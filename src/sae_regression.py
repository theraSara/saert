import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
SAE_DIR = ROOT / "results" / "sae"
REG_DIR = ROOT / "results" / "regression"

N_LAYERS = 12
N_CV_FOLDS = 5
RANDOM_STATE = 42
ALPHAS = np.logspace(-4, 1, 40)
CONTROLS = ["zipf_freq", "word_length", "is_sentence_final"]
RT_COL = "log_primary_RT"


def make_row_uid(corpus: str, df: pd.DataFrame) -> pd.Series:
    return (
        corpus + ":" + df["text_id"].astype(str)
        + ":" + df["word_position"].astype(int).astype(str)
    )


def get_cv_splitter(groups: np.ndarray, n_splits: int = N_CV_FOLDS):
    if groups is None:
        return KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        ), None
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        return GroupKFold(n_splits=min(n_splits, len(unique_groups))), groups
    return KFold(
        n_splits=min(n_splits, len(groups)),
        shuffle=True,
        random_state=RANDOM_STATE,
    ), None


def load_corpus_data(corpus: str) -> tuple[pd.DataFrame, dict]:
    rt_path = REG_DIR / f"{corpus}_residualized_RT.csv"
    if not rt_path.exists():
        raise FileNotFoundError(
            f"Residualized RT not found at {rt_path}\n"
            "Run src/baseline.py first."
        )

    model_path = REG_DIR / f"{corpus}_baseline_model.pkl"
    with open(model_path, "rb") as f:
        baseline = pickle.load(f)

    df = pd.read_csv(rt_path)
    if "row_uid" not in df.columns:
        df["row_uid"] = make_row_uid(corpus, df)

    required = [RT_COL, "surprisal", "row_uid", "text_id"] + CONTROLS
    df = df.dropna(subset=required).reset_index(drop=True)

    print(f"  Loaded {len(df):,} words for {corpus}")
    if "heldout" in baseline:
        heldout = baseline["heldout"]
        print(f"  Held-out R2 controls:           {heldout['r2_controls']:.4f}")
        print(f"  Held-out R2 controls+surprisal: {heldout['r2_controls_surprisal']:.4f}")
        print(f"  Held-out delta surprisal:       {heldout['delta_r2_surprisal']:.4f}")

    return df, baseline


def load_sae_features(corpus: str, layer: int, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    stats_path = SAE_DIR / f"{corpus}_sae_L{layer:02d}_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"SAE stats not found at {stats_path}\n"
            "Run src/sae_extraction.py first."
        )

    data = np.load(stats_path, allow_pickle=True)
    dense_topk = data["dense_topk"]
    top_k_ids = data["top_k_ids"]

    if dense_topk.shape[0] != len(df):
        raise ValueError(
            f"{corpus} layer {layer}: SAE matrix has {dense_topk.shape[0]} rows "
            f"but regression data has {len(df)} rows. Regenerate upstream artifacts."
        )

    if "row_uid" in data:
        sae_uids = data["row_uid"].astype(str)
        df_uids = df["row_uid"].astype(str).to_numpy()
        if not np.array_equal(sae_uids, df_uids):
            mismatches = np.where(sae_uids != df_uids)[0][:5]
            examples = [
                f"{i}: sae={sae_uids[i]} df={df_uids[i]}"
                for i in mismatches
            ]
            raise ValueError(
                f"{corpus} layer {layer}: row_uid mismatch between SAE and RT data. "
                + "; ".join(examples)
            )

    return dense_topk, top_k_ids


def residualize_against_controls(
    X_train: np.ndarray,
    X_test: np.ndarray,
    C_train: np.ndarray,
    C_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if X_train.shape[1] == 0:
        return X_train, X_test
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(C_train, X_train)
    return X_train - model.predict(C_train), X_test - model.predict(C_test)


def fit_incremental_model(
    C_train: np.ndarray,
    C_test: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray | None,
    use_lasso: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    control_model = make_pipeline(StandardScaler(), LinearRegression())
    control_model.fit(C_train, y_train)
    control_pred_train = control_model.predict(C_train)
    control_pred_test = control_model.predict(C_test)

    if X_train.shape[1] == 0:
        return control_pred_test, np.array([]), np.nan

    y_resid_train = y_train - control_pred_train
    X_resid_train, X_resid_test = residualize_against_controls(
        X_train, X_test, C_train, C_test
    )

    if use_lasso:
        inner_cv, inner_groups = get_cv_splitter(groups_train)
        model = GridSearchCV(
            make_pipeline(
                StandardScaler(),
                Lasso(max_iter=10000, random_state=RANDOM_STATE),
            ),
            param_grid={"lasso__alpha": ALPHAS},
            cv=inner_cv,
            scoring="r2",
        )
        model.fit(X_resid_train, y_resid_train, groups=inner_groups)
        resid_pred = model.predict(X_resid_test)
        coef = model.best_estimator_.named_steps["lasso"].coef_
        alpha = float(model.best_params_["lasso__alpha"])
    else:
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(X_resid_train, y_resid_train)
        resid_pred = model.predict(X_resid_test)
        coef = model.named_steps["linearregression"].coef_
        alpha = np.nan

    return control_pred_test + resid_pred, np.asarray(coef), alpha


def fit_final_coefficients(
    C: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, float]:
    _, coef, alpha = fit_incremental_model(
        C, C, X, X, y, groups, use_lasso=True
    )
    return coef, alpha


def grouped_bootstrap_metrics(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    n_boot: int = 1000,
) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    rows = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
        if np.var(y[idx]) == 0:
            continue
        row = {name: r2_score(y[idx], pred[idx]) for name, pred in predictions.items()}
        row["delta_surprisal"] = row["controls_surprisal"] - row["controls"]
        row["delta_sae_over_surprisal"] = row["combined"] - row["controls_surprisal"]
        row["delta_sae_over_controls"] = row["controls_sae"] - row["controls"]
        rows.append(row)
    boot = pd.DataFrame(rows)
    out = {}
    for col in boot.columns:
        out[f"{col}_ci_low"] = boot[col].quantile(0.025)
        out[f"{col}_ci_high"] = boot[col].quantile(0.975)
    return out


def analyse_layer(corpus: str, layer: int, df: pd.DataFrame) -> tuple[dict, list[dict], pd.DataFrame] | None:
    y = df[RT_COL].to_numpy(dtype=float)
    C = df[CONTROLS].to_numpy(dtype=float)
    surp = df[["surprisal"]].to_numpy(dtype=float)
    groups = df["text_id"].to_numpy()

    try:
        feat_matrix, top_k_ids = load_sae_features(corpus, layer, df)
    except FileNotFoundError as e:
        print(f"    [SKIP] {e}")
        return None

    col_var = feat_matrix.var(axis=0)
    nonzero = col_var > 0
    feat_use = feat_matrix[:, nonzero]
    ids_use = top_k_ids[nonzero]

    splitter, split_groups = get_cv_splitter(groups)
    split_args = (C, y, split_groups) if split_groups is not None else (C, y)

    preds = {
        "controls": np.full(len(df), np.nan),
        "controls_surprisal": np.full(len(df), np.nan),
        "controls_sae": np.full(len(df), np.nan),
        "combined": np.full(len(df), np.nan),
    }
    fold_rows = []
    fold_feature_rows = []
    alpha_sae = []
    alpha_combined = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(*split_args), start=1):
        C_train, C_test = C[train_idx], C[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx] if split_groups is not None else None

        controls_pred, _, _ = fit_incremental_model(
            C_train, C_test,
            np.empty((len(train_idx), 0)), np.empty((len(test_idx), 0)),
            y_train, groups_train, use_lasso=False
        )
        surp_pred, _, _ = fit_incremental_model(
            C_train, C_test, surp[train_idx], surp[test_idx],
            y_train, groups_train, use_lasso=False
        )
        sae_pred, _, sae_alpha = fit_incremental_model(
            C_train, C_test, feat_use[train_idx], feat_use[test_idx],
            y_train, groups_train, use_lasso=True
        )
        combined_X = np.column_stack([surp, feat_use])
        combined_pred, combined_coef, combined_alpha = fit_incremental_model(
            C_train, C_test, combined_X[train_idx], combined_X[test_idx],
            y_train, groups_train, use_lasso=True
        )

        preds["controls"][test_idx] = controls_pred
        preds["controls_surprisal"][test_idx] = surp_pred
        preds["controls_sae"][test_idx] = sae_pred
        preds["combined"][test_idx] = combined_pred
        alpha_sae.append(sae_alpha)
        alpha_combined.append(combined_alpha)

        feature_coefs = combined_coef[1:] if len(combined_coef) else np.array([])
        selected = np.where(feature_coefs != 0)[0]
        for j in selected:
            fold_feature_rows.append({
                "fold": fold,
                "layer": layer,
                "feature_id": int(ids_use[j]),
                "coef": float(feature_coefs[j]),
            })

        fold_rows.append({
            "corpus": corpus,
            "layer": layer,
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "r2_controls": r2_score(y_test, controls_pred),
            "r2_controls_surprisal": r2_score(y_test, surp_pred),
            "r2_controls_sae": r2_score(y_test, sae_pred),
            "r2_combined": r2_score(y_test, combined_pred),
        })

    r2_controls = r2_score(y, preds["controls"])
    r2_surp = r2_score(y, preds["controls_surprisal"])
    r2_sae = r2_score(y, preds["controls_sae"])
    r2_combined = r2_score(y, preds["combined"])
    ci = grouped_bootstrap_metrics(y, preds, groups)

    n_active_per_word = (feat_use > 0).sum(axis=1)
    r_l0, p_l0 = pearsonr(n_active_per_word, y)

    final_combined_X = np.column_stack([surp, feat_use])
    final_coef, final_alpha = fit_final_coefficients(C, final_combined_X, y, groups)
    final_feature_coef = final_coef[1:] if len(final_coef) else np.array([])

    fold_features = pd.DataFrame(fold_feature_rows)
    n_outer_folds = len(fold_rows)
    top_features = []
    for j, coef in enumerate(final_feature_coef):
        if coef == 0:
            continue
        feature_id = int(ids_use[j])
        if fold_features.empty:
            selected_coefs = pd.Series(dtype=float)
        else:
            selected_coefs = fold_features.loc[
                fold_features["feature_id"] == feature_id, "coef"
            ]
        signs = np.sign(selected_coefs.to_numpy())
        final_sign = np.sign(coef)
        sign_agreement = float((signs == final_sign).mean()) if len(signs) else 0.0
        top_features.append({
            "corpus": corpus,
            "layer": layer,
            "feature_id": feature_id,
            "coef": float(coef),
            "direction": "positive" if coef > 0 else "negative",
            "mean_act": float(feat_use[:, j].mean()),
            "pct_active": float((feat_use[:, j] > 0).mean() * 100),
            "selection_rate": float(len(selected_coefs) / n_outer_folds),
            "mean_cv_coef": float(selected_coefs.mean()) if len(selected_coefs) else 0.0,
            "sign_agreement": sign_agreement,
        })
    top_features = sorted(top_features, key=lambda x: abs(x["coef"]), reverse=True)

    result = {
        "corpus": corpus,
        "layer": layer,
        "n_words": len(df),
        "n_sae_features_available": int(feat_use.shape[1]),
        "n_sae_features_selected_final": len(top_features),
        "n_sae_features_selected_stable_60pct": int(sum(f["selection_rate"] >= 0.6 for f in top_features)),
        "heldout_r2_controls": r2_controls,
        "heldout_r2_controls_surprisal": r2_surp,
        "heldout_r2_controls_sae": r2_sae,
        "heldout_r2_combined": r2_combined,
        "heldout_delta_r2_surprisal": r2_surp - r2_controls,
        "heldout_delta_r2_sae_over_surprisal": r2_combined - r2_surp,
        "heldout_delta_r2_sae_over_controls": r2_sae - r2_controls,
        "lasso_alpha_sae_mean": float(np.nanmean(alpha_sae)),
        "lasso_alpha_combined_mean": float(np.nanmean(alpha_combined)),
        "lasso_alpha_combined_final": final_alpha,
        "r_l0_rt": r_l0,
        "p_l0_rt": p_l0,
        **ci,
    }

    print(
        f"    Layer {layer:2d} | "
        f"R2 ctrl={r2_controls:.4f} | "
        f"ctrl+surp={r2_surp:.4f} | "
        f"ctrl+SAE={r2_sae:.4f} | "
        f"combined={r2_combined:.4f} | "
        f"delta SAE|surp={result['heldout_delta_r2_sae_over_surprisal']:.4f} | "
        f"stable feats={result['n_sae_features_selected_stable_60pct']}"
    )

    return result, top_features, pd.DataFrame(fold_rows)


def run_sae_regression(corpus: str):
    print(f"\n{'='*65}")
    print(f"  SAE REGRESSION - {corpus.upper()}")
    print(f"{'='*65}")

    df, _ = load_corpus_data(corpus)
    all_results = []
    all_top_features = []
    all_folds = []

    for layer in range(1, N_LAYERS + 1):
        out = analyse_layer(corpus, layer, df)
        if out is None:
            continue
        result, top_features, fold_df = out
        all_results.append(result)
        all_top_features.extend(top_features)
        all_folds.append(fold_df)

    results_df = pd.DataFrame(all_results)
    results_path = REG_DIR / f"{corpus}_sae_regression.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results saved -> {results_path}")

    if all_top_features:
        feat_df = pd.DataFrame(all_top_features)
        feat_path = REG_DIR / f"{corpus}_sae_top_features.csv"
        feat_df.to_csv(feat_path, index=False)
        print(f"  Top features  -> {feat_path}")
    else:
        feat_df = pd.DataFrame()

    if all_folds:
        fold_path = REG_DIR / f"{corpus}_sae_cv_folds.csv"
        pd.concat(all_folds, ignore_index=True).to_csv(fold_path, index=False)
        print(f"  CV folds      -> {fold_path}")

    if not results_df.empty:
        best = results_df.loc[results_df["heldout_delta_r2_sae_over_surprisal"].idxmax()]
        print(
            f"\n  Best layer by held-out delta R2(SAE|surprisal): "
            f"Layer {int(best['layer'])} "
            f"({best['heldout_delta_r2_sae_over_surprisal']:.4f})"
        )

    return results_df, feat_df


def main():
    parser = argparse.ArgumentParser(
        description="SparseRT - Step 05: Grouped held-out SAE regression"
    )
    parser.add_argument(
        "--corpus",
        choices=["provo", "natural_stories", "both"],
        default="both",
    )
    args = parser.parse_args()

    corpora = (
        ["provo", "natural_stories"] if args.corpus == "both"
        else [args.corpus]
    )

    for corpus in corpora:
        try:
            run_sae_regression(corpus)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n[SKIP] {e}\n")

    print("\nSAE regression complete.")
    print("Next step: inspect stability before feature interpretation.")


if __name__ == "__main__":
    main()
