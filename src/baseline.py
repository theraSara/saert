import argparse
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
OUT_DIR   = ROOT / "results" / "regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RT_MIN_MS = 100
RT_MAX_MS = 3000

SENT_FINAL_RE = re.compile(r'[.!?]["\']?$')

CONTROLS        = ["zipf_freq", "word_length", "is_sentence_final"]
FULL_PREDICTORS = CONTROLS + ["surprisal"]

RT_COL = "log_primary_RT"
N_CV_FOLDS = 5
RANDOM_STATE = 42


def is_sentence_final(word: str) -> int:
    return int(bool(SENT_FINAL_RE.search(str(word))))


def apply_rt_exclusions(df: pd.DataFrame, corpus: str) -> pd.DataFrame:
    n_before = len(df)
    df = df[(df["primary_RT"] >= RT_MIN_MS) & (df["primary_RT"] <= RT_MAX_MS)]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  RT exclusion [{RT_MIN_MS}–{RT_MAX_MS} ms]: "
              f"dropped {n_dropped} rows ({n_before} → {len(df)})")
    else:
        print(f"  RT exclusion [{RT_MIN_MS}–{RT_MAX_MS} ms]: 0 rows dropped ✓")
    return df

def load_surprisal_data(corpus: str) -> pd.DataFrame:
    surp_dir = ROOT / "results" / "surprisal"
    path = surp_dir / f"{corpus}_surprisal.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Surprisal data not found at {path}\n"
        )

    df = pd.read_csv(path)
    n_raw = len(df)

    df = apply_rt_exclusions(df, corpus)

    df = df.dropna(subset=["surprisal"])

    df["is_sentence_final"] = df["word"].apply(is_sentence_final)
    n_final = df["is_sentence_final"].sum()
    print(f"  Sentence-final words flagged: {n_final} "
          f"({100*n_final/len(df):.1f}%)")

    required = [RT_COL, "zipf_freq", "word_length", "surprisal"]
    df = df.dropna(subset=required)
    df = df[df[RT_COL] > 0]

    n_dropped = n_raw - len(df)
    print(f"  Final dataset: {len(df):,} rows "
          f"({n_dropped} total dropped from {n_raw:,})")

    for col in FULL_PREDICTORS:
        mu, sd = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / sd if sd > 0 else 0.0

    return df


def fit_ols(df: pd.DataFrame, predictors: list[str], outcome: str = RT_COL):
    """OLS with HC3 robust standard errors on standardised predictors."""
    pred_z  = [f"{p}_z" for p in predictors]
    formula = f"{outcome} ~ " + " + ".join(pred_z)
    result  = smf.ols(formula, data=df).fit(cov_type="HC3")
    return result


def compute_delta_r2(r2_full: float, r2_reduced: float) -> float:
    return r2_full - r2_reduced


def compute_residualized_rt(df: pd.DataFrame, controls_result) -> pd.Series:
    """
    RT' = log_RT - fitted(controls only) + intercept
    """
    intercept = controls_result.params["Intercept"]
    rt_prime  = df[RT_COL] - controls_result.fittedvalues + intercept
    return rt_prime


def get_cv_splitter(groups: pd.Series, n_splits: int = N_CV_FOLDS):
    n_groups = groups.nunique()
    n_splits = min(n_splits, n_groups)
    if n_splits >= 2:
        return GroupKFold(n_splits=n_splits), groups.to_numpy()
    return KFold(
        n_splits=min(N_CV_FOLDS, len(groups)),
        shuffle=True,
        random_state=RANDOM_STATE,
    ), None


def grouped_bootstrap_r2(
    y: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    r2_values = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
        if np.var(y[idx]) > 0:
            r2_values.append(r2_score(y[idx], y_pred[idx]))
    if not r2_values:
        return np.nan, np.nan
    return tuple(np.percentile(r2_values, [2.5, 97.5]))


def fit_grouped_heldout(df: pd.DataFrame) -> dict:
    y = df[RT_COL].to_numpy(dtype=float)
    groups = df["text_id"].to_numpy()
    splitter, split_groups = get_cv_splitter(df["text_id"])
    X_controls = df[CONTROLS].to_numpy(dtype=float)
    X_full = df[FULL_PREDICTORS].to_numpy(dtype=float)
    preds_controls = np.full(len(df), np.nan)
    preds_full = np.full(len(df), np.nan)
    fold_rows = []

    split_args = (X_controls, y, split_groups) if split_groups is not None else (X_controls, y)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(*split_args), start=1):
        controls_model = make_pipeline(StandardScaler(), LinearRegression())
        full_model = make_pipeline(StandardScaler(), LinearRegression())
        controls_model.fit(X_controls[train_idx], y[train_idx])
        full_model.fit(X_full[train_idx], y[train_idx])
        preds_controls[test_idx] = controls_model.predict(X_controls[test_idx])
        preds_full[test_idx] = full_model.predict(X_full[test_idx])
        fold_rows.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "r2_controls": r2_score(y[test_idx], preds_controls[test_idx]),
            "r2_controls_surprisal": r2_score(y[test_idx], preds_full[test_idx]),
        })

    r2_controls = r2_score(y, preds_controls)
    r2_full = r2_score(y, preds_full)
    ci_controls = grouped_bootstrap_r2(y, preds_controls, groups)
    ci_full = grouped_bootstrap_r2(y, preds_full, groups)
    return {
        "pred_controls": preds_controls,
        "pred_controls_surprisal": preds_full,
        "r2_controls": r2_controls,
        "r2_controls_surprisal": r2_full,
        "delta_r2_surprisal": r2_full - r2_controls,
        "r2_controls_ci_low": ci_controls[0],
        "r2_controls_ci_high": ci_controls[1],
        "r2_controls_surprisal_ci_low": ci_full[0],
        "r2_controls_surprisal_ci_high": ci_full[1],
        "folds": pd.DataFrame(fold_rows),
    }

def print_regression_summary(
    corpus, controls_result, full_result,
    r2_controls, r2_full, delta_r2_surprisal, df, heldout
):
    sep  = "═" * 62
    thin = "─" * 62
    lines = [
        sep,
        f"  BASELINE REGRESSION — {corpus.upper()}",
        f"  N = {len(df):,} words  |  DV = log(RT ms)",
        f"  Exclusions: RT<{RT_MIN_MS}ms or >{RT_MAX_MS}ms; NaN surprisal",
        f"  Controls: zipf_freq, word_length, is_sentence_final",
        sep,
        f"  {'Predictor':<22} {'β':>8} {'SE':>8} {'t':>8} {'p':>10}",
        thin,
    ]

    for pred in FULL_PREDICTORS:
        col_z = f"{pred}_z"
        if col_z in full_result.params:
            b   = full_result.params[col_z]
            se  = full_result.bse[col_z]
            t   = full_result.tvalues[col_z]
            p   = full_result.pvalues[col_z]
            sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
            lines.append(
                f"  {pred:<22} {b:>8.4f} {se:>8.4f} {t:>8.3f} {p:>10.4f} {sig}"
            )

    lines += [
        thin, "",
        "  In-sample model comparison (R²):",
        thin,
        f"  Controls only:     {r2_controls:.4f}",
        f"  + Surprisal:       {r2_full:.4f}",
        f"  ΔR² surprisal:     {delta_r2_surprisal:.4f}  "
        f"({100*delta_r2_surprisal:.2f}% unique variance)",
        "",
        "  Grouped held-out model comparison (R²):",
        thin,
        f"  Controls only:     {heldout['r2_controls']:.4f} "
        f"[{heldout['r2_controls_ci_low']:.4f}, {heldout['r2_controls_ci_high']:.4f}]",
        f"  + Surprisal:       {heldout['r2_controls_surprisal']:.4f} "
        f"[{heldout['r2_controls_surprisal_ci_low']:.4f}, "
        f"{heldout['r2_controls_surprisal_ci_high']:.4f}]",
        f"  ΔR² surprisal:     {heldout['delta_r2_surprisal']:.4f}",
        "",
        "  Surprisal (bits):",
        f"    Mean {df['surprisal'].mean():.3f}   Median {df['surprisal'].median():.3f}"
        f"   SD {df['surprisal'].std():.3f}",
        "",
        "  RT' distribution:",
        f"    Mean {df['RT_prime'].mean():.4f}   SD {df['RT_prime'].std():.4f}"
        f"   Skew {df['RT_prime'].skew():.3f}",
        f"    Corr(surprisal, RT') = {df['surprisal'].corr(df['RT_prime']):.4f}",
        f"    Sentence-final words: {df['is_sentence_final'].sum():,} "
        f"| Mean RT (final): {df[df['is_sentence_final']==1]['primary_RT'].mean():.1f} ms"
        f" vs (non-final): {df[df['is_sentence_final']==0]['primary_RT'].mean():.1f} ms",
        sep,
    ]

    summary_str = "\n".join(lines)
    print("\n" + summary_str + "\n")
    return summary_str

def run_baseline_regression(corpus: str):
    print(f"\n{'─'*62}")
    print(f"  Running baseline regression for: {corpus}")
    print(f"{'─'*62}")

    df = load_surprisal_data(corpus)

    print("  Fitting controls-only model...")
    controls_result  = fit_ols(df, CONTROLS)
    r2_controls      = controls_result.rsquared

    print("  Fitting full model (controls + surprisal)...")
    full_result      = fit_ols(df, FULL_PREDICTORS)
    r2_full          = full_result.rsquared

    print("  Fitting surprisal-only model...")
    surp_only_result = fit_ols(df, ["surprisal"])
    r2_surp_only     = surp_only_result.rsquared

    delta_r2_surp    = compute_delta_r2(r2_full, r2_controls)

    print("  Running grouped held-out baseline...")
    heldout = fit_grouped_heldout(df)

    print("  Computing RT'...")
    df["RT_prime"] = compute_residualized_rt(df, controls_result)
    df["controls_oof_pred"] = heldout["pred_controls"]
    df["controls_surprisal_oof_pred"] = heldout["pred_controls_surprisal"]
    df["RT_resid_controls_oof"] = df[RT_COL] - df["controls_oof_pred"]

    summary_str = print_regression_summary(
        corpus, controls_result, full_result,
        r2_controls, r2_full, delta_r2_surp, df, heldout
    )

    summary_path = OUT_DIR / f"{corpus}_regression_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_str + "\n\n")
        f.write("=== FULL MODEL (statsmodels) ===\n\n")
        f.write(str(full_result.summary()))
        f.write("\n\n=== CONTROLS-ONLY MODEL ===\n\n")
        f.write(str(controls_result.summary()))
    print(f"  Summary to {summary_path}")

    model_path = OUT_DIR / f"{corpus}_baseline_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "controls_result":  controls_result,
            "full_result":      full_result,
            "surp_only_result": surp_only_result,
            "r2_controls":      r2_controls,
            "r2_full":          r2_full,
            "r2_surp_only":     r2_surp_only,
            "delta_r2_surp":    delta_r2_surp,
            "heldout":          {k: v for k, v in heldout.items() if k != "folds"},
            "controls":         CONTROLS,
            "full_predictors":  FULL_PREDICTORS,
            "rt_col":           RT_COL,
            "n_obs":            len(df),
        }, f)
    print(f"  Model  → {model_path}")

    cv_path = OUT_DIR / f"{corpus}_baseline_cv_folds.csv"
    heldout["folds"].to_csv(cv_path, index=False)
    print(f"  CV     → {cv_path}")

    out_cols = [
        "hidden_row_idx",                           # direct index into .npy files
        "row_uid",
        "text_id", "word_position", "word", "word_clean",
        "primary_RT", "log_primary_RT",
        "surprisal", "surprisal_z",
        "zipf_freq", "zipf_freq_z",
        "word_length", "word_length_z",
        "is_sentence_final", "is_sentence_final_z",
        "start_token_idx", "end_token_idx", "hidden_token_idx", "n_tokens",
        "hidden_context_size", "hidden_context_stride",
        "RT_prime", "RT_resid_controls_oof",
        "controls_oof_pred", "controls_surprisal_oof_pred",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    rt_out = OUT_DIR / f"{corpus}_residualized_RT.csv"
    df[out_cols].to_csv(rt_out, index=False)
    print(f"  RT'    → {rt_out}")

    return df, full_result, controls_result

def main():
    parser = argparse.ArgumentParser(
        description="SparseRT — Baseline Regression + RT Residualization"
    )
    parser.add_argument(
        "--corpus",
        choices=["provo", "natural_stories", "both"],
        default="both",
    )
    args = parser.parse_args()

    corpora = (
        ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]
    )
    for corpus in corpora:
        try:
            run_baseline_regression(corpus)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {e}\n")

    print("\nBaseline regression complete.")

if __name__ == "__main__":
    main()
