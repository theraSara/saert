"""Reproducible aggregate-RT baselines with leave-one-text-out evaluation."""

import argparse
import json
from pathlib import Path
import tempfile
import uuid

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .pipeline_utils import file_record, normalize_words, numeric, read_table, run_metadata, sha256_file, write_json
    from .validate_outputs import validate_corpus
except ImportError:
    from pipeline_utils import file_record, normalize_words, numeric, read_table, run_metadata, sha256_file, write_json
    from validate_outputs import validate_corpus

ROOT = Path(__file__).resolve().parent.parent
OUTCOME = "log_primary_RT"
SURPRISALS = {"long": "surprisal", "matched": "surprisal_matched_context"}
LEXICAL = ["zipf_freq", "word_length", "is_sentence_final"]


def build_design(frame, lags=2):
    """Construct history on complete stimuli, then identify eligible RT rows."""
    if lags not in range(0, 4):
        raise ValueError("Specify 0..3 spillover lags before examining outcomes")
    df = normalize_words(frame, "baseline stimuli", complete=True)
    for col in ["primary_RT", OUTCOME] + LEXICAL + list(SURPRISALS.values()):
        df[col] = numeric(df[col], col)
    predictors = LEXICAL + list(SURPRISALS.values())
    if not np.isfinite(df[predictors].to_numpy()).all():
        raise ValueError("Missing/invalid stimulus predictors; repair extraction rather than impute them")
    if (df[list(SURPRISALS.values())] < 0).any().any():
        raise ValueError("Negative surprisal")
    df["log_word_position"] = np.log1p(df["word_position"])
    controls = LEXICAL + ["log_word_position"]
    # Distinct position flags absorb structural zero fills without duplicating
    # availability indicators (which would introduce exact collinearity).
    for k in range(1, max(1, lags) + 1):
        name = f"is_text_position_{k}"
        df[name] = df["word_position"].eq(k).astype(int)
        controls.append(name)
    for lag in range(1, lags + 1):
        df[f"lag{lag}_available"] = df["word_position"].gt(lag).astype(int)
        for col in predictors:
            name = f"{col}_lag{lag}"
            df[name] = df.groupby("text_id", sort=False)[col].shift(lag).fillna(0.)
            if col in LEXICAL:
                controls.append(name)
    models = {"controls": controls.copy()}
    for context, col in SURPRISALS.items():
        models[f"{context}_current"] = controls + [col]
        if lags:
            models[f"{context}_spillover"] = controls + [col] + [f"{col}_lag{k}" for k in range(1, lags+1)]
    positive = df["primary_RT"].gt(0) & df["primary_RT"].notna()
    if not np.allclose(df.loc[positive, OUTCOME], np.log(df.loc[positive, "primary_RT"]), rtol=1e-7, atol=1e-7):
        raise ValueError("Outcome must be log of the arithmetic mean RT, in milliseconds")
    df["analysis_status"] = np.where(positive, "included", "missing_or_nonpositive_RT")
    eligible = df.loc[positive].copy().reset_index(drop=True)
    if eligible["text_id"].nunique() < 3:
        raise ValueError("At least three texts with eligible RT observations are required")
    return eligible, df, models


def fit_model(train, columns):
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(train[columns].to_numpy(dtype=float), train[OUTCOME].to_numpy(dtype=float))
    return model


def model_parameters(model, columns, model_name, fold):
    scaler, regression = model.steps[0][1], model.steps[1][1]
    coefficients = regression.coef_.astype(float)
    return {"model": model_name, "fold": fold, "columns": columns,
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef_standardized": coefficients.tolist(), "intercept_standardized": float(regression.intercept_),
            "coef_raw": (coefficients / scaler.scale_).tolist(),
            "intercept_raw": float(regression.intercept_ - np.dot(coefficients / scaler.scale_, scaler.mean_))}


def evaluate_models(df, models):
    """All learned preprocessing is fitted inside each training split."""
    predictions = df.copy()
    y = df[OUTCOME].to_numpy(dtype=float)
    groups = df["text_id"].to_numpy()
    predictions["fold"] = -1
    for name in models:
        predictions[f"pred_{name}"] = np.nan
    folds, parameters = [], []
    for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(df, groups=groups), 1):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        predictions.loc[test_idx, "fold"] = fold
        folds.append({"fold": fold, "train_text_ids": sorted(map(int, train["text_id"].unique())),
                      "test_text_ids": sorted(map(int, test["text_id"].unique())),
                      "n_train": len(train), "n_test": len(test)})
        for name, columns in models.items():
            model = fit_model(train, columns)
            predictions.loc[test_idx, f"pred_{name}"] = model.predict(test[columns].to_numpy(dtype=float))
            parameters.append(model_parameters(model, columns, name, fold))
    if not np.isfinite(predictions[[f"pred_{n}" for n in models]].to_numpy()).all():
        raise ValueError("Incomplete out-of-fold prediction coverage")
    predictions["RT_resid_controls_oof"] = y - predictions["pred_controls"]
    # This separate full-data adjustment is DESCRIPTIVE, never an evaluation target.
    descriptive_model = fit_model(df, models["controls"])
    predictions["RT_prime_descriptive"] = y - descriptive_model.predict(df[models["controls"]].to_numpy()) + descriptive_model.steps[1][1].intercept_
    for name, columns in models.items():
        parameters.append(model_parameters(fit_model(df, columns), columns, name, "full_data_descriptive"))
    return predictions, folds, parameters


def paired_metrics(predictions, models, bootstrap=2000, seed=42):
    """Paired text bootstrap of FIXED OOF predictions, without model refitting."""
    if bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    names = list(models)
    y = predictions[OUTCOME].to_numpy(dtype=float)
    pred = predictions[[f"pred_{n}" for n in names]].to_numpy(dtype=float)
    stats, per_text = [], []
    for text_id, group in predictions.groupby("text_id", sort=True):
        yg = group[OUTCOME].to_numpy(dtype=float)
        pg = group[[f"pred_{n}" for n in names]].to_numpy(dtype=float)
        sse = ((yg[:, None]-pg)**2).sum(axis=0)
        sae = abs(yg[:, None]-pg).sum(axis=0)
        stats.append(np.r_[len(group), yg.sum(), np.square(yg).sum(), sse, sae])
        for j, name in enumerate(names):
            per_text.append({"text_id": int(text_id), "model": name, "n": len(group),
                "r2": float(r2_score(yg, pg[:, j])) if len(yg)>1 and np.var(yg)>0 else np.nan,
                "mse": sse[j]/len(group), "rmse": np.sqrt(sse[j]/len(group)),
                "delta_mse_vs_controls": (sse[0]-sse[j])/len(group)})
    stats = np.array(stats)
    rng = np.random.default_rng(seed)
    sampled = stats[rng.integers(0, len(stats), size=(bootstrap, len(stats)))].sum(axis=1)
    sst = sampled[:, 2] - sampled[:, 1]**2 / sampled[:, 0]
    valid = sst > 0
    if not valid.any():
        raise ValueError("All bootstrap samples have zero outcome variance")
    br2 = 1 - sampled[valid, 3:3+len(names)] / sst[valid, None]
    actual_r2 = np.array([r2_score(y, pred[:, j]) for j in range(len(names))])
    summary = []
    for j, name in enumerate(names):
        rlo, rhi = np.quantile(br2[:, j], [.025, .975])
        dlo, dhi = np.quantile(br2[:, j]-br2[:, 0], [.025, .975])
        summary.append({"model": name, "n": len(y), "n_texts": len(stats),
            "n_predictors": len(models[name]), "r2": actual_r2[j], "r2_ci_low": rlo, "r2_ci_high": rhi,
            "rmse_log_ms": np.sqrt(mean_squared_error(y, pred[:, j])),
            "mae_log_ms": np.mean(abs(y-pred[:, j])),
            "delta_r2_vs_controls": actual_r2[j]-actual_r2[0],
            "delta_r2_ci_low": dlo, "delta_r2_ci_high": dhi})
    pairs = [(name, "controls") for name in names if name != "controls"]
    pairs += [(f"{c}_spillover", f"{c}_current") for c in SURPRISALS if f"{c}_spillover" in names]
    pairs += [("long_spillover", "matched_spillover")] if "long_spillover" in names else [("long_current", "matched_current")]
    contrasts = []
    for candidate, reference in pairs:
        j, k = names.index(candidate), names.index(reference)
        lo, hi = np.quantile(br2[:, j]-br2[:, k], [.025, .975])
        contrasts.append({"candidate": candidate, "reference": reference,
            "delta_r2": actual_r2[j]-actual_r2[k], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(summary), pd.DataFrame(contrasts), pd.DataFrame(per_text)


def run_baseline(corpus, data_dir, results_dir, output_dir, lags=2, bootstrap=2000, seed=42, overwrite=False):
    output = Path(output_dir)
    manifest_name = f"{corpus}_baseline_manifest.json"
    if not overwrite and list(output.glob(f"{corpus}_*")):
        raise FileExistsError(f"Baseline artifacts already exist: {output}; use a new directory or --overwrite")
    print(f"Validating {corpus} extraction", flush=True)
    quality = validate_corpus(corpus, data_dir, results_dir)
    source_path = Path(results_dir) / f"{corpus}_surprisal.csv"
    source_manifest = Path(results_dir) / f"{corpus}_extraction_manifest.json"
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    eligible, all_rows, models = build_design(read_table(source_path), lags)
    print(f"{corpus}: {len(eligible):,} eligible rows; {eligible.text_id.nunique()} held-out texts", flush=True)
    predictions, folds, parameters = evaluate_models(eligible, models)
    summary, contrasts, per_text = paired_metrics(predictions, models, bootstrap, seed)
    run_id = str(uuid.uuid4())
    for df in [predictions, all_rows, summary, contrasts, per_text]:
        df.insert(0, "baseline_id", run_id)
        if "corpus" not in df:
            df.insert(1, "corpus", corpus)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".baseline-", dir=output) as tmp:
        tmp = Path(tmp)
        for suffix, df in [("predictions", predictions), ("analysis_rows", all_rows),
                           ("model_comparison", summary), ("contrasts", contrasts), ("per_text", per_text)]:
            df.to_csv(tmp / f"{corpus}_{suffix}.csv", index=False)
        write_json(tmp / f"{corpus}_fitted_models.json", {"baseline_id": run_id, "fits": parameters})
        manifest = {**run_metadata(), "schema_version": 1, "baseline_id": run_id, "corpus": corpus,
            "extraction_id": source["extraction_id"], "preparation_id": source["preparation_id"],
            "inputs": {"scores": file_record(source_path), "extraction_manifest": file_record(source_manifest)},
            "quality": quality, "n_eligible": len(eligible),
            "exclusions": all_rows.analysis_status.value_counts().to_dict(),
            "outcome": "natural log of arithmetic mean RT in ms; unweighted region/word observations",
            "rt_policy": "No new aggregate RT cutoffs; missing/nonpositive RT excluded AFTER constructing lags",
            "lag_policy": "previous presented units within text, before RT filtering; zero structural fill with initial-position indicators",
            "spillover_lags": lags, "models": models, "folds": folds,
            "evaluation": "leave one complete text out; training-only StandardScaler and unpenalized OLS",
            "bootstrap": {"replicates": bootstrap, "seed": seed, "unit": "text",
                          "method": "paired percentile 95% interval over fixed OOF predictions; no model refitting"},
            "context_settings": {"long": source["surprisal"], "matched": source["hidden"]},
            "limitations": ["Aggregate-RT exploratory analysis; not participant-level mixed effects",
                "Fixed-prediction bootstrap does not include model-training variability; only 10 Natural Stories clusters",
                "No participant or item random effects; text ID defines splits, not a numeric predictor",
                "RT_prime_descriptive is an in-sample visualization adjustment, not pure comprehension or a valid CV target",
                "Previously inspected corpora are not untouched confirmatory test sets"],
            "source_code": [file_record(__file__), file_record(Path(__file__).with_name("validate_outputs.py")),
                            file_record(Path(__file__).with_name("pipeline_utils.py"))],
            "outputs": {p.name: {"sha256": sha256_file(p)} for p in tmp.iterdir()}}
        write_json(tmp / manifest_name, manifest)
        for p in tmp.iterdir():
            if p.name != manifest_name:
                p.replace(output / p.name)
        (tmp / manifest_name).replace(output / manifest_name)
    print(summary[["model", "r2", "delta_r2_vs_controls", "delta_r2_ci_low", "delta_r2_ci_high"]].to_string(index=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["both", "provo", "natural_stories"], default="both")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/prepared_v2")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/surprisal_bos")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/baseline_bos")
    parser.add_argument("--spillover-lags", type=int, choices=range(4), default=2)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    corpora = ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]
    for corpus in corpora:
        run_baseline(corpus, args.data_dir, args.results_dir, args.output_dir,
                     args.spillover_lags, args.bootstrap, args.seed, args.overwrite)


if __name__ == "__main__":
    main()
