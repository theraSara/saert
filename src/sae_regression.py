"""Nested text-held-out ridge: unpenalized baseline plus dense or sparse features."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

try:
    from .baseline import build_design, OUTCOME
    from .pipeline_utils import file_record, read_table, sha256_file, write_json, run_metadata
    from .surprisal import HOOK_FILES
    from .validate_outputs import validate_corpus
except ImportError:
    from baseline import build_design, OUTCOME
    from pipeline_utils import file_record, read_table, sha256_file, write_json, run_metadata
    from surprisal import HOOK_FILES
    from validate_outputs import validate_corpus

ROOT = Path(__file__).resolve().parent.parent


def prepare_fit(controls, y, features, config):
    """All nuisance projection, feature filtering and scales use this training split."""
    scaler = StandardScaler().fit(controls)
    c = np.column_stack([np.ones(len(y)), scaler.transform(controls)])
    u, s, _ = np.linalg.svd(c, full_matrices=False)
    q = u[:, s > s[0] * max(c.shape) * np.finfo(float).eps]
    pinv = np.linalg.pinv(c)
    base_coef = pinv @ y
    x = features.astype(np.float64)
    if sparse.issparse(x):
        means = np.asarray(x.mean(axis=0)).ravel()
        var = np.asarray(x.power(2).mean(axis=0)).ravel() - means**2
        counts = x.getnnz(axis=0)
    else:
        var = np.var(x, axis=0)
        counts = np.count_nonzero(x, axis=0)
    keep = (var > config["variance_floor"]) & (counts >= config["min_active_training_rows"])
    scale = np.sqrt(np.maximum(var[keep], config["variance_floor"]))
    z = x[:, keep]
    z = z.multiply(1/scale).tocsr() if sparse.issparse(z) else z / scale
    qz = np.asarray((z.T @ q).T)
    residual = y - q @ (q.T @ y)
    rhs = np.asarray(z.T @ residual).ravel() / len(y)
    prepared = dict(c=c, q=q, pinv=pinv, scaler=scaler, z=z, qz=qz, y=y,
                    rhs=rhs, keep=keep, scale=scale, base_coef=base_coef, n=len(y))
    if not sparse.issparse(z) and z.shape[1]:
        gram = (z.T @ z - qz.T @ qz) / len(y)
        eigenvalues, vectors = np.linalg.eigh((gram + gram.T)/2)
        prepared.update(eigenvalues=np.maximum(eigenvalues, 0), vectors=vectors)
    return prepared


def ridge_path(fit, test_controls, test_features, alphas, config, keep_coefficients=False):
    c_test = np.column_stack([np.ones(len(test_controls)), fit["scaler"].transform(test_controls)])
    raw_test = test_features[:, fit["keep"]]
    z_test = raw_test.multiply(1/fit["scale"]).tocsr() if sparse.issparse(raw_test) else raw_test/fit["scale"]
    predictions, coefficients, iterations = [], [], []
    z, qz, n = fit["z"], fit["qz"], fit["n"]
    p = z.shape[1]
    previous = np.zeros(p)
    if sparse.issparse(z):
        diag = np.maximum((np.asarray(z.power(2).sum(axis=0)).ravel() - (qz*qz).sum(axis=0))/n, 0)
    for alpha in alphas:
        count = [0]
        if p == 0:
            beta = np.zeros(0)
        elif "vectors" in fit:
            v = fit["vectors"]
            beta = v @ ((v.T @ fit["rhs"])/(fit["eigenvalues"] + alpha))
        else:
            operator = LinearOperator((p, p), matvec=lambda b: (z.T @ (z @ b)-qz.T @ (qz @ b))/n + alpha*b, dtype=np.float64)
            preconditioner = LinearOperator((p, p), matvec=lambda b: b/(diag+alpha), dtype=np.float64)
            def callback(_):
                count[0] += 1
            beta, info = cg(operator, fit["rhs"], x0=previous, M=preconditioner,
                            rtol=config["cg_rtol"], atol=0, maxiter=config["cg_maxiter"], callback=callback)
            if info != 0:
                raise RuntimeError(f"Ridge CG did not converge: alpha={alpha}, info={info}; no silent fallback")
        previous = beta
        base_coef = fit["pinv"] @ (fit["y"] - z @ beta)
        pred = c_test @ base_coef + z_test @ beta
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite ridge prediction")
        predictions.append(pred)
        iterations.append(count[0])
        if keep_coefficients:
            raw = np.zeros(len(fit["keep"]))
            raw[fit["keep"]] = beta / fit["scale"]
            c_raw = base_coef[1:] / fit["scaler"].scale_
            intercept = base_coef[0] - c_raw @ fit["scaler"].mean_
            coefficients.append((raw, np.r_[intercept, c_raw]))
    return np.column_stack(predictions), coefficients, iterations


def nested_predictions(frame, controls, features, config, *, progress=False, max_folds=None):
    y = frame[OUTCOME].to_numpy(float)
    c = frame[controls].to_numpy(float)
    groups = frame.text_id.to_numpy()
    if len(np.unique(groups)) < config["inner_splits"] + 1:
        raise ValueError("Too few stories for the specified nested group splits")
    alphas = np.array(config["alphas"], dtype=float)
    if np.any(alphas <= 0) or np.any(np.diff(alphas) >= 0):
        raise ValueError("Alphas must be positive, unique, and descending")
    prediction = np.full(len(y), np.nan)
    baseline = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1)
    folds, feature_coefs, control_coefs = [], [], []
    for fold, (train, test) in enumerate(LeaveOneGroupOut().split(c, y, groups), 1):
        if max_folds and fold > max_folds:
            break
        started = time.perf_counter()
        inner_sse = np.zeros(len(alphas))
        inner_n = 0
        inner_records = []
        for inner, (tr, va) in enumerate(GroupKFold(config["inner_splits"]).split(c[train], groups=groups[train]), 1):
            a, b = train[tr], train[va]
            fit = prepare_fit(c[a], y[a], features[a], config)
            preds, _, iterations = ridge_path(fit, c[b], features[b], alphas, config)
            sse = ((preds-y[b, None])**2).sum(axis=0)
            inner_sse += sse
            inner_n += len(b)
            inner_records.append({"inner_fold": inner, "train_text_ids": np.unique(groups[a]).astype(int).tolist(),
                "validation_text_ids": np.unique(groups[b]).astype(int).tolist(), "sse_by_alpha": sse.tolist(),
                "n_validation": len(b), "n_retained_features": int(fit["keep"].sum()), "cg_iterations": iterations})
        chosen = int(np.argmin(inner_sse))
        fit = prepare_fit(c[train], y[train], features[train], config)
        preds, coeffs, iterations = ridge_path(fit, c[test], features[test], [alphas[chosen]], config, True)
        prediction[test] = preds[:, 0]
        c_test = np.column_stack([np.ones(len(test)), fit["scaler"].transform(c[test])])
        baseline[test] = c_test @ fit["base_coef"]
        fold_ids[test] = fold
        feature_coefs.append(coeffs[0][0])
        control_coefs.append(coeffs[0][1])
        folds.append({"fold": fold, "train_text_ids": np.unique(groups[train]).astype(int).tolist(),
            "test_text_ids": np.unique(groups[test]).astype(int).tolist(), "alpha": float(alphas[chosen]),
            "inner_mse_by_alpha": (inner_sse/inner_n).tolist(), "selected_inner_mse": float(inner_sse[chosen]/inner_n),
            "inner_folds": inner_records, "n_retained_features": int(fit["keep"].sum()),
            "cg_iterations": iterations[0], "seconds": time.perf_counter()-started})
        if progress and (fold == 1 or fold % 10 == 0 or fold == len(np.unique(groups))):
            print(f"  fold {fold}/{len(np.unique(groups))}: {folds[-1]['seconds']:.1f}s, alpha={alphas[chosen]:g}", flush=True)
    return prediction, baseline, fold_ids, folds, np.array(feature_coefs), np.array(control_coefs)


def verify_saved(folder, config_hash, input_records):
    m = json.loads((folder/"manifest.json").read_text())
    if m["config_hash"] != config_hash or m["inputs"] != input_records:
        raise ValueError(f"Resume inputs/config changed: {folder}")
    if m["source_code"]["sha256"] != sha256_file(__file__):
        raise ValueError(f"Regression implementation changed; use a fresh output directory: {folder}")
    for name, rec in m["outputs"].items():
        if Path(name).name != name or sha256_file(folder/name) != rec["sha256"]:
            raise ValueError(f"Changed saved regression output: {folder/name}")
    return m


def run_one(corpus, label, kind, representation, frame, columns, paths, config, config_path, output, resume, smoke):
    folder = output/corpus/kind/label/representation
    source = paths["features"] if representation == "sae" else paths["dense"]
    inputs = {"scores": file_record(paths["scores"]), "features": file_record(source),
              "sae_manifest": file_record(paths["sae_manifest"]),
              "baseline_predictions": file_record(paths["baseline_predictions"])}
    config_hash = sha256_file(config_path)
    if (folder/"manifest.json").exists():
        if not resume:
            raise FileExistsError(f"{folder} exists; use --resume or a fresh output directory")
        verify_saved(folder, config_hash, inputs)
        print(f"Verified completed: {corpus}/{kind}/{label}/{representation}", flush=True)
        return
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Incomplete nonempty output folder: {folder}")
    features = sparse.load_npz(source) if representation == "sae" else np.load(source, allow_pickle=False)
    features = features[frame.hidden_row_idx.to_numpy(int)]
    print(f"Fitting {corpus}/{kind}/{label}/{representation}: {features.shape}", flush=True)
    values, base, fold_ids, folds, coefficients, controls = nested_predictions(frame, columns, features, config,
                                                       progress=True, max_folds=1 if smoke else None)
    if smoke:
        print(f"SMOKE ONLY: completed one outer fold in {folds[0]['seconds']:.1f}s; no research outputs written", flush=True)
        return
    if not np.isfinite(values).all():
        raise ValueError("Incomplete outer prediction coverage")
    baseline = read_table(paths["baseline_predictions"])
    if frame.row_uid.tolist() != baseline.row_uid.tolist() or not np.allclose(base, baseline.pred_matched_spillover, atol=1e-8, rtol=0):
        raise ValueError("Nested evaluator's baseline differs from verified baseline")
    result = frame[["row_uid", "text_id", "word_position", OUTCOME]].copy()
    result["fold"] = fold_ids
    result["prediction"] = values
    result["baseline_prediction"] = base
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ridge_", dir=folder.parent) as temporary:
        temp = Path(temporary)
        result.to_csv(temp/"predictions.csv", index=False)
        np.savez_compressed(temp/"coefficients.npz", feature_coef_raw=coefficients, control_coef_raw=controls)
        write_json(temp/"folds.json", {"controls": ["intercept"]+columns, "folds": folds})
        manifest = {**run_metadata(), "schema_version": 1, "corpus": corpus, "label": label,
                    "hook": next(h for h, l in HOOK_FILES.items() if l == label),
                    "kind": kind, "representation": representation, "config": config, "config_hash": config_hash,
                    "inputs": inputs, "source_code": file_record(__file__), "n_rows": len(frame),
                    "n_features_input": features.shape[1], "n_outer_folds": len(folds),
                    "coefficient_policy": "raw feature units, original feature column IDs; intercept then baseline columns; row order is folds.json",
                    "outputs": {p.name: {"sha256": sha256_file(p)} for p in temp.iterdir()}}
        write_json(temp/"manifest.json", manifest)
        for p in temp.iterdir():
            if p.name != "manifest.json":
                p.replace(folder/p.name)
        (temp/"manifest.json").replace(folder/"manifest.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", choices=["both", "provo", "natural_stories"], default="both")
    p.add_argument("--data-dir", type=Path, default=ROOT/"data/prepared_v2")
    p.add_argument("--results-dir", type=Path, default=ROOT/"results/surprisal_bos")
    p.add_argument("--baseline-dir", type=Path, default=ROOT/"results/baseline_bos")
    p.add_argument("--sae-dir", type=Path, default=ROOT/"results/sae_bos")
    p.add_argument("--output-dir", type=Path, default=ROOT/"results/sae_regression_v2")
    p.add_argument("--config", type=Path, default=ROOT/"configs/sae_regression.json")
    p.add_argument("--hooks", nargs="+", choices=list(HOOK_FILES.values()), default=list(HOOK_FILES.values()))
    p.add_argument("--states", nargs="+", choices=["prefix", "post"], default=["prefix", "post"])
    p.add_argument("--representations", nargs="+", choices=["dense", "sae"], default=["dense", "sae"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="One outer fold per combination; prints timing only and writes no results")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if config["context"] != "matched" or config["spillover_lags"] != 2:
        raise ValueError("This version verifies the frozen matched-context, two-lag baseline")
    corpora = ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]
    with threadpool_limits(limits=config["threads"]):
        for corpus in corpora:
            validate_corpus(corpus, args.data_dir, args.results_dir)
            baseline_manifest = json.loads((args.baseline_dir/f"{corpus}_baseline_manifest.json").read_text())
            for name, rec in baseline_manifest["outputs"].items():
                if sha256_file(args.baseline_dir/name) != rec["sha256"]:
                    raise ValueError(f"Baseline changed: {name}")
            scores = args.results_dir/f"{corpus}_surprisal.csv"
            frame, _, models = build_design(read_table(scores), config["spillover_lags"])
            for label in args.hooks:
                folder = args.sae_dir/corpus/label
                m = json.loads((folder/"manifest.json").read_text())
                if sha256_file(args.results_dir/f"{corpus}_extraction_manifest.json") != m["inputs"]["extraction_manifest"]["sha256"]:
                    raise ValueError("SAE extraction source changed")
                for name, rec in m["outputs"].items():
                    if sha256_file(folder/name) != rec["sha256"]:
                        raise ValueError(f"SAE output changed: {folder/name}")
                rows = read_table(folder/"rows.csv")
                if rows.iloc[frame.hidden_row_idx.to_numpy(int)].row_uid.tolist() != frame.row_uid.tolist():
                    raise ValueError("SAE/source row alignment mismatch")
                for kind in args.states:
                    stem = "prefix_hidden" if kind == "prefix" else "hidden"
                    paths = {"scores": scores, "features": folder/f"{kind}_features.npz",
                             "dense": args.results_dir/f"{corpus}_{stem}_{label}.npy",
                             "sae_manifest": folder/"manifest.json",
                             "baseline_predictions": args.baseline_dir/f"{corpus}_predictions.csv"}
                    if sha256_file(paths["dense"]) != m["inputs"][kind]["sha256"]:
                        raise ValueError("SAE and dense states have different provenance")
                    for representation in args.representations:
                        run_one(corpus, label, kind, representation, frame, models["matched_spillover"], paths,
                                config, args.config, args.output_dir, args.resume, args.smoke)


if __name__ == "__main__":
    main()
