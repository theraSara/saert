"""Foldwise residualized-RT prediction: surprisal, dense states and SAE features on RT'.

Two-stage design (the "RT prime" analysis)
------------------------------------------
For every outer leave-one-text-out fold, using TRAINING TEXTS ONLY:

  Stage 1  OLS   log RT ~ stage-1 predictors          ->  RT' = log RT - fitted
  Stage 2  model RT' ~ representation                 ->  predict RT' on the held-out text

Residualization is refit inside every outer fold AND inside every inner
alpha-selection fold, so no held-out text ever informs its own target.

Targets
-------
resid_controls            stage 1 = lexical/position controls (+ lexical spillover)
                          asks: once lexical effects are removed, what do surprisal,
                          dense states and SAE features each explain?
resid_controls_surprisal  stage 1 = controls + matched surprisal + 2 surprisal lags
                          asks: what do dense/SAE explain that surprisal leaves behind?

Relation to sae_regression.py
-----------------------------
sae_regression.py fits ONE joint model (controls + surprisal unpenalized, features
ridge-penalized), so its gains are the one-stage answer to the
resid_controls_surprisal question. Here the stage-1 variables are removed from the
OUTCOME ONLY: the stage-2 features are never adjusted for the stage-1 controls.
This is the classic psycholinguistic RT' procedure, and it is conservative:
predictors correlated with the stage-1 variables lose the shared variance.
Surviving gains are therefore robust to the "it's just lexical frequency" critique
(for LINEAR lexical effects; nonlinear lexical structure is not ruled out).

Within stage 2, the combined models fit surprisal and the feature block JOINTLY.
prepare_fit/ridge_path solve that joint problem via Frisch-Waugh-Lovell projection,
which is a solution method, not a change of model: predictions equal those of a
direct joint solver to machine precision (verified, max |diff| ~ 1e-13).

Representations
---------------
surprisal         OLS on matched surprisal + 2 lags                    (resid_controls only)
dense             ridge on the 768-d hidden state (unpenalized intercept)
sae               ridge on 24,576 SAE activations (unpenalized intercept)
surprisal+dense   joint: surprisal unpenalized + raw dense features ridge (resid_controls only)
surprisal+sae     joint: surprisal unpenalized + raw SAE features ridge   (resid_controls only)
Ridge reuses prepare_fit/ridge_path from sae_regression.py; alpha is chosen by
inner GroupKFold exactly as in the main regression.

Metrics (paired text bootstrap of fixed out-of-fold predictions, as in baseline.py)
  delta_r2_pp     R2 gain on the raw log-RT scale over stage 1 alone
  r2_resid_pp     R2 of the residualized target itself
  vs_surprisal_pp paired gain over the surprisal-only model (resid_controls only)

Usage
-----
  python src/residual_rt_prediction.py --corpus both --resume
  python src/residual_rt_prediction.py --corpus provo --hooks L01 --targets resid_controls
  python src/residual_rt_prediction.py --summarize-only
"""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

try:
    from .baseline import build_design, OUTCOME
    from .sae_regression import prepare_fit, ridge_path
    from .pipeline_utils import file_record, read_table, sha256_file, write_json, run_metadata
    from .surprisal import HOOK_FILES
    from .validate_outputs import validate_corpus
except ImportError:
    from baseline import build_design, OUTCOME
    from sae_regression import prepare_fit, ridge_path
    from pipeline_utils import file_record, read_table, sha256_file, write_json, run_metadata
    from surprisal import HOOK_FILES
    from validate_outputs import validate_corpus

ROOT = Path(__file__).resolve().parent.parent

# target name -> (build_design model whose columns form stage 1, baseline prediction column)
TARGETS = {
    "resid_controls":           ("controls",          "pred_controls"),
    "resid_controls_surprisal": ("matched_spillover", "pred_matched_spillover"),
}
REPRESENTATIONS = ["surprisal", "dense", "sae", "surprisal+dense", "surprisal+sae"]
STAGE2_DESIGN = {
    "surprisal": "OLS of RT' on the surprisal block (current + 2 lags)",
    "dense": "ridge of RT' on dense hidden state; intercept unpenalized",
    "sae": "ridge of RT' on SAE activations; intercept unpenalized",
    "surprisal+dense": "joint model of RT': surprisal block unpenalized + dense features ridge-penalized "
                       "(solved via prepare_fit/ridge_path FWL projection; identical to a direct joint solve)",
    "surprisal+sae": "joint model of RT': surprisal block unpenalized + SAE features ridge-penalized "
                     "(solved via prepare_fit/ridge_path FWL projection; identical to a direct joint solve)",
}
# Hooks chosen by nested CV in the main regression; used for the printed summary.
PRIMARY_HOOKS = {"provo": "L01", "natural_stories": "L08"}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: OLS residualization (mirrors baseline.fit_model: scaler + OLS on train)
# ══════════════════════════════════════════════════════════════════════════════

def ols_fit(x, y):
    scaler = StandardScaler().fit(x)
    design = np.column_stack([np.ones(len(y)), scaler.transform(x)])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    return scaler, coef


def ols_predict(model, x):
    scaler, coef = model
    return np.column_stack([np.ones(len(x)), scaler.transform(x)]) @ coef


def oof_stage1(y, c1, groups):
    """Out-of-fold stage-1 predictions (used to verify against baseline.py)."""
    out = np.full(len(y), np.nan)
    for train, test in LeaveOneGroupOut().split(c1, y, groups):
        out[test] = ols_predict(ols_fit(c1[train], y[train]), c1[test])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: predict the residual target
# ══════════════════════════════════════════════════════════════════════════════

def stage2_path(rep, a, b, r_a, surp, feats, const, alphas, config):
    """Fit on rows a (target r_a), predict rows b. Returns (len(b), len(alphas))."""
    if rep == "surprisal":
        p = ols_predict(ols_fit(surp[a], r_a), surp[b])
        return np.repeat(p[:, None], len(alphas), axis=1)
    # Surprisal enters as unpenalized nuisance columns; otherwise an inert
    # constant column (prepare_fit needs >=1 control column; the intercept is
    # always added, so a zero column only contributes a dropped null direction).
    nuis = surp if rep.startswith("surprisal+") else const
    fit = prepare_fit(nuis[a], r_a, feats[a], config)
    preds, _, _ = ridge_path(fit, nuis[b], feats[b], alphas, config)
    return preds


def nested_residual(y, c1, surp, feats, const, rep, groups, config, progress=False):
    alphas = np.array(config["alphas"], dtype=float)
    stage1 = np.full(len(y), np.nan)
    pred = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1)
    records = []
    n_texts = len(np.unique(groups))
    for fold, (train, test) in enumerate(LeaveOneGroupOut().split(c1, y, groups), 1):
        started = time.perf_counter()
        m1 = ols_fit(c1[train], y[train])
        r_train = y[train] - ols_predict(m1, c1[train])
        stage1[test] = ols_predict(m1, c1[test])
        if rep == "surprisal":
            alpha, inner_mse = None, None
            chosen_alphas = alphas[:1]
        else:
            sse = np.zeros(len(alphas))
            n_val = 0
            for tr, va in GroupKFold(config["inner_splits"]).split(train, groups=groups[train]):
                a, b = train[tr], train[va]
                m1_inner = ols_fit(c1[a], y[a])          # stage 1 refit: no inner leakage
                r_a = y[a] - ols_predict(m1_inner, c1[a])
                r_b = y[b] - ols_predict(m1_inner, c1[b])
                p = stage2_path(rep, a, b, r_a, surp, feats, const, alphas, config)
                sse += ((p - r_b[:, None]) ** 2).sum(axis=0)
                n_val += len(b)
            chosen = int(np.argmin(sse))
            alpha, inner_mse = float(alphas[chosen]), (sse / n_val).tolist()
            chosen_alphas = alphas[chosen:chosen + 1]
        p = stage2_path(rep, train, test, r_train, surp, feats, const, chosen_alphas, config)
        pred[test] = p[:, 0]
        fold_ids[test] = fold
        records.append({"fold": fold, "test_text_ids": np.unique(groups[test]).astype(int).tolist(),
                        "alpha": alpha, "inner_mse_by_alpha": inner_mse,
                        "seconds": time.perf_counter() - started})
        if progress and (fold == 1 or fold % 10 == 0 or fold == n_texts):
            a_str = "OLS" if alpha is None else f"alpha={alpha:g}"
            print(f"    fold {fold}/{n_texts}: {records[-1]['seconds']:.1f}s, {a_str}", flush=True)
    return stage1, pred, fold_ids, records


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation: paired text bootstrap of fixed out-of-fold predictions
# ══════════════════════════════════════════════════════════════════════════════

def paired_scores(y, stage1, preds, groups, bootstrap=2000, seed=42, reference=None):
    """preds: {name: stage-2 residual predictions}. Returns {name: metrics}."""
    names = list(preds)
    texts = np.unique(groups)
    per_text = []
    for t in texts:
        m = groups == t
        yt, s1 = y[m], stage1[m]
        r = yt - s1
        row = [m.sum(), yt.sum(), (yt ** 2).sum(), r.sum(), (r ** 2).sum(), ((yt - s1) ** 2).sum()]
        row += [((r - preds[n][m]) ** 2).sum() for n in names]
        per_text.append(row)
    stats = np.array(per_text)
    idx_sse = {n: 6 + j for j, n in enumerate(names)}

    def metrics(s):
        sst = s[..., 2] - s[..., 1] ** 2 / s[..., 0]
        ssr = s[..., 4] - s[..., 3] ** 2 / s[..., 0]
        return sst, ssr

    total = stats.sum(axis=0)
    sst, ssr = metrics(total)
    rng = np.random.default_rng(seed)
    boot = stats[rng.integers(0, len(stats), size=(bootstrap, len(stats)))].sum(axis=1)
    b_sst, b_ssr = metrics(boot)
    valid = (b_sst > 0) & (b_ssr > 0)
    boot, b_sst, b_ssr = boot[valid], b_sst[valid], b_ssr[valid]

    out = {}
    for n in names:
        k = idx_sse[n]
        d = (total[5] - total[k]) / sst
        bd = (boot[:, 5] - boot[:, k]) / b_sst
        rr = 1 - total[k] / ssr
        brr = 1 - boot[:, k] / b_ssr
        res = {"delta_r2_pp": 100 * d,
               "ci_low_pp": 100 * np.quantile(bd, .025), "ci_high_pp": 100 * np.quantile(bd, .975),
               "r2_resid_pp": 100 * rr,
               "r2_resid_ci_low_pp": 100 * np.quantile(brr, .025),
               "r2_resid_ci_high_pp": 100 * np.quantile(brr, .975),
               "texts_improved": int((stats[:, k] < stats[:, 5]).sum()), "n_texts": len(texts)}
        if reference is not None and reference in preds and n != reference:
            kr = idx_sse[reference]
            res["vs_surprisal_pp"] = 100 * (total[kr] - total[k]) / sst
            bc = (boot[:, kr] - boot[:, k]) / b_sst
            res["vs_surprisal_ci_low_pp"] = 100 * np.quantile(bc, .025)
            res["vs_surprisal_ci_high_pp"] = 100 * np.quantile(bc, .975)
        out[n] = res
    return out


# ══════════════════════════════════════════════════════════════════════════════
# One (corpus, target, [kind, hook], representation) run with provenance
# ══════════════════════════════════════════════════════════════════════════════

def run_one(folder, meta, y, c1, surp, feats, const, rep, groups, frame, config, config_hash,
            inputs, resume):
    if (folder / "manifest.json").exists():
        if not resume:
            raise FileExistsError(f"{folder} exists; use --resume or a fresh --output-dir")
        m = json.loads((folder / "manifest.json").read_text())
        if m["config_hash"] != config_hash or m["source_code"]["sha256"] != sha256_file(__file__):
            raise ValueError(f"Config or implementation changed since {folder}; use a fresh output dir")
        for name, rec in m["outputs"].items():
            if sha256_file(folder / name) != rec["sha256"]:
                raise ValueError(f"Changed saved output: {folder / name}")
        print(f"  Verified completed: {folder.relative_to(folder.parents[3])}", flush=True)
        return
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Incomplete nonempty output folder: {folder}")
    print(f"  Fitting {meta['corpus']}/{meta['target']}/{meta.get('kind', '-')}/"
          f"{meta.get('label', '-')}/{rep}", flush=True)
    stage1, pred, fold_ids, records = nested_residual(y, c1, surp, feats, const, rep, groups,
                                                      config, progress=True)
    if not (np.isfinite(stage1).all() and np.isfinite(pred).all()):
        raise ValueError("Incomplete or nonfinite out-of-fold predictions")
    result = frame[["row_uid", "text_id", "word_position"]].copy()
    result[OUTCOME] = y
    result["fold"] = fold_ids
    result["stage1_prediction"] = stage1
    result["resid_target"] = y - stage1
    result["stage2_prediction"] = pred
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="resid_", dir=folder.parent) as tmp:
        tmp = Path(tmp)
        result.to_csv(tmp / "predictions.csv", index=False)
        write_json(tmp / "folds.json", {"folds": records})
        manifest = {**run_metadata(), "schema_version": 1, **meta, "representation": rep,
                    "config": config, "config_hash": config_hash, "inputs": inputs,
                    "source_code": file_record(__file__), "n_rows": len(frame),
                    "design": "two-stage; stage 1 OLS refit in every outer and inner training split; "
                              "stage-1 variables removed from the outcome only (conservative RT' estimator)",
                    "stage2_design": STAGE2_DESIGN[rep],
                    "outputs": {p.name: {"sha256": sha256_file(p)} for p in tmp.iterdir()}}
        write_json(tmp / "manifest.json", manifest)
        for p in tmp.iterdir():
            if p.name != "manifest.json":
                p.replace(folder / p.name)
        (tmp / "manifest.json").replace(folder / "manifest.json")


# ══════════════════════════════════════════════════════════════════════════════
# Summary across all saved runs
# ══════════════════════════════════════════════════════════════════════════════

def summarize(output, bootstrap, seed):
    rows = []
    for corpus_dir in sorted(p for p in output.iterdir() if p.is_dir()):
        for target_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
            ref_path = target_dir / "surprisal" / "predictions.csv"
            ref = read_table(ref_path) if ref_path.exists() else None
            for pred_path in sorted(target_dir.rglob("predictions.csv")):
                rel = pred_path.parent.relative_to(target_dir).parts
                df = read_table(pred_path)
                preds = {"model": df.stage2_prediction.to_numpy(float)}
                if ref is not None and rel != ("surprisal",):
                    if ref.row_uid.tolist() != df.row_uid.tolist() or not np.allclose(
                            ref.stage1_prediction, df.stage1_prediction, atol=1e-8, rtol=0):
                        raise ValueError(f"Stage-1 mismatch vs surprisal reference: {pred_path}")
                    preds["surprisal_ref"] = ref.stage2_prediction.to_numpy(float)
                score = paired_scores(df[OUTCOME].to_numpy(float), df.stage1_prediction.to_numpy(float),
                                      preds, df.text_id.to_numpy(), bootstrap, seed,
                                      reference="surprisal_ref" if "surprisal_ref" in preds else None)
                kind, label, rep = (("-", "-", rel[0]) if len(rel) == 1 else rel)
                rows.append({"corpus": corpus_dir.name, "target": target_dir.name, "kind": kind,
                             "label": label, "representation": rep, **score["model"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary.csv", index=False)
    return summary


def print_primary(summary):
    if summary.empty:
        return
    print("\n" + "═" * 92)
    print("  PRIMARY HOOKS (L01 Provo / L08 NS, chosen by nested CV in the main regression)")
    print("  ΔR² = gain on raw log-RT over stage 1 alone; 95% paired text-bootstrap CI")
    print("═" * 92)
    for (corpus, target), grp in summary.groupby(["corpus", "target"], sort=False):
        hook = PRIMARY_HOOKS.get(corpus)
        sel = grp[(grp.label == hook) | (grp.representation == "surprisal")]
        print(f"\n  {corpus} | target = {target}")
        for _, r in sel.iterrows():
            tag = r.representation if r.kind == "-" else f"{r.representation} ({r.kind})"
            vs = (f"   vs surprisal {r.vs_surprisal_pp:+.2f} [{r.vs_surprisal_ci_low_pp:+.2f}, "
                  f"{r.vs_surprisal_ci_high_pp:+.2f}]" if "vs_surprisal_pp" in r and pd.notna(r.get("vs_surprisal_pp"))
                  else "")
            print(f"    {tag:26s} ΔR² {r.delta_r2_pp:+6.2f} pp [{r.ci_low_pp:+.2f}, {r.ci_high_pp:+.2f}]"
                  f"  texts {r.texts_improved}/{r.n_texts}{vs}")
    print("\n  Full per-hook table: summary.csv (choosing the best hook from it is post hoc).")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", choices=["both", "provo", "natural_stories"], default="both")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/prepared_v2")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results/surprisal_bos")
    p.add_argument("--baseline-dir", type=Path, default=ROOT / "results/baseline_bos")
    p.add_argument("--sae-dir", type=Path, default=ROOT / "results/sae_bos")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/residual_rt")
    p.add_argument("--config", type=Path, default=ROOT / "configs/sae_regression.json")
    p.add_argument("--targets", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    p.add_argument("--hooks", nargs="+", choices=list(HOOK_FILES.values()), default=list(HOOK_FILES.values()))
    p.add_argument("--states", nargs="+", choices=["prefix", "post"], default=["post", "prefix"])
    p.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=REPRESENTATIONS)
    p.add_argument("--include-ns-l01", action="store_true", help="Include NS L01 (excluded by default: out-of-distribution SAE inputs, see NOTES.md)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()

    config = json.loads(args.config.read_text())
    if config["context"] != "matched" or config["spillover_lags"] != 2:
        raise ValueError("Expected the frozen matched-context, two-lag configuration")
    bootstrap, seed = int(config.get("bootstrap", 2000)), 42
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        config_hash = sha256_file(args.config)
        corpora = ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]
        with threadpool_limits(limits=config["threads"]):
            for corpus in corpora:
                print(f"\n{'═' * 60}\n  {corpus.upper()}\n{'═' * 60}", flush=True)
                validate_corpus(corpus, args.data_dir, args.results_dir)
                scores = args.results_dir / f"{corpus}_surprisal.csv"
                frame, _, models = build_design(read_table(scores), config["spillover_lags"])
                y = frame[OUTCOME].to_numpy(float)
                groups = frame.text_id.to_numpy()
                surp_cols = [c for c in models["matched_spillover"] if c not in models["controls"]]
                surp = frame[surp_cols].to_numpy(float)
                const = np.zeros((len(frame), 1))
                baseline = read_table(args.baseline_dir / f"{corpus}_predictions.csv")
                if baseline.row_uid.tolist() != frame.row_uid.tolist():
                    raise ValueError("Baseline/design row alignment mismatch")

                for target in args.targets:
                    model_key, baseline_col = TARGETS[target]
                    stage1_cols = models[model_key]
                    c1 = frame[stage1_cols].to_numpy(float)
                    # Stage 1 must reproduce the verified baseline exactly.
                    check = oof_stage1(y, c1, groups)
                    if not np.allclose(check, baseline[baseline_col], atol=1e-7, rtol=0):
                        raise ValueError(f"Stage 1 for {target} differs from baseline {baseline_col}")
                    print(f"\n  target={target}: stage 1 matches baseline '{baseline_col}' ✓", flush=True)
                    meta = {"corpus": corpus, "target": target, "stage1_columns": stage1_cols,
                            "surprisal_columns": surp_cols}
                    reps = list(args.representations)
                    if target == "resid_controls_surprisal":
                        dropped = [r for r in reps if r.startswith("surprisal")]
                        reps = [r for r in reps if not r.startswith("surprisal")]
                        if dropped:
                            print(f"  (skipping {dropped}: surprisal is already in stage 1)", flush=True)
                    base_inputs = {"scores": file_record(scores), "baseline_predictions": file_record(args.baseline_dir / f"{corpus}_predictions.csv")}

                    if "surprisal" in reps:
                        run_one(args.output_dir / corpus / target / "surprisal", meta, y, c1, surp, None,
                                const, "surprisal", groups, frame, config, config_hash, base_inputs, args.resume)

                    feature_reps = [r for r in reps if r != "surprisal"]
                    if not feature_reps:
                        continue
                    for label in args.hooks:
                        if corpus == "natural_stories" and label == "L01" and not args.include_ns_l01:
                            print("  (skipping NS L01: out-of-distribution SAE inputs; --include-ns-l01 to force)")
                            continue
                        folder = args.sae_dir / corpus / label
                        rows = read_table(folder / "rows.csv")
                        if rows.iloc[frame.hidden_row_idx.to_numpy(int)].row_uid.tolist() != frame.row_uid.tolist():
                            raise ValueError(f"SAE/source row alignment mismatch: {folder}")
                        for kind in args.states:
                            stem = "prefix_hidden" if kind == "prefix" else "hidden"
                            sources = {"dense": args.results_dir / f"{corpus}_{stem}_{label}.npy", "sae": folder / f"{kind}_features.npz"}
                            loaded = {}
                            for rep in feature_reps:
                                src = "sae" if rep.endswith("sae") else "dense"
                                if src not in loaded:
                                    raw = (sparse.load_npz(sources[src]).tocsr() if src == "sae" else np.load(sources[src], allow_pickle=False))
                                    loaded[src] = raw[frame.hidden_row_idx.to_numpy(int)]
                                inputs = {**base_inputs, "features": file_record(sources[src])}
                                run_one(args.output_dir / corpus / target / kind / label / rep,
                                        {**meta, "kind": kind, "label": label}, y, c1, surp, loaded[src],
                                        const, rep, groups, frame, config, config_hash, inputs, args.resume)
                            del loaded

    print("\nSummarizing all saved runs (paired text bootstrap)...", flush=True)
    summary = summarize(args.output_dir, bootstrap, seed)
    print_primary(summary)
    print(f"\nDone. Results: {args.output_dir}/summary.csv")


if __name__ == "__main__":
    main()