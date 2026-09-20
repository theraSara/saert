"""Saved nested-regression comparisons; inner-selected hooks, never outer winners."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from .pipeline_utils import file_record, read_table, sha256_file, write_json, run_metadata
    from .surprisal import HOOK_FILES
    from .utils import PALETTE, CORPUS_LABELS, set_style, save_figure, annotated_heatmap
except ImportError:
    from pipeline_utils import file_record, read_table, sha256_file, write_json, run_metadata
    from surprisal import HOOK_FILES
    from utils import PALETTE, CORPUS_LABELS, set_style, save_figure, annotated_heatmap


def paired_score(frame, reference="baseline_prediction", bootstrap=2000, seed=42):
    y, p, b = [frame[c].to_numpy(float) for c in ["log_primary_RT", "prediction", reference]]
    sst = ((y-y.mean())**2).sum()
    actual = ((y-b)**2).sum()-((y-p)**2).sum()
    records = []
    for _, group in frame.groupby("text_id", sort=True):
        yg, pg, bg = [group[c].to_numpy(float) for c in ["log_primary_RT", "prediction", reference]]
        records.append([len(yg), yg.sum(), (yg*yg).sum(), ((yg-bg)**2).sum()-((yg-pg)**2).sum()])
    a = np.array(records)
    rng = np.random.default_rng(seed)
    sampled = a[rng.integers(0, len(a), size=(bootstrap, len(a)))].sum(axis=1)
    variance = sampled[:, 2]-sampled[:, 1]**2/sampled[:, 0]
    interval = np.quantile(sampled[variance > 0, 3]/variance[variance > 0], [.025, .975])
    return {"r2": 1-((y-p)**2).sum()/sst, "delta_r2": actual/sst,
            "ci_low": interval[0], "ci_high": interval[1], "rmse_log_ms": np.sqrt(np.mean((y-p)**2)),
            "mae_log_ms": np.mean(abs(y-p)), "texts_improved": int((a[:, 3] > 0).sum()),
            "n_texts": len(a), "n_rows": len(y),
            "max_abs_error_log_ms": float(abs(y-p).max()),
            "largest_error_share_of_sse": float(((y-p)**2).max()/max(((y-p)**2).sum(), 1e-30))}


def audit_extreme_predictions(root, summary):
    """Descriptive audit only: >2 log-unit errors trigger inspection, never exclusion."""
    from scipy import sparse
    try:
        from .baseline import build_design
    except ImportError:
        from baseline import build_design
    records = []
    flagged = summary.query("representation == 'sae' and label != 'inner_selected' and max_abs_error_log_ms > 2")
    for run in flagged.itertuples():
        folder = Path(root)/run.corpus/run.kind/run.label/'sae'
        m = json.loads((folder/'manifest.json').read_text())
        folds = json.loads((folder/'folds.json').read_text())
        frame, _, _ = build_design(read_table(m['inputs']['scores']['path']), m['config']['spillover_lags'])
        p = read_table(folder/'predictions.csv')
        if frame.row_uid.tolist() != p.row_uid.tolist():
            raise ValueError('Outlier audit row alignment mismatch')
        x = sparse.load_npz(m['inputs']['features']['path']).astype(float)[frame.hidden_row_idx.to_numpy(int)]
        worst = int(np.argmax(abs(p.prediction-p.log_primary_RT)))
        fold = next(f for f in folds['folds'] if f['fold'] == int(p.iloc[worst].fold))
        fold_index = folds['folds'].index(fold)
        with np.load(folder/'coefficients.npz') as co:
            coefficients = co['feature_coef_raw'][fold_index]
            controls = co['control_coef_raw'][fold_index]
        train = frame.text_id.isin(fold['train_text_ids']).to_numpy()
        test = frame.text_id.isin(fold['test_text_ids']).to_numpy()
        c = np.column_stack([np.ones(len(frame)), frame[folds['controls'][1:]].to_numpy(float)])
        reconstructed = c[test] @ controls + x[test] @ coefficients
        if not np.allclose(reconstructed, p.prediction[test], rtol=1e-10, atol=1e-8):
            raise ValueError('Saved coefficients do not reproduce anomalous predictions')
        activations = x[worst].toarray().ravel()
        contributions = activations*coefficients
        feature = int(np.argmax(abs(contributions)))
        training_values = x[train, feature].toarray().ravel()
        records.append(dict(corpus=run.corpus, kind=run.kind, label=run.label,
            row_uid=p.iloc[worst].row_uid, word=frame.iloc[worst].word,
            prediction=float(p.iloc[worst].prediction), observed_log_rt=float(p.iloc[worst].log_primary_RT),
            feature_id=feature, activation=activations[feature],
            training_max=training_values.max(), training_sd=training_values.std(),
            training_active_rows=int(np.count_nonzero(training_values)),
            feature_contribution=contributions[feature],
            coefficient_reproduction_error=float(abs(reconstructed-p.prediction[test]).max())))
    return pd.DataFrame(records, columns=['corpus', 'kind', 'label', 'row_uid', 'word', 'prediction',
        'observed_log_rt', 'feature_id', 'activation', 'training_max', 'training_sd',
        'training_active_rows', 'feature_contribution', 'coefficient_reproduction_error'])


def choose_inner_hooks(runs):
    """Select each outer fold's hook solely by its inner-validation MSE."""
    labels = list(runs)
    output, choices = [], []
    first = runs[labels[0]]["predictions"]
    for fold in sorted(first.fold.unique()):
        candidates = [(next(f for f in runs[label]["folds"] if f["fold"] == fold), label) for label in labels]
        chosen, label = min(candidates, key=lambda pair: pair[0]["selected_inner_mse"])
        output.append(runs[label]["predictions"].loc[lambda d: d.fold == fold])
        choices.append({"fold": int(fold), "label": label, "alpha": chosen["alpha"],
                        "inner_validation_mse": chosen["selected_inner_mse"]})
    result = pd.concat(output).set_index("row_uid").loc[first.row_uid].reset_index()
    return result, choices


def collect(root, corpora):
    root = Path(root)
    summary, choices, selected, sources, tuning = [], [], [], [], []
    verified = {}
    all_runs = {}
    configuration = None
    for corpus in corpora:
        all_runs[corpus] = {}
        for kind in ["prefix", "post"]:
            all_runs[corpus][kind] = {}
            for representation in ["dense", "sae"]:
                runs = {}
                for label in HOOK_FILES.values():
                    folder = root/corpus/kind/label/representation
                    m = json.loads((folder/"manifest.json").read_text())
                    if configuration is None:
                        configuration = m["config"]
                    if m["config"] != configuration or (m["corpus"], m["kind"], m["label"], m["representation"]) != (corpus, kind, label, representation):
                        raise ValueError("Mixed regression specifications")
                    for name, rec in m["outputs"].items():
                        if Path(name).name != name or sha256_file(folder/name) != rec["sha256"]:
                            raise ValueError(f"Changed regression artifact: {folder/name}")
                    for rec in m["inputs"].values():
                        if rec["path"] not in verified:
                            verified[rec["path"]] = sha256_file(rec["path"])
                        if verified[rec["path"]] != rec["sha256"]:
                            raise ValueError(f"Changed regression source: {rec['path']}")
                    p = read_table(folder/"predictions.csv")
                    folds = json.loads((folder/"folds.json").read_text())["folds"]
                    if not p.row_uid.is_unique or len(p) != m["n_rows"] or not np.isfinite(p[["prediction", "baseline_prediction"]]).all().all():
                        raise ValueError("Incomplete predictions")
                    for f in folds:
                        test = set(p.loc[p.fold == f["fold"], "text_id"])
                        train = set(f["train_text_ids"])
                        if test != set(f["test_text_ids"]) or test & train:
                            raise ValueError("Outer split leakage")
                        for inner in f["inner_folds"]:
                            tr, va = set(inner["train_text_ids"]), set(inner["validation_text_ids"])
                            if tr & va or tr | va != train:
                                raise ValueError("Inner split leakage")
                        tuning.append({"corpus": corpus, "kind": kind, "representation": representation,
                                       "label": label, "fold": f["fold"], "alpha": f["alpha"],
                                       "retained_features": f["n_retained_features"], "seconds": f["seconds"]})
                    if runs:
                        reference = next(iter(runs.values()))["predictions"]
                        if p.row_uid.tolist() != reference.row_uid.tolist() or not np.allclose(p[["log_primary_RT", "baseline_prediction"]], reference[["log_primary_RT", "baseline_prediction"]], atol=1e-10):
                            raise ValueError("Hooks use different evaluation rows or outcomes")
                    runs[label] = {"predictions": p, "folds": folds}
                    summary.append({"corpus": corpus, "kind": kind, "representation": representation,
                                    "label": label, **paired_score(p, bootstrap=configuration["bootstrap"], seed=configuration["seed"])})
                    sources.append(file_record(folder/"manifest.json"))
                chosen, selection = choose_inner_hooks(runs)
                all_runs[corpus][kind][representation] = chosen
                summary.append({"corpus": corpus, "kind": kind, "representation": representation,
                                "label": "inner_selected", **paired_score(chosen, bootstrap=configuration["bootstrap"], seed=configuration["seed"])})
                choices.extend([{"corpus": corpus, "kind": kind, "representation": representation, **r} for r in selection])
                selected.append(chosen.assign(corpus=corpus, kind=kind, representation=representation))
    paired = []
    for corpus in corpora:
        for kind in ["prefix", "post"]:
            a, b = [all_runs[corpus][kind][r] for r in ["sae", "dense"]]
            if a.row_uid.tolist() != b.row_uid.tolist():
                raise ValueError("Dense and SAE evaluation rows differ")
            compare = a.assign(dense_prediction=b.prediction.to_numpy())
            paired.append({"corpus": corpus, "kind": kind, **paired_score(compare, reference="dense_prediction",
                           bootstrap=configuration["bootstrap"], seed=configuration["seed"])})
    return pd.DataFrame(summary), pd.DataFrame(paired), pd.DataFrame(choices), pd.concat(selected), pd.DataFrame(tuning), sources


def primary_table(summary, paired):
    rows = []
    primary = summary[summary.label == "inner_selected"]
    for (corpus, kind), group in primary.groupby(["corpus", "kind"], sort=False):
        s = group.set_index("representation")
        a, d = s.loc["sae"], s.loc["dense"]
        contrast = paired[(paired.corpus == corpus) & (paired.kind == kind)].iloc[0]
        rows.append({"Corpus": CORPUS_LABELS[corpus], "State": "Before word" if kind == "prefix" else "After word",
            "SAE gain (pp)": 100*a.delta_r2, "SAE 95% interval": f"[{100*a.ci_low:+.2f}, {100*a.ci_high:+.2f}]",
            "Dense gain (pp)": 100*d.delta_r2, "SAE minus dense (pp)": 100*contrast.delta_r2,
            "Difference 95% interval": f"[{100*contrast.ci_low:+.2f}, {100*contrast.ci_high:+.2f}]",
            "Texts improved": f"{int(a.texts_improved)}/{int(a.n_texts)}"})
    return pd.DataFrame(rows)


def prediction_figure(summary, corpora):
    set_style()
    fig, axes = plt.subplots(1, len(corpora), figsize=(11, 4.4), squeeze=False, layout="constrained", sharex=True)
    for ax, corpus in zip(axes.flat, corpora):
        sub = summary[(summary.corpus == corpus) & (summary.label == "inner_selected")].set_index(["kind", "representation"])
        labels = []
        for i, (kind, rep) in enumerate([(k, r) for k in ["prefix", "post"] for r in ["dense", "sae"]]):
            row = sub.loc[(kind, rep)]
            color = PALETTE["plum"] if rep == "sae" else PALETTE["green"]
            ax.hlines(i, 100*row.ci_low, 100*row.ci_high, color=color, linewidth=2.5)
            ax.scatter(100*row.delta_r2, i, color=color, s=50, zorder=3)
            labels.append(f"{'Before' if kind == 'prefix' else 'After'} word · {rep.upper() if rep == 'sae' else 'dense'}")
            ax.annotate(f"{100*row.delta_r2:+.2f}", (100*row.delta_r2, i), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
        ax.set_yticks(range(4), labels)
        ax.set_ylim(3.65, -.65)
        ax.axvline(0, color=PALETTE["muted"], linewidth=1)
        ax.grid(axis="x", color=PALETTE["grid"])
        ax.set_title(CORPUS_LABELS[corpus])
        ax.set_xlabel("Held-out R² gain over surprisal baseline (pp)")
    fig.suptitle("Do representations improve reading-time prediction?\nHook and regularization chosen using training stories only", fontsize=13)
    return fig


def layer_figure(summary, corpora):
    set_style()
    fig, axes = plt.subplots(1, len(corpora), figsize=(12, 7), squeeze=False, layout="constrained")
    from matplotlib.colors import Normalize
    sub = summary[summary.label != "inner_selected"]
    # Display-only saturation prevents a failed model from washing out every other cell.
    # Printed values and exported CSVs remain unchanged, including extreme failures.
    norm = Normalize(-10, 10, clip=True)
    for ax, corpus in zip(axes.flat, corpora):
        values = sub[sub.corpus == corpus].pivot(index="label", columns=["kind", "representation"], values="delta_r2").reindex(list(HOOK_FILES.values()))
        values = values.reindex(columns=pd.MultiIndex.from_tuples([(k, r) for k in ["prefix", "post"] for r in ["dense", "sae"]]))*100
        values.columns = ["Before\ndense", "Before\nSAE", "After\ndense", "After\nSAE"]
        values.index = [str(i) for i in range(13)]
        annotated_heatmap(ax, values, fmt="+.2f", signed=True, norm=norm, title=CORPUS_LABELS[corpus],
                          colorbar_label="ΔR² (pp); colors saturate at ±10")
        ax.set_ylabel("Completed transformer blocks")
    fig.suptitle("Where does additional predictive information appear?\nExploratory hook profile; each number is a held-out improvement", fontsize=13)
    return fig


def export_reports(root, output, corpora):
    summary, paired, choices, predictions, tuning, sources = collect(root, corpora)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = {"model_comparison": summary, "sae_vs_dense": paired, "selected_hooks": choices,
              "selected_predictions": predictions, "tuning": tuning, "primary_results": primary_table(summary, paired),
              "feature_shift_audit": audit_extreme_predictions(root, summary)}
    for name, df in tables.items():
        df.to_csv(output/f"{name}.csv", index=False)
    figures = {"heldout_representation_gain": prediction_figure(summary, corpora), "layer_gain_heatmap": layer_figure(summary, corpora)}
    outputs = [output/f"{name}.csv" for name in tables]
    for name, figure in figures.items():
        outputs.extend(save_figure(figure, output, name))
    write_json(output/"diagnostics_manifest.json", {**run_metadata(), "inputs": sources,
        "source_code": [file_record(__file__), file_record(Path(__file__).with_name("utils.py"))],
        "outputs": {p.name: file_record(p) for p in outputs},
        "uncertainty": "95% paired text bootstrap conditional on fixed OOF predictions; no model refitting or participant uncertainty",
        "selection": "Primary results select hooks on inner-validation MSE only; layer profiles are exploratory"})
    return tables, figures


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regression-dir", type=Path, default=root/"results/sae_regression_v2")
    p.add_argument("--output-dir", type=Path, default=root/"results/sae_regression_v2/diagnostics")
    p.add_argument("--corpus", choices=["both", "provo", "natural_stories"], default="both")
    args = p.parse_args()
    corpora = ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]
    tables, figures = export_reports(args.regression_dir, args.output_dir, corpora)
    for fig in figures.values():
        plt.close(fig)
    print(tables["primary_results"].to_string(index=False))


if __name__ == "__main__":
    main()
