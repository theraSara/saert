"""Tables and exportable figures from saved, hash-verified baseline predictions."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

try:
    from .pipeline_utils import file_record, read_table, sha256_file, write_json
    from .utils import PALETTE, set_style, save_figure
except ImportError:
    from pipeline_utils import file_record, read_table, sha256_file, write_json
    from utils import PALETTE, set_style, save_figure

LABELS = {"provo": "Provo FFD", "natural_stories": "Natural Stories SPR"}
COLORS = {"long": PALETTE["plum"], "matched": PALETTE["green"]}
MODEL_LABELS = {"controls": "Controls", "long_current": "1,024: current",
    "long_spillover": "1,024: current + spillover", "matched_current": "128: current",
    "matched_spillover": "128: current + spillover"}


def load_results(directory, corpora=("provo", "natural_stories")):
    root = Path(directory)
    results = {}
    for corpus in corpora:
        path = root / f"{corpus}_baseline_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1 or manifest["corpus"] != corpus:
            raise ValueError("Unexpected baseline manifest")
        for name, record in manifest["outputs"].items():
            if Path(name).name != name or sha256_file(root/name) != record["sha256"]:
                raise ValueError(f"Changed baseline output: {name}")
        frames = {name: read_table(root/f"{corpus}_{name}.csv") for name in
                  ["predictions", "analysis_rows", "model_comparison", "contrasts", "per_text"]}
        for name, frame in frames.items():
            if set(frame.baseline_id) != {manifest["baseline_id"]}:
                raise ValueError(f"Mixed baseline runs in {name}")
        pred = frames["predictions"]
        if not pred.row_uid.is_unique or len(pred) != manifest["n_eligible"]:
            raise ValueError("Prediction coverage mismatch")
        for fold in manifest["folds"]:
            actual = set(pred.loc[pred.fold == fold["fold"], "text_id"])
            if actual != set(fold["test_text_ids"]) or actual & set(fold["train_text_ids"]):
                raise ValueError("Invalid story-held-out split")
        for row in frames["model_comparison"].itertuples():
            actual = r2_score(pred.log_primary_RT, pred[f"pred_{row.model}"])
            if not np.isclose(actual, row.r2, atol=1e-10):
                raise ValueError("Summary R2 differs from saved predictions")
        results[corpus] = {**frames, "manifest": manifest, "manifest_record": file_record(path)}
    return results


def quality_table(results):
    rows = []
    for corpus, run in results.items():
        p, m = run["predictions"], run["manifest"]
        rows.append({"corpus": corpus, "measure": m["quality"]["RT"],
            "texts": p.text_id.nunique(), "stimulus_units": m["quality"]["rows"],
            "eligible_RT_units": len(p), "excluded_RT_units": m["quality"]["rows"]-len(p),
            "first_units_scored": m["quality"]["first_words_scored"],
            "max_model_tokens_per_text": int((run["analysis_rows"].groupby("text_id").end_token_idx.max()+2).max()),
            "mean_RT_ms": p.primary_RT.mean(), "median_RT_ms": p.primary_RT.median(),
            "mean_surprisal_bits": p.surprisal.mean(),
            "max_context_score_difference_bits": (p.surprisal-p.surprisal_matched_context).abs().max()})
    return pd.DataFrame(rows)


def comparison_table(results):
    rows = []
    for corpus, run in results.items():
        for row in run["model_comparison"].itertuples():
            rows.append({"Corpus": LABELS[corpus], "Model": MODEL_LABELS.get(row.model, row.model),
                "Held-out R2": row.r2, "RMSE (log ms)": row.rmse_log_ms,
                "Delta R2 (pp)": row.delta_r2_vs_controls*100,
                "95% interval (pp)": f"[{row.delta_r2_ci_low*100:.2f}, {row.delta_r2_ci_high*100:.2f}]"})
    return pd.DataFrame(rows)


def collinearity_table(results):
    rows = []
    for corpus, run in results.items():
        suffix = "spillover" if run["manifest"]["spillover_lags"] else "current"
        for name in [f"long_{suffix}", f"matched_{suffix}"]:
            columns = run["manifest"]["models"][name]
            x = run["predictions"][columns].to_numpy(dtype=float)
            for j, col in enumerate(columns):
                variance = np.var(x[:, j])
                if variance == 0:
                    vif = np.nan
                else:
                    model = LinearRegression().fit(np.delete(x, j, axis=1), x[:, j])
                    explained = model.score(np.delete(x, j, axis=1), x[:, j])
                    vif = 1/(1-explained) if explained < 1-1e-12 else np.inf
                rows.append({"corpus": corpus, "model": name, "predictor": col, "VIF_descriptive": vif})
    return pd.DataFrame(rows)


def heterogeneity_table(results):
    rows = []
    for corpus, run in results.items():
        suffix = "spillover" if run["manifest"]["spillover_lags"] else "current"
        for model in [f"long_{suffix}", f"matched_{suffix}"]:
            group = run["per_text"].loc[lambda d: d.model == model]
            if group.empty:
                continue
            rows.append({"corpus": corpus, "model": model, "texts": len(group),
                "texts_with_lower_MSE": int((group.delta_mse_vs_controls > 0).sum()),
                "min_MSE_reduction": group.delta_mse_vs_controls.min(),
                "max_MSE_reduction": group.delta_mse_vs_controls.max()})
    return pd.DataFrame(rows)


def style():
    set_style()


def improvement_figure(results):
    set_style()
    fig, axes = plt.subplots(1, len(results), figsize=(11, 4.8), squeeze=False, sharex=True, layout="constrained")
    for ax, (corpus, run) in zip(axes.flat, results.items()):
        table = run["model_comparison"].set_index("model")
        identical = np.allclose(run["predictions"].surprisal, run["predictions"].surprisal_matched_context, atol=1e-10, rtol=0)
        labels = []
        for context in (["long"] if identical else ["matched", "long"]):
            for suffix in (["current", "spillover"] if run["manifest"]["spillover_lags"] else ["current"]):
                row = table.loc[f"{context}_{suffix}"]
                y = len(labels)
                value, lo, hi = 100*np.array([row.delta_r2_vs_controls, row.delta_r2_ci_low, row.delta_r2_ci_high])
                context_label = "Both contexts" if identical else ("128 tokens" if context == "matched" else "1,024 tokens")
                labels.append(f"{context_label}\n{'Current word' if suffix == 'current' else 'Current + 2 previous words'}")
                ax.hlines(y, lo, hi, color=COLORS[context], linewidth=2.5)
                ax.scatter(value, y, color=COLORS[context], s=55, zorder=3)
                ax.annotate(f"{value:+.2f}", (value, y), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels)-.4, -.65)
        ax.axvline(0, color=PALETTE["muted"], linewidth=1)
        ax.set_title(LABELS[corpus] + ("\nContext predictions are identical" if identical else ""))
        ax.set_xlabel("Held-out RÂ² improvement (percentage points)")
        ax.grid(axis="x", color=PALETTE["grid"])
    fig.suptitle("Does surprisal improve reading-time prediction?\nPoints: improvement Â· bars: conditional 95% text-bootstrap intervals", fontsize=13)
    return fig


def export_report(results, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {"corpus_quality": quality_table(results), "model_comparison": comparison_table(results),
              "collinearity": collinearity_table(results), "per_text_heterogeneity": heterogeneity_table(results),
              "paired_contrasts": pd.concat([r["contrasts"] for r in results.values()], ignore_index=True)}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    outputs = [f"{name}.csv" for name in tables]
    for name, func in [("heldout_improvement", improvement_figure)]:
        fig = func(results)
        outputs.extend(p.name for p in save_figure(fig, output, name))
        plt.close(fig)
    note = ["# Baseline results and figure selection", "",
        "Primary outcome: natural log of arithmetic mean RT; Provo uses FFD and Natural Stories uses SPR.",
        "Leave-one-text-out OLS, identical eligible rows and controls across context comparisons.", "",
        "## Main-text candidate", "",
        "`heldout_improvement.pdf` shows the incremental predictive value of current and spillover surprisal. "
        "Use the accompanying model-comparison table for exact RÂ²/RMSE values. Intervals are paired 95% text-bootstrap "
        "intervals conditional on fixed OOF predictions, not model-refitting or participant uncertainty.", "",
        "## Supplementary candidate", "",
        "Per-text heterogeneity and collinearity are saved as diagnostic CSVs. "
        "Only the held-out improvement figure is exported by the current design. Older plot files are superseded. "
        "LaTeX tables are exported through notebook 90.", ""]
    for corpus, run in results.items():
        note += [f"## {LABELS[corpus]}", ""]
        for row in run["model_comparison"].itertuples():
            if row.model.endswith("spillover"):
                note.append(f"- {MODEL_LABELS[row.model]}: held-out RÂ² {row.r2:.4f}; Î”RÂ² {100*row.delta_r2_vs_controls:.2f} pp "
                            f"[{100*row.delta_r2_ci_low:.2f}, {100*row.delta_r2_ci_high:.2f}].")
        contrast = run["contrasts"].query("candidate == 'long_spillover' and reference == 'matched_spillover'")
        if len(contrast):
            r = contrast.iloc[0]
            note += [f"- Paired 1,024 minus 128 context difference: {100*r.delta_r2:.3f} pp "
                     f"[{100*r.ci_low:.3f}, {100*r.ci_high:.3f}]."]
        note += [""]
    note += ["## Interpretation limits", "",
        "These are exploratory aggregate-RT baselines, not a participant-level mixed-effects analysis or a causal human mechanism. "
        "Only ten independent story clusters are available for Natural Stories. First-unit scores use BOS; missing history is "
        "zero-filled with initial-position controls. Provo contains interest areas, including punctuation/compound regions.", "",
        "A positive surprisal increment does not predict whether SAE features will add further value. "
        "The older weak-surprisal results should not be cited as validation of this corrected run."]
    (output/"figure_selection.md").write_text("\n".join(note)+"\n", encoding="utf-8")
    outputs.append("figure_selection.md")
    write_json(output/"diagnostics_manifest.json", {"baseline_sources": {c:r["manifest_record"] for c,r in results.items()},
        "source_code": [file_record(__file__), file_record(Path(__file__).with_name("utils.py"))], "outputs": {n:{"sha256":sha256_file(output/n)} for n in outputs}})
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/baseline_bos"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline_bos/diagnostics"))
    parser.add_argument("--corpus", choices=["both", "provo", "natural_stories"], default="both")
    args = parser.parse_args()
    corpora = ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]
    tables = export_report(load_results(args.baseline_dir, corpora), args.output_dir)
    print(tables["model_comparison"].to_string(index=False))
    print(f"Tables and figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
