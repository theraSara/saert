"""Validate SAE outputs and export descriptive fidelity tables and figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
try:
    from .pipeline_utils import file_record, read_table, run_metadata, sha256_file, write_json
    from .surprisal import HOOK_FILES
    from .utils import set_style, annotated_heatmap, save_figure
except ImportError:
    from pipeline_utils import file_record, read_table, run_metadata, sha256_file, write_json
    from surprisal import HOOK_FILES
    from utils import set_style, annotated_heatmap, save_figure


def load_results(root, corpora, labels=None):
    root = Path(root)
    labels = labels or list(HOOK_FILES.values())
    geometry, behaviors, sources = [], [], []
    sampling = {}
    for corpus in corpora:
        for label in labels:
            folder = root / corpus / label
            manifest_path = folder / "manifest.json"
            m = json.loads(manifest_path.read_text())
            if m["schema_version"] != 1 or m["corpus"] != corpus or HOOK_FILES[m["hook"]] != label:
                raise ValueError(f"Manifest mismatch: {folder}")
            for name, record in m["outputs"].items():
                if Path(name).name != name or sha256_file(folder / name) != record["sha256"]:
                    raise ValueError(f"Changed SAE output: {folder / name}")
            for record in m["inputs"].values():
                if sha256_file(record["path"]) != record["sha256"]:
                    raise ValueError(f"Changed SAE input: {record['path']}")
            source = Path(m["inputs"]["extraction_manifest"]["path"])
            originals = read_table(source.parent / f"{corpus}_hidden_rows.csv")
            rows = read_table(folder / "rows.csv")
            if rows.row_uid.tolist() != originals.row_uid.tolist():
                raise ValueError("SAE rows differ from source extraction")
            if not np.array_equal(rows.hidden_row_idx, np.arange(len(rows))):
                raise ValueError("SAE rows are not in array order")
            for kind in ["post", "prefix"]:
                features = sparse.load_npz(folder / f"{kind}_features.npz")
                diag = read_table(folder / f"{kind}_reconstruction.csv")
                if features.shape != (m["n_rows"], m["feature_width"]) or features.dtype != np.float32:
                    raise ValueError("Invalid sparse feature shape/dtype")
                if not np.isfinite(features.data).all() or (features.data <= 0).any():
                    raise ValueError("Invalid sparse activation values")
                if diag.row_uid.tolist() != rows.row_uid.tolist() or not np.array_equal(features.getnnz(axis=1), diag.n_active):
                    raise ValueError("Sparse features and reconstruction diagnostics disagree")
            geometry.append(read_table(folder / "geometry.csv"))
            if "behavior.csv" in m["outputs"]:
                signature = json.dumps(m["fidelity_windows"], sort_keys=True)
                if corpus in sampling and sampling[corpus] != signature:
                    raise ValueError("Fidelity samples differ across hooks")
                sampling[corpus] = signature
                b = read_table(folder / "behavior.csv")
                weights = b.n_scored_tokens.to_numpy(float)
                average = lambda name: float(np.average(b[name], weights=weights))
                for policy, suffix in [("protect_bos", ""), ("protect_window_start", "_preserve_window_start")]:
                    original = average("original_nll_bits")
                    recon = average(f"reconstruction{suffix}_nll_bits")
                    zero = average(f"zero{suffix}_nll_bits")
                    denom = zero - original
                    behaviors.append({"corpus": corpus, "label": label, "hook": m["hook"], "policy": policy,
                    "n_windows": len(b), "n_scored_tokens": int(weights.sum()),
                    "original_nll_bits": original, "reconstruction_nll_bits": recon, "zero_nll_bits": zero,
                    "nll_increase_bits": recon-original, "kl_bits": average(f"reconstruction{suffix}_kl_bits"),
                    "loss_recovered": (zero-recon)/denom if denom > 1e-8 else np.nan,
                    "identity_kl_bits": average("identity_kl_bits")})
            sources.append(file_record(manifest_path))
    return pd.concat(geometry, ignore_index=True), pd.DataFrame(behaviors), sources


def export_reports(root, output, corpora, labels=None):
    geometry, behavior, sources = load_results(root, corpora, labels)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    geometry.to_csv(output / "geometry_summary.csv", index=False)
    behavior.to_csv(output / "behavior_summary.csv", index=False)
    set_style()
    from matplotlib.colors import SymLogNorm
    labels = labels or list(HOOK_FILES.values())
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout="constrained")
    gcols, bcols = {}, {}
    for corpus in corpora:
        display_name = "Provo" if corpus == "provo" else "Natural Stories"
        for kind, group, name in [("prefix", "text_token", "Before word"), ("post", "all", "After word")]:
            part = geometry[(geometry.corpus == corpus) & (geometry.kind == kind) & (geometry.group == group)].set_index("label")
            gcols[f"{display_name}\n{name}"] = part.reindex(labels).mean_relative_l2_error*100
        if not behavior.empty:
            b = behavior[behavior.corpus == corpus].pivot(index="label", columns="policy", values="nll_increase_bits").reindex(labels)
            if np.allclose(b.protect_bos, b.protect_window_start, atol=1e-10, rtol=0):
                bcols[f"{display_name}\nBoth policies\n(identical)"] = b.protect_bos
            else:
                bcols[f"{display_name}\nPreserve BOS\nonly"] = b.protect_bos
                bcols[f"{display_name}\nPreserve each\nwindow start"] = b.protect_window_start
    geo = pd.DataFrame(gcols)
    geo.index = [str(list(HOOK_FILES.values()).index(label)) for label in labels]
    annotated_heatmap(axes[0], geo, fmt=".1f", title="How much of each state changes?",
                      colorbar_label="Average relative error (%) · lower is better")
    if bcols:
        beh = pd.DataFrame(bcols)
        beh.index = geo.index
        extent = max(float(abs(beh).max().max()), .01)
        annotated_heatmap(axes[1], beh, fmt="+.3f", signed=True, errors=True,
            norm=SymLogNorm(linthresh=.1, vmin=-extent, vmax=extent),
            title="How much do predictions change?",
            colorbar_label="Extra loss (bits/token) · lower is better; nonlinear color scale")
    else:
        axes[1].text(.5, .5, "Behavioral fidelity was not run", ha="center", transform=axes[1].transAxes)
        axes[1].axis("off")
    for ax in axes:
        ax.set_ylabel("Completed transformer blocks")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("SAE fidelity: read the numbers, then the color\nBefore-word geometry excludes BOS prefixes; behavioral checks use four sampled windows per corpus by default", fontsize=12)
    generated = save_figure(fig, output, "sae_fidelity")
    files = ["geometry_summary.csv", "behavior_summary.csv"] + [p.name for p in generated]
    write_json(output / "diagnostics_manifest.json", {**run_metadata(), "inputs": sources,
        "source_code": [file_record(__file__), file_record(Path(__file__).with_name("utils.py"))], "outputs": {name: file_record(output / name) for name in files},
        "interpretation": "Descriptive quality diagnostics; no uncertainty intervals, RT effects, or causal human claims."})
    return geometry, behavior, fig


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-dir", type=Path, default=root / "results/sae_bos")
    p.add_argument("--output-dir", type=Path, default=root / "results/sae_bos/diagnostics")
    p.add_argument("--corpus", choices=["both", "provo", "natural_stories"], default="both")
    p.add_argument("--hooks", nargs="+", choices=list(HOOK_FILES.values()))
    args = p.parse_args()
    corpora = ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]
    g, b, fig = export_reports(args.sae_dir, args.output_dir, corpora, args.hooks)
    plt.close(fig)
    print(b.to_string(index=False))


if __name__ == "__main__":
    main()
