"""Reusable, data-only LaTeX table export with explicit columns and provenance."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
try:
    from .pipeline_utils import file_record, read_table, sha256_file, write_json
except ImportError:
    from pipeline_utils import file_record, read_table, sha256_file, write_json


def escape_latex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(ch, ch) for ch in str(value))


def portable_record(path):
    record = file_record(path)
    root = Path(__file__).resolve().parents[1]
    absolute = Path(path).resolve()
    record['path'] = absolute.relative_to(root).as_posix() if absolute.is_relative_to(root) else absolute.name
    return record


def export_table(frame, destination, *, caption, label, columns=None, rename=None, decimals=None,
                 signed=(), emphasize=(), notes="", sources=(), wide=None):
    """No automatic best-result selection or significance stars. Explicit presentation."""
    if not re.fullmatch(r"[A-Za-z0-9:._-]+", label):
        raise ValueError("Use a plain LaTeX label without whitespace or commands")
    columns = list(columns or frame.columns)
    if len(set(columns)) != len(columns) or not set(columns).issubset(frame.columns):
        raise ValueError("Missing or duplicate export columns")
    chosen = frame[columns].copy()
    decimals = decimals or {}
    formatted = chosen.copy().astype(object)
    for col in columns:
        for index, value in chosen[col].items():
            if pd.isna(value):
                text = "---"
            elif isinstance(value, (float, np.floating, int, np.integer)) and col in decimals:
                text = format(value, f"{'+' if col in signed else ''}.{decimals[col]}f")
            else:
                text = str(value)
            text = escape_latex(text)
            if col in emphasize and not pd.isna(value):
                text = r"\textbf{" + text + "}"
            formatted.at[index, col] = text
    formatted.columns = [escape_latex((rename or {}).get(c, c)) for c in columns]
    wide = len(columns) > 4 if wide is None else bool(wide)
    latex = formatted.to_latex(index=False, escape=False, caption=escape_latex(caption), label=label,
                               position="t" if wide else "htbp", column_format="l"*len(columns))
    if wide:
        latex = latex.replace(r"\begin{table}", r"\begin{table*}").replace(r"\end{table}", r"\end{table*}")
    if notes:
        latex = latex.replace(r"\end{tabular}", r"\end{tabular}"+"\n\\par\\smallskip\n\\begin{minipage}{\\linewidth}\\footnotesize\n"+escape_latex(notes)+"\n\\end{minipage}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(latex, encoding="utf-8", newline="\n")
    chosen.rename(columns=rename or {}).to_csv(destination.with_suffix(".csv"), index=False, lineterminator="\n")
    write_json(destination.with_suffix(".manifest.json"), {"inputs": [portable_record(p) for p in sources],
        "source_code": portable_record(__file__), "caption": caption, "label": label, "columns": columns,
        "rename": rename or {}, "decimals": decimals, "signed": list(signed), "emphasize": list(emphasize),
        "notes": notes, "wide": wide,
        "outputs": {p.name: portable_record(p) for p in [destination, destination.with_suffix('.csv')]}})
    manifest_path = destination.with_suffix('.manifest.json')
    manifest_path.write_bytes(manifest_path.read_bytes().replace(b'\r\n', b'\n'))
    return latex


def verified_csv(path):
    path = Path(path)
    manifest_path = path.parent/"diagnostics_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(path) != manifest["outputs"][path.name]["sha256"]:
        raise ValueError(f"Report has changed since generation: {path}")
    return read_table(path)


def project_tables(baseline_dir, sae_dir, regression_dir):
    """Curated defaults. The notebook can also call export_table on any DataFrame."""
    tables = {}
    path = Path(baseline_dir)/"diagnostics/model_comparison.csv"
    if path.exists():
        frame = verified_csv(path)
        frame = frame[frame.Model.str.contains("spillover")].copy()
        frame["Model"] = frame.Model.str.split(":").str[0] + " tokens"
        # Provo's two contexts are exactly identical; state this instead of two duplicate rows.
        duplicate = frame.Corpus.str.startswith("Provo")
        provo = frame[duplicate].copy()
        if len(provo) == 2 and np.allclose(provo[["Held-out R2", "Delta R2 (pp)"]].iloc[0], provo[["Held-out R2", "Delta R2 (pp)"]].iloc[1], atol=1e-12, rtol=0):
            provo = provo.iloc[:1].copy()
            provo["Model"] = "Both (identical)"
        frame = pd.concat([provo, frame[~duplicate]], ignore_index=True)
        tables["baseline"] = {"frame": frame, "sources": [path], "columns": ["Corpus", "Model", "Held-out R2", "Delta R2 (pp)", "95% interval (pp)"],
            "rename": {"Model": "Context", "Held-out R2": "R2", "Delta R2 (pp)": "Gain (pp)"}, "wide": True,
            "caption": "Surprisal improvement beyond lexical and position controls.", "label": "tab:baseline",
            "decimals": {"Held-out R2": 3, "Delta R2 (pp)": 2}, "signed": ["Delta R2 (pp)"], "emphasize": ["Delta R2 (pp)"],
            "notes": "Current and two previous-word surprisals. Intervals: paired text bootstrap of fixed held-out predictions, excluding refitting and participant uncertainty. pp = percentage points."}
    path = Path(sae_dir)/"diagnostics/behavior_summary.csv"
    if path.exists():
        frame = verified_csv(path)
        frame = frame[frame.policy.eq("protect_window_start")][["corpus", "label", "nll_increase_bits", "kl_bits"]]
        frame["corpus"] = frame.corpus.map({"provo": "Provo", "natural_stories": "Natural Stories"})
        tables["sae_fidelity"] = {"frame": frame, "sources": [path],
            "wide": True,
            "rename": {"corpus": "Corpus", "label": "Hook label", "nll_increase_bits": "Added NLL (bits/token)", "kl_bits": "KL (bits/token)"},
            "caption": "Reconstruction fidelity with window-start states preserved.", "label": "tab:sae-fidelity",
            "decimals": {"nll_increase_bits": 3, "kl_bits": 3}, "signed": ["nll_increase_bits"], "emphasize": ["nll_increase_bits"],
            "notes": "Descriptive sampled-window diagnostic. Lower added loss is better. This restricted intervention does not validate excluded window-start states; consult the notebook for both policies."}
    path = Path(regression_dir)/"diagnostics/primary_results.csv"
    if path.exists():
        frame = verified_csv(path)
        tables["sae_prediction"] = {"frame": frame, "sources": [path],
            "wide": True, "rename": {"SAE gain (pp)": "SAE gain", "Dense gain (pp)": "Dense gain", "SAE 95% interval": "95% interval", "Texts improved": "Better texts"},
            "columns": ["Corpus", "State", "SAE gain (pp)", "SAE 95% interval", "Dense gain (pp)", "Texts improved"],
            "caption": "Nested story-held-out representation gains beyond the surprisal baseline.", "label": "tab:sae-prediction",
            "decimals": {"SAE gain (pp)": 2, "Dense gain (pp)": 2}, "signed": ["SAE gain (pp)", "Dense gain (pp)"], "emphasize": ["SAE gain (pp)"],
            "notes": "Gains and intervals are R2 percentage points. Hook and ridge strength selected in inner grouped validation. Intervals condition on fixed outer-fold predictions. Better texts refers to SAE versus the surprisal baseline. Bold identifies the primary metric, not significance."}
        tables["sae_dense_contrast"] = {"frame": frame, "sources": [path],
            "wide": True,
            "columns": ["Corpus", "State", "SAE minus dense (pp)", "Difference 95% interval"],
            "caption": "Paired SAE-versus-dense prediction comparison.", "label": "tab:sae-dense",
            "decimals": {"SAE minus dense (pp)": 2}, "signed": ["SAE minus dense (pp)"], "emphasize": ["SAE minus dense (pp)"],
            "notes": "Positive values favor SAE. Each representation selects its hook using inner validation. These conditional bootstrap intervals do not account for all exploratory comparisons."}
    return tables
