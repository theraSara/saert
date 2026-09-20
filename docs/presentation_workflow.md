# Figures and tables

All current analysis notebooks use `src/utils.py`. User preference: **no
orange/blue palette and no dotted/dashed curves as the primary distinction**.
The palette uses plum, forest green, muted rose and neutral grays. Matplotlib
defaults, heatmaps, figure export and notebook table formatting are defined
centrally. Notebook tables update when you rerun them. To restyle the saved
figures, rerun the local diagnostic exporters, then `bash run.sh export-reports`.
Model fitting is unaffected. Portable notebooks display the exported figures.

## Reading rules

- Give every condition a direct label or its own heatmap column.
- Show identical conditions once and explicitly label them identical. Provo's
  two SAE boundary policies and two surprisal context regimes coincide here.
- Print the numbers inside heatmaps. State units and whether larger or smaller
  is better. Use symmetric color scales for signed predictive gains and a common
  scale across comparable panels.
- Fidelity loss spans several orders of magnitude, so its color scale is
  nonlinear, explicitly labeled; numerical annotations remain untransformed.
- The RT hook heatmap saturates colors at ±10 percentage points. Exact printed
  values and CSVs retain extreme failed predictions, so one failure does not
  wash out the useful comparisons. This changes display only, never scores.
- Main RT metric: held-out ΔR² in percentage points with paired uncertainty.
  Highlight this column; do not make users search a wide table for it.
- Bold identifies the metric to read, not statistical significance. Color also
  does not establish significance. Put uncertainty next to the effect.
- Keep raw diagnostics in CSVs or collapsed notebook sections. Add a figure
  only when it answers a specific question better than a short table.
- Scientific export formats are PDF, SVG and 300-dpi PNG, through the shared
  export function. No plot requires rerunning regression.

Current notebooks:

| Notebook | Main question |
|---|---|
| 01_surprisal_baseline | Does surprisal add predictive value? |
| 02_sae_reconstruction | How much does reconstruction alter activations and predictions? |
| 03_sae_reading_time_prediction | Do sparse features improve over surprisal and dense states? |
| 04_feature_scaling_robustness | Does the earliest-hook result survive alternative scaling? |
| 90_export_latex_tables | How should a selected result table appear in the paper? |

Earlier notebooks remain in the ignored `local_archive/notebooks/` on the
research machine. The five current notebooks read the portable `reports/`
bundle and work without original corpora or model arrays. Full local result
directories and superseded diagnostic plots are excluded from GitHub.

## LaTeX workshop

`90_export_latex_tables.ipynb` reads the current report CSVs, lets you choose tables, and
exports `.tex`, the unrounded selected `.csv`, and a `.manifest.json` recording
input hashes, column order, precision, caption, notes and emphasis. Add
`\usepackage{booktabs}` in the paper preamble. Repeated export replaces derived
tables, not analysis results. Default exports split the primary prediction table
from the SAE-versus-dense contrast to avoid one unreadable wide table.
The built-in paper tables use `table*` to span both columns of an ACL-style
document. For a small custom table, pass `wide=False`; `wide=True` requests a
spanning float. The exporter defaults to spanning when more than four columns
are selected. Text escaping and number preservation are tested; compilation
into a paper PDF requires a local TeX installation.

For any new table, use `export_table` from `src/table_export.py` on a DataFrame.
Choose columns, decimals and units explicitly. Captions/text are escaped for
LaTeX, and optional emphasis is column-based rather than automatically selecting
the largest number. The notebook includes a disabled custom-CSV example to edit.
Exports now go to `reports/tables/` and use repository-relative provenance.

```sh
bash run.sh notebook both
bash run.sh sae-notebook both
bash run.sh regression-notebook both
bash run.sh latex both
```

`scripts/create_report_notebooks.py` is a developer utility to recreate notebook sources
and **clear outputs**. Normal use should execute existing notebooks with run.sh;
do not rebuild them after making personal edits without preserving those edits.
