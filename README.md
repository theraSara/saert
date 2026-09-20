# SparseRT

Sparse GPT-2 features and human reading times, using Provo and Natural Stories.
This repository contains the current research code, **portable result notebooks**,
and the progress presentation. Raw corpora, model weights and full experiment
outputs stay local.

## Present the current work

- [Progress slides](presentations/2026-09-21-progress/saert-progress.pptx)
- [Meeting guide and likely questions](presentations/2026-09-21-progress/meeting-guide.md)
- [Speaker notes](presentations/2026-09-21-progress/speaker-notes.md)
- [Current findings](docs/sae_prediction_findings.md) and [project plan](docs/project_plan.md)

The deck works without Python. The notebooks include saved outputs and can also
run using the small report bundle, without GPUs or model downloads.

## Get it on another computer

First clone:

```bash
git clone --depth 1 https://github.com/theraSara/saert.git
cd saert
```

If you already cloned it:

```bash
cd saert
git pull --ff-only origin main
```

To run the report notebooks on macOS/Linux or Windows Git Bash:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows Git Bash instead:
# source .venv/Scripts/activate
python -m pip install -r requirements-view.txt
python -m jupyter lab notebooks
```

Alternatively, use a Python 3.11 Conda environment and install the same viewing
requirements. Choose that environment's kernel in Jupyter.

## Notebook order

| Notebook | Question |
|---|---|
| [01_surprisal_baseline](notebooks/01_surprisal_baseline.ipynb) | Does surprisal improve prediction? |
| [02_sae_reconstruction](notebooks/02_sae_reconstruction.ipynb) | Does SAE reconstruction preserve model behavior? |
| [03_sae_reading_time_prediction](notebooks/03_sae_reading_time_prediction.ipynb) | Do sparse features add information beyond surprisal and dense states? |
| [04_feature_scaling_robustness](notebooks/04_feature_scaling_robustness.ipynb) | Does the first-hook result survive alternative scaling? |
| [90_export_latex_tables](notebooks/90_export_latex_tables.ipynb) | Export selected result tables for a paper |

These notebooks read `reports/`, use the shared style in `src/utils.py`, and do
not refit models. Verify the bundle or execute a notebook from the terminal:

```bash
bash run.sh verify-reports
bash run.sh regression-notebook
bash run.sh scaling-notebook
bash run.sh latex
```

## Repository layout

```text
configs/          Analysis specifications fixed for each experiment
src/              Research modules, report utilities and shared plotting style
scripts/          Curated report export, notebook generation and slide builders
tests/            Alignment, evaluation, leakage and portability checks
notebooks/        Five current, executed report notebooks
reports/          Small aggregate CSVs, selected figures and LaTeX tables
docs/             Methods, findings, project plan and setup instructions
presentations/    Dated progress meeting materials
run.sh            Documented commands for research and reports
```

Local only, ignored by Git: `data/`, `results/`, `naturalstories/`,
`local_archive/`, environments, downloaded weights, activations, fitted models,
and temporary presentation builds. Historical notebooks remain in the local
archive; they are superseded by the numbered notebooks above.

## Reproduce the full research pipeline

This requires the original corpora and `requirements.txt`, not just the viewing
requirements. See [environment setup](docs/setup.md), [data and extraction
methods](docs/corrected_pipeline.md), and `bash run.sh help`.

```bash
bash run.sh prepare both
bash run.sh extract both
bash run.sh verify both
bash run.sh baseline both
bash run.sh diagnostics both
bash run.sh sae both
bash run.sh sae-report both
bash run.sh regression both
bash run.sh regression-report both
bash run.sh scaling both
bash run.sh export-reports
```

Existing fits remain protected. Use fresh output directories for repeat or
changed experiments. `export-reports` copies a checked allowlist from the
completed default experiment directories; it never publishes raw trials,
word-level predictions, activations or full result trees. A fresh full run needs
all specified stages before that export. There is no external upload in this
command: it only writes the local `reports/` directory.

The fitting modules keep their established names to preserve recorded code
hashes and resume behavior. `src/data.py` prepares data, `src/surprisal.py`
extracts model quantities, `src/baseline.py` fits baselines, and the `sae_*`
modules handle extraction, prediction and diagnostics.

## Interpretation

These are exploratory aggregate-RT results. Held-out predictive gains do not
establish feature meaning or human causal mechanisms. Reported bootstrap
intervals condition on saved predictions and exclude participant and model-
refitting uncertainty. The targeted scaling follow-up does not replace the
original all-hook analysis. See the methods documents for exact comparisons.
