#!/usr/bin/env bash
# SparseRT: run from Git Bash (Windows), macOS, or Linux after `conda activate saert`.
# Corrected baseline, SAE extraction, and fidelity diagnostics.
set -euo pipefail
script_dir=.
if [[ "${BASH_SOURCE[0]}" == */* ]]; then script_dir="${BASH_SOURCE[0]%/*}"; fi
cd -- "$script_dir"

usage() {
  while IFS= read -r help_line; do printf '%s\n' "$help_line"; done <<'HELP'
Usage: bash run.sh STAGE [CORPUS]

Stages:
  check     Show the interpreter; check dependencies and imports.
  test      Run the offline correctness tests (no model download).
  prepare   Prepare complete stimuli and RTs; Provo uses IA_ID/IA_LABEL and FFD.
  extract   Extract BOS-conditioned surprisal, post-word states, and prefix states.
  verify    Check saved manifests, row alignment, scores, and activation arrays.
  baseline  Fit leave-one-text-out baselines for both context settings.
  diagnostics Export tables and PDF/SVG/PNG figures from saved baseline results.
  notebook  Open/run the portable baseline report (no model/data downloads).
  sae-pilot Extract L03 for both state types and check fidelity (separate directory).
  sae       Extract all 13 hooks, retaining every feature, with fidelity checks.
  sae-report Validate SAE files and export reconstruction/fidelity diagnostics.
  sae-notebook Execute 02_sae_reconstruction.ipynb from saved SAE outputs.
  regression Fit nested dense/SAE ridge comparisons; resume verified completed fits.
  regression-notebook Execute 03_sae_reading_time_prediction.ipynb from saved predictions.
  latex     Execute 90_export_latex_tables.ipynb (table exports only).
  scaling   Run the fixed-L01 scaling sensitivity to a NEW SCALING_ROOT (both corpora).
  scaling-notebook Execute 04_feature_scaling_robustness.ipynb from saved follow-up results.
  export-reports Copy a verified allowlist of local aggregate results into reports/.
  verify-reports Verify the shareable report bundle without raw data or models.
  regression-report Regenerate regression diagnostics from full local predictions.
  all       Run research setup through baseline diagnostics (requires raw corpora).
  help      Show these instructions. Running without arguments also shows help.

CORPUS is both (default), natural_stories, or provo.

For the outputs already generated:
  bash run.sh verify both
  bash run.sh baseline both
  bash run.sh notebook both

For a fresh run:
  bash run.sh check
  bash run.sh test
  bash run.sh prepare both
  bash run.sh extract both
  bash run.sh verify both

Or run all stages to NEW directories:
  PREPARED_ROOT=data/prepared_repeat RESULTS_ROOT=results/surprisal_repeat BASELINE_ROOT=results/baseline_repeat bash run.sh all both

Optional environment variables:
  PYTHON_BIN       Python executable (default: python from the active environment).
  RAW_ROOT         Raw corpus directory (default: data).
  PREPARED_ROOT    Prepared corpus directory (default: data/prepared_v2).
  RESULTS_ROOT     Extraction directory (default: results/surprisal_bos).
  BASELINE_ROOT    Baseline output directory (default: results/baseline_bos).
  REPORT_ROOT      Figure/table directory (default: BASELINE_ROOT/diagnostics).
  SAE_ROOT         Full SAE output directory (default: results/sae_bos).
  SAE_PILOT_ROOT   Pilot output directory (default: results/sae_pilot).
  SAE_REVISION     Immutable SAE checkpoint SHA (a pinned default is supplied).
  FIDELITY_WINDOWS Windows sampled per corpus, at most one per text (default: 4).
  SAE_BATCH_SIZE   Encoding batch size (default: 128).
  REGRESSION_ROOT  Nested regression outputs (default: results/sae_regression_v2).
  RIDGE_CONFIG     Frozen regression config (default: configs/sae_regression.json).
  LaTeX notebook exports use reports/tables/; full experiment outputs stay in results/.
  SCALING_ROOT     Fixed-L01 follow-up (default: results/scaling_sensitivity_v1).
  SPILLOVER_LAGS   Previous-word lags, 0..3 (default: 2; fixed before fitting).
  BOOTSTRAP        Paired text-bootstrap replicates (default: 2000).
  PROVO_RT         FFD (default, matches our runs), GZD, or TRT.
  GPT2_REVISION    Defaults to the exact GPT-2 commit used in our completed runs.
  DEVICE           cpu, cuda, or mps; SAE defaults to CPU, surprisal auto-selects.
  HF_HUB_OFFLINE   Set to 1 to use cached Hugging Face files only.
  OVERWRITE        Set to 1 to explicitly replace the selected stage's outputs.

Different RT measures should use separate output directories. Changing preparation
or extraction invalidates downstream artifacts; rerun every dependent stage.
This runner never silently skips existing files or enables overwrite by default.

If Git Bash reports a missing SSL_CERT_FILE, run this in your terminal first:
  export SSL_CERT_FILE="$(python -s -m certifi)"
That uses the installed CA bundle; it does not disable HTTPS verification.
Keep an institution-provided certificate bundle if your network requires it.
HELP
}

stage="${1:-help}"
corpus="${2:-both}"
if (( $# > 2 )); then usage >&2; exit 2; fi
case "$stage" in help|-h|--help) usage; exit 0 ;; esac
case "$stage" in check|test|prepare|extract|verify|baseline|diagnostics|notebook|sae-pilot|sae|sae-report|sae-notebook|regression|regression-notebook|regression-report|latex|scaling|scaling-notebook|export-reports|verify-reports|all) ;; *) usage >&2; exit 2 ;; esac
case "$corpus" in both|natural_stories|provo) ;; *) echo "Unknown corpus: $corpus" >&2; exit 2 ;; esac

python_bin="${PYTHON_BIN:-python}"
raw_root="${RAW_ROOT:-data}"
prepared_root="${PREPARED_ROOT:-data/prepared_v2}"
results_root="${RESULTS_ROOT:-results/surprisal_bos}"
baseline_root="${BASELINE_ROOT:-results/baseline_bos}"
report_root="${REPORT_ROOT:-$baseline_root/diagnostics}"
sae_root="${SAE_ROOT:-results/sae_bos}"
sae_pilot_root="${SAE_PILOT_ROOT:-results/sae_pilot}"
sae_revision="${SAE_REVISION:-57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9}"
regression_root="${REGRESSION_ROOT:-results/sae_regression_v2}"
ridge_config="${RIDGE_CONFIG:-configs/sae_regression.json}"
provo_rt="${PROVO_RT:-FFD}"
# Pinned to the revision in both completed BOS extraction manifests.
revision="${GPT2_REVISION:-607a30d783dfa663caf39e06633721c8d4cfcd7e}"
export PYTHONNOUSERSITE=1
# The completed runs used four CPU threads; these affect performance, not windows.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

overwrite_args=()
case "${OVERWRITE:-0}" in
  0) ;;
  1) overwrite_args=(--overwrite) ;;
  *) echo "OVERWRITE must be 0 or 1" >&2; exit 2 ;;
esac
case "$provo_rt" in FFD|GZD|TRT) ;; *) echo "PROVO_RT must be FFD, GZD, or TRT" >&2; exit 2 ;; esac
device_args=()
if [[ -n "${DEVICE:-}" ]]; then device_args=(--device "$DEVICE"); fi

run_command() {
  # Print the actual command for inspection and copying; no eval or shell expansion.
  printf '\nRunning:'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

check_environment() {
  run_command "$python_bin" -s -c 'import sys; print("Python:", sys.executable); print(sys.version)'
  run_command "$python_bin" -s -m pip check
  run_command "$python_bin" -s -c 'import torch, pandas, numpy, scipy, sklearn, statsmodels, transformers, transformer_lens, sae_lens; print("Imports OK; Torch:", torch.__version__); print("CUDA:", torch.cuda.is_available())'
}

run_tests() {
  run_command "$python_bin" -s -m unittest discover -s tests -v
}

prepare_data() {
  # Natural Stories: all_stories.tok + processed_wordinfo.tsv (+ participant RTs).
  # Provo: original official CSV; IA_ID/IA_LABEL preserve displayed regions.
  provo_args=()
  if [[ "$corpus" != natural_stories ]]; then provo_args=(--provo-rt "$provo_rt"); fi
  run_command "$python_bin" -s src/data.py \
    --corpus "$corpus" --data-dir "$raw_root" --output-dir "$prepared_root" \
    --tokenizer-revision "$revision" "${provo_args[@]}" "${overwrite_args[@]}"
}

extract_surprisal() {
  # One BOS per story; all windows include BOS in their token budget when present.
  # Long-context surprisal: 1024/512; states and matched surprisal: 128/64.
  # --save-prefix adds pre-word arrays alongside final-subtoken arrays.
  run_command "$python_bin" -s src/surprisal.py \
    --corpus "$corpus" --data-dir "$prepared_root" --output-dir "$results_root" \
    --revision "$revision" --window-size 1024 --stride 512 \
    --hidden-context 128 --hidden-stride 64 --save-prefix \
    "${device_args[@]}" "${overwrite_args[@]}"
}

verify_outputs() {
  run_command "$python_bin" -s src/validate_outputs.py \
    --corpus "$corpus" --data-dir "$prepared_root" --results-dir "$results_root"
}

fit_baseline() {
  run_command "$python_bin" -s src/baseline.py \
    --corpus "$corpus" --data-dir "$prepared_root" --results-dir "$results_root" \
    --output-dir "$baseline_root" --spillover-lags "${SPILLOVER_LAGS:-2}" \
    --bootstrap "${BOOTSTRAP:-2000}" --seed 42 "${overwrite_args[@]}"
}

export_diagnostics() {
  run_command "$python_bin" -s src/baseline_diagnostics.py \
    --corpus "$corpus" --baseline-dir "$baseline_root" --output-dir "$report_root"
}

execute_notebook() {
  run_command "$python_bin" -s src/run_notebook.py --analysis baseline
}

extract_sae() {
  target_root="$sae_root"
  hook_args=()
  if [[ "$stage" == sae-pilot ]]; then target_root="$sae_pilot_root"; hook_args=(--hooks L03); fi
  run_command "$python_bin" -s src/sae_extraction.py \
    --corpus "$corpus" --data-dir "$prepared_root" --results-dir "$results_root" \
    --output-dir "$target_root" --revision "$sae_revision" \
    --fidelity-windows "${FIDELITY_WINDOWS:-4}" --batch-size "${SAE_BATCH_SIZE:-128}" \
    --seed 42 --threads "${OMP_NUM_THREADS:-4}" \
    "${hook_args[@]}" "${device_args[@]}" "${overwrite_args[@]}"
}

sae_report() {
  run_command "$python_bin" -s src/sae_diagnostics.py \
    --corpus "$corpus" --sae-dir "$sae_root" --output-dir "$sae_root/diagnostics"
}

sae_notebook() {
  run_command "$python_bin" -s src/run_notebook.py --analysis sae
}

fit_sae_regression() {
  run_command "$python_bin" -s src/sae_regression.py \
    --corpus "$corpus" --data-dir "$prepared_root" --results-dir "$results_root" \
    --baseline-dir "$baseline_root" --sae-dir "$sae_root" --output-dir "$regression_root" \
    --config "$ridge_config" --resume
}

regression_notebook() {
  run_command "$python_bin" -s src/run_notebook.py --analysis regression
}

latex_notebook() {
  run_command "$python_bin" -s src/run_notebook.py --analysis latex
}

case "$stage" in
  check) check_environment ;;
  test) run_tests ;;
  prepare) prepare_data ;;
  extract) extract_surprisal ;;
  verify) verify_outputs ;;
  baseline) fit_baseline ;;
  diagnostics) export_diagnostics ;;
  notebook) execute_notebook ;;
  sae-pilot)
    extract_sae
    run_command "$python_bin" -s src/sae_diagnostics.py \
      --corpus "$corpus" --sae-dir "$sae_pilot_root" \
      --output-dir "$sae_pilot_root/diagnostics" --hooks L03
    ;;
  sae) extract_sae ;;
  sae-report) sae_report ;;
  sae-notebook) sae_notebook ;;
  regression) fit_sae_regression ;;
  regression-notebook) regression_notebook ;;
  regression-report) run_command "$python_bin" -s src/sae_regression_diagnostics.py --corpus "$corpus" --regression-dir "$regression_root" --output-dir "$regression_root/diagnostics" ;;
  export-reports) run_command "$python_bin" -s scripts/export_report_bundle.py ;;
  verify-reports) run_command "$python_bin" -s scripts/export_report_bundle.py --verify-only ;;
  latex) latex_notebook ;;
  scaling)
    if [[ "$corpus" != both ]]; then echo 'The declared scaling follow-up covers both corpora.' >&2; exit 2; fi
    run_command "$python_bin" -s src/scaling_sensitivity.py --source-root "$regression_root" --output-dir "${SCALING_ROOT:-results/scaling_sensitivity_v1}"
    ;;
  scaling-notebook)
    run_command "$python_bin" -s src/run_notebook.py --analysis scaling
    ;;
  all)
    check_environment
    run_tests
    prepare_data
    extract_surprisal
    verify_outputs
    fit_baseline
    export_diagnostics
    ;;
esac
