# Nested SAE regression: implemented specification

This is the first corrected predictive comparison of SAE features. It uses
`configs/sae_regression.json`, frozen before inspecting its held-out outcomes.
The older Lasso notebooks and `results/sae`/`results/regression` artifacts are not
inputs. This implementation covers the ridge stage of the project plan;
elastic-net selection, feature interpretation and interventions remain later work.

## Model and target

For each corpus, hook and state type, predict natural log arithmetic mean RT
using the exact corrected matched-context baseline (lexical/position controls,
current surprisal and two previous-word surprisals), plus either a dense state
or the corresponding SAE features. Post-word states observe the current word;
prefix states do not. No additional feature spillover is introduced in this run.

Fit the objective

`mean((y - C b - Z w)^2) + alpha * ||w||²`,

where C includes an intercept and all baseline predictors, and Z contains
training-scaled added features. Baseline coefficients b are **unpenalized**.
The implementation projects the feature objective and y off the training control
space, solves ridge for w, then refits b conditional on w. It does not use saved
global or OOF RT residuals. Sparse features remain sparse; control projection is
applied algebraically without materializing a dense residualized feature matrix.

Within every training split, discard features with variance ≤ 1e-12 or fewer
than five nonzero observations. Scale retained features by training standard
deviation. An unpenalized intercept absorbs feature means, so sparse centering
is unnecessary. Original column IDs are preserved in saved raw coefficients,
with zeros for filtered features. Ridge coefficients are not feature-meaning
annotations or a sparse candidate list.

Dense ridge uses an eigendecomposition of the projected Gram matrix. Sparse
ridge uses preconditioned conjugate gradients, warm-started along the descending
alpha grid. The relative residual tolerance is 1e-6, maximum 1,500 iterations.
Failure to converge stops the run; it is not silently discarded or replaced.
Tests compare both solvers and a joint partially penalized solution.

## Splits and selection

- Outer leave-one-text-out evaluation: 55 Provo and 10 Natural Stories folds.
- Three inner GroupKFold splits on outer-training stories only.
- Fixed alpha grid: 100, 10, 1, 0.1, 0.01 for mean-squared-error normalization.
- Select alpha by pooled word/region-weighted inner-validation MSE. Exact ties
  choose the earlier, larger alpha. Filtering/scaling/control fits are recomputed
  in each inner split.
- Refit on outer-training data, then predict the untouched outer-test story.
- Run both representations and both state types at all 13 recorded hook locations.

Per-hook outer scores are exploratory profiles. For the main comparison,
`sae_regression_diagnostics.py` selects a hook separately in each outer fold
using **only that hook's selected inner-validation MSE**. Dense and SAE each
select their own hook; prefix and post-word are separate prespecified analyses.
Never report the largest outer-profile cell as an unbiased selected-model score.
Inner splits are the same deterministic grouping across hooks. These corpora have
already been explored in earlier work; this is not a pristine confirmatory test.

The evaluator reproduces the saved baseline predictions within 1e-8 and checks
the same row order. Baseline and new predictions therefore use identical outcomes,
groups and controls. Hyperparameters at grid boundaries are reported, not
automatically extended after viewing outer performance.

## Metrics and plots

The main metric is paired ΔR² against controls + surprisal, in percentage points.
The SAE-versus-dense contrast is separately paired by identical held-out rows.
Report RMSE/MAE, absolute R² and per-text improvement counts as secondary metrics.
Intervals use 2,000 paired whole-text bootstrap resamples of fixed OOF predictions,
seed 42. They do not include model-refitting uncertainty, participant uncertainty
or correction for all exploratory comparisons. Natural Stories has ten clusters.

The main figure gives each state/representation its own labeled interval row.
The hook profile is an annotated heatmap with a common symmetric scale across
corpora. Green means better prediction, plum means worse prediction. A positive
cell is not a significance marker. No layers, stories or outcomes are omitted
because their result is unfavorable.
Colors saturate at ±10 percentage points to preserve readability around ordinary
gains; printed values and CSVs are never clipped. A descriptive audit inspects
SAE runs with any error above two log units, traces the largest feature
contribution, and verifies the prediction against saved coefficients. This
threshold triggers reporting only; it does not filter data or choose models.

## Artifacts and reproduction

```sh
bash run.sh regression both
bash run.sh regression-notebook both
bash run.sh latex both
```

Regression automatically verifies and resumes completed combinations. Changed
input hashes, config or implementation require a new `REGRESSION_ROOT`; there is
no silent overwrite. Each `<corpus>/<state>/<hook>/<representation>/` contains
`predictions.csv`, `folds.json`, `coefficients.npz` and `manifest.json` installed
last. Coefficients are in original feature and control units, with the intercept
first in the control array, and fold ordering recorded. NumPy loading needs no
pickle. The config and complete split/tuning metadata accompany every result.

`--smoke` executes one outer fold per selected combination and writes no research
outputs. For example:

```sh
python -s src/sae_regression.py --corpus natural_stories --hooks L03 --states post --smoke
```

The report layer validates hashes and split membership, writes all selected
predictions and hook choices, and generates the main comparison plus the full
hook heatmap. `03_sae_reading_time_prediction.ipynb` reads these artifacts without refitting.
The dedicated `90_export_latex_tables.ipynb` handles table production from saved reports.

Results support a predictive comparison only. Additional association does not
identify a human processing mechanism; post-word lexical identity, participant
effects and feature stability require the planned follow-up analyses.
The SAE representation is wider and nonlinear relative to the dense state;
matching the readout family and tuning grid does not equate effective complexity.
A gain alone therefore does not establish that learned feature meanings are
responsible. Random-feature/capacity controls are useful follow-up comparisons.
