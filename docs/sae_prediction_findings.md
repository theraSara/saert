# First corrected SAE prediction results

20 September 2026. Exploratory ridge analysis with the configuration frozen in
`configs/sae_regression.json` before inspecting held-out results. This is not a
preregistration. Reproduce with `bash run.sh regression both` (verified resume)
and `bash run.sh regression-notebook both`.

## Read the primary comparison

All effects below are held-out R² **percentage-point gains over the same
controls + current/two-lag matched-context surprisal baseline**. Outcomes are
log arithmetic mean RT: Provo first fixation duration and Natural Stories SPR.
Each outer test text is excluded from preprocessing, regularization and hook
selection. Main comparisons select the hook using inner-validation error.

- **Provo, before word:** SAE +2.87 [1.77, 3.98]; dense +1.07.
  Paired SAE minus dense +1.80 [0.67, 2.88]. SAE improves error in 43/55 texts.
- **Provo, after word:** SAE +6.58 [5.42, 7.78]; dense +2.80.
  Paired SAE minus dense +3.78 [2.73, 4.88]. SAE improves error in 50/55 texts.
- **Natural Stories, before word:** SAE +0.82 [0.19, 1.37]; dense +2.61.
  Paired SAE minus dense −1.79 [−3.49, 0.25], leaving the direction uncertain
  under this interval procedure. SAE improves error in 7/10 stories.
- **Natural Stories, after word:** SAE +3.84 [2.35, 5.18]; dense +6.23.
  Paired SAE minus dense −2.39 [−4.02, −0.45]. SAE improves error in 10/10 stories.

Intervals are paired whole-text bootstraps conditional on fixed out-of-fold
predictions. They omit participant and model-refitting uncertainty. Text counts
are descriptive, not independent replications of an experiment. Dense and SAE
readouts differ in dimensionality and nonlinearity; they do not isolate the
effect of sparsity alone. Post-word states observe the target word, so their
gains are not evidence of anticipation.

## Two findings that shape the next experiment

**Provo selects the earliest SAE hook in all 55 folds, for both states.** This is
`blocks.0.hook_resid_pre`: token and position information before the transformer
blocks mix context. The before-word representation is of the preceding input
position (BOS for the first word); the after-word representation observes the
current word's final subtoken. This favors testing lexical identity, previous-word
identity and position explanations before claims about syntactic integration.

**Natural Stories' earliest SAE hook has an extreme generalization failure.**
The after-word state at story 8, word 907, `saucer`, and the following before-word
state at word 908, `or`, share a large activation of feature 16980. Its held-out
activation is about 0.777, versus a training maximum of 0.000034 and standard
deviation 0.00000122. It is nonzero on 12 training rows, passing the declared
filter. Training-only standardization therefore permits extreme extrapolation;
the prediction reaches 117.64 log ms after-word and 75.25 before-word.

Saved coefficients reproduce these predictions to numerical precision. This
identifies the statistical source of the failure, not a linguistic interpretation
of feature 16980. The feature-shift audit is regenerated with the report and
records full-precision values. No outliers are deleted, predictions clipped, or
hyperparameters retuned after seeing this failure. Inner hook selection never
chooses L01 for Natural Stories, so the primary comparisons above do not contain
this hook. It remains visible in the exploratory heatmap and per-hook scores.

Many SAE fits also choose the largest declared ridge strength: 419/715 Provo
before-word fits, 246/715 Provo after-word fits, 53/130 Natural Stories before-word
fits and 17/130 after-word fits. These counts cover every hook, not only the
inner-selected models. The fixed grid is a limitation, not a tuned optimum
established beyond its boundary.

## What to do next

**Follow-up completed:** a fixed-L01 training-derived median-SD floor preserves
Provo's after-word gain and removes the extreme Natural Stories errors at this
hook. Details: [scaling_sensitivity_methods.md](scaling_sensitivity_methods.md).
This is an exploratory response to the observed failure. The original all-hook
results above remain unchanged; an all-hook scaling comparison is still pending.

1. Declare a separate scaling robustness analysis using training-derived scale
   floors or an explicitly specified alternative scaling rule; retain this
   analysis unchanged. Choose rules on training data and document this as an
   exploratory response to an observed failure. Fresh external data provide a
   stronger check than repeated exploration of these same held-out texts.
2. Add training-only lexical/position comparison models, especially for Provo's
   pre-first-block result. Inspect story-level consistency before interpretation.
3. Nominate a small feature set using training-only selection and stability
   analysis. Ridge coefficients alone do not provide validated feature labels.
4. Build independent feature dossiers and linguistic checks. Expand fidelity
   checks before interventions or gain tuning. Keep participant-level robustness
   in the paper plan.

## Reproducible artifacts

- `notebooks/03_sae_reading_time_prediction.ipynb`: two main figures, focused tables and
  collapsible diagnostics; no model fitting occurs in the notebook.
- `results/sae_regression_v2/diagnostics/primary_results.csv`: primary effects.
- `model_comparison.csv`: all hooks and inner-selected models, including failures.
- `selected_hooks.csv`: each outer fold's training-selected hook.
- `feature_shift_audit.csv`: extreme-error feature contributions and train scales.
- `notebooks/90_export_latex_tables.ipynb`: all paper table exports from verified reports.

Numerical solvers, nested evaluation and limitations are documented in
[sae_regression_methods.md](sae_regression_methods.md). Plots use shared styles
from `src/utils.py`; the hook heatmap saturates colors at ±10 pp while keeping
all printed values unchanged. This is a display choice, never an analysis filter.
