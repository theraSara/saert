# Corrected aggregate reading-time baseline

This is an exploratory predictive analysis of the corrected BOS extraction.
It does not establish a human cognitive mechanism or replace participant-level
mixed-effects inference. The corpora have already been examined during development.

## Outcome and inclusion

- Provo: first fixation duration (FFD), averaged over retained positive-duration
  participant observations for each displayed interest area.
- Natural Stories: the authors' processed arithmetic mean self-paced RT (SPR).
- Outcome: natural logarithm of the arithmetic mean RT in milliseconds. This is
  not the mean of participant log RTs. Observations are unweighted word/region means.
- No new 100–3,000 ms exclusion on aggregate means. Missing/nonpositive means
  are excluded only after lag construction. Preparation-time filtering remains
  recorded in the prepared manifests. No new participant exclusions are added.
- All presented units are retained as context, including the first word and
  punctuation/compound interest areas. Provo uses IA_ID order, not Word_Number.

## Predictors fixed before this run

All five default models use the same controls and eligible observations:

- Current wordfreq Zipf frequency, lexical length, and prepared sentence-final flag.
- `log(1 + word_position)` and separate indicators for the first and second units.
- Previous-one and previous-two frequency, length, and sentence-final flags.

The comparison models are controls alone; controls plus current surprisal;
and controls plus current, previous-one, and previous-two surprisals. The latter
two specifications are fitted separately for the 1,024-token/stride-512 regime
and the 128-token/stride-64 regime matched to hidden-state extraction. No model
includes both surprisal contexts simultaneously. The two-lag default is a
declared modeling choice, not a tuned optimum; CLI alternatives are sensitivity
analyses and should not be selected retrospectively as if confirmatory.

Lags follow presented word/region order within each text, including across sentence
boundaries, and never cross texts. They are constructed before RT exclusions.
Unavailable initial history is filled with zero, with separate initial-position
controls; zero is a structural encoding, not observed surprisal. There is no
lagged RT predictor, and no outcome-dependent feature selection or penalty tuning.

Sentence-final status is a punctuation heuristic, not a validated syntactic parse.
Text identity defines held-out groups and is not treated as a numeric predictor.
Participant/item random effects and nonlinear controls remain future robustness work.

## Evaluation and uncertainty

Each entire text is held out once: 55 Provo folds and 10 Natural Stories folds.
StandardScaler and unpenalized linear regression fit training texts only. Saved
fold specifications enumerate train/test text IDs. Every eligible unit receives
exactly one held-out prediction from each model on the same split.

Primary summaries: pooled held-out R² and ΔR² against controls, RMSE and MAE in
natural-log-ms units. R² uses the pooled outcome mean as its reference. Negative
R² and negative improvements are possible and are not clipped. Percentage points
are `100 * ΔR²`, not relative percent improvement or a ratio to a tiny baseline gain.

Intervals: 2,000 paired bootstrap resamples of whole texts, seed 42, with 2.5th and
97.5th percentiles. Each resample uses the same text draw for every model. Paired
contrasts compare long versus matched context and spillover versus current-only.
These intervals condition on the fixed out-of-fold predictions: models are not
refitted inside the bootstrap. They do not include training-set or participant
uncertainty and are particularly limited by the ten Natural Stories clusters.
No significance stars or independent-row OLS p-values are reported.

Per-text plots use controls MSE minus candidate MSE; positive values mean lower
error. They do not divide by each short text's potentially small RT variance.

## Residuals and provenance

`RT_resid_controls_oof` is observed log mean RT minus the held-out controls
prediction. It is a diagnostic residual, not automatically a valid training
target for downstream cross-validation. The SAE evaluator must fit its own
control adjustments inside its training and inner-validation folds.

`RT_prime_descriptive` uses a full-data controls fit and adds its centered-predictor
intercept back. It is explicitly in-sample and descriptive. Neither variable is
"pure comprehension cost" or measured in milliseconds.

Each corpus writes predictions, all stimulus rows with inclusion reasons, model
comparison and paired-contrast tables, per-text scores, portable fitted parameters
(including scalers), and a manifest with source hashes, extraction IDs, settings,
and split membership. No trained pickle is required to inspect the results.

The notebook checks artifact hashes and recomputes R² from saved predictions.
It does not refit models. Exported tables and figures carry a diagnostics manifest.

## Reproduce

```bash
bash run.sh baseline both
bash run.sh diagnostics both
bash run.sh notebook both
```

Use a fresh `BASELINE_ROOT` for sensitivity runs. The older SAE scripts are not
yet connected to these outputs and must not be used unchanged.

## Figure selection

`heldout_improvement.pdf` is the main-text candidate: incremental predictive value
of current and spillover surprisal, with conditional paired intervals.
`model_comparison.tex` provides exact numbers. `per_text_performance.pdf` is a
supplementary candidate showing heterogeneity. RT/surprisal distributions and
collinearity checks are diagnostics rather than mechanistic results.

## Methodological references

- [On the Role of Context in Reading Time Prediction](https://aclanthology.org/2024.emnlp-main.179/): related context/spillover comparisons; not an exact reproduction of its analysis.
- [scikit-learn: LeaveOneGroupOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html).
- [scikit-learn: avoiding preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
