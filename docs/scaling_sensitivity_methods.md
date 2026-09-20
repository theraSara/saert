# Fixed-hook scaling sensitivity

This exploratory follow-up was specified after inspecting the original L01
Natural Stories failure and before fitting the new readouts. It is not an
independent confirmatory experiment. The original analysis remains unchanged.

## Specification

Both corpora, before- and after-word states, SAE L01 only. This is the
pre-first-block residual hook. Preserve the original words, outcomes, controls,
matched surprisal, two spillover lags, five ridge strengths, grouped inner folds
and outer leave-one-story-out evaluation. Retain the same training-only
nonconstant/minimum-active-row feature filter.

Within each inner or outer **training split**, compute retained features'
standard deviations. The denominator for feature j becomes
`max(SD_j, median(SD across retained features))`. No alternative floor values are
tuned. Recompute the baseline projection under the scaled representation and
fit the same partially penalized ridge objective. Do not clip test activations
or predictions, use test outcomes for tuning, or remove anomalous rows.

`src/scaling_sensitivity.py` reuses the original preprocessing, ridge solver and
metric functions. Its separate grouped evaluation records every floor, alpha,
training/validation/test membership, coefficients and predictions. A synthetic
test checks equivalence to the joint penalized solution; another checks that
changing a held-out story's outcome cannot change its fitted predictions.

## Results

Gain over M1, in R² percentage points, at fixed L01:

| Corpus / state | Original | Scale floor | Conditional 95% interval for floor gain |
|---|---:|---:|---|
| Provo before | +2.87 | +3.41 | +2.53 to +4.33 |
| Provo after | +6.58 | +6.53 | +5.28 to +7.82 |
| Natural Stories before | −3521.03 | +0.62 | +0.20 to +0.95 |
| Natural Stories after | −9131.48 | +2.13 | +1.47 to +2.81 |

The original extreme negative values reflect real prediction failures, not
percentages of explained variance bounded by −100%. R² can be arbitrarily
negative. Natural Stories maximum absolute error falls from 69.47 to 1.07 log ms
before-word and 111.88 to 1.10 after-word. These are worst-case errors, not RMSE.

The new Provo before-word gain is descriptively higher. This analysis does not
establish that the change is superior, optimal, or transferable to all hooks.
Natural Stories' original inner selector avoided L01, so these first-hook scores
must not replace its original selected-model results. Intervals condition on
fixed OOF predictions and exclude participant/refitting uncertainty.

## Reproduction

```
# Read the completed follow-up without fitting again:
bash run.sh scaling-notebook both
# A new full follow-up run requires a new directory:
SCALING_ROOT=results/scaling_sensitivity_repeat bash run.sh scaling both
```

The completed artifacts are in `results/scaling_sensitivity_v1/`. The readable
summary is `notebooks/04_feature_scaling_robustness.ipynb`. The notebook 90 custom-table
workflow accepts `comparison.csv` for LaTeX export.
