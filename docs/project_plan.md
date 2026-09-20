# SparseRT project plan

Updated 20 September 2026. This is a prospective work plan following exploratory
baseline and fidelity analyses; it is not a retrospective preregistration.

**Working title:** SparseRT: From Sparse Language-Model Features to Human
Reading-Time Prediction.

**Central question:** Which sparse GPT-2 features predict human reading times
beyond lexical controls and surprisal, what linguistic patterns do they track,
and do controlled changes to those features alter model predictions in ways
that generalize to held-out reading-time data?

The project has three distinct evidence levels: predictive association with RT,
validated feature interpretation, and intervention effects inside GPT-2. The
third level does not establish a causal mechanism in human readers.

## 1. Findings established so far

### Corrected baseline

Outcomes are natural logs of arithmetic mean RT: Provo first fixation duration
(FFD), and Natural Stories self-paced reading time (SPR). Evaluation holds out
entire texts. These R² values describe word/region means, not individual trials.

| Corpus | Units / texts | Controls R² | Controls + current and two lagged surprisals R² | Increment, percentage points | Conditional 95% interval for increment |
|---|---:|---:|---:|---:|---:|
| Provo FFD | 2,743 / 55 | 0.1793 | 0.1941 | +1.48 | +0.54 to +2.35 |
| Natural Stories SPR, 1,024-token regime | 10,256 / 10 | 0.3656 | 0.4094 | +4.38 | +2.34 to +6.34 |
| Natural Stories SPR, 128-token regime | 10,256 / 10 | 0.3656 | 0.4071 | +4.16 | +1.80 to +6.00 |

The intervals resample whole texts while keeping fitted out-of-fold predictions
fixed. They do not capture model-refitting or participant uncertainty. Natural
Stories has only ten independent story groups.

Surprisal adds useful held-out predictive information under this specification.
The earlier “surprisal explains almost nothing” and “SAEs explain 88–341 times
more” narratives must not be used as results of the corrected pipeline. Corrected
SAE RT ridge regression is now complete; see the dated update below. Its
configuration was fixed before inspecting its held-out results.

Natural Stories' 1,024-minus-128 context difference is +0.225 percentage points,
with interval −0.721 to +1.131. This does not establish a context advantage or
equivalence. Provo's two regimes coincide because the texts fit in both.
Corpus differences cannot be attributed solely to reading modality: texts,
readers, preprocessing, outcome distributions, and measurement procedures differ.

### SAE extraction and fidelity

There are 13 hook locations across GPT-2 small's 12 blocks: 12 pre-block hooks
and the final post-block residual. Each has a pinned 24,576-feature dictionary.
All features are saved for both corpora and both state types (52 sparse matrices).
Sampled live/saved states match exactly. Identity interventions leave predictions
unchanged. These checks establish implementation consistency, not human validity.

Geometric reconstruction and preservation of model behavior are different:

| Sampled Natural Stories intervention | NLL increase when only actual BOS is preserved | NLL increase when every window's initial position is preserved |
|---|---:|---:|
| L02: before block 1 | 6.395 bits/token | 0.032 bits/token |
| L03: before block 2 | 3.158 bits/token | 0.093 bits/token |

This isolates a strong sensitivity to reconstructing rolling-window starts at
these hooks. The restricted intervention excludes that position; it does not
show that the excluded state is faithfully reconstructed. At the final residual
hook, reconstruction still increases NLL by 0.533 bits/token in Provo and 0.679
in Natural Stories. These diagnostics use only four windows per corpus (236 and
319 scored tokens), so broader validation remains necessary for interventions.

No corrected result yet validates a linguistic feature meaning or a benefit
from gain tuning. Ridge comparisons now establish predictive associations,
with hook selection inside training folds. A zero SAE
activation does not demonstrate that GPT-2 suppressed a human computation.

### Predictive comparison update — 20 September 2026

Phase C's first ridge comparison is complete: 104 corpus/state/hook/readout
combinations, with outer story-held-out evaluation and training-only selection
of regularization and the main comparison's hook. Full details and limitations
are in [sae_prediction_findings.md](sae_prediction_findings.md).

SAE gains beyond controls plus matched-context surprisal are +2.87 / +6.58
percentage points before/after the word in Provo, and +0.82 / +3.84 in Natural
Stories. SAE outperforms dense states in Provo; dense states have the larger
Natural Stories gains. The before-word Natural Stories SAE-minus-dense interval
includes zero. These conditional bootstrap intervals do not include participant
or model-refitting uncertainty.

Provo's SAE selector chooses the pre-first-block hook in every outer fold.
That is a lexical/position representation before transformer contextual mixing,
so this result does not establish syntactic integration. Natural Stories has an
extreme first-hook feature-scaling failure, retained in the exploratory results;
inner selection avoids it. Investigate lexical identity and scaling sensitivity
before selecting features for a cognitive interpretation. Original Phase C
specifications below are retained to distinguish the plan from this update.

The targeted fixed-L01 scaling follow-up is also complete. A training-derived
median-SD floor preserves Provo's after-word gain (+6.58 to +6.53 pp) and removes
the extreme Natural Stories first-hook errors. It does not replace the original
all-hook analysis. See [scaling_sensitivity_methods.md](scaling_sensitivity_methods.md)
and notebook 04. The immediate milestone remains lexical controls and a declared
all-hook scaling sensitivity before candidate interpretation.

Sources: `results/baseline_bos/*_model_comparison.csv`, `*_contrasts.csv`,
`results/sae_bos/diagnostics/behavior_summary.csv`, `geometry_summary.csv`.
Definitions and reproducibility details are in [baseline_methods.md](baseline_methods.md)
and [sae_methods.md](sae_methods.md).

## 2. Positioning and related work

The target is computational psycholinguistics informed by model interpretability:
test the incremental value of representations, interpret a reproducible subset,
and validate their role in the model with controlled interventions.

There is direct overlap to address before claiming novelty. Guo, Wu, and Yiu's
[Sparse Autoencoders Map Brain–LLM Alignment onto Cortical Semantic Topography](https://arxiv.org/html/2605.23035v1)
already reports SAE-based RT analyses using Provo and Natural Stories. Its §5.3
uses semantic features projected to 50 principal components in mixed-effects
models. Therefore neither “SAE features predict RT” nor “we use both corpora” is
a sufficient novelty claim. Audit its methods and implementation before asserting
which questions remain unresolved; this plan does not treat a brief literature
check as a complete novelty review.

Our proposed emphasis is a focused GPT-2-small study connecting individually
identified features, story-held-out generalization, and carefully validated
feature interventions. Prefix/post-word distinctions and reconstruction/window
controls support that claim. Their novelty must still be checked against broader
literature, and their scientific value depends on the eventual results.

[Reverse-Engineering the Reader](https://aclanthology.org/2024.emnlp-main.526/)
optimizes LM surprisal-based RT alignment by fine-tuning. It motivates a later
comparison with targeted feature gains; it does not show that its improvements
arose by restoring features suppressed by an SAE.

[On the Role of Context in Reading Time Prediction](https://aclanthology.org/2024.emnlp-main.179/)
motivates careful separation of frequency and contextual predictors. Our
frequency control and incremental comparisons do not isolate “pure cognition.”

[Sparse Readout Prism](https://arxiv.org/abs/2609.01936) decomposes readout weights
to explain logit/lens scores. Keep it as an optional interpretation extension,
after verifying its compatibility and incremental value; it is not required to
run the next predictive experiment.

## 3. Phase plan and completion criteria

| Phase | Main question | Status | Deliverable / completion criterion |
|---|---|---|---|
| A. Data and baseline | Are stimuli, RTs, context and baseline evaluation correct? | Implemented; initial analysis complete | Versioned inputs, held-out predictions, notebook 01; participant robustness remains |
| B. SAE fidelity | Are the correct dictionaries applied, and what does reconstruction change? | Extraction complete; preliminary fidelity complete | Notebook 02; expand samples before feature interventions |
| C. Predictive comparison | Do sparse features add held-out RT information? | Initial ridge comparison complete | Nested splits, dense controls, notebook 03; scaling/lexical sensitivity next |
| D. Feature interpretation | Which reproducible features track which linguistic patterns? | Pending C | Feature dossiers, negative examples, independent annotation/tests |
| E. Model interventions | Does changing a feature affect predictions and RT alignment? | Pending B expansion and D | Paired intervention results with reconstruction and perturbation controls |
| F. Robustness and paper | Which conclusions survive alternative analyses? | Plan now; run after C | Participant analysis, outcome sensitivities, complete result/limitation tables |

### Phase C: the immediate implementation

Freeze GPT-2, SAE checkpoints, word alignment and baseline covariates. Keep each
corpus separate. The first comparison uses the **128-token matched-context
surprisal with two spillover lags**; the 1,024-token baseline is a declared
sensitivity. This makes context availability explicit rather than attributing
a context mismatch to the representation.

Use the same original log mean RT outcome and controls in all models:

| Model | Predictors | Purpose |
|---|---|---|
| M0 | Lexical/position controls | Existing reference |
| M1 | Controls + current and two lagged surprisals | Primary baseline |
| M2 | M1 + dense hidden state at one hook | Tests information beyond scalar surprisal |
| M3 | M1 + SAE features at that hook | Tests sparse representations |
| M4, secondary | Controls + SAE features, without surprisal | Describes shared predictive information; not the primary contrast |

Run M2/M3 separately on post-word and prefix states. Do not combine all hooks in
the first experiment. Post-word states observe the current word; prefix states
do not. A post-word advantage alone is not evidence for anticipation. Initially
add only the selected state's current feature vector; feature spillover can be
a separately declared sensitivity, rather than multiplying the search space now.

Primary contrast: held-out M3 minus M1. Secondary contrast: M3 minus M2. SAE need
not outperform a dense representation to provide useful interpretability, but
any predictive loss must be quantified. Do not conclude that an equal-performing
SAE is a “strict superset” of surprisal or dense information.

Evaluation protocol:

1. Outer leave-one-text-out splits: 55 Provo and 10 Natural Stories folds,
   reusing the baseline grouping and eligible rows.
2. Inner grouped validation within the training texts only; proposed initial
   default is three group folds, with a fixed split seed/order recorded.
3. Fit feature filtering, scaling, dimensionality reduction (if any), and
   regularization using inner training data. All-zero/constant-feature filtering
   is permitted within a fold. Never use a global top-512 list.
4. Keep controls and baseline surprisal unpenalized while regularizing the added
   representation block, or use an equivalent training-only partial-regression
   implementation. Test equivalence on synthetic data. A generic pipeline that
   penalizes all predictors differently from M1 is not the intended comparison.
5. Begin with ridge for both dense and sparse representations, using comparable
   inner-validation budgets. Add elastic-net/Lasso as a secondary sparse-readout
   analysis for candidate selection. Freeze explicit hyperparameter grids in a
   configuration before inspecting its RT results; record convergence failures.
6. Refit on all outer-training texts with the chosen configuration; predict the
   outer-test text once. Store every fold, parameter choice, prediction and row ID.
7. Report all hook profiles. If reporting a single automatically chosen hook,
   choose it inside inner validation. Picking the global outer-CV winner and
   reporting its score as unbiased performance creates selection optimism.

Do not train on globally residualized RT or assume previous OOF residuals are
automatically safe targets: nuisance fits must respect the current inner/outer
split. Residuals remain useful diagnostics. Controls do not purify RT into one
psychological process.

Report paired ΔR², RMSE/MAE and per-text error changes on identical rows. Reuse
the conditional text bootstrap with its limitations stated. Handle multiple
hooks/analyses explicitly; an exploratory layer profile is not 13 independent
confirmatory discoveries. Participant-level uncertainty is addressed in Phase F.

**Deliverables:** replace `src/sae_regression.py`; add a frozen analysis config,
`results/sae_regression_v2/` with fold-level provenance and predictions, and
`notebooks/03_sae_reading_time_prediction.ipynb`. Test that changing a test-story outcome
cannot change that story's predictions, selections or hyperparameters.

**Decision:** if added value is absent, report that result and diagnose the
representation/readout tradeoff. Do not keep searching hooks, features and
outcomes until a favorable result appears. If gains are present but driven by
one story or basic lexical identity, resolve that before a cognitive explanation.

### Phase D: interpretation with independent validation

Use training-only selection to nominate a small set of features or correlated
feature groups. Examine stability across training folds and regularization
choices; a large coefficient alone is not a stable linguistic finding. Record
activation prevalence and effect scale, not just a signed coefficient.

For each candidate, create a dossier containing:

- Hook, checkpoint revision and feature ID; prefix/post-word role and selection rule.
- Top activations and a random sample of moderate activations from a separate
  text corpus, plus near-miss and nonactivating examples. Look for lexical,
  punctuation, position and frequency explanations before complex labels.
- A provisional description annotated without seeing RT-effect signs, preferably
  by two team members; permit mixed or uninterpretable features.
- New minimal-pair contexts and operational linguistic labels testing the
  description beyond the examples used to invent it.
- Held-out predictive effects of the selected feature/group and stability,
  with selection and evaluation kept separate.

A decoder's vocabulary projection can suggest a hypothesis. Applying LayerNorm
to an isolated decoder direction is not an exact causal attribution through the
full model. Validate interpretations in context and allow polysemantic features.
Do not match equal feature IDs across different hook dictionaries.

For the professor's “zeroed or minimally used” question, distinguish **SAE
encoding zeros**, **regression coefficients set to zero**, and **features we
experimentally ablate**. They arise from different procedures. Inspect encoder
preactivations/thresholds and SAE reconstruction residuals where relevant, but
do not call a negative result evidence that a human computation is absent.

**Deliverable:** `src/feature_analysis.py` plus a small, auditable feature report
and `04_feature_interpretation.ipynb`. If identities are unstable, prefer a
validated group-level account over a story about one chosen feature number.

### Phase E: connect identified features to model behavior

First expand fidelity checks to all texts and multiple predeclared positions,
including initial and rolling windows, BOS, and later hooks. Report both
boundary policies; select a documented operating policy for interventions.
There is no universal pass threshold: compare reconstruction-induced changes
with the feature effects being interpreted and diagnose disproportionate damage.

For a candidate decoder direction d_j and activation z_j, one useful primary
intervention is `h_changed = h + (gain - 1) * z_j * d_j`. Gain 1 leaves the
original residual unchanged and keeps SAE reconstruction error in the stream.
Compare this with a reconstruction-based intervention separately; do not
attribute reconstruction damage to removal of a chosen feature.

Predeclare a bounded gain grid (for example 0, 0.5, 1, 1.5, 2), hook and eligible
positions. Intervention sites must precede the target subtoken whose probability
is evaluated. Editing the state *after* a token has been observed cannot change
that token's already-computed probability. For multi-subtoken words, specify
whether to edit before the word only or before every subtoken and sum the
resulting conditional log probabilities consistently.

Compare original model, identity, reconstruction-only, candidate feature changes,
and randomly selected features matched for activation frequency, magnitude and
decoder-direction norm. Track perturbation size, downstream KL/NLL and general
text perplexity as well as reading-time predictive performance.

Train feature choice, gain selection and any RT readout inside the training
splits. Evaluate held-out changes in RT prediction error and surprisal, with
frozen-readout and refitted-readout comparisons clearly separated. Measured
human RT does not change when GPT-2 is edited; only its predictions and their
alignment to the existing RT observations change.

Multiplying a zero activation cannot activate it. If investigating allegedly
missing features, define a separate activation-injection or matched-context
patching experiment with training-derived magnitudes and controls. Do not treat
injection as proof of the feature's natural role.

**Deliverable:** `src/feature_interventions.py`, reproducible intervention tables,
and `05_feature_interventions.ipynb`. Gain tuning is an optional stronger branch
after selective intervention validation, not an assumption that alignment must
improve. Full LM fine-tuning and training a second SAE are outside the core plan.

### Phase F: robustness and publication

Use participant-level RTs to check aggregation effects and participant/item
dependence, with a prespecified mixed-effects structure and convergence checks.
Avoid treating thousands of trials of the same stimulus as independent feature
replications. Predict held-out stories using a clearly defined policy for new
item random effects; distinguish this from within-sample coefficient inference.

Keep FFD as the current Provo primary measure. Add gaze duration as a declared
secondary analysis, and total reading time only if its extra question is worth
the scope. Do not silently switch measures because a result looks stronger.
Check frequency/length nonlinearities, spillover choices, BOS/first-word handling,
lexical identity and influential stories. Cross-corpus replication means repeating
the procedure or testing frozen selected features, not assuming interchangeable
RT scales or asserting that a difference identifies modality-specific cognition.

Because both corpora have been explored, label current evaluation exploratory.
An independent dataset or replication can provide stronger confirmation; a new
random split of already inspected results is not a new untouched experiment.

For AACL, prioritize one clear empirical question with controlled evaluation and
well-supported interpretations. A plausible main-text package is: incremental
held-out prediction figure, a small validated feature table, and a controlled
intervention figure if that phase succeeds. Baseline, fidelity and sensitivity
tables belong in the main text or supplement according to their role. Preserve
negative outcomes, exact analysis scope and replication limitations. Venue
deadlines and submission requirements need a separate current check when the
team chooses a submission cycle.

## 4. Team discussion and scope

Suggested responsibilities to agree with the team, not assignments already made:

| Role | Useful responsibility |
|---|---|
| Project lead / you | Pipeline, nested prediction runs, experiment ledger and reproducibility |
| Professor | Core claim, theoretical interpretation and scope decisions |
| Postdoc | Statistical design, leakage/uncertainty review, participant-level robustness |
| Contributing PhD student | Independent feature annotation and intervention/minimal-pair validation |

The current CPU machine has completed extraction. Start regression locally and
profile one fold before scaling. Use the university GPUs for large context
searches, expanded interventions or later gain tuning when profiling warrants it;
additional compute alone does not fix an evaluation problem.

The next team checkpoint is after Phase C: assess genuine held-out increments,
dense-versus-sparse tradeoffs, prefix/post-word differences and story consistency.
Then choose a bounded interpretation/intervention branch. Maintain a dated
experiment ledger separating predeclared settings, debugging changes and
result-driven exploration.

## 5. Immediate checklist

- [ ] Audit closely related SAE/RT work and record the proposed distinction.
- [ ] Freeze the Phase C comparison, splits, hyperparameter grids and selection policy.
- [ ] Implement and test corrected nested SAE/dense regression.
- [ ] Run a computational smoke test, then both corpora across all recorded hooks.
- [ ] Execute notebook 03 and review every result, including failures and null gains.
- [ ] Decide whether stable candidates justify Phase D; expand fidelity before Phase E.

Existing commands and notebooks remain documented in `run.sh`. Proposed Phase
C–F scripts/notebooks above are future deliverables, not files claimed to exist.
