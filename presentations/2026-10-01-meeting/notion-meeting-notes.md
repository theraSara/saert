# SparseRT meeting notes — 1 hour

**Date:** 2026-10-01  
**Project:** Sparse Autoencoder features and human reading times  
**Datasets:** Provo eye-tracking FFD + Natural Stories self-paced reading  
**Model:** GPT-2 small + pretrained residual-stream SAEs

---

## 1. Goal for today’s meeting

I want to check whether the current direction is sound before turning the feature-interpretation results into the main story.

The current project asks:

> Do sparse internal features from GPT-2 predict human reading-time difficulty beyond lexical controls and surprisal, and do the predictive features point toward interpretable linguistic mechanisms?

The current answer is cautiously positive:

- Surprisal is a reliable baseline predictor after lexical controls.
- SAE features add held-out predictive signal, especially in Provo and in after-word states.
- Natural Stories is more mixed: dense states are stronger than SAE features in the main selected comparison, but SAE features still add signal in some settings.
- Feature interpretation is promising, but it should be framed as exploratory until labels are validated on independent examples or interventions.

---

## 2. Suggested 1-hour agenda

| Time | Topic | Goal |
|---:|---|---|
| 0–5 min | Project framing | Confirm the research question and scope. |
| 5–15 min | Baseline results | Show that lexical controls + surprisal are working. |
| 15–25 min | SAE fidelity | Show that SAE inputs/reconstructions are reasonable enough to analyze. |
| 25–40 min | SAE vs dense prediction | Discuss the main held-out prediction result. |
| 40–50 min | Residualized RT and scaling caveat | Explain RT′ and the Natural Stories L01 scaling issue. |
| 50–60 min | Feature interpretation + next steps | Decide what is safe to claim and what must be validated next. |

---

## 3. Notebook order for the presentation

Use the notebooks in this order:

1. `notebooks/01_baseline.ipynb`  
   **Purpose:** data quality, reading-time distributions, and surprisal baseline.

2. `notebooks/02_sae_fidelity.ipynb`  
   **Purpose:** show SAE reconstruction quality and activation behavior before using SAE features for prediction.

3. `notebooks/03_rt_prediction.ipynb`  
   **Purpose:** main result: SAE features vs dense states vs surprisal.

4. `notebooks/04_residual_rt.ipynb`  
   **Purpose:** ask whether SAE/dense representations predict reading-time residuals after controls and surprisal.

5. `notebooks/05_feature_interpretation.ipynb`  
   **Purpose:** exploratory interpretation of top predictive features.

For the meeting, notebooks 1–4 should carry the main argument. Notebook 5 should be presented as the next stage, not as final evidence.

---

## 4. Baseline result: surprisal is doing useful work

The baseline is not trivial. Surprisal improves held-out prediction beyond lexical controls.

| Dataset | Controls R² | Best surprisal model | Held-out R² | ΔR² over controls |
|---|---:|---|---:|---:|
| Provo FFD | 0.179 | 1,024/current + spillover | 0.194 | +1.48 pp |
| Natural Stories SPR | 0.366 | 1,024/current + spillover | 0.409 | +4.38 pp |

**Interpretation:**

- The pipeline is recovering the expected psycholinguistic baseline: more surprising words predict longer reading times.
- Spillover helps, especially in Natural Stories.
- These baseline results justify asking whether internal representations add anything beyond surprisal.

**What I should say:**

> Before looking inside the model, I checked that the standard surprisal story is present. It is: surprisal improves held-out reading-time prediction after controls, with a stronger gain in Natural Stories than Provo.

---

## 5. Main SAE prediction result

The main comparison asks whether SAE features add held-out predictive value relative to the controls + surprisal baseline, and how that compares to dense hidden states.

| Dataset | State | SAE gain | SAE 95% interval | Dense gain | SAE − dense | Texts improved |
|---|---|---:|---|---:|---:|---:|
| Provo FFD | Before word | +2.87 pp | [+1.77, +3.98] | +1.07 pp | +1.80 pp | 43/55 |
| Provo FFD | After word | +6.58 pp | [+5.42, +7.78] | +2.80 pp | +3.78 pp | 50/55 |
| Natural Stories SPR | Before word | +0.82 pp | [+0.19, +1.37] | +2.61 pp | −1.79 pp | 7/10 |
| Natural Stories SPR | After word | +3.84 pp | [+2.35, +5.18] | +6.23 pp | −2.39 pp | 10/10 |

**Interpretation:**

- Provo: SAE features outperform dense states in both before-word and after-word settings.
- Natural Stories: both representations help, but dense states outperform SAE features in the main selected comparison.
- After-word states are stronger than before-word states, which is expected because after-word states already include the target word.

**Careful claim:**

> SAE features clearly contain reading-time-relevant information, but they are not uniformly better than dense states. The strongest SAE advantage is in Provo; Natural Stories is more supportive of dense-state prediction.

---

## 6. Hook/layer story

The selected hooks suggest different explanations across datasets.

| Dataset | Main selected SAE hook | Interpretation |
|---|---|---|
| Provo | L01 | Likely lexical, orthographic, position, or local word-form effects. Be cautious about syntax claims. |
| Natural Stories | L08 | More plausibly contextual or higher-level, but still needs validation. |

**Important caveat:**

Provo selecting L01 is not a weakness, but it changes the story. L01 is before contextual transformer processing, so a Provo result at L01 should not be described as a deep syntactic mechanism without further controls.

**What I should ask:**

> Should I frame the Provo result as evidence that sparse lexical/position features are strong predictors of eye-tracking RT, while Natural Stories motivates the higher-level feature interpretation?

---

## 7. Residualized RT analysis

The RT′ analysis asks whether internal representations predict reading-time variation after removing controls, and in some settings after removing surprisal too.

This is the stronger cognitive-processing framing:

> RT′ = reading time after removing lower-level predictors.

Current interpretation:

- The residualized analysis supports the idea that representations contain signal not captured by lexical controls alone.
- The most conservative version, after controls + surprisal, should be treated as the cleaner follow-up analysis.
- It is useful for feature nomination, but not yet causal/mechanistic proof.

**What I should say:**

> I am using RT′ to avoid mistaking frequency or word length effects for cognitive difficulty. This lets us ask whether SAE features relate to residual processing cost rather than only word recognition difficulty.

---

## 8. Scaling sensitivity caveat

There was a serious diagnostic issue with fixed Natural Stories L01 SAE models.

| Dataset | State | Original L01 gain | Gain with training-SD floor | Worst original error | Worst floor error |
|---|---|---:|---:|---:|---:|
| Provo | Before word | +2.87 pp | +3.41 pp | 0.40 | 0.40 |
| Provo | After word | +6.58 pp | +6.53 pp | 0.41 | 0.40 |
| Natural Stories | Before word | −3521.03 pp | +0.62 pp | 69.47 | 1.07 |
| Natural Stories | After word | −9131.48 pp | +2.13 pp | 111.88 | 1.10 |

**Interpretation:**

- This is a scaling/pathology issue, not a substantive psycholinguistic result.
- The main Natural Stories selected-hook comparison is not based on this failed L01 condition.
- The floor sensitivity is useful because it explains the failure without deleting hard examples or using test information.

**What I should say:**

> I found and audited a failure mode: tiny training variance in a feature can create extreme held-out predictions. The scaling floor fixes the diagnostic condition, but I will not present the original L01 Natural Stories failure as a main finding.

---

## 9. Feature interpretation status

Feature interpretation is currently exploratory.

Current methods include:

- top predictive features from regression,
- max-activating examples,
- logit-lens style token summaries,
- Neuronpedia labels where available,
- SRP-style decomposition.

**Safe claim:**

> I can identify candidate features that are consistently predictive and inspect what contexts activate them.

**Unsafe claim for now:**

> These features are the causal mechanism of human reading difficulty.

That causal/mechanistic claim needs more evidence:

- independent validation examples,
- held-out feature-label tests,
- ablations or feature steering,
- controls against random matched features,
- participant-level robustness if possible.

---

## 10. What seems good enough now

Strong parts:

- Complete stimulus context and token alignment are handled carefully.
- First-word/BOS conditioning is explicit.
- Grouped held-out evaluation is used rather than random row splits.
- Regression uses training-only preprocessing and nested selection.
- The report bundle is portable and verified.
- The notebooks are now presentation-ready and organized as a clear research arc.

This is good enough for a progress meeting and collaboration discussion.

For a paper, it is promising but still needs more validation around the interpretation step.

---

## 11. What still needs work before paper submission

Priority items:

1. **Clarify the main claim.**  
   I should say “SAE features predict reading-time variance” before saying “SAE features explain mechanisms.”

2. **Validate feature labels.**  
   Do not rely only on manual impressions or Neuronpedia labels.

3. **Add stronger controls for Provo L01.**  
   Especially lexical identity, word position, punctuation, sentence boundary, and orthographic features.

4. **Check participant-level robustness if available.**  
   Current aggregate RT is useful, but reviewers may ask about mixed-effects or participant variation.

5. **Separate before-word and after-word claims.**  
   After-word states are easier to predict from because the model has seen the word. Before-word states are more anticipatory and cognitively interesting.

6. **Keep Natural Stories L01 as a diagnostic appendix.**  
   It should not drive the main result.

---

## 12. Questions to ask the professor

1. Is the main framing better as:
   - “SAE features reveal sparse predictors of reading difficulty,” or
   - “SAE features expose model-human processing mismatches”?

2. Should Provo L01 be treated as a main finding, or mostly as a lexical-control motivation?

3. For Natural Stories, should the paper emphasize dense states outperforming SAE features, or should the SAE feature interpretability still be the focus?

4. What level of feature interpretation is enough for the next milestone?
   - max-activating examples,
   - Neuronpedia labels,
   - SRP/logit-lens evidence,
   - manual linguistic categories,
   - intervention/ablation?

5. Should the next step be feature validation or participant-level/mixed-effects robustness?

---

## 13. One-minute summary to say aloud

The project is now structured as a validated pipeline rather than only a pilot. First, the baseline behaves as expected: surprisal improves held-out reading-time prediction beyond lexical controls. Second, SAE reconstructions are checked before using them. Third, SAE features add predictive signal, especially for Provo and after-word states, although Natural Stories shows dense states are stronger in the main comparison. Fourth, residualized RT lets us ask whether features predict processing cost beyond lower-level predictors. The feature interpretation notebook is promising, but I am treating it as exploratory until the feature labels are validated more independently.

---

## 14. Desired outcome from today

By the end of the meeting, I want to decide:

- what the central claim should be,
- whether the current results are enough to move deeper into feature interpretation,
- which caveats need to be foregrounded,
- and what validation step should come next.
