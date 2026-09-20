# Presenting SparseRT to the team

Use slides 1–14 for a 12–15 minute update. Keep slides 15–17 as backup.
Speaker notes in the PowerPoint and `speaker_notes.md` provide slide-by-slide wording.
If the meeting allows only five minutes, show slides 2, 4, 7, 12 and 13.

## Opening

“I have completed the corrected data, surprisal and SAE prediction pipeline.
SAE features improve reading-time prediction beyond controls and surprisal in
both corpora, but their advantage over dense representations differs by corpus.
I also found and investigated an extreme scaling failure. The immediate next
step is separating lexical and scaling effects before interpreting individual
features or intervening on them.”

## Order of explanation

1. **Question:** distinguish predictive association, feature meaning and model
   intervention. This explains how the current work supports the professor's
   eventual question about minimally used features.
2. **Workflow:** describe the output of each stage rather than walking through
   scripts. Explain data alignment and matched context only as needed.
3. **Evidence:** establish that the corrected surprisal baseline works, then
   compare SAE and dense representations on the same held-out stories.
4. **What changed our thinking:** Provo selects the earliest hook in every fold.
   Its token/position content makes lexical controls a priority.
5. **New experiment:** a training-derived scaling floor preserves Provo's
   after-word gain and removes the extreme Natural Stories first-hook errors.
6. **Decision:** agree on the next controlled experiment and the evidence needed
   for a feature interpretation or intervention claim.

## New result you can explain in one minute

“One feature barely activated on the training stories but activated strongly on
a held-out word. Scaling by its very small training standard deviation amplified
the prediction. I tested a fixed alternative: no retained feature gets a scale
smaller than the median feature standard deviation in that training split.
Everything, including the floor, is computed within training data. Provo's
after-word gain stayed almost the same, from 6.58 to 6.53 R² percentage points.
The maximum Natural Stories after-word error at that hook fell from 111.88 to
1.10 log milliseconds. This is an exploratory sensitivity, not independent
confirmation or a replacement of our all-hook results.”

The full follow-up runs 130 outer fits across four fixed-L01 conditions. It does
not retrain GPT-2 or the SAE, change measured RTs, delete words, or clip predictions.

## Likely questions and short answers

**What does a 6.58-point gain mean?**
It means held-out R² increases by 0.0658 beyond the controls-plus-surprisal model.
It is not a 6.58% reduction in reading time. The outcome is log mean RT.

**Are SAE features always better than dense states?**
No. Provo favors SAE. Natural Stories' after-word comparison favors dense states.
The before-word Natural Stories difference remains uncertain under the reported
conditional interval. SAE is also wider and nonlinear, so the comparison does
not isolate sparsity alone.

**Why ridge rather than Lasso now?**
This first analysis estimates whether the representation contains useful signal
while retaining correlated predictors. Lasso or elastic net is a later
training-only candidate-selection analysis. A large ridge coefficient does not
by itself identify a stable linguistic feature.

**Are these significant effects?**
Report the effect and its interval. The intervals resample whole texts with
predictions fixed, omitting model-refitting and participant uncertainty. The
study is exploratory, and Natural Stories has only ten story groups. Avoid
treating every hook as an independent confirmatory discovery.

**Why does the first hook perform well in Provo?**
We do not yet know the full explanation. Its input consists of token and position
information before transformer-block contextual mixing. This makes lexical
identity and position plausible explanations to test before syntax or integration.

**Did you find the feature humans need but GPT-2 suppresses?**
No. Zero activation, zero regression weight and experimental ablation are
different. We need stable feature candidates, independent linguistic validation,
and controlled model interventions before making a mechanistic account.

**Does SAE reconstruction measure feature importance?**
Reconstruction can change many directions at once. The fidelity checks quantify
that background damage. A later feature intervention should preserve the original
residual when gain equals one and compare against identity and matched random
features. Model effects still do not establish human causal mechanisms.

**Are the new scaling results the new primary results?**
No. They cover one fixed hook only. The original all-hook selector remains the
primary exploratory comparison. A full scaling sensitivity requires its own
declared specification and all-hook evaluation.

**Can this become an AACL paper?**
The current result supports a research direction, not a publication guarantee.
The intended contribution needs a carefully controlled representation comparison
and validated feature evidence. Novelty requires a focused comparison with prior
SAE and reading-time work. Participant-level robustness would strengthen the
psycholinguistic claims.

## Concrete next milestone

Agree on a training-only lexical identity/position comparison and one declared
all-hook scaling sensitivity. Then nominate a small feature set using stability
within training data. Interpret it with independently sampled contexts and
minimal pairs. Expand fidelity checks before gain tuning or feature ablation.

Before the meeting, open the PPTX in your presentation application and rehearse
the transitions between the baseline, SAE comparison and scaling follow-up.
The deck has editable chart/table content and speaker notes. Local render review
does not substitute for checking the application and projector you will use.
