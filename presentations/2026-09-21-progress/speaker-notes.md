# SparseRT meeting notes

Suggested pace: slides 1–14 in 12–15 minutes. Slides 15–17 are optional backup.

## 1. SparseRT

Opening, about 30 seconds: I have built a corrected pipeline from reading-time data to sparse GPT-2 features and tested whether they improve prediction on unseen stories. We now have predictive evidence, a meaningful difference between corpora, and a new scaling check. The next question is what drives the gains before we interpret features or intervene on them.

Source: docs/project_plan.md; docs/sae_prediction_findings.md

## 2. Research question and evidence levels

About 50 seconds: The professor’s question concerns features that may be weakly used or removed. I separate three claims. Prediction asks whether features help explain held-out RTs. Interpretation asks what those features track. Intervention asks what changes in the model when we edit them. We have completed an initial predictive comparison. We have not established a causal mechanism in human readers.

Source: docs/project_plan.md

## 3. Data and outcome definition

About 45 seconds: Provo and Natural Stories provide complementary measurements, but they also differ in texts and participants. We cannot attribute the result difference entirely to reading modality. The current outcome is the log of the arithmetic mean RT for each displayed region or word, not the average of individual log RTs. Participant-level modeling is a later robustness analysis.

Source: docs/baseline_methods.md; docs/project_plan.md

## 4. Workflow and current progress

About 60 seconds: This is the project map. The first four stages are implemented. I keep the language-model probability path separate from the representation path. The final two stages need additional evidence, rather than automatically labeling a predictive feature as a cognitive process. Every stage saves reproducible artifacts and metadata.

Source: docs/project_plan.md; run.sh

## 5. A fair comparison beyond surprisal

About 60 seconds: Each model predicts the same original outcome on the same held-out words. The high-dimensional models retain unpenalized controls and surprisal, with ridge only on the added representation. We hold out one whole text at a time and tune the penalty inside three grouped training folds. We also select the reported hook inside training data. We do not pick the best test-layer result.

Source: docs/sae_regression_methods.md; configs/sae_regression.json

## 6. Surprisal adds useful predictive information

About 45 seconds: The corrected baseline contradicts the original weak-surprisal narrative. With lexical and position controls and two spillover lags, surprisal adds held-out predictive value in both datasets. These gains are over controls only. Later SAE gains use the stronger controls-plus-surprisal baseline. The model has a 1,024-token limit, but the primary representation comparison uses matched 128-token windows.

Source: results/baseline_bos/diagnostics/model_comparison.csv; docs/baseline_methods.md

## 7. SAE gains differ across corpora

About 75 seconds: All bars show gains beyond controls and surprisal. SAE features add predictive value in both corpora, with larger after-word gains. Provo favors the SAE readout. Natural Stories has larger dense-state gains. After-word features observe the target word, so their gain can include lexical information and is not evidence of anticipation. The next slide gives the paired comparison and its uncertainty.

Source: results/sae_regression_v2/diagnostics/primary_results.csv

## 8. The paired comparison favors different readouts

About 60 seconds: Positive differences favor SAE. Negative differences favor dense states. Provo favors SAE for both state timings. Natural Stories favors dense states after the word, while its before-word interval includes zero. These conditional intervals do not include model-refitting variation or multiple-comparison correction. This result supports a representation comparison, not a general claim that sparse models are more human-like.

Source: results/sae_regression_v2/diagnostics/primary_results.csv

## 9. Provo points toward a lexical explanation

About 60 seconds: In every Provo outer fold, inner validation selects L01 for the SAE model, before and after the word. This hook is before the first transformer block. It contains token and position information before contextual mixing by the transformer blocks. That is a strong reason to test lexical and position explanations before naming a syntax or integration mechanism. Natural Stories uses later hooks, but that does not isolate reading modality as the cause.

Source: results/sae_regression_v2/diagnostics/selected_hooks.csv; src/surprisal.py

## 10. Fidelity checks constrain future interventions

About 55 seconds: Applying an SAE and replacing the hidden state with its reconstruction can alter the model. In a small fixed window sample, preserving each rolling-window start sharply reduces reconstruction damage at early Natural Stories hooks. This is a boundary sensitivity, and not proof that every state reconstructs faithfully. We need broader fidelity tests before attributing a behavioral change to one feature.

Source: results/sae_bos/diagnostics/behavior_summary.csv; docs/sae_methods.md

## 11. One feature exposed a scaling failure

About 60 seconds: In Natural Stories story eight, the earliest-hook feature 16980 has tiny training activations but a much larger held-out activation. Dividing by its tiny training standard deviation creates an extreme input to the linear readout. Saved coefficients reproduce the resulting errors. The original inner selector never chose this hook for Natural Stories, so the main selected-model results are unaffected. This prompted a declared follow-up sensitivity, not deletion of a difficult word.

Source: results/sae_regression_v2/diagnostics/feature_shift_audit.csv

## 12. The scaling follow-up retains useful signal

About 75 seconds: I ran the new check on the fixed earliest hook in both corpora, keeping the same nested splits and ridge grid. Provo’s after-word gain stays at about six and a half points. Natural Stories’ catastrophic errors disappear and this hook now adds modest predictive value. Its maximum absolute after-word error drops from 111.88 to 1.10 log milliseconds. This is a targeted exploratory sensitivity after observing a failure. It does not replace the original all-hook comparison or establish a new best hook.

Source: configs/scaling_sensitivity.json; results/scaling_sensitivity_v1/comparison.csv

## 13. The next research steps

About 70 seconds: The first priority is separating lexical and scale effects from the interpretability claim. Then nominate features using training data, test stability, and validate linguistic labels using independent texts and minimal pairs. Only after those steps should we change gains or ablate features in GPT-2. Broad reconstruction checks, identity interventions and matched random-feature controls are necessary to interpret the model effects. Participant-level RT robustness also remains part of the paper plan.

Source: docs/project_plan.md; docs/sae_prediction_findings.md

## 14. Questions for the research meeting

About 60 seconds: I would like the team to agree on the next experiment and on what constitutes a sufficient mechanistic claim. My recommendation is to prioritize lexical controls and scaling robustness, then identify a small reproducible feature set. We should also agree on whether participant-level analysis is necessary before feature interpretation or can run alongside it. A defensible contribution needs to explain a representation effect rather than only show improved prediction.

Source: docs/project_plan.md

## 15. Appendix: model and evaluation details

Use only if asked. GPT-2 small is frozen, with pinned checkpoint revisions. There are twelve pre-block hooks and the final post-block residual. Prefix states precede the first subtoken, and post states use the last subtoken. Word surprisal sums subtoken surprisals. BOS supplies first-word conditioning. The primary comparison uses 128/64 windows for both representation and surprisal. Every preprocessing operation in regression fits on training data.

Source: src/surprisal.py; docs/sae_regression_methods.md; configs/sae_regression.json

## 16. Appendix: three different meanings of zero

Use only if asked about the professor’s original question. An SAE activation can be zero because of its encoding rule. A regression coefficient can be zero because of the fitting procedure, especially Lasso, but the current ridge readout is not feature selection. Experimental ablation is a separate intervention we choose. These should never be conflated. Multiplying a zero activation by a gain still produces zero, so investigating missing features would require a separately designed injection or patching experiment.

Source: docs/project_plan.md

## 17. Appendix: reproducible evidence

Use if asked where to inspect results. The project saves raw predictions, fold choices and input hashes. The numbered notebooks are the current analysis workflow. Older outputs are superseded. The LaTeX notebook reads validated report tables. The scaling follow-up uses its own outputs and never overwrites the original analysis. Main caveats are aggregate RT, ten Natural Stories groups, exploration of these data, readout-capacity differences and incomplete mechanistic validation.

Source: run.sh; docs/presentation_workflow.md; docs/project_plan.md
