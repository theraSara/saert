# SAE extraction and fidelity

This stage converts the verified `results/surprisal_bos` activations into sparse
features. It does not fit an RT model or change the saved original surprisals.

## Checkpoints and representation

The release is `gpt2-small-res-jb`, repository
[`jbloom/GPT2-Small-SAEs-Reformatted`](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted/tree/57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9),
pinned to commit `57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9`. Both config and
weights are downloaded at that revision and their hashes are recorded. SAELens 6
loads the pinned disk files with the release's `center_writing_weights=True`
override. The code rejects unimplemented activation normalization.

There are **13 locations, not 13 transformer layers**: `L01` through `L12` map
to `blocks.0.hook_resid_pre` through `blocks.11.hook_resid_pre` (0–11 completed
blocks), and `final_resid_post` maps to `blocks.11.hook_resid_post` (12 completed
blocks). Every location has its own 24,576-feature dictionary; feature IDs are
only meaningful together with the hook and checkpoint revision.

The model, hook, dimensions, 128-token context, extraction processing flags,
source hashes and row order are checked. Legacy training configs do not record
every processing flag or their BOS policy. Modern loader defaults are not proof
of historical settings. Empirical fidelity checks supplement the metadata;
they do not eliminate this provenance limitation.

For every stimulus word, save two **separate** CSR matrices:

- `post_features.npz`: encode the state at the current word's final subtoken.
  These features observe the current word and can reflect its lexical identity.
- `prefix_features.npz`: encode the position preceding the first subtoken,
  using that target subtoken's original scoring window. The first word has a
  BOS prefix. This is a different predictive question from post-word analysis.

Do not average subtoken states before encoding. All feature columns are retained,
including inactive columns, in original checkpoint order. Load with
`scipy.sparse.load_npz`; no pickle is needed. Do not densify the whole corpus.
`rows.csv` preserves `row_uid` and `hidden_row_idx`. The extraction includes
stimulus rows without usable RT because RT filtering belongs to regression.

## Geometric checks

Per-row outputs contain the active-feature count, squared reconstruction error,
squared input norm, relative L2 error, and cosine similarity. The summary's
centered reconstruction R² is

`1 - sum ||x - reconstruction(x)||² / sum ||x - mean(x)||²`.

It is not variance in human RT. It can be negative. When the denominator is
effectively zero (identical BOS prefixes), R² is undefined, not zero or one.
The all-prefix summary mixes BOS and ordinary states, so inspect the separate
`text_token` and `bos` summaries. Large BOS vectors can otherwise make aggregate
reconstruction statistics look misleadingly favorable.

`feature_stats.csv` reports every feature's activation count and mean solely for
quality assessment. It is **not** a feature-selection list. Any filtering,
scaling, RT-based selection or tuning must be fitted inside subsequent training
folds, including inner folds used to choose regularization.

## Behavioral checks

The default is four deterministic windows per corpus, at most one per sampled
text, seed 42. Selection does not use RT or SAE results. The first selected text
contributes its initial window; other window positions are sampled from the
same 128/64 window plan as extraction. Exact text IDs, token IDs, bounds, and
claimed targets are recorded. Only targets owned by the sampled window are
scored, so overlapping windows do not duplicate target scores.

Reload the exact original GPT-2 revision and processing configuration. Recompute
sampled hidden states and compare both saved post-word and prefix states against
the live cache. Require agreement within `atol=2e-4, rtol=2e-5`. At each hook,
compare the original forward pass with these interventions **one hook at a time**:

1. Identity replacement: must preserve the original prediction distribution.
2. SAE encode/decode at every non-BOS position, preserving actual BOS when present.
3. Zero ablation at exactly those same positions.
4. Reconstruction and zero-ablation sensitivities that also preserve local
   position zero of rolling windows, even when it is an ordinary text token.

This second boundary policy was added after the L03 pilot, before full extraction:
reconstructing the initial ordinary token of rolling Natural Stories windows
caused a much larger prediction change than reconstructing the retained word
states suggested. Keep both policies visible. Preserving the window start is a
restricted intervention, not evidence that the SAE faithfully reconstructs that
excluded state, and is not permission to add a fresh BOS at each window.

Report token-weighted next-token NLL in bits, its change from the original model,
and KL(original distribution || reconstructed distribution) in bits. Also report
`(NLL_zero - NLL_reconstruction) / (NLL_zero - NLL_original)` when the denominator
is positive. This loss-recovery ratio uses the matching boundary policy, can lie
outside [0, 1], and is not a percentage of reading time explained. A very damaging
zero ablation can make recovery look favorable; read KL and absolute loss change
alongside it. Protecting BOS leaves the first-token prediction unchanged at the
intervened hook; that is not a test of reconstructing BOS.

The small sample is a smoke test, with **no population confidence intervals**.
It is not an RT regression, feature ablation, gain-tuning experiment, or proof
that a feature represents a human computation. Poor fidelity requires diagnosis
before intervention claims. Expand the fixed sample before a paper's final
mechanistic experiments, using separate output directories.

## Files and commands

```sh
bash run.sh sae-pilot both
bash run.sh sae both
bash run.sh sae-report both
bash run.sh sae-notebook both
```

Pilot output is `results/sae_pilot`; full output is `results/sae_bos`. Existing
extraction outputs require a new directory or explicit `OVERWRITE=1`. Reports
can be regenerated from saved features. The `all` stage intentionally still
covers preparation through the baseline notebook; SAE stages are explicit.

Each `<corpus>/<hook label>/` contains matrices, row mapping, per-row
reconstruction diagnostics, complete feature statistics, `geometry.csv`,
`behavior.csv`, and a manifest installed last as a completion marker. `--hooks`
allows selected locations; `--fidelity-windows 0` explicitly skips behavioral
checks and must not be described as validated behavioral fidelity.

`02_sae_reconstruction.ipynb` and `src/sae_diagnostics.py` validate saved output hashes,
row mapping, sparse shape/counts and consistency of the sampled windows across
hooks before exporting CSV/LaTeX tables and PDF/SVG/PNG figures. The figure is a
quality diagnostic or potential supplement, not an RT improvement figure.

## Next analysis

### Completed run: 20 September 2026

The CPU run completed all 13 hooks for Provo (2,743 regions) and Natural Stories
(10,256 words): 26 corpus/hook manifests and 52 sparse matrices. Live/saved
post-word and prefix states matched exactly on the sampled positions. Identity
interventions preserved prediction distributions. All 33 offline tests passed.

The retained post-word states have mean relative reconstruction errors of
approximately 0.045–0.298 in Provo and 0.048–0.326 in Natural Stories across
hooks. These values should be compared by location, not interpreted as RT effects.

On the fixed four-window sample, Natural Stories L02 is a clear boundary failure:
its NLL increase is 6.395 bits/token when only actual BOS is protected, versus
0.032 when every window start is protected. L03 changes from 3.158 to 0.093.
The final residual output still increases NLL by 0.533 bits/token in Provo and
0.679 in Natural Stories under the restricted policy. Thus boundary protection
addresses a specific early-hook problem; it does not make reconstruction
behaviorally exact at every hook. With only 236 scored Provo tokens and 319
Natural Stories tokens, these are diagnostic observations, not final corpus-wide
effect estimates. The full saved tables retain both policies and every hook.

The source artifacts are suitable for the next predictive comparison, with
fidelity limitations documented. Any feature-level model intervention will need
a reconstruction baseline, explicit boundary treatment and broader validation.

### Regression plan

Replace the legacy SAE regression with nested story-held-out evaluation. Fit the
same eligible RT outcomes and controls as the corrected baseline, compare
controls + surprisal with controls + surprisal + SAE features, and include dense
state baselines with comparable tuning. Keep prefix/post-word questions separate.
Train preprocessing and regularization in inner training folds; do not treat
previously computed global or outer-OOF residuals as an independent training target.
Use the original outcome with controls in each fit, or cross-fit nuisance models
within the appropriate training split. Reserve feature interpretation and
intervention choices for training data, then validate on held-out stories.
