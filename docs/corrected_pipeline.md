# Corrected data and surprisal pipeline

This implementation separates complete presented stimuli from RT availability.
It does not implement new regression models, SAE interventions, or confirmatory
human cognitive claims. The original saved results must be re-evaluated after
regeneration. See the dated research audit for the scientific motivation.

## Inputs

All paths below are relative to the chosen `--data-dir` (default `data`).

| Corpus | Required | Optional |
|---|---|---|
| Provo | `provo/Provo_Corpus-Eyetracking_Data.csv`; `--provo-rt FFD`, `GZD`, or `TRT` | External stimulus table keyed by interest area; source-column override, participant-column override, trial RT bounds |
| Natural Stories | `natural_stories/all_stories.tok`; `natural_stories/processed_wordinfo.tsv` | `processed_RTs.tsv` (official spelling) or `processed_RT.tsv`, but not both |

The official Provo eye-tracking file supplies the stimulus sequence through
`Text_ID`, `IA_ID`, and `IA_LABEL`. Use these eye-tracking interest areas, not
`Word_Number`/`Word`, which are norm-related fields and can be missing or differ
from the displayed regions. The paper explicitly documents this distinction in
[Table 2](https://doi.org/10.3758/s13428-017-0908-4). The loader retains the original
norm fields as `norm_word_position` and `norm_word` in participant output. Missing
norm metadata does not exclude a valid eye-tracking observation.

The local official file contains 2,743 consecutive interest areas across 55 texts,
with consistent labels across participants. This is a region count, not a claim
that each region corresponds to exactly one ordinary lexical word. Original
case, punctuation, and spelling in `IA_LABEL` are retained; only surrounding
whitespace is stripped. Repeated identical region definitions are collapsed and
conflicting definitions fail. A separate stimulus download is not required.

For an optional `--provo-stimuli` override, provide `text_id`, `word_position`,
`word`, where positions correspond to **IA_ID**; the table must match the regions
exactly. A norm-based table is not automatically interchangeable with this key.

Illustrative schema (not a replacement corpus):

```tsv
text_id	word_position	word
1	1	There
1	2	are
1	3	now
```

Every text must start at position 1 and have contiguous positions. These checks
detect initial/internal omissions, but cannot prove that an unmarked trailing
word or an entire text is missing: verify completeness against the authoritative
source. IDs must be positive integers. Word strings, punctuation, and case are
checked exactly across stimulus and RT files; mismatches are not normalized away.
Original line breaks/spacing are not available in these word tables: model text
is explicitly defined as corpus words joined by one ASCII space.

## RT and frequency policy

- FFD: `IA_FIRST_FIXATION_DURATION`.
- GZD: `IA_FIRST_RUN_DWELL_TIME`.
- TRT: `IA_DWELL_TIME`.
- No fallback between measures. An explicit column override must be justified
  from the source data dictionary.
- Provo requires participant IDs (`Participant_ID`, with legacy `Part_ID` fallback), rejects duplicate
  participant × word rows, preserves all source trial columns, and records
  `raw_RT`, `analysis_RT`, and `rt_status`.
- Nonpositive or missing durations do not enter duration means. They remain in
  the participant table and do not remove the stimulus word. Skipping and
  first-pass/progressive-fixation eligibility need a prespecified analysis policy;
  this loader does not infer those flags or perform new participant exclusions.
- Optional `--provo-rt-min` / `--provo-rt-max` operate on trials before aggregation.
  There is no implicit 100–3000 ms cutoff. Counts, SD, mean RT, and mean log RT
  are retained. `log_primary_RT` is **log of mean RT**, not mean log RT.
- Natural Stories retains upstream `meanItemRT`, `nItem`, and optional SD/geometric
  summaries. If participant RTs are supplied, they are validated and copied, but
  do not silently replace the authors' aggregates. Use the corrected upstream
  post-2021 release; source hashes identify the local files, not their validity.
- `zipf_freq` uses the English `wordfreq` large list. The duplicate
  `log_unigram_freq` predictor is removed. `word_clean` is only for lexical
  predictors; the original `word` is used for model input.
- All stimulus rows are retained, even when `primary_RT` is missing. Regression
  exclusion should occur later, after context and lagged predictors are constructed.

## Outputs from data.py

`--output-dir` is a root containing one directory per corpus. If omitted, these
files go under `--data-dir/<corpus>`:

- `<corpus>_prepared.csv`: all stimulus words, RT summaries, lexical predictors,
  exact tokenizer spans/IDs, `row_uid`, `rt_measure`, and `preparation_id`.
- `story_token_order.csv`: complete ordered words for both corpora.
- `<corpus>_participants.csv`: retained participant records when available.
- `<corpus>_prepared_manifest.json`: source/output SHA256 hashes, tokenizer
  fingerprint, code hashes, versions, counts, and filtering settings.

No tokenizer or model is downloaded merely by importing a module or requesting
`--help`. Raw inputs are validated before tokenizer loading.

## Window and scoring policy

`surprisal.py` requires the new prepared manifest and validates artifact hashes,
word identity, row IDs, spans, and exact token IDs. It does not accept an old
prepared CSV as if it had the new provenance.

Default long-context scores use a 1,024-token window, stride 512. SAE-state
extraction uses 128-token windows, stride 64. Both are configurable; require
`2 <= window <= 1024` and `1 <= stride < window`.

For an overlapped token, the earliest covering window has the most preceding
tokens. Only newly covered tokens are retained from subsequent windows. Redundant
trailing windows are not evaluated. This is maximum context **among the evaluated
windows**, not maximum possible context for each token. Set stride 1 for the
rolling policy under this convention if the compute cost is acceptable.

The target token is included in the input window; logits at the preceding
position predict it. Thus this implementation scores with at most `window-1`
preceding tokens. Absolute positions restart at zero in each window. One explicit
GPT-2 BOS token (`<|endoftext|>`, ID 50256) is prepended to each complete story
before windowing. It counts toward the window limit and is never reinserted at
window boundaries. Its logits predict the first text token. Every word, including
the first word, therefore has a finite score; multiple subtokens are summed as
usual. This is a document-start boundary convention, not an estimate using an
unknown preceding passage. The BOS token itself is not an RT observation.

Extraction schema 3 records the BOS policy and token ID. Prepared spans and
`token_idx`/`hidden_token_idx` retain text-only indices. Explicit `model_token_idx`
columns and all window starts use the augmented sequence (BOS at 0, first text
token at 1). Context counts include BOS when it remains in the window.

Scores are cross-entropy in bits. Word scores sum all assigned subtoken scores,
including attached punctuation and any explicit preceding separator token.
Character offsets and exact decoded-text round trips validate the partition;
there are no guessed alignment fallbacks or silently clamped invalid spans.

Both long-context `surprisal` and `surprisal_matched_context` are emitted. The latter
uses exactly the context/window policy used to extract SAE hidden states.

## Hidden-state semantics

L01–L12 preserve the existing filename convention:

| File label | Hook | Completed transformer blocks |
|---|---|---:|
| L01 | `blocks.0.hook_resid_pre` | 0 |
| L03 | `blocks.2.hook_resid_pre` | 2 |
| L12 | `blocks.11.hook_resid_pre` | 11 |
| final_resid_post | `blocks.11.hook_resid_post` | 12 |

All 13 sites are extracted. The downstream SAE script still processes its existing
12 pre-block sites; adding the final-output SAE is a separate downstream change.

Normal hidden arrays use the final subtoken of the observed current word. They
are post-observation predictors, not pre-word anticipation. Optional `--save-prefix`
arrays contain the state before the first subtoken **in that target's short-context
scoring window**. Simply shifting already-selected post-token states would give
the wrong window at some boundaries. The first word's prefix is now the BOS
activation and is finite. Its `prefix_kind` is `bos`, `prefix_model_token_idx` is
0, and text-only `prefix_token_idx` is -1 (a sentinel, not an array index).
It is a boundary representation with no preceding linguistic content.

Model loading remains on TransformerLens's legacy HookedTransformer interface
with explicit folding/centering flags. Versions, resolved model commit, device,
precision, hook names, and code hashes are recorded. Matching a 128-token training
context is a declared operating condition, not proof of SAE fidelity. Actual
checkpoint-specific BOS and weight-processing compatibility still need validation.

## Outputs from surprisal.py

- `<corpus>_surprisal.csv`: all prepared words plus both scores, row indices,
  extraction ID, and word-level context metadata.
- `<corpus>_token_windows.csv`: token IDs, word ownership, offsets, scores, window
  starts, and actual preceding-context lengths under both policies.
- `<corpus>_hidden_rows.csv`: immutable mapping between array row index and word ID.
- `<corpus>_hidden_L01.npy` … `_L12.npy`, plus `_hidden_final_resid_post.npy`.
- Optional corresponding `_prefix_hidden_*.npy` arrays.
- `<corpus>_extraction_manifest.json`: all settings and SHA256 hashes; written last
  as the run completion record.

States are written into memory-mapped arrays, keeping corpus-wide output arrays
off the accelerator. No SAE intervention occurs in this script. Outputs are
staged before publishing; replacement of multiple files is not a filesystem-wide
atomic transaction, so consumers should verify the final manifest hashes.

Existing outputs require `--overwrite`. A fresh output directory is preferable
while comparing versions. Stale optional participant/prefix artifacts trigger an
error rather than silently surviving as if they belonged to a new run.

## Run commands and downstream transition

```sh
python -s src/data.py --corpus natural_stories --output-dir data/prepared_v2
python -s src/surprisal.py --corpus natural_stories --data-dir data/prepared_v2 --output-dir results/surprisal_bos --save-prefix

python -s src/data.py --corpus provo --provo-rt FFD --output-dir data/prepared_v2
python -s src/surprisal.py --corpus provo --data-dir data/prepared_v2 --output-dir results/surprisal_bos --save-prefix
```

Use `GZD` if gaze duration is the selected research outcome. Pin matching commit
SHAs via `--tokenizer-revision` and `--revision` for repeatable extraction.

Existing schema-2 prepared data can be reused. Existing no-BOS extraction results
must be regenerated in full, including hidden states: adding BOS changes model
positions and sliding-window boundaries. Use the new `surprisal_bos` directory;
do not splice first-word scores into old arrays. Previously fitted downstream
models must also be rerun using a consistent extraction version.

The corrected baseline now consumes these versioned paths, validates extraction
manifests, preserves the prepared sentence-final control, and removes the old
aggregate RT cutoffs. See `baseline_methods.md` and run `bash run.sh baseline both`,
then `bash run.sh notebook both`. The older SAE scripts are not yet connected to
this new evaluation; do not interpret their old outputs as a continuation of it.

## Validation

```sh
python -s -m unittest discover -s tests -v
```

Offline tests cover complete stimulus preservation, explicit RT selection,
trial-before-mean exclusion, exact alignment, punctuation/Unicode, explicit BOS handling,
word sums, 1,024-token boundaries, prefix-window consistency, artifact hashes,
and causal score/state behavior in a small randomly initialized transformer.
They do not certify the unavailable raw corpora or pretrained SAE reconstruction.
