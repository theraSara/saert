"""
02_surprisal_extraction.py — SparseRT Pipeline
================================================
Extracts per-word surprisal and per-layer hidden states from GPT-2 small
using TransformerLens (NOT the HuggingFace API).

WHY TransformerLens (not HuggingFace):
---------------------------------------
The SAELens pre-trained SAEs (gpt2-small-res-jb, Bloom 2024) were trained
using TransformerLens with hook point blocks.{N}.hook_resid_pre.
TransformerLens folds layer norm parameters into the weights (fold_ln=True
by default), which changes the activation scale relative to raw HuggingFace
hidden states. Using HuggingFace hidden states with these SAEs produces
reconstruction errors in the thousands (instead of <1.0) and L0 values of
10,000+ (instead of 20–200) — the SAE is simply seeing a completely
different input distribution than it was trained on.

By extracting hidden states with TransformerLens we guarantee:
  1. Correct hook point naming (blocks.N.hook_resid_pre)
  2. Correct activation scale (folded layer norm)
  3. Off-by-one alignment: our "layer N" = hook_resid_pre of block N-1,
     matching the SAE's training hook exactly

What this script computes
--------------------------
Per word:
  1. surprisal   — −log₂ p(word | context) in bits, summed over BPE tokens
  2. final-subtoken hidden state at each of 12 hook points
     (hook_resid_pre of blocks 0–11), shape per layer: (n_words, 768)

GPT-2 small max = 1024 tokens. Natural Stories stories exceed this.
For long stories, surprisal uses a strided 1024-token window and assigns each
token the estimate from the window where it has the most left context.
Hidden states for SAE extraction use 128-token windows, matching the
gpt2-small-res-jb SAE training context.

Outputs
-------
results/surprisal/
    {corpus}_surprisal.csv           — per-word surprisal
    {corpus}_hidden_L{n:02d}.npy    — hidden states layer n, (n_words, 768)

Usage
-----
    python src/02_surprisal_extraction.py --corpus provo
    python src/02_surprisal_extraction.py --corpus natural_stories
    python src/02_surprisal_extraction.py --corpus both
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
PROVO_DIR = DATA_DIR / "provo"
NS_DIR    = DATA_DIR / "natural_stories"
OUT_DIR   = ROOT / "results" / "surprisal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model constants ────────────────────────────────────────────────────────────
MODEL_NAME  = "gpt2"
N_LAYERS    = 12
HIDDEN_SIZE = 768
MAX_LENGTH  = 1024
WINDOW_SIZE = 1024
STRIDE      = 512
SAE_CONTEXT_SIZE = 128
SAE_HIDDEN_STRIDE = 64


HOOK_NAMES = [f"blocks.{i}.hook_resid_pre" for i in range(N_LAYERS)]



def load_model(device: str):

    print(f"Loading GPT-2 small via TransformerLens on {device}...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        center_writing_weights=True,
        fold_ln=True,
        device=device,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  GPT-2 small loaded ({n_params:.0f}M parameters)")
    print(f"  Hook points: {HOOK_NAMES[0]} ... {HOOK_NAMES[-1]}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Core extraction: single window
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_forward_pass(
    token_ids: list[int],
    model,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    input_tensor = torch.tensor([token_ids], dtype=torch.long)  # (1, n_tokens)

    logits, cache = model.run_with_cache(
        input_tensor,
        names_filter=HOOK_NAMES,   # only cache what we need → saves memory
        return_type="logits",
        prepend_bos=False,         # we already excluded BOS in tokenization
    )
    # logits shape: (1, n_tokens, vocab_size) → squeeze batch dim
    logits = logits.squeeze(0)   # (n_tokens, vocab_size)

    # ── Surprisal ─────────────────────────────────────────────────────────
    n_tokens   = len(token_ids)
    surprisals = np.full(n_tokens, np.nan, dtype=np.float32)
    log_probs  = torch.nn.functional.log_softmax(logits, dim=-1)  # (n_tok, vocab)

    for t in range(1, n_tokens):
        lp = log_probs[t - 1, token_ids[t]].item()
        surprisals[t] = -lp / np.log(2)     # nats → bits

    # ── Hidden states ──────────────────────────────────────────────────────
    # cache[hook] shape: (1, n_tokens, 768) — squeeze batch dim
    hidden = np.stack(
        [cache[hook].squeeze(0).cpu().float().numpy() for hook in HOOK_NAMES],
        axis=0,
    )  # (n_layers, n_tokens, 768)

    return surprisals, hidden

def extract_with_sliding_window(
    token_ids: list[int],
    model,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_tokens      = len(token_ids)
    all_surprisals = np.full(n_tokens, np.nan, dtype=np.float32)
    all_hidden     = np.zeros((N_LAYERS, n_tokens, HIDDEN_SIZE), dtype=np.float32)
    best_left_context = np.full(n_tokens, -1, dtype=np.int32)

    starts = list(range(0, n_tokens, STRIDE))

    for start in tqdm(starts, desc="    Windows", leave=False):
        end        = min(start + WINDOW_SIZE, n_tokens)
        surp_win, hidden_win = run_forward_pass(token_ids[start:end], model, device)

        for local_idx, global_idx in enumerate(range(start, end)):
            if local_idx > best_left_context[global_idx]:
                all_surprisals[global_idx] = surp_win[local_idx]
                all_hidden[:, global_idx, :] = hidden_win[:, local_idx, :]
                best_left_context[global_idx] = local_idx

    return all_surprisals, all_hidden


def extract_sequence(token_ids, model, device):
    if len(token_ids) <= MAX_LENGTH:
        return run_forward_pass(token_ids, model, device)
    return extract_with_sliding_window(token_ids, model, device)


def extract_hidden_for_sae(token_ids: list[int], model, device: str) -> np.ndarray:

    n_tokens = len(token_ids)
    if n_tokens <= SAE_CONTEXT_SIZE:
        _, hidden = run_forward_pass(token_ids, model, device)
        return hidden

    all_hidden = np.zeros((N_LAYERS, n_tokens, HIDDEN_SIZE), dtype=np.float32)
    best_left_context = np.full(n_tokens, -1, dtype=np.int32)
    starts = list(range(0, n_tokens, SAE_HIDDEN_STRIDE))

    for start in tqdm(starts, desc="    SAE hidden windows", leave=False):
        end = min(start + SAE_CONTEXT_SIZE, n_tokens)
        _, hidden_win = run_forward_pass(token_ids[start:end], model, device)

        for local_idx, global_idx in enumerate(range(start, end)):
            if local_idx > best_left_context[global_idx]:
                all_hidden[:, global_idx, :] = hidden_win[:, local_idx, :]
                best_left_context[global_idx] = local_idx

    return all_hidden


# ══════════════════════════════════════════════════════════════════════════════
# Word-level aggregation (BPE subwords → words)
# ══════════════════════════════════════════════════════════════════════════════

def build_text_and_word_offsets(words: list[str]) -> tuple[str, list[tuple[int, int]]]:
    parts = []
    spans = []
    cursor = 0
    for i, word in enumerate(words):
        if i > 0:
            parts.append(" ")
            cursor += 1
        word = "" if pd.isna(word) else str(word)
        start = cursor
        parts.append(word)
        cursor += len(word)
        spans.append((start, cursor))
    return "".join(parts), spans


def build_spans_from_offsets(
    word_offsets: list[tuple[int, int]],
    token_offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    spans = []
    for word_start, word_end in word_offsets:
        hits = [
            tok_idx
            for tok_idx, (tok_start, tok_end) in enumerate(token_offsets)
            if tok_end > tok_start and tok_start < word_end and tok_end > word_start
        ]
        spans.append((hits[0], hits[-1]) if hits else (-1, -1))
    return spans


def normalise_token_offsets(offsets):
    if hasattr(offsets, "detach"):
        offsets = offsets.detach().cpu().tolist()
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]
    return [tuple(x) for x in offsets]


def align_words_to_tokens(
    words: list[str],
    token_ids: list[int],
    model,
) -> list[tuple[int,int]]:
    """
    Align words to the token positions used by the forward pass.

    Key constraint: the alignment must index into the exact same token sequence
    that the forward pass used. We use tokenizer character offsets because GPT-2
    word_ids() can drift when punctuation is attached to words.

    If the count still differs (should not happen), we fall back to a
    character-offset approach against the decoded actual token_ids.
    """
    full_text, word_offsets = build_text_and_word_offsets(words)

    # Use character offsets, not word_ids(). GPT-2 pre-tokenization can split
    # punctuation into separate pseudo-words, which makes word_ids() drift away
    # from corpus word positions.
    encoding = model.tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=False,
    )
    n_enc    = encoding["input_ids"].shape[1]

    if n_enc == len(token_ids):
        offsets = normalise_token_offsets(encoding["offset_mapping"])
        return build_spans_from_offsets(word_offsets, offsets)

    # Fallback: token counts differ — use offset_mapping for character alignment
    # This handles edge cases where to_tokens and tokenizer differ slightly
    enc2    = model.tokenizer(
        full_text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=False,
    )
    offsets = normalise_token_offsets(enc2["offset_mapping"])
    n_enc2  = len(offsets)

    spans_enc2 = build_spans_from_offsets(word_offsets, offsets)

    if n_enc2 == len(token_ids):
        tqdm.write(f'    ⚠ Alignment fallback 1 (offset_mapping): {len(words)} words, enc={n_enc}, enc2={n_enc2}, to_tokens={len(token_ids)}')
        return spans_enc2

    tqdm.write(f'    ⚠ Alignment fallback 2 (decoded chars): {len(words)} words, enc={n_enc}, enc2={n_enc2}, to_tokens={len(token_ids)}')
    # Last resort: if both encodings disagree with to_tokens,
    # decode actual token_ids and do char alignment on decoded strings
    decoded   = [model.tokenizer.decode([tid]) for tid in token_ids]
    full_dec  = "".join(decoded)
    # Map each char in full_dec to its token index
    char_to_tok: list[int] = []
    for ti, s in enumerate(decoded):
        char_to_tok.extend([ti] * len(s))
    # Strip leading spaces from BPE tokens to align with full_text
    # Both texts should have same non-space characters
    ft_ns = [i for i, c in enumerate(full_text)  if c != " "]
    fd_ns = [i for i, c in enumerate(full_dec)   if c != " "]
    ft_to_fd = {ft: fd for ft, fd in zip(ft_ns, fd_ns)}
    spans_dec = []
    for wcs, wce in word_offsets:
        hits = set()
        for ft_idx in range(wcs, wce):
            fd_idx = ft_to_fd.get(ft_idx)
            if fd_idx is not None and fd_idx < len(char_to_tok):
                hits.add(char_to_tok[fd_idx])
        tok_sorted = sorted(hits)
        spans_dec.append((tok_sorted[0], tok_sorted[-1]) if tok_sorted else (-1, -1))
    return spans_dec


def aggregate_to_words(
    df: pd.DataFrame,
    token_surprisals: np.ndarray,
    token_hidden: np.ndarray,
    word_spans: list[tuple[int,int]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map token-level outputs to word-level using freshly computed spans.

    Parameters
    ----------
    df            : word-level DataFrame for this text (sorted by word_position)
    token_surprisals : (n_tokens,) surprisal array
    token_hidden  : (n_layers, n_tokens, 768) hidden state array
    word_spans    : list of (start, end) inclusive token indices per word,
                   computed by align_words_to_tokens() using the same
                   tokenizer as the forward pass

    Surprisal  : sum over subword tokens. If any subtoken surprisal is
                 undefined, the whole word surprisal is undefined.
    Hidden state: final subtoken activation per layer. This keeps SAE inputs
                  on the token activation distribution.
    """
    n_tokens = token_surprisals.shape[0]
    n_words  = len(df)
    word_surp   = np.full(n_words, np.nan, dtype=np.float32)
    word_hidden = np.zeros((N_LAYERS, n_words, HIDDEN_SIZE), dtype=np.float32)

    for i, (s, e) in enumerate(word_spans):
        if s < 0 or e < 0 or s >= n_tokens:
            # Invalid span — leave as NaN
            continue
        e_excl = min(e + 1, n_tokens)   # clamp to sequence length
        token_range = token_surprisals[s:e_excl]
        word_surp[i] = np.nan if np.isnan(token_range).any() else token_range.sum()
        word_hidden[:, i, :] = token_hidden[:, e_excl - 1, :]

    return word_surp, word_hidden


# ══════════════════════════════════════════════════════════════════════════════
# Per-corpus extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_corpus(corpus: str, model, device: str):
    """Extract surprisal + hidden states for one corpus."""

    if corpus == "provo":
        data_path      = PROVO_DIR / "provo_prepared.csv"
        tok_order_path = None
    else:
        data_path      = NS_DIR / "natural_stories_prepared.csv"
        tok_order_path = NS_DIR / "story_token_order.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Prepared data not found at {data_path}\n"
            "Run 01_data_prep.py first."
        )

    df        = pd.read_csv(data_path)
    tok_order = pd.read_csv(tok_order_path) if tok_order_path and tok_order_path.exists() else None

    records          = []
    hidden_by_layer  = [[] for _ in range(N_LAYERS)]
    text_ids         = df["text_id"].unique()

    print(f"\nExtracting from {corpus} ({len(text_ids)} texts)...")

    for text_id in tqdm(text_ids, desc=f"  {corpus}"):
        text_df = df[df["text_id"] == text_id].sort_values("word_position")

        if tok_order is not None:
            full_words = (
                tok_order[tok_order["text_id"] == text_id]
                .sort_values("word_position")["word"]
                .fillna("").tolist()
            )
        else:
            full_words = text_df["word"].fillna("").tolist()

        # Tokenize using HuggingFace tokenizer directly (no truncation).
        # Do NOT use model.to_tokens() — it silently truncates to MAX_LENGTH
        # and may prepend BOS even with prepend_bos=False on some TL versions.
        # We tokenize once, get both token_ids and word alignment from the
        # same encoding object, guaranteeing perfect consistency.
        full_text, full_word_offsets = build_text_and_word_offsets(full_words)
        encoding  = model.tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=False,         # no truncation — we handle long seqs ourselves
            add_special_tokens=False, # no BOS/EOS — match prepend_bos=False
        )
        token_ids = encoding["input_ids"][0].tolist()   # list[int], full length
        token_offsets = normalise_token_offsets(encoding["offset_mapping"])

        n_tokens = len(token_ids)
        if n_tokens > MAX_LENGTH:
            tqdm.write(
                f"    Story {text_id}: {n_tokens} tokens > {MAX_LENGTH} "
                f"→ using sliding window"
            )

        token_surp, _ = extract_sequence(token_ids, model, device)
        token_hidden = extract_hidden_for_sae(token_ids, model, device)

        word_spans = build_spans_from_offsets(full_word_offsets, token_offsets)

        # text_df has only the words with RT; word_spans covers ALL full_words
        # (including punctuation-only tokens in NS story_token_order).
        # We need to map text_df words to their positions in full_words.
        # For Provo: full_words == text_df words (same list).
        # For NS: full_words comes from story_token_order which may have extra
        # tokens. We match by word_position.
        if tok_order is not None:
            # Build position → span lookup from full story
            full_positions = (
                tok_order[tok_order["text_id"] == text_id]
                .sort_values("word_position")["word_position"]
                .tolist()
            )
            pos_to_span = {pos: word_spans[i] for i, pos in enumerate(full_positions)}
            text_spans = [
                pos_to_span.get(int(row["word_position"]), (-1, -1))
                for _, row in text_df.iterrows()
            ]
        else:
            text_spans = word_spans   # Provo: 1-to-1

        word_surp, word_hidden = aggregate_to_words(
            text_df, token_surp, token_hidden, text_spans
        )

        # Diagnostic: flag zero-surprisal words (suspicious, should not happen)
        zero_mask = (word_surp == 0) & ~np.isnan(word_surp)
        if zero_mask.sum() > 0:
            zero_idx = np.where(zero_mask)[0]
            zero_words = text_df.iloc[zero_idx]["word"].values
            tqdm.write(
                f"    ⚠ Story {text_id}: {len(zero_idx)} zero-surprisal words — "
                f"examples: {zero_words[:3]}"
            )

        for i, (_, row) in enumerate(text_df.iterrows()):
            span_start, span_end = text_spans[i]
            row_uid = (
                f"{corpus}:{row['text_id']}:{int(row['word_position'])}"
            )
            records.append({
                "hidden_row_idx": len(records),   # direct index into .npy arrays
                "row_uid":        row_uid,
                "text_id":        text_id,
                "word_position":  row["word_position"],
                "word":           row["word"],
                "token_ids_str":  str(token_ids[span_start:span_end + 1]) if span_start >= 0 else "[]",
                "n_tokens":       int(span_end - span_start + 1) if span_start >= 0 else 0,
                "start_token_idx": int(span_start),
                "end_token_idx":   int(span_end),
                "hidden_token_idx": int(span_end),
                "hidden_context_size": SAE_CONTEXT_SIZE,
                "hidden_context_stride": SAE_HIDDEN_STRIDE,
                "surprisal":      float(word_surp[i]),
            })
            for layer in range(N_LAYERS):
                hidden_by_layer[layer].append(word_hidden[layer, i, :])

    # ── Surprisal CSV ──────────────────────────────────────────────────────
    surp_df = pd.DataFrame(records)
    stale_alignment_cols = [
        "token_ids_str", "n_tokens", "start_token_idx", "end_token_idx",
        "hidden_token_idx", "hidden_context_size", "hidden_context_stride",
        "row_uid",
    ]
    df_for_merge = df.drop(columns=[c for c in stale_alignment_cols if c in df.columns])
    surp_df = df_for_merge.merge(
        surp_df[[
            "text_id", "word_position", "row_uid", "surprisal", "hidden_row_idx",
            "token_ids_str", "n_tokens", "start_token_idx", "end_token_idx",
            "hidden_token_idx", "hidden_context_size", "hidden_context_stride",
        ]],
        on=["text_id", "word_position"],
        how="left",
        validate="one_to_one",   # catches duplicate key merges early
    )
    surp_out = OUT_DIR / f"{corpus}_surprisal.csv"
    surp_df.to_csv(surp_out, index=False)

    # ── Hidden state .npy files ────────────────────────────────────────────
    print(f"  Saving hidden states ({N_LAYERS} layers × {len(records)} words × {HIDDEN_SIZE} dims)...")
    for layer in range(N_LAYERS):
        hs = np.stack(hidden_by_layer[layer], axis=0)   # (n_words, 768)
        np.save(OUT_DIR / f"{corpus}_hidden_L{layer+1:02d}.npy", hs)

    # ── Summary ────────────────────────────────────────────────────────────
    valid = surp_df["surprisal"].dropna()
    n_nan = surp_df["surprisal"].isna().sum()
    n_zero = (surp_df["surprisal"] == 0).sum()
    n_neg = (surp_df["surprisal"] < 0).sum()
    n_texts = surp_df["text_id"].nunique()

    print(f"\n{'─'*60}")
    print(f"  {corpus.upper()} — EXTRACTION SUMMARY")
    print(f"{'─'*60}")
    print(f"  Words extracted:      {len(surp_df):,}")
    print(f"  Valid surprisals:     {len(valid):,}")
    print(f"  NaN:                  {n_nan} (expected ≈ {n_texts})")
    print(f"  Zero surprisal:       {n_zero} {'⚠️ BUG!' if n_zero > 0 else '✓'}")
    print(f"  Negative:             {n_neg} {'⚠️ BUG!' if n_neg > 0 else '✓'}")
    print(f"")
    print(f"  Mean:                 {valid.mean():.3f} bits (typical: 4–8)")
    print(f"  Median:               {valid.median():.3f} bits")
    print(f"  Std:                  {valid.std():.3f} bits")
    print(f"  Min / Max:            {valid.min():.3f} / {valid.max():.3f} bits")

    # Diagnostic guidance
    if n_zero > 0:
        print(f"\n  ⚠️ ERROR: {n_zero} zero-surprisal words found!")
        print(f"    GPT-2 should almost never assign p=1 to tokens.")
        print(f"    This indicates a tokenization or alignment bug.")
        print(f"    Check if words are mapping to wrong token positions.")
    if valid.mean() < 2.0:
        print(f"\n  ⚠️ WARNING: Mean surprisal too low ({valid.mean():.3f})")
        print(f"    Suggests tokenizer mismatch or wrong token sequences.")
    if valid.mean() > 20.0:
        print(f"\n  ⚠️ WARNING: Mean surprisal too high ({valid.mean():.3f})")
        print(f"    Check tokenization and probability computation.")
    print(f"{'─'*60}\n")
    print(f"  Surprisal → {surp_out}")
    print(f"  Hidden states → {OUT_DIR}/{corpus}_hidden_L01.npy ... L12.npy\n")

    return surp_df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SparseRT — Step 02: Surprisal + Hidden State Extraction (TransformerLens)"
    )
    parser.add_argument("--corpus", choices=["provo", "natural_stories", "both"], default="both")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print("Auto-detected Apple Silicon MPS")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Auto-detected CUDA GPU")
        else:
            device = "cpu"
            print("No GPU — using CPU")
    else:
        device = args.device

    model  = load_model(device)
    corpora = ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]

    for corpus in corpora:
        try:
            extract_corpus(corpus, model, device)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {e}\n")

    print("Surprisal extraction complete.")
    print("Next step: python src/03_baseline_regression.py")
    print("Then re-run: python src/04_sae_extraction.py")


if __name__ == "__main__":
    main()
