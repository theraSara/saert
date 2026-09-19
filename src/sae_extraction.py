import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
OUT_DIR  = ROOT / "results" / "sae"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SURP_DIR = ROOT / "results" / "surprisal"
REG_DIR  = ROOT / "results" / "regression"

# ── SAE constants ──────────────────────────────────────────────────────────────
SAE_RELEASE  = "gpt2-small-res-jb"          # Bloom (2024) all-layer release
N_LAYERS     = 12
N_FEATURES   = 24_576                       # SAE width (32× expansion)
HIDDEN_SIZE  = 768
ALL_LAYERS   = list(range(1, N_LAYERS + 1)) # 1-indexed throughout pipeline

TOP_K = 512


def load_sae(layer: int, device: str):
    from sae_lens import SAE

    # hook_resid_pre of block index = layer - 1  (0-indexed blocks)
    hook_name = f"blocks.{layer - 1}.hook_resid_pre"

    sae, cfg_dict, _ = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=hook_name,
        device=device,
    )
    sae.eval()
    return sae


def get_sae_width(sae) -> int:
    cfg = getattr(sae, "cfg", None)
    for name in ("d_sae", "dict_size", "d_hidden"):
        value = getattr(cfg, name, None) if cfg is not None else None
        if value is not None:
            return int(value)
    return N_FEATURES


@torch.no_grad()
def encode_hidden_states(
    hidden_states: np.ndarray,   # (n_words, 768)
    sae,
    device: str,
    batch_size: int = 512,
) -> tuple[
    list[np.ndarray], list[np.ndarray], np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    n_words = hidden_states.shape[0]
    feat_indices = []
    feat_values  = []
    n_active_list  = []
    recon_err_list = []
    input_norm_list = []
    recon_err_rel_list = []

    for start in range(0, n_words, batch_size):
        end   = min(start + batch_size, n_words)
        batch = torch.tensor(
            hidden_states[start:end], dtype=torch.float32, device=device
        )  # (B, 768)

        # Forward through SAE
        # sae.encode() returns the feature activation vector f(x)
        feature_acts = sae.encode(batch)   # (B, 24576)

        # Reconstruction for error computation
        recon = sae.decode(feature_acts)   # (B, 768)
        input_norm = batch.norm(dim=-1)
        err = (batch - recon).norm(dim=-1)
        err_rel = err / input_norm.clamp_min(1e-8)
        recon_err_list.append(err.cpu().numpy())
        input_norm_list.append(input_norm.cpu().numpy())
        recon_err_rel_list.append(err_rel.cpu().numpy())

        feature_acts_np = feature_acts.cpu().numpy()  # (B, 24576)

        for i in range(end - start):
            acts = feature_acts_np[i]           # (24576,)
            active_mask = acts > 0
            indices = np.where(active_mask)[0].astype(np.int32)
            values  = acts[active_mask].astype(np.float32)
            feat_indices.append(indices)
            feat_values.append(values)
            n_active_list.append(len(indices))

    n_active    = np.array(n_active_list, dtype=np.int16)
    recon_error = np.concatenate(recon_err_list).astype(np.float32)
    input_norm = np.concatenate(input_norm_list).astype(np.float32)
    recon_error_rel = np.concatenate(recon_err_rel_list).astype(np.float32)

    return feat_indices, feat_values, n_active, recon_error, input_norm, recon_error_rel


def compute_top_k_dense(
    feat_indices: list[np.ndarray],
    feat_values:  list[np.ndarray],
    n_words: int,
    top_k_feature_ids: np.ndarray,   # (K,) — which features to keep densely
) -> np.ndarray:
    K     = len(top_k_feature_ids)
    dense = np.zeros((n_words, K), dtype=np.float32)

    # Build reverse lookup: feature_id → column index in dense matrix
    feat_to_col = {fid: col for col, fid in enumerate(top_k_feature_ids)}

    for word_idx, (indices, values) in enumerate(zip(feat_indices, feat_values)):
        for feat_id, val in zip(indices, values):
            col = feat_to_col.get(int(feat_id))
            if col is not None:
                dense[word_idx, col] = val

    return dense


def select_top_k_features(
    feat_indices: list[np.ndarray],
    n_words: int,
    top_k: int = TOP_K,
    n_features: int = N_FEATURES,
) -> np.ndarray:
    """
    Select the top-K features by activation frequency across all words.
    These are the features most likely to be informative for regression.
    """
    max_seen = -1
    for indices in feat_indices:
        if len(indices):
            max_seen = max(max_seen, int(indices.max()))
    n_features = max(n_features, max_seen + 1)
    top_k = min(top_k, n_features)

    # Count how many words each feature fires on
    freq = np.zeros(n_features, dtype=np.int32)
    for indices in feat_indices:
        freq[indices] += 1
    # Return indices of the top-K most frequent features
    top_k_ids = np.argsort(freq)[-top_k:][::-1].astype(np.int32)
    return top_k_ids


# ══════════════════════════════════════════════════════════════════════════════
# Per-corpus extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_corpus(corpus: str, layers: list[int], device: str, batch_size: int = 512):
    """
    Run SAE extraction for one corpus across specified layers.

    For each layer:
      1. Load hidden states from script 02 output
      2. Load the corresponding pre-trained SAE
      3. Encode hidden states → sparse feature activations
      4. Save sparse activations + dense top-K summary
      5. Accumulate per-word metadata (n_active, recon_err)
    """
    # ── Load residualized RT data (for word ordering / metadata) ──────────
    rt_path = REG_DIR / f"{corpus}_residualized_RT.csv"
    if not rt_path.exists():
        raise FileNotFoundError(
            f"Residualized RT not found at {rt_path}\n"
            "Run 03_baseline_regression.py first."
        )
    rt_df = pd.read_csv(rt_path)
    if "row_uid" not in rt_df.columns:
        rt_df["row_uid"] = (
            corpus + ":" + rt_df["text_id"].astype(str)
            + ":" + rt_df["word_position"].astype(int).astype(str)
        )
    if "hidden_row_idx" not in rt_df.columns:
        rt_df["hidden_row_idx"] = np.arange(len(rt_df), dtype=int)
    if "hidden_context_size" not in rt_df.columns:
        rt_df["hidden_context_size"] = np.nan
    if "hidden_context_stride" not in rt_df.columns:
        rt_df["hidden_context_stride"] = np.nan
    n_words = len(rt_df)
    print(f"\n  {corpus}: {n_words:,} words | processing {len(layers)} layers")

    # ── Accumulate metadata across layers ─────────────────────────────────
    # metadata_rows: one dict per (word, layer) — used for script 05
    metadata_rows = []

    for layer in tqdm(layers, desc=f"  Layers ({corpus})"):

        # ── Load hidden states for this layer ─────────────────────────────
        hs_path = SURP_DIR / f"{corpus}_hidden_L{layer:02d}.npy"
        if not hs_path.exists():
            print(f"    [SKIP] Hidden states not found: {hs_path}")
            continue

        # Load the full hidden state file, then select only rows that have
        # valid RT data using hidden_row_idx from residualized_RT.csv.
        # This excludes the zero-vector rows (NaN surprisal words) which
        # would cause the SAE to fire thousands of spurious features.
        hidden_full = np.load(hs_path)   # (n_full_words, 768)

        if "hidden_row_idx" in rt_df.columns:
            row_indices = rt_df["hidden_row_idx"].astype(int).values
            if row_indices.max() >= hidden_full.shape[0] or row_indices.min() < 0:
                raise ValueError(
                    f"{corpus} layer {layer}: hidden_row_idx is outside hidden array bounds"
                )
            hidden = hidden_full[row_indices]   # (n_words, 768) — valid words only
        else:
            # Fallback: assume 1-to-1 alignment (Provo with no NaN drops)
            hidden = hidden_full[:n_words]

        assert hidden.shape == (n_words, HIDDEN_SIZE), (
            f"Shape mismatch: expected ({n_words}, {HIDDEN_SIZE}), "
            f"got {hidden.shape}"
        )

        # Safety check: warn if any zero-norm hidden states remain
        norms = np.linalg.norm(hidden, axis=1)
        n_zero = (norms == 0).sum()
        if n_zero > 0:
            tqdm.write(f"    ⚠ {n_zero} zero-norm hidden states after filtering — check alignment")

        # ── Load SAE ───────────────────────────────────────────────────────
        tqdm.write(f"    Loading SAE for layer {layer}...")
        sae = load_sae(layer, device)
        n_features = get_sae_width(sae)

        # ── Encode hidden states → feature activations ────────────────────
        tqdm.write(f"    Encoding {n_words:,} hidden states...")
        (
            feat_indices, feat_values, n_active, recon_error,
            input_norm, recon_error_rel
        ) = encode_hidden_states(
            hidden, sae, device, batch_size=batch_size
        )

        # ── Select top-K features and build dense summary ─────────────────
        top_k_ids = select_top_k_features(feat_indices, n_words, TOP_K, n_features)
        dense_topk = compute_top_k_dense(
            feat_indices, feat_values, n_words, top_k_ids
        )  # (n_words, TOP_K)

        # ── Save sparse activations ────────────────────────────────────────
        # Store as variable-length arrays: indices and values per word
        # scipy sparse or ragged arrays — we use numpy object arrays
        sparse_out = OUT_DIR / f"{corpus}_sae_L{layer:02d}_sparse.npz"
        np.savez_compressed(
            sparse_out,
            feat_indices  = np.array(feat_indices, dtype=object),
            feat_values   = np.array(feat_values,  dtype=object),
            n_active      = n_active,
            recon_error   = recon_error,
            input_norm     = input_norm,
            recon_error_rel = recon_error_rel,
            row_uid        = rt_df["row_uid"].astype(str).to_numpy(),
            hidden_row_idx = rt_df["hidden_row_idx"].astype(int).to_numpy(),
        )

        # ── Save dense top-K summary ───────────────────────────────────────
        stats_out = OUT_DIR / f"{corpus}_sae_L{layer:02d}_stats.npz"
        np.savez_compressed(
            stats_out,
            dense_topk     = dense_topk,       # (n_words, TOP_K)
            top_k_ids      = top_k_ids,         # (TOP_K,)
            n_active       = n_active,
            recon_error    = recon_error,
            input_norm      = input_norm,
            recon_error_rel = recon_error_rel,
            row_uid         = rt_df["row_uid"].astype(str).to_numpy(),
            hidden_row_idx  = rt_df["hidden_row_idx"].astype(int).to_numpy(),
            hidden_input_kind = np.array("final_subtoken_resid_pre"),
            hidden_context_size = rt_df["hidden_context_size"].to_numpy(),
            hidden_context_stride = rt_df["hidden_context_stride"].to_numpy(),
        )

        # ── Accumulate metadata ────────────────────────────────────────────
        for word_idx in range(n_words):
            metadata_rows.append({
                "layer":        layer,
                "word_idx":     word_idx,
                "row_uid":      rt_df.iloc[word_idx]["row_uid"],
                "hidden_row_idx": int(rt_df.iloc[word_idx]["hidden_row_idx"]),
                "hidden_context_size": rt_df.iloc[word_idx]["hidden_context_size"],
                "hidden_context_stride": rt_df.iloc[word_idx]["hidden_context_stride"],
                "n_active":     int(n_active[word_idx]),
                "recon_error":  float(recon_error[word_idx]),
                "input_norm":   float(input_norm[word_idx]),
                "recon_error_rel": float(recon_error_rel[word_idx]),
            })

        # ── Layer summary ──────────────────────────────────────────────────
        tqdm.write(
            f"    Layer {layer:2d}: "
            f"mean active features = {n_active.mean():.1f}  "
            f"mean rel recon error = {recon_error_rel.mean():.4f}  "
            f"saved → {sparse_out.name}"
        )

        # Free memory before loading next SAE
        del sae, hidden, feat_indices, feat_values, dense_topk
        torch.cuda.empty_cache() if device == "cuda" else None

    # ── Save metadata CSV ──────────────────────────────────────────────────
    meta_df = pd.DataFrame(metadata_rows)

    # Merge with RT data for convenience in script 05
    # (word_idx aligns with row order in residualized_RT.csv)
    rt_df_indexed = rt_df.reset_index(drop=True)
    rt_df_indexed["word_idx"] = rt_df_indexed.index

    meta_merged = meta_df.merge(
        rt_df_indexed[["word_idx", "text_id", "word_position", "word",
                        "surprisal", "RT_prime", "is_sentence_final"]],
        on="word_idx",
        how="left",
    )

    meta_out = OUT_DIR / f"{corpus}_sae_metadata.csv"
    meta_merged.to_csv(meta_out, index=False)
    print(f"\n  Metadata saved → {meta_out}")

    # ── Print overall summary ──────────────────────────────────────────────
    print_sae_summary(corpus, meta_df, layers)

    return meta_df


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

def print_sae_summary(corpus: str, meta_df: pd.DataFrame, layers: list[int]):
    print(f"\n{'═'*60}")
    print(f"  SAE EXTRACTION SUMMARY — {corpus.upper()}")
    print(f"{'═'*60}")
    print(f"  {'Layer':>6}  {'Mean L0':>10}  {'Median L0':>10}  {'Mean RelErr':>12}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*12}")
    for layer in layers:
        sub = meta_df[meta_df["layer"] == layer]
        if sub.empty:
            continue
        print(
            f"  {layer:>6}  "
            f"{sub['n_active'].mean():>10.1f}  "
            f"{sub['n_active'].median():>10.1f}  "
            f"{sub['recon_error_rel'].mean():>12.4f}"
        )
    print(f"{'═'*60}")
    print()
    print("  Interpretation:")
    print("  L0 and relative reconstruction error should be compared across layers/corpora.")
    print("  Large relative errors or many zero-norm inputs indicate hook/alignment problems.")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SparseRT — Step 04: SAE Feature Extraction"
    )
    parser.add_argument(
        "--corpus",
        choices=["provo", "natural_stories", "both"],
        default="both",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=ALL_LAYERS,
        help="Which layers to extract (default: all 12). "
             "Example: --layers 8 9 10 11 12"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help="Compute device. Auto-detected if not set. "
             "M3 Mac: mps. GPU workstation: cuda."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Words per SAE forward pass (default 512). "
             "Reduce if you hit memory errors."
    )
    args = parser.parse_args()

    # ── Device selection ───────────────────────────────────────────────────
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

    layers = sorted(set(args.layers))
    print(f"Layers to process: {layers}")
    print(f"SAE release: {SAE_RELEASE}  |  Features per layer: {N_FEATURES:,}")

    corpora = (
        ["provo", "natural_stories"] if args.corpus == "both"
        else [args.corpus]
    )

    for corpus in corpora:
        try:
            extract_corpus(corpus, layers, device, batch_size=args.batch_size)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {e}\n")

    print("SAE extraction complete.")
    print("Next step: python src/05_sae_regression.py")


if __name__ == "__main__":
    main()
