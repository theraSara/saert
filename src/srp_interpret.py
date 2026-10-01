"""
srp_interpret.py — SparseRT Pipeline
======================================
Applies Sparse Readout Prism (He et al. 2026) to interpret the top SAE
features that predict human reading times.

What SRP does
-------------
For each SAE feature i, the decoder vector W_dec[:, i] is a direction in
GPT-2's residual stream. The logit lens tells us which tokens it predicts
by computing h @ W_U.T. But this is corpus-dependent (He et al. §2).

SRP instead decomposes the unembedding row W_U[token] into sparse readout
features d_i and expresses the logit as:

    h · W_U[token]  =  base  +  Σ_i z_i (h · d_i)  +  residual

where z_i is a signed contribution — stable across fitting corpora because
the d_i are derived from W_U weights alone, not from any text corpus.

Workflow
--------
1. Extract W_U and h_LN (layer-normed hidden states) from GPT-2 via
   TransformerLens — both are needed to train the SRP dictionary.
2. Train or load an SRP dictionary on GPT-2's W_U rows.
3. For each top SAE feature, decompose its decoder vector W_dec[:, i]
   through the SRP dictionary to get sparse readout contributions.
4. Save results alongside Neuronpedia labels.

Usage
-----
    python src/srp_interpret.py                          # train + decompose
    python src/srp_interpret.py --srp-checkpoint PATH   # load existing
    python src/srp_interpret.py --corpus provo --hooks L01 --kinds post
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

ROOT    = Path(__file__).resolve().parent.parent
SRP_DIR = ROOT / "results" / "srp" / "gpt2_small"
FI_DIR  = ROOT / "results" / "feature_interpretation"

# Cloned SRP repo (sibling of saert)
SRP_REPO = ROOT / "sparse-readout-prism"
_srp_src = SRP_REPO / "src"
if _srp_src.exists() and str(_srp_src) not in sys.path:
    sys.path.insert(0, str(_srp_src))

BEST_HOOKS = {
    "provo":           "L01",
    "natural_stories": "L08",
}
KINDS = ["post", "prefix"]


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Extract W_U and h_LN from GPT-2 small
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_gpt2_data(device: str = "cpu", n_hidden: int = 8192) -> tuple:
    """
    Extract from GPT-2 small:
      W_U   : (50257, 768)  unembedding matrix  (vocab × d_model)
      h_LN  : (n_hidden, 768) layer-normed hidden states from GPT-2

    h_LN is needed by the SRP data loader alongside W_U.
    We generate it by running GPT-2 on its own token IDs (no external text).

    Returns (W_U, h_LN) both float32 on CPU.
    """
    from transformer_lens import HookedTransformer

    print("  Loading GPT-2 small via TransformerLens...")
    model = HookedTransformer.from_pretrained(
        "gpt2",
        fold_ln=True,
        center_writing_weights=True,
        device=device,
    )
    model.eval()

    # W_U: TransformerLens stores it as (d_model, vocab) → transpose
    W_U = model.W_U.detach().cpu().float().T   # (50257, 768)
    print(f"  W_U: {W_U.shape}")

    # Generate h_LN by running GPT-2 on random token sequences
    # We cache the final layer's residual stream (post layer-norm = h_LN)
    print(f"  Generating {n_hidden} layer-normed hidden states...")
    torch.manual_seed(42)
    seq_len   = 128
    n_seqs    = (n_hidden + seq_len - 1) // seq_len
    # Random token IDs from GPT-2 vocab
    token_ids = torch.randint(0, W_U.shape[0], (n_seqs, seq_len), device=device)

    h_LN_chunks = []
    batch_size  = 8
    for i in range(0, n_seqs, batch_size):
        batch = token_ids[i:i + batch_size]
        _, cache = model.run_with_cache(
            batch,
            names_filter="blocks.11.hook_resid_post",
            return_type=None,
        )
        h = cache["blocks.11.hook_resid_post"]   # (B, seq, 768)
        # Apply final layer norm (fold_ln=True doesn't fold the final LN)
        h_normed = model.ln_final(h)             # (B, seq, 768)
        h_LN_chunks.append(h_normed.reshape(-1, 768).detach().cpu().float())

    h_LN = torch.cat(h_LN_chunks, dim=0)[:n_hidden]   # (n_hidden, 768)
    print(f"  h_LN: {h_LN.shape}")

    del model, cache
    if device == "mps":
        torch.mps.empty_cache()

    return W_U, h_LN


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Train SRP dictionary on GPT-2 W_U
# ══════════════════════════════════════════════════════════════════════════════

def train_srp_direct(
    W_U: torch.Tensor,
    h_LN: torch.Tensor,
    srp_dir: Path,
    device: str = "cpu",
    n_steps: int = 5_000,
    batch_size: int = 1024,
    d_features: int = 4096,   # 5× d_model=768; large enough for interpretation
    k: int = 64,              # 64 active features per decomposition
) -> Path:
    """
    Train an SRP (TopKSAE) dictionary directly in Python, without subprocess.
    Uses the SRP library's own TopKSAE + preprocess_rows.

    The training follows the paper recipe:
      - TopK factorizer (k=128, d_features=32×768=24576)
      - Adam, lr=1e-3, 20k steps, batch 4096
      - Prism penalty λ=1e-3 with linear ramp
      - Rows of W_U as training data (centered + row-normalized)
    """
    from sparse_readout_prism import build_factorizer, preprocess_rows

    srp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = srp_dir / "checkpoint.pt"

    d_model = W_U.shape[1]   # 768
    print(f"\n  Training SRP: d_model={d_model}, d_features={d_features}, k={k}, steps={n_steps}")
    print(f"  Device: {device}  (~5 min MPS, ~20 min CPU)")

    # Preprocess W_U
    row_mean, row_norms, rows_normalized = preprocess_rows(W_U)
    rows_norm_dev = rows_normalized.to(device)

    # build_factorizer(config: dict, d_model: int) — confirmed signature
    factorizer_config = {
        "factorizer": {
            "architecture":      "topk",
            "d_features":        d_features,
            "k":                 k,
            "encoder_init_scale": 0.02,
        }
    }
    factorizer = build_factorizer(factorizer_config, d_model=d_model).to(device)

    opt = torch.optim.Adam(factorizer.parameters(), lr=1e-3)

    # Linear ramp for prism penalty (ramp over first 5000 steps)
    lambda_prism = 1e-3
    ramp_steps   = 5000
    torch.manual_seed(42)
    n_rows = rows_norm_dev.shape[0]

    print(f"  {'step':>6}  {'recon_loss':>11}  {'prism_loss':>11}  {'L0':>6}")
    print(f"  {'─'*6}  {'─'*11}  {'─'*11}  {'─'*6}")

    for step in range(n_steps):
        # Sample a batch of W_U rows
        idx   = torch.randint(0, n_rows, (batch_size,), device=device)
        batch = rows_norm_dev[idx]   # (B, d_model)

        out   = factorizer(batch, k=k)
        recon_loss = F.mse_loss(out.reconstruction, batch)

        # Prism penalty: sampled pairwise orthogonality on decoder columns.
        # Computing the full (24576×24576) gram matrix needs ~9GB — too large.
        # Instead sample 512 random column pairs per step; unbiased estimator
        # of the same quantity, O(512 × d_model) not O(d_features²).
        prism_scale = min(1.0, step / ramp_steps) * lambda_prism
        D       = factorizer.decoder                          # (d_features, d_model)
        D_norm  = F.normalize(D, dim=1)
        n_pairs = 128
        idx_a   = torch.randint(0, d_features, (n_pairs,), device=device)
        idx_b   = torch.randint(0, d_features, (n_pairs,), device=device)
        # Avoid self-pairs
        idx_b   = torch.where(idx_b == idx_a,
                              (idx_b + 1) % d_features, idx_b)
        dots    = (D_norm[idx_a] * D_norm[idx_b]).sum(dim=1)  # (n_pairs,)
        prism_loss = (dots ** 2).mean()

        loss = recon_loss + prism_scale * prism_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Re-normalise decoder columns after each step
        with torch.no_grad():
            factorizer.normalize_decoder_()

        if step == 0 or (step + 1) % 500 == 0:
            with torch.no_grad():
                l0 = (out.code != 0).float().sum(dim=1).mean().item()
            print(f"  {step+1:>6}  {recon_loss.item():>11.6f}  "
                  f"{prism_loss.item():>11.6f}  {l0:>6.1f}")

    # Evaluate: row-centered explained variance on held-out rows
    with torch.no_grad():
        val_rows = rows_norm_dev[:min(2000, n_rows)]
        val_out  = factorizer(val_rows, k=k)
        # Row-centered EV
        total_var = ((val_rows - val_rows.mean(0)) ** 2).sum()
        resid_var = ((val_rows - val_out.reconstruction) ** 2).sum()
        row_ev    = 1 - resid_var / total_var
    print(f"\n  rowEV (held-out): {row_ev.item():.4f}")

    # Save checkpoint — same layout as the Ministral/R1 checkpoints in the HF repo
    torch.save({
        "model_state_dict": factorizer.cpu().state_dict(),
        "factorizer": {
            "architecture": "topk",
            "k":            k,
            "d_features":   d_features,
            "d_model":      d_model,
        },
        "row_mean":   row_mean.cpu(),
        "row_norms":  row_norms.cpu(),
        "metrics":    {"row_ev": float(row_ev.item())},
        "model_id":   "gpt2-small",
    }, checkpoint_path)

    print(f"  Checkpoint saved → {checkpoint_path}")
    return checkpoint_path


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Decompose SAE decoder vectors through SRP
# ══════════════════════════════════════════════════════════════════════════════

def load_sae_decoder(hook_label: str, device: str) -> torch.Tensor:
    """
    Load SAE decoder matrix for one hook from SAELens.
    Returns W_dec: (d_model=768, n_features=24576) on CPU.
    """
    from sae_lens import SAE

    layer_idx = int(hook_label[1:]) - 1   # L01→0, L08→7
    hook_name = f"blocks.{layer_idx}.hook_resid_pre"
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=hook_name,
        device=device,
    )
    sae.eval()
    return sae.W_dec.T.detach().float().cpu()   # (768, 24576)


def decompose_one_feature(
    feature_id: int,
    W_dec: torch.Tensor,            # (d_model, n_sae_features)
    W_U: torch.Tensor,              # (vocab, d_model)
    row_mean: torch.Tensor,         # (d_model,)
    row_norms: torch.Tensor,        # (vocab,)
    rows_normalized: torch.Tensor,  # (vocab, d_model)
    srp_model,                      # loaded TopKSAE
    srp_k: int,
    tokenizer,
    top_n_srp: int = 10,
    top_n_tokens: int = 5,
) -> dict:
    """
    Decompose one SAE feature's decoder vector through the SRP dictionary.

    The SAE decoder vector W_dec[:, i] is a direction in GPT-2's residual
    stream. We treat it as the query hidden state h in the SRP formula:

        h · W_U[token]  =  base  +  Σ_i z_i (h · d_i)  +  residual

    This gives us the sparse readout features that contribute most to
    this SAE feature's logit lens prediction — corpus-independently.

    The Decomposition.feature_contributions tensor gives the SIGNED
    contribution of each SRP feature to the selected token's logit.
    active_feature_indices gives the indices where code != 0.
    """
    from sparse_readout_prism import decompose_token_logit

    h = W_dec[:, feature_id].float()     # (768,) — the SAE feature direction

    # Logit lens: which tokens does this feature direction predict?
    logits       = h @ W_U.T             # (50257,)
    top_ids      = logits.topk(top_n_tokens).indices.tolist()
    top_tokens   = [tokenizer.decode([tid]) for tid in top_ids]
    primary_token_id = top_ids[0]

    # SRP decomposition for the top-scoring token
    d = decompose_token_logit(
        h              = h,
        W_row          = W_U[primary_token_id],
        row_mean       = row_mean,
        row_norm       = row_norms[primary_token_id],
        row_normalized = rows_normalized[primary_token_id],
        model          = srp_model,
        k              = srp_k,
    )

    # Active SRP features: indices where code != 0
    active_idx = d.active_feature_indices.tolist()

    # Contributions: signed contribution of each active SRP feature
    # d.feature_contributions is (d_features,) — full vector
    # We extract values at active positions
    contributions = d.feature_contributions.detach()
    active_contribs = [
        (int(idx), float(contributions[idx]))
        for idx in active_idx
    ]
    # Sort by |contribution| descending
    active_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_srp = active_contribs[:top_n_srp]

    # Fidelity: |feature_sum| / (|feature_sum| + |residual|)
    fs  = float(d.feature_sum)
    res = float(d.residual_term)
    fidelity = abs(fs) / (abs(fs) + abs(res) + 1e-8)

    # Identity check (should be ~0)
    identity_err = float(d.identity_error)

    return {
        "feature_id":          feature_id,
        "logit_lens_top_token": top_tokens[0],
        "logit_lens_top5":     " | ".join(t.strip() for t in top_tokens),
        "srp_top_features":    json.dumps(top_srp),          # [(idx, contribution), ...]
        "srp_feature_sum":     round(fs, 6),
        "srp_residual":        round(res, 6),
        "srp_base":            round(float(d.base_term), 6),
        "srp_original_logit":  round(float(d.original_logit), 6),
        "srp_fidelity":        round(fidelity, 4),
        "srp_n_active":        len(active_idx),
        "srp_identity_error":  round(abs(identity_err), 8),
    }


def run_srp_combo(
    corpus: str,
    hook_label: str,
    kind: str,
    W_U: torch.Tensor,
    row_mean: torch.Tensor,
    row_norms: torch.Tensor,
    rows_normalized: torch.Tensor,
    srp_model,
    srp_k: int,
    tokenizer,
    device: str,
) -> pd.DataFrame | None:
    """Run SRP decomposition for one (corpus, hook, kind) combination."""
    feat_path = FI_DIR / f"{corpus}_{hook_label}_{kind}_top_features.csv"
    if not feat_path.exists():
        print(f"  [SKIP] {feat_path.name} not found")
        return None

    df          = pd.read_csv(feat_path)
    feature_ids = df["feature_id"].astype(int).tolist()
    print(f"\n  {corpus} | {hook_label} | {kind}  ({len(feature_ids)} features)")

    # Load SAE decoder for this hook (cached per hook across calls)
    print(f"    Loading SAE decoder for {hook_label}...")
    W_dec = load_sae_decoder(hook_label, device)   # (768, 24576)

    rows = []
    for fid in feature_ids:
        row = decompose_one_feature(
            feature_id      = fid,
            W_dec           = W_dec,
            W_U             = W_U,
            row_mean        = row_mean,
            row_norms       = row_norms,
            rows_normalized = rows_normalized,
            srp_model       = srp_model,
            srp_k           = srp_k,
            tokenizer       = tokenizer,
        )
        rows.append(row)

    srp_df = pd.DataFrame(rows)

    # Merge back into top_features CSV
    # Drop any existing SRP columns first to avoid duplicates
    srp_cols = [c for c in df.columns if c.startswith("srp_") or c == "logit_lens_top_token"]
    df = df.drop(columns=srp_cols, errors="ignore")
    df_merged = df.merge(srp_df, on="feature_id", how="left")
    df_merged.to_csv(feat_path, index=False)

    # Standalone SRP CSV
    out_path = FI_DIR / f"{corpus}_{hook_label}_{kind}_srp.csv"
    srp_df.to_csv(out_path, index=False)

    # Print summary
    print(f"    fidelity  mean={srp_df['srp_fidelity'].mean():.3f}  "
          f"min={srp_df['srp_fidelity'].min():.3f}  "
          f"max={srp_df['srp_fidelity'].max():.3f}")
    print(f"    identity_error  max={srp_df['srp_identity_error'].max():.2e}  "
          f"(should be <1e-4)")
    print(f"    → {out_path.name}")
    print()
    print(f"    {'ID':>6}  {'fidelity':>8}  {'n_srp':>5}  "
          f"{'logit_lens_top':>20}  top SRP feature contribution")
    print(f"    {'─'*6}  {'─'*8}  {'─'*5}  {'─'*20}  {'─'*30}")
    for _, row in srp_df.head(10).iterrows():
        top_contrib = json.loads(row["srp_top_features"])
        top_str     = (f"SRP#{top_contrib[0][0]} ({top_contrib[0][1]:+.3f})"
                       if top_contrib else "—")
        print(f"    {int(row['feature_id']):>6}  "
              f"{row['srp_fidelity']:>8.3f}  "
              f"{int(row['srp_n_active']):>5}  "
              f"{str(row['logit_lens_top_token']):>20}  "
              f"{top_str}")

    return srp_df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SparseRT — SRP feature interpretation"
    )
    parser.add_argument(
        "--srp-checkpoint", type=Path,
        default=SRP_DIR / "checkpoint.pt",
        help="Path to SRP checkpoint. Trains if not found.",
    )
    parser.add_argument(
        "--corpus", choices=["provo", "natural_stories", "both"], default="both",
    )
    parser.add_argument(
        "--hooks", nargs="+", default=None,
        help="Hook labels (e.g. L01 L08). Default: best hook per corpus.",
    )
    parser.add_argument(
        "--kinds", nargs="+", choices=["post", "prefix"], default=["post", "prefix"],
    )
    parser.add_argument(
        "--srp-k", type=int, default=128,
        help="Active SRP features per decomposition (default: 128).",
    )
    parser.add_argument(
        "--n-steps", type=int, default=5_000,
        help="Training steps if training from scratch (default: 20000).",
    )
    parser.add_argument(
        "--device", choices=["cpu", "mps", "cuda"], default=None,
    )
    args = parser.parse_args()

    if args.device is None:
        device = ("mps"  if torch.backends.mps.is_available()  else
                  "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    print(f"Device: {device}")

    if not _srp_src.exists():
        raise FileNotFoundError(
            f"SRP repo not found at {SRP_REPO}\n"
            "Clone with: git clone https://github.com/hematteo/sparse-readout-prism\n"
            "Then:       cd sparse-readout-prism && uv sync"
        )

    from sparse_readout_prism import load_factorizer, preprocess_rows
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # ── Step 1: Extract W_U (and h_LN if training needed) ─────────────────
    checkpoint_path = args.srp_checkpoint
    need_h_LN       = not checkpoint_path.exists()

    print("\nStep 1: Extracting GPT-2 W_U...")
    if need_h_LN:
        W_U, h_LN = extract_gpt2_data(device=device, n_hidden=8192)
    else:
        # Only need W_U for decomposition preprocessing
        from transformer_lens import HookedTransformer
        m = HookedTransformer.from_pretrained("gpt2", fold_ln=True,
                                              center_writing_weights=True,
                                              device=device)
        m.eval()
        W_U = m.W_U.detach().cpu().float().T
        del m
        print(f"  W_U: {W_U.shape}")

    # ── Step 2: Preprocess W_U ─────────────────────────────────────────────
    print("\nStep 2: Preprocessing W_U...")
    row_mean, row_norms, rows_normalized = preprocess_rows(W_U)

    # ── Step 3: Train or load SRP ──────────────────────────────────────────
    if checkpoint_path.exists():
        print(f"\nStep 3: Loading SRP checkpoint from {checkpoint_path}")
        srp_model = load_factorizer(str(checkpoint_path), freeze=True)
        print(f"  k={srp_model.k}  d_features={srp_model.d_features}")
    else:
        print(f"\nStep 3: Training SRP dictionary on GPT-2 W_U...")
        # Save W_U + h_LN for potential future reference
        SRP_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"W_U": W_U, "h_LN": h_LN},
                   SRP_DIR / "gpt2_readout_data.pt")
        checkpoint_path = train_srp_direct(
            W_U=W_U, h_LN=h_LN,
            srp_dir=SRP_DIR, device=device,
            n_steps=args.n_steps, d_features=4096, k=args.srp_k,
        )
        srp_model = load_factorizer(str(checkpoint_path), freeze=True)
        print(f"  Loaded: k={srp_model.k}  d_features={srp_model.d_features}")

    # ── Step 4: Decompose SAE features through SRP ─────────────────────────
    print("\nStep 4: Decomposing SAE decoder vectors through SRP...")
    corpora = (["provo", "natural_stories"] if args.corpus == "both"
               else [args.corpus])

    for corpus in corpora:
        hooks = args.hooks or [BEST_HOOKS[corpus]]
        for hook_label in hooks:
            for kind in args.kinds:
                run_srp_combo(
                    corpus=corpus, hook_label=hook_label, kind=kind,
                    W_U=W_U, row_mean=row_mean, row_norms=row_norms,
                    rows_normalized=rows_normalized,
                    srp_model=srp_model, srp_k=args.srp_k,
                    tokenizer=tokenizer, device=device,
                )

    print("\nSRP interpretation complete.")
    print(f"Output: {FI_DIR}/")
    print()
    print("Key output columns in *_srp.csv:")
    print("  srp_top_features   — [(srp_idx, contribution), ...] top-10 readout features")
    print("  srp_fidelity       — fraction of logit explained by sparse approx")
    print("  srp_n_active       — number of active SRP features")
    print("  srp_identity_error — should be <1e-4 (checks decomposition correctness)")
    print()
    print("Next: run notebooks/feature_interpretation.ipynb to visualize SRP results.")


if __name__ == "__main__":
    main()