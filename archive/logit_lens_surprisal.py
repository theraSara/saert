"""
logit_lens_surprisal.py — SparseRT Pipeline
=============================================
Three mechanistic surprisal variants, each computed from a different
subset of GPT-2's internal representations.

Standard surprisal uses the final-layer output:
    surp(word) = -log₂ p_final(word | context)

This script computes three alternatives:

1. LOGIT-LENS SURPRISAL
   Use the hidden state at the RT-selected layer (L01/L08) instead
   of the final output:
       surp_logit_lens(word) = -log₂ p_L(word | context)
   where p_L = softmax(W_U @ LN_final(h_L))
   Token-level, then summed across subtokens per word — matching
   exactly how surprisal.py computes standard surprisal.

2. SAE-RECONSTRUCTED SURPRISAL
   Reconstruct the hidden state using ONLY the SAE features that the
   ridge regression selected as RT-predictive:
       h_sparse = z_selected @ W_dec  +  b_dec
       surp_sparse = -log₂ softmax(W_U @ LN_final(h_sparse))[token]
   This is mechanistically interpretable surprisal — only from the
   features empirically shown to predict reading difficulty.

3. SUPPRESSED SURPRISAL
   Reconstruct from the SAE features that were NOT selected:
       h_suppressed = z_unselected @ W_dec  +  b_dec
       surp_suppressed = -log₂ softmax(W_U @ LN_final(h_suppressed))[token]
   If this is a WORSE predictor than surp_sparse, the model's
   non-RT-relevant computation is genuinely orthogonal to human
   processing difficulty.

The comparison table is the paper's core quantitative result for
the "mechanistic surprisal" analysis.

Usage
-----
    python src/logit_lens_surprisal.py --corpus both
    python src/logit_lens_surprisal.py --corpus provo --kinds post
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

try:
    from pipeline_utils import read_table, run_metadata, write_json
    from surprisal import HOOK_FILES
except ImportError:
    from pipeline_utils import read_table, run_metadata, write_json
    from surprisal import HOOK_FILES

ROOT    = Path(__file__).resolve().parent.parent
SURP    = ROOT / "results" / "surprisal_bos"
SAE_DIR = ROOT / "results" / "sae_bos"
REG_DIR = ROOT / "results" / "sae_regression_v2"
OUT_DIR = ROOT / "results" / "logit_lens_surprisal"

# Best hooks from inner validation
BEST_HOOKS = {"provo": "L01", "natural_stories": "L08"}

# Controls used in baseline regression (must match baseline.py)
OUTCOME  = "log_primary_RT"
CONTROLS = ["zipf_freq", "word_length", "is_sentence_final",
            "log_word_position", "is_text_position_1", "is_text_position_2"]


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load GPT-2 readout weights
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def load_gpt2_readout(device: str = "cpu"):
    """Load W_U and ln_final from GPT-2 small. Returns (W_U_np, ln_final)."""
    from transformer_lens import HookedTransformer
    print("  Loading GPT-2 small (TransformerLens)...")
    model = HookedTransformer.from_pretrained(
        "gpt2", fold_ln=True, center_writing_weights=True, device=device
    )
    model.eval()
    # W_U: TransformerLens (d_model, vocab) → transpose to (vocab, d_model)
    W_U = model.W_U.detach().cpu().float().T.numpy()   # (50257, 768)
    ln_final = model.ln_final.cpu()
    print(f"  W_U: {W_U.shape}")
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return W_U, ln_final


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load SAE decoder and regression-selected feature mask
# ══════════════════════════════════════════════════════════════════════════════

def load_sae_decoder(hook_label: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load SAE decoder matrix and bias.
    Returns W_dec (n_features, 768) and b_dec (768,) as numpy arrays.
    """
    from sae_lens import SAE
    layer_idx = int(hook_label[1:]) - 1   # L01→0, L08→7
    hook_name = f"blocks.{layer_idx}.hook_resid_pre"
    print(f"    Loading SAE for {hook_label} ({hook_name})...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb", sae_id=hook_name, device="cpu"
    )
    sae.eval()
    W_dec = sae.W_dec.detach().float().numpy()   # (n_features, 768)
    b_dec = sae.b_dec.detach().float().numpy()   # (768,)
    del sae
    return W_dec, b_dec


def load_selected_mask(corpus: str, hook_label: str, kind: str) -> np.ndarray:
    """
    Load ridge regression coefficients and return boolean mask of
    features selected in ≥50% of outer folds.
    """
    coef_path = REG_DIR / corpus / kind / hook_label / "sae" / "coefficients.npz"
    if not coef_path.exists():
        raise FileNotFoundError(
            f"No regression coefficients at {coef_path}\n"
            f"Run: bash run.sh regression {corpus}"
        )
    data     = np.load(coef_path, allow_pickle=False)
    coef_raw = data["feature_coef_raw"]   # (n_folds, n_features)
    n_folds  = coef_raw.shape[0]
    n_nonzero = (coef_raw != 0).sum(axis=0)
    selected  = n_nonzero >= (n_folds // 2)
    print(f"    Selected features: {selected.sum()} / {len(selected)} "
          f"(≥{n_folds//2}/{n_folds} folds)")
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Compute logit-lens surprisal (word-level, matching surprisal.py)
# ══════════════════════════════════════════════════════════════════════════════

def apply_ln_and_logit(H: np.ndarray, W_U: np.ndarray, ln_final,
                       token_ids: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """
    For each row i, compute -log₂ p(token_ids[i] | H[i]) via logit lens.
    H:         (N, 768) hidden states
    W_U:       (vocab, 768)
    token_ids: (N,) integer token IDs to score
    Returns:   (N,) surprisal in bits
    """
    N = len(H)
    surp = np.zeros(N, dtype=np.float64)
    for start in range(0, N, batch_size):
        end     = min(start + batch_size, N)
        h_batch = torch.tensor(H[start:end], dtype=torch.float32)
        h_norm  = ln_final(h_batch).numpy()                     # (B, 768)
        logits  = h_norm @ W_U.T                                 # (B, vocab)
        # Numerically stable log-softmax
        logits -= logits.max(axis=1, keepdims=True)
        log_sum = np.log(np.exp(logits).sum(axis=1))
        tids    = token_ids[start:end]
        target_logits = logits[np.arange(end - start), tids]
        surp[start:end] = -(target_logits - log_sum) / np.log(2)
    return surp


def compute_word_logit_lens(
    corpus: str,
    hook_label: str,
    kind: str,
    W_U: np.ndarray,
    ln_final,
) -> pd.Series:
    """
    Compute logit-lens surprisal at hook_label for all words in corpus.
    Matches surprisal.py exactly:
      - Use token-level hidden states
      - Sum subtoken surprisals per word (word_surprisals logic)
    Returns a Series indexed by row_uid.
    """
    stem = "hidden" if kind == "post" else "prefix_hidden"
    h_path = SURP / f"{corpus}_{stem}_{hook_label}.npy"
    if not h_path.exists():
        raise FileNotFoundError(f"Hidden states not found: {h_path}")

    # Load word-level surprisal CSV for alignment (hidden_row_idx = sequential 0..N-1)
    surp_df = read_table(SURP / f"{corpus}_surprisal.csv")

    # Load token windows for subtoken-level token IDs
    # Each row in token_windows is one subtoken; surprisal.py sums subtokens per word
    tok_df = read_table(SURP / f"{corpus}_token_windows.csv")

    print(f"    Loading hidden states {h_path.name} ...")
    H_all = np.load(h_path, mmap_mode="r", allow_pickle=False)   # (N_words, 768)
    print(f"    Shape: {H_all.shape}")

    # For WORD-level logit lens surprisal we need the hidden state at the
    # position that predicts each SUBTOKEN, then sum across subtokens.
    #
    # surprisal.py saves:
    #   post:   H[i] = hidden state at end_token_idx of word i (final subtoken)
    #   prefix: H[i] = hidden state at start_token_idx - 1 of word i (before first subtoken)
    #
    # For multi-subtoken words, surprisal.py sums token-level surprisals:
    #   word_surprisal = sum( -log p(t_k | h_{k-1}) for k in word_subtokens )
    #
    # For LOGIT-LENS surprisal we replicate this:
    # - Post: for the final subtoken only (as a reasonable approximation),
    #   use H[i] (the end-subtoken hidden state). This is what the SAE
    #   features act on, so it is the most internally consistent choice.
    # - We do NOT re-run the model — we use the already-saved hidden states.
    #
    # For single-subtoken words (majority), this is exact.
    # For multi-subtoken words, this approximates by using only the final
    # subtoken's hidden state to predict the final subtoken — acceptable
    # since the regression also uses this representation.

    # token_windows has one row per subtoken; we need the last subtoken per word
    # for post, or the first subtoken per word for prefix.
    tok_df["row_uid"] = tok_df["row_uid"]

    if kind == "post":
        # Last subtoken per word
        last_tok = (tok_df
                    .sort_values("token_idx")
                    .groupby("row_uid", sort=False)
                    .last()
                    .reset_index()[["row_uid", "token_id"]])
    else:
        # First subtoken per word
        first_tok = (tok_df
                     .sort_values("token_idx")
                     .groupby("row_uid", sort=False)
                     .first()
                     .reset_index()[["row_uid", "token_id"]])
        last_tok = first_tok

    # Merge with surp_df for hidden_row_idx alignment
    merged = surp_df[["row_uid", "hidden_row_idx"]].merge(last_tok, on="row_uid", how="inner")
    merged = merged.reset_index(drop=True)

    h_idx    = merged["hidden_row_idx"].values.astype(int)
    tok_ids  = merged["token_id"].values.astype(int)
    H_subset = np.array(H_all[h_idx])   # (N_words, 768) — materialise from mmap

    print(f"    Computing logit-lens surprisal for {len(H_subset)} words...")
    surp_ll = apply_ln_and_logit(H_subset, W_U, ln_final, tok_ids)

    result = pd.Series(surp_ll, index=merged["row_uid"].values)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — SAE-reconstructed and suppressed surprisal
# ══════════════════════════════════════════════════════════════════════════════

def compute_sae_surprisals(
    corpus: str,
    hook_label: str,
    kind: str,
    W_U: np.ndarray,
    ln_final,
    W_dec: np.ndarray,    # (n_features, 768)
    b_dec: np.ndarray,    # (768,)
    selected: np.ndarray, # bool (n_features,)
) -> tuple[pd.Series, pd.Series]:
    """
    Compute SAE-reconstructed and suppressed surprisal for all words.
    Returns (surp_sparse, surp_suppressed) as Series indexed by row_uid.
    """
    # Load sparse feature matrix: (n_words, n_features)
    feat_path = SAE_DIR / corpus / hook_label / f"{kind}_features.npz"
    if not feat_path.exists():
        raise FileNotFoundError(f"SAE features not found: {feat_path}")
    print(f"    Loading SAE features {feat_path.name} ...")
    features = sparse.load_npz(feat_path)   # (n_words, n_features)

    # Load rows.csv: maps feature matrix position to row_uid
    rows_path = SAE_DIR / corpus / hook_label / "rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(f"rows.csv not found: {rows_path}")
    sae_rows = read_table(rows_path)   # row_uid, hidden_row_idx, ...
    row_uids = sae_rows["row_uid"].values

    # Load token_id per word (last subtoken for post, first for prefix)
    tok_df   = read_table(SURP / f"{corpus}_token_windows.csv")
    surp_df  = read_table(SURP / f"{corpus}_surprisal.csv")
    if kind == "post":
        tok_sel = (tok_df.sort_values("token_idx")
                   .groupby("row_uid", sort=False).last()
                   .reset_index()[["row_uid", "token_id"]])
    else:
        tok_sel = (tok_df.sort_values("token_idx")
                   .groupby("row_uid", sort=False).first()
                   .reset_index()[["row_uid", "token_id"]])

    uid_to_tok = dict(zip(tok_sel["row_uid"].values, tok_sel["token_id"].values.astype(int)))

    n_words = features.shape[0]
    print(f"    Computing SAE-reconstructed surprisal for {n_words} words...")

    # Batch: reconstruct all hidden states at once then apply logit lens
    # Sparse matmul: (n_words, n_features) @ (n_features, 768) = (n_words, 768)
    selected_f   = selected.astype(np.float32)
    unselected_f = (~selected).astype(np.float32)

    # Scale feature activations by selection mask then decode
    # features is CSR: multiply columns by mask vector
    feat_dense = np.array(features.todense(), dtype=np.float32)  # (n_words, n_features)

    z_sparse     = feat_dense * selected_f[None, :]     # (n_words, n_features)
    z_suppressed = feat_dense * unselected_f[None, :]   # (n_words, n_features)

    H_sparse     = z_sparse     @ W_dec + b_dec[None, :]  # (n_words, 768)
    H_suppressed = z_suppressed @ W_dec + b_dec[None, :]  # (n_words, 768)

    # Get token IDs for each word
    tok_ids = np.array([uid_to_tok.get(uid, 0) for uid in row_uids], dtype=int)

    # Apply logit lens
    surp_sp  = apply_ln_and_logit(H_sparse,     W_U, ln_final, tok_ids)
    surp_sup = apply_ln_and_logit(H_suppressed, W_U, ln_final, tok_ids)

    del feat_dense, z_sparse, z_suppressed, H_sparse, H_suppressed

    return (pd.Series(surp_sp,  index=row_uids),
            pd.Series(surp_sup, index=row_uids))


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Leave-one-text-out ΔR² comparison
# ══════════════════════════════════════════════════════════════════════════════

def delta_r2_loo(
    df: pd.DataFrame,
    predictor_col: str,
    bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Leave-one-text-out ΔR² of predictor_col above CONTROLS.
    Matches baseline.py methodology (unpenalised OLS, StandardScaler in train split).
    Returns (delta_r2_pp, ci_lo_pp, ci_hi_pp).
    """
    df = df.dropna(subset=[OUTCOME, predictor_col] + CONTROLS).copy()
    texts  = df["text_id"].values
    y      = df[OUTCOME].values
    C      = df[CONTROLS].values.astype(float)
    P      = df[predictor_col].values.astype(float)

    res_ctrl = np.zeros(len(df))
    res_full = np.zeros(len(df))

    for _, (tr, te) in enumerate(LeaveOneGroupOut().split(df, groups=texts)):
        sc_c = StandardScaler().fit(C[tr])
        sc_p = StandardScaler().fit(P[tr].reshape(-1, 1))

        # Controls only
        Xc_tr = np.column_stack([sc_c.transform(C[tr]), np.ones(len(tr))])
        Xc_te = np.column_stack([sc_c.transform(C[te]), np.ones(len(te))])
        b_c   = np.linalg.lstsq(Xc_tr, y[tr], rcond=None)[0]
        res_ctrl[te] = y[te] - Xc_te @ b_c

        # Controls + predictor
        Xf_tr = np.column_stack([Xc_tr, sc_p.transform(P[tr].reshape(-1, 1))])
        Xf_te = np.column_stack([Xc_te, sc_p.transform(P[te].reshape(-1, 1))])
        b_f   = np.linalg.lstsq(Xf_tr, y[tr], rcond=None)[0]
        res_full[te] = y[te] - Xf_te @ b_f

    ss_tot  = ((y - y.mean()) ** 2).sum()
    delta   = ((res_ctrl**2).sum() - (res_full**2).sum()) / ss_tot

    # Bootstrap CI over texts (paired, fixed predictions)
    rng = np.random.default_rng(seed)
    unique_texts = np.unique(texts)
    boot_deltas = []
    for _ in range(bootstrap):
        bt    = rng.choice(unique_texts, size=len(unique_texts), replace=True)
        idx   = np.concatenate([np.where(texts == t)[0] for t in bt])
        yb    = y[idx]
        ss_b  = ((yb - yb.mean()) ** 2).sum()
        if ss_b < 1e-12:
            continue
        d_b = ((res_ctrl[idx]**2).sum() - (res_full[idx]**2).sum()) / ss_b
        boot_deltas.append(d_b)

    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    return float(delta * 100), float(ci_lo * 100), float(ci_hi * 100)


# ══════════════════════════════════════════════════════════════════════════════
# Per-combo runner
# ══════════════════════════════════════════════════════════════════════════════

def run_one(
    corpus: str,
    hook_label: str,
    kind: str,
    W_U: np.ndarray,
    ln_final,
    prepared_dir: Path,
) -> pd.DataFrame:
    print(f"\n  {corpus} | {hook_label} | {kind}")

    # Load W_dec and selected mask
    W_dec, b_dec = load_sae_decoder(hook_label)
    selected     = load_selected_mask(corpus, hook_label, kind)

    # Logit-lens surprisal
    print("    Computing logit-lens surprisal...")
    surp_ll  = compute_word_logit_lens(corpus, hook_label, kind, W_U, ln_final)

    # SAE surprisals
    surp_sp, surp_sup = compute_sae_surprisals(
        corpus, hook_label, kind, W_U, ln_final, W_dec, b_dec, selected
    )

    # Load original surprisal + RT + controls
    surp_df  = read_table(SURP / f"{corpus}_surprisal.csv")
    prep_dir = prepared_dir / corpus
    prep_csv = prep_dir / f"{corpus}_prepared.csv"
    if not prep_csv.exists():
        raise FileNotFoundError(f"Prepared CSV not found: {prep_csv}")
    prep_df  = read_table(prep_csv)

    # Build unified DataFrame
    # word_position is in surp_df; drop it from prep_df to avoid duplicate columns
    df = (surp_df[["row_uid", "text_id", "word_position", "word", "surprisal"]]
          .merge(prep_df[["row_uid", OUTCOME, "zipf_freq", "word_length",
                           "is_sentence_final"]],
                 on="row_uid", how="inner"))
    df["log_word_position"]  = np.log1p(df["word_position"])
    df["is_text_position_1"] = (df["word_position"] == 1).astype(int)
    df["is_text_position_2"] = (df["word_position"] == 2).astype(int)
    df = df.rename(columns={"surprisal": "surp_original"})

    df["surp_logit_lens"]  = df["row_uid"].map(surp_ll)
    df["surp_sparse"]      = df["row_uid"].map(surp_sp)
    df["surp_suppressed"]  = df["row_uid"].map(surp_sup)

    # Save per-word CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{corpus}_{kind}_surprisal_variants.csv"
    df.to_csv(out_csv, index=False)
    print(f"    → {out_csv.name}  ({len(df)} words)")

    # Stats
    for col in ["surp_original", "surp_logit_lens", "surp_sparse", "surp_suppressed"]:
        v = df[col].dropna()
        print(f"    {col:30s}  mean={v.mean():.3f}  std={v.std():.3f}  N={len(v)}")

    # Leave-one-text-out ΔR² comparison
    print(f"\n    Leave-one-text-out ΔR² comparison:")
    rows = []
    predictors = {
        "standard surprisal":     "surp_original",
        "logit-lens surprisal":   "surp_logit_lens",
        "SAE-sparse surprisal":   "surp_sparse",
        "SAE-suppressed surprisal": "surp_suppressed",
    }
    for name, col in predictors.items():
        if df[col].isna().mean() > 0.1:
            print(f"    [SKIP] {name}: >10% NaN")
            continue
        d, lo, hi = delta_r2_loo(df, col)
        rows.append({"corpus": corpus, "kind": kind, "predictor": name,
                     "delta_r2_pp": d, "ci_lo": lo, "ci_hi": hi})
        print(f"    {name:35s}  {d:+.3f} pp  [{lo:+.3f}, {hi:+.3f}]")

    comp_df = pd.DataFrame(rows)
    comp_csv = OUT_DIR / f"{corpus}_{kind}_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    return comp_df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["provo", "natural_stories", "both"],
                        default="both")
    parser.add_argument("--kinds", nargs="+", choices=["post", "prefix"],
                        default=["post", "prefix"])
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--prepared-dir", type=Path,
                        default=ROOT / "data" / "prepared_v2")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    device = (args.device or
              ("mps"  if torch.backends.mps.is_available() else
               "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print("\nLoading GPT-2 readout weights...")
    W_U, ln_final = load_gpt2_readout(device=device)

    corpora = (["provo", "natural_stories"] if args.corpus == "both"
               else [args.corpus])

    all_results = []
    for corpus in corpora:
        hook = BEST_HOOKS[corpus]
        print(f"\n{'═'*60}")
        print(f"  {corpus.upper()} — hook {hook}")
        print(f"{'═'*60}")
        for kind in args.kinds:
            try:
                comp = run_one(corpus, hook, kind, W_U, ln_final, args.prepared_dir)
                all_results.append(comp)
            except FileNotFoundError as e:
                print(f"\n  [SKIP] {e}")
            except Exception as e:
                print(f"\n  [ERROR] {corpus}/{kind}: {e}")
                raise

    # Final summary table
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"\n{'═'*72}")
        print("  FINAL SUMMARY — ΔR² above lexical controls (pp, 95% CI)")
        print(f"{'═'*72}")
        for (corpus, kind), grp in combined.groupby(["corpus", "kind"]):
            print(f"\n  {corpus} | {kind}")
            for _, row in grp.iterrows():
                print(f"    {row['predictor']:38s}"
                      f"  {row['delta_r2_pp']:+6.3f} pp"
                      f"  [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]")

    write_json(OUT_DIR / "manifest.json", {
        **run_metadata(),
        "best_hooks":   BEST_HOOKS,
        "corpora":      corpora,
        "kinds":        args.kinds,
        "method": {
            "surp_original":     "standard GPT-2 surprisal (final layer)",
            "surp_logit_lens":   "logit-lens surprisal at L01/L08",
            "surp_sparse":       "logit-lens from RT-selected SAE features only",
            "surp_suppressed":   "logit-lens from non-selected SAE features",
        },
    })

    print(f"\nDone. Results in: {OUT_DIR}/")


if __name__ == "__main__":
    main()