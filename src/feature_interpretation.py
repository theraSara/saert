import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from transformer_lens import HookedTransformer

try:
    from .pipeline_utils import file_record, read_table, run_metadata, write_json
    from .surprisal import HOOK_FILES, load_prepared
except ImportError:
    from pipeline_utils import file_record, read_table, run_metadata, write_json
    from surprisal import HOOK_FILES, load_prepared

ROOT = Path(__file__).resolve().parent.parent

BEST_HOOKS = {
    "provo":            {"post": "L01", "prefix": "L01"},
    "natural_stories":  {"post": "L08", "prefix": "L08"},
}

DEFAULT_TOP_N = 30
CONTEXT_WINDOW = 5


def load_sae_decoder(hook_label: str, device: str):
    """
    Load the decoder matrix for one SAE hook from SAELens.
    Returns W_dec: (hidden_size, n_features) float32 tensor.
    """
    from sae_lens import SAE

    layer_idx = int(hook_label[1:]) - 1   # L01 → 0, L08 → 7
    hook_name = f"blocks.{layer_idx}.hook_resid_pre"

    sae, _, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=hook_name,
        device=device,
    )
    sae.eval()
    return sae.W_dec.T.detach().float() 


@torch.no_grad()
def apply_logit_lens(decoder: torch.Tensor, model: HookedTransformer,
    feature_ids: np.ndarray, top_k: int = 20,) -> list[dict]:
    """
    For each feature in feature_ids, compute:
        logit_lens(feature_i) = W_unembed(LayerNorm(W_dec[:, i]))
    Returns top-k token predictions per feature.

    This is corpus-independent: it uses only the model's weights, not any reading time data. 
    The result is stable across fitting corpora (He et al. 2026 corpus conditionality argument).
    """
    results = []
    W_dec = decoder.to(model.cfg.device)  

    for feat_id in feature_ids:
        # Decoder vector for this feature
        d = W_dec[:, feat_id].unsqueeze(0)    # (1, hidden_size)

        # Apply final layer norm (fold_ln=True means LN params are in weights)
        d_normed = model.ln_final(d)           # (1, hidden_size)

        # Project through unembedding matrix
        logits = model.unembed(d_normed)       # (1, 1, vocab_size)
        logits = logits.squeeze()              # (vocab_size,)

        # Top-k tokens
        topk_logits, topk_ids = torch.topk(logits, top_k)
        topk_tokens = [model.tokenizer.decode([tid.item()]) for tid in topk_ids]

        # Bottom-k tokens (what fires against this feature)
        botk_logits, botk_ids = torch.topk(-logits, 10)
        botk_tokens = [model.tokenizer.decode([tid.item()]) for tid in botk_ids]

        results.append({
            "feature_id":       int(feat_id),
            "logit_lens_top20": " | ".join(f"{t.strip()!r}" for t in topk_tokens),
            "logit_lens_bot10": " | ".join(f"{t.strip()!r}" for t in botk_tokens),
            "logit_lens_label": "",   # filled manually after inspection
        })

    return results

def find_max_activating(
    features: sparse.csr_matrix,   # (n_words, n_features)
    feature_ids: np.ndarray,
    frame: pd.DataFrame,            # aligned word-level DataFrame
    top_n: int = 10,
    context_window: int = CONTEXT_WINDOW,
) -> pd.DataFrame:
    """
    For each feature, find the top-n words where it activates most strongly.
    Returns a DataFrame with one row per (feature, example).
    """
    rows = []
    # Convert to CSC for efficient column slicing
    features_csc = features.tocsc()

    for feat_id in feature_ids:
        col = np.asarray(features_csc[:, feat_id].todense()).ravel()
        if col.max() == 0:
            continue

        top_word_indices = np.argsort(col)[-top_n:][::-1]

        for rank, word_idx in enumerate(top_word_indices):
            activation = float(col[word_idx])
            if activation == 0:
                continue

            row_data = frame.iloc[word_idx]
            text_id  = row_data["text_id"]
            word_pos = row_data["word_position"]

            # Build context window: surrounding words from same text
            text_frame = frame[frame["text_id"] == text_id].sort_values("word_position")
            word_positions = text_frame["word_position"].values
            target_idx_in_text = np.searchsorted(word_positions, word_pos)
            lo = max(0, target_idx_in_text - context_window)
            hi = min(len(text_frame), target_idx_in_text + context_window + 1)
            context_words = text_frame.iloc[lo:hi]["word"].tolist()
            target_in_context = target_idx_in_text - lo

            # Mark the target word with brackets
            context_words[target_in_context] = f"[{context_words[target_in_context]}]"
            context_str = " ".join(context_words)

            rows.append({
                "feature_id":       int(feat_id),
                "rank":             rank + 1,
                "row_uid":          row_data["row_uid"],
                "word":             row_data["word"],
                "word_position":    int(word_pos),
                "text_id":          int(text_id),
                "activation":       round(activation, 4),
                "context":          context_str,
                "RT_prime":         round(float(row_data.get("RT_resid_controls_oof",row_data.get("log_primary_RT", 0))), 4),
            })

    return pd.DataFrame(rows)

def summarise_coefficients(
    coef_matrix: np.ndarray,   # (n_folds, n_features) — raw feature coefficients
    feature_stats: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """
    Average ridge coefficients across outer folds (only folds where the feature
    was not zeroed by the variance floor). Rank by mean absolute coefficient.
    """
    # Mean and SD of coefficient across folds (NaN where feature was zeroed)
    with np.errstate(invalid="ignore"):
        mean_coef = np.nanmean(coef_matrix, axis=0)
        std_coef  = np.nanstd(coef_matrix, axis=0)
        n_nonzero = np.sum(coef_matrix != 0, axis=0)

    feature_ids = np.arange(coef_matrix.shape[1])
    abs_mean    = np.abs(mean_coef)

    # Sort by absolute mean coefficient, take top_n
    ranked = np.argsort(abs_mean)[-top_n:][::-1]

    rows = []
    for feat_id in ranked:
        # Merge with feature_stats if available
        pct_active = float("nan")
        if feature_stats is not None and feat_id < len(feature_stats):
            row = feature_stats.iloc[feat_id]
            pct_active = float(row.get("pct_active", float("nan")))

        rows.append({
            "feature_id":    int(feat_id),
            "mean_coef":     round(float(mean_coef[feat_id]), 6),
            "std_coef":      round(float(std_coef[feat_id]), 6),
            "abs_mean_coef": round(float(abs_mean[feat_id]), 6),
            "sign":          "positive" if mean_coef[feat_id] > 0 else "negative",
            "n_folds_nonzero": int(n_nonzero[feat_id]),
            "pct_active":    round(pct_active, 2) if not np.isnan(pct_active) else float("nan"),
            "interpretation_note": (
                "active → harder to read" if mean_coef[feat_id] > 0
                else "active → easier to read"
            ),
        })

    return pd.DataFrame(rows)

def interpret_one(
    corpus: str,
    hook_label: str,
    kind: str,              # "post" or "prefix"
    regression_dir: Path,
    sae_dir: Path,
    data_dir: Path,
    output: Path,
    model: HookedTransformer,
    device: str,
    top_n: int = DEFAULT_TOP_N,
):
    """Full interpretation pipeline for one (corpus, hook, kind) combination."""

    print(f"\n  {corpus} | {hook_label} | {kind}")

    # ── Load regression coefficients ───────────────────────────────────────
    # Path: regression_dir/{corpus}/{kind}/{label}/sae/coefficients.npz
    # This is written by sae_regression.py (run_one, line 164: folder = output/corpus/kind/label/repr)
    reg_folder = regression_dir / corpus / kind / hook_label / "sae"
    coef_path  = reg_folder / "coefficients.npz"
    if not coef_path.exists():
        print(f"    [SKIP] No coefficients at {coef_path}")
        print(f"    Run the regression first:")
        print(f"      bash run.sh regression {corpus}")
        print(f"    Or for a single hook:")
        print(f"      python src/sae_regression.py --corpus {corpus} --hooks {hook_label}")
        return None

    coef_data   = np.load(coef_path, allow_pickle=False)
    coef_matrix = coef_data["feature_coef_raw"]   # (n_folds, n_features)
    print(f"    Coefficients: {coef_matrix.shape} (folds × features)")

    # ── Load SAE sparse feature activations ────────────────────────────────
    # Path: sae_dir/{corpus}/{label}/{kind}_features.npz
    # This is written by sae_extraction.py
    sae_folder    = sae_dir / corpus / hook_label
    features_path = sae_folder / f"{kind}_features.npz"
    stats_path    = sae_folder / f"{kind}_feature_stats.csv"

    if not features_path.exists():
        print(f"    [SKIP] No SAE features at {features_path}")
        return None

    features = sparse.load_npz(features_path)   # (n_words, n_features)
    print(f"    Features: {features.shape} sparse")

    feature_stats = pd.read_csv(stats_path) if stats_path.exists() else None

    # ── Load word-level DataFrame ──────────────────────────────────────────
    # load_prepared expects the *parent* directory; it appends corpus/ internally
    frame, _, _ = load_prepared(data_dir, corpus)
    frame = frame.reset_index(drop=True)

    # Merge baseline predictions for RT context (RT_resid_controls_oof)
    pred_path = regression_dir / corpus / kind / hook_label / "sae" / "predictions.csv"
    if pred_path.exists():
        preds = read_table(pred_path)[["row_uid", "baseline_prediction"]]
        frame = frame.merge(preds, on="row_uid", how="left")
        frame["RT_prime"] = frame["log_primary_RT"] - frame["baseline_prediction"]

    # Align frame rows to SAE feature rows.
    # sae_extraction.py writes a rows.csv in sae_dir/{corpus}/{label}/
    # with the row_uid order used when building the sparse feature matrix.
    rows_csv = sae_folder / "rows.csv"
    if rows_csv.exists():
        sae_rows = read_table(rows_csv)[["row_uid"]]
        frame = sae_rows.merge(frame, on="row_uid", how="left").reset_index(drop=True)
    # Otherwise assume feature matrix rows align with prepared frame rows in order

    # Coefficient summary: top features ─────────────────────────────────
    print(f"    Summarising top {top_n} features by |mean coefficient|...")
    top_features = summarise_coefficients(coef_matrix, feature_stats, top_n)
    feature_ids  = top_features["feature_id"].values

    # SAE decoder logit lens 
    print(f"    Computing logit lens for {len(feature_ids)} features...")
    decoder = load_sae_decoder(hook_label, device)
    lens_results = apply_logit_lens(decoder, model, feature_ids, top_k=20)
    lens_df = pd.DataFrame(lens_results)

    # Merge logit lens into top features
    top_features = top_features.merge(lens_df, on="feature_id", how="left")

    # ── Max-activating examples ────────────────────────────────────────────
    print(f"    Finding max-activating examples...")
    max_examples = find_max_activating(
        features, feature_ids, frame,
        top_n=10, context_window=CONTEXT_WINDOW,
    )

    # ── Save outputs ───────────────────────────────────────────────────────
    stem = f"{corpus}_{hook_label}_{kind}"
    top_features.to_csv(output / f"{stem}_top_features.csv", index=False)
    max_examples.to_csv(output / f"{stem}_max_activating.csv", index=False)

    print(f"    → {stem}_top_features.csv ({len(top_features)} features)")
    print(f"    → {stem}_max_activating.csv ({len(max_examples)} examples)")

    # Print top-10 for quick inspection
    print(f"\n    Top 10 features at {corpus}/{hook_label}/{kind}:")
    print(f"    {'ID':>6}  {'coef':>8}  {'pct%':>6}  {'sign':>10}  logit_lens (top 5 tokens)")
    print(f"    {'─'*6}  {'─'*8}  {'─'*6}  {'─'*10}  {'─'*40}")
    for _, row in top_features.head(10).iterrows():
        top5 = " | ".join(row["logit_lens_top20"].split(" | ")[:5]) if pd.notna(row.get("logit_lens_top20")) else "—"
        print(f"    {int(row['feature_id']):>6}  {row['mean_coef']:>+8.4f}  "
              f"{row['pct_active']:>5.1f}%  {row['sign']:>10}  {top5}")

    return {"top_features": top_features, "max_examples": max_examples}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", choices=["provo", "natural_stories", "both"], default="both"
    )
    parser.add_argument(
        "--hook", default=None,
        help="Override best-hook selection. Default: use BEST_HOOKS table (L01 for Provo, L08 for Natural Stories)."
    )
    parser.add_argument(
        "--kinds", nargs="+", choices=["post", "prefix"], default=["post", "prefix"]
    )
    parser.add_argument(
        "--top-n", type=int, default=DEFAULT_TOP_N,
        help=f"Number of top features to interpret per (corpus, hook, kind). Default: {DEFAULT_TOP_N}."
    )
    parser.add_argument(
        "--regression-dir", type=Path,
        default=ROOT / "results" / "sae_regression_v2"
    )
    parser.add_argument(
        "--sae-dir", type=Path,
        default=ROOT / "results" / "sae_bos"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "data" / "prepared_v2"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "feature_interpretation"
    )
    parser.add_argument(
        "--device", choices=["cpu", "mps", "cuda"], default=None
    )
    args = parser.parse_args(argv)

    # Device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Load model once (needed for logit lens)
    print("Loading GPT-2 small via TransformerLens...")
    model = HookedTransformer.from_pretrained(
        "gpt2", fold_ln=True, center_writing_weights=True, device=device
    )
    model.eval()
    print(f"  Loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpora = ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]

    all_outputs = []
    for corpus in corpora:
        for kind in args.kinds:
            hook_label = args.hook or BEST_HOOKS[corpus][kind]
            try:
                result = interpret_one(
                    corpus, hook_label, kind,
                    args.regression_dir, args.sae_dir, args.data_dir,
                    args.output_dir, model, device, args.top_n
                )
                if result is not None:
                    all_outputs.append({
                        "corpus": corpus, "hook": hook_label, "kind": kind,
                        "n_features": len(result["top_features"]),
                        "n_examples": len(result["max_examples"]),
                    })
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
            except Exception as e:
                print(f"  [ERROR] {corpus}/{hook_label}/{kind}: {e}")
                raise

    # Write manifest
    write_json(args.output_dir / "feature_interpretation_manifest.json", {
        **run_metadata(),
        "corpora": corpora,
        "kinds": args.kinds,
        "best_hooks": BEST_HOOKS,
        "top_n": args.top_n,
        "context_window": CONTEXT_WINDOW,
        "method": {
            "logit_lens": "W_unembed(LayerNorm(W_dec[:, i])) — corpus-independent",
            "max_activating": f"Top-10 words by activation value, {CONTEXT_WINDOW}-word context window",
            "reference": "Bloom & Lin 2024; He et al. 2026 (arXiv:2609.01936)"
        },
        "outputs": all_outputs,
    })

    print(f"\nFeature interpretation complete.")
    print(f"Output: {args.output_dir}")
    print()
    print("Next steps:")
    print("  1. Open results/feature_interpretation/*.csv")
    print("  2. Fill in 'logit_lens_label' column with brief linguistic description")
    print("     e.g.: 'past-tense verbs', 'sentence boundaries', 'rare nouns'")
    print("  3. Cross-check labels against max_activating examples")
    print("  4. Group features into linguistic categories for the paper")


if __name__ == "__main__":
    main()