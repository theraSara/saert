"""Pinned SAE extraction and geometric/behavioral fidelity; no RT feature selection."""
import argparse
import gc
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from scipy import sparse
import torch

try:
    from .pipeline_utils import file_record, read_table, run_metadata, sha256_file, window_plan, write_json
    from .surprisal import HOOK_FILES, MODEL_PROCESSING, load_model
    from .validate_outputs import validate_corpus
except ImportError:
    from pipeline_utils import file_record, read_table, run_metadata, sha256_file, window_plan, write_json
    from surprisal import HOOK_FILES, MODEL_PROCESSING, load_model
    from validate_outputs import validate_corpus

ROOT = Path(__file__).resolve().parent.parent
SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
SAE_REVISION = "57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9"


def check_compatibility(raw, hook, extraction):
    metadata = raw.get("metadata", raw)
    recorded_hook = metadata.get("hook_name", metadata.get("hook_point"))
    if recorded_hook != hook or metadata.get("model_name") not in {"gpt2", "gpt2-small"}:
        raise ValueError(f"Wrong checkpoint model/hook: {recorded_hook}, expected {hook}")
    if raw.get("d_in") != 768 or not isinstance(raw.get("d_sae"), int):
        raise ValueError("Invalid checkpoint dimensions")
    if metadata.get("context_size") != extraction["hidden"]["window_size"]:
        raise ValueError("Saved hidden-state context does not match SAE training context")
    if extraction["model"]["processing"] != MODEL_PROCESSING:
        raise ValueError("Model processing flags differ from the audited extraction")
    if extraction["hidden"]["pooling"] != "final_subtoken" or not extraction["hidden"]["save_prefix"]:
        raise ValueError("Expected final-subtoken and pre-word prefix arrays")
    if raw.get("normalize_activations", "none") not in {"none", False, None}:
        raise ValueError("This loader requires explicit implementation of checkpoint normalization")
    return {"model_hook_dimensions_context": "checked",
            "known_release_processing": {"center_writing_weights": True},
            "training_bos_policy": metadata.get("prepend_bos", "not recorded in original config"),
            "limitation": "Legacy config does not establish every training-time processing flag; loader defaults are not historical evidence."}


def load_checkpoint(hook, revision, device, extractions):
    from huggingface_hub import hf_hub_download
    from sae_lens import SAE
    from sae_lens.loading.pretrained_sae_loaders import sae_lens_disk_loader
    paths = {name: Path(hf_hub_download(SAE_REPO, f"{hook}/{name}", revision=revision))
             for name in ("cfg.json", "sae_weights.safetensors")}
    raw = json.loads(paths["cfg.json"].read_text())
    checks = [check_compatibility(raw, hook, m) for m in extractions]

    def converter(path, device, cfg_overrides=None):
        overrides = {"model_from_pretrained_kwargs": {"center_writing_weights": True},
                     **(cfg_overrides or {})}
        return sae_lens_disk_loader(path, device, cfg_overrides=overrides)

    sae = SAE.load_from_disk(paths["cfg.json"].parent, device=device, dtype="float32", converter=converter)
    sae.eval()
    if sae.cfg.d_in != 768 or sae.cfg.d_sae != raw["d_sae"] or sae.cfg.normalize_activations != "none":
        raise ValueError("Converted SAE differs from validated checkpoint")
    provenance = {"repo": SAE_REPO, "revision": revision, "hook": hook,
                  "files": {k: file_record(v) for k, v in paths.items()},
                  "original_config": raw, "compatibility": checks[0],
                  "width": sae.cfg.d_sae, "architecture": sae.cfg.architecture()}
    return sae, provenance


@torch.inference_mode()
def encode_array(hidden, sae, device="cpu", batch_size=128):
    if hidden.ndim != 2 or hidden.shape[1] != sae.cfg.d_in or not np.isfinite(hidden).all():
        raise ValueError("Invalid hidden-state array")
    parts, records = [], []
    for start in range(0, len(hidden), batch_size):
        x = torch.tensor(np.array(hidden[start:start+batch_size]), dtype=torch.float32, device=device)
        z = sae.encode(x)
        reconstruction = sae.decode(z)
        if z.shape != (len(x), sae.cfg.d_sae) or not torch.isfinite(z).all() or (z < 0).any():
            raise ValueError("Expected finite nonnegative features from this ReLU SAE")
        if reconstruction.shape != x.shape or not torch.isfinite(reconstruction).all():
            raise ValueError("Invalid reconstruction")
        parts.append(sparse.csr_matrix(z.cpu().numpy()))
        error = (x - reconstruction).square().sum(-1)
        norm2 = x.square().sum(-1)
        cosine = torch.nn.functional.cosine_similarity(x, reconstruction, dim=-1)
        records.append(pd.DataFrame({"n_active": (z > 0).sum(-1).cpu().numpy(),
            "squared_error": error.cpu().numpy(), "squared_input_norm": norm2.cpu().numpy(),
            "relative_l2_error": (error / norm2.clamp_min(1e-20)).sqrt().cpu().numpy(),
            "cosine_similarity": cosine.cpu().numpy()}))
    return sparse.vstack(parts, format="csr"), pd.concat(records, ignore_index=True)


def geometry_summary(hidden, diagnostics, mask):
    x = np.asarray(hidden[mask], dtype=np.float64)
    d = diagnostics.loc[mask]
    sse = float(d.squared_error.sum())
    centered_ss = float(np.square(x - x.mean(axis=0)).sum())
    return {"n_rows": len(x), "mean_active": float(d.n_active.mean()),
            "fraction_zero_rows": float(d.n_active.eq(0).mean()),
            "mean_relative_l2_error": float(d.relative_l2_error.mean()),
            "mean_cosine_similarity": float(d.cosine_similarity.mean()),
            "reconstruction_r2_centered": 1 - sse / centered_ss if centered_ss > 1e-12 else None}


def select_windows(tokens, manifest, count, seed):
    """At most one window per selected text; selection never sees RT or features."""
    text_ids = np.sort(tokens.text_id.unique())
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(text_ids, size=min(count, len(text_ids)), replace=False))
    windows = []
    for i, text_id in enumerate(chosen):
        group = tokens[tokens.text_id == text_id].sort_values("token_idx")
        ids = [manifest["bos_token_id"]] + group.token_id.astype(int).tolist()
        plans = list(window_plan(len(ids), manifest["hidden"]["window_size"], manifest["hidden"]["stride"]))
        start, end, claim_start, claim_end = plans[0 if i == 0 else int(rng.integers(len(plans)))]
        windows.append({"text_id": int(text_id), "start": start, "end": end,
                        "claim_start": max(claim_start, start+1), "claim_end": claim_end,
                        "ids": ids[start:end]})
    return windows


@torch.inference_mode()
def prepare_fidelity(model, windows, device):
    for w in windows:
        inputs = torch.tensor([w["ids"]], device=device)
        logits, cache = model.run_with_cache(inputs, names_filter=list(HOOK_FILES), prepend_bos=False)
        positions = torch.arange(w["claim_start"]-w["start"]-1, w["claim_end"]-w["start"]-1, device=device)
        logp = logits[0, positions].float().log_softmax(-1)
        target = inputs[0, positions+1]
        w.update({"positions": positions.cpu(), "target": target.cpu(), "logp": logp.cpu(),
                  "original_nll_bits": float(-logp.gather(1, target[:, None]).mean() / np.log(2)),
                  "cache": {hook: cache[hook][0].cpu() for hook in HOOK_FILES}})
    return windows


def verify_live_states(rows, states, windows, hook, kind):
    checked, max_error = 0, 0.0
    start_col = "hidden_window_start" if kind == "post" else "prefix_window_start"
    pos_col = "hidden_model_token_idx" if kind == "post" else "prefix_model_token_idx"
    for w in windows:
        mask = (rows.text_id == w["text_id"]) & (rows[start_col] == w["start"])
        selected = rows.loc[mask]
        if selected.empty:
            continue
        positions = selected[pos_col].to_numpy(int) - w["start"]
        live = w["cache"][hook][positions].numpy()
        saved = np.asarray(states[selected.hidden_row_idx.to_numpy(int)])
        if not np.allclose(live, saved, atol=2e-4, rtol=2e-5):
            raise ValueError(f"Live/saved state mismatch at {hook}, {kind}")
        checked += len(selected)
        max_error = max(max_error, float(np.abs(live-saved).max()))
    if not checked:
        raise ValueError("No saved rows checked against live model states")
    return {"n_rows": checked, "max_absolute_error": max_error}


def replacement_hook(sae, mode, protect_first):
    def replace(activation, hook):
        if mode == "identity":
            return activation.clone()
        result = activation.clone()
        first = 1 if protect_first else 0
        original = activation[:, first:, :]
        if mode.startswith("reconstruction"):
            result[:, first:, :] = sae.decode(sae.encode(original))
        elif mode.startswith("zero"):
            result[:, first:, :] = 0
        else:
            raise ValueError(f"Unknown intervention: {mode}")
        return result
    return replace


@torch.inference_mode()
def behavioral_fidelity(model, sae, hook, windows, device):
    records = []
    for w in windows:
        original_logp = w["logp"].to(device)
        original_p = original_logp.exp()
        record = {k: w[k] for k in ["text_id", "start", "end", "claim_start", "claim_end", "original_nll_bits"]}
        record["n_scored_tokens"] = len(w["target"])
        first_state = w["cache"][hook][0:1].to(device)
        first_recon = sae.decode(sae.encode(first_state))
        record["initial_position_relative_error"] = float((first_state-first_recon).norm()/first_state.norm().clamp_min(1e-10))
        for mode in ["identity", "reconstruction", "zero", "reconstruction_preserve_window_start", "zero_preserve_window_start"]:
            logits = model.run_with_hooks(torch.tensor([w["ids"]], device=device), prepend_bos=False,
                fwd_hooks=[(hook, replacement_hook(sae, mode, w["start"] == 0 or mode.endswith("preserve_window_start")))])
            logp = logits[0, w["positions"].to(device)].float().log_softmax(-1)
            nll = -logp.gather(1, w["target"].to(device)[:, None]).mean() / np.log(2)
            kl = (original_p * (original_logp - logp)).sum(-1).mean() / np.log(2)
            if not torch.isfinite(nll) or not torch.isfinite(kl):
                raise ValueError("Nonfinite behavioral fidelity")
            record[f"{mode}_nll_bits"] = float(nll)
            record[f"{mode}_kl_bits"] = float(kl)
            if mode == "identity" and not torch.allclose(logp, original_logp, atol=2e-4, rtol=2e-5):
                raise ValueError("Identity intervention changed model predictions")
        records.append(record)
    return pd.DataFrame(records)


def save_hook(corpus, hook, sae, checkpoint, rows, extraction_path, results_root, output,
              device, batch_size, model, windows, seed, overwrite):
    label = HOOK_FILES[hook]
    destination = output / corpus / label
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"{destination} is not empty; choose new output or use --overwrite")
    summaries, live_checks, inputs = [], {}, {"extraction_manifest": file_record(extraction_path)}
    with tempfile.TemporaryDirectory(prefix="sae_", dir=destination.parent) as temporary:
        temp = Path(temporary)
        row_map = rows[["row_uid", "hidden_row_idx", "text_id", "word_position", "word", "prefix_kind"]].copy()
        row_map.to_csv(temp / "rows.csv", index=False)
        for kind, stem in [("post", "hidden"), ("prefix", "prefix_hidden")]:
            path = results_root / f"{corpus}_{stem}_{label}.npy"
            inputs[kind] = file_record(path)
            hidden = np.load(path, mmap_mode="r", allow_pickle=False)
            features, diagnostics = encode_array(hidden, sae, device, batch_size)
            sparse.save_npz(temp / f"{kind}_features.npz", features)
            pd.concat([row_map, diagnostics], axis=1).to_csv(temp / f"{kind}_reconstruction.csv", index=False)
            counts = features.getnnz(axis=0)
            pd.DataFrame({"feature_id": np.arange(features.shape[1]), "n_active": counts,
                          "activation_rate": counts / len(rows),
                          "mean_activation": np.asarray(features.mean(axis=0)).ravel()}).to_csv(temp / f"{kind}_feature_stats.csv", index=False)
            groups = {"all": np.ones(len(rows), dtype=bool)}
            if kind == "prefix":
                groups.update({"text_token": rows.prefix_kind.eq("text_token").to_numpy(),
                               "bos": rows.prefix_kind.eq("bos").to_numpy()})
            for group, mask in groups.items():
                if mask.any():
                    summaries.append({"corpus": corpus, "hook": hook, "label": label, "kind": kind,
                                      "group": group, **geometry_summary(hidden, diagnostics, mask)})
            if windows:
                live_checks[kind] = verify_live_states(rows, hidden, windows, hook, kind)
            del hidden, features
        pd.DataFrame(summaries).to_csv(temp / "geometry.csv", index=False)
        if windows:
            behavioral_fidelity(model, sae, hook, windows, device).to_csv(temp / "behavior.csv", index=False)
        manifest = {**run_metadata(), "schema_version": 1, "corpus": corpus, "hook": hook,
            "checkpoint": checkpoint, "inputs": inputs, "n_rows": len(rows), "feature_width": sae.cfg.d_sae,
            "selection": "none; every SAE feature retained in scipy CSR, original checkpoint feature IDs",
            "post_policy": "state at final subtoken; observes current word",
            "prefix_policy": "before first subtoken in that token's matched-context window",
            "live_state_checks": live_checks, "fidelity_seed": seed,
            "fidelity_windows": [{k: w[k] for k in ["text_id", "start", "end", "claim_start", "claim_end", "ids"]} for w in windows],
            "intervention": "one hook at all non-BOS positions; compare identity, reconstruction, zero; also test preservation of every window's initial position",
            "limitations": ["Sampled fidelity is diagnostic, not a population estimate or a feature-level causal test.",
                            "No changes to RT, baseline predictions, or original surprisal files.",
                            "Global feature counts are descriptive; selection/scaling belongs inside later training folds."],
            "source_code": [file_record(__file__)],
            "outputs": {p.name: {"sha256": sha256_file(p)} for p in temp.iterdir()}}
        write_json(temp / "manifest.json", manifest)
        for p in temp.iterdir():
            if p.name != "manifest.json":
                p.replace(destination / p.name)
        (temp / "manifest.json").replace(destination / "manifest.json")
    print(f"Saved {corpus} {label}: {len(rows):,} rows x {sae.cfg.d_sae:,} features, post + prefix", flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["both", "provo", "natural_stories"], default="both")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/prepared_v2")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/surprisal_bos")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/sae_bos")
    parser.add_argument("--hooks", nargs="+", choices=list(HOOK_FILES.values()), default=list(HOOK_FILES.values()))
    parser.add_argument("--revision", default=SAE_REVISION)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fidelity-windows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("Use an immutable 40-character checkpoint commit SHA")
    if min(args.batch_size, args.threads) < 1 or args.fidelity_windows < 0 or len(set(args.hooks)) != len(args.hooks):
        parser.error("Invalid batch size, thread/window count, or duplicate hooks")
    torch.set_num_threads(args.threads)
    corpora = ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]
    manifests, frames, paths = {}, {}, {}
    for corpus in corpora:
        for label in args.hooks:
            destination = args.output_dir / corpus / label
            if destination.exists() and any(destination.iterdir()) and not args.overwrite:
                raise FileExistsError(f"Refusing to replace {destination}; choose new output or --overwrite")
        print(f"Validating source artifacts: {corpus}", flush=True)
        validate_corpus(corpus, args.data_dir, args.results_dir)
        paths[corpus] = args.results_dir / f"{corpus}_extraction_manifest.json"
        manifests[corpus] = json.loads(paths[corpus].read_text())
        frames[corpus] = read_table(args.results_dir / f"{corpus}_surprisal.csv")
    model, windows = None, {c: [] for c in corpora}
    if args.fidelity_windows:
        revisions = {m["model"]["resolved_revision"] for m in manifests.values()}
        if len(revisions) != 1:
            raise ValueError("Corpora have different model revisions")
        model, _, _ = load_model(args.device, revisions.pop())
        for corpus in corpora:
            tokens = read_table(args.results_dir / f"{corpus}_token_windows.csv")
            selected = select_windows(tokens, manifests[corpus], args.fidelity_windows, args.seed)
            windows[corpus] = prepare_fidelity(model, selected, args.device)
    for hook, label in HOOK_FILES.items():
        if label not in args.hooks:
            continue
        print(f"Loading pinned checkpoint: {hook}", flush=True)
        sae, checkpoint = load_checkpoint(hook, args.revision, args.device, list(manifests.values()))
        for corpus in corpora:
            save_hook(corpus, hook, sae, checkpoint, frames[corpus], paths[corpus], args.results_dir,
                      args.output_dir, args.device, args.batch_size, model, windows[corpus], args.seed, args.overwrite)
        del sae
        gc.collect()
    print("SAE extraction complete. Review geometry and behavioral fidelity before feature regression.", flush=True)


if __name__ == "__main__":
    main()
