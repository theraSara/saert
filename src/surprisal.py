import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import uuid
import numpy as np
import pandas as pd

try:
    from .pipeline_utils import (KEYS, file_record, normalize_words, read_table, run_metadata,
                                 sha256_file, tokenize_words, tokenizer_hash, validate_word_matches,
                                 window_plan, write_json)
except ImportError:
    from pipeline_utils import (KEYS, file_record, normalize_words, read_table, run_metadata,
                                sha256_file, tokenize_words, tokenizer_hash, validate_word_matches,
                                window_plan, write_json)

ROOT = Path(__file__).resolve().parent.parent
# Preserve old filenames but specify the actual location, not an inferred cognitive stage.
HOOK_FILES = {f"blocks.{i}.hook_resid_pre": f"L{i+1:02d}" for i in range(12)}
HOOK_FILES["blocks.11.hook_resid_post"] = "final_resid_post"
MODEL_PROCESSING = {"center_writing_weights": True, "center_unembed": True,
                    "fold_ln": True, "fold_value_biases": True,
                    "refactor_factored_attn_matrices": False}


def load_model(device, revision):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformer_lens import HookedTransformer
    config = AutoConfig.from_pretrained("gpt2", revision=revision)
    resolved = getattr(config, "_commit_hash", None)
    if not resolved:
        raise ValueError("Could not resolve GPT-2 revision to a commit SHA")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", revision=resolved, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Fast GPT-2 tokenizer required")
    model = HookedTransformer.from_pretrained(
        "gpt2", revision=resolved, tokenizer=tokenizer, device=device,
        dtype=torch.float32, default_prepend_bos=False, **MODEL_PROCESSING)
    model.eval()
    if (model.cfg.n_layers, model.cfg.d_model, model.cfg.n_ctx) != (12, 768, 1024):
        raise ValueError("Expected GPT-2 small: 12 layers, width 768, context 1024")
    return model, tokenizer, resolved


def run_forward_pass(token_ids, model, device, hooks=()):
    import torch
    if not 1 <= len(token_ids) <= 1024:
        raise ValueError("Forward pass must contain 1..1024 tokens")
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        if hooks:
            logits, cache = model.run_with_cache(inputs, names_filter=list(hooks),
                return_type="logits", prepend_bos=False)
            hidden = np.stack([cache[h][0].detach().cpu().float().numpy() for h in hooks])
            if not np.isfinite(hidden).all():
                raise ValueError("Nonfinite hidden states")
        else:
            logits = model(inputs, return_type="logits", prepend_bos=False)
            hidden = None
        scores = np.full(len(token_ids), np.nan, dtype=np.float32)
        if len(token_ids) > 1:
            loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), inputs[0, 1:], reduction="none")
            scores[1:] = (loss / np.log(2)).cpu().numpy()
            if not np.isfinite(scores[1:]).all() or (scores[1:] < 0).any():
                raise ValueError("Invalid token surprisal")
    return scores, hidden


def extract_sequence(token_ids, model, device, window_size=1024, stride=512, hooks=(),
                     save_prefix=False, forward_fn=None):
    forward_fn = forward_fn or run_forward_pass
    n = len(token_ids)
    plan = list(window_plan(n, window_size, stride))
    result = {"surprisal": np.full(n, np.nan, dtype=np.float32),
              "window_start": np.full(n, -1, dtype=np.int64),
              "left_context": np.full(n, -1, dtype=np.int64),
              "hidden": None, "prefix_hidden": None}
    for start, end, claim_start, claim_end in plan:
        score, states = forward_fn(token_ids[start:end], model, device, hooks)
        count = end - start
        if score.shape != (count,):
            raise ValueError("Forward score shape mismatch")
        dest = slice(claim_start, claim_end)
        local_start = claim_start - start
        local = slice(local_start, count)
        result["surprisal"][dest] = score[local]
        result["window_start"][dest] = start
        result["left_context"][dest] = np.arange(local_start, count)
        if hooks:
            if states is None or states.shape[:2] != (len(hooks), count):
                raise ValueError("Forward hidden-state shape mismatch")
            if result["hidden"] is None:
                shape = (len(hooks), n, states.shape[-1])
                result["hidden"] = np.full(shape, np.nan, dtype=np.float32)
                if save_prefix:
                    result["prefix_hidden"] = np.full(shape, np.nan, dtype=np.float32)
            result["hidden"][:, dest, :] = states[:, local, :]
            if save_prefix:
                first_target = max(claim_start, start + 1)
                result["prefix_hidden"][:, first_target:claim_end, :] = states[:, first_target-start-1:count-1, :]
    if not np.isnan(result["surprisal"][0]) or not np.isfinite(result["surprisal"][1:]).all():
        raise ValueError("Expected exactly one unscored initial input position")
    if (result["window_start"] < 0).any() or (result["surprisal"][1:] < 0).any():
        raise ValueError("Invalid window coverage or scores")
    return result


def extract_story(token_ids, bos_token_id, model, device, window_size=1024, stride=512,
                  hooks=(), save_prefix=False, forward_fn=None):
    if not token_ids:
        raise ValueError("Cannot score an empty story")
    if not isinstance(bos_token_id, (int, np.integer)) or bos_token_id < 0:
        raise ValueError("A valid BOS token ID is required to score the first word")
    result = extract_sequence([int(bos_token_id)] + list(token_ids), model, device,
                              window_size, stride, hooks, save_prefix, forward_fn)
    for name in ["surprisal", "window_start", "left_context"]:
        result[name] = result[name][1:]
    for name in ["hidden", "prefix_hidden"]:
        if result[name] is not None:
            result[name] = result[name][:, 1:, :]
            if not np.isfinite(result[name]).all():
                raise ValueError(f"Nonfinite story {name}")
    if not np.isfinite(result["surprisal"]).all():
        raise ValueError("Every story token must have finite BOS-conditioned surprisal")
    return result


def word_surprisals(scores, spans):
    values = []
    for start, end in spans:
        if not 0 <= start <= end < len(scores):
            raise ValueError(f"Out-of-bounds word span {(start, end)}")
        values.append(float(np.sum(scores[start:end+1], dtype=np.float64)))
    return np.asarray(values)


def load_prepared(directory, corpus):
    directory = Path(directory) / corpus
    path = directory / f"{corpus}_prepared.csv"
    manifest_path = directory / f"{corpus}_prepared_manifest.json"
    stimulus_path = directory / "story_token_order.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}; regenerate using corrected src/data.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or manifest.get("corpus") != corpus:
        raise ValueError("Incompatible preparation manifest")
    for artifact in [path, stimulus_path]:
        expected = manifest["outputs"][artifact.name]["sha256"]
        if sha256_file(artifact) != expected:
            raise ValueError(f"Prepared artifact changed since preparation: {artifact}")
    frame = normalize_words(read_table(path), corpus, complete=True)
    stimuli = normalize_words(read_table(stimulus_path), "prepared stimulus order", complete=True)
    validate_word_matches(stimuli, frame, corpus)
    if not frame[KEYS].equals(stimuli[KEYS]):
        raise ValueError("Prepared rows must cover the entire stimulus sequence")
    for col in ["row_uid", "preparation_id", "primary_RT", "log_primary_RT", "rt_measure",
                "token_ids_str", "start_token_idx", "end_token_idx"]:
        if col not in frame:
            raise ValueError(f"Prepared file missing {col}")
    expected_uids = corpus + ":" + frame["text_id"].astype(str) + ":" + frame["word_position"].astype(str)
    if not frame["row_uid"].equals(expected_uids) or not frame["row_uid"].is_unique:
        raise ValueError("Prepared row IDs disagree with word keys")
    if set(frame["preparation_id"]) != {manifest["preparation_id"]}:
        raise ValueError("Mixed preparation IDs")
    return frame, manifest, {"prepared": file_record(path), "manifest": file_record(manifest_path),
                              "stimuli": file_record(stimulus_path)}


def extract_corpus(corpus, frame, preparation, input_records, model, tokenizer, device, output,
                   revision, window_size=1024, stride=512, hidden_context=128, hidden_stride=64,
                   save_prefix=False, overwrite=False):
    output = Path(output)
    list(window_plan(1, window_size, stride))
    list(window_plan(1, hidden_context, hidden_stride))
    run_id = str(uuid.uuid4())
    hooks = list(HOOK_FILES)
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        raise ValueError("Tokenizer has no BOS token; first-word scoring requires it")
    names = [f"{corpus}_surprisal.csv", f"{corpus}_token_windows.csv",
             f"{corpus}_hidden_rows.csv", f"{corpus}_extraction_manifest.json"]
    names += [f"{corpus}_hidden_{label}.npy" for label in HOOK_FILES.values()]
    if save_prefix:
        names += [f"{corpus}_prefix_hidden_{label}.npy" for label in HOOK_FILES.values()]
    elif output.exists() and list(output.glob(f"{corpus}_prefix_hidden_*.npy")):
        raise ValueError("Old prefix arrays exist; use a fresh output directory or --save-prefix")
    if not overwrite and any((output / name).exists() for name in names):
        raise FileExistsError(f"Extraction output exists in {output}; use --output-dir or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    manifest_name = f"{corpus}_extraction_manifest.json"
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=output) as temp, ExitStack() as handles:
        temp = Path(temp)
        width = model.cfg.d_model
        arrays, prefixes = {}, {}
        for hook, label in HOOK_FILES.items():
            array = np.lib.format.open_memmap(temp / f"{corpus}_hidden_{label}.npy",
                mode="w+", dtype="float32", shape=(len(frame), width))
            handles.callback(array._mmap.close)
            arrays[hook] = array
            if save_prefix:
                array = np.lib.format.open_memmap(temp / f"{corpus}_prefix_hidden_{label}.npy",
                    mode="w+", dtype="float32", shape=(len(frame), width))
                handles.callback(array._mmap.close)
                prefixes[hook] = array
        rows, token_rows = [], []
        for text_id, group in frame.groupby("text_id", sort=True):
            print(f"{corpus}: text {text_id} ({len(group)} words)")
            _, ids, offsets, spans = tokenize_words(group["word"].tolist(), tokenizer)
            if tokenizer_hash(tokenizer) != preparation["tokenizer"]["backend_sha256"]:
                raise ValueError("Tokenizer differs from preparation; use matching revisions and rerun data.py")
            for record, (s, e) in zip(group.to_dict("records"), spans):
                if (int(record["start_token_idx"]), int(record["end_token_idx"])) != (s, e) or json.loads(record["token_ids_str"]) != ids[s:e+1]:
                    raise ValueError(f"Preparation/extraction token alignment differs at {record['row_uid']}")
            same_context = (window_size, stride) == (hidden_context, hidden_stride)
            short = extract_story(ids, bos_token_id, model, device, hidden_context, hidden_stride, hooks, save_prefix)
            long = short if same_context else extract_story(ids, bos_token_id, model, device, window_size, stride)
            long_words = word_surprisals(long["surprisal"], spans)
            short_words = word_surprisals(short["surprisal"], spans)
            base_row = len(rows)
            ends = [end for _, end in spans]
            starts = [start for start, _ in spans]
            for h, hook in enumerate(hooks):
                arrays[hook][base_row:base_row+len(group)] = short["hidden"][h, ends]
                if save_prefix:
                    prefixes[hook][base_row:base_row+len(group)] = short["prefix_hidden"][h, starts]
            owner = {}
            for i, (record, (s, e)) in enumerate(zip(group.to_dict("records"), spans)):
                row = dict(record)
                row.update({"extraction_id": run_id, "hidden_row_idx": base_row + i,
                    "surprisal": long_words[i], "surprisal_matched_context": short_words[i],
                    "surprisal_units": "bits", "hidden_token_idx": e,
                    "hidden_model_token_idx": e+1,
                    "hidden_input_kind": "final_subtoken_resid", "hidden_context_size": hidden_context,
                    "hidden_context_stride": hidden_stride, "surprisal_context_size": window_size,
                    "surprisal_context_stride": stride,
                    "hidden_window_start": int(short["window_start"][e]),
                    "hidden_left_context": int(short["left_context"][e]),
                    "surprisal_left_context_min": int(long["left_context"][s:e+1].min()),
                    "surprisal_left_context_max": int(long["left_context"][s:e+1].max()),
                    "prefix_token_idx": s-1, "prefix_model_token_idx": s,
                    "prefix_kind": "bos" if s == 0 else "text_token", "prefix_available": True,
                    "prefix_window_start": int(short["window_start"][s]),
                    "prefix_context_tokens": int(short["left_context"][s])})
                rows.append(row)
                for t in range(s, e+1):
                    owner[t] = record["row_uid"]
            for t, token_id in enumerate(ids):
                token_rows.append({"extraction_id": run_id, "text_id": text_id, "token_idx": t,
                    "model_token_idx": t+1,
                    "token_id": token_id, "row_uid": owner[t], "char_start": offsets[t][0],
                    "char_end": offsets[t][1], "surprisal": float(long["surprisal"][t]),
                    "surprisal_matched_context": float(short["surprisal"][t]),
                    "surprisal_window_start": int(long["window_start"][t]),
                    "surprisal_left_context": int(long["left_context"][t]),
                    "hidden_window_start": int(short["window_start"][t]),
                    "hidden_left_context": int(short["left_context"][t])})
        result = pd.DataFrame(rows)
        if len(result) != len(frame) or not result["row_uid"].is_unique:
            raise ValueError("Extraction row coverage mismatch")
        for col in ["surprisal", "surprisal_matched_context"]:
            if not np.isfinite(result[col]).all() or (result[col] < 0).any():
                raise ValueError(f"Every word, including story-initial words, needs finite nonnegative {col}")
        result.to_csv(temp / f"{corpus}_surprisal.csv", index=False)
        pd.DataFrame(token_rows).to_csv(temp / f"{corpus}_token_windows.csv", index=False)
        result[["extraction_id", "hidden_row_idx", "row_uid", "text_id", "word_position", "word",
                "hidden_token_idx", "hidden_model_token_idx", "prefix_kind", "prefix_available"]].to_csv(temp / f"{corpus}_hidden_rows.csv", index=False)
        for array in list(arrays.values()) + list(prefixes.values()):
            array.flush()
            array._mmap.close()
        arrays.clear()
        prefixes.clear()
        outputs = {p.name: {"sha256": sha256_file(p)} for p in temp.iterdir()}
        manifest = {**run_metadata(), "schema_version": 3, "extraction_id": run_id,
            "preparation_id": preparation["preparation_id"], "corpus": corpus,
            "model": {"name": "gpt2", "resolved_revision": revision, "dtype": "float32",
                      "device": device, "processing": MODEL_PROCESSING},
            "tokenizer_backend_sha256": tokenizer_hash(tokenizer),
            "bos_policy": "one explicit BOS per story before windowing; never reinserted at window boundaries",
            "bos_token_id": int(bos_token_id), "bos_token": tokenizer.bos_token,
            "index_policy": "text token indices exclude BOS; model indices and window starts include BOS at 0",
            "context_counts": "preceding model tokens, including BOS when present in the window",
            "window_policy": "first covering window; maximum left context among evaluated windows",
            "position_policy": "absolute position IDs reset to zero within each window",
            "surprisal": {"window_size": window_size, "stride": stride, "units": "bits"},
            "hidden": {"window_size": hidden_context, "stride": hidden_stride, "pooling": "final_subtoken",
                       "save_prefix": save_prefix, "prefix_policy": "before first subtoken, in its scoring window"},
            "hook_files": {hook: {"file": f"{corpus}_hidden_{label}.npy",
                 "completed_blocks": i if i < 12 else 12} for i, (hook, label) in enumerate(HOOK_FILES.items())},
            "inputs": input_records, "outputs": outputs,
            "source_code": [file_record(__file__), file_record(Path(__file__).with_name("pipeline_utils.py"))],
            "n_words": len(result), "n_texts": int(result["text_id"].nunique()),
            "note": "No SAE intervention performed; short context is an operating condition, not a fidelity guarantee"}
        write_json(temp / manifest_name, manifest)
        for name in names:
            if name != manifest_name:
                (temp / name).replace(output / name)
        (temp / manifest_name).replace(output / manifest_name)
    print(f"Saved {corpus}: {len(result):,} complete stimulus rows and {len(hooks)} hook arrays -> {output}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["provo", "natural_stories", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "surprisal")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--hidden-context", type=int, default=128)
    parser.add_argument("--hidden-stride", type=int, default=64)
    parser.add_argument("--save-prefix", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    list(window_plan(1, args.window_size, args.stride))
    list(window_plan(1, args.hidden_context, args.hidden_stride))
    corpora = ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]
    prepared = {c: load_prepared(args.data_dir, c) for c in corpora}
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer, resolved = load_model(device, args.revision)
    for corpus in corpora:
        frame, manifest, inputs = prepared[corpus]
        extract_corpus(corpus, frame, manifest, inputs, model, tokenizer, device, args.output_dir,
                       resolved, args.window_size, args.stride, args.hidden_context, args.hidden_stride,
                       args.save_prefix, args.overwrite)


if __name__ == "__main__":
    main()
