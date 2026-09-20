"""Read-only integrity checks for corrected BOS extraction artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .pipeline_utils import read_table, sha256_file
    from .surprisal import HOOK_FILES, load_prepared
except ImportError:
    from pipeline_utils import read_table, sha256_file
    from surprisal import HOOK_FILES, load_prepared


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_corpus(corpus, data_dir, results_dir):
    root = Path(results_dir)
    prepared, preparation, inputs = load_prepared(data_dir, corpus)
    manifest = json.loads((root / f"{corpus}_extraction_manifest.json").read_text(encoding="utf-8"))
    require(manifest["schema_version"] == 3, "Expected BOS extraction schema 3")
    require(manifest["corpus"] == corpus and manifest["bos_token_id"] == 50256, "Corpus/BOS mismatch")
    require(manifest["preparation_id"] == preparation["preparation_id"], "Preparation ID mismatch")
    for key in ["prepared", "manifest", "stimuli"]:
        require(manifest["inputs"][key]["sha256"] == inputs[key]["sha256"], f"Changed preparation input: {key}")
    for name, record in manifest["outputs"].items():
        require(Path(name).name == name, "Output manifest must use plain filenames")
        require(sha256_file(root / name) == record["sha256"], f"Checksum mismatch: {name}")

    rows = read_table(root / f"{corpus}_surprisal.csv")
    n = len(prepared)
    require(len(rows) == manifest["n_words"] == n, "Word count mismatch")
    require(rows["row_uid"].tolist() == prepared["row_uid"].tolist(), "Word row order mismatch")
    require(rows["word"].tolist() == prepared["word"].tolist(), "Word text mismatch")
    require(set(rows["extraction_id"]) == {manifest["extraction_id"]}, "Mixed extraction IDs")
    require(np.array_equal(rows["hidden_row_idx"], np.arange(n)), "Array row index mismatch")
    require(np.array_equal(rows["hidden_model_token_idx"], rows["end_token_idx"] + 1), "Hidden token index mismatch")
    for column in ["surprisal", "surprisal_matched_context"]:
        values = pd.to_numeric(rows[column], errors="raise").to_numpy(dtype=float)
        require(np.isfinite(values).all() and (values >= 0).all(), f"Invalid {column}")
    first = rows.loc[rows["word_position"] == 1]
    require(len(first) == manifest["n_texts"], "Missing first word or story")
    require(first["prefix_kind"].eq("bos").all() and first["prefix_model_token_idx"].eq(0).all(), "First-word prefix is not BOS")
    require(rows["prefix_available"].astype(str).str.lower().eq("true").all(), "Unavailable prefix")
    mapping = read_table(root / f"{corpus}_hidden_rows.csv")
    require(mapping["row_uid"].tolist() == rows["row_uid"].tolist(), "Hidden row mapping mismatch")
    require(mapping["hidden_row_idx"].tolist() == rows["hidden_row_idx"].tolist(), "Hidden mapping indices mismatch")

    tokens = read_table(root / f"{corpus}_token_windows.csv")
    require(set(tokens["extraction_id"]) == {manifest["extraction_id"]}, "Mixed token extraction IDs")
    require(set(tokens["row_uid"]) == set(rows["row_uid"]), "Token/word coverage mismatch")
    require(np.array_equal(tokens["model_token_idx"], tokens["token_idx"] + 1), "BOS offset mismatch")
    for label, config, column in [("surprisal", "surprisal", "surprisal"),
                                  ("hidden", "hidden", "surprisal_matched_context")]:
        context = tokens[f"{label}_left_context"].to_numpy()
        require(((context >= 1) & (context < manifest[config]["window_size"])).all(), f"Invalid {label} context length")
        require(np.array_equal(context, tokens["model_token_idx"] - tokens[f"{label}_window_start"]), f"Invalid {label} window indices")
        values = pd.to_numeric(tokens[column], errors="raise").to_numpy(dtype=float)
        require(np.isfinite(values).all() and (values >= 0).all(), f"Invalid token {column}")
        sums = tokens.assign(score=values).groupby("row_uid")["score"].sum().reindex(rows["row_uid"])
        require(np.allclose(sums, rows[column].to_numpy(dtype=float), rtol=1e-6, atol=1e-6), f"Token sums differ from word {column}")

    require(set(manifest["hook_files"]) == set(HOOK_FILES), "Unexpected recorded hooks")
    array_count = 0
    for hook, label in HOOK_FILES.items():
        require(manifest["hook_files"][hook]["file"] == f"{corpus}_hidden_{label}.npy", "Hook filename mismatch")
        kinds = ["hidden", "prefix_hidden"] if manifest["hidden"]["save_prefix"] else ["hidden"]
        for kind in kinds:
            name = f"{corpus}_{kind}_{label}.npy"
            require(name in manifest["outputs"], f"Unrecorded array: {name}")
            array = np.load(root / name, mmap_mode="r", allow_pickle=False)
            try:
                require(array.shape == (n, 768) and array.dtype == np.float32, f"Array shape/dtype mismatch: {name}")
                require(np.isfinite(array).all(), f"Nonfinite array: {name}")
            finally:
                array._mmap.close()
            array_count += 1
    return {"corpus": corpus, "RT": ",".join(sorted(set(rows["rt_measure"]))),
            "texts": len(first), "rows": n, "finite_scores": n,
            "first_words_scored": len(first), "arrays_checked": array_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["both", "provo", "natural_stories"], default="both")
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared_v2"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/surprisal_bos"))
    args = parser.parse_args()
    corpora = ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]
    summary = [validate_corpus(c, args.data_dir, args.results_dir) for c in corpora]
    print(pd.DataFrame(summary).to_string(index=False))
    print("Integrity checks passed. This does not validate RT prediction or SAE fidelity.")


if __name__ == "__main__":
    main()
