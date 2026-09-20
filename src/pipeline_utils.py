from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd

KEYS = ["text_id", "word_position"]
ALIASES = {"Text_ID": "text_id", "Word_Number": "word_position", "Word": "word",
           "item": "text_id", "zone": "word_position"}


def read_table(path):
    path = Path(path)
    return pd.read_csv(path, sep="\t" if path.suffix in {".tsv", ".tok"} else ",",
                       keep_default_na=False, encoding="utf-8-sig")


def normalize_words(frame, label, *, complete=False, unique=True):
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns})
    if not df.columns.is_unique:
        raise ValueError(f"{label}: duplicate/ambiguous column names")
    missing = set(KEYS + ["word"]) - set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    for col in KEYS:
        values = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(values).all() or (values < 1).any() or (values % 1 != 0).any():
            raise ValueError(f"{label}: {col} must contain positive integer IDs")
        df[col] = values.astype("int64")
    if df.empty:
        raise ValueError(f"{label}: empty table")
    if df["word"].map(lambda w: not isinstance(w, str) or not w or w != w.strip()).any():
        raise ValueError(f"{label}: missing words or leading/trailing whitespace")
    if unique and df.duplicated(KEYS).any():
        raise ValueError(f"{label}: duplicate word keys")
    if complete:
        for text_id, group in df.groupby("text_id", sort=True):
            positions = sorted(group["word_position"].tolist())
            if positions != list(range(1, len(positions) + 1)):
                missing = sorted(set(range(1, max(positions) + 1)) - set(positions))
                raise ValueError(f"{label}: text {text_id} is incomplete; missing positions "
                                 f"{missing[:12]}. Use complete presented stimuli, not RT rows.")
    return df.sort_values(KEYS, kind="stable").reset_index(drop=True)


def validate_word_matches(stimuli, observations, label):
    lookup = stimuli[KEYS + ["word"]].rename(columns={"word": "stimulus_word"})
    joined = observations.merge(lookup, on=KEYS, how="left", validate="many_to_one", indicator=True)
    bad = (joined["_merge"] != "both") | (joined["word"] != joined["stimulus_word"])
    if bad.any():
        examples = joined.loc[bad, KEYS + ["word", "stimulus_word"]].head().to_dict("records")
        raise ValueError(f"{label}: stimulus/observation mismatch: {examples}")


def numeric(series, label):
    values = series.replace({"": np.nan, ".": np.nan, "NA": np.nan, "NaN": np.nan,
                             "nan": np.nan, "N/A": np.nan})
    try:
        values = pd.to_numeric(values, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid numeric values in {label}") from exc
    if np.isinf(values).any():
        raise ValueError(f"Infinite values in {label}")
    return values


def build_text_and_word_offsets(words):
    parts, spans, cursor = [], [], 0
    for i, word in enumerate(words):
        if not isinstance(word, str) or not word or word != word.strip():
            raise ValueError(f"Invalid stimulus word at index {i}: {word!r}")
        if i:
            parts.append(" ")
            cursor += 1
        spans.append((cursor, cursor + len(word)))
        parts.append(word)
        cursor += len(word)
    if not parts:
        raise ValueError("Cannot tokenize an empty stimulus")
    return "".join(parts), spans


def tokenize_words(words, tokenizer):
    text, word_offsets = build_text_and_word_offsets(words)
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False,
                         truncation=False)
    ids = list(encoding["input_ids"])
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    if len(ids) != len(offsets) or not ids:
        raise ValueError("Tokenizer returned inconsistent IDs/offsets")
    if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != text:
        raise ValueError("Tokenizer round trip does not reproduce the exact constructed stimulus")
    owners = {}
    for word_idx, (left, right) in enumerate(word_offsets):
        hits = [i for i, (a, b) in enumerate(offsets) if b > a and a < right and b > left]
        if not hits or hits != list(range(hits[0], hits[-1] + 1)):
            raise ValueError(f"Noncontiguous or empty token span for word {word_idx}")
        for i in hits:
            if i in owners:
                raise ValueError(f"Token {i} overlaps multiple corpus words")
            owners[i] = word_idx
    for i, (a, b) in enumerate(offsets):
        if i not in owners:
            if b <= a or not text[a:b].isspace():
                raise ValueError(f"Unassigned non-whitespace token {i}: offset {(a, b)}")
            candidates = [j for j, (left, _) in enumerate(word_offsets) if left >= b]
            if not candidates:
                raise ValueError("Unassigned trailing whitespace token")
            owners[i] = candidates[0]
    spans = []
    for j in range(len(words)):
        hits = sorted(i for i, owner in owners.items() if owner == j)
        if hits != list(range(hits[0], hits[-1] + 1)):
            raise ValueError(f"Noncontiguous final token assignment for word {j}")
        spans.append((hits[0], hits[-1]))
    if [i for a, b in spans for i in range(a, b + 1)] != list(range(len(ids))):
        raise ValueError("Word spans do not partition the model token sequence")
    return text, ids, offsets, spans


def window_plan(n_tokens, window_size, stride):
    if not 2 <= window_size <= 1024 or not 1 <= stride < window_size:
        raise ValueError("Require 2 <= window size <= 1024 and 1 <= stride < window size")
    if n_tokens < 1:
        raise ValueError("Empty token sequence")
    start, claimed = 0, 0
    while claimed < n_tokens:
        end = min(start + window_size, n_tokens)
        yield start, end, claimed, end
        claimed = end
        start += stride


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def tokenizer_hash(tokenizer):
    return hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest()


def run_metadata():
    versions = {}
    for name in ["numpy", "pandas", "torch", "transformers", "transformer-lens", "sae-lens", "wordfreq"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(), "versions": versions}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")
