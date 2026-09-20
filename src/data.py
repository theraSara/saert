"""Prepare complete stimuli and aggregate RT observations without changing context."""

import argparse
import json
from pathlib import Path
import re
import tempfile
import uuid
import numpy as np
import pandas as pd

try:
    from .pipeline_utils import (KEYS, file_record, normalize_words, numeric, read_table,
                                 run_metadata, tokenize_words, tokenizer_hash,
                                 validate_word_matches, write_json)
except ImportError:
    from pipeline_utils import (KEYS, file_record, normalize_words, numeric, read_table,
                                run_metadata, tokenize_words, tokenizer_hash,
                                validate_word_matches, write_json)

ROOT = Path(__file__).resolve().parent.parent
RT_COLUMNS = {"FFD": "IA_FIRST_FIXATION_DURATION", "GZD": "IA_FIRST_RUN_DWELL_TIME",
              "TRT": "IA_DWELL_TIME"}


def clean_word(word):
    return re.sub(r"^[^\w']+|[^\w']+$", "", word).lower()


def load_stimuli(path):
    # Norms files can repeat words; conflicting forms must not be collapsed.
    raw = normalize_words(read_table(path), str(path), unique=False)
    raw = raw[KEYS + ["word"]].drop_duplicates()
    return normalize_words(raw, str(path), complete=True)


def prepare_provo(stimulus_path, rt_path, measure, rt_column=None, participant_column=None,
                  rt_min=None, rt_max=None):
    raw = read_table(rt_path)
    has_regions = {"IA_ID", "IA_LABEL"}.issubset(raw.columns)
    if has_regions:
        # Norm-based Word_Number/Word can be missing or differ from the actual
        # eye-tracking interest areas. Preserve them, but never use them as RT keys.
        raw = raw.rename(columns={"Word_Number": "norm_word_position", "Word": "norm_word"})
        raw["word_position"] = raw["IA_ID"]
        raw["word"] = raw["IA_LABEL"].str.strip()
    trials = normalize_words(raw, "Provo eye tracking", unique=False)
    if stimulus_path is None:
        if not has_regions:
            raise ValueError("Provo needs IA_ID/IA_LABEL or an explicit complete stimulus table")
        stimuli = normalize_words(trials[KEYS + ["word"]].drop_duplicates(),
                                  "Provo presented interest areas", complete=True)
    else:
        stimuli = load_stimuli(stimulus_path)
    if participant_column is None:
        participant_column = "Participant_ID" if "Participant_ID" in trials else "Part_ID"
    validate_word_matches(stimuli, trials, "Provo")
    column = rt_column or RT_COLUMNS[measure]
    if column not in trials:
        raise ValueError(f"Requested {measure} requires {column!r}; available columns: "
                         f"{list(trials.columns)}. No alternative RT measure will be substituted.")
    if participant_column not in trials or trials[participant_column].astype(str).str.strip().eq("").any():
        raise ValueError(f"Missing participant IDs in {participant_column!r}")
    if trials.duplicated([participant_column] + KEYS).any():
        raise ValueError("Duplicate Provo participant x word rows; resolve before aggregation")
    raw_rt = numeric(trials[column], column)
    reason = pd.Series("included", index=trials.index)
    reason.loc[raw_rt.isna()] = "missing"
    reason.loc[raw_rt.le(0)] = "nonpositive"
    if rt_min is not None:
        reason.loc[raw_rt.gt(0) & raw_rt.lt(rt_min)] = "below_min"
    if rt_max is not None:
        reason.loc[raw_rt.gt(rt_max)] = "above_max"
    trials["participant_id"] = trials[participant_column].astype(str)
    trials["raw_RT"] = raw_rt
    trials["rt_status"] = reason
    trials["analysis_RT"] = raw_rt.where(reason.eq("included"))
    trials["log_analysis_RT"] = np.log(trials["analysis_RT"])
    aggregate = trials.groupby(KEYS, as_index=False).agg(
        primary_RT=("analysis_RT", "mean"), n_participants=("analysis_RT", "count"),
        sdRT=("analysis_RT", "std"), mean_log_RT=("log_analysis_RT", "mean"),
        n_trials=("raw_RT", "size"))
    prepared = stimuli.merge(aggregate, on=KEYS, how="left", validate="one_to_one")
    for col in ["n_participants", "n_trials"]:
        prepared[col] = prepared[col].fillna(0).astype(int)
    prepared[measure] = prepared["primary_RT"]
    prepared["rt_measure"] = measure
    prepared["rt_source_column"] = column
    prepared["stimulus_unit"] = "interest_area" if has_regions else "word"
    trials["rt_measure"] = measure
    inputs = {"rt": file_record(rt_path)}
    if stimulus_path is not None:
        inputs["stimuli"] = file_record(stimulus_path)
    return prepared, trials, inputs, {
        "measure": measure, "source_column": column, "rt_min_ms": rt_min, "rt_max_ms": rt_max,
        "participant_column": participant_column,
        "stimulus_source": "external table" if stimulus_path is not None else "unique Text_ID/IA_ID/IA_LABEL from eye-tracking file",
        "position_source": "IA_ID" if has_regions else "Word_Number or word_position",
        "word_source": "IA_LABEL, surrounding whitespace stripped" if has_regions else "Word or word",
        "norm_metadata": "Original Word_Number and Word retained as norm_word_position and norm_word in participant output" if has_regions else "unchanged",
        "trial_status_counts": reason.value_counts().to_dict(),
        "participant_exclusions": "No new participant-quality exclusions; source rows retained",
        "skipping": "Nonpositive/missing durations excluded from duration means; stimuli retained",
    }


def prepare_natural_stories(directory):
    directory = Path(directory)
    tok_path, info_path = directory / "all_stories.tok", directory / "processed_wordinfo.tsv"
    stimuli = load_stimuli(tok_path)
    info = normalize_words(read_table(info_path), "Natural Stories wordinfo")
    validate_word_matches(stimuli, info, "Natural Stories")
    if not {"meanItemRT", "nItem"}.issubset(info):
        raise ValueError("Natural Stories requires meanItemRT and nItem")
    info = info.rename(columns={"meanItemRT": "primary_RT", "nItem": "n_participants",
                                "sdItemRT": "sdRT", "gmeanItemRT": "gRT", "gsdItemRT": "gsdRT"})
    for col in ["primary_RT", "n_participants", "sdRT", "gRT", "gsdRT"]:
        if col in info:
            info[col] = numeric(info[col], col)
    bad = info["primary_RT"].notna() & (info["primary_RT"].le(0) | info["n_participants"].isna()
                                       | info["n_participants"].le(0))
    if bad.any():
        raise ValueError("Natural Stories contains invalid aggregate RT/counts; check source release")
    counts = info["n_participants"].dropna()
    if (counts < 0).any() or (counts % 1 != 0).any():
        raise ValueError("Natural Stories nItem must contain nonnegative integer counts")
    prepared = stimuli.merge(info.drop(columns="word"), on=KEYS, how="left", validate="one_to_one")
    prepared["RT"] = prepared["primary_RT"]
    prepared["rt_measure"] = "SPR"
    prepared["rt_source_column"] = "meanItemRT"
    inputs = {"stimuli": file_record(tok_path), "wordinfo": file_record(info_path)}
    available = [p for p in [directory / "processed_RTs.tsv", directory / "processed_RT.tsv"] if p.exists()]
    if len(available) > 1:
        raise ValueError("Both processed_RT.tsv and processed_RTs.tsv exist; keep one authoritative file")
    trials = None
    if available:
        trials = normalize_words(read_table(available[0]), "Natural Stories participant RTs", unique=False)
        validate_word_matches(stimuli, trials, "Natural Stories participants")
        if not {"WorkerId", "RT"}.issubset(trials):
            raise ValueError("Participant file must contain WorkerId and RT")
        trials["participant_id"] = trials["WorkerId"].astype(str)
        if trials["participant_id"].str.strip().eq("").any() or trials.duplicated(["participant_id"] + KEYS).any():
            raise ValueError("Missing or duplicate Natural Stories participant x word IDs")
        trials["raw_RT"] = numeric(trials["RT"], "participant RT")
        trials["rt_measure"] = "SPR"
        inputs["participant_rt"] = file_record(available[0])
    return prepared, trials, inputs, {
        "measure": "SPR", "source_column": "meanItemRT",
        "aggregation": "Upstream meanItemRT retained; participant file not reaggregated",
        "participant_data_available": trials is not None,
        "source_note": "Use the corrected post-2021 Natural Stories release; hashes identify local inputs",
    }


def add_predictors_and_alignment(prepared, corpus, tokenizer, frequency_fn=None):
    if frequency_fn is None:
        from wordfreq import zipf_frequency
        frequency_fn = lambda w: zipf_frequency(w, "en", wordlist="large")
    df = normalize_words(prepared, corpus, complete=True)
    df["word_clean"] = df["word"].map(clean_word)
    df["zipf_freq"] = df["word_clean"].map(frequency_fn)
    df["word_length"] = df["word_clean"].str.len()
    df["is_sentence_final"] = df["word"].str.contains(r'''[.!?]["'”’)]*$''', regex=True).astype(int)
    df["corpus"] = corpus
    df["row_uid"] = corpus + ":" + df["text_id"].astype(str) + ":" + df["word_position"].astype(str)
    df["log_primary_RT"] = np.log(df["primary_RT"].where(df["primary_RT"].gt(0)))
    df["rt_aggregation"] = "log_of_arithmetic_mean_ms"
    records = []
    for _, group in df.groupby("text_id", sort=True):
        _, ids, _, spans = tokenize_words(group["word"].tolist(), tokenizer)
        for uid, (start, end) in zip(group["row_uid"], spans):
            records.append({"row_uid": uid, "start_token_idx": start, "end_token_idx": end,
                            "n_tokens": end - start + 1, "token_ids_str": json.dumps(ids[start:end+1])})
    return df.merge(pd.DataFrame(records), on="row_uid", validate="one_to_one", how="left")


def save_prepared(df, trials, corpus, output, inputs, settings, tokenizer, revision, overwrite=False):
    output = Path(output)
    names = [f"{corpus}_prepared.csv", "story_token_order.csv", f"{corpus}_prepared_manifest.json"]
    participant_name = f"{corpus}_participants.csv"
    if trials is None and (output / participant_name).exists():
        raise ValueError("Existing participant output but no participant input; use a fresh output directory")
    if trials is not None:
        names.append(participant_name)
    if not overwrite and any((output / name).exists() for name in names):
        raise FileExistsError(f"Prepared output exists in {output}; use a fresh --output-dir or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    df = df.copy()
    df["preparation_id"] = run_id
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as temp:
        temp = Path(temp)
        df.to_csv(temp / names[0], index=False)
        df[KEYS + ["word", "row_uid"]].to_csv(temp / names[1], index=False)
        if trials is not None:
            trials.to_csv(temp / participant_name, index=False)
        outputs = {p.name: {"sha256": file_record(p)["sha256"]} for p in temp.iterdir()}
        manifest = {**run_metadata(), "schema_version": 2, "preparation_id": run_id,
                    "corpus": corpus, "inputs": inputs, "outputs": outputs, "settings": settings,
                    "source_code": [file_record(__file__), file_record(Path(__file__).with_name("pipeline_utils.py"))],
                    "tokenizer": {"name": "gpt2", "requested_revision": revision,
                                  "backend_sha256": tokenizer_hash(tokenizer)},
                    "text_reconstruction": "Corpus words joined by one ASCII space; case/punctuation preserved",
                    "n_words": len(df), "n_texts": int(df["text_id"].nunique()),
                    "n_missing_rt": int(df["primary_RT"].isna().sum())}
        write_json(temp / names[2], manifest)
        for name in names:
            if name != names[2]:
                (temp / name).replace(output / name)
        (temp / names[2]).replace(output / names[2])
    print(f"{corpus}: {len(df):,} stimulus words, {df['primary_RT'].notna().sum():,} RT means -> {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["provo", "natural_stories", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, help="Prepared-data root; one subdirectory per corpus")
    parser.add_argument("--provo-stimuli", type=Path, help="Optional complete stimulus table keyed by IA_ID; default derives regions from official IA_ID/IA_LABEL")
    parser.add_argument("--provo-rt", choices=list(RT_COLUMNS), help="Required for Provo; no automatic fallback")
    parser.add_argument("--provo-rt-column", help="Explicit override after consulting the data dictionary")
    parser.add_argument("--provo-participant-column", help="Default: Participant_ID, with Part_ID fallback for legacy input")
    parser.add_argument("--provo-rt-min", type=float)
    parser.add_argument("--provo-rt-max", type=float)
    parser.add_argument("--tokenizer-revision", default="main", help="Use a commit SHA for reproducibility")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    corpora = ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]
    if "provo" in corpora and args.provo_rt is None:
        parser.error("Provo requires --provo-rt (FFD, GZD, or TRT)")
    bounds = [x for x in [args.provo_rt_min, args.provo_rt_max] if x is not None]
    if any(not np.isfinite(x) or x <= 0 for x in bounds) or (len(bounds) == 2 and bounds[0] > bounds[1]):
        parser.error("RT bounds must be positive, finite, and min <= max")
    loaded = {}
    for corpus in corpora:
        if corpus == "provo":
            loaded[corpus] = prepare_provo(args.provo_stimuli,
                args.data_dir / "provo" / "Provo_Corpus-Eyetracking_Data.csv", args.provo_rt,
                args.provo_rt_column, args.provo_participant_column, args.provo_rt_min, args.provo_rt_max)
        else:
            loaded[corpus] = prepare_natural_stories(args.data_dir / corpus)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", revision=args.tokenizer_revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with offsets is required")
    for corpus in corpora:
        df, trials, inputs, settings = loaded[corpus]
        df = add_predictors_and_alignment(df, corpus, tokenizer)
        save_prepared(df, trials, corpus, (args.output_dir or args.data_dir) / corpus,
                      inputs, settings, tokenizer, args.tokenizer_revision, args.overwrite)


if __name__ == "__main__":
    main()
