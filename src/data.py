
import argparse
import os
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import GPT2TokenizerFast
from wordfreq import word_frequency, zipf_frequency
from tqdm import tqdm
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROVO_DIR = DATA_DIR / "provo"
NS_DIR = DATA_DIR / "natural_stories"

print("Loading GPT-2 tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def get_word_frequency(word: str) -> float:
    return zipf_frequency(word.lower(), "en")

def get_unigram_frequency(word: str) -> float:
    freq = word_frequency(word.lower(), "en", wordlist="large")
    return np.log10(freq + 1e-10)  # avoid log(0)

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


def align_words_to_gpt2_tokens(words: list[str]) -> list[dict]:
    full_text, word_offsets = build_text_and_word_offsets(words)
    encoding = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_ids = encoding["input_ids"]
    token_offsets = encoding["offset_mapping"]

    aligned = []
    for word_idx, (word, (word_start, word_end)) in enumerate(zip(words, word_offsets)):
        tok_indices = [
            tok_idx
            for tok_idx, (tok_start, tok_end) in enumerate(token_offsets)
            if tok_end > tok_start and tok_start < word_end and tok_end > word_start
        ]
        if not tok_indices:
            print(f"    Warning: word {word_idx} ({word!r}) matched no tokens")
            start_token_idx = -1
            end_token_idx = -1
            span_ids = []
            span_strs = []
        else:
            start_token_idx = tok_indices[0]
            end_token_idx = tok_indices[-1]
            span_ids = token_ids[start_token_idx: end_token_idx + 1]
            span_strs = [tokenizer.decode([t]) for t in span_ids]
        aligned.append({
            "word":            word,
            "token_ids":       span_ids,
            "token_strs":      span_strs,
            "n_tokens":        len(tok_indices),
            "start_token_idx": start_token_idx,
            "end_token_idx":   end_token_idx,
        })
    return aligned

def load_provo() -> pd.DataFrame:
    et_path = PROVO_DIR / "Provo_Corpus-Eyetracking_Data.csv"
    if not et_path.exists():
        raise FileNotFoundError(
            f"Provo eye-tracking data not found at {et_path}\n"
            "Download from https://osf.io/sjefs/ and place in data/provo/"
        )

    print("Loading Provo corpus...")
    et = pd.read_csv(et_path)
    col_map = {
        "Text_ID": "text_id",
        "Word_Number": "word_position",
        "Word": "word",
        "IA_FIRST_FIXATION_DURATION": "FFD",
        "IA_GAZE_DURATION": "GZD",
        "IA_DWELL_TIME": "TRT",
        "Part_ID": "participant_id",
    }
    cols_present = {k: v for k, v in col_map.items() if k in et.columns}
    et = et[list(cols_present.keys())].rename(columns=cols_present)
    rt_cols = [c for c in ["FFD", "GZD", "TRT"] if c in et.columns]
    agg_dict = {c: "mean" for c in rt_cols}
    agg_dict["word"] = "first"
    provo = (
        et.groupby(["text_id", "word_position"])
        .agg(agg_dict)
        .reset_index()
    )
    provo["word_clean"] = provo["word"].apply(clean_word)

    print("  Adding frequency measures...")
    provo["zipf_freq"] = provo["word_clean"].apply(get_word_frequency)
    provo["log_unigram_freq"] = provo["word_clean"].apply(get_unigram_frequency)
    provo["word_length"] = provo["word_clean"].apply(len)
    provo["word_position"] = provo["word_position"].astype(int)

    print("  Aligning tokens per text...")
    provo = add_token_alignment_provo(provo)

    for col in rt_cols:
        if col in provo.columns:
            provo[f"log_{col}"] = np.log(provo[col].clip(lower=1))

    primary_rt = "GZD" if "GZD" in provo.columns else rt_cols[0]
    n_missing = provo[primary_rt].isna().sum()
    print(f"  Kept complete stimulus context; {n_missing} rows have missing RT")
    provo["corpus"] = "provo"
    provo["primary_RT"] = provo[primary_rt]
    provo["log_primary_RT"] = np.log(provo["primary_RT"].clip(lower=1))
    provo["row_uid"] = make_row_uid("provo", provo)
    print(f"  Provo prepared: {len(provo)} word-level observations across {provo['text_id'].nunique()} texts")

    return provo

def add_token_alignment_provo(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for text_id, group in tqdm(df.groupby("text_id"), desc="  Token alignment"):
        group = group.sort_values("word_position")
        words = group["word"].tolist()
        try:
            aligned = align_words_to_gpt2_tokens(words)
        except Exception as e:
            print(f"    Warning: tokenization failed for text {text_id}: {e}")
            aligned = [{"token_ids": [], "n_tokens": 1,
                        "start_token_idx": i, "end_token_idx": i} for i in range(len(words))]
        for i, row_dict in enumerate(aligned):
            results.append({
                "text_id": text_id,
                "word_position": group.iloc[i]["word_position"],
                "token_ids_str": str(row_dict["token_ids"]),  
                "n_tokens": row_dict["n_tokens"],
                "start_token_idx": row_dict["start_token_idx"],
                "end_token_idx": row_dict["end_token_idx"],
            })
    token_df = pd.DataFrame(results)
    df = df.merge(token_df, on=["text_id", "word_position"], how="left")
    return df

def load_natural_stories() -> pd.DataFrame:
    wordinfo_path = NS_DIR / "processed_wordinfo.tsv"
    tok_path = NS_DIR / "all_stories.tok"
    if not wordinfo_path.exists():
        raise FileNotFoundError(
            f"Natural Stories word info not found at {wordinfo_path}\n"
            "Expected file: processed_wordinfo.tsv\n"
        )
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Natural Stories token file not found at {tok_path}\n"
            "Expected file: all_stories.tok\n"
        )
    print("Loading Natural Stories corpus...")
    wordinfo = pd.read_csv(wordinfo_path, sep="\t")
    wordinfo.columns = [c.strip() for c in wordinfo.columns]

    tok = pd.read_csv(tok_path, sep="\t")
    tok.columns = [c.strip() for c in tok.columns]
    wordinfo_keys = set(zip(wordinfo["item"], wordinfo["zone"]))
    tok_keys = set(zip(tok["item"], tok["zone"]))
    only_in_tok = tok_keys - wordinfo_keys
    only_in_wordinfo = wordinfo_keys - tok_keys
    if only_in_tok:
        print(f"  Note: {len(only_in_tok)} positions in all_stories.tok have no RT")
    if only_in_wordinfo:
        print(f"  Warning: {len(only_in_wordinfo)} positions in processed_wordinfo.tsv "
              f"have no matching token in all_stories.tok ")

    ns = wordinfo.rename(columns={
        "item":        "text_id",
        "zone":        "word_position",
        "meanItemRT":  "RT",
        "sdItemRT":    "sdRT",
        "gmeanItemRT": "gRT",
        "gsdItemRT":   "gsdRT",
        "nItem":       "n_participants",
    })

    tok_ordered = (
        tok.rename(columns={"item": "text_id", "zone": "word_position"}).sort_values(["text_id", "word_position"])
    )
    tok_out_path = NS_DIR / "story_token_order.csv"
    tok_ordered.to_csv(tok_out_path, index=False)
    print(f"  Full story token order saved → {tok_out_path}")
    n_missing = ns["RT"].isna().sum()
    print(f"  Kept complete RT table; {n_missing} rows have missing RT")

    ns["word_clean"] = ns["word"].apply(clean_word)

    print("  Adding frequency measures...")
    ns["zipf_freq"] = ns["word_clean"].apply(get_word_frequency)
    ns["log_unigram_freq"] = ns["word_clean"].apply(get_unigram_frequency)
    ns["word_length"] = ns["word_clean"].apply(len)

    print("  Aligning tokens per story...")
    ns = add_token_alignment_ns(ns, tok_ordered)
    ns["log_RT"] = np.log(ns["RT"].clip(lower=1))
    ns["log_gRT"] = np.log(ns["gRT"].clip(lower=1))
    ns["corpus"] = "natural_stories"
    ns["primary_RT"] = ns["RT"]         
    ns["log_primary_RT"] = ns["log_RT"]
    ns["row_uid"] = make_row_uid("natural_stories", ns)

    print(f"  Natural Stories prepared: {len(ns):,} word-level observations across {ns['text_id'].nunique()} stories")

    return ns

def add_token_alignment_ns(df: pd.DataFrame, tok_ordered: pd.DataFrame) -> pd.DataFrame:
    results = []
    for text_id, full_group in tqdm(
        tok_ordered.groupby("text_id"), desc="  Token alignment"
    ):
        full_group = full_group.sort_values("word_position")
        all_words = full_group["word"].fillna("").tolist()
        all_positions = full_group["word_position"].tolist()

        try:
            aligned = align_words_to_gpt2_tokens(all_words)
        except Exception as e:
            print(f"    Warning: tokenization failed for story {text_id}: {e}")
            aligned = [{"token_ids": [], "n_tokens": 1,
                        "start_token_idx": i, "end_token_idx": i} for i in range(len(all_words))]

        for i, row_dict in enumerate(aligned):
            results.append({
                "text_id": text_id,
                "word_position": all_positions[i],
                "token_ids_str": str(row_dict["token_ids"]),
                "n_tokens": row_dict["n_tokens"],
                "start_token_idx": row_dict["start_token_idx"],
                "end_token_idx": row_dict["end_token_idx"],
            })

    token_df = pd.DataFrame(results)
    df = df.merge(token_df, on=["text_id", "word_position"], how="left")
    return df

def clean_word(word: str) -> str:
    if not isinstance(word, str):
        return ""
    word = re.sub(r"^[^\w']+|[^\w']+$", "", word)
    return word.lower()


def make_row_uid(corpus: str, df: pd.DataFrame) -> pd.Series:
    return (
        corpus
        + ":"
        + df["text_id"].astype(str)
        + ":"
        + df["word_position"].astype(int).astype(str)
    )

def print_summary(df: pd.DataFrame, corpus_name: str):
    print(f"  {corpus_name.upper()} — Summary")
    print(f"  Rows (word observations): {len(df):,}")
    print(f"  Unique texts/stories:     {df['text_id'].nunique()}")
    print(f"  Unique words (types):     {df['word_clean'].nunique():,}")
    print(f"  Mean RT (ms):             {df['primary_RT'].mean():.1f}")
    print(f"  Median RT (ms):           {df['primary_RT'].median():.1f}")
    print(f"  Mean Zipf frequency:      {df['zipf_freq'].mean():.2f}")
    print(f"  Multi-token words:        {(df['n_tokens'] > 1).sum():,} ({100*(df['n_tokens'] > 1).mean():.1f}%)")
    print(f"  Missing RT:               {df['primary_RT'].isna().sum()}")
    if "row_uid" in df.columns:
        print(f"  Unique row ids:           {df['row_uid'].is_unique}")

def main():
    parser = argparse.ArgumentParser(description="SparseRT — Step 01: Data Preparation")
    parser.add_argument(
        "--corpus",
        choices=["provo", "natural_stories", "both"],
        default="both"
    )
    args = parser.parse_args()

    if args.corpus in ("provo", "both"):
        try:
            provo = load_provo()
            print_summary(provo, "provo")
            out_path = PROVO_DIR / "provo_prepared.csv"
            provo.to_csv(out_path, index=False)
            print(f"  Saved → {out_path}\n")
        except FileNotFoundError as e:
            print(f"\n[SKIP] {e}\n")

    if args.corpus in ("natural_stories", "both"):
        try:
            ns = load_natural_stories()
            print_summary(ns, "natural_stories")
            out_path = NS_DIR / "natural_stories_prepared.csv"
            ns.to_csv(out_path, index=False)
            print(f"  Saved → {out_path}\n")
        except FileNotFoundError as e:
            print(f"\n[SKIP] {e}\n")

    print("Data preparation complete.")

if __name__ == "__main__":
    main()
