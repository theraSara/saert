import argparse, json, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
FI_DIR = ROOT / "results" / "feature_interpretation"
REG_DIR = ROOT / "results" / "sae_regression_v2"
CACHE_DIR = ROOT / "results" / "neuronpedia_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Neuronpedia API
# Endpoint: GET https://www.neuronpedia.org/api/feature/{model_id}/{layer}-res-jb/{feature_id}
# model_id : gpt2-small
# layer    : 0-indexed integer (our L01 = 0, L08 = 7, L12 = 11)
# feature_id: integer
# Returns JSON file with the following:
#   explanations[].description  — GPT-4 / Claude auto-interp explanation
#   explanations[].typeName     — method used (e.g. "oai_token-act-pair")
#   pos_str                     — top positive logit tokens
#   neg_str                     — top negative logit tokens
#   activations_density         — fraction of tokens that activate this feature

# Usage
#     python src/neuronpedia_fetch.py --layers 1 8  # Fetch only for the best hooks (fast: ~4 min)
#     python src/neuronpedia_fetch.py --layers all  # Fetch all layers (full: ~36 min, run once)
#     python src/neuronpedia_fetch.py --corpus provo --layers 1  # Specific corpus only
#     python src/neuronpedia_fetch.py --layers 1 8 --dry-run   # Dry run (show what would be fetched, no API calls)

MODEL_ID = "gpt2-small"
BASE_URL = "https://www.neuronpedia.org/api/feature"
SLEEP_BETWEEN = 1.5

ALL_LAYERS = [f"L{n:02d}" for n in range(1, 13)]
KINDS = ["post", "prefix"]
BEST_HOOKS = {
    "provo": "L01",
    "natural_stories": "L08"
}

# for layer label (1-indexed in the pipeline) -> Neuronpedia 0-indexed layer
def hook_label_to_np_layer(hook_label: str) -> int:
    # convert the layer labels L01 -> 0, L08 -> 7
    return int(hook_label[1:]) - 1

def cache_path(hook_label: str, feature_id: int) -> Path:
    layer = hook_label_to_np_layer(hook_label)
    source = f"{layer}-res-jb"
    return CACHE_DIR / MODEL_ID / source / f"{feature_id}.json"

def load_cached(hook_label: str, feature_id: int) -> dict | None:
    p = cache_path(hook_label, feature_id)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def save_cached(hook_label: str, feature_id: int, data: dict):
    p = cache_path(hook_label, feature_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f: 
        json.dump(data, f, indent=2)

def fetch_feature(hook_label:str, feature_id: int, dry_run: bool = False) -> dict | None:
    cached = load_cached(hook_label, feature_id)
    if cached is not None:
        return cached
    layer = hook_label_to_np_layer(hook_label)
    source = f"{layer}-res-jb"
    url = f"{BASE_URL}/{MODEL_ID}/{source}/{feature_id}"
    if dry_run:
        print(f"DRY RUN would fetch: {url}")
        return None
    
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "SparseRT-Research/1.0"})
        if r.status_code == 200:
            data = r.json()
            save_cached(hook_label, feature_id, data)
            return data
        elif r.status_code == 404:
            # feature not found (empty result in the cache)
            save_cached(hook_label, feature_id, {"error": "not found", "feature_id": feature_id})
            return None
        else: 
            print(f" HTTP {r.status_code} for {url}")
            return None 
    except Exception as e:
        print(f" Error fetching {url}: {e}")
        return None
    
def parse_response(data: dict) -> dict:
    if data is None or "error" in data:
        return {
            "np_explanation": "",
            "np_explanation_model": "",
            "np_layer": "",
            "np_pos_logits": "",
            "np_neg_logits": "",
            "np_density": float("nan"),
            "np_url": "",
        }
    explanations = data.get("explanations", [])
    best_exp = ""
    best_model = ""
    if explanations:
        for exp in explanations:
            if "gpt-4" in exp.get("typeName", "").lower():
                best_exp = exp.get("description", "")
                best_model = exp.get("typeName", "")
                break 
        if not best_exp and explanations:
            best_exp = explanations[0].get("description", "")
            best_model = explanations[0].get("typeName", "")
    
    # top tokens positive/negaative logits
    pos_logits = data.get("pos_str", "")
    neg_logts = data.get("neg_str", "")

    density = data.get("activation_density", float("nan"))
    np_layer = data.get("layer", "")
    np_index = data.get("index", data.get("feature_id", ""))
    source = data.get("sourceSet", {}).get("id", "") if isinstance(data.get("sourceSet"), dict) else ""
    np_url = (f"https://www.neuronpedia.org/{MODEL_ID}/{source}/{np_index}" if source else "")

    return {
            "np_explanation": best_exp,
            "np_explanation_model": best_model,
            "np_layer": np_layer,
            "np_pos_logits": pos_logits,
            "np_neg_logits": neg_logts,
            "np_density": density,
            "np_url": np_url,
        }

def load_feature_ids(corpus: str, hook_label: str, kind: str) -> list[int]:
    path = FI_DIR / f"{corpus}_{hook_label}_{kind}_top_features.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df["feature_id"].astype(int).tolist()

def fetch_and_merge(corpus: str, hook_label: str, kind: str, dry_run: bool = False) -> pd.DataFrame | None:
    feature_ids = load_feature_ids(corpus, hook_label, kind)
    if not feature_ids:
        return None
    
    print(f"\n {corpus} | {hook_label} | {kind} - {len(feature_ids)} features")

    rows = []
    cached_count = 0
    fetched_count = 0

    for fid in feature_ids:
        cached = load_cached(hook_label, fid)
        if cached is not None:
            cached_count += 1
            parsed = parse_response(cached)
        else:
            parsed = parse_response(fetch_feature(hook_label, fid, dry_run))
            fetched_count += 1
            if not dry_run:
                time.sleep(SLEEP_BETWEEN)

        rows.append({"feature_id": fid, **parsed})

    print(f"Cached: {cached_count} | Fetched: {fetched_count}")

    csv_path = FI_DIR / f"{corpus}_{hook_label}_{kind}_top_features.csv"
    if not csv_path.exists():
        return pd.DataFrame(rows)
    
    df = pd.read_csv(csv_path)
    np_df = pd.DataFrame(rows)
    df = df.merge(np_df, on="feature_id", how="left")
    df.to_csv(csv_path, index=False)
    
    out_path = FI_DIR / f"{corpus}_{hook_label}_{kind}_neuronpedia.csv"
    np_df.to_csv(out_path, index=False)

    print(f"\n Neuronpedia explanations (feature_id -> GPT-4 description):")
    for _, row in np_df.iterrows():
        exp = row["np_explanation"][:70] if row["np_explanation"] else "- no explanation -"
        url = row["np_url"]
        print(f" F{int(row['feature_id']):>6}: {exp}")

        if url:
            print(f" {url}")
    
    return df 

def main():
    parser = argparse.ArgumentParser(description="SparseRT - Fetch Neuronpedia feature explanations")
    parser.add_argument("--corpus", choices=["provo", "natural_stories", "both"], default="both",)
    parser.add_argument("--layers", nargs="+", default=["1", "8"],
        help="Layers to fetch. Use 'all' for all 12, or integers like '1 8'. Default: '1 8' (best hooks from inner validation).",
    )
    parser.add_argument("--kinds", nargs="+", choices=["post", "prefix"], default=["post", "prefix"],
    )
    parser.add_argument("--dry-run", action="store_true",
        help="Print what would be fetched without making API calls.",
    )
    args = parser.parse_args()

    if args.layers == ["all"]:
        hook_labels = ALL_LAYERS
    else:
        hook_labels = [f"L{int(l):02d}" for l in args.layers]

    corpora = (
        ["provo", "natural_stories"] if args.corpus == "both" else [args.corpus]
    )

    print(f"Neuronpedia fetch — SparseRT")
    print(f"  Model:   {MODEL_ID}")
    print(f"  Corpora: {corpora}")
    print(f"  Hooks:   {hook_labels}")
    print(f"  Kinds:   {args.kinds}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Sleep between requests: {SLEEP_BETWEEN}s")
    print(f"  Cache: {CACHE_DIR}")
    print()

    total = 0
    for corpus in corpora:
        for hook_label in hook_labels:
            for kind in args.kinds:
                result = fetch_and_merge(corpus, hook_label, kind, args.dry_run)
                if result is not None:
                    total += len(result)

    print(f"\nDone. Total features processed: {total}")
    print(f"Results saved to: {FI_DIR}/")
    print(f"Cache saved to:   {CACHE_DIR}/")
    print()
    print("Next steps:")
    print("  1. Open results/feature_interpretation/*_neuronpedia.csv")
    print("  2. The 'np_explanation' column has GPT-4 descriptions — use these")
    print("     as your authoritative labels instead of the logit-lens labels")
    print("  3. The 'np_url' column links directly to the Neuronpedia dashboard")
    print("     for visual inspection of each feature")

if __name__ == "__main__":
    main()
