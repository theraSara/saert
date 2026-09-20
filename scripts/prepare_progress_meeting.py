"""Prepare the dated progress deck from the portable report bundle, without raw data."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.report_bundle import verify_bundle,read_summary
from src.utils import PALETTE

def main():
    manifest=verify_bundle()
    sources={"primary":("sae_prediction","primary_results"),"scores":("sae_prediction","model_comparison"),
             "baseline":("baseline","model_comparison"),"sensitivity":("scaling_sensitivity","comparison")}
    data={key:read_summary(*source).to_dict(orient="records") for key,source in sources.items()}
    data["palette"]=PALETTE
    folder=ROOT/".presentation_build";folder.mkdir(exist_ok=True)
    (folder/"data.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Prepared presentation data from checked reports/")

if __name__=="__main__":main()
