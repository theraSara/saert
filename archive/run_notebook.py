"""Execute a portable report notebook using this interpreter, without fitting models."""
import argparse
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

NAMES = {"baseline":"01_surprisal_baseline.ipynb", "sae":"02_sae_reconstruction.ipynb",
         "regression":"03_sae_reading_time_prediction.ipynb", "scaling":"04_feature_scaling_robustness.ipynb",
         "latex":"90_export_latex_tables.ipynb"}

def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",choices=NAMES,default="baseline")
    parser.add_argument("--reports-dir",type=Path,default=root/"reports")
    args=parser.parse_args()
    path=root/"notebooks"/NAMES[args.analysis]
    notebook=nbformat.read(path,as_version=4)
    env=dict(os.environ,PYTHONNOUSERSITE="1",MPLBACKEND="Agg",SAERT_REPORTS_DIR=str(args.reports_dir.resolve()))
    client=NotebookClient(notebook,timeout=300,kernel_name="python3",resources={"metadata":{"path":str(root)}})
    manager=client.create_kernel_manager()
    manager.kernel_spec.argv=[sys.executable,"-s","-m","ipykernel_launcher","-f","{connection_file}"]
    client.km=manager
    client.execute(env=env)
    nbformat.write(notebook,path)
    print("Executed:",path.relative_to(root))

if __name__=="__main__":main()
