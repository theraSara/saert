"""Check the indexed sharing boundary and the portable report snapshot."""
from pathlib import Path
import json
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.report_bundle import verify_bundle

def main():
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    names=[n for n in names if n]
    forbidden=('data/','results/','naturalstories/','local_archive/','.presentation_build/','.chart-data-')
    binary_model={'.npy','.npz','.pkl','.pt','.pth','.safetensors'}
    total=0
    for name in names:
        path=ROOT/name
        if name.startswith(forbidden) or path.suffix in binary_model:
            raise ValueError(f'Local-only artifact is tracked: {name}')
        if not path.is_file():raise ValueError(f'Indexed path missing: {name}')
        total+=path.stat().st_size
        if path.stat().st_size>5_000_000:raise ValueError(f'Unexpectedly large shared file: {name}')
        if path.suffix in {'.md','.json','.csv','.ipynb','.tex','.txt','.yml'}:
            raw=path.read_text(encoding='utf-8-sig')
            if re.search(r'(?:hf_[A-Za-z0-9]{25,}|ghp_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})',raw):
                raise ValueError(f'Possible credential in {name}')
            if re.search(r'C:(?:\\\\|\\|/)Users(?:\\\\|\\|/)',raw):
                raise ValueError(f'Machine-specific user path in {name}')
        if path.suffix=='.ipynb':
            nb=json.loads(path.read_text(encoding='utf-8'))
            if any(o.get('output_type')=='error' for c in nb['cells'] for o in c.get('outputs',[])):
                raise ValueError(f'Failed notebook output: {name}')
    manifest=verify_bundle()
    print(f'{len(names)} tracked files, {total/1e6:.2f} MB. {len(manifest["files"])} report artifacts verified. Sharing boundary passed.')

if __name__=='__main__':main()
