"""Export an explicit allowlist of aggregate reports; never copy full results trees."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.report_bundle import digest, verify_bundle

SECTIONS = {
 'baseline': ('results/baseline_bos/diagnostics',
   ['model_comparison.csv','paired_contrasts.csv','corpus_quality.csv','collinearity.csv','per_text_heterogeneity.csv','heldout_improvement.png','heldout_improvement.pdf']),
 'sae_fidelity': ('results/sae_bos/diagnostics',
   ['behavior_summary.csv','geometry_summary.csv','sae_fidelity.png','sae_fidelity.pdf']),
 'sae_prediction': ('results/sae_regression_v2/diagnostics',
   ['primary_results.csv','model_comparison.csv','sae_vs_dense.csv','feature_shift_audit.csv','heldout_representation_gain.png','heldout_representation_gain.pdf','layer_gain_heatmap.png','layer_gain_heatmap.pdf']),
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    output=ROOT/'reports'
    if args.verify_only:
        m=verify_bundle(output); print(f'Verified {len(m["files"])} portable artifacts'); return
    files={}
    def register(target,source,transformation='none'):
        if target.suffix in {'.csv','.json','.md','.tex'}:
            target.write_bytes(target.read_bytes().replace(b'\r\n',b'\n'))
            transformation = 'LF line endings' if transformation == 'none' else transformation+'; LF line endings'
        files[target.relative_to(output).as_posix()]={'sha256':digest(target),'bytes':target.stat().st_size,
            'source':source.relative_to(ROOT).as_posix(),'source_sha256':digest(source),'transformation':transformation}
    for section,(directory,names) in SECTIONS.items():
        source=ROOT/directory
        m=json.loads((source/'diagnostics_manifest.json').read_text(encoding='utf-8'))
        dest=output/section;dest.mkdir(parents=True,exist_ok=True)
        for name in names:
            p=source/name
            if digest(p)!=m['outputs'][name]['sha256']:raise ValueError(f'Changed source: {p}')
            target=dest/name;shutil.copyfile(p,target);register(target,p)
    source=ROOT/'results/sae_regression_v2/diagnostics'
    m=json.loads((source/'diagnostics_manifest.json').read_text())
    for name in ['selected_hooks.csv','tuning.csv']:
        if digest(source/name)!=m['outputs'][name]['sha256']:raise ValueError(name)
    hooks=pd.read_csv(source/'selected_hooks.csv').groupby(['corpus','kind','representation','label']).size().reset_index(name='fold_count')
    target=output/'sae_prediction/hook_selection.csv';hooks.to_csv(target,index=False);register(target,source/'selected_hooks.csv','Grouped fold counts')
    tuning=pd.read_csv(source/'tuning.csv').groupby(['corpus','kind','representation']).agg(fits=('alpha','size'),
        smallest_alpha_choices=('alpha',lambda x:int((x==.01).sum())),largest_alpha_choices=('alpha',lambda x:int((x==100).sum()))).reset_index()
    target=output/'sae_prediction/regularization_summary.csv';tuning.to_csv(target,index=False);register(target,source/'tuning.csv','Grouped grid-boundary counts')
    source=ROOT/'results/scaling_sensitivity_v1'
    m=json.loads((source/'manifest.json').read_text())
    if digest(source/'comparison.csv')!=m['summary']['sha256']:raise ValueError('Changed scaling summary')
    dest=output/'scaling_sensitivity';dest.mkdir(exist_ok=True)
    shutil.copyfile(source/'comparison.csv',dest/'comparison.csv');register(dest/'comparison.csv',source/'comparison.csv')
    (output/'manifest.json').write_text(json.dumps({'schema_version':1,'description':'Aggregate results only. Paths are repository-relative. Full inputs and fits stay local.',
        'files':files},indent=2)+'\n',encoding='utf-8',newline='\n')
    verify_bundle(output)
    print(f'Exported and verified {len(files)} portable artifacts ({sum(r["bytes"] for r in files.values())/1e6:.2f} MB)')

if __name__=='__main__':main()
