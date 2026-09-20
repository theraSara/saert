"""Fixed first-hook SAE sensitivity using a training-derived median SD floor."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from threadpoolctl import threadpool_limits
try:
    from . import sae_regression as ridge
    from .sae_regression_diagnostics import paired_score
    from .pipeline_utils import file_record, sha256_file, write_json, read_table, run_metadata
except ImportError:
    import sae_regression as ridge
    from sae_regression_diagnostics import paired_score
    from pipeline_utils import file_record, sha256_file, write_json, read_table, run_metadata

ROOT = Path(__file__).resolve().parents[1]


def prepare_floor(controls, y, features, config):
    """Filtering and floor depend only on this training partition, never its test rows."""
    fit = ridge.prepare_fit(controls, y, features, config)
    if not sparse.issparse(features):
        raise ValueError('This declared sensitivity is for sparse SAE inputs')
    floor = float(np.median(fit['scale'])) if len(fit['scale']) else 0.
    scale = np.maximum(fit['scale'], floor)
    z = features.astype(float)[:, fit['keep']].multiply(1/scale).tocsr()
    fit.update(scale=scale, z=z, qz=np.asarray((z.T @ fit['q']).T), scale_floor=floor,
               rhs=np.asarray(z.T @ (y-fit['q'] @ (fit['q'].T @ y))).ravel()/len(y))
    return fit


def fit_nested(frame, controls, features, config):
    y = frame[ridge.OUTCOME].to_numpy(float)
    c = frame[controls].to_numpy(float)
    groups = frame.text_id.to_numpy()
    alphas = np.array(config['alphas'])
    pred, baseline = np.full(len(y), np.nan), np.full(len(y), np.nan)
    fold_ids = np.zeros(len(y), dtype=int)
    records, feature_coefs, control_coefs = [], [], []
    for fold, (train, test) in enumerate(LeaveOneGroupOut().split(c, groups=groups), 1):
        started = time.perf_counter()
        sse = np.zeros(len(alphas)); count = 0; inner_records = []
        for tr, va in GroupKFold(config['inner_splits']).split(c[train], groups=groups[train]):
            a, b = train[tr], train[va]
            fit = prepare_floor(c[a], y[a], features[a], config)
            predictions, _, iterations = ridge.ridge_path(fit, c[b], features[b], alphas, config)
            errors = ((predictions-y[b, None])**2).sum(axis=0)
            sse += errors; count += len(b)
            inner_records.append(dict(train_text_ids=np.unique(groups[a]).tolist(),
                validation_text_ids=np.unique(groups[b]).tolist(), scale_floor=fit['scale_floor'],
                sse_by_alpha=errors.tolist(), n_validation=len(b), cg_iterations=iterations))
        chosen = int(np.argmin(sse))
        fit = prepare_floor(c[train], y[train], features[train], config)
        p, co, iterations = ridge.ridge_path(fit, c[test], features[test], [alphas[chosen]], config, True)
        pred[test] = p[:, 0]
        baseline[test] = np.column_stack([np.ones(len(test)), fit['scaler'].transform(c[test])]) @ fit['base_coef']
        fold_ids[test] = fold
        feature_coefs.append(co[0][0]); control_coefs.append(co[0][1])
        records.append(dict(fold=fold, train_text_ids=np.unique(groups[train]).tolist(),
            test_text_ids=np.unique(groups[test]).tolist(), alpha=float(alphas[chosen]),
            scale_floor=fit['scale_floor'], inner_mse_by_alpha=(sse/count).tolist(),
            inner_folds=inner_records, cg_iterations=iterations, seconds=time.perf_counter()-started))
        if fold == 1 or fold % 10 == 0:
            print(f'  fold {fold}: alpha={alphas[chosen]:g}, SD floor={fit["scale_floor"]:.5g}', flush=True)
    return pred, baseline, fold_ids, records, np.array(feature_coefs), np.array(control_coefs)


def run(corpus, kind, source_root, output, config, specification):
    source = source_root/corpus/kind/'L01'/'sae'
    m = json.loads((source/'manifest.json').read_text())
    ridge.verify_saved(source, sha256_file(ROOT/'configs/sae_regression.json'), m['inputs'])
    for rec in m['inputs'].values():
        if sha256_file(rec['path']) != rec['sha256']:
            raise ValueError('Original source artifact changed')
    folder = output/corpus/kind
    if folder.exists():
        raise FileExistsError(f'Use a fresh output directory: {folder}')
    frame, _, models = ridge.build_design(read_table(m['inputs']['scores']['path']), config['spillover_lags'])
    reference = read_table(source/'predictions.csv')
    if reference.row_uid.tolist() != frame.row_uid.tolist():
        raise ValueError('Sensitivity/reference alignment mismatch')
    features = sparse.load_npz(m['inputs']['features']['path'])[frame.hidden_row_idx.to_numpy(int)]
    print(f'{corpus}/{kind}: training-only scale floor sensitivity', flush=True)
    pred, base, folds, records, fco, cco = fit_nested(frame, models['matched_spillover'], features, config)
    if not np.isfinite(pred).all() or not np.allclose(base, reference.baseline_prediction, atol=1e-8, rtol=0):
        raise ValueError('Nonfinite prediction or baseline mismatch')
    result = reference.copy()
    result['original_prediction'] = reference.prediction
    result['prediction'] = pred
    result['fold'] = folds
    folder.mkdir(parents=True)
    result.to_csv(folder/'predictions.csv', index=False)
    write_json(folder/'folds.json', {'controls': ['intercept']+models['matched_spillover'], 'folds': records})
    np.savez_compressed(folder/'coefficients.npz', feature_coef_raw=fco, control_coef_raw=cco)
    write_json(folder/'manifest.json', {**run_metadata(), 'corpus': corpus, 'state': kind, 'hook': 'L01',
        'specification': specification, 'ridge_config': config,
        'inputs': {'original_manifest': file_record(source/'manifest.json'), **m['inputs']},
        'source_code': [file_record(__file__), file_record(ridge.__file__)],
        'outputs': {p.name: file_record(p) for p in folder.iterdir()}})
    orig_score = paired_score(result.assign(prediction=result.original_prediction), bootstrap=config['bootstrap'])
    new_score = paired_score(result, bootstrap=config['bootstrap'])
    contrast = paired_score(result, reference='original_prediction', bootstrap=config['bootstrap'])
    return dict(corpus=corpus, state=kind, hook='L01', original_gain_pp=100*orig_score['delta_r2'],
        floor_gain_pp=100*new_score['delta_r2'], floor_ci_low_pp=100*new_score['ci_low'],
        floor_ci_high_pp=100*new_score['ci_high'], floor_minus_original_pp=100*contrast['delta_r2'],
        original_max_abs_error=orig_score['max_abs_error_log_ms'],
        floor_max_abs_error=new_score['max_abs_error_log_ms'],
        texts_improved=new_score['texts_improved'], n_texts=new_score['n_texts'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT/'results/sae_regression_v2')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/scaling_sensitivity_v1')
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/sae_regression.json').read_text())
    specification = json.loads((ROOT/'configs/scaling_sensitivity.json').read_text())
    rows = []
    with threadpool_limits(limits=config['threads']):
        for corpus in ['provo', 'natural_stories']:
            for kind in ['prefix', 'post']:
                rows.append(run(corpus, kind, args.source_root, args.output_dir, config, specification))
    pd.DataFrame(rows).to_csv(args.output_dir/'comparison.csv', index=False)
    write_json(args.output_dir/'manifest.json', {**run_metadata(), 'specification': specification,
        'config': file_record(ROOT/'configs/scaling_sensitivity.json'),
        'source_code': file_record(__file__), 'summary': file_record(args.output_dir/'comparison.csv'),
        'runs': [file_record(p) for p in args.output_dir.glob('*/*/manifest.json')]})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
