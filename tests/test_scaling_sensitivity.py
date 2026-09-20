import unittest
import numpy as np
from scipy import sparse
from src.scaling_sensitivity import prepare_floor, fit_nested
import pandas as pd
from src.sae_regression import ridge_path

class ScalingSensitivityTests(unittest.TestCase):
    def test_training_floor_and_joint_solution(self):
        rng = np.random.default_rng(13)
        c = rng.normal(size=(50, 2)); x = rng.normal(size=(50, 5)); x[:, 0] *= 1e-5
        y = rng.normal(size=50)
        cfg = dict(variance_floor=1e-14, min_active_training_rows=5, cg_rtol=1e-10, cg_maxiter=1000)
        fit = prepare_floor(c, y, sparse.csr_matrix(x), cfg)
        expected = np.maximum(x.std(axis=0), np.median(x.std(axis=0)))
        np.testing.assert_allclose(fit['scale'], expected)
        design = np.column_stack([np.ones(50), fit['scaler'].transform(c), x/expected])
        penalty = np.diag([0]*3+[2]*5)
        coef = np.linalg.solve(design.T@design/50+penalty, design.T@y/50)
        pred,_,_ = ridge_path(fit,c,sparse.csr_matrix(x),[2],cfg)
        np.testing.assert_allclose(pred[:,0],design@coef,atol=1e-8)

    def test_heldout_outcomes_do_not_change_its_predictions(self):
        rng = np.random.default_rng(17)
        frame = pd.DataFrame({'text_id': np.repeat(np.arange(5), 12),
            'log_primary_RT': rng.normal(size=60), 'control': rng.normal(size=60)})
        x = sparse.csr_matrix(abs(rng.normal(size=(60, 6))))
        cfg = dict(variance_floor=1e-12, min_active_training_rows=5, cg_rtol=1e-8,
                   cg_maxiter=1000, alphas=[10.,1.], inner_splits=3)
        first = fit_nested(frame, ['control'], x, cfg)
        changed = frame.copy(); changed.loc[changed.text_id.eq(0), 'log_primary_RT'] += 100
        second = fit_nested(changed, ['control'], x, cfg)
        np.testing.assert_allclose(first[0][:12], second[0][:12])
        self.assertEqual(first[3][0]['alpha'], second[3][0]['alpha'])
        self.assertEqual(first[3][0]['scale_floor'], second[3][0]['scale_floor'])

if __name__ == '__main__': unittest.main()
