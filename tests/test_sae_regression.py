import copy
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from src.sae_regression import prepare_fit, ridge_path, nested_predictions

CONFIG = json.loads((Path(__file__).resolve().parents[1]/"configs/sae_regression.json").read_text())


class NestedRidgeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(81)
        self.config = {**CONFIG, "min_active_training_rows": 1, "alphas": [10., 1., .1]}

    def test_matches_joint_partially_penalized_solution(self):
        c = self.rng.normal(size=(50, 3))
        x = self.rng.normal(size=(50, 12))
        y = 2 + c @ np.array([1., -2., .2]) + x[:, 0] + self.rng.normal(size=50)*.1
        fit = prepare_fit(c, y, x, self.config)
        pred, coefficients, _ = ridge_path(fit, c, x, [1.], self.config, True)
        joint = np.column_stack([fit["c"], fit["z"]])
        penalty = np.diag(np.r_[np.zeros(4), np.ones(12)*len(y)])
        expected = joint @ np.linalg.solve(joint.T @ joint + penalty, joint.T @ y)
        np.testing.assert_allclose(pred[:, 0], expected, atol=1e-9)
        raw, ctrl = coefficients[0]
        np.testing.assert_allclose(ctrl[0]+c @ ctrl[1:]+x @ raw, expected, atol=1e-9)

    def test_sparse_and_dense_agree(self):
        x = self.rng.uniform(size=(65, 20))
        x[x < .8] = 0
        c = self.rng.normal(size=(65, 2))
        y = self.rng.normal(size=65)
        predictions = []
        for matrix in [x, sparse.csr_matrix(x)]:
            fit = prepare_fit(c[:50], y[:50], matrix[:50], self.config)
            pred, _, _ = ridge_path(fit, c[50:], matrix[50:], self.config["alphas"], self.config)
            predictions.append(pred)
        np.testing.assert_allclose(*predictions, atol=2e-6)

    def test_test_story_outcomes_cannot_change_its_fit(self):
        frame = pd.DataFrame({"text_id": np.repeat([1, 2, 3, 4], 12), "control": self.rng.normal(size=48),
                              "log_primary_RT": self.rng.normal(size=48)})
        x = sparse.csr_matrix(self.rng.uniform(size=(48, 8)))
        a = nested_predictions(frame, ["control"], x, self.config, max_folds=1)
        altered = frame.copy()
        altered.loc[altered.text_id == 1, "log_primary_RT"] += 100
        b = nested_predictions(altered, ["control"], x, self.config, max_folds=1)
        np.testing.assert_array_equal(a[0][:12], b[0][:12])
        np.testing.assert_array_equal(a[4], b[4])
        self.assertEqual(a[3][0]["alpha"], b[3][0]["alpha"])
        self.assertEqual(a[3][0]["inner_folds"], b[3][0]["inner_folds"])

    def test_filtering_only_uses_training_rows_and_constant_features(self):
        x = np.column_stack([np.arange(10), np.ones(10), np.zeros(10)])
        c = self.rng.normal(size=(10, 1))
        fit = prepare_fit(c, np.arange(10), sparse.csr_matrix(x), self.config)
        np.testing.assert_array_equal(fit["keep"], [True, False, False])
        all_zero = prepare_fit(c, np.arange(10), sparse.csr_matrix((10, 2)), self.config)
        pred, _, _ = ridge_path(all_zero, c, sparse.csr_matrix((10, 2)), [1], self.config)
        np.testing.assert_allclose(pred[:, 0], all_zero["c"] @ all_zero["base_coef"])


if __name__ == "__main__":
    unittest.main()
