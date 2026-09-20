from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.sae_regression_diagnostics import choose_inner_hooks, paired_score
from src.table_export import export_table


class ReportingTests(unittest.TestCase):
    def test_hook_selection_ignores_outer_test_accuracy(self):
        frame = pd.DataFrame({"row_uid": ["a", "b", "c", "d"], "fold": [1, 1, 2, 2],
                              "text_id": [1, 1, 2, 2], "log_primary_RT": [0., 1., 0., 1.],
                              "prediction": [100., 100., 100., 100.], "baseline_prediction": [.5]*4})
        runs = {"A": {"predictions": frame.copy(), "folds": [{"fold": 1, "selected_inner_mse": 1., "alpha": 1.}, {"fold": 2, "selected_inner_mse": 3., "alpha": 1.}]},
                "B": {"predictions": frame.assign(prediction=0), "folds": [{"fold": 1, "selected_inner_mse": 2., "alpha": 1.}, {"fold": 2, "selected_inner_mse": 1., "alpha": 1.}]}}
        chosen, selection = choose_inner_hooks(runs)
        self.assertEqual([s["label"] for s in selection], ["A", "B"])
        np.testing.assert_array_equal(chosen.prediction, [100, 100, 0, 0])

    def test_identical_predictions_have_zero_paired_interval(self):
        p = pd.DataFrame({"text_id": np.repeat([1,2,3], 3), "log_primary_RT": np.arange(9),
                          "prediction": np.arange(9)+1, "baseline_prediction": np.arange(9)+1})
        score = paired_score(p, bootstrap=100)
        self.assertEqual([score[k] for k in ["delta_r2", "ci_low", "ci_high"]], [0, 0, 0])

    def test_latex_escaping_precision_and_unrounded_csv(self):
        frame = pd.DataFrame({"Name": ["A&B_1"], "Gain": [.123456]})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"table.tex"
            latex = export_table(frame, path, caption="An effect & comparison", label="tab:test",
                                 decimals={"Gain": 2}, signed=["Gain"], emphasize=["Gain"])
            self.assertIn(r"A\&B\_1", latex)
            self.assertIn(r"\textbf{+0.12}", latex)
            self.assertEqual(pd.read_csv(path.with_suffix('.csv')).Gain.iloc[0], .123456)
            self.assertTrue(path.with_suffix('.manifest.json').exists())
            wide = export_table(frame, path, caption="Wide table", label="tab:wide", wide=True)
            self.assertIn(r"\begin{table*}[t]", wide)
            self.assertIn(r"\end{table*}", wide)


if __name__ == "__main__":
    unittest.main()
