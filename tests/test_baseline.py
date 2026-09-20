"""Checks for leakage, lag alignment, and paired held-out comparisons."""
import unittest
import numpy as np
import pandas as pd
from src.baseline import build_design, evaluate_models, paired_metrics


def synthetic_data():
    rng = np.random.default_rng(12)
    n = 80
    df = pd.DataFrame({"text_id": np.repeat([1, 2, 3, 4], 20),
        "word_position": np.tile(np.arange(1, 21), 4), "word": [f"word{i}" for i in range(n)],
        "zipf_freq": rng.uniform(2, 7, n), "word_length": rng.integers(1, 12, n),
        "is_sentence_final": np.tile([0]*9+[1], 8), "surprisal": rng.uniform(1, 15, n)})
    df["surprisal_matched_context"] = df["surprisal"] + rng.uniform(0, 1, n)
    df["log_primary_RT"] = 5.5 - .05*df.zipf_freq + .025*df.surprisal + rng.normal(0, .01, n)
    df["primary_RT"] = np.exp(df.log_primary_RT)
    return df


class BaselineTests(unittest.TestCase):
    def test_lags_before_filtering_and_no_cross_text_history(self):
        raw = synthetic_data()
        raw.loc[1, ["primary_RT", "log_primary_RT"]] = np.nan
        eligible, full, models = build_design(raw, 2)
        third = eligible[(eligible.text_id == 1) & (eligible.word_position == 3)].iloc[0]
        self.assertEqual(third.surprisal_lag1, raw.loc[1, "surprisal"])
        self.assertEqual(third.surprisal_lag2, raw.loc[0, "surprisal"])
        starts = full[full.word_position == 1]
        self.assertTrue(starts.surprisal_lag1.eq(0).all())
        self.assertTrue(starts.surprisal_lag2.eq(0).all())
        self.assertTrue(starts.is_text_position_1.eq(1).all())
        self.assertEqual(len(full)-len(eligible), 1)
        self.assertNotIn("text_id", models["controls"])

    def test_no_aggregate_cutoffs_and_log_validation(self):
        raw = synthetic_data()
        raw.loc[0:1, "primary_RT"] = [90, 3500]
        raw.loc[0:1, "log_primary_RT"] = np.log([90, 3500])
        eligible, _, _ = build_design(raw)
        self.assertEqual(len(eligible), len(raw))
        raw.loc[0, "log_primary_RT"] += 1
        with self.assertRaisesRegex(ValueError, "log of"):
            build_design(raw)

    def test_missing_predictors_fail_and_no_random_split_fallback(self):
        raw = synthetic_data()
        raw.loc[5, "zipf_freq"] = np.nan
        with self.assertRaisesRegex(ValueError, "predictors"):
            build_design(raw)
        with self.assertRaisesRegex(ValueError, "three texts"):
            build_design(synthetic_data().query("text_id == 1"))

    def test_heldout_outcomes_cannot_change_their_predictions(self):
        df, _, models = build_design(synthetic_data())
        predictions, folds, parameters = evaluate_models(df, models)
        changed = df.copy()
        changed.loc[changed.text_id == 1, "log_primary_RT"] += 100
        new_predictions, _, _ = evaluate_models(changed, models)
        cols = [f"pred_{name}" for name in models]
        np.testing.assert_allclose(predictions.loc[df.text_id == 1, cols], new_predictions.loc[df.text_id == 1, cols])
        for split in folds:
            self.assertFalse(set(split["train_text_ids"]) & set(split["test_text_ids"]))
            self.assertEqual(len(split["test_text_ids"]), 1)
        fit = next(p for p in parameters if p["fold"] == 1 and p["model"] == "long_spillover")
        train = df[df.text_id != 1]
        np.testing.assert_allclose(fit["mean"], train[fit["columns"]].mean().to_numpy())
        expected = df.loc[df.text_id == 1, fit["columns"]].to_numpy() @ np.array(fit["coef_raw"]) + fit["intercept_raw"]
        np.testing.assert_allclose(expected, predictions.loc[df.text_id == 1, "pred_long_spillover"])

    def test_paired_bootstrap_identical_predictions_have_zero_difference(self):
        df, _, _ = build_design(synthetic_data())
        df["pred_controls"] = df.log_primary_RT + .1
        df["pred_long_current"] = df.pred_controls
        df["pred_matched_current"] = df.pred_controls
        models = {"controls": ["zipf_freq"], "long_current": ["surprisal"], "matched_current": ["surprisal_matched_context"]}
        summary, contrasts, _ = paired_metrics(df, models, bootstrap=100)
        np.testing.assert_allclose(summary[["delta_r2_vs_controls", "delta_r2_ci_low", "delta_r2_ci_high"]], 0)
        np.testing.assert_allclose(contrasts[["delta_r2", "ci_low", "ci_high"]], 0)
        repeat, _, _ = paired_metrics(df, models, bootstrap=100)
        pd.testing.assert_frame_equal(summary, repeat)

    def test_oof_residual_and_model_comparison_use_same_rows(self):
        df, _, models = build_design(synthetic_data())
        predictions, _, _ = evaluate_models(df, models)
        np.testing.assert_allclose(predictions.RT_resid_controls_oof, df.log_primary_RT-predictions.pred_controls)
        summary, _, per_text = paired_metrics(predictions, models, bootstrap=100)
        self.assertTrue(summary.n.eq(len(df)).all())
        self.assertEqual(len(per_text), len(models)*4)
        self.assertGreater(summary.set_index("model").loc["long_current", "delta_r2_vs_controls"], 0)


if __name__ == "__main__":
    unittest.main()
