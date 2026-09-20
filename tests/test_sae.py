"""SAE alignment, sparse completeness, and intervention boundary checks."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import torch
from src.sae_extraction import (check_compatibility, encode_array, geometry_summary,
    replacement_hook, select_windows, verify_live_states)
from src.surprisal import MODEL_PROCESSING


class IdentitySAE:
    cfg = SimpleNamespace(d_in=3, d_sae=3)
    def encode(self, x):
        return x
    def decode(self, z):
        return z


class SAETests(unittest.TestCase):
    def test_sparse_complete_identity_and_zero_rows(self):
        x = np.array([[1, 0, 2], [0, 0, 0], [0, 3, 1]], dtype=np.float32)
        z, d = encode_array(x, IdentitySAE(), batch_size=2)
        np.testing.assert_array_equal(z.toarray(), x)
        np.testing.assert_array_equal(d.n_active, [2, 0, 2])
        summary = geometry_summary(x, d, np.ones(3, dtype=bool))
        self.assertEqual(summary["reconstruction_r2_centered"], 1)
        self.assertEqual(summary["fraction_zero_rows"], 1/3)

    def test_reject_nonfinite_or_wrong_dimensions(self):
        for x in [np.ones((2, 4)), np.array([[1, np.nan, 2]])]:
            with self.assertRaises(ValueError):
                encode_array(x, IdentitySAE())

    def test_hook_and_context_must_match(self):
        raw = dict(model_name="gpt2-small", hook_point="blocks.2.hook_resid_pre", d_in=768, d_sae=24576, context_size=128)
        extraction = {"hidden": {"window_size": 128, "pooling": "final_subtoken", "save_prefix": True},
                      "model": {"processing": MODEL_PROCESSING}}
        self.assertIn("not recorded", check_compatibility(raw, raw["hook_point"], extraction)["training_bos_policy"])
        with self.assertRaises(ValueError):
            check_compatibility(raw, "blocks.3.hook_resid_pre", extraction)
        wrong = copy.deepcopy(extraction)
        wrong["hidden"]["window_size"] = 1024
        with self.assertRaises(ValueError):
            check_compatibility(raw, raw["hook_point"], wrong)

    def test_bos_protection_only_for_actual_initial_window(self):
        x = torch.ones(1, 4, 3)
        initial = replacement_hook(IdentitySAE(), "zero", True)(x, None)
        self.assertTrue(torch.equal(initial[:, 0], x[:, 0]))
        self.assertEqual(float(initial[:, 1:].sum()), 0)
        rolling = replacement_hook(IdentitySAE(), "zero", False)(x, None)
        self.assertEqual(float(rolling.sum()), 0)
        self.assertTrue(torch.equal(x, replacement_hook(IdentitySAE(), "reconstruction", False)(x, None)))

    def test_window_selection_and_ownership(self):
        tokens = pd.DataFrame({"text_id": np.repeat([1, 2, 3], 180), "token_idx": np.tile(np.arange(180), 3), "token_id": 9})
        m = {"bos_token_id": 50256, "hidden": {"window_size": 128, "stride": 64}}
        a = select_windows(tokens, m, 3, 42)
        self.assertEqual(a, select_windows(tokens, m, 3, 42))
        for w in a:
            self.assertLessEqual(len(w["ids"]), 128)
            self.assertGreater(w["claim_start"], w["start"])
            self.assertEqual(w["ids"][0] == 50256, w["start"] == 0)

    def test_live_validation_detects_permuted_saved_rows(self):
        rows = pd.DataFrame({"text_id": [1, 1], "hidden_window_start": [0, 0],
                             "hidden_model_token_idx": [1, 2], "hidden_row_idx": [0, 1]})
        x = np.array([[1, 2, 3], [2, 4, 6]], dtype=np.float32)
        window = {"text_id": 1, "start": 0, "cache": {"hook": torch.tensor(np.vstack([x[:1], x]))}}
        self.assertEqual(verify_live_states(rows, x, [window], "hook", "post")["n_rows"], 2)
        with self.assertRaises(ValueError):
            verify_live_states(rows, x[::-1], [window], "hook", "post")


if __name__ == "__main__":
    unittest.main()
