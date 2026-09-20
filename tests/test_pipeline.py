"""Offline correctness tests: no corpus downloads or pretrained weights required."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from src import data, surprisal
from src.pipeline_utils import (normalize_words, numeric, read_table, tokenize_words,
                                validate_word_matches, window_plan)


def make_tokenizer():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(["There are words. Café isn't easy! NA unbelievable 👋"],
        trainers.BpeTrainer(vocab_size=270, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=["<|endoftext|>"]))
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>",
                                  bos_token="<|endoftext|>")


def mock_forward(ids, model, device, hooks=()):
    n = len(ids)
    scores = np.arange(n, dtype=np.float32) + 1
    scores[0] = np.nan
    # Each state explicitly records the window's first token and local position.
    states = np.stack([np.column_stack([np.full(n, ids[0]), np.arange(n)])
                       for _ in hooks]).astype(np.float32) if hooks else None
    return scores, states


class StimulusTests(unittest.TestCase):
    def setUp(self):
        self.stimulus = pd.DataFrame({"text_id": [1, 1, 1], "word_position": [1, 2, 3],
                                      "word": ["There", "are", "words."]})

    def test_missing_initial_and_internal_words_fail(self):
        for idx in [0, 1]:
            with self.subTest(idx=idx), self.assertRaisesRegex(ValueError, "incomplete"):
                normalize_words(self.stimulus.drop(index=idx), "test", complete=True)

    def test_duplicate_keys_and_fractional_positions_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_words(pd.concat([self.stimulus, self.stimulus.iloc[[0]]]), "test")
        invalid = self.stimulus.astype({"word_position": float})
        invalid.loc[0, "word_position"] = 1.5
        with self.assertRaisesRegex(ValueError, "integer"):
            normalize_words(invalid, "test")

    def test_word_mismatch_is_not_silently_joined(self):
        observed = self.stimulus.copy()
        observed.loc[1, "word"] = "were"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_word_matches(self.stimulus, observed, "test")

    def test_missing_rt_keeps_stimulus_and_trial_filters_precede_mean(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.stimulus.to_csv(root / "stimuli.csv", index=False)
            trials = pd.DataFrame({"Text_ID": [1, 1, 1], "Word_Number": [2, 2, 3],
                "Word": ["are", "are", "words."], "Part_ID": ["A", "B", "A"],
                "IA_FIRST_FIXATION_DURATION": [200, 20, 0], "IA_FIRST_RUN_DWELL_TIME": [350, 400, 0]})
            trials.to_csv(root / "rt.csv", index=False)
            result, kept_trials, _, settings = data.prepare_provo(root / "stimuli.csv", root / "rt.csv", "FFD", rt_min=100)
            self.assertEqual(result["word"].tolist(), ["There", "are", "words."])
            self.assertTrue(np.isnan(result.loc[0, "primary_RT"]))
            self.assertTrue(np.isnan(result.loc[2, "primary_RT"]))
            self.assertEqual(result.loc[1, "primary_RT"], 200)
            self.assertEqual(result.loc[1, "n_participants"], 1)
            self.assertEqual(len(kept_trials), 3)
            self.assertEqual(settings["trial_status_counts"]["below_min"], 1)
            gaze, _, _, _ = data.prepare_provo(root / "stimuli.csv", root / "rt.csv", "GZD")
            self.assertEqual(gaze.loc[1, "primary_RT"], 375)

    def test_requested_measure_never_falls_back(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.stimulus.to_csv(root / "stimuli.csv", index=False)
            trials = self.stimulus.assign(Part_ID="A", IA_FIRST_FIXATION_DURATION=200)
            trials.to_csv(root / "rt.csv", index=False)
            with self.assertRaisesRegex(ValueError, "No alternative"):
                data.prepare_provo(root / "stimuli.csv", root / "rt.csv", "GZD")

    def test_provo_interest_areas_recover_missing_norms_and_keep_displayed_forms(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rt.csv"
            pd.DataFrame({"Text_ID": [1]*6, "IA_ID": [1, 2, 3]*2,
                "IA_LABEL": ["There ", "are ", "words. "]*2,
                "Word_Number": ["NA", "7", "8"]*2,
                "Word": ["NA", "are", "normalized"]*2,
                "Participant_ID": ["A"]*3 + ["B"]*3,
                "IA_FIRST_FIXATION_DURATION": [200, 300, 400, 220, 320, 420]}).to_csv(path, index=False)
            prepared, trials, _, settings = data.prepare_provo(None, path, "FFD")
            self.assertEqual(prepared["word"].tolist(), ["There", "are", "words."])
            self.assertEqual(prepared["word_position"].tolist(), [1, 2, 3])
            self.assertEqual(prepared["primary_RT"].tolist(), [210, 310, 410])
            self.assertEqual(len(trials), 6)
            self.assertEqual(trials.iloc[0]["norm_word"], "NA")
            self.assertEqual(trials.iloc[-1]["norm_word_position"], "8")
            self.assertEqual(settings["participant_column"], "Participant_ID")
            self.assertEqual(settings["position_source"], "IA_ID")

    def test_provo_conflicting_interest_area_labels_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rt.csv"
            pd.DataFrame({"Text_ID": [1, 1], "IA_ID": [1, 1], "IA_LABEL": ["One", "Two"],
                          "Participant_ID": ["A", "B"], "IA_FIRST_FIXATION_DURATION": [200, 300]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                data.prepare_provo(None, path, "FFD")

    def test_natural_stories_order_missing_rt_and_literal_na(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.stimulus.to_csv(root / "all_stories.tok", sep="\t", index=False)
            self.stimulus.iloc[[2, 1]].rename(columns={"text_id": "item", "word_position": "zone"}).assign(
                meanItemRT=[300, 200], nItem=[4, 3]).to_csv(root / "processed_wordinfo.tsv", sep="\t", index=False)
            prepared, trials, _, _ = data.prepare_natural_stories(root)
            self.assertIsNone(trials)
            self.assertEqual(prepared["word_position"].tolist(), [1, 2, 3])
            self.assertTrue(np.isnan(prepared.loc[0, "primary_RT"]))
            self.assertEqual(prepared.loc[2, "primary_RT"], 300)
            (root / "literal.csv").write_text("word\nNA\n", encoding="utf-8")
            self.assertEqual(read_table(root / "literal.csv").loc[0, "word"], "NA")

    def test_numeric_garbage_fails(self):
        with self.assertRaises(ValueError):
            numeric(pd.Series(["200", "typo"]), "RT")
        self.assertTrue(numeric(pd.Series(["."]), "RT").isna().all())


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = make_tokenizer()

    def test_byte_level_punctuation_unicode_and_multitoken_words(self):
        words = ["'Admiral", "Café", "isn't", "unbelievable!", "👋", "NA"]
        text, ids, _, spans = tokenize_words(words, self.tokenizer)
        self.assertEqual(text, " ".join(words))
        self.assertEqual(self.tokenizer.decode(ids), text)
        self.assertEqual([i for s, e in spans for i in range(s, e+1)], list(range(len(ids))))
        self.assertTrue(any(e > s for s, e in spans))

    def test_empty_words_fail(self):
        with self.assertRaises(ValueError):
            tokenize_words(["word", ""], self.tokenizer)

    def test_cross_word_token_fails(self):
        class BrokenTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [1], "offset_mapping": [(0, len(text))]}
            def decode(self, ids, **kwargs):
                return "a b"
        with self.assertRaisesRegex(ValueError, "multiple corpus words"):
            tokenize_words(["a", "b"], BrokenTokenizer())


class WindowTests(unittest.TestCase):
    def test_bos_story_boundaries_and_prefix_states(self):
        for n in [1, 127, 128, 1023, 1024, 1025, 1300]:
            for size, stride in [(1024, 512), (128, 64)]:
                with self.subTest(n=n, size=size):
                    calls = []
                    def forward(ids, model, device, hooks):
                        calls.append(list(ids))
                        return mock_forward(ids, model, device, hooks)
                    ids = list(range(1, n+1))
                    result = surprisal.extract_story(ids, 0, None, "cpu", size, stride,
                        hooks=["h"], save_prefix=True, forward_fn=forward)
                    plan = list(window_plan(n+1, size, stride))
                    self.assertEqual(calls, [([0]+ids)[s:e] for s, e, _, _ in plan])
                    self.assertEqual(sum(call.count(0) for call in calls), 1)
                    self.assertTrue(all(len(call) <= size for call in calls))
                    self.assertEqual(result["surprisal"].shape, (n,))
                    self.assertTrue(np.isfinite(result["surprisal"]).all())
                    self.assertEqual(result["left_context"][0], 1)
                    np.testing.assert_array_equal(result["prefix_hidden"][0, 0], [0, 0])
                    expected = [max(t-s for s, e, _, _ in plan if s <= t < e) for t in range(1, n+1)]
                    np.testing.assert_array_equal(result["left_context"], expected)
                    np.testing.assert_array_equal(result["prefix_hidden"][0, :, 1], np.array(expected)-1)

    def test_bos_first_multitoken_word_matches_manual_chain_rule(self):
        logits = torch.tensor([[[0., 2., -1.], [2., 0., -1.], [-1., 0., 2.]]])
        class FakeModel:
            def __call__(self, inputs, **kwargs):
                self.inputs = inputs.tolist()
                return logits
        model = FakeModel()
        result = surprisal.extract_story([1, 0], 2, model, "cpu")
        expected = -torch.log_softmax(logits, -1)[0, [0, 1], [1, 0]].sum().item() / np.log(2)
        self.assertEqual(model.inputs, [[2, 1, 0]])
        self.assertAlmostEqual(surprisal.word_surprisals(result["surprisal"], [(0, 1)])[0], expected, places=6)
        for bos in [None, -1]:
            with self.assertRaises(ValueError):
                surprisal.extract_story([1], bos, model, "cpu")

    def test_boundaries_exact_coverage_and_longest_evaluated_context(self):
        for n in [1, 127, 128, 129, 1023, 1024, 1025, 1300, 2049]:
            for size, stride in [(1024, 512), (128, 64), (8, 1), (8, 7)]:
                with self.subTest(n=n, size=size, stride=stride):
                    out = surprisal.extract_sequence(list(range(n)), None, "cpu", size, stride,
                        hooks=["test"], save_prefix=True, forward_fn=mock_forward)
                    plan = list(window_plan(n, size, stride))
                    expected = [max(t-start for start, end, _, _ in plan if start <= t < end) for t in range(n)]
                    np.testing.assert_array_equal(out["left_context"], expected)
                    np.testing.assert_array_equal(out["hidden"][0, :, 1], expected)
                    self.assertTrue(np.isnan(out["surprisal"][0]))
                    self.assertTrue(np.isfinite(out["surprisal"][1:]).all())
                    self.assertTrue(np.isnan(out["prefix_hidden"][:, 0]).all())
                    np.testing.assert_array_equal(out["prefix_hidden"][0, 1:, 1], np.array(expected[1:])-1)

    def test_prefix_uses_target_window_at_boundary(self):
        out = surprisal.extract_sequence(list(range(1300)), None, "cpu", hooks=["h"],
                                         save_prefix=True, forward_fn=mock_forward)
        np.testing.assert_array_equal(out["hidden"][0, 1023], [0, 1023])
        np.testing.assert_array_equal(out["prefix_hidden"][0, 1024], [512, 511])

    def test_invalid_windows_fail(self):
        for size, stride in [(1025, 512), (1, 1), (128, 128), (128, 0)]:
            with self.assertRaises(ValueError):
                list(window_plan(5, size, stride))

    def test_word_sum_propagates_nan_and_rejects_truncation(self):
        scores = np.array([np.nan, 2., 3., 4.])
        values = surprisal.word_surprisals(scores, [(0, 1), (2, 3)])
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(values[1], 7)
        with self.assertRaises(ValueError):
            surprisal.word_surprisals(scores, [(2, 8)])

    def test_shifted_cross_entropy_matches_hand_computation(self):
        logits = torch.tensor([[[0., 2., -1.], [2., 0., -1.], [-1., 0., 2.]]])
        class FakeModel:
            def __call__(self, inputs, **kwargs):
                self.inputs = inputs
                return logits
        model = FakeModel()
        scores, states = surprisal.run_forward_pass([2, 1, 0], model, "cpu")
        expected = -torch.log_softmax(logits, -1)[0, [0, 1], [1, 0]].numpy() / np.log(2)
        np.testing.assert_allclose(scores[1:], expected, rtol=1e-6)
        self.assertTrue(np.isnan(scores[0]))
        self.assertIsNone(states)

    def test_real_small_transformer_causality_and_single_window_parity(self):
        from transformer_lens import HookedTransformer, HookedTransformerConfig
        model = HookedTransformer(HookedTransformerConfig(n_layers=2, d_model=16, n_ctx=32,
            d_head=8, n_heads=2, d_mlp=32, d_vocab=20, act_fn="gelu", normalization_type="LN",
            seed=4, device="cpu"))
        model.eval()
        hooks = ["blocks.0.hook_resid_pre", "blocks.1.hook_resid_post"]
        tokens = [2, 3, 4, 5, 6, 7]
        scores, hidden = surprisal.run_forward_pass(tokens, model, "cpu", hooks)
        extracted = surprisal.extract_sequence(tokens, model, "cpu", 16, 8, hooks, True)
        np.testing.assert_allclose(extracted["surprisal"], scores, equal_nan=True)
        np.testing.assert_allclose(extracted["hidden"], hidden)
        changed_scores, changed_hidden = surprisal.run_forward_pass(tokens[:-1]+[9], model, "cpu", hooks)
        np.testing.assert_allclose(changed_scores[:-1], scores[:-1], equal_nan=True)
        np.testing.assert_allclose(changed_hidden[:, :-1], hidden[:, :-1], atol=1e-6)
        story = surprisal.extract_story(tokens, 1, model, "cpu", 16, 8, hooks, True)
        with torch.inference_mode():
            bos_logits, bos_cache = model.run_with_cache(torch.tensor([[1]]), names_filter=hooks,
                                                       return_type="logits", prepend_bos=False)
        first_expected = -torch.log_softmax(bos_logits[0, 0], -1)[tokens[0]].item() / np.log(2)
        self.assertAlmostEqual(story["surprisal"][0], first_expected, places=5)
        for h, hook in enumerate(hooks):
            np.testing.assert_allclose(story["prefix_hidden"][h, 0], bos_cache[hook][0, 0].numpy(), atol=1e-6)
        changed = surprisal.extract_story([9]+tokens[1:], 1, model, "cpu", 16, 8, hooks, True)
        np.testing.assert_allclose(changed["prefix_hidden"][:, 0], story["prefix_hidden"][:, 0], atol=1e-6)


class ArtifactTests(unittest.TestCase):
    def test_preparation_extraction_roundtrip_and_hash_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tokenizer = make_tokenizer()
            raw = pd.DataFrame({"text_id": [1, 1, 1], "word_position": [1, 2, 3],
                "word": ["There", "are", "words."], "primary_RT": [np.nan, 200., 300.],
                "rt_measure": ["FFD"]*3})
            prepared = data.add_predictors_and_alignment(raw, "provo", tokenizer, lambda w: 5.)
            data.save_prepared(prepared, None, "provo", root / "data/provo", {}, {}, tokenizer, "test")
            with self.assertRaises(FileExistsError):
                data.save_prepared(prepared, None, "provo", root / "data/provo", {}, {}, tokenizer, "test")
            frame, manifest, inputs = surprisal.load_prepared(root / "data", "provo")
            with patch.object(surprisal, "run_forward_pass", mock_forward):
                result = surprisal.extract_corpus("provo", frame, manifest, inputs,
                    SimpleNamespace(cfg=SimpleNamespace(d_model=2)), tokenizer, "cpu", root / "out",
                    "test", window_size=16, stride=8, hidden_context=8, hidden_stride=4, save_prefix=True)
            self.assertEqual(result["word"].tolist(), raw["word"].tolist())
            self.assertEqual(result["hidden_row_idx"].tolist(), [0, 1, 2])
            self.assertTrue(np.isfinite(result[["surprisal", "surprisal_matched_context"]]).all().all())
            self.assertTrue(result["prefix_available"].all())
            self.assertEqual(result.loc[0, "prefix_kind"], "bos")
            self.assertEqual(result.loc[0, "prefix_model_token_idx"], 0)
            self.assertEqual(result.loc[0, "prefix_token_idx"], -1)
            self.assertEqual(result.loc[0, "primary_RT"], "")  # missing RT survives CSV round trip
            self.assertEqual(np.load(root / "out/provo_hidden_L01.npy").shape, (3, 2))
            prefix = np.load(root / "out/provo_prefix_hidden_L01.npy")
            np.testing.assert_array_equal(prefix[0], [tokenizer.bos_token_id, 0])
            self.assertTrue(np.isfinite(prefix).all())
            extraction = json.loads((root / "out/provo_extraction_manifest.json").read_text())
            self.assertEqual(extraction["schema_version"], 3)
            self.assertEqual(extraction["bos_token_id"], tokenizer.bos_token_id)
            token_rows = pd.read_csv(root / "out/provo_token_windows.csv")
            np.testing.assert_array_equal(token_rows["model_token_idx"], token_rows["token_idx"]+1)
            self.assertTrue(np.isfinite(token_rows["surprisal"]).all())
            self.assertEqual(len(extraction["hook_files"]), 13)
            self.assertEqual(extraction["hook_files"]["blocks.2.hook_resid_pre"]["completed_blocks"], 2)
            self.assertEqual(extraction["hook_files"]["blocks.11.hook_resid_post"]["completed_blocks"], 12)
            path = root / "data/provo/provo_prepared.csv"
            path.write_text(path.read_text().replace("There", "THERE"))
            with self.assertRaisesRegex(ValueError, "changed since"):
                surprisal.load_prepared(root / "data", "provo")


if __name__ == "__main__":
    unittest.main()
