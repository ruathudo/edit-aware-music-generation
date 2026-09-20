"""
Tests for the edit-aware model, its loss, and the label mapping.

The label mapping is the part most likely to be quietly wrong: it joins the
aligner's per-event classes to the tokenizer's positions, and a misalignment
there would train the head to point at the wrong notes while every loss curve
still looked healthy.

Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.align import DELETE, INSERT_BEFORE, KEEP, SUB_DUR, SUB_PITCH  # noqa: E402
from src.edit_model import (  # noqa: E402
    IGNORE_INDEX,
    EditAwareTransformer,
    EditModelConfig,
    edit_aware_loss,
    edit_class_weights,
)
from src.evaluation import edit_prediction_scores, paired_comparison  # noqa: E402
from src.refinement import make_collate, token_edit_labels  # noqa: E402
from src.tokenizer import MelodyTokenizer, TokenizerConfig  # noqa: E402

Q = 24


def make_tokenizer():
    return MelodyTokenizer(TokenizerConfig(pitch_min=55, pitch_max=80, durations=(6, 12, 24, 48)))


def make_model(tokenizer, use_edit_head=True, seed=0):
    torch.manual_seed(seed)
    config = EditModelConfig(
        vocab_size=len(tokenizer), d_model=32, n_heads=2,
        n_encoder_layers=2, n_decoder_layers=2, d_ff=64, max_len=128,
        use_edit_head=use_edit_head,
    )
    return EditAwareTransformer(config, pad_id=tokenizer.pad_id).eval()


class TestTokenEditLabels(unittest.TestCase):
    def setUp(self):
        self.tok = make_tokenizer()

    def test_identical_melodies_are_all_keep(self):
        notes = [(60, Q), (62, Q), (64, Q)]
        tokens, labels = token_edit_labels(notes, notes, self.tok)
        self.assertEqual(len(tokens), len(labels))
        self.assertTrue(all(l in (KEEP, IGNORE_INDEX) for l in labels))

    def test_structural_tokens_are_ignored_not_keep(self):
        """Labelling BOS/BAR as KEEP would inflate the majority class."""
        notes = [(60, Q), (62, Q)]
        tokens, labels = token_edit_labels(notes, notes, self.tok)
        self.assertEqual(labels[0], IGNORE_INDEX)  # BOS

    def test_pitch_substitution_lands_on_the_right_event(self):
        source = [(60, Q), (62, Q), (64, Q)]
        target = [(60, Q), (67, Q), (64, Q)]
        tokens, labels = token_edit_labels(source, target, self.tok)
        _, spans = self.tok.encode_with_spans(source)

        pitch_at, duration_at = spans[1]
        self.assertEqual(labels[pitch_at], SUB_PITCH)
        self.assertEqual(labels[duration_at], SUB_PITCH)
        # Neighbours untouched
        self.assertEqual(labels[spans[0][0]], KEEP)
        self.assertEqual(labels[spans[2][0]], KEEP)

    def test_duration_substitution_is_typed_separately(self):
        source = [(60, Q), (62, Q)]
        target = [(60, Q), (62, 48)]
        _, labels = token_edit_labels(source, target, self.tok)
        _, spans = self.tok.encode_with_spans(source)
        self.assertEqual(labels[spans[1][0]], SUB_DUR)

    def test_deletion_is_marked_on_the_spurious_note(self):
        source = [(60, Q), (62, Q), (64, Q)]
        target = [(60, Q), (64, Q)]
        _, labels = token_edit_labels(source, target, self.tok)
        _, spans = self.tok.encode_with_spans(source)
        self.assertEqual(labels[spans[1][0]], DELETE)

    def test_insertion_is_marked_before_the_following_note(self):
        source = [(60, Q), (64, Q)]
        target = [(60, Q), (62, Q), (64, Q)]
        _, labels = token_edit_labels(source, target, self.tok)
        _, spans = self.tok.encode_with_spans(source)
        self.assertEqual(labels[spans[1][0]], INSERT_BEFORE)

    def test_trailing_insertion_lands_on_eos(self):
        source = [(60, Q)]
        target = [(60, Q), (62, Q)]
        tokens, labels = token_edit_labels(source, target, self.tok)
        self.assertEqual(tokens[-1], self.tok.eos_id)
        self.assertEqual(labels[-1], INSERT_BEFORE)

    def test_labels_align_with_bar_tokens_present(self):
        """The mapping must survive BAR tokens, which break index arithmetic."""
        notes = [(60, 48), (62, 48), (64, 24), (65, 24)]
        target = [(60, 48), (67, 48), (64, 24), (65, 24)]
        tokens, labels = token_edit_labels(notes, target, self.tok, bar_starts={2})
        self.assertIn(self.tok.bar_id, tokens)
        _, spans = self.tok.encode_with_spans(notes, {2})
        self.assertEqual(labels[spans[1][0]], SUB_PITCH)
        self.assertEqual(labels[tokens.index(self.tok.bar_id)], IGNORE_INDEX)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.tok = make_tokenizer()
        self.model = make_model(self.tok)

    def test_output_shapes(self):
        source = torch.randint(5, 20, (3, 11))
        target_in = torch.randint(5, 20, (3, 9))
        logits, edit_logits = self.model(source, target_in)
        self.assertEqual(logits.shape, (3, 9, len(self.tok)))
        self.assertEqual(edit_logits.shape, (3, 11, 5))

    def test_source_and_target_lengths_may_differ(self):
        """Insertions and deletions change length; the legacy design assumed
        they did not."""
        logits, edits = self.model(torch.randint(5, 20, (2, 17)), torch.randint(5, 20, (2, 4)))
        self.assertEqual(logits.size(1), 4)
        self.assertEqual(edits.size(1), 17)

    def test_encoder_is_bidirectional(self):
        """A later source token must be able to influence an earlier position --
        that is what makes typed edit classification well-posed."""
        source = torch.randint(5, 20, (1, 10))
        with torch.no_grad():
            before = self.model.encode(source)
            altered = source.clone()
            altered[0, 8] = (altered[0, 8] % 14) + 5
            after = self.model.encode(altered)
        self.assertFalse(torch.allclose(before[:, 0], after[:, 0], atol=1e-6))

    def test_decoder_is_causal(self):
        source = torch.randint(5, 20, (1, 8))
        target = torch.randint(5, 20, (1, 10))
        with torch.no_grad():
            memory = self.model.encode(source)
            base = self.model.decode(target, memory, source)
            altered = target.clone()
            altered[0, 7] = (altered[0, 7] % 14) + 5
            after = self.model.decode(altered, memory, source)
        torch.testing.assert_close(base[:, :7], after[:, :7], rtol=1e-4, atol=1e-5)

    def test_baseline_has_no_edit_head(self):
        model = make_model(self.tok, use_edit_head=False)
        logits, edits = model(torch.randint(5, 20, (2, 7)), torch.randint(5, 20, (2, 5)))
        self.assertIsNone(edits)
        with self.assertRaises(RuntimeError):
            model.predict_edits(torch.randint(5, 20, (1, 7)))

    def test_b2_is_parameter_matched_to_the_edit_aware_model(self):
        """B2 keeps the head and zeroes alpha, so the control differs only in
        whether the auxiliary loss contributes gradient."""
        with_head = make_model(self.tok, use_edit_head=True)
        without = make_model(self.tok, use_edit_head=False)
        self.assertEqual(
            with_head.n_parameters - without.n_parameters,
            with_head.config.d_model * 5 + 5,
        )

    def test_rejects_overlong_sequences(self):
        with self.assertRaises(ValueError):
            self.model(torch.randint(5, 20, (1, 200)), torch.randint(5, 20, (1, 4)))


class TestLoss(unittest.TestCase):
    def setUp(self):
        self.tok = make_tokenizer()
        self.model = make_model(self.tok)

    def test_alpha_zero_reduces_to_generative_loss(self):
        logits = torch.randn(2, 5, len(self.tok))
        target = torch.randint(1, 20, (2, 5))
        edits = torch.randn(2, 7, 5)
        labels = torch.randint(0, 5, (2, 7))

        total, gen, edit = edit_aware_loss(
            logits, target, edits, labels, self.tok.pad_id, alpha=0.0
        )
        self.assertAlmostEqual(float(total), gen, places=6)
        self.assertEqual(edit, 0.0)

    def test_alpha_scales_the_auxiliary_term(self):
        torch.manual_seed(1)
        logits = torch.randn(2, 5, len(self.tok))
        target = torch.randint(1, 20, (2, 5))
        edits = torch.randn(2, 7, 5)
        labels = torch.randint(0, 5, (2, 7))

        _, gen, edit = edit_aware_loss(logits, target, edits, labels, self.tok.pad_id, alpha=1.0)
        total2, _, _ = edit_aware_loss(logits, target, edits, labels, self.tok.pad_id, alpha=2.0)
        self.assertAlmostEqual(float(total2), gen + 2 * edit, places=4)

    def test_ignore_index_is_excluded(self):
        torch.manual_seed(2)
        logits = torch.randn(1, 4, len(self.tok))
        target = torch.randint(1, 20, (1, 4))
        edits = torch.randn(1, 6, 5)

        labels = torch.full((1, 6), IGNORE_INDEX)
        labels[0, 2] = KEEP
        total, _, edit = edit_aware_loss(logits, target, edits, labels, self.tok.pad_id, alpha=1.0)
        self.assertTrue(np_isfinite(edit))

    def test_class_weights_favour_rare_classes(self):
        labels = torch.tensor([KEEP] * 95 + [SUB_PITCH] * 4 + [DELETE])
        weights = edit_class_weights(labels)
        self.assertGreater(weights[DELETE], weights[SUB_PITCH])
        self.assertGreater(weights[SUB_PITCH], weights[KEEP])

    def test_class_weights_ignore_masked_positions(self):
        labels = torch.tensor([KEEP] * 10 + [IGNORE_INDEX] * 100)
        weights = edit_class_weights(labels)
        self.assertTrue(torch.isfinite(weights).all())


def np_isfinite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


class TestEvaluationMetrics(unittest.TestCase):
    def test_perfect_prediction_scores_one(self):
        actual = [KEEP, SUB_PITCH, DELETE, KEEP]
        scores = edit_prediction_scores(actual, actual)
        self.assertAlmostEqual(scores["macro_f1"], 1.0)
        self.assertAlmostEqual(scores["localization"]["f1"], 1.0)

    def test_predicting_keep_everywhere_collapses_macro_f1(self):
        """The failure mode class weighting exists to prevent."""
        actual = [KEEP] * 90 + [SUB_PITCH] * 5 + [DELETE] * 5
        predicted = [KEEP] * 100
        scores = edit_prediction_scores(predicted, actual)
        self.assertLess(scores["macro_f1"], 0.5)
        self.assertEqual(scores["localization"]["recall"], 0.0)
        self.assertGreater(scores["accuracy"], 0.85)  # accuracy still looks fine

    def test_ignored_positions_do_not_count(self):
        actual = [KEEP, IGNORE_INDEX, SUB_PITCH]
        predicted = [KEEP, SUB_DUR, SUB_PITCH]
        self.assertAlmostEqual(edit_prediction_scores(predicted, actual)["accuracy"], 1.0)

    def test_paired_comparison_detects_an_improvement(self):
        baseline = np.arange(60, dtype=float) + 10
        treatment = np.arange(60, dtype=float) + 5
        result = paired_comparison(baseline, treatment, n_boot=500)
        self.assertAlmostEqual(result["mean_delta"], 5.0)
        self.assertEqual(result["win_rate"], 100.0)
        self.assertGreater(result["ci95"][0], 0)

    def test_paired_comparison_on_a_tie(self):
        values = np.ones(30)
        result = paired_comparison(values, values.copy(), n_boot=200)
        self.assertEqual(result["mean_delta"], 0.0)
        self.assertEqual(result["tie_rate"], 100.0)


class TestCollate(unittest.TestCase):
    def test_pads_and_shifts_the_target(self):
        from src.refinement import RefinementExample

        tok = make_tokenizer()
        batch = [
            RefinementExample([1, 2, 3], [KEEP, KEEP, KEEP], [1, 4, 5, 2], [], None, "D1"),
            RefinementExample([1, 2], [KEEP, KEEP], [1, 4, 2], [], None, "D1"),
        ]
        source, labels, target_in, target_out, _ = make_collate(tok.pad_id)(batch)

        self.assertEqual(source.shape, (2, 3))
        self.assertEqual(labels[1, 2], IGNORE_INDEX)  # padded label position
        self.assertEqual(target_in.shape, target_out.shape)
        self.assertEqual(target_in[0].tolist(), [1, 4, 5])
        self.assertEqual(target_out[0].tolist(), [4, 5, 2])


if __name__ == "__main__":
    unittest.main()
