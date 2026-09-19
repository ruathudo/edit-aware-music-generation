"""
Tests for the judge's musical metrics and the LM plumbing.

The metrics are pure functions over note lists, so they test without a trained
model. Anything needing weights or the corpora skips cleanly.

Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.align import REST  # noqa: E402
from src.corpus import infer_bar_starts  # noqa: E402
from src.judge import (  # noqa: E402
    MAJOR_SCALE,
    interval_histogram,
    l1_distance,
    max_interval,
    metric_alignment,
    pitch_class_histogram,
    scale_conformance,
    scale_of,
    sounding,
    tonic_pitch_class,
)
from src.model import LMConfig, MelodyLM, token_nll  # noqa: E402

Q = 24  # ticks per quarter


class TestKeyParsing(unittest.TestCase):
    def test_natural_tonics(self):
        self.assertEqual(tonic_pitch_class("C"), 0)
        self.assertEqual(tonic_pitch_class("A"), 9)

    def test_flat_is_a_hyphen(self):
        """The corpus writes B-flat as 'B-', not 'Bb'."""
        self.assertEqual(tonic_pitch_class("B-"), 10)
        self.assertEqual(tonic_pitch_class("A-"), 8)

    def test_sharp_and_wraparound(self):
        self.assertEqual(tonic_pitch_class("B#"), 0)
        self.assertEqual(tonic_pitch_class("C-"), 11)

    def test_unknown_is_none(self):
        self.assertIsNone(tonic_pitch_class(None))
        self.assertIsNone(tonic_pitch_class("H"))

    def test_c_major_is_the_white_notes(self):
        self.assertEqual(scale_of("C", "major"), frozenset(MAJOR_SCALE))

    def test_minor_admits_the_raised_seventh(self):
        """Folk minor melodies use a leading tone constantly; excluding it would
        score ordinary cadences as out-of-scale."""
        a_minor = scale_of("A", "minor")
        self.assertIn(8, a_minor)   # G# leading tone
        self.assertIn(7, a_minor)   # G natural


class TestScaleConformance(unittest.TestCase):
    def test_diatonic_melody_scores_one(self):
        notes = [(60, Q), (62, Q), (64, Q), (65, Q), (67, Q)]
        self.assertEqual(scale_conformance(notes, "C", "major"), 1.0)

    def test_chromatic_notes_lower_the_score(self):
        notes = [(60, Q), (61, Q), (62, Q), (63, Q)]  # two out of key
        self.assertAlmostEqual(scale_conformance(notes, "C", "major"), 0.5)

    def test_rests_are_not_counted(self):
        notes = [(60, Q), (REST, Q), (64, Q)]
        self.assertEqual(scale_conformance(notes, "C", "major"), 1.0)

    def test_unknown_key_is_none(self):
        self.assertIsNone(scale_conformance([(60, Q)], None, None))

    def test_empty_is_none(self):
        self.assertIsNone(scale_conformance([], "C", "major"))


class TestHistograms(unittest.TestCase):
    def test_pitch_class_histogram_is_duration_weighted(self):
        hist = pitch_class_histogram([(60, 3 * Q), (62, Q)])
        self.assertAlmostEqual(hist[0], 0.75)
        self.assertAlmostEqual(hist[2], 0.25)
        self.assertAlmostEqual(hist.sum(), 1.0)

    def test_octave_equivalence(self):
        a = pitch_class_histogram([(60, Q)])
        b = pitch_class_histogram([(72, Q)])
        self.assertEqual(l1_distance(a, b), 0.0)

    def test_identical_melodies_have_zero_distance(self):
        notes = [(60, Q), (64, Q), (67, Q)]
        self.assertEqual(l1_distance(pitch_class_histogram(notes), pitch_class_histogram(notes)), 0.0)
        self.assertEqual(l1_distance(interval_histogram(notes), interval_histogram(notes)), 0.0)

    def test_intervals_skip_rests(self):
        """A rest should not break the melodic line into a spurious interval."""
        with_rest = interval_histogram([(60, Q), (REST, Q), (64, Q)])
        without = interval_histogram([(60, Q), (64, Q)])
        self.assertEqual(l1_distance(with_rest, without), 0.0)

    def test_max_interval(self):
        self.assertEqual(max_interval([(60, Q), (72, Q), (65, Q)]), 12)
        self.assertEqual(max_interval([(60, Q)]), 0)
        self.assertEqual(max_interval([]), 0)

    def test_sounding_filters_rests(self):
        self.assertEqual(sounding([(60, Q), (REST, Q)]), [(60, Q)])


class TestMetricAlignment(unittest.TestCase):
    def test_downbeats_score_highest(self):
        """Four quarter notes in 4/4: one downbeat, three ordinary beats."""
        notes = [(60, Q)] * 4
        score = metric_alignment(notes, "4/4")
        self.assertAlmostEqual(score, (1.0 + 0.5 + 0.5 + 0.5) / 4)

    def test_syncopation_scores_lower_than_on_beat(self):
        on_beat = metric_alignment([(60, Q)] * 4, "4/4")
        off_beat = metric_alignment([(60, Q // 2)] + [(60, Q)] * 3 + [(60, Q // 2)], "4/4")
        self.assertLess(off_beat, on_beat)

    def test_longer_notes_dominate(self):
        """Duration weighting is what makes this structural rather than a count."""
        long_downbeat = metric_alignment([(60, 3 * Q), (62, Q)], "4/4")
        short_downbeat = metric_alignment([(60, Q), (62, 3 * Q)], "4/4")
        self.assertGreater(long_downbeat, short_downbeat)

    def test_pickup_offset_shifts_the_grid(self):
        # Durations must be unequal: four equal quarters in 4/4 are rotationally
        # symmetric, so an offset would permute the strengths without changing
        # the weighted mean.
        notes = [(60, 3 * Q), (62, Q)]
        aligned = metric_alignment(notes, "4/4", start_offset=0)
        shifted = metric_alignment(notes, "4/4", start_offset=Q)
        self.assertAlmostEqual(aligned, 0.875)
        self.assertAlmostEqual(shifted, 0.625)
        self.assertGreater(aligned, shifted)

    def test_unknown_metre_is_none(self):
        self.assertIsNone(metric_alignment([(60, Q)], None))
        self.assertIsNone(metric_alignment([(60, Q)], "free"))


class TestInferredBars(unittest.TestCase):
    def test_bars_fall_every_metre_width(self):
        notes = [(60, Q)] * 8  # two bars of 4/4
        self.assertEqual(infer_bar_starts(notes, "4/4"), frozenset({4}))

    def test_pickup_offset_moves_the_first_barline(self):
        notes = [(60, Q)] * 8
        # starting one quarter into the bar, the downbeat arrives after 3 notes
        self.assertEqual(infer_bar_starts(notes, "4/4", start_offset=Q), frozenset({3, 7}))

    def test_no_metre_yields_no_bars(self):
        self.assertEqual(infer_bar_starts([(60, Q)] * 8, None), frozenset())

    def test_rests_occupy_time(self):
        notes = [(60, 2 * Q), (REST, 2 * Q), (62, Q)]
        self.assertEqual(infer_bar_starts(notes, "4/4"), frozenset({2}))


class TestLanguageModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = LMConfig(vocab_size=40, d_model=32, n_heads=2, n_layers=2, d_ff=64, max_len=64)
        self.model = MelodyLM(self.config, pad_id=0).eval()

    def test_output_shape(self):
        tokens = torch.randint(1, 40, (3, 17))
        self.assertEqual(self.model(tokens).shape, (3, 17, 40))

    def test_is_causal(self):
        """A change at position t must not alter logits before t."""
        tokens = torch.randint(1, 40, (1, 12))
        with torch.no_grad():
            base = self.model(tokens)
            altered = tokens.clone()
            altered[0, 8] = (altered[0, 8] % 38) + 1
            after = self.model(altered)
        torch.testing.assert_close(base[:, :8], after[:, :8], rtol=1e-4, atol=1e-5)

    def test_nll_ignores_padding(self):
        """Appending pad must change neither the total NLL nor the token count."""
        tokens = torch.randint(1, 40, (1, 10))
        padded = torch.cat([tokens, torch.zeros(1, 6, dtype=torch.long)], dim=1)
        with torch.no_grad():
            total_a, count_a = token_nll(self.model, tokens)
            total_b, count_b = token_nll(self.model, padded)
        self.assertEqual(int(count_a.item()), int(count_b.item()))
        self.assertAlmostEqual(float(total_a), float(total_b), places=3)

    def test_nll_counts_predicted_tokens(self):
        tokens = torch.randint(1, 40, (2, 11))
        _, count = token_nll(self.model, tokens)
        self.assertTrue(torch.all(count == 10))

    def test_rejects_overlong_sequences(self):
        with self.assertRaises(ValueError):
            self.model(torch.randint(1, 40, (1, 65)))

    def test_hidden_states_shape(self):
        tokens = torch.randint(1, 40, (2, 9))
        with torch.no_grad():
            self.assertEqual(self.model.hidden_states(tokens).shape, (2, 9, 32))


if __name__ == "__main__":
    unittest.main()
