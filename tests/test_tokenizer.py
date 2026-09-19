"""
Unit tests for the rest- and bar-aware tokenizer.

The properties that matter: encode/decode round-trips exactly, spans point at the
right tokens (replacing the `1 + 2*t` arithmetic that REST and BAR break), rests
survive, and bar placement follows the metre rather than a note count.

Run: python -m unittest discover -s tests -v
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import REST, correction_cost, edit_script  # noqa: E402
from src.tokenizer import (  # noqa: E402
    BAR,
    MelodyTokenizer,
    TokenizerConfig,
    corpus_stats,
    frames_to_events,
)


def random_events(rng, n=None, allow_rest=True):
    n = n if n is not None else rng.randint(0, 14)
    out = []
    for _ in range(n):
        pitch = REST if (allow_rest and rng.random() < 0.2) else rng.randint(21, 108)
        out.append((pitch, rng.randint(1, 8)))
    return out


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tok = MelodyTokenizer()

    def test_decode_inverts_encode(self):
        rng = random.Random(0)
        for _ in range(500):
            events = random_events(rng)
            self.assertEqual(self.tok.decode(self.tok.encode(events)), events)

    def test_rests_survive_the_round_trip(self):
        events = [(60, 2), (REST, 4), (64, 1)]
        self.assertEqual(self.tok.decode(self.tok.encode(events)), events)

    def test_empty_sequence(self):
        tokens = self.tok.encode([])
        self.assertEqual(tokens, [self.tok.bos_id, self.tok.eos_id])
        self.assertEqual(self.tok.decode(tokens), [])

    def test_brackets_are_bos_and_eos(self):
        tokens = self.tok.encode([(60, 1)])
        self.assertEqual(tokens[0], self.tok.bos_id)
        self.assertEqual(tokens[-1], self.tok.eos_id)


class TestSpans(unittest.TestCase):
    def setUp(self):
        self.tok = MelodyTokenizer()

    def test_spans_point_at_the_right_tokens(self):
        """The property that replaces `1 + 2*t`: spans must locate every event
        correctly even with BAR and REST tokens interleaved."""
        rng = random.Random(1)
        for _ in range(400):
            events = random_events(rng)
            tokens, spans = self.tok.encode_with_spans(events)

            self.assertEqual(len(spans), len(events))
            for (pitch, dur), (p_idx, d_idx) in zip(events, spans):
                self.assertEqual(tokens[p_idx], self.tok.pitch_id(pitch))
                self.assertEqual(tokens[d_idx], self.tok.token_to_id[f"D{dur}"])

    def test_legacy_arithmetic_is_wrong_once_bars_appear(self):
        """Guards the reason spans exist: with a BAR token in the stream, the old
        `1 + 2*t` index no longer lands on note t."""
        events = [(60, 8), (62, 8), (64, 4)]  # third event starts bar 1
        tokens, spans = self.tok.encode_with_spans(events)
        self.assertIn(self.tok.bar_id, tokens)
        self.assertNotEqual(spans[2][0], 1 + 2 * 2)
        self.assertEqual(tokens[spans[2][0]], self.tok.pitch_id(64))


class TestBarPlacement(unittest.TestCase):
    def setUp(self):
        self.tok = MelodyTokenizer(TokenizerConfig(ticks_per_bar=16))

    def test_no_bar_token_before_the_first_bar(self):
        tokens = self.tok.encode([(60, 4), (62, 4)])
        self.assertNotIn(self.tok.bar_id, tokens)

    def test_bar_emitted_at_the_boundary(self):
        # Two events of 8 ticks fill bar 0; the third starts bar 1.
        tokens = self.tok.encode([(60, 8), (62, 8), (64, 4)])
        self.assertEqual(tokens.count(self.tok.bar_id), 1)
        self.assertEqual(tokens[5], self.tok.bar_id)

    def test_event_spanning_a_barline_belongs_to_its_starting_bar(self):
        # A 12-tick event starting at tick 8 crosses into bar 1 but starts in bar 0.
        tokens = self.tok.encode([(60, 8), (62, 12), (64, 1)])
        ids = [self.tok.id_to_token[t] for t in tokens]
        self.assertEqual(ids.index(BAR), ids.index("P64") - 1)

    def test_bars_can_be_disabled(self):
        tok = MelodyTokenizer(TokenizerConfig(use_bar=False))
        tokens = tok.encode([(60, 8), (62, 8), (64, 4)])
        self.assertNotIn(tok.bar_id, tokens)

    def test_bar_tokens_do_not_reach_the_event_sequence(self):
        """Bars are token-level only, so the aligner's note schema is untouched."""
        events = [(60, 8), (62, 8), (64, 4)]
        self.assertEqual(self.tok.decode(self.tok.encode(events)), events)


class TestMalformedInput(unittest.TestCase):
    def setUp(self):
        self.tok = MelodyTokenizer()

    def test_unpaired_pitch_is_skipped_not_raised(self):
        tokens = [self.tok.bos_id, self.tok.pitch_id(60), self.tok.eos_id]
        events, stats = self.tok.decode(tokens, return_stats=True)
        self.assertEqual(events, [])
        self.assertEqual(stats["unpaired_pitch"], 1)

    def test_orphan_duration_is_counted(self):
        tokens = [self.tok.bos_id, self.tok.token_to_id["D4"], self.tok.eos_id]
        events, stats = self.tok.decode(tokens, return_stats=True)
        self.assertEqual(events, [])
        self.assertEqual(stats["orphan_duration"], 1)

    def test_pad_is_ignored(self):
        events = [(60, 2), (REST, 1)]
        tokens = self.tok.encode(events) + [self.tok.pad_id] * 5
        self.assertEqual(self.tok.decode(tokens), events)

    def test_out_of_vocabulary_input_raises_on_encode(self):
        tok = MelodyTokenizer(TokenizerConfig(durations=(1, 2, 4)))
        with self.assertRaises(ValueError):
            tok.encode([(60, 3)])
        with self.assertRaises(ValueError):
            tok.encode([(200, 1)])
        self.assertFalse(tok.supports([(60, 3)]))
        self.assertTrue(tok.supports([(60, 2)]))


class TestFrameParsing(unittest.TestCase):
    def test_rests_become_events(self):
        # pitch 60 held 2, then rest 3, then pitch 64
        frames = [60, 128, 129, 129, 129, 64]
        self.assertEqual(frames_to_events(frames), [(60, 2), (REST, 3), (64, 1)])

    def test_repeated_pitch_is_two_onsets(self):
        self.assertEqual(frames_to_events([59, 59]), [(59, 1), (59, 1)])

    def test_parse_accounts_for_every_frame(self):
        """The legacy parser dropped rest frames, compressing time. This one must
        conserve total duration exactly."""
        rng = random.Random(2)
        for _ in range(300):
            frames = [rng.choice([60, 62, 64, 128, 129]) for _ in range(64)]
            if frames[0] == 128:
                frames[0] = 60  # a leading hold has nothing to extend
            self.assertEqual(sum(d for _, d in frames_to_events(frames)), 64)


class TestCorpusStats(unittest.TestCase):
    def test_retention_falls_faster_than_coverage(self):
        """One out-of-vocabulary duration rejects a whole melody, which is why the
        legacy allowlist lost 37% of melodies while covering 97% of tokens."""
        sequences = [[(60, 1)] * 9 + [(60, 5)] for _ in range(10)]
        stats = corpus_stats(sequences, TokenizerConfig(durations=(1, 2, 3, 4)))
        self.assertAlmostEqual(stats["duration_coverage"], 0.9)
        self.assertEqual(stats["melody_retention"], 0.0)

    def test_full_vocabulary_retains_everything(self):
        sequences = [[(60, 1), (REST, 5), (64, 7)]]
        stats = corpus_stats(sequences)
        self.assertEqual(stats["melody_retention"], 1.0)
        self.assertAlmostEqual(stats["rest_fraction"], 1 / 3)


class TestAlignerInterop(unittest.TestCase):
    """The tokenizer must not disturb the note schema the aligner works on."""

    def test_rests_are_alignable_events(self):
        source = [(60, 1), (REST, 2), (64, 1)]
        target = [(60, 1), (62, 2), (64, 1)]
        self.assertEqual(correction_cost(source, target), 1.0)
        script = edit_script(source, target)
        self.assertEqual([op.op for op in script], ["sub_pitch"])
        self.assertEqual(script[0].payload, (REST, 62))

    def test_decoded_output_feeds_the_aligner(self):
        rng = random.Random(3)
        tok = MelodyTokenizer()
        for _ in range(200):
            source, target = random_events(rng), random_events(rng)
            decoded = tok.decode(tok.encode(source))
            self.assertEqual(correction_cost(decoded, target), correction_cost(source, target))


if __name__ == "__main__":
    unittest.main()
