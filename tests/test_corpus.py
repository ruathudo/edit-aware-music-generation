"""
Tests for the MTC / Essen corpus loader.

Unit tests run on synthetic records so they need no download. The tests that do
need the corpora skip cleanly when `data/corpora/` is empty, since the data is
gitignored and redistribution is not permitted for either collection.

Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import REST, correction_cost  # noqa: E402
from src.corpus import (  # noqa: E402
    TICKS_PER_QUARTER,
    Melody,
    bar_ticks,
    NoteEvent,
    duration_vocabulary,
    load,
    melody_from_record,
    split_by_tunefamily,
)
from src.tokenizer import MelodyTokenizer, TokenizerConfig  # noqa: E402


def record(pitches, durations, rests=None, beats=None, tunefamily="fam", **kw):
    n = len(pitches)
    features = {
        "midipitch": pitches,
        "duration_frac": durations,
        "restduration_frac": rests or [None] * n,
        "beat": beats or [1.0] * n,
        "scaledegree": [1] * n,
        "beatstrength": [1.0] * n,
        "phrase_ix": [0] * n,
        "phrase_end": [False] * n,
        "tonic": ["C"] * n,
        "mode": ["major"] * n,
        "timesignature": ["4/4"] * n,
    }
    features.update(kw)
    return {"id": "X1", "tunefamily": tunefamily, "features": features}


class TestDurationConversion(unittest.TestCase):
    def test_quarter_is_one_tick_unit(self):
        m = melody_from_record(record([60], ["1"]), "T")
        self.assertEqual(m.events[0].duration, TICKS_PER_QUARTER)

    def test_triplets_are_exact(self):
        """The reason TPQ is 24: triplet eighths (1/3 quarter) must not round."""
        m = melody_from_record(record([60, 62, 64], ["1/3", "1/3", "1/3"]), "T")
        self.assertEqual([e.duration for e in m.events], [8, 8, 8])
        self.assertEqual(sum(e.duration for e in m.events), TICKS_PER_QUARTER)

    def test_common_divisions_are_exact(self):
        m = melody_from_record(record([60] * 5, ["1/2", "1/4", "1/8", "3/2", "1/6"]), "T")
        self.assertEqual([e.duration for e in m.events], [12, 6, 3, 36, 4])

    def test_unrepresentable_duration_raises(self):
        with self.assertRaises(ValueError):
            melody_from_record(record([60], ["1/5"]), "T")


class TestRestMaterialization(unittest.TestCase):
    """`restduration_frac[i]` is the rest *following* note i, not preceding it.

    Confirmed against the corpus's own `nextisrest` flag. Getting this backwards
    shifts every rest one position and drops bar accuracy by ~25 points, without
    breaking any round-trip test -- which is why these assert the direction
    explicitly.
    """

    def test_rest_becomes_an_event_after_its_note(self):
        m = melody_from_record(record([60, 64], ["1", "1"], rests=["1/2", None]), "T")
        self.assertEqual(
            [(e.pitch, e.duration) for e in m.events],
            [(60, 24), (REST, 12), (64, 24)],
        )

    def test_rests_carry_no_annotations(self):
        m = melody_from_record(record([60, 64], ["1", "1"], rests=["1", None]), "T")
        rest = m.events[1]
        self.assertTrue(rest.is_rest)
        self.assertIsNone(rest.scaledegree)
        self.assertIsNone(rest.beat)

    def test_trailing_rest_is_kept(self):
        m = melody_from_record(record([60], ["1"], rests=["1"]), "T")
        self.assertEqual(len(m.events), 2)
        self.assertTrue(m.events[1].is_rest)

    def test_n_sounding_excludes_rests(self):
        m = melody_from_record(record([60, 64], ["1", "1"], rests=["1", None]), "T")
        self.assertEqual(len(m), 3)
        self.assertEqual(m.n_sounding, 2)


class TestBarDetection(unittest.TestCase):
    def test_bar_starts_where_the_beat_resets(self):
        m = melody_from_record(record([60, 62, 64, 65], ["1"] * 4, beats=[1.0, 2.0, 1.0, 2.0]), "T")
        self.assertEqual(m.bar_starts, frozenset({2}))

    def test_pickup_bar_is_handled(self):
        """A melody starting on beat 4 has its first barline before the downbeat,
        which cumulative-duration inference would get wrong."""
        m = melody_from_record(record([60, 62, 64], ["1"] * 3, beats=[4.0, 1.0, 2.0]), "T")
        self.assertEqual(m.bar_starts, frozenset({1}))

    def test_rest_stays_in_its_own_notes_bar(self):
        m = melody_from_record(
            record([60, 64], ["1", "1"], rests=["1", None], beats=[1.0, 1.0]), "T"
        )
        # events: note, its trailing rest, then the next bar's note at index 2.
        self.assertEqual(m.bar_starts, frozenset({2}))

    def test_missing_beats_yield_no_bars(self):
        m = melody_from_record(record([60, 62], ["1", "1"], beats=[None, None]), "T")
        self.assertEqual(m.bar_starts, frozenset())


class TestSplitting(unittest.TestCase):
    def _melody(self, i, family):
        return Melody(id=f"m{i}", source="T", events=(NoteEvent(60, 24),), tunefamily=family)

    def test_no_tune_family_spans_two_splits(self):
        """A random split would leak variants of a test melody into training."""
        melodies = [self._melody(i, f"fam{i % 10}") for i in range(200)]
        splits = split_by_tunefamily(melodies, {"train": 0.6, "val": 0.2, "test": 0.2})

        seen: dict[str, str] = {}
        for name, group in splits.items():
            for m in group:
                self.assertEqual(seen.setdefault(m.tunefamily, name), name)

    def test_every_melody_lands_somewhere_exactly_once(self):
        melodies = [self._melody(i, f"fam{i % 7}") for i in range(100)]
        splits = split_by_tunefamily(melodies, {"a": 0.5, "b": 0.5})
        ids = [m.id for g in splits.values() for m in g]
        self.assertEqual(sorted(ids), sorted(m.id for m in melodies))

    def test_unfamilied_melodies_are_singletons(self):
        melodies = [self._melody(i, None) for i in range(50)]
        splits = split_by_tunefamily(melodies, {"train": 0.8, "test": 0.2})
        self.assertEqual(sum(len(g) for g in splits.values()), 50)
        self.assertTrue(len(splits["test"]) > 0)

    def test_split_is_deterministic(self):
        melodies = [self._melody(i, f"fam{i % 9}") for i in range(90)]
        a = split_by_tunefamily(melodies, {"train": 0.7, "test": 0.3}, seed=7)
        b = split_by_tunefamily(melodies, {"train": 0.7, "test": 0.3}, seed=7)
        self.assertEqual([m.id for m in a["test"]], [m.id for m in b["test"]])


class TestRealCorpus(unittest.TestCase):
    """Smoke tests against the downloaded corpora; skipped when absent."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.mels = load("MTC-ANN", limit=300)
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc))

    def test_loads_and_is_well_formed(self):
        self.assertGreater(len(self.mels), 100)
        for m in self.mels[:50]:
            self.assertGreater(len(m.events), 0)
            self.assertTrue(all(e.duration > 0 for e in m.events))
            self.assertTrue(all(e.is_rest or 0 < e.pitch < 128 for e in m.events))

    def test_tune_families_are_present(self):
        families = {m.tunefamily for m in self.mels}
        self.assertGreater(len(families), 5)
        self.assertNotIn(None, families)

    def test_round_trips_through_the_tokenizer(self):
        vocab = duration_vocabulary(self.mels)
        lo = min(e.pitch for m in self.mels for e in m.events if not e.is_rest)
        hi = max(e.pitch for m in self.mels for e in m.events if not e.is_rest)
        tok = MelodyTokenizer(TokenizerConfig(pitch_min=lo, pitch_max=hi, durations=vocab))

        for m in self.mels[:200]:
            tokens = tok.encode(m.notes, m.bar_starts)
            self.assertEqual(tok.decode(tokens), m.notes)

    def test_barlines_are_emitted_from_annotations(self):
        vocab = duration_vocabulary(self.mels)
        lo, hi = 21, 108
        tok = MelodyTokenizer(TokenizerConfig(pitch_min=lo, pitch_max=hi, durations=vocab))
        barred = [m for m in self.mels if m.bar_starts][:20]
        self.assertTrue(barred, "expected some melodies to have barlines")
        for m in barred:
            tokens = tok.encode(m.notes, m.bar_starts)
            self.assertEqual(tokens.count(tok.bar_id), len(m.bar_starts - {0}))

    def test_decoded_melodies_align_against_themselves_at_zero_cost(self):
        for m in self.mels[:100]:
            self.assertEqual(correction_cost(m.notes, m.notes), 0.0)

    def test_detected_bars_hold_the_duration_their_metre_implies(self):
        """The regression guard for rest placement.

        Checks the representation against real musical structure rather than
        against itself: an interior bar in a fixed metre must contain exactly one
        bar's worth of ticks. This is what exposed rests being attached to the
        wrong note -- accuracy sat at 76% until the direction was corrected, while
        every round-trip test still passed.
        """
        ok = bad = 0
        for m in self.mels:
            if m.freemeter or not m.bar_starts:
                continue
            starts = sorted(m.bar_starts)
            for a, b in zip(starts, starts[1:]):
                expected = bar_ticks(m.events[a].timesignature or m.timesignature)
                if not expected:
                    continue
                total = sum(e.duration for e in m.events[a:b])
                ok += total == expected
                bad += total != expected

        self.assertGreater(ok + bad, 500, "expected a meaningful number of interior bars")
        self.assertGreater(ok / (ok + bad), 0.95, f"bar accuracy {ok}/{ok + bad}")


if __name__ == "__main__":
    unittest.main()
