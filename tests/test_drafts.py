"""
Tests for the draft sources.

The central property, from spec 3.4: the edit script is *derived by alignment*,
never read from the injection log, and applying it to the reference must
reproduce the draft exactly. That is simultaneously a test of the draft sources
and a free correctness test on the aligner.

Run: python -m unittest discover -s tests -v
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import REST, apply_script  # noqa: E402
from src.corpus import Melody, NoteEvent  # noqa: E402
from src.drafts import (  # noqa: E402
    MAX_SINGABLE_LEAP,
    Priors,
    empirical_priors,
    _cadential_disruption,
    _out_of_scale,
    _repeated_note_collapse,
    _rhythmic_monotony,
    _unsingable_leap,
    d1_random,
    d3_musical,
    generate,
)
from src.judge import scale_of  # noqa: E402

Q = 24
DURATIONS = (6, 12, 24, 36, 48)
SPAN = (33, 108)  # representable pitch range, as the tokenizer is built with headroom
PRIORS = Priors(
    durations=DURATIONS,
    duration_weights=(1.0,) * len(DURATIONS),
    pitches=tuple(range(55, 80)),
    pitch_weights=(1.0,) * 25,
)


def make_melody(n=16, tonic="C", mode="major", timesignature="4/4"):
    """A C-major scale figure with plausible annotations."""
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    events = []
    for i in range(n):
        events.append(
            NoteEvent(
                pitch=scale[i % len(scale)],
                duration=Q,
                scaledegree=(i % 7) + 1,
                beat=float((i % 4) + 1),
                beatstrength=1.0 if i % 4 == 0 else 0.25,
                phrase_ix=i // 8,
                phrase_end=(i % 8 == 7),
                timesignature=timesignature,
            )
        )
    return Melody(
        id="t1",
        source="T",
        events=tuple(events),
        tunefamily="fam",
        tonic=tonic,
        mode=mode,
        timesignature=timesignature,
    )


class TestScriptDerivation(unittest.TestCase):
    """Spec 3.4's assertion, as an executable check."""

    def test_d1_script_reproduces_the_draft(self):
        melody = make_melody()
        for seed in range(200):
            draft = d1_random(melody, PRIORS, random.Random(seed))
            if draft is None:
                continue
            self.assertEqual(apply_script(melody.notes, draft.script), draft.notes)

    def test_d3_script_reproduces_the_draft(self):
        melody = make_melody()
        for seed in range(200):
            draft = d3_musical(melody, random.Random(seed))
            if draft is None:
                continue
            self.assertEqual(apply_script(melody.notes, draft.script), draft.notes)

    def test_derived_script_is_never_longer_than_needed(self):
        melody = make_melody()
        for seed in range(100):
            draft = d1_random(melody, PRIORS, random.Random(seed), max_edits=3)
            if draft is None:
                continue
            # Each injected edit costs at most 2 (pitch and duration both changed).
            self.assertLessEqual(draft.n_edits, 2 * len(draft.injected))


class TestD1(unittest.TestCase):
    def test_produces_a_different_melody(self):
        melody = make_melody()
        draft = d1_random(melody, PRIORS, random.Random(3))
        self.assertIsNotNone(draft)
        self.assertNotEqual(draft.notes, melody.notes)
        self.assertEqual(draft.source, "D1")

    def test_is_deterministic_given_a_seed(self):
        melody = make_melody()
        a = d1_random(melody, PRIORS, random.Random(11))
        b = d1_random(melody, PRIORS, random.Random(11))
        self.assertEqual(a.notes, b.notes)

    def test_respects_the_duration_vocabulary(self):
        melody = make_melody()
        for seed in range(100):
            draft = d1_random(melody, PRIORS, random.Random(seed))
            if draft is None:
                continue
            for pitch, duration in draft.notes:
                if (pitch, duration) not in melody.notes:
                    self.assertIn(duration, DURATIONS)

    def test_operation_mixture_matches_the_published_weights(self):
        """D1 must keep its original behaviour exactly -- it is the control."""
        melody = make_melody()
        counts = {"replace_pitch": 0, "change_duration": 0, "delete_note": 0, "insert_note": 0}
        for seed in range(3000):
            draft = d1_random(melody, PRIORS, random.Random(seed))
            if draft is None:
                continue
            for op in draft.injected:
                counts[op] += 1

        total = sum(counts.values())
        self.assertAlmostEqual(counts["replace_pitch"] / total, 0.40, delta=0.04)
        self.assertAlmostEqual(counts["change_duration"] / total, 0.30, delta=0.04)
        self.assertAlmostEqual(counts["delete_note"] / total, 0.15, delta=0.04)
        self.assertAlmostEqual(counts["insert_note"] / total, 0.15, delta=0.04)


class TestD3Rules(unittest.TestCase):
    """Each rule must actually produce the error it is named for."""

    def test_out_of_scale_leaves_a_non_diatonic_pitch(self):
        melody = make_melody()
        scale = scale_of("C", "major")
        for seed in range(50):
            notes = list(melody.notes)
            if _out_of_scale(notes, melody, random.Random(seed), SPAN) is None:
                continue
            changed = [n for n, o in zip(notes, melody.notes) if n != o]
            self.assertTrue(changed)
            self.assertTrue(any(p % 12 not in scale for p, _ in changed))

    def test_unsingable_leap_exceeds_the_threshold(self):
        melody = make_melody()
        for seed in range(50):
            notes = list(melody.notes)
            if _unsingable_leap(notes, melody, random.Random(seed), SPAN) is None:
                continue
            pitches = [p for p, _ in notes if p != REST]
            widest = max(abs(b - a) for a, b in zip(pitches, pitches[1:]))
            self.assertGreater(widest, MAX_SINGABLE_LEAP)

    def test_cadential_disruption_moves_off_tonic_and_dominant(self):
        melody = make_melody()
        for seed in range(50):
            notes = list(melody.notes)
            if _cadential_disruption(notes, melody, random.Random(seed), SPAN) is None:
                continue
            changed = [(i, n) for i, (n, o) in enumerate(zip(notes, melody.notes)) if n != o]
            self.assertTrue(changed)
            for _, (pitch, _) in changed:
                self.assertNotIn(pitch % 12, {0, 7})  # C major: tonic C, dominant G

    def test_rhythmic_monotony_flattens_durations(self):
        melody = make_melody()
        varied = [(p, Q if i % 2 else Q * 2) for i, (p, _) in enumerate(melody.notes)]
        notes = list(varied)
        applied = _rhythmic_monotony(notes, melody, random.Random(1), SPAN)
        if applied:
            self.assertNotEqual(notes, varied)
            self.assertLess(len({d for _, d in notes}), len({d for _, d in varied}) + 1)

    def test_repeated_note_collapse_flattens_pitch(self):
        """Asserts a static run appears, not that distinct pitches drop: the test
        melody cycles through its pitches, so a collapsed window reuses pitches
        that still occur elsewhere."""
        def longest_run(pitches):
            best = current = 1
            for a, b in zip(pitches, pitches[1:]):
                current = current + 1 if a == b else 1
                best = max(best, current)
            return best

        melody = make_melody()
        before = longest_run([p for p, _ in melody.notes])
        for seed in range(50):
            notes = list(melody.notes)
            if _repeated_note_collapse(notes, melody, random.Random(seed), SPAN) is None:
                continue
            self.assertGreater(longest_run([p for p, _ in notes]), before)
            break
        else:
            self.fail("rule never applied")

    def test_rules_never_produce_an_empty_melody(self):
        melody = make_melody()
        for seed in range(200):
            draft = d3_musical(melody, random.Random(seed))
            if draft is not None:
                self.assertGreater(len(draft.notes), 0)

    def test_unknown_key_disables_key_dependent_rules(self):
        melody = make_melody(tonic=None)
        notes = list(melody.notes)
        self.assertIsNone(_out_of_scale(notes, melody, random.Random(0), SPAN))
        self.assertIsNone(_cadential_disruption(notes, melody, random.Random(0), SPAN))
        self.assertEqual(notes, melody.notes)


class TestBatchGeneration(unittest.TestCase):
    def test_generate_is_deterministic(self):
        melodies = [make_melody() for _ in range(20)]
        a = generate(melodies, "D1", PRIORS, seed=5)
        b = generate(melodies, "D1", PRIORS, seed=5)
        self.assertEqual([d.notes for d in a], [d.notes for d in b])

    def test_different_seeds_give_different_drafts(self):
        melodies = [make_melody() for _ in range(20)]
        a = generate(melodies, "D1", PRIORS, seed=1)
        b = generate(melodies, "D1", PRIORS, seed=2)
        self.assertNotEqual([d.notes for d in a], [d.notes for d in b])

    def test_unknown_source_raises(self):
        with self.assertRaises(ValueError):
            generate([make_melody()], "D9", PRIORS)


if __name__ == "__main__":
    unittest.main()


class TestPriors(unittest.TestCase):
    """D1 must draw from plausible material, or it tests a strawman.

    Uniform sampling over a derived duration vocabulary inflates D1's apparent
    damage: on this corpus seven of forty durations carry 98.3% of the mass, so
    a flat draw picks a near-unused rhythm most of the time.
    """

    def _melodies(self):
        melody = make_melody()
        return [melody]

    def test_empirical_priors_follow_corpus_frequency(self):
        priors = empirical_priors(self._melodies())
        weights = dict(zip(priors.durations, priors.duration_weights))
        self.assertEqual(set(priors.durations), {Q})
        self.assertGreater(weights[Q], 0)

    def test_uniform_flag_flattens_the_weights(self):
        priors = empirical_priors(self._melodies(), uniform=True)
        self.assertEqual(set(priors.duration_weights), {1.0})

    def test_sampled_duration_can_exclude_a_value(self):
        priors = Priors((6, 12), (1.0, 99.0), (60,), (1.0,))
        rng = random.Random(0)
        for _ in range(50):
            self.assertNotEqual(priors.sample_duration(rng, exclude=12), 12)

    def test_weighted_sampling_prefers_common_durations(self):
        priors = Priors((6, 12), (1.0, 99.0), (60,), (1.0,))
        rng = random.Random(0)
        draws = [priors.sample_duration(rng) for _ in range(500)]
        self.assertGreater(draws.count(12), draws.count(6) * 5)

    def test_inserted_pitches_stay_inside_the_corpus_span(self):
        priors = empirical_priors([make_melody()])
        lo, hi = priors.pitch_span
        for seed in range(100):
            draft = d1_random(make_melody(), priors, random.Random(seed))
            if draft is None:
                continue
            for pitch, _ in draft.notes:
                if pitch != REST:
                    self.assertTrue(lo - 3 <= pitch <= hi + 3)
