"""
Unit tests for the Wagner-Fischer aligner.

The aligner underpins ground-truth edit labels, Correction Cost and fidelity, so
a bug here would silently corrupt every number in the paper. These run before
anything else in the pipeline and need no corpus and no torch -- alignment is
schema-dependent, not corpus-dependent.

Run: python -m unittest discover -s tests -v
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import (  # noqa: E402
    DELETE,
    INSERT_BEFORE,
    KEEP,
    SUB_DUR,
    SUB_PITCH,
    CostVector,
    align,
    apply_script,
    correction_cost,
    edit_labels,
    edit_script,
)


def random_melody(rng, length=None):
    n = length if length is not None else rng.randint(0, 12)
    return [(rng.randint(21, 108), rng.choice([1, 2, 3, 4, 6, 8])) for _ in range(n)]


class TestCorrectionCost(unittest.TestCase):
    def test_identity_is_free(self):
        melody = [(60, 1), (62, 2), (64, 4)]
        self.assertEqual(correction_cost(melody, melody), 0.0)
        self.assertEqual(edit_script(melody, melody), [])

    def test_empty_sequences(self):
        self.assertEqual(correction_cost([], []), 0.0)
        self.assertEqual(correction_cost([(60, 1), (62, 1)], []), 2.0)
        self.assertEqual(correction_cost([], [(60, 1), (62, 1)]), 2.0)

    def test_pitch_only_substitution_costs_one(self):
        self.assertEqual(correction_cost([(60, 1)], [(67, 1)]), 1.0)

    def test_duration_only_substitution_costs_one(self):
        self.assertEqual(correction_cost([(60, 1)], [(60, 4)]), 1.0)

    def test_both_attributes_differ_costs_two(self):
        self.assertEqual(correction_cost([(60, 1)], [(67, 4)]), 2.0)

    def test_weighted_cost_vector_is_honoured(self):
        costs = CostVector(delete=1.0, insert=1.0, pitch_sub=5.0, dur_sub=0.5)
        self.assertEqual(correction_cost([(60, 1)], [(60, 4)], costs), 0.5)
        # A pitch substitution at 5.0 is dearer than delete+insert at 2.0.
        self.assertEqual(correction_cost([(60, 1)], [(67, 1)], costs), 2.0)


class TestEditScript(unittest.TestCase):
    def test_substitution_emits_typed_ops(self):
        script = edit_script([(60, 1)], [(67, 4)])
        self.assertEqual([op.op for op in script], ["sub_pitch", "sub_dur"])
        self.assertEqual(script[0].payload, (60, 67))
        self.assertEqual(script[1].payload, (1, 4))

    def test_deletion_is_localized(self):
        script = edit_script([(60, 1), (62, 2), (64, 4)], [(60, 1), (64, 4)])
        self.assertEqual([op.op for op in script], ["delete"])
        self.assertEqual(script[0].src_index, 1)
        self.assertEqual(script[0].payload, (62, 2))

    def test_insertion_is_localized(self):
        script = edit_script([(60, 1), (64, 4)], [(60, 1), (62, 2), (64, 4)])
        self.assertEqual([op.op for op in script], ["insert"])
        self.assertEqual(script[0].src_index, 1)
        self.assertEqual(script[0].payload, (62, 2))

    def test_trailing_insertion_attaches_past_the_end(self):
        script = edit_script([(60, 1)], [(60, 1), (62, 2)])
        self.assertEqual([op.op for op in script], ["insert"])
        self.assertEqual(script[0].src_index, 1)

    def test_script_is_ordered_by_position(self):
        rng = random.Random(0)
        for _ in range(200):
            source, target = random_melody(rng), random_melody(rng)
            keys = [op.sort_key() for op in edit_script(source, target)]
            self.assertEqual(keys, sorted(keys))

    def test_round_trip_reproduces_target(self):
        """The strongest property: the derived script, applied, must yield the target."""
        rng = random.Random(1)
        for _ in range(500):
            source, target = random_melody(rng), random_melody(rng)
            self.assertEqual(apply_script(source, edit_script(source, target)), target)

    def test_script_cost_matches_correction_cost(self):
        """Under unit costs each op costs 1, so script length must equal CC."""
        rng = random.Random(2)
        for _ in range(500):
            source, target = random_melody(rng), random_melody(rng)
            self.assertEqual(len(edit_script(source, target)), correction_cost(source, target))


class TestEditLabels(unittest.TestCase):
    def test_length_includes_eos_slot(self):
        rng = random.Random(3)
        for _ in range(100):
            source, target = random_melody(rng), random_melody(rng)
            self.assertEqual(len(edit_labels(source, target)), len(source) + 1)

    def test_identity_is_all_keep(self):
        melody = [(60, 1), (62, 2)]
        self.assertEqual(edit_labels(melody, melody), [KEEP, KEEP, KEEP])

    def test_class_assignment(self):
        self.assertEqual(edit_labels([(60, 1)], [(67, 1)]), [SUB_PITCH, KEEP])
        self.assertEqual(edit_labels([(60, 1)], [(60, 4)]), [SUB_DUR, KEEP])
        self.assertEqual(edit_labels([(60, 1)], [(67, 4)]), [SUB_PITCH, KEEP])

    def test_deletion_and_insertion_classes(self):
        labels = edit_labels([(60, 1), (62, 2)], [(60, 1)])
        self.assertEqual(labels, [KEEP, DELETE, KEEP])

        labels = edit_labels([(60, 1), (64, 4)], [(60, 1), (62, 2), (64, 4)])
        self.assertEqual(labels, [KEEP, INSERT_BEFORE, KEEP])

    def test_trailing_insertion_lands_on_eos_slot(self):
        self.assertEqual(edit_labels([(60, 1)], [(60, 1), (62, 2)]), [KEEP, INSERT_BEFORE])

    def test_insertion_outranks_substitution(self):
        # Something is missing before position 0 *and* position 0 differs.
        labels = edit_labels([(64, 4)], [(60, 1), (67, 4)])
        self.assertEqual(labels[0], INSERT_BEFORE)

    def test_labels_never_leave_the_vocabulary(self):
        rng = random.Random(4)
        for _ in range(300):
            source, target = random_melody(rng), random_melody(rng)
            self.assertTrue(set(edit_labels(source, target)) <= {0, 1, 2, 3, 4})


class TestLegacyAgreement(unittest.TestCase):
    """Guards against the rebuild silently changing the published metric."""

    def test_matches_legacy_minimal_edit_distance(self):
        try:
            from src.train import minimal_edit_distance
        except ImportError as exc:
            self.skipTest(f"legacy pipeline unavailable: {exc}")

        rng = random.Random(5)
        for _ in range(300):
            source, target = random_melody(rng), random_melody(rng)
            legacy, _ = minimal_edit_distance(source, target)
            self.assertEqual(
                correction_cost(source, target),
                legacy,
                msg=f"diverged on {source} -> {target}",
            )

    def test_agrees_with_injected_d1_scripts(self):
        """Spec 3.4: derive the script by alignment and assert it agrees with the
        injected one. Costs may differ (alignment finds the *minimal* script, the
        injected one need not be minimal), but applying either must give M_0."""
        try:
            from src.train import simulate_edits
        except ImportError as exc:
            self.skipTest(f"legacy pipeline unavailable: {exc}")

        rng = random.Random(6)
        for seed in range(200):
            clean = random_melody(rng, length=rng.randint(6, 16))
            corrupted, injected = simulate_edits(clean, max_edits=5, seed=seed)

            derived = edit_script(clean, corrupted)
            self.assertEqual(apply_script(clean, derived), corrupted)
            self.assertLessEqual(
                correction_cost(clean, corrupted),
                2 * len(injected),
                msg="derived script should never exceed the injected one's cost",
            )


class TestAlignmentInternals(unittest.TestCase):
    def test_ties_prefer_diagonal(self):
        """Both attributes differing: sub costs 2 and delete+insert costs 2.
        Preferring the diagonal keeps it one substitution, not two structural ops."""
        _, moves = align([(60, 1)], [(67, 4)])
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0][0], 0)  # _DIAG

    def test_moves_cover_both_sequences(self):
        rng = random.Random(7)
        for _ in range(200):
            source, target = random_melody(rng), random_melody(rng)
            _, moves = align(source, target)
            consumed_src = sum(1 for kind, _, _ in moves if kind in (0, 1))
            consumed_tgt = sum(1 for kind, _, _ in moves if kind in (0, 2))
            self.assertEqual(consumed_src, len(source))
            self.assertEqual(consumed_tgt, len(target))


if __name__ == "__main__":
    unittest.main()
