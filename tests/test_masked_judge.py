import random
import unittest

import numpy as np
import torch

from src.masked_judge import (
    MaskedLMConfig,
    MaskedMelodyLM,
    event_groups,
    mask_events,
    score_events,
)


class EventGroupsTest(unittest.TestCase):
    def test_each_event_masks_pitch_and_duration_together_and_eos_alone(self):
        tokens = [1, 10, 50, 3, 11, 51, 2]      # BOS, ev, BAR, ev, EOS
        spans = [(1, 2), (4, 5)]
        self.assertEqual(event_groups(tokens, spans, eos_id=2), [[1, 2], [4, 5], [6]])

    def test_no_eos_group_when_sequence_lacks_eos(self):
        self.assertEqual(event_groups([1, 10, 50], [(1, 2)], eos_id=2), [[1, 2]])


class MaskEventsTest(unittest.TestCase):
    def test_masks_whole_events_and_targets_only_masked_positions(self):
        tokens = [1, 10, 50, 11, 51, 2]
        groups = [[1, 2], [3, 4], [5]]
        for seed in range(50):
            masked, targets = mask_events(tokens, groups, 0.3, 99, random.Random(seed))
            for group in groups:
                hidden = [masked[i] == 99 for i in group]
                self.assertTrue(all(hidden) or not any(hidden), "an event was half-masked")
            for i, (m, t) in enumerate(zip(masked, targets)):
                self.assertEqual(t != -100, m == 99)
                if t != -100:
                    self.assertEqual(t, tokens[i])

    def test_always_masks_at_least_one_event(self):
        masked, _ = mask_events([1, 10, 50, 2], [[1, 2], [3]], 0.0, 99, random.Random(0))
        self.assertIn(99, masked)


class ScoreEventsTest(unittest.TestCase):
    def test_scores_are_joint_nll_shared_by_both_tokens_of_an_event(self):
        torch.manual_seed(0)
        model = MaskedMelodyLM(MaskedLMConfig(vocab_size=20, d_model=16, n_heads=2,
                                              n_layers=1, d_ff=32), pad_id=0)
        tokens = [1, 5, 12, 6, 13, 2]
        groups = [[1, 2], [3, 4], [5]]
        scores = score_events(model, tokens, groups, batch_size=2, device="cpu")
        self.assertEqual(scores[0], 0.0)                 # BOS is never scored
        self.assertEqual(scores[1], scores[2])
        self.assertEqual(scores[3], scores[4])
        self.assertTrue(np.all(scores[1:] > 0))

        # Joint score equals the sum of the two tokens' -log p with the event masked.
        x = torch.tensor([tokens])
        x[0, [1, 2]] = model.config.mask_id
        logp = model(x).log_softmax(-1)[0]
        expected = -(logp[1, 5] + logp[2, 12]).item()
        self.assertAlmostEqual(scores[1], expected, places=5)


if __name__ == "__main__":
    unittest.main()
