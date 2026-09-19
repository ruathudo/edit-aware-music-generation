"""
Tests for grammar-constrained sampling.

The property that matters: every sampled sequence decodes to exactly the events
the model chose, with no token silently dropped. If sampling could emit an
unpaired pitch, the tolerant decoder would discard it and the resulting D2 draft
would differ from its reference by a note the generator never decided to change
-- a decoding artifact masquerading as an edit.

Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.model import LMConfig, MelodyLM  # noqa: E402
from src.sampling import Grammar, sample_continuations  # noqa: E402
from src.tokenizer import MelodyTokenizer, TokenizerConfig  # noqa: E402


def make_tokenizer():
    return MelodyTokenizer(TokenizerConfig(pitch_min=60, pitch_max=72, durations=(6, 12, 24)))


def make_model(tokenizer, seed=0):
    torch.manual_seed(seed)
    config = LMConfig(
        vocab_size=len(tokenizer), d_model=32, n_heads=2, n_layers=2, d_ff=64, max_len=128
    )
    return MelodyLM(config, pad_id=tokenizer.pad_id).eval()


class TestGrammar(unittest.TestCase):
    def setUp(self):
        self.tok = make_tokenizer()
        self.grammar = Grammar(self.tok)

    def test_only_durations_may_follow_a_pitch(self):
        mask = self.grammar.next_mask(self.tok.pitch_id(60))
        allowed = {self.tok.id_to_token[i] for i in mask.nonzero().flatten().tolist()}
        self.assertTrue(all(t.startswith("D") for t in allowed))

    def test_only_durations_may_follow_a_rest(self):
        mask = self.grammar.next_mask(self.tok.rest_id)
        allowed = {self.tok.id_to_token[i] for i in mask.nonzero().flatten().tolist()}
        self.assertTrue(all(t.startswith("D") for t in allowed))

    def test_a_duration_may_be_followed_by_event_bar_or_end(self):
        mask = self.grammar.next_mask(self.tok.token_to_id["D12"])
        allowed = {self.tok.id_to_token[i] for i in mask.nonzero().flatten().tolist()}
        self.assertIn("<EOS>", allowed)
        self.assertIn("<BAR>", allowed)
        self.assertIn("P60", allowed)
        self.assertFalse(any(t.startswith("D") for t in allowed))

    def test_a_barline_must_be_followed_by_an_event(self):
        mask = self.grammar.next_mask(self.tok.bar_id)
        allowed = {self.tok.id_to_token[i] for i in mask.nonzero().flatten().tolist()}
        self.assertNotIn("<EOS>", allowed)
        self.assertNotIn("<BAR>", allowed)
        self.assertIn("P60", allowed)

    def test_bos_behaves_like_a_barline(self):
        after_bos = self.grammar.next_mask(self.tok.bos_id)
        self.assertTrue(torch.equal(after_bos, self.grammar.after_bar))


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.tok = make_tokenizer()
        self.model = make_model(self.tok)

    def _prefix(self, events):
        return self.tok.encode(events)[:-1]  # drop EOS so there is something to continue

    def test_output_is_always_well_formed(self):
        """No token may be dropped on decode -- that is the whole point."""
        prefixes = [self._prefix([(60, 12), (62, 12)]) for _ in range(8)]
        out = sample_continuations(self.model, self.tok, prefixes, max_new_tokens=40, seed=1)
        for tokens in out:
            events, stats = self.tok.decode(tokens, return_stats=True)
            self.assertEqual(stats.get("unpaired_pitch", 0), 0)
            self.assertEqual(stats.get("orphan_duration", 0), 0)
            self.assertGreater(len(events), 0)

    def test_prefix_is_preserved(self):
        events = [(60, 12), (62, 24), (64, 6)]
        prefix = self._prefix(events)
        out = sample_continuations(self.model, self.tok, [prefix], max_new_tokens=20, seed=2)
        self.assertEqual(out[0][: len(prefix)], prefix)

    def test_continuation_extends_the_prefix(self):
        prefix = self._prefix([(60, 12), (62, 12)])
        out = sample_continuations(self.model, self.tok, [prefix], max_new_tokens=20, seed=3)
        self.assertGreater(len(out[0]), len(prefix))

    def test_is_deterministic_given_a_seed(self):
        prefixes = [self._prefix([(60, 12), (62, 12)])] * 4
        a = sample_continuations(self.model, self.tok, prefixes, max_new_tokens=30, seed=5)
        b = sample_continuations(self.model, self.tok, prefixes, max_new_tokens=30, seed=5)
        self.assertEqual(a, b)

    def test_temperature_zero_is_greedy_and_repeatable(self):
        prefixes = [self._prefix([(60, 12)])] * 3
        out = sample_continuations(
            self.model, self.tok, prefixes, max_new_tokens=20, temperature=0.0, seed=9
        )
        self.assertEqual(out[0], out[1])
        self.assertEqual(out[1], out[2])

    def test_ragged_prefix_lengths_in_one_batch(self):
        """Batching must not let a short sequence read a longer one's padding."""
        prefixes = [
            self._prefix([(60, 12)]),
            self._prefix([(60, 12), (62, 12), (64, 12), (65, 12)]),
        ]
        out = sample_continuations(self.model, self.tok, prefixes, max_new_tokens=25, seed=7)
        for prefix, tokens in zip(prefixes, out):
            self.assertEqual(tokens[: len(prefix)], prefix)
            self.assertNotIn(self.tok.pad_id, tokens)

    def test_never_exceeds_the_model_context(self):
        prefix = self._prefix([(60, 12)] * 3)
        out = sample_continuations(self.model, self.tok, [prefix], max_new_tokens=500, seed=11)
        self.assertLessEqual(len(out[0]), self.model.config.max_len)

    def test_top_k_restricts_the_support(self):
        prefixes = [self._prefix([(60, 12), (62, 12)])] * 16
        out = sample_continuations(
            self.model, self.tok, prefixes, max_new_tokens=20, top_k=1, seed=4
        )
        self.assertEqual(len({tuple(t) for t in out}), 1)


if __name__ == "__main__":
    unittest.main()
