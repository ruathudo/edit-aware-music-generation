"""
Autoregressive sampling from a melody LM, constrained to the token grammar.

The grammar is fixed by the encoding: a pitch or REST is always followed by a
duration, a duration may be followed by another event, a barline, or EOS, and a
barline is followed by an event. Sampling is masked to obey it.

Constraining is a deliberate choice, not a convenience. D2 exists to produce
drafts that are *plausible but worse* -- the deployment scenario the paper
claims. If sampling were unconstrained, a small generator would sometimes emit
an unpaired pitch, the tolerant decoder would silently drop it, and the draft
would differ from its reference by a note the model never really "decided" to
change. That contaminates the edit script with decoding artifacts and tests the
model's grasp of syntax rather than of music.
"""
import torch

from src.model import MelodyLM
from src.tokenizer import MelodyTokenizer


class Grammar:
    """Masks over the vocabulary for what may legally follow each token class."""

    def __init__(self, tokenizer: MelodyTokenizer, device=None):
        size = len(tokenizer)
        device = device or "cpu"

        pitch_ids = [
            tokenizer.token_to_id[t]
            for t in tokenizer.vocab
            if t.startswith("P")
        ] + [tokenizer.rest_id]
        duration_ids = [
            tokenizer.token_to_id[t] for t in tokenizer.vocab if t.startswith("D")
        ]

        def mask(ids):
            m = torch.zeros(size, dtype=torch.bool, device=device)
            m[list(ids)] = True
            return m

        # After a pitch or rest, only a duration is legal.
        self.after_event = mask(duration_ids)
        # After a duration: another event, a barline, or the end.
        self.after_duration = mask(pitch_ids + [tokenizer.bar_id, tokenizer.eos_id])
        # After a barline, an event must follow.
        self.after_bar = mask(pitch_ids)

        self.pitch_ids = set(pitch_ids)
        self.duration_ids = set(duration_ids)
        self.bar_id = tokenizer.bar_id
        self.eos_id = tokenizer.eos_id
        self.bos_id = tokenizer.bos_id

    def next_mask(self, last_token: int) -> torch.Tensor:
        if last_token in self.pitch_ids:
            return self.after_event
        if last_token == self.bar_id or last_token == self.bos_id:
            return self.after_bar
        return self.after_duration


@torch.no_grad()
def sample_continuations(
    model: MelodyLM,
    tokenizer: MelodyTokenizer,
    prefixes: list[list[int]],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
    device=None,
    forbid_eos: bool = False,
) -> list[list[int]]:
    """Continue each prefix until EOS, batched.

    Sequences are right-padded and each one's own position is tracked, so the
    batch may contain prefixes of different lengths. Padding only ever sits after
    a sequence's current end, which the causal mask already excludes from
    attention.
    """
    device = device or next(model.parameters()).device
    model.eval()

    grammar = Grammar(tokenizer, device)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    batch = len(prefixes)
    lengths = torch.tensor([len(p) for p in prefixes], device=device)
    limit = min(model.config.max_len, int(lengths.max()) + max_new_tokens)

    tokens = torch.full((batch, limit), tokenizer.pad_id, dtype=torch.long, device=device)
    for i, prefix in enumerate(prefixes):
        tokens[i, : len(prefix)] = torch.tensor(prefix[:limit], device=device)

    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        active = (~finished) & (lengths < limit)
        if not bool(active.any()):
            break

        logits = model(tokens[:, : int(lengths.max())])
        step = logits[torch.arange(batch, device=device), lengths - 1]

        # Mask to the grammar, per sequence.
        last = tokens[torch.arange(batch, device=device), lengths - 1]
        allowed = torch.stack([grammar.next_mask(int(t)) for t in last.cpu()]).to(device)
        if forbid_eos:
            # Infilling needs a fixed number of events, so the model must not be
            # able to end the sequence early.
            allowed = allowed.clone()
            allowed[:, grammar.eos_id] = False
        step = step.masked_fill(~allowed, float("-inf"))

        if temperature <= 0:
            choice = step.argmax(dim=-1)
        else:
            step = step / temperature
            if top_k:
                kth = step.topk(min(top_k, step.size(-1)), dim=-1).values[:, -1:]
                step = step.masked_fill(step < kth, float("-inf"))
            probs = torch.softmax(step, dim=-1)
            choice = torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(-1).to(device)

        write = active & (~finished)
        positions = lengths.clamp(max=limit - 1)
        tokens[torch.arange(batch, device=device)[write], positions[write]] = choice[write]
        lengths = lengths + write.long()
        finished = finished | (choice == grammar.eos_id) & write

    return [
        _trim_incomplete(tokens[i, : int(lengths[i])].tolist(), grammar)
        for i in range(batch)
    ]


def _trim_incomplete(tokens: list[int], grammar: Grammar) -> list[int]:
    """Drop a trailing token that cannot end a sequence.

    Generation stops on EOS, but also on `max_new_tokens` and on the context
    limit, and those two can land mid-pair. A trailing pitch with no duration
    would be silently dropped by the decoder, so the draft would lose a note the
    generator never chose to remove -- a truncation artifact that would show up
    in the edit script as a deletion.
    """
    while tokens and (tokens[-1] in grammar.pitch_ids or tokens[-1] == grammar.bar_id):
        tokens = tokens[:-1]
    return tokens
