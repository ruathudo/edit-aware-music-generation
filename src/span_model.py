"""
Span-level detection of generated material.

The per-position edit head sits at ~1.5x chance on generator drafts and does not
improve even on drafts the judge certifies as degraded. The damage is not at any
particular note: the melody is measurably worse as a whole, and the *unchanged*
notes around a spliced span also become dearer under the judge (+0.0677, against
+0.2162 for the span itself). The generator writes a locally coherent passage;
what is wrong is how it sits in the piece.

So this scores *regions* rather than positions. For every candidate span it pools
the encoder states across the span and concatenates the states immediately either
side of it, then picks the span most likely to be generated. The boundary states
are in the feature deliberately -- a spliced passage's most visible defect is
that it does not rejoin its surroundings, and that evidence lives at the seam,
not inside the span.

Framed as a choice among candidates rather than independent binary decisions,
because the generating process inserts exactly one span. That also gives an
honest chance baseline: 1/|candidates|, usually a few hundredths of a percent.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SpanModelConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 1024

    def to_dict(self) -> dict:
        return asdict(self)


def enumerate_spans(
    n_events: int,
    lengths: tuple[int, ...] = (2, 3, 4, 5, 6),
    margin: int = 2,
) -> list[tuple[int, int]]:
    """Candidate (start, length) pairs in event space.

    `lengths` and `margin` mirror the generating process, which is an assumption a
    deployed system would not have -- a real user's draft comes with no span-length
    prior. It is stated rather than hidden, and widening the range only makes the
    task harder, never easier.
    """
    out = []
    for length in lengths:
        for start in range(margin, max(margin, n_events - length - margin) + 1):
            if start + length <= n_events:
                out.append((start, length))
    return out


class SpanDetector(nn.Module):
    """Bidirectional encoder plus a span-scoring head."""

    def __init__(self, config: SpanModelConfig, pad_id: int = 0):
        super().__init__()
        self.config = config
        self.pad_id = pad_id

        self.embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.d_ff, dropout=config.dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.d_model)

        # interior mean, plus the states just before and just after the span
        self.head = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        hidden = self.drop(self.embed(tokens) + self.pos_embed(pos))
        hidden = self.encoder(hidden, src_key_padding_mask=(tokens == self.pad_id))
        return self.norm(hidden)

    def forward(
        self,
        tokens: torch.Tensor,
        span_bounds: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score each candidate span.

        tokens      (B, T)      draft token ids
        span_bounds (B, C, 2)   token [start, end) per candidate
        span_mask   (B, C)      True where the candidate is real rather than padding

        Returns (B, C) logits, with padded candidates set to -inf so they cannot
        win an argmax or carry probability mass in the softmax.
        """
        memory = self.encode(tokens)
        b, t, d = memory.shape

        # Prefix sums give the interior mean of any span in constant time, which
        # matters when every melody contributes a few hundred candidates.
        zero = memory.new_zeros(b, 1, d)
        cumulative = torch.cat([zero, memory.cumsum(dim=1)], dim=1)  # (B, T+1, D)

        starts = span_bounds[..., 0].clamp(0, t)
        ends = span_bounds[..., 1].clamp(0, t)
        widths = (ends - starts).clamp(min=1).unsqueeze(-1).float()

        gather = lambda source, index: source.gather(
            1, index.unsqueeze(-1).expand(-1, -1, d)
        )
        interior = (gather(cumulative, ends) - gather(cumulative, starts)) / widths

        before = gather(memory, (starts - 1).clamp(min=0))
        after = gather(memory, ends.clamp(max=t - 1))

        logits = self.head(torch.cat([interior, before, after], dim=-1)).squeeze(-1)
        return logits.masked_fill(~span_mask, float("-inf"))


@torch.no_grad()
def surprisal_span_scores(
    per_token_surprisal: torch.Tensor,
    span_bounds: torch.Tensor,
    span_mask: torch.Tensor,
) -> torch.Tensor:
    """Zero-shot baseline: rank spans by the judge's mean surprisal.

    The judge needs no training for this and never saw an edit label, so it tests
    whether pooling the same signal over a region recovers what per-position
    surprisal could not. If the trained detector cannot beat it, the encoder is
    contributing nothing beyond what the judge already knew.
    """
    b, t = per_token_surprisal.shape
    zero = per_token_surprisal.new_zeros(b, 1)
    cumulative = torch.cat([zero, per_token_surprisal.cumsum(dim=1)], dim=1)

    starts = span_bounds[..., 0].clamp(0, t)
    ends = span_bounds[..., 1].clamp(0, t)
    widths = (ends - starts).clamp(min=1).float()

    totals = cumulative.gather(1, ends) - cumulative.gather(1, starts)
    return (totals / widths).masked_fill(~span_mask, float("-inf"))
