"""
Decoder-only melody language model.

One architecture serves three roles, differing only in configuration and
training data: the judge `J` (spec 4), the generator `G` that produces D2
drafts, and the backbone of the edit-aware model. Keeping them one class is what
makes `B2` -- the parameter-matched imitation baseline the spec asks for -- a
config change rather than a second implementation.

Implemented with `nn.TransformerEncoder` under a causal mask, which is the
standard decoder-only formulation. The legacy pipeline used
`nn.TransformerDecoder(x, x, tgt_mask=...)`, which additionally runs
cross-attention of `x` over itself: the same function, computed with an extra
attention block per layer.
"""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LMConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 1024

    def to_dict(self) -> dict:
        return asdict(self)


def causal_mask(size: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)


class MelodyLM(nn.Module):
    def __init__(self, config: LMConfig, pad_id: int = 0):
        super().__init__()
        self.config = config
        self.pad_id = pad_id

        self.embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        # Nested tensors are unused under norm_first; disabling silences a warning.
        self.encoder = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.d_model)
        self.fc = nn.Linear(config.d_model, config.vocab_size)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids -> (B, T, V) next-token logits."""
        b, t = tokens.shape
        if t > self.config.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.config.max_len}")

        pos = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        h = self.drop(self.embed(tokens) + self.pos_embed(pos))

        h = self.encoder(
            h,
            mask=causal_mask(t, tokens.device),
            src_key_padding_mask=(tokens == self.pad_id),
        )
        return self.fc(self.norm(h))

    def hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        """Final hidden states, for the edit head to attach to (spec 5.1)."""
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        h = self.drop(self.embed(tokens) + self.pos_embed(pos))
        h = self.encoder(
            h,
            mask=causal_mask(t, tokens.device),
            src_key_padding_mask=(tokens == self.pad_id),
        )
        return self.norm(h)


def token_nll(
    model: MelodyLM,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence negative log-likelihood.

    Returns (total_nll, n_predicted_tokens), both shape (B,). Padding is excluded
    from both, so `total / count` is a length-normalized score.

    Length normalization is not cosmetic here. Insertions and deletions change a
    draft's length, so an unnormalized total would partly measure how long the
    draft is rather than how implausible it is -- and D1/D3 both insert and
    delete notes. `Judge.score` reports both and uses the per-token value.
    """
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]

    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=model.pad_id,
        reduction="none",
    ).reshape(targets.shape)

    valid = (targets != model.pad_id).float()
    return (ce * valid).sum(dim=1), valid.sum(dim=1)
