"""
A two-sided judge: how plausible is a note given everything around it?

`J` is autoregressive, so it scores a note only against what precedes it. That
is the right tool for whole-melody plausibility, but the wrong one for asking
whether *this note* is wrong: a note can be fine given its past and break what
follows, and a causal model charges that break to the *following* notes. The
D2 splice measurements show exactly this -- unchanged notes after an infilled
span score +0.0677 dearer.

`MaskedJudge` is trained the way that question is posed. Whole events are
masked -- pitch and duration together, so neither half leaks the other -- and
the model predicts them from both sides. A note's score is

    -log p(pitch, duration | every other token in the melody)

Same size as `J`, same training half of Essen, so the only difference between
the two judges is the direction of context.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from src.tokenizer import MelodyTokenizer


@dataclass(frozen=True)
class MaskedLMConfig:
    vocab_size: int          # tokenizer vocabulary; MASK is appended after it
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 1024

    @property
    def mask_id(self) -> int:
        return self.vocab_size


class MaskedMelodyLM(nn.Module):
    def __init__(self, config: MaskedLMConfig, pad_id: int = 0):
        super().__init__()
        self.config = config
        self.pad_id = pad_id
        self.embed = nn.Embedding(config.vocab_size + 1, config.d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads, dim_feedforward=config.d_ff,
            dropout=config.dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.d_model)
        # Predicts real tokens only; MASK is never a target.
        self.fc = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        h = self.drop(self.embed(tokens) + self.pos_embed(pos))
        h = self.encoder(h, src_key_padding_mask=(tokens == self.pad_id))
        return self.fc(self.norm(h))


def event_groups(tokens: list[int], spans: list[tuple[int, int]], eos_id: int) -> list[list[int]]:
    """Token positions masked together: each event's pitch+duration, then EOS alone.

    EOS is included because it is a labelled slot (trailing insertions land
    there), so every labelled position gets a score.
    """
    groups = [[p, d] for p, d in spans]
    if tokens and tokens[-1] == eos_id:
        groups.append([len(tokens) - 1])
    return groups


def mask_events(tokens, groups, rate, mask_id, rng):
    """Training input: mask a random subset of whole events (at least one)."""
    chosen = [g for g in groups if rng.random() < rate] or [rng.choice(groups)]
    masked = list(tokens)
    targets = [-100] * len(tokens)
    for group in chosen:
        for i in group:
            targets[i] = tokens[i]
            masked[i] = mask_id
    return masked, targets


@torch.no_grad()
def score_events(model, tokens, groups, batch_size=512, device=None) -> np.ndarray:
    """Per-token score: each group masked alone, scored by -log p of its tokens.

    Both positions of an event receive the event's joint score, matching how
    edit labels are broadcast to both tokens.
    """
    device = device or next(model.parameters()).device
    model.eval()
    out = np.zeros(len(tokens), dtype=np.float64)
    base = torch.tensor(tokens, dtype=torch.long)
    for start in range(0, len(groups), batch_size):
        chunk = groups[start:start + batch_size]
        batch = base.unsqueeze(0).repeat(len(chunk), 1)
        for row, group in enumerate(chunk):
            batch[row, group] = model.config.mask_id
        logp = model(batch.to(device)).log_softmax(dim=-1).cpu()
        for row, group in enumerate(chunk):
            nll = -sum(float(logp[row, i, tokens[i]]) for i in group)
            out[group] = nll
    return out


def train_masked_judge(
    train_seqs, val_seqs, tokenizer: MelodyTokenizer, config: MaskedLMConfig | None = None,
    epochs=60, batch_size=64, lr=3e-4, patience=6, rate=0.15, seed=4321, device=None,
):
    """`*_seqs` are (tokens, groups) pairs. Early stopping on validation masked NLL.

    Validation masks are drawn once with a fixed seed, so the stopping signal is
    not noise from re-sampling which events are hidden.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    config = config or MaskedLMConfig(vocab_size=len(tokenizer))
    model = MaskedMelodyLM(config, pad_id=tokenizer.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    val_rng = random.Random(seed + 1)
    val_fixed = [mask_events(t, g, rate, config.mask_id, val_rng) for t, g in val_seqs]

    def batches(pairs, shuffle):
        order = list(range(len(pairs)))
        if shuffle:
            rng.shuffle(order)
        for s in range(0, len(order), batch_size):
            chunk = [pairs[i] for i in order[s:s + batch_size]]
            x = pad_sequence([torch.tensor(m) for m, _ in chunk], batch_first=True,
                             padding_value=tokenizer.pad_id)
            y = pad_sequence([torch.tensor(t) for _, t in chunk], batch_first=True,
                             padding_value=-100)
            yield x.to(device), y.to(device)

    def loss_of(x, y):
        logits = model(x)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=-100)

    history = {"train": [], "val": []}
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        pairs = [mask_events(t, g, rate, config.mask_id, rng) for t, g in train_seqs]
        running, n = 0.0, 0
        for x, y in batches(pairs, shuffle=True):
            loss = loss_of(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss); n += 1
        model.eval()
        with torch.no_grad():
            vals = [float(loss_of(x, y)) for x, y in batches(val_fixed, shuffle=False)]
        val = float(np.mean(vals))
        history["train"].append(running / max(n, 1))
        history["val"].append(val)
        print(f"  epoch {epoch:3d}  train {history['train'][-1]:.4f}  val {val:.4f}", flush=True)
        if best - val > 1e-4:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch}; best val {best:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), history


def save_masked_judge(model: MaskedMelodyLM, path, extra=None):
    torch.save({"state_dict": model.state_dict(), "config": asdict(model.config),
                "pad_id": model.pad_id, **(extra or {})}, path)


def load_masked_judge(path, device=None) -> MaskedMelodyLM:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    model = MaskedMelodyLM(MaskedLMConfig(**blob["config"]), pad_id=blob["pad_id"])
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()
