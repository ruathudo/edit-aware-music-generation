"""
The edit-aware refinement model (spec 5).

Encoder-decoder, not the legacy decoder-only stack. Two reasons:

1. Spec 5.1 requires the edit head to sit on *input* positions -- the positions
   of the draft `M_0` -- rather than on decoder outputs.
2. Deciding whether note `i` is an error requires seeing what follows it. A
   causal model reading the draft cannot do that: at position `i` it has only
   the prefix. Bidirectional encoding of the draft is what makes typed edit
   classification a well-posed problem rather than a guess from left context.

The legacy pipeline also aligned `corrupted[:, :-1]` against `clean[:, 1:]`
position by position, which silently assumes the draft and the reference have
the same length. Insertions and deletions break that assumption, and both D1 and
D3 produce them. Encoder-decoder removes the assumption entirely.

The generative head predicts the corrected melody; the edit head predicts, for
each input position, which of five operations applies:

    KEEP, SUB_PITCH, SUB_DUR, INSERT_BEFORE, DELETE

which is what makes "where *and how*" accurate rather than aspirational -- the
published version predicted a binary mask and claimed typed correction.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.align import EDIT_CLASSES

IGNORE_INDEX = -100


@dataclass(frozen=True)
class EditModelConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 1024
    n_edit_classes: int = len(EDIT_CLASSES)
    use_edit_head: bool = True
    use_edit_hint: bool = False  # oracle input; diagnostics only
    use_surprisal: bool = False  # judge per-token surprisal as an input feature

    def to_dict(self) -> dict:
        return asdict(self)


def causal_mask(size: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)


class EditAwareTransformer(nn.Module):
    """Encoder-decoder over melody tokens, with an optional typed edit head.

    `use_edit_head=False` gives baseline B1. Keeping the head while setting
    alpha to zero gives B2, the parameter-matched control: identical
    architecture and identical parameter count, differing only in whether the
    auxiliary loss contributes gradient. That is a tighter control than
    rescaling a baseline's width to match, because nothing but the loss changes.
    """

    def __init__(self, config: EditModelConfig, pad_id: int = 0):
        super().__init__()
        self.config = config
        self.pad_id = pad_id

        # Source and target share a vocabulary, so they share an embedding.
        self.embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.n_encoder_layers, enable_nested_tensor=False
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.n_decoder_layers)

        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.fc = nn.Linear(config.d_model, config.vocab_size)

        self.edit_head = (
            nn.Linear(config.d_model, config.n_edit_classes) if config.use_edit_head else None
        )

        # Oracle channel, used only by the diagnostics: it injects ground-truth
        # edit labels at the input so the decoder can be asked what it would do
        # given perfect localization. Never enabled for a reported model -- it
        # leaks the answer by construction.
        self.edit_hint = (
            nn.Embedding(config.n_edit_classes + 1, config.d_model)
            if config.use_edit_hint else None
        )

        # The judge's per-token surprisal, projected into the model's space.
        # Unlike the oracle hint this leaks nothing: the judge is trained on a
        # disjoint corpus half and has never seen an edit label. It is a genuine
        # inference-time feature, not a diagnostic.
        self.surprisal_proj = (
            nn.Linear(1, config.d_model) if config.use_surprisal else None
        )

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        if t > self.config.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.config.max_len}")
        pos = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        return self.drop(self.embed(tokens) + self.pos_embed(pos))

    def encode(
        self,
        source: torch.Tensor,
        edit_hint: torch.Tensor | None = None,
        surprisal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Bidirectional encoding of the draft. No causal mask here by design."""
        hidden = self._embed(source)
        if self.surprisal_proj is not None and surprisal is not None:
            hidden = hidden + self.surprisal_proj(surprisal.unsqueeze(-1))
        if self.edit_hint is not None and edit_hint is not None:
            # IGNORE_INDEX maps to the final row rather than indexing negatively.
            safe = edit_hint.clone()
            safe[safe == IGNORE_INDEX] = self.config.n_edit_classes
            hidden = hidden + self.edit_hint(safe)

        memory = self.encoder(hidden, src_key_padding_mask=(source == self.pad_id))
        return self.encoder_norm(memory)

    def decode(self, target_in: torch.Tensor, memory: torch.Tensor, source: torch.Tensor):
        hidden = self.decoder(
            self._embed(target_in),
            memory,
            tgt_mask=causal_mask(target_in.size(1), target_in.device),
            tgt_key_padding_mask=(target_in == self.pad_id),
            memory_key_padding_mask=(source == self.pad_id),
        )
        return self.fc(self.decoder_norm(hidden))

    def forward(self, source: torch.Tensor, target_in: torch.Tensor, edit_hint=None, surprisal=None):
        """Returns (token_logits, edit_logits); edit_logits is None without the head."""
        memory = self.encode(source, edit_hint, surprisal)
        token_logits = self.decode(target_in, memory, source)
        edit_logits = self.edit_head(memory) if self.edit_head is not None else None
        return token_logits, edit_logits

    @torch.no_grad()
    def predict_edits(self, source: torch.Tensor, surprisal=None) -> torch.Tensor:
        if self.edit_head is None:
            raise RuntimeError("model has no edit head")
        return self.edit_head(self.encode(source, surprisal=surprisal)).argmax(dim=-1)


def edit_class_weights(
    labels: torch.Tensor,
    n_classes: int = len(EDIT_CLASSES),
    ignore_index: int = IGNORE_INDEX,
    scheme: str = "inverse",
    max_ratio: float = 10.0,
) -> torch.Tensor:
    """Class weights for the edit head, normalized to mean 1.

    KEEP is ~94% of positions, so an unweighted objective can reach a good loss
    by predicting it everywhere. But the obvious correction over-shoots badly:
    plain inverse frequency weights INSERT_BEFORE about **105x** KEEP on this
    data, and the head duly learns to flag nearly everything -- measured at
    recall 0.56 against precision 0.010, with KEEP recall collapsing to 0.29
    despite precision 0.97. The imbalance fix became a worse imbalance.

    Schemes:
      ``none``     uniform; lets KEEP dominate.
      ``sqrt``     inverse square-root frequency; a much gentler correction.
      ``inverse``  plain inverse frequency; the overshooting default.
      ``capped``   inverse frequency with the max:min ratio clamped to
                   `max_ratio`, keeping the ordering but bounding the extreme.
    """
    valid = labels[labels != ignore_index]
    counts = torch.bincount(valid.flatten(), minlength=n_classes).float().clamp(min=1.0)

    if scheme == "none":
        weights = torch.ones(n_classes)
    elif scheme == "sqrt":
        weights = counts.sum().sqrt() / counts.sqrt()
    elif scheme == "inverse":
        weights = counts.sum() / counts
    elif scheme == "capped":
        weights = counts.sum() / counts
        weights = weights.clamp(max=weights.min() * max_ratio)
    else:
        raise ValueError(f"unknown weighting scheme {scheme!r}")

    return weights / weights.mean()


def focal_edit_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Focal loss over edit classes (spec 5.1's alternative to re-weighting).

    Down-weights positions the model already gets right, so the easy KEEP
    majority stops dominating the gradient without inflating rare classes by a
    hundredfold the way inverse frequency does.
    """
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )
    valid = (targets.reshape(-1) != ignore_index).float()
    focal = ((1.0 - torch.exp(-ce)) ** gamma) * ce
    return (focal * valid).sum() / valid.sum().clamp(min=1.0)


def edit_aware_loss(
    token_logits: torch.Tensor,
    target_out: torch.Tensor,
    edit_logits: torch.Tensor | None,
    edit_labels: torch.Tensor | None,
    pad_id: int,
    alpha: float = 0.0,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, float]:
    """L = L_gen + alpha * L_edit  (spec 5.2).

    L_gen is cross-entropy over the musical token vocabulary; L_edit is
    class-weighted cross-entropy over the five edit classes at input positions.
    alpha = 0 reduces this to the imitation-only baseline exactly, so the
    baseline and the edit-aware model share one code path.
    """
    generative = F.cross_entropy(
        token_logits.reshape(-1, token_logits.size(-1)),
        target_out.reshape(-1),
        ignore_index=pad_id,
    )

    if edit_logits is None or edit_labels is None or alpha == 0.0:
        return generative, generative.detach().item(), 0.0

    edit = F.cross_entropy(
        edit_logits.reshape(-1, edit_logits.size(-1)),
        edit_labels.reshape(-1),
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
    )
    return generative + alpha * edit, generative.detach().item(), edit.detach().item()
