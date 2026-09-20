"""
The refinement task: given a draft `M_0`, produce the reference `M_star`.

Holds the three pieces that connect the aligner, the tokenizer and the model:
per-token edit labels, a cached draft corpus, and the batching.

Draft generation is cached because D2 requires sampling from the generator for
the whole corpus (spec 10). The cache key is a hash of the generating config, so
changing a sampling parameter produces a new cache rather than silently reusing
stale drafts.
"""
import hashlib
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.align import KEEP, edit_labels
from src.corpus import Melody
from src.drafts import Draft
from src.edit_model import IGNORE_INDEX
from src.tokenizer import MelodyTokenizer

Notes = list[tuple[int, int]]


def token_edit_labels(
    source_notes: Notes,
    target_notes: Notes,
    tokenizer: MelodyTokenizer,
    bar_starts=None,
) -> tuple[list[int], list[int]]:
    """Encode the draft and label every token with its edit class.

    Returns (source_tokens, labels) of equal length.

    The aligner produces one label per *event*; the encoder sees *tokens*. Each
    event's label is broadcast to both of its tokens -- the class already names
    the attribute at fault (SUB_PITCH against SUB_DUR), so both positions carry
    the same target and the head gets supervision at each.

    Structural tokens (BOS, BAR) are set to IGNORE_INDEX rather than KEEP. They
    are not editable positions, and labelling them KEEP would inflate an already
    dominant majority class and make the head's accuracy look better than it is.
    The EOS position is a real slot: it carries the trailing-insertion label.
    """
    tokens, spans = tokenizer.encode_with_spans(source_notes, bar_starts)
    labels = [IGNORE_INDEX] * len(tokens)

    event_labels = edit_labels(source_notes, target_notes)

    for index, (pitch_at, duration_at) in enumerate(spans):
        labels[pitch_at] = event_labels[index]
        labels[duration_at] = event_labels[index]

    # Last element of event_labels is the EOS slot, where trailing insertions land.
    if tokens and tokens[-1] == tokenizer.eos_id:
        labels[-1] = event_labels[len(source_notes)]

    return tokens, labels


@dataclass
class RefinementExample:
    source: list[int]
    source_labels: list[int]
    target: list[int]
    draft_notes: Notes
    reference: Melody
    draft_source: str
    surprisal: list[float] | None = None


class RefinementDataset(Dataset):
    """Encoded (draft, reference) pairs with token-level edit labels."""

    def __init__(
        self,
        drafts: list[Draft],
        tokenizer: MelodyTokenizer,
        max_len: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.examples: list[RefinementExample] = []
        self.dropped = 0

        for draft in drafts:
            reference = draft.reference
            if not (tokenizer.supports(draft.notes) and tokenizer.supports(reference.notes)):
                self.dropped += 1
                continue

            source, labels = token_edit_labels(
                draft.notes, reference.notes, tokenizer, reference.bar_starts
            )
            target = tokenizer.encode(reference.notes, reference.bar_starts)

            if len(source) > max_len or len(target) > max_len:
                self.dropped += 1
                continue

            self.examples.append(
                RefinementExample(source, labels, target, draft.notes, reference, draft.source)
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RefinementExample:
        return self.examples[index]

    def label_tensor(self) -> torch.Tensor:
        """All labels flattened, for computing class weights."""
        return torch.tensor(
            [label for example in self.examples for label in example.source_labels],
            dtype=torch.long,
        )


def make_collate(pad_id: int):
    """Batch into (source, source_labels, target_in, target_out, metadata).

    The decoder is teacher-forced, so the target is split into the shifted input
    and the expected output.
    """

    def collate(batch: list[RefinementExample]):
        source = pad_sequence(
            [torch.tensor(e.source, dtype=torch.long) for e in batch],
            batch_first=True, padding_value=pad_id,
        )
        labels = pad_sequence(
            [torch.tensor(e.source_labels, dtype=torch.long) for e in batch],
            batch_first=True, padding_value=IGNORE_INDEX,
        )
        target = pad_sequence(
            [torch.tensor(e.target, dtype=torch.long) for e in batch],
            batch_first=True, padding_value=pad_id,
        )
        return source, labels, target[:, :-1], target[:, 1:], batch

    return collate


# ---------------------------------------------------------------- draft cache

def _config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def cached_drafts(
    melodies: list[Melody],
    build,
    config: dict,
    cache_dir: str | Path = "data/drafts",
    force: bool = False,
) -> list[Draft]:
    """Generate drafts once and reuse them.

    `build` is a zero-argument callable producing the drafts. The cache key
    includes the generating config *and* the reference ids, so a changed
    sampling temperature or a changed split both invalidate it -- silently
    reusing drafts built under different settings is the failure this prevents.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _config_hash({**config, "ids": [m.id for m in melodies]})
    path = cache_dir / f"drafts_{key}.pkl"

    if path.exists() and not force:
        with path.open("rb") as handle:
            drafts = pickle.load(handle)
        print(f"  drafts: reused {len(drafts)} from {path.name}")
        return drafts

    drafts = build()
    with path.open("wb") as handle:
        pickle.dump(drafts, handle)
    print(f"  drafts: built {len(drafts)} -> {path.name}")
    return drafts


def build_draft_corpus(
    melodies: list[Melody],
    tokenizer: MelodyTokenizer,
    priors,
    sources: tuple[str, ...] = ("D2-infill",),
    generator=None,
    temperature: float = 1.0,
    per_melody: int = 1,
    seed: int = 0,
    max_edits: int = 5,
) -> list[Draft]:
    """Assemble drafts from one or more sources.

    D2-infill is the default: it is the only source that is both local enough
    for the edit framing and on-manifold enough to be a plausible draft.
    """
    from src.drafts import d2_infill, generate

    out: list[Draft] = []
    for source in sources:
        if source in ("D1", "D3"):
            out.extend(
                generate(
                    melodies, source, priors,
                    seed=seed, max_edits=max_edits, per_melody=per_melody,
                )
            )
        elif source == "D2-infill":
            if generator is None:
                raise ValueError("D2-infill requires a generator")
            for repeat in range(per_melody):
                out.extend(
                    d2_infill(
                        melodies, generator, tokenizer,
                        random.Random(seed + repeat),
                        temperature=temperature,
                        seed=seed + repeat,
                    )
                )
        else:
            raise ValueError(f"unknown draft source {source!r}")
    return out


# ---------------------------------------------------------------- judge surprisal

@torch.no_grad()
def attach_surprisal(dataset, judge, batch_size: int = 64, device=None):
    """Store the judge's per-token surprisal on every example.

    The judge already knows which notes are implausible -- that is the whole
    reason it exists -- and a position the draft makes surprising is exactly a
    candidate edit site. Nothing about this leaks: `J` is trained on a disjoint
    corpus half and has never seen an edit label.

    Surprisal is standardized *within each melody*, so the feature says "how
    surprising is this note relative to the rest of this melody" rather than
    "how unusual is this melody overall". Localization is a within-sequence
    question, and the absolute level mostly reflects the melody's own difficulty.

    It also predicts a sharp split. On D1 the corruption is off-manifold and
    should light up; on D2-infill the span is plausible everywhere and should
    not. That makes this a fix and a test of the manifold result at once.
    """
    import torch.nn.functional as F

    device = device or next(judge.model.parameters()).device
    pad_id = judge.tokenizer.pad_id

    for start in range(0, len(dataset), batch_size):
        chunk = dataset.examples[start : start + batch_size]
        tokens = pad_sequence(
            [torch.tensor(e.source, dtype=torch.long) for e in chunk],
            batch_first=True, padding_value=pad_id,
        ).to(device)

        logits = judge.model(tokens[:, :-1])
        targets = tokens[:, 1:]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=pad_id, reduction="none",
        ).reshape(targets.shape)

        # BOS has no prediction of its own; pad the front so indices line up.
        full = torch.cat([torch.zeros(len(chunk), 1, device=device), ce], dim=1)
        valid = (tokens != pad_id).float()

        count = valid.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (full * valid).sum(dim=1, keepdim=True) / count
        var = (((full - mean) ** 2) * valid).sum(dim=1, keepdim=True) / count
        standardized = ((full - mean) / var.sqrt().clamp(min=1e-6)) * valid

        for example, row in zip(chunk, standardized.cpu()):
            example.surprisal = row[: len(example.source)].tolist()

    return dataset


def surprisal_tensor(batch, width: int, device) -> torch.Tensor | None:
    """Padded (B, width) surprisal for a collated batch, or None if unattached."""
    if not batch or batch[0].surprisal is None:
        return None
    out = torch.zeros(len(batch), width)
    for i, example in enumerate(batch):
        values = example.surprisal[:width]
        out[i, : len(values)] = torch.tensor(values, dtype=torch.float)
    return out.to(device)
