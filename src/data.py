"""
Torch plumbing between `Melody` objects and the language model.

Deliberately thin: tokenization lives in `src.tokenizer`, corpus semantics in
`src.corpus`. This module only batches and pads.
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.corpus import (
    Melody,
    duration_vocabulary,
    load,
    pitch_range,
    split_by_tunefamily,
)
from src.tokenizer import MelodyTokenizer, TokenizerConfig


def standard_splits(seed: int = 42) -> dict[str, list[Melody]]:
    """The corpus partition every experiment shares.

    Essen is halved between the judge and the generator because spec 4 requires
    `J` to be trained on data disjoint from `G` -- otherwise the degradation
    filter is circular, admitting exactly the drafts `G` was taught to produce.

    MTC-ANN is held out whole. It is the validator's reference set and the source
    of D4 variant pairs, so `J` must never see it: a judge that had trained on
    these melodies would score their human-authored variants as familiar rather
    than as plausible, which is precisely the confound the calibration control
    exists to rule out.

    Splits are by tune family, and deterministic given `seed`, so `G` can
    reconstruct exactly this partition without a file on disk.
    """
    essen = load("ESSEN")
    halves = split_by_tunefamily(essen, {"judge": 0.5, "generator": 0.5}, seed=seed)
    judge = split_by_tunefamily(halves["judge"], {"train": 0.9, "val": 0.1}, seed=seed + 1)
    generator = split_by_tunefamily(halves["generator"], {"train": 0.9, "val": 0.1}, seed=seed + 2)

    return {
        "judge_train": judge["train"],
        "judge_val": judge["val"],
        "generator_train": generator["train"],
        "generator_val": generator["val"],
        "heldout": load("MTC-ANN"),
    }


def build_tokenizer(melodies: list[Melody], pitch_pad: int = 12, **overrides) -> MelodyTokenizer:
    """Fit a tokenizer to a corpus.

    The pitch range and duration set are derived from the data rather than
    hardcoded, which is the fix for the legacy allowlist that silently rejected
    whole melodies for one unlisted duration. Build this on the *union* of every
    split so held-out melodies cannot contain an unencodable value.

    `pitch_pad` widens the range beyond what the corpus contains, by default an
    octave either side. This is not slack for its own sake: D3's out-of-range
    rule exists precisely to push notes outside the melody's established
    tessitura, so a vocabulary clipped to the corpus could not represent the
    error it is supposed to inject. The cost is a couple of dozen embedding rows.
    """
    lo, hi = pitch_range(melodies)
    config = TokenizerConfig(
        pitch_min=overrides.pop("pitch_min", max(0, lo - pitch_pad)),
        pitch_max=overrides.pop("pitch_max", min(127, hi + pitch_pad)),
        durations=overrides.pop("durations", duration_vocabulary(melodies)),
        **overrides,
    )
    return MelodyTokenizer(config)


class MelodyTokenDataset(Dataset):
    """Melodies encoded once, up front.

    Encoding is deterministic and the corpora are small, so caching it avoids
    re-tokenizing every epoch. Sequences longer than `max_len` are dropped rather
    than truncated -- a truncated melody is a different melody, and would
    contaminate the judge's notion of a plausible ending.
    """

    def __init__(self, melodies: list[Melody], tokenizer: MelodyTokenizer, max_len: int = 1024):
        self.tokenizer = tokenizer
        self.melodies: list[Melody] = []
        self.encoded: list[list[int]] = []
        self.dropped = 0

        for melody in melodies:
            if not tokenizer.supports(melody.notes):
                self.dropped += 1
                continue
            tokens = tokenizer.encode(melody.notes, melody.bar_starts)
            if len(tokens) > max_len:
                self.dropped += 1
                continue
            self.melodies.append(melody)
            self.encoded.append(tokens)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.encoded[index], dtype=torch.long)


def make_collate(pad_id: int):
    def collate(batch: list[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(batch, batch_first=True, padding_value=pad_id)

    return collate


def make_loader(
    dataset: MelodyTokenDataset,
    pad_id: int,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate(pad_id),
        num_workers=num_workers,
    )
