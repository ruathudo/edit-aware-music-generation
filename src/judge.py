"""
The automatic degradation validator (spec 4).

This is the component that replaces the user study the reviewers asked for, and
it is deliberately built before any edit-aware model: it is the only part of the
plan that can *falsify* the premise. If random perturbation turns out to degrade
melodies after all, the rebuild's motivation weakens and that should surface on
day three rather than day thirty.

The judge `J` is a melody language model trained on data disjoint from anything
it scores. Per draft it reports:

  - ``delta_nll``  -- the primary signal, per-token so that insertions and
    deletions do not smuggle sequence length into the score.
  - scale conformance, pitch-class distance, interval distance, maximum
    interval, and metric alignment -- the musical criteria, all computed from
    the notes themselves so they work on drafts, which carry no annotations.

A caution the spec does not state, and which matters for whether any of this is
believable: ``ΔNLL`` measures *deviation from the corpus distribution*, which is
not the same thing as musical badness. An unusual-but-good melody is penalized
like a bad one. Two calibration controls are therefore required before the filter
gates any training data, and `calibration_report` produces both:

  1. the spread of ``NLL_J`` over held-out *real* melodies, as a noise floor; and
  2. human-authored variant pairs (MTC tune families), which should score
     ``ΔNLL ≈ 0``. If they do not, `J` is measuring novelty rather than
     degradation and every retention number downstream is suspect.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.align import REST
from src.corpus import Melody, bar_ticks, beat_ticks, infer_bar_starts, pickup_offset
from src.data import MelodyTokenDataset, build_tokenizer, make_collate
from src.model import LMConfig, MelodyLM, token_nll
from src.tokenizer import MelodyTokenizer, TokenizerConfig

Notes = list[tuple[int, int]]

# ---------------------------------------------------------------- musical metrics

_BASE_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Scale degrees as semitone offsets from the tonic. The minor set includes the
# raised seventh: folk melodies in minor routinely use it as a leading tone, and
# excluding it would score ordinary cadences as out-of-scale.
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10, 11)


def tonic_pitch_class(tonic: str | None) -> int | None:
    """Parse a tonic name to a pitch class. '-' is flat, '#' is sharp."""
    if not tonic:
        return None
    pitch_class = _BASE_PITCH_CLASS.get(tonic[0].upper())
    if pitch_class is None:
        return None
    for char in tonic[1:]:
        if char == "-":
            pitch_class -= 1
        elif char == "#":
            pitch_class += 1
    return pitch_class % 12


def scale_of(tonic: str | None, mode: str | None) -> frozenset[int] | None:
    root = tonic_pitch_class(tonic)
    if root is None:
        return None
    degrees = MINOR_SCALE if (mode or "").lower().startswith("min") else MAJOR_SCALE
    return frozenset((root + d) % 12 for d in degrees)


def sounding(notes: Notes) -> Notes:
    return [(p, d) for p, d in notes if p != REST]


def scale_conformance(notes: Notes, tonic: str | None, mode: str | None) -> float | None:
    """Fraction of sounding notes whose pitch class is in the key."""
    scale = scale_of(tonic, mode)
    pitches = [p for p, _ in sounding(notes)]
    if scale is None or not pitches:
        return None
    return sum(p % 12 in scale for p in pitches) / len(pitches)


def pitch_class_histogram(notes: Notes) -> np.ndarray:
    """Duration-weighted, normalized 12-bin pitch-class histogram."""
    hist = np.zeros(12)
    for pitch, duration in sounding(notes):
        hist[pitch % 12] += duration
    total = hist.sum()
    return hist / total if total else hist


def interval_histogram(notes: Notes, span: int = 13) -> np.ndarray:
    """Normalized histogram of melodic intervals, clipped to +/- span semitones.

    Rests are skipped rather than breaking the melodic line, so the interval is
    measured between successive sounding notes.
    """
    pitches = [p for p, _ in sounding(notes)]
    hist = np.zeros(2 * span + 1)
    for a, b in zip(pitches, pitches[1:]):
        hist[int(np.clip(b - a, -span, span)) + span] += 1
    total = hist.sum()
    return hist / total if total else hist


def max_interval(notes: Notes) -> int:
    pitches = [p for p, _ in sounding(notes)]
    return max((abs(b - a) for a, b in zip(pitches, pitches[1:])), default=0)


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())


def metric_alignment(
    notes: Notes,
    timesignature: str | None,
    start_offset: int = 0,
) -> float | None:
    """Duration-weighted mean metric strength of note onsets.

    Downbeat scores 1.0, another beat 0.5, a half-beat subdivision 0.25, anything
    else 0. Weighting by duration is what makes this a *structural* measure: it
    asks whether the long notes land on strong positions, which is the property
    D3's metric-misalignment rule breaks.
    """
    width = bar_ticks(timesignature)
    per_beat = beat_ticks(timesignature)
    if not width or not per_beat:
        return None

    position = start_offset
    score = weight = 0.0

    for pitch, duration in notes:
        if pitch != REST:
            within = position % width
            if within == 0:
                strength = 1.0
            elif within % per_beat == 0:
                strength = 0.5
            elif (within * 2) % per_beat == 0:
                strength = 0.25
            else:
                strength = 0.0
            score += strength * duration
            weight += duration
        position += duration

    return score / weight if weight else None


# ---------------------------------------------------------------- the judge

@dataclass
class DraftScore:
    delta_nll: float
    nll_draft: float
    nll_reference: float
    delta_scale_conformance: float | None
    delta_metric_alignment: float | None
    pitch_class_distance: float
    interval_distance: float
    max_interval_draft: int
    max_interval_reference: int
    n_notes_draft: int
    n_notes_reference: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class Judge:
    """A trained melody LM plus the musical criteria, used to score drafts."""

    def __init__(self, model: MelodyLM, tokenizer: MelodyTokenizer, device=None):
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

    # -------- persistence --------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "lm_config": self.model.config.to_dict(),
                "tokenizer_config": {
                    "pitch_min": self.tokenizer.config.pitch_min,
                    "pitch_max": self.tokenizer.config.pitch_max,
                    "durations": list(self.tokenizer.config.durations),
                    "ticks_per_bar": self.tokenizer.config.ticks_per_bar,
                    "use_bar": self.tokenizer.config.use_bar,
                },
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device=None) -> "Judge":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        blob = torch.load(path, map_location=device, weights_only=False)

        tconf = dict(blob["tokenizer_config"])
        tconf["durations"] = tuple(tconf["durations"])
        tokenizer = MelodyTokenizer(TokenizerConfig(**tconf))

        model = MelodyLM(LMConfig(**blob["lm_config"]), pad_id=tokenizer.pad_id)
        model.load_state_dict(blob["state_dict"])
        return cls(model, tokenizer, device)

    # -------- scoring --------

    @torch.no_grad()
    def nll(self, sequences: list[tuple[Notes, frozenset]], batch_size: int = 64):
        """Per-token and total NLL for (notes, bar_starts) pairs."""
        collate = make_collate(self.tokenizer.pad_id)
        totals, counts = [], []

        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            encoded = [
                torch.tensor(self.tokenizer.encode(notes, bars), dtype=torch.long)
                for notes, bars in chunk
            ]
            batch = collate(encoded).to(self.device)
            total, count = token_nll(self.model, batch)
            totals.append(total.cpu().numpy())
            counts.append(count.cpu().numpy())

        total = np.concatenate(totals) if totals else np.zeros(0)
        count = np.concatenate(counts) if counts else np.zeros(0)
        return total / np.maximum(count, 1), total

    def nll_of_melodies(self, melodies: list[Melody], batch_size: int = 64):
        """NLL of real melodies, using their annotated barlines."""
        return self.nll([(m.notes, m.bar_starts) for m in melodies], batch_size)

    def score_drafts(
        self,
        pairs: list[tuple[Notes, Melody]],
        batch_size: int = 64,
    ) -> list[DraftScore]:
        """Score (draft_notes, reference_melody) pairs.

        Both sides get barlines from `infer_bar_starts` under the reference's
        metre and pickup, so neither benefits from annotation the other lacks.
        """
        encoded: list[tuple[Notes, frozenset]] = []
        for draft, reference in pairs:
            offset = pickup_offset(reference)
            ts = reference.timesignature
            encoded.append((draft, infer_bar_starts(draft, ts, offset)))
            encoded.append((reference.notes, infer_bar_starts(reference.notes, ts, offset)))

        per_token, _ = self.nll(encoded, batch_size)

        out = []
        for i, (draft, reference) in enumerate(pairs):
            nll_draft, nll_ref = float(per_token[2 * i]), float(per_token[2 * i + 1])
            ref_notes = reference.notes
            offset = pickup_offset(reference)

            sc_d = scale_conformance(draft, reference.tonic, reference.mode)
            sc_r = scale_conformance(ref_notes, reference.tonic, reference.mode)
            ma_d = metric_alignment(draft, reference.timesignature, offset)
            ma_r = metric_alignment(ref_notes, reference.timesignature, offset)

            out.append(
                DraftScore(
                    delta_nll=nll_draft - nll_ref,
                    nll_draft=nll_draft,
                    nll_reference=nll_ref,
                    delta_scale_conformance=(None if sc_d is None or sc_r is None else sc_d - sc_r),
                    delta_metric_alignment=(None if ma_d is None or ma_r is None else ma_d - ma_r),
                    pitch_class_distance=l1_distance(
                        pitch_class_histogram(draft), pitch_class_histogram(ref_notes)
                    ),
                    interval_distance=l1_distance(
                        interval_histogram(draft), interval_histogram(ref_notes)
                    ),
                    max_interval_draft=max_interval(draft),
                    max_interval_reference=max_interval(ref_notes),
                    n_notes_draft=len(sounding(draft)),
                    n_notes_reference=len(sounding(ref_notes)),
                )
            )
        return out

    def is_degraded(
        self,
        score: DraftScore,
        tau: float,
        require_musical_criterion: bool = True,
    ) -> bool:
        """The admission filter (spec 4): clear the NLL threshold, and degrade on
        at least one musical criterion."""
        if score.delta_nll <= tau:
            return False
        if not require_musical_criterion:
            return True
        return any(
            [
                (score.delta_scale_conformance or 0) < 0,
                (score.delta_metric_alignment or 0) < 0,
                score.pitch_class_distance > 0,
                score.interval_distance > 0,
                score.max_interval_draft > score.max_interval_reference,
            ]
        )


# ---------------------------------------------------------------- training

def train_judge(
    train_melodies: list[Melody],
    val_melodies: list[Melody],
    tokenizer: MelodyTokenizer,
    config: LMConfig | None = None,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 3e-4,
    patience: int = 6,
    seed: int = 1234,
    device=None,
    log_every: int = 1,
) -> tuple[Judge, dict]:
    """Train `J`. Seeded separately from the generator, as spec 4 requires."""
    from torch.utils.data import DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = config or LMConfig(vocab_size=len(tokenizer))
    model = MelodyLM(config, pad_id=tokenizer.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    train_ds = MelodyTokenDataset(train_melodies, tokenizer, config.max_len)
    val_ds = MelodyTokenDataset(val_melodies, tokenizer, config.max_len)
    collate = make_collate(tokenizer.pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    print(
        f"judge: {model.n_parameters/1e6:.2f}M params | "
        f"train {len(train_ds)} (dropped {train_ds.dropped}) | val {len(val_ds)}"
    )

    history = {"train": [], "val": []}
    best = float("inf")
    best_state = None
    stale = 0

    for epoch in range(epochs):
        model.train()
        running = count = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            total, n = token_nll(model, batch)
            loss = total.sum() / n.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += float(total.sum())
            count += float(n.sum())
        train_nll = running / max(count, 1)

        model.eval()
        running = count = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                total, n = token_nll(model, batch)
                running += float(total.sum())
                count += float(n.sum())
        val_nll = running / max(count, 1)

        history["train"].append(train_nll)
        history["val"].append(val_nll)
        if log_every and epoch % log_every == 0:
            print(f"  epoch {epoch+1:3d}  train {train_nll:.4f}  val {val_nll:.4f}")

        if best - val_nll > 1e-4:
            best, stale = val_nll, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch+1}; best val {best:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_nll"] = best
    return Judge(model, tokenizer, device), history


def calibration_report(judge: Judge, held_out: list[Melody], variant_pairs=None) -> dict:
    """The two controls that make `ΔNLL` interpretable.

    Without these the filter is just a novelty detector with a threshold on it.
    """
    per_token, _ = judge.nll_of_melodies(held_out)
    report = {
        "held_out_n": len(held_out),
        "nll_mean": float(per_token.mean()),
        "nll_std": float(per_token.std()),
        "nll_p05": float(np.percentile(per_token, 5)),
        "nll_p95": float(np.percentile(per_token, 95)),
    }

    if variant_pairs:
        scores = judge.score_drafts(variant_pairs)
        deltas = np.array([s.delta_nll for s in scores])
        report["variant_pairs"] = {
            "n": len(scores),
            "delta_nll_mean": float(deltas.mean()),
            "delta_nll_median": float(np.median(deltas)),
            "delta_nll_std": float(deltas.std()),
            "fraction_positive": float((deltas > 0).mean()),
        }
    return report


def save_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
    return path
