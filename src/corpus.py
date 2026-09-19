"""
Corpus loading for the Meertens Tune Collections and the Essen Folksong Collection.

Both arrive as gzipped JSONL feature sequences from Zenodo record 3551003 (the
data behind the MTCFeatures package), which is preferable to parsing **kern for
two reasons: the licence is explicit (CC BY-NC-SA 3.0, versus CCARH's
"licensed not sold" terms on the raw kern), and every annotation the spec asks us
to *infer* is already present.

That last point removes work from several places in the plan:
  - `tonic`/`mode`/`scaledegree` mean D3's out-of-scale rule needs no
    Krumhansl-Schmuckler key finding, and the judge's scale-conformance metric
    reads straight off the annotation.
  - `beat`/`beatstrength`/`timesignature` give D3 its metric-misalignment rule
    and give the tokenizer real barlines.
  - `phrase_ix`/`phrase_end` give D3 its cadential rule.
  - `tunefamily` gives D4 human-authored variant pairs without recruitment.

Durations are fractional quarter-lengths (triplets appear as 1/3), so they are
converted to integer ticks. TICKS_PER_QUARTER = 24 represents every duration in
both corpora exactly -- verified across 462k notes, zero failures -- because the
observed denominators are all divisors of 24.

Data is *not* committed: `data/` is gitignored, and neither corpus permits
redistribution. Run `python -m src.corpus fetch` to download.
"""
import gzip
import json
import random
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from src.align import REST

TICKS_PER_QUARTER = 24

DATA_DIR = Path("data/corpora")

# Zenodo record 3551003. MD5s are the upstream manifest's, checked on download.
CORPORA = {
    "MTC-ANN": {
        "filename": "MTC-ANN-2.0.1_sequences-1.1.jsonl.gz",
        "md5": "1a15615a4f7222702276565a0253e329",
        "note": "360 melodies in 26 tune families; every melody has a variant.",
    },
    "MTC-FS-INST": {
        "filename": "MTC-FS-INST-2.0_sequences-1.1.jsonl.gz",
        "md5": "2c5ef01870e9e202df392fed5b34b061",
        "note": "Large instrumental/vocal set; no tune-family annotation.",
    },
    "ESSEN": {
        "filename": "essen_sequences-1.1.jsonl.gz",
        "md5": "c9290572aefb97af60b206775d050876",
        "note": "8469 melodies; cross-corpus generalization check.",
    },
}

_BASE_URL = "https://zenodo.org/record/3551003/files"


@dataclass(frozen=True)
class NoteEvent:
    """One event: a sounding note, or a rest when `pitch is REST`.

    Annotations are carried per event and are None on rests, which have no
    scale degree or phrase membership of their own.
    """
    pitch: int
    duration: int  # ticks
    scaledegree: int | None = None
    beat: float | None = None
    beatstrength: float | None = None
    phrase_ix: int | None = None
    phrase_end: bool = False
    timesignature: str | None = None

    @property
    def is_rest(self) -> bool:
        return self.pitch == REST


@dataclass(frozen=True)
class Melody:
    """A melody plus the annotations D3 and the judge need."""
    id: str
    source: str
    events: tuple[NoteEvent, ...]
    bar_starts: frozenset[int] = frozenset()
    tunefamily: str | None = None
    tonic: str | None = None
    mode: str | None = None
    timesignature: str | None = None
    freemeter: bool = False

    @property
    def notes(self) -> list[tuple[int, int]]:
        """The (pitch, duration) view the aligner and tokenizer consume."""
        return [(e.pitch, e.duration) for e in self.events]

    @property
    def n_sounding(self) -> int:
        return sum(1 for e in self.events if not e.is_rest)

    def __len__(self) -> int:
        return len(self.events)


def _ticks(frac: str | float) -> int:
    """Convert a fractional quarter-length to integer ticks."""
    value = Fraction(frac) * TICKS_PER_QUARTER
    if value.denominator != 1:
        raise ValueError(f"duration {frac} is not representable at TPQ={TICKS_PER_QUARTER}")
    return int(value)


def beat_ticks(timesignature: str | None) -> int | None:
    """Ticks per counted beat, e.g. an eighth in 6/8 and a quarter in 3/4."""
    if not timesignature or "/" not in timesignature:
        return None
    try:
        denominator = int(timesignature.split("/")[1])
    except ValueError:
        return None
    value = Fraction(4, denominator) * TICKS_PER_QUARTER
    return int(value) if value.denominator == 1 else None


def bar_ticks(timesignature: str | None) -> int | None:
    """Ticks in a full bar of the given metre."""
    if not timesignature or "/" not in timesignature:
        return None
    try:
        numerator = int(timesignature.split("/")[0])
    except ValueError:
        return None
    per_beat = beat_ticks(timesignature)
    return numerator * per_beat if per_beat is not None else None


def infer_bar_starts(
    notes: list[tuple[int, int]],
    timesignature: str | None,
    start_offset: int = 0,
) -> frozenset[int]:
    """Bar starts computed from cumulative duration, for melodies with no annotation.

    Drafts have no `beat` column, so their barlines cannot be read off the corpus.
    Scoring a draft against its reference therefore has to infer bars for *both*
    by this same function -- mixing annotated bars on one side with inferred bars
    on the other would make the comparison measure the inference method as much
    as the draft.

    `start_offset` is how far into a bar the melody begins, which is what encodes
    a pickup; see `pickup_offset`.
    """
    width = bar_ticks(timesignature)
    if not width:
        return frozenset()

    starts: set[int] = set()
    position = start_offset
    current = position // width

    for index, (_, duration) in enumerate(notes):
        bar = position // width
        if bar > current:
            starts.add(index)
            current = bar
        position += duration

    return frozenset(starts)


def pickup_offset(melody: Melody) -> int:
    """How far into its first bar a melody begins, in ticks."""
    for event in melody.events:
        if event.beat is not None:
            per_beat = beat_ticks(event.timesignature or melody.timesignature)
            if per_beat is None:
                return 0
            return int(round((event.beat - 1) * per_beat))
    return 0


def _first(values):
    for v in values or []:
        if v is not None:
            return v
    return None


def melody_from_record(record: dict, source: str) -> Melody:
    """Build a Melody from one MTCFeatures JSONL record.

    Rests are materialized as events. The corpus stores a rest as
    `restduration_frac` on the note that *precedes* it -- verified against the
    `nextisrest` flag, which co-occurs with it 4876 times and with the previous
    note's flag zero times -- so the rest event is emitted after its note.
    """
    f = record["features"]
    pitches = f["midipitch"]
    n = len(pitches)

    beats = f.get("beat") or [None] * n
    rests = f.get("restduration_frac") or [None] * n

    events: list[NoteEvent] = []
    bar_starts: set[int] = set()

    for i in range(n):
        # A new bar begins where the beat count stops advancing. Reading it from
        # the annotation rather than from cumulative duration is what makes
        # pickup bars and mid-melody metre changes come out right. The
        # comparison is <= rather than <: two onsets cannot share a metric
        # position within one bar, so an equal beat also means a new bar.
        new_bar = (
            i > 0
            and beats[i] is not None
            and beats[i - 1] is not None
            and beats[i] <= beats[i - 1]
        )
        if new_bar:
            bar_starts.add(len(events))

        timesig = _get(f, "timesignature", i)
        events.append(
            NoteEvent(
                pitch=int(pitches[i]),
                duration=_ticks(f["duration_frac"][i]),
                scaledegree=_get(f, "scaledegree", i),
                beat=beats[i],
                beatstrength=_get(f, "beatstrength", i),
                phrase_ix=_get(f, "phrase_ix", i),
                phrase_end=bool(_get(f, "phrase_end", i)),
                timesignature=timesig,
            )
        )

        # The rest follows its note, so it falls at the end of that note's bar
        # and the next barline is marked after it.
        if rests[i]:
            events.append(NoteEvent(pitch=REST, duration=_ticks(rests[i])))

    return Melody(
        id=record["id"],
        source=source,
        events=tuple(events),
        bar_starts=frozenset(bar_starts),
        tunefamily=record.get("tunefamily") or None,
        tonic=_first(f.get("tonic")),
        mode=_first(f.get("mode")),
        timesignature=_first(f.get("timesignature")),
        freemeter=bool(record.get("freemeter")),
    )


def _get(features: dict, key: str, i: int):
    seq = features.get(key)
    return seq[i] if seq is not None and i < len(seq) else None


def load(name: str, limit: int | None = None, data_dir: Path | None = None) -> list[Melody]:
    """Load a corpus by registry name."""
    if name not in CORPORA:
        raise KeyError(f"unknown corpus {name!r}; known: {sorted(CORPORA)}")

    path = (data_dir or DATA_DIR) / CORPORA[name]["filename"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.corpus fetch {name}"
        )

    out = []
    with gzip.open(path, "rt") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            out.append(melody_from_record(json.loads(line), name))
    return out


def duration_vocabulary(melodies: list[Melody]) -> tuple[int, ...]:
    """The exact set of durations present, in ticks.

    Derived rather than guessed: the legacy pipeline's hardcoded allowlist
    rejected whole melodies for a single unlisted duration.
    """
    return tuple(sorted({e.duration for m in melodies for e in m.events}))


def pitch_range(melodies: list[Melody]) -> tuple[int, int]:
    pitches = [e.pitch for m in melodies for e in m.events if not e.is_rest]
    return (min(pitches), max(pitches)) if pitches else (0, 0)


def split_by_tunefamily(
    melodies: list[Melody],
    fractions: dict[str, float],
    seed: int = 42,
) -> dict[str, list[Melody]]:
    """Split so that no tune family spans two splits.

    This matters more than it looks. Tune-family members are variants of one
    another, so a random split would put a near-duplicate of a test melody into
    training and quietly inflate every number. It also keeps the judge `J` honest:
    the spec requires `J` to be trained on data disjoint from `G`, and family
    leakage would violate that even with disjoint melody ids.

    Melodies with no tune family are treated as singleton families.
    """
    if not fractions:
        raise ValueError("fractions must be non-empty")

    groups: dict[str, list[Melody]] = {}
    for i, m in enumerate(melodies):
        key = m.tunefamily or f"__singleton_{i}"
        groups.setdefault(key, []).append(m)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    total = sum(fractions.values())
    names = list(fractions)
    quotas = {k: len(melodies) * v / total for k, v in fractions.items()}

    out: dict[str, list[Melody]] = {k: [] for k in names}
    for key in keys:
        # Assign each family to whichever split is furthest below its quota,
        # which keeps sizes close despite families varying in size.
        target = max(names, key=lambda k: quotas[k] - len(out[k]))
        out[target].extend(groups[key])
    return out


def summary(melodies: list[Melody]) -> dict:
    """Corpus statistics for the paper's dataset table."""
    if not melodies:
        return {"melodies": 0}

    from collections import Counter

    lengths = [len(m) for m in melodies]
    rests = sum(1 for m in melodies for e in m.events if e.is_rest)
    events = sum(lengths)
    family_sizes = Counter(m.tunefamily for m in melodies if m.tunefamily)
    families = set(family_sizes)
    lo, hi = pitch_range(melodies)

    return {
        "melodies": len(melodies),
        "events": events,
        "events_per_melody_mean": events / len(melodies),
        "events_per_melody_max": max(lengths),
        "rest_fraction": rests / max(events, 1),
        "pitch_range": (lo, hi),
        "distinct_durations": len(duration_vocabulary(melodies)),
        "tune_families": len(families),
        "in_a_family_of_2plus": sum(
            1 for m in melodies if m.tunefamily and family_sizes[m.tunefamily] >= 2
        ),
        "freemeter": sum(1 for m in melodies if m.freemeter),
        "with_bars": sum(1 for m in melodies if m.bar_starts),
    }


# ---------- fetching ----------

def fetch(name: str, data_dir: Path | None = None) -> Path:
    """Download one corpus from Zenodo and verify its MD5."""
    import hashlib
    import urllib.request

    spec = CORPORA[name]
    target = (data_dir or DATA_DIR)
    target.mkdir(parents=True, exist_ok=True)
    path = target / spec["filename"]

    if not path.exists():
        url = f"{_BASE_URL}/{spec['filename']}?download=1"
        print(f"downloading {name} <- {url}")
        urllib.request.urlretrieve(url, path)

    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if digest != spec["md5"]:
        raise ValueError(f"MD5 mismatch for {path}: {digest} != {spec['md5']}")
    print(f"{name}: {path} (md5 ok)")
    return path


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args and args[0] == "fetch":
        for name in (args[1:] or ["MTC-ANN", "ESSEN"]):
            fetch(name)
    else:
        for name in ("MTC-ANN", "ESSEN"):
            try:
                mels = load(name)
            except FileNotFoundError as exc:
                print(f"{name}: {exc}")
                continue
            print(f"\n=== {name} ===")
            for k, v in summary(mels).items():
                print(f"  {k:24s} {v}")
