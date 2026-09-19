"""
Draft sources (spec 3.2), behind one interface.

Each source turns a reference melody `M_star` into a draft `M_0` plus the edit
script relating them. The script is always *derived by alignment* rather than
taken from the injection log, per spec 3.4 -- the injected operations are kept
alongside it so the two can be cross-checked, which is a free correctness test
on the aligner.

  D1  random perturbation -- the published scheme, reimplemented unchanged. Its
      role is now to be the control the validator exposes as weak, so it must
      keep its original behaviour exactly and must not be quietly improved.

  D3  musically grounded degradation -- errors chosen because music theory says
      they are errors. Every rule here is driven by a corpus annotation
      (`scaledegree`, `beatstrength`, `phrase_end`, `tonic`), which is why the
      corpus change had to come first: on the old 4-bar Lakh-derived data none
      of these signals existed.

D2 (model-generated) needs a trained generator and is not implemented yet; it is
deliberately off the validator's critical path, since the headline contrast is
D1 versus D3 and neither needs `G`.
"""
import random
from dataclasses import dataclass, field

from src.align import EditOp, REST, edit_script
from src.corpus import Melody, bar_ticks, beat_ticks
from src.judge import MAX_SINGABLE_LEAP

Notes = list[tuple[int, int]]


@dataclass(frozen=True)
class Priors:
    """Marginal pitch and duration distributions, for D1 to draw from.

    Sampling *uniformly* over a derived duration vocabulary is not a faithful
    translation of the published scheme, and getting this wrong inflates D1's
    apparent damage badly. The original drew from six durations that were all
    common; this corpus yields 40, of which seven carry 98.3% of the mass, so a
    uniform draw picks a sub-1%-frequency rhythm 82% of the time. That is not
    "random perturbation" -- it is an unrealistic-rhythm generator, and it tests
    a strawman rather than the reviewers' actual objection, which presumes the
    random edit draws from plausible material and may therefore land on a valid
    alternative.
    """
    durations: tuple[int, ...]
    duration_weights: tuple[float, ...]
    pitches: tuple[int, ...]
    pitch_weights: tuple[float, ...]

    @property
    def pitch_span(self) -> tuple[int, int]:
        return (min(self.pitches), max(self.pitches))

    def sample_duration(self, rng, exclude=None) -> int:
        for _ in range(16):
            value = rng.choices(self.durations, weights=self.duration_weights, k=1)[0]
            if value != exclude:
                return value
        alternatives = [d for d in self.durations if d != exclude]
        return rng.choice(alternatives) if alternatives else exclude

    def sample_pitch(self, rng) -> int:
        return rng.choices(self.pitches, weights=self.pitch_weights, k=1)[0]


def empirical_priors(melodies: list[Melody], uniform: bool = False) -> Priors:
    """Marginals measured from a corpus.

    `uniform=True` reproduces the naive translation -- flat over the derived
    vocabulary -- which is kept only so the report can show how much of D1's
    measured degradation is an artifact of that choice.
    """
    from collections import Counter

    durations = Counter()
    pitches = Counter()
    for melody in melodies:
        for event in melody.events:
            durations[event.duration] += 1
            if not event.is_rest:
                pitches[event.pitch] += 1

    dur_keys = tuple(sorted(durations))
    pit_keys = tuple(sorted(pitches))
    if uniform:
        return Priors(dur_keys, (1.0,) * len(dur_keys), pit_keys, (1.0,) * len(pit_keys))
    return Priors(
        dur_keys,
        tuple(float(durations[d]) for d in dur_keys),
        pit_keys,
        tuple(float(pitches[p]) for p in pit_keys),
    )


@dataclass
class Draft:
    """A corrupted melody together with how it differs from its reference."""
    notes: Notes
    reference: Melody
    source: str
    injected: list[str] = field(default_factory=list)
    script: list[EditOp] = field(default_factory=list)

    @property
    def n_edits(self) -> int:
        return len(self.script)


def _sounding_indices(notes: Notes) -> list[int]:
    return [i for i, (p, _) in enumerate(notes) if p != REST]


def finalize(notes: Notes, reference: Melody, source: str, injected: list[str]) -> Draft | None:
    """Derive the alignment-based script and package the draft."""
    if not notes or notes == reference.notes:
        return None
    return Draft(
        notes=notes,
        reference=reference,
        source=source,
        injected=injected,
        script=edit_script(reference.notes, notes),
    )


# ------------------------------------------------------------------ D1

# The published mixture: pitch 40%, duration 30%, delete 15%, insert 15%.
D1_OPERATIONS = ("replace_pitch", "change_duration", "delete_note", "insert_note")
D1_WEIGHTS = (0.40, 0.30, 0.15, 0.15)


def d1_random(
    melody: Melody,
    priors: Priors,
    rng: random.Random,
    max_edits: int = 5,
) -> Draft | None:
    """Uniformly random perturbation -- the legacy corruption scheme.

    Reproduced deliberately, including its weaknesses: a random pitch
    substitution in an unaccompanied melody may well produce a perfectly valid
    alternative line, which is the reviewers' central objection and the thing the
    validator is built to measure.
    """
    notes = list(melody.notes)
    injected: list[str] = []
    used: set[int] = set()
    pitch_range = priors.pitch_span

    for _ in range(rng.randint(1, max_edits)):
        op = rng.choices(D1_OPERATIONS, weights=D1_WEIGHTS, k=1)[0]

        if op == "insert_note":
            position = rng.randint(0, len(notes))
            notes.insert(position, (priors.sample_pitch(rng), priors.sample_duration(rng)))
            injected.append(op)
            used = {i + 1 if i >= position else i for i in used}
            used.add(position)
            continue

        candidates = [i for i in _sounding_indices(notes) if i not in used]
        if not candidates:
            continue
        index = rng.choice(candidates)
        pitch, duration = notes[index]

        if op == "replace_pitch":
            shifted = pitch
            while shifted == pitch:
                shifted = max(pitch_range[0], min(pitch_range[1], pitch + rng.choice([-2, -1, 1, 2])))
            notes[index] = (shifted, duration)
        elif op == "change_duration":
            replacement = priors.sample_duration(rng, exclude=duration)
            if replacement is None or replacement == duration:
                continue
            notes[index] = (pitch, replacement)
        else:  # delete_note
            notes.pop(index)
            used = {i - 1 if i > index else i for i in used if i != index}

        used.add(index)
        injected.append(op)

    return finalize(notes, melody, "D1", injected)


# ------------------------------------------------------------------ D3

def _scale_pitch_classes(melody: Melody) -> frozenset[int] | None:
    from src.judge import scale_of

    return scale_of(melody.tonic, melody.mode)


def _out_of_scale(notes: Notes, melody: Melody, rng: random.Random, span) -> str | None:
    """Move a note to a non-diatonic degree of the inferred key."""
    scale = _scale_pitch_classes(melody)
    if not scale:
        return None
    candidates = [i for i in _sounding_indices(notes) if notes[i][0] % 12 in scale]
    if not candidates:
        return None

    index = rng.choice(candidates)
    pitch, duration = notes[index]
    for shift in rng.sample([1, -1, 2, -2], 4):
        if (pitch + shift) % 12 not in scale and span[0] <= pitch + shift <= span[1]:
            notes[index] = (pitch + shift, duration)
            return "out_of_scale"
    return None


def _unsingable_leap(notes: Notes, melody: Melody, rng: random.Random, span) -> str | None:
    """Introduce an interval wider than a singer would plausibly manage."""
    indices = _sounding_indices(notes)
    if len(indices) < 2:
        return None

    index = rng.choice(indices[1:])
    previous = notes[indices[indices.index(index) - 1]][0]
    pitch, duration = notes[index]

    direction = 1 if rng.random() < 0.5 else -1
    size = MAX_SINGABLE_LEAP + rng.randint(1, 5)
    leap = previous + direction * size
    if not (span[0] <= leap <= span[1]):
        leap = previous - direction * size
    if not (span[0] <= leap <= span[1]) or leap == pitch:
        return None

    notes[index] = (leap, duration)
    return "unsingable_leap"


def _metric_misalignment(notes: Notes, melody: Melody, rng: random.Random, span) -> str | None:
    """Shift a metrically strong note off the beat by resizing its predecessor.

    Targets notes the corpus marks as metrically salient, so the damage is to
    structure rather than to a random duration.
    """
    width = bar_ticks(melody.timesignature)
    per_beat = beat_ticks(melody.timesignature)
    if not width or not per_beat or per_beat < 2:
        return None

    strong = [
        i
        for i, event in enumerate(melody.events)
        if i > 0 and (event.beatstrength or 0) >= 0.5 and i < len(notes)
    ]
    if not strong:
        return None

    index = rng.choice(strong)
    pitch, duration = notes[index - 1]
    nudge = per_beat // 2 or 1
    shifted = duration + (nudge if rng.random() < 0.5 else -nudge)
    if shifted <= 0 or shifted == duration:
        return None

    notes[index - 1] = (pitch, shifted)
    return "metric_misalignment"


def _cadential_disruption(notes: Notes, melody: Melody, rng: random.Random, span) -> str | None:
    """Move a phrase-final note away from its point of resolution."""
    from src.judge import tonic_pitch_class

    root = tonic_pitch_class(melody.tonic)
    if root is None:
        return None

    endings = [
        i
        for i, event in enumerate(melody.events)
        if event.phrase_end and i < len(notes) and notes[i][0] != REST
    ]
    if not endings:
        endings = [i for i in _sounding_indices(notes)[-1:]]
    if not endings:
        return None

    index = rng.choice(endings)
    pitch, duration = notes[index]
    for shift in rng.sample([1, -1, 2, -2, 6], 5):
        if (pitch + shift) % 12 not in {root, (root + 7) % 12} and span[0] <= pitch + shift <= span[1]:
            notes[index] = (pitch + shift, duration)
            return "cadential_disruption"
    return None


def _rhythmic_monotony(notes: Notes, melody: Melody, rng: random.Random, span) -> str | None:
    """Flatten a varied rhythmic figure to a single repeated duration."""
    indices = _sounding_indices(notes)
    if len(indices) < 4:
        return None

    start = rng.randint(0, max(0, len(indices) - 4))
    window = indices[start : start + rng.randint(3, 6)]
    durations = {notes[i][1] for i in window}
    if len(durations) < 2:
        return None

    flat = rng.choice(sorted(durations))
    for i in window:
        notes[i] = (notes[i][0], flat)
    return "rhythmic_monotony"


def _repeated_note_collapse(notes: Notes, melody: Melody, rng: random.Random, span) -> str | None:
    """Replace a melodic figure with a static repeated pitch."""
    indices = _sounding_indices(notes)
    if len(indices) < 4:
        return None

    start = rng.randint(0, max(0, len(indices) - 4))
    window = indices[start : start + rng.randint(3, 5)]
    if len({notes[i][0] for i in window}) < 2:
        return None

    held = notes[window[0]][0]
    for i in window:
        notes[i] = (held, notes[i][1])
    return "repeated_note_collapse"


D3_RULES = (
    _out_of_scale,
    _unsingable_leap,
    _metric_misalignment,
    _cadential_disruption,
    _rhythmic_monotony,
    _repeated_note_collapse,
)


def d3_musical(
    melody: Melody,
    rng: random.Random,
    max_edits: int = 5,
    rules=D3_RULES,
    pitch_range: tuple[int, int] = (33, 108),
) -> Draft | None:
    """Musically grounded degradation.

    Severity is kept comparable to D1 by drawing the same 1-5 edits, so the two
    sources differ in the *kind* of error they introduce rather than the amount.
    """
    notes = list(melody.notes)
    injected: list[str] = []

    for _ in range(rng.randint(1, max_edits)):
        rule = rng.choice(rules)
        applied = rule(notes, melody, rng, pitch_range)
        if applied:
            injected.append(applied)

    if not injected:
        return None
    return finalize(notes, melody, "D3", injected)


# ------------------------------------------------------------------ batch helpers

def generate(
    melodies: list[Melody],
    source: str,
    priors: Priors,
    seed: int = 0,
    max_edits: int = 5,
    per_melody: int = 1,
    pitch_range: tuple[int, int] = (33, 108),
) -> list[Draft]:
    """Generate drafts for a corpus, deterministically given `seed`."""
    drafts: list[Draft] = []
    for i, melody in enumerate(melodies):
        for k in range(per_melody):
            rng = random.Random(seed * 1_000_003 + i * 101 + k)
            if source == "D1":
                draft = d1_random(melody, priors, rng, max_edits)
            elif source == "D3":
                draft = d3_musical(melody, rng, max_edits, pitch_range=pitch_range)
            else:
                raise ValueError(f"unknown draft source {source!r}")
            if draft is not None:
                drafts.append(draft)
    return drafts


# ------------------------------------------------------------------ D2

def d2_continuations(
    melodies: list[Melody],
    model,
    tokenizer,
    rng: random.Random,
    temperature: float = 1.0,
    top_k: int | None = None,
    prefix_fraction: tuple[float, float] = (0.25, 0.50),
    seed: int = 0,
) -> list[Draft]:
    """Drafts produced by continuing a reference's own prefix with a generator.

    Spec 3.2: keep the first k notes of a reference and sample the rest from `G`.

    This is the draft source the validator result promoted from optional to
    central. D1 and D3 both move a melody a short distance *off* the data
    manifold, which is why they are ~100x more implausible per unit of change
    than genuine musical variation. A generator's output is on-manifold by
    construction, so D2 is the only source that can be plausible *and* worse --
    which is the deployment scenario the paper has always claimed to address.

    The known risk, which the validator exists to settle rather than assume: a
    continuation may be a valid alternative melody rather than a flawed one. That
    is why the filter is mandatory for D2, and why the report tracks how far the
    drafts drift -- a continuation that diverges wildly is not an "edit" at all.
    """
    from src.sampling import sample_continuations

    prefixes, keepers = [], []
    for melody in melodies:
        events = melody.notes
        if len(events) < 8:
            continue
        low, high = prefix_fraction
        k = rng.randint(max(1, int(len(events) * low)), max(2, int(len(events) * high)))

        tokens, spans = tokenizer.encode_with_spans(events, melody.bar_starts)
        # Cut before the k-th event's pitch token, so the prefix ends on a
        # completed (pitch, duration) pair and the grammar resumes cleanly.
        prefixes.append(tokens[: spans[k][0]])
        keepers.append(melody)

    if not prefixes:
        return []

    generated = sample_continuations(
        model, tokenizer, prefixes,
        temperature=temperature, top_k=top_k, seed=seed,
    )

    drafts = []
    for melody, tokens in zip(keepers, generated):
        notes = tokenizer.decode(tokens)
        draft = finalize(notes, melody, "D2", [f"continuation@T{temperature}"])
        if draft is not None:
            drafts.append(draft)
    return drafts
