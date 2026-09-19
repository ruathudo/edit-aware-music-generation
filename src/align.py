"""
Wagner-Fischer alignment over note tuples.

Shared primitive underneath three consumers: ground-truth edit-script derivation
(draft sources D1-D4), Correction Cost and fidelity (evaluation), and the 5-class
per-position labels for the edit head. A bug here silently corrupts every reported
number, so this module is built and tested before anything else runs.

Alignment is schema-dependent, not corpus-dependent: it operates on note tuples and
neither knows nor cares which corpus produced them. What it *is* coupled to is the
note schema and the cost vector, both of which are parameters here rather than
inlined constants.
"""
from dataclasses import dataclass

# Sentinel pitch for an explicit rest. Real MIDI pitches are 21-108, so a negative
# value cannot collide with one.
REST = -1

# A note is (pitch, duration); pitch may be REST.
Note = tuple[int, int]


@dataclass(frozen=True)
class CostVector:
    """Per-operation costs for alignment.

    Defaults reproduce the published Correction Cost (all unit costs, so a
    substitution differing in both attributes costs 2). The weighted CC variant
    is obtained by passing a non-uniform vector.
    """
    delete: float = 1.0
    insert: float = 1.0
    pitch_sub: float = 1.0
    dur_sub: float = 1.0


DEFAULT_COSTS = CostVector()


# 5-class edit vocabulary for the per-position head.
KEEP, SUB_PITCH, SUB_DUR, INSERT_BEFORE, DELETE = range(5)
EDIT_CLASSES = ["KEEP", "SUB_PITCH", "SUB_DUR", "INSERT_BEFORE", "DELETE"]

# Traceback moves.
_DIAG, _DEL, _INS = 0, 1, 2

# Canonical ordering at a shared source index: an insertion lands before the note
# it precedes, and attribute modifications come before deletions (spec 3.4).
_OP_ORDER = {"insert": 0, "sub_pitch": 1, "sub_dur": 2, "delete": 3}


@dataclass(frozen=True)
class EditOp:
    """One canonical edit turning the source melody into the target.

    src_index indexes the source (M_0); for "insert" it is the position the new
    note is inserted *before*, so it may equal len(source). tgt_index indexes the
    target (M_star) and is None for deletions.
    """
    op: str
    src_index: int
    tgt_index: int | None = None
    payload: object = None

    def sort_key(self) -> tuple[int, int]:
        return (self.src_index, _OP_ORDER[self.op])


def substitution_cost(a: Note, b: Note, costs: CostVector = DEFAULT_COSTS) -> float:
    """Cost of aligning note a to note b: additive over the two attributes."""
    return (
        (costs.pitch_sub if a[0] != b[0] else 0.0)
        + (costs.dur_sub if a[1] != b[1] else 0.0)
    )


def align(
    source: list[Note],
    target: list[Note],
    costs: CostVector = DEFAULT_COSTS,
) -> tuple[float, list[tuple[int, int | None, int | None]]]:
    """Align source to target, returning (total_cost, moves).

    Each move is (kind, src_index, tgt_index) with kind in {_DIAG, _DEL, _INS},
    ordered left to right. Backpointers are recorded during the forward pass
    rather than recovered by recomputing costs during traceback, so tie-breaking
    is decided once and cannot drift between the two passes.

    Ties prefer the diagonal. This matters: when a note differs in both pitch and
    duration, substitution costs 2 and delete+insert also costs 2. Preferring the
    diagonal keeps it a single substitution, which is both more interpretable as
    an edit script and consistent with the published metric's behaviour.
    """
    n, m = len(source), len(target)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bp = [[_DIAG] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i * costs.delete
        bp[i][0] = _DEL
    for j in range(1, m + 1):
        dp[0][j] = j * costs.insert
        bp[0][j] = _INS

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + substitution_cost(source[i - 1], target[j - 1], costs)
            dele = dp[i - 1][j] + costs.delete
            ins = dp[i][j - 1] + costs.insert

            best = min(diag, dele, ins)
            dp[i][j] = best
            if best == diag:
                bp[i][j] = _DIAG
            elif best == dele:
                bp[i][j] = _DEL
            else:
                bp[i][j] = _INS

    moves: list[tuple[int, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = bp[i][j]
        if move == _DIAG and i > 0 and j > 0:
            moves.append((_DIAG, i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == _DEL and i > 0:
            moves.append((_DEL, i - 1, None))
            i -= 1
        else:
            # Insertion happens before source position i.
            moves.append((_INS, i, j - 1))
            j -= 1

    moves.reverse()
    return dp[n][m], moves


def correction_cost(
    source: list[Note],
    target: list[Note],
    costs: CostVector = DEFAULT_COSTS,
) -> float:
    """Minimal cost of turning source into target."""
    return align(source, target, costs)[0]


def edit_script(
    source: list[Note],
    target: list[Note],
    costs: CostVector = DEFAULT_COSTS,
) -> list[EditOp]:
    """Derive the canonical edit script turning source into target.

    A substitution differing in both attributes emits two ops (sub_pitch and
    sub_dur) at the same index, so the script's length in unit costs equals the
    Correction Cost under the default vector.
    """
    _, moves = align(source, target, costs)
    script: list[EditOp] = []

    for kind, i, j in moves:
        if kind == _DIAG:
            src_note, tgt_note = source[i], target[j]
            if src_note[0] != tgt_note[0]:
                script.append(EditOp("sub_pitch", i, j, (src_note[0], tgt_note[0])))
            if src_note[1] != tgt_note[1]:
                script.append(EditOp("sub_dur", i, j, (src_note[1], tgt_note[1])))
        elif kind == _DEL:
            script.append(EditOp("delete", i, None, source[i]))
        else:
            script.append(EditOp("insert", i, j, target[j]))

    return canonicalize(script)


def canonicalize(script: list[EditOp]) -> list[EditOp]:
    """Order a script by source position, insertions then modifications then deletions.

    Python's sort is stable, so multiple insertions before the same position keep
    their target order.
    """
    return sorted(script, key=EditOp.sort_key)


def apply_script(source: list[Note], script: list[EditOp]) -> list[Note]:
    """Apply an edit script to source, returning the resulting melody.

    Inverse of edit_script(): applying a derived script to its own source must
    reproduce the target exactly, which is the round-trip property the tests
    assert. Also used to build M_0 from an injected script in D1/D3.
    """
    by_index: dict[int, list[EditOp]] = {}
    for op in script:
        by_index.setdefault(op.src_index, []).append(op)

    out: list[Note] = []
    for i in range(len(source) + 1):
        ops = by_index.get(i, [])

        for op in ops:
            if op.op == "insert":
                out.append(op.payload)

        if i == len(source):
            break
        if any(op.op == "delete" for op in ops):
            continue

        pitch, dur = source[i]
        for op in ops:
            if op.op == "sub_pitch":
                pitch = op.payload[1]
            elif op.op == "sub_dur":
                dur = op.payload[1]
        out.append((pitch, dur))

    return out


def edit_labels(
    source: list[Note],
    target: list[Note],
    costs: CostVector = DEFAULT_COSTS,
    both_differ: int = SUB_PITCH,
) -> list[int]:
    """Per-position 5-class labels over source positions, for the edit head.

    Returns len(source) + 1 labels. The trailing slot corresponds to the EOS
    position and exists so that insertions past the end of the source have
    somewhere to attach; it can only ever be KEEP or INSERT_BEFORE.

    Two lossy reductions are unavoidable when collapsing a script to one label
    per position, and both are deliberate:
      - A note differing in *both* pitch and duration has no dedicated class in
        the 5-class vocabulary, so it is labelled `both_differ` (SUB_PITCH by
        default). The full script from edit_script() retains both operations.
      - Several insertions before the same position collapse to one
        INSERT_BEFORE; the label marks that something is missing here, not how
        much.

    When several operations land on one position, the first in canonical order
    wins (insert, then pitch, then duration, then delete). Insertion outranking
    substitution is deliberate -- a missing note is the more structural error.
    Insertion and deletion can co-occur at a position only under a non-uniform
    cost vector, since unit costs make the aligner prefer the diagonal there.
    """
    labels = [KEEP] * (len(source) + 1)

    for op in edit_script(source, target, costs):
        if labels[op.src_index] != KEEP:
            continue

        if op.op == "insert":
            labels[op.src_index] = INSERT_BEFORE
        elif op.op == "delete":
            labels[op.src_index] = DELETE
        elif _differs_in_both(op, source, target):
            labels[op.src_index] = both_differ
        else:
            labels[op.src_index] = SUB_PITCH if op.op == "sub_pitch" else SUB_DUR

    return labels


def _differs_in_both(op: EditOp, source: list[Note], target: list[Note]) -> bool:
    if op.tgt_index is None:
        return False
    src_note, tgt_note = source[op.src_index], target[op.tgt_index]
    return src_note[0] != tgt_note[0] and src_note[1] != tgt_note[1]
