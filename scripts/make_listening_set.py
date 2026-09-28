"""
Build the listening-test item set and its answer key.

    python scripts/make_listening_set.py

Two tasks, both on MTC test material (the paper's main test set):

  spot   Excerpts from D1, D3, D2 (trained generator) and D2 ep2 (2-epoch generator)
         drafts, plus untouched references as catch trials. Raters click the notes
         that sound wrong and rate naturalness 1-5. Each excerpt comes from a
         different reference melody, so no rater hears two versions of one tune.
  pair   Two versions of an excerpt that differ at one highlighted note.
         main:  an ep2 repair that missed the reference, against the reference's
                note, in the output's context (does "different" mean "wrong"?)
         check: the same repair against the draft's note it replaced (attention
                check; the judge prefers the repair 97% of the time)

The page gets only opaque ids, notes and the highlighted slot
(`results/listening/items.json`). Everything needed to score the ratings --
source, true edit positions, which side is which, judge scores -- goes to
`results/listening/key.json`, which raters never see.
"""
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import align, edit_labels  # noqa: E402
from src.masked_judge import load_masked_judge  # noqa: E402
from scripts.repair_plausibility import slot_surprisal, test_examples  # noqa: E402

DIAG = 0  # src.align's diagonal move
SEED = 2026
SPOT_PER_SOURCE = 20
PAIRS_MAIN, PAIRS_CHECK = 30, 10
MAX_EXCERPT, PAIR_CONTEXT = 20, 8
SWEEPS = {
    "D1": "results/refiner/headline_r2/mtc_d1",
    "D3": "results/refiner/headline_r2/mtc_d3",
    "D2": "results/refiner/headline_r2/mtc_d2infill",
    "D2-ep2": "results/refiner/headline_ep2/mtc_d2infill",
}
OUT = Path("results/listening")


def excerpt_window(n, positions, rng):
    """[start, end) of at most MAX_EXCERPT events covering every position, or None."""
    if n <= MAX_EXCERPT:
        return 0, n
    if not positions:
        start = rng.randrange(0, n - MAX_EXCERPT + 1)
        return start, start + MAX_EXCERPT
    lo, hi = min(positions), max(positions)
    if hi - lo + 1 > MAX_EXCERPT:
        return None
    start = rng.randrange(max(0, hi - MAX_EXCERPT + 1), min(lo, n - MAX_EXCERPT) + 1)
    return start, start + MAX_EXCERPT


def main():
    rng = random.Random(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    examples = {}
    for name, directory in SWEEPS.items():
        config = json.loads(Path(directory).with_suffix(".json").read_text())["config"]
        tokenizer, examples[name] = test_examples(config, device)

    # ---- spot task: one excerpt per reference melody, disjoint across sources
    refs = sorted({ex.reference.id for ex in examples["D1"]})
    rng.shuffle(refs)
    unused = list(refs)
    items, key = [], {}

    def add_item(task, payload, answer):
        # Opaque ids: a sequential number would reveal the source, since items are
        # generated one source at a time.
        item_id = f"{task[0]}-{rng.randrange(16**6):06x}"
        while item_id in key:
            item_id = f"{task[0]}-{rng.randrange(16**6):06x}"
        items.append({"id": item_id, "task": task, **payload})
        key[item_id] = answer

    for source in ("D1", "D3", "D2", "D2-ep2", "reference"):
        pool = examples["D1"] if source == "reference" else examples[source]
        by_ref = {}
        for ex in pool:
            by_ref.setdefault(ex.reference.id, []).append(ex)
        taken = 0
        for ref_id in list(unused):
            if taken == SPOT_PER_SOURCE:
                break
            ex = rng.choice(by_ref.get(ref_id, [None]))
            if ex is None:
                continue
            ref_notes = list(ex.reference.notes)
            notes = ref_notes if source == "reference" else list(ex.draft_notes)
            labels = [0] * (len(notes) + 1) if source == "reference" else edit_labels(notes, ref_notes)
            wrong = [i for i, lab in enumerate(labels[:len(notes)]) if lab != 0]
            if source != "reference" and not wrong:
                continue
            window = excerpt_window(len(notes), wrong, rng)
            if window is None:
                continue
            s, e = window
            add_item("spot", {"notes": notes[s:e]},
                     {"source": source, "reference_id": ref_id, "window": [s, e],
                      "wrong_positions": [i - s for i in wrong if s <= i < e]})
            unused.remove(ref_id)
            taken += 1
        print(f"spot {source:9s} {taken} items")

    # ---- pair task: ep2 mismatched repairs (seed 0 outputs)
    masked = load_masked_judge("models/masked_judge.pth", device)
    outputs = json.loads((Path(SWEEPS["D2-ep2"]) / "outputs_alpha0.5_seed0_binary.json").read_text())
    candidates = []
    for ex, out in zip(examples["D2-ep2"], outputs):
        draft = [tuple(n) for n in out["draft"]]
        assert draft == [tuple(n) for n in ex.draft_notes]
        gated = [tuple(n) for n in out["gated_0.8"]]
        if gated == draft or len(gated) != len(draft):
            continue
        ref = list(ex.reference.notes)
        partner = {i: j for kind, i, j in align(draft, ref)[1] if kind == DIAG}
        for i, (before, after) in enumerate(zip(draft, gated)):
            j = partner.get(i)
            if before != after and j is not None and before != ref[j] and after != ref[j]:
                candidates.append((ex, gated, i, before, after, ref[j]))
    rng.shuffle(candidates)
    seen, chosen = set(), []
    for c in candidates:                      # one pair per reference melody
        if c[0].reference.id not in seen:
            seen.add(c[0].reference.id); chosen.append(c)
    for kind, group in (("main", chosen[:PAIRS_MAIN]), ("check", chosen[PAIRS_MAIN:PAIRS_MAIN + PAIRS_CHECK])):
        for ex, gated, i, before, after, ref_note in group:
            other = ref_note if kind == "main" else before
            s = max(0, i - PAIR_CONTEXT); e = min(len(gated), i + PAIR_CONTEXT + 1)
            a_is_repair = rng.random() < 0.5
            a_note, b_note = (after, other) if a_is_repair else (other, after)
            ctx_a, ctx_b = list(gated[s:e]), list(gated[s:e])
            ctx_a[i - s], ctx_b[i - s] = a_note, b_note
            surprisal = slot_surprisal(masked, tokenizer, gated, ex.reference.bar_starts, i,
                                       [after, other], device)
            add_item("pair", {"a": ctx_a, "b": ctx_b, "slot": i - s},
                     {"kind": kind, "reference_id": ex.reference.id,
                      "a_is": "repair" if a_is_repair else ("reference" if kind == "main" else "draft"),
                      "b_is": ("reference" if kind == "main" else "draft") if a_is_repair else "repair",
                      "judge_surprisal": {"repair": surprisal[0],
                                          ("reference" if kind == "main" else "draft"): surprisal[1]}})
        print(f"pair {kind:5s} {len(group)} items")

    rng.shuffle(items)   # stored order carries no information either
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "items.json").write_text(json.dumps({"version": 1, "ticks_per_quarter": 24,
                                                "items": items}, separators=(",", ":")))
    (OUT / "key.json").write_text(json.dumps({"version": 1, "seed": SEED, "key": key}, indent=1))
    print(f"\n{len(items)} items -> {OUT / 'items.json'}; key -> {OUT / 'key.json'}")


if __name__ == "__main__":
    main()
