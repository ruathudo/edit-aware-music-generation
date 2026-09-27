"""
What does the decoder write where the gate opens? Localization versus repair.

    python scripts/repair_accuracy.py

Gated decoding substitutes one event at a time, so output event i replaces draft
event i. Aligning each draft to its reference classifies every event the gate
changed:

  at a true substitution   the draft event was aligned to a *different* reference
                           event -- the gate found a real error
  broke a correct note     the draft event already matched the reference
  deletion / other         the draft event has no reference partner (the fix
                           was to delete it, which the gate cannot do)

and, at true substitutions, whether the decoder wrote the reference's event
exactly, or at least its pitch. Localization quality is the first split;
repair quality is the second. Reads the saved gated outputs of the headline
runs (seeds 0-2, alpha 0.5, tau 0.8) and of the epoch-2 generator runs.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import align  # noqa: E402
from src.judge import save_json  # noqa: E402

DIAG = 0  # src.align's diagonal move
CONDITIONS = [
    ("MTC", "D1", "results/refiner/headline_r2/mtc_d1"),
    ("MTC", "D3", "results/refiner/headline_r2/mtc_d3"),
    ("MTC", "D2 ep2", "results/refiner/headline_ep2/mtc_d2infill"),
    ("Essen", "D1", "results/refiner/headline_r2/essen_d1"),
    ("Essen", "D3", "results/refiner/headline_r2/essen_d3"),
    ("Essen", "D2 ep2", "results/refiner/headline_ep2/essen_d2infill"),
]


def tally(path):
    t = dict(changed=0, at_sub=0, exact=0, pitch=0, broke=0, other=0, skipped=0)
    for example in json.loads(path.read_text()):
        draft = [tuple(n) for n in example["draft"]]
        ref = [tuple(n) for n in example["reference"]]
        gated = [tuple(n) for n in example["gated_0.8"]]
        if len(gated) != len(draft):   # the gate only substitutes; guard anyway
            t["skipped"] += 1
            continue
        partner = {i: j for kind, i, j in align(draft, ref)[1] if kind == DIAG}
        for i, (before, after) in enumerate(zip(draft, gated)):
            if before == after:
                continue
            t["changed"] += 1
            j = partner.get(i)
            if j is None:
                t["other"] += 1
            elif before == ref[j]:
                t["broke"] += 1
            else:
                t["at_sub"] += 1
                t["exact"] += after == ref[j]
                t["pitch"] += after[0] == ref[j][0]
    return t


def main():
    rows = []
    for corpus, source, directory in CONDITIONS:
        paths = sorted(Path(directory).glob("outputs_alpha0.5_seed*_binary.json"))
        if not paths:
            print(f"{corpus} {source}: no saved outputs, skipped")
            continue
        per_seed = [tally(p) for p in paths]
        t = {k: sum(s[k] for s in per_seed) for k in per_seed[0]}
        c, sub = t["changed"], max(t["at_sub"], 1)
        row = {"corpus": corpus, "source": source, "seeds": len(paths), "changed": c,
               "share_true_substitution": t["at_sub"] / c, "share_broke_correct": t["broke"] / c,
               "share_deletion_or_other": t["other"] / c, "exact_repair": t["exact"] / sub,
               "pitch_repair": t["pitch"] / sub, "skipped_examples": t["skipped"]}
        rows.append(row)
        print(f"{corpus:5s} {source:7s} changed {c:5d}  true sub {row['share_true_substitution']:.2f}  "
              f"broke {row['share_broke_correct']:.2f}  del/other {row['share_deletion_or_other']:.2f}"
              f"  | exact repair {row['exact_repair']:.2f}  pitch {row['pitch_repair']:.2f}")
    path = Path("results/refiner/controls/repair_accuracy.json")
    save_json({"rows": rows}, path)
    print(f"\nrepair accuracy -> {path}")


if __name__ == "__main__":
    main()
