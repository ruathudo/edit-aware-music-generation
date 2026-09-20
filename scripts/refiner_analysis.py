"""
Paired analysis of the refinement sweep (spec 6.1).

    python scripts/refiner_analysis.py results/refiner/sweep_seed0.json

The published paper defined this analysis in its Section 6.2 and never reported
it. It is what turns an aggregate percentage into a claim that survives review:
win rate, paired delta, a bootstrap 95% interval, and a signed-rank test.

Two comparisons matter here, and the second is the one that decides whether the
task is being done at all:

  - each condition against the imitation baseline B1, which is the paper's
    actual claim;
  - each condition against B3, copying the draft unchanged. With realistic
    drafts only a few notes in fifty need changing, so copying is a strong local
    optimum. A model that cannot beat B3 has not learned to correct anything,
    whatever its Correction Cost looks like in isolation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import paired_comparison  # noqa: E402
from src.judge import save_json  # noqa: E402


def load_costs(label: str, directory: Path) -> np.ndarray | None:
    path = directory / f"costs_{label}.npy"
    return np.load(path) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", nargs="?", default="results/refiner/sweep_seed0.json")
    parser.add_argument("--out", default="results/refiner/paired_analysis.json")
    args = parser.parse_args()

    sweep = json.loads(Path(args.sweep).read_text())
    directory = Path(args.sweep).parent
    rows = sweep["rows"]

    b3 = load_costs("b3", directory)
    baseline_label = next(
        (r["label"] for r in rows if r["alpha"] == 0.0 and not r["use_edit_head"]), None
    )
    b1 = load_costs(baseline_label, directory) if baseline_label else None

    print(f"sweep: {args.sweep}")
    print(f"B1 (imitation, no head): {baseline_label}")
    print(f"B3 (copy input): CC {b3.mean():.3f}\n" if b3 is not None else "")

    analysis = []
    header = (
        f"{'condition':22s} {'CC':>7s} {'fid':>7s} {'macroF1':>8s} "
        f"{'vs B3 win%':>11s} {'vs B3 p':>9s} {'vs B1 win%':>11s} {'vs B1 p':>9s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        costs = load_costs(r["label"], directory)
        if costs is None:
            continue

        entry = {"label": r["label"], "alpha": r["alpha"],
                 "use_edit_head": r["use_edit_head"],
                 "correction_cost": r["correction_cost"],
                 "fidelity_per_note": r["fidelity_per_note"]}

        if b3 is not None and len(b3) == len(costs):
            entry["vs_b3"] = paired_comparison(b3, costs)
        if b1 is not None and len(b1) == len(costs):
            entry["vs_b1"] = paired_comparison(b1, costs)
        if "edit_scores" in r:
            entry["macro_f1"] = r["edit_scores"]["macro_f1"]
            entry["localization_f1"] = r["edit_scores"]["localization"]["f1"]
        analysis.append(entry)

        def fmt(cmp_key, field):
            block = entry.get(cmp_key)
            if not block:
                return "     -"
            if field == "win":
                return f"{block['win_rate']:10.1f}%"
            p = block["wilcoxon"]["p_value"]
            return f"{p:9.2e}" if p is not None else "        -"

        print(
            f"{r['label']:22s} {r['correction_cost']:7.3f} {r['fidelity_per_note']:7.4f} "
            f"{entry.get('macro_f1', float('nan')):8.3f} "
            f"{fmt('vs_b3','win')} {fmt('vs_b3','p')} "
            f"{fmt('vs_b1','win')} {fmt('vs_b1','p')}"
        )

    # The alpha curve, over conditions that actually used the head.
    curve = [(e["alpha"], e["correction_cost"]) for e in analysis if e["use_edit_head"]]
    if curve:
        print("\nalpha curve (Correction Cost, lower is better):")
        for alpha, cost in sorted(curve):
            print(f"  alpha {alpha:<5} CC {cost:.3f}")
        best = min(curve, key=lambda c: c[1])
        print(f"  best: alpha {best[0]} at CC {best[1]:.3f}")

    save_json({"source": args.sweep, "analysis": analysis}, args.out)
    print(f"\nanalysis -> {args.out}")


if __name__ == "__main__":
    main()
