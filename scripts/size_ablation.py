"""
Does the gated gain survive scale? (spec 7, model-size axis)

    python scripts/size_ablation.py

"One model size" was an explicit complaint on the rejected paper, so this
sweeps layers x d_model and reports, for each, whether precision-gated decoding
still beats the copy-input baseline.

The quantity to read is **gated vs B3**, not raw Correction Cost. Bigger models
decode better in absolute terms, but the claim under test is that the gate
converts localization into a real improvement over doing nothing -- and that has
to hold across capacities, not only at the one configuration that was tuned.

Localization **precision** is reported alongside, because the diagnostics showed
it is what the gain actually depends on: below roughly 0.5, editing costs more
than it saves and the gate correctly declines.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judge import save_json  # noqa: E402


def summarize(path: Path, alpha: float = 0.5, threshold: str = "0.9") -> dict | None:
    blob = json.loads(path.read_text())
    directory = path.with_suffix("")
    b3_file = directory / "costs_b3.npy"
    if not b3_file.exists():
        return None

    b3 = np.load(b3_file).mean()
    rows = [r for r in blob["rows"] if r["use_edit_head"] and r["alpha"] == alpha]
    if not rows:
        return None

    deltas, precisions = [], []
    for row in rows:
        costs = directory / f"costs_{row['label']}_gated{threshold}.npy"
        if costs.exists():
            deltas.append(b3 - np.load(costs).mean())
        block = row.get("edit_scores", {}).get("localization", {})
        if block:
            precisions.append(block.get("precision", np.nan))

    if not deltas:
        return None

    match = re.search(r"size_L(\d+)_D(\d+)", path.stem)
    return {
        "layers": int(match.group(1)) if match else None,
        "d_model": int(match.group(2)) if match else None,
        "parameters": rows[0].get("n_parameters"),
        "b3": float(b3),
        "free_cc": float(np.mean([r["correction_cost"] for r in rows])),
        "gated_delta_mean": float(np.mean(deltas)),
        "gated_delta_std": float(np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0),
        "gated_pct": float(np.mean(deltas) / b3 * 100),
        "seeds_positive": int(sum(d > 0 for d in deltas)),
        "seeds": len(deltas),
        "loc_precision": float(np.nanmean(precisions)) if precisions else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="results/refiner/size_L*_D*.json")
    parser.add_argument("--out", default="results/refiner/size_ablation.json")
    args = parser.parse_args()

    entries = [s for s in (summarize(p) for p in sorted(Path().glob(args.glob))) if s]
    if not entries:
        print("no completed size runs found")
        return

    entries.sort(key=lambda e: (e["layers"] or 0, e["d_model"] or 0))
    header = (f"{'layers':>7s} {'d_model':>8s} {'params':>9s} {'free CC':>9s} "
              f"{'gated vs B3':>16s} {'%':>7s} {'seeds+':>7s} {'locP':>6s}")
    print(header)
    print("-" * len(header))
    for e in entries:
        params = f"{e['parameters']/1e6:.1f}M" if e["parameters"] else "-"
        print(f"{e['layers']:>7d} {e['d_model']:>8d} {params:>9s} {e['free_cc']:9.3f} "
              f"{e['gated_delta_mean']:+10.3f} ±{e['gated_delta_std']:.3f} "
              f"{e['gated_pct']:+6.2f}% {e['seeds_positive']}/{e['seeds']:<5d} "
              f"{e['loc_precision']:6.3f}")

    holds = [e for e in entries if e["seeds_positive"] == e["seeds"]]
    print(f"\ngated beats B3 on every seed in {len(holds)}/{len(entries)} configurations")
    if holds:
        best = max(holds, key=lambda e: e["gated_pct"])
        print(f"best: {best['layers']} layers, d_model {best['d_model']} "
              f"-> {best['gated_pct']:+.2f}%")
        spread = [e["gated_pct"] for e in holds]
        print(f"range across configurations where it holds: "
              f"{min(spread):+.2f}% to {max(spread):+.2f}%")

    save_json({"entries": entries}, args.out)
    print(f"\nablation -> {args.out}")


if __name__ == "__main__":
    main()
