"""
Threshold-free localization, compared across draft sources.

    python scripts/localization_report.py results/refiner/headline_r2

Under the globally selected setting the D2 head collapses to KEEP -- it flags a
dozen positions where D1's flags ~2,300 -- so argmax precision on D2 is computed
over almost nothing and cannot be compared with D1's. These measures do not
depend on where the head puts its threshold:

- **AP** (average precision, area under precision-recall) of P(edit) against
  the true edit positions, and **AP lift** = AP / base rate. A head with no
  information scores AP equal to the base rate, lift 1.0.
- **AUROC**, the probability a random edited position outranks a random kept one.
- **precision@k**, with k the true number of edits: precision if the head had to
  flag exactly as many positions as there are errors.

It also reports **gate precision** -- precision among the positions the gate
actually hands to the decoder, P(edit) >= tau -- which is the quantity gating
depends on, as opposed to the head's argmax precision.

Positions are token positions from the saved probability arrays (two per event,
sharing a label), so every measure is at the same granularity for every source.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judge import save_json  # noqa: E402


def average_precision(scores, truth):
    order = np.argsort(-scores, kind="stable")
    hits = truth[order].astype(np.float64)
    if hits.sum() == 0:
        return float("nan")
    precision_at = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision_at * hits).sum() / hits.sum())


def auroc(scores, truth):
    # Mann-Whitney U with average ranks for ties.
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = truth.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def measure(npz, tau):
    z = np.load(npz)
    p, y = z["p_edit"].astype(np.float64), z["is_edit"].astype(np.int64)
    base = float(y.mean())
    k = int(y.sum())
    top = np.argsort(-p, kind="stable")[:k]
    gate = p >= tau
    ap = average_precision(p, y)
    return {
        "base_rate": base,
        "ap": ap,
        "ap_lift": ap / base,
        "auroc": auroc(p, y),
        "precision_at_k": float(y[top].mean()),
        "gate_flags": int(gate.sum()),
        "gate_precision": float(y[gate].mean()) if gate.any() else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    root = Path(args.directory)

    order = {"D3": 0, "D1": 1, "D2-infill": 2}
    table = []
    for sweep in sorted(root.glob("*.json")):
        if sweep.name in ("headline.json", "localization.json"):
            continue
        blob = json.loads(sweep.read_text())
        config = blob["config"]
        (tau,) = config["gated_thresholds"]
        alpha = max(config["alphas"])
        per_seed = [
            measure(npz, tau)
            for npz in sorted(sweep.with_suffix("").glob(f"probs_alpha{alpha}_seed*_*.npz"))
        ]
        if not per_seed:
            continue
        row = {"corpus": "Essen" if config.get("essen_test") else "MTC",
               "source": config["sources"][0], "tau": tau, "n_seeds": len(per_seed),
               "per_seed": per_seed}
        for key in per_seed[0]:
            vals = [m[key] for m in per_seed]
            finite = [v for v in vals if v == v]
            row[key] = statistics.mean(finite) if finite else float("nan")
            row[key + "_sd"] = statistics.stdev(finite) if len(finite) > 1 else float("nan")
        table.append(row)
    table.sort(key=lambda r: (r["corpus"] != "MTC", order.get(r["source"], 9)))

    head = (f"{'corpus':6s} {'source':10s} {'base':>6s} {'AP':>13s} {'lift':>6s} "
            f"{'AUROC':>13s} {'P@k':>6s} {'gate flags':>11s} {'gate P':>7s}")
    print(head)
    print("-" * len(head))
    for r in table:
        print(f"{r['corpus']:6s} {r['source']:10s} {r['base_rate']:6.3f} "
              f"{r['ap']:6.3f} ±{r['ap_sd']:5.3f} {r['ap_lift']:5.1f}x "
              f"{r['auroc']:6.3f} ±{r['auroc_sd']:5.3f} {r['precision_at_k']:6.3f} "
              f"{r['gate_flags']:11.0f} {r['gate_precision']:7.3f}")

    out = root / "localization.json"
    save_json({"rows": table}, out)
    print(f"\nlocalization -> {out}")


if __name__ == "__main__":
    main()
