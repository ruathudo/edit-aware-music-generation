"""
Aggregate a multi-seed sweep: mean +/- std per condition (spec 6.2).

    python scripts/seed_aggregate.py results/refiner/seeds_d2.json

Single-run numbers are not trustworthy here, and this repo has a concrete
demonstration of why: running the identical configuration twice (same seed) moved
the alpha=0 Correction Cost by 0.133, while the effect being claimed was 0.259.
Run-to-run nondeterminism was half the effect size.

Two rules this script exists to enforce:

  - **Never take a max across seeds.** Reporting the best seed per condition is
    cherry-picking, and with an effect this close to the noise floor it would
    manufacture a result. Everything here is a mean over seeds with its spread.
  - **Compare seed-matched.** The edit-aware model from seed 0 is compared with
    the baseline from seed 0, and the per-seed effects are then summarized. A
    within-run paired test says whether one *trained model* beats another; it
    says nothing about whether the training procedure reliably produces better
    models, which is the actual claim.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import paired_comparison  # noqa: E402
from src.judge import save_json  # noqa: E402


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1) if len(values) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = Path(args.sweep)
    sweep = json.loads(path.read_text())
    directory = path.with_suffix("")
    rows = sweep["rows"]
    b3_mean = sweep["b3_copy_input"]["correction_cost_mean"]
    b3_costs = np.load(directory / "costs_b3.npy")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["alpha"], r["use_edit_head"])].append(r)

    sources = ",".join(sweep["config"].get("sources", ["?"]))
    seeds = sorted({r["seed"] for r in rows})
    print(f"sweep: {path.name}   drafts: {sources}   seeds: {seeds}")
    print(f"B3 copy-input CC {b3_mean:.3f}\n")

    header = (f"{'condition':18s} {'CC mean±std':>16s} {'lift':>9s} "
              f"{'fidelity':>9s} {'macroF1':>13s}")
    print(header)
    print("-" * len(header))

    summary = []
    for (alpha, use_head) in sorted(groups, key=lambda k: (not k[1], k[0])):
        group = groups[(alpha, use_head)]
        cc_m, cc_s = mean_std([r["correction_cost"] for r in group])
        lift_m, lift_s = mean_std(
            [(r["draft_cost"] - r["correction_cost"]) / r["draft_cost"] for r in group]
        )
        fid_m, _ = mean_std([r["fidelity_per_note"] for r in group])
        f1_m, f1_s = mean_std(
            [r.get("edit_scores", {}).get("macro_f1") for r in group]
        )

        name = f"alpha={alpha}" + ("" if use_head else " (B1)")
        print(f"{name:18s} {cc_m:8.3f} ±{cc_s:6.3f} {lift_m:+8.2%} "
              f"{fid_m:9.4f} {f1_m:7.3f} ±{f1_s:5.3f}")
        summary.append({
            "alpha": alpha, "use_edit_head": use_head, "n_seeds": len(group),
            "cc_mean": cc_m, "cc_std": cc_s,
            "lift_mean": lift_m, "lift_std": lift_s,
            "fidelity_mean": fid_m, "macro_f1_mean": f1_m, "macro_f1_std": f1_s,
        })

    # Seed-matched effects against the same seed's B1.
    b1_by_seed = {
        r["seed"]: r["label"] for r in rows if r["alpha"] == 0.0 and not r["use_edit_head"]
    }
    if b1_by_seed:
        print(f"\nseed-matched effect vs B1 (positive = edit-aware better)")
        print(f"{'condition':18s} {'delta CC mean±std':>20s} {'per-seed deltas':>28s} {'pooled p':>10s}")
        print("-" * 80)

        for (alpha, use_head) in sorted(groups, key=lambda k: k[0]):
            if not use_head:
                continue
            per_seed, pooled_a, pooled_b = [], [], []
            for r in groups[(alpha, use_head)]:
                b1_label = b1_by_seed.get(r["seed"])
                if not b1_label:
                    continue
                ea = np.load(directory / f"costs_{r['label']}.npy")
                b1 = np.load(directory / f"costs_{b1_label}.npy")
                per_seed.append(float(b1.mean() - ea.mean()))
                pooled_a.append(b1)
                pooled_b.append(ea)

            if not per_seed:
                continue
            m, s = mean_std(per_seed)
            pooled = paired_comparison(np.concatenate(pooled_a), np.concatenate(pooled_b),
                                       n_boot=2000)
            p = pooled["wilcoxon"]["p_value"]
            deltas = " ".join(f"{d:+.3f}" for d in per_seed)
            print(f"{'alpha=' + str(alpha):18s} {m:+11.3f} ±{s:6.3f} {deltas:>28s} "
                  f"{(f'{p:.2e}' if p is not None else '-'):>10s}")

            for entry in summary:
                if entry["alpha"] == alpha and entry["use_edit_head"]:
                    entry["vs_b1_delta_mean"] = m
                    entry["vs_b1_delta_std"] = s
                    entry["vs_b1_per_seed"] = per_seed
                    entry["vs_b1_pooled_p"] = p

        print("\nRead the spread, not just the mean: if the per-seed deltas straddle")
        print("zero, the effect is not reliably reproduced by the training procedure,")
        print("whatever a pooled within-run p-value says.")

    # Against B3, which is deterministic and needs no seed.
    print(f"\nvs B3 copy-input (positive = model better than doing nothing)")
    print(f"{'condition':18s} {'delta CC mean±std':>20s} {'beats B3':>10s}")
    print("-" * 52)
    for (alpha, use_head) in sorted(groups, key=lambda k: (not k[1], k[0])):
        deltas, wins = [], []
        for r in groups[(alpha, use_head)]:
            costs = np.load(directory / f"costs_{r['label']}.npy")
            deltas.append(float(b3_costs.mean() - costs.mean()))
            wins.append(float((b3_costs > costs).mean() * 100))
        m, s = mean_std(deltas)
        w, _ = mean_std(wins)
        name = f"alpha={alpha}" + ("" if use_head else " (B1)")
        print(f"{name:18s} {m:+11.3f} ±{s:6.3f} {w:9.1f}%")

    out = args.out or str(path.with_name(path.stem + "_aggregated.json"))
    save_json({"source": str(path), "b3": b3_mean, "conditions": summary}, out)
    print(f"\naggregated -> {out}")


if __name__ == "__main__":
    main()
