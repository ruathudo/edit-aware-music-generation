"""
Summarize the headline runs produced by scripts/run_headline.sh.

    python scripts/headline_report.py results/refiner/headline

One row per corpus x draft source, all under the single setting chosen on D1
validation. Gains are (CC_B3 - CC) / CC_B3 per seed, then averaged; spread is
the sample sd of the per-seed percentages. Win/loss are per-melody against B3,
averaged over seeds, read from the saved cost arrays.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judge import save_json  # noqa: E402


def pct(b3, cc):
    return 100 * (b3 - cc) / b3


def summarize(path):
    blob = json.loads(path.read_text())
    config = blob["config"]
    (tau,) = config["gated_thresholds"]
    alpha = max(config["alphas"])
    b3 = blob["b3_copy_input"]["correction_cost_mean"]
    cost_dir = path.with_suffix("")
    b3_costs = np.load(cost_dir / "costs_b3.npy")

    rows = blob["rows"]
    seeds = sorted({r["seed"] for r in rows})
    b1 = {r["seed"]: r for r in rows if not r["use_edit_head"]}
    b2 = {r["seed"]: r for r in rows if r["use_edit_head"] and r["alpha"] == 0.0}
    ea = {r["seed"]: r for r in rows if r["use_edit_head"] and r["alpha"] == alpha}

    gated, win, loss, prec, rec = [], [], [], [], []
    for s in seeds:
        r = ea[s]
        gated.append(pct(b3, r["gated"][str(tau)]["correction_cost"]))
        costs = np.load(cost_dir / f"costs_{r['label']}_gated{tau}.npy")
        win.append(float(np.mean(costs < b3_costs)))
        loss.append(float(np.mean(costs > b3_costs)))
        prec.append(r["edit_scores"]["localization"]["precision"])
        rec.append(r["edit_scores"]["localization"]["recall"])

    def ms(xs):
        return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else float("nan"))

    return {
        "corpus": "Essen" if config.get("essen_test") else "MTC",
        "source": config["sources"][0],
        "setting": {"weighting": config["weighting"], "alpha": alpha, "tau": tau},
        "n": ea[seeds[0]]["n"],
        "b3_cc": b3,
        "b1_vs_b3_pct": ms([pct(b3, b1[s]["correction_cost"]) for s in seeds]),
        "b2_vs_b3_pct": ms([pct(b3, b2[s]["correction_cost"]) for s in seeds]),
        "free_vs_b3_pct": ms([pct(b3, ea[s]["correction_cost"]) for s in seeds]),
        "gated_vs_b3_pct": ms(gated),
        "gated_per_seed": gated,
        "seeds_positive": sum(g > 0 for g in gated),
        "win": statistics.mean(win),
        "loss": statistics.mean(loss),
        "loc_precision": ms(prec),
        "loc_recall": ms(rec),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()

    order = {"D3": 0, "D1": 1, "D2-infill": 2}
    rows = [summarize(p) for p in sorted(Path(args.directory).glob("*.json"))
            if p.name != "headline.json"]
    rows.sort(key=lambda r: (r["corpus"] != "MTC", order.get(r["source"], 9)))

    s = rows[0]["setting"]
    print(f"setting: weighting={s['weighting']} alpha={s['alpha']} tau={s['tau']}  "
          f"(selected on D1 validation)\n")
    head = (f"{'corpus':6s} {'source':10s} {'n':>5s} {'B3':>6s} {'B1':>8s} {'free':>8s} "
            f"{'gated':>15s} {'seeds+':>6s} {'win':>6s} {'loss':>6s} {'locP':>13s} {'locR':>6s}")
    print(head)
    print("-" * len(head))
    for r in rows:
        g, sd = r["gated_vs_b3_pct"]
        p, psd = r["loc_precision"]
        print(f"{r['corpus']:6s} {r['source']:10s} {r['n']:5d} {r['b3_cc']:6.3f} "
              f"{r['b1_vs_b3_pct'][0]:+7.1f}% {r['free_vs_b3_pct'][0]:+7.1f}% "
              f"{g:+7.2f}% ±{sd:4.2f} {r['seeds_positive']:>4d}/3 "
              f"{r['win']:6.1%} {r['loss']:6.1%} {p:6.3f} ±{psd:5.3f} {r['loc_recall'][0]:6.3f}")
        print(f"{'':18s} gated per seed: {', '.join(f'{x:+.2f}%' for x in r['gated_per_seed'])}")

    out = Path(args.directory) / "headline.json"
    save_json({"rows": rows}, out)
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
