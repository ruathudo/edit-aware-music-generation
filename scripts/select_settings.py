"""
Choose alpha, tau and the class weighting on the validation split, once, globally.

    python scripts/select_settings.py results/refiner/select/d1_w*_s*.json

Every headline number before this existed inherited settings chosen after seeing
test results: `weighting_ablation.py` ranked weightings by test macro-F1, and
alpha=0.5 / tau=0.9 were fixed after reading test tables. This script is the
replacement. Its inputs are sweeps run with `--eval-splits val`, so test drafts
are never decoded while choosing, and it refuses any file that carries test
results.

The criterion is mean validation gated Correction Cost across seeds -- the
quantity the method is actually trying to lower -- jointly over all three
settings. Selection is on D1 only and applied to every draft source unchanged:
the paper's claim is about how sources differ, so the method must not be
re-tuned per source.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judge import save_json  # noqa: E402


def load(paths):
    """(weighting, alpha, tau) -> {seed: gated CC}, plus val B3 and loc scores."""
    gated = defaultdict(dict)
    precision = defaultdict(dict)
    b3 = None
    for path in paths:
        blob = json.loads(Path(path).read_text())
        config = blob["config"]
        if blob.get("primary_split") != "val" or "test" in config.get("eval_splits", []):
            raise SystemExit(f"{path}: carries test results; selection must be val-only")
        b3_here = blob["b3_by_split"]["val"]["correction_cost_mean"]
        if b3 is not None and abs(b3 - b3_here) > 1e-9:
            raise SystemExit(f"{path}: validation B3 differs ({b3_here} vs {b3}); "
                             "sweeps were not run on the same drafts")
        b3 = b3_here
        for row in blob["rows"]:
            if not row["use_edit_head"]:
                continue
            key = (config["weighting"], row["alpha"])
            precision[key][row["seed"]] = row["edit_scores"]["localization"]["precision"]
            for tau, block in row.get("gated", {}).items():
                gated[key + (float(tau),)][row["seed"]] = block["correction_cost"]
    return gated, precision, b3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweeps", nargs="+")
    parser.add_argument("--out", default="results/refiner/select/selection.json")
    args = parser.parse_args()

    gated, precision, b3 = load(args.sweeps)
    seed_counts = {len(v) for v in gated.values()}
    if len(seed_counts) != 1:
        raise SystemExit(f"unequal seed coverage across settings: {sorted(seed_counts)}")

    table = []
    for (weighting, alpha, tau), by_seed in gated.items():
        gains = [100 * (b3 - cc) / b3 for cc in by_seed.values()]
        table.append({
            "weighting": weighting, "alpha": alpha, "tau": tau,
            "val_gated_cc": statistics.mean(by_seed.values()),
            "val_gain_pct": statistics.mean(gains),
            "val_gain_sd_pp": statistics.stdev(gains) if len(gains) > 1 else float("nan"),
            "seeds_positive": sum(g > 0 for g in gains),
            "n_seeds": len(gains),
            "val_loc_precision": statistics.mean(precision[(weighting, alpha)].values()),
        })
    table.sort(key=lambda r: r["val_gated_cc"])
    best = table[0]

    print(f"validation B3 CC {b3:.4f}   settings {len(table)}   seeds {best['n_seeds']}\n")
    print(f"{'weighting':9s} {'alpha':>5s} {'tau':>5s} {'val CC':>8s} {'gain':>8s} "
          f"{'sd pp':>6s} {'seeds+':>6s} {'locP':>6s}")
    for r in table[:15]:
        print(f"{r['weighting']:9s} {r['alpha']:5.1f} {r['tau']:5.2f} {r['val_gated_cc']:8.4f} "
              f"{r['val_gain_pct']:+7.2f}% {r['val_gain_sd_pp']:6.2f} "
              f"{r['seeds_positive']:>3d}/{r['n_seeds']:<2d} {r['val_loc_precision']:6.3f}")

    # How decisive is the choice? Settings within one seed-sd of the winner are
    # statistically indistinguishable from it on this evidence.
    tied = [r for r in table if best["val_gain_pct"] - r["val_gain_pct"] <= best["val_gain_sd_pp"]]
    print(f"\nselected: weighting={best['weighting']} alpha={best['alpha']} tau={best['tau']} "
          f"(val gain {best['val_gain_pct']:+.2f}%)")
    print(f"{len(tied)} setting(s) within one seed-sd of it")

    # Previous test-selected setting, for the record.
    old = next((r for r in table if r["weighting"] == "none" and r["alpha"] == 0.5
                and r["tau"] == 0.9), None)
    if old:
        print(f"previous test-selected setting (none, 0.5, 0.9): val gain "
              f"{old['val_gain_pct']:+.2f}%, rank {table.index(old) + 1}/{len(table)}")

    save_json({"selected": best, "val_b3_cc": b3, "tied_within_one_sd": tied,
               "table": table, "sweeps": args.sweeps}, args.out)
    print(f"\nselection -> {args.out}")


if __name__ == "__main__":
    main()
