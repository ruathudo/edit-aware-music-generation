"""
Table 2 of the paper: how each draft source sits under the judge.

    python scripts/plausibility_table.py

Computed on exactly the drafts Table 1 describes -- the MTC test drafts for
D1-D3 (three per reference) and the validator's D4 variant pairs -- all scored
by `J`, so every column comes from one set of drafts and one definition:

  surface      Correction Cost / reference length (mean)
  dNLL         per-token NLL(draft) - NLL(reference) (mean, median)
  per unit     median of dNLL / surface, per draft
  <= 0         share of drafts no less plausible than their reference
  > tau        share judged degraded, tau = 95th pct of the variants' dNLL
  rho          Spearman(surface, dNLL), 95% CI by resampling tune families
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judge import save_json  # noqa: E402
from scripts.decoupling import cluster_bootstrap, spearman  # noqa: E402
import scripts.decoupling as dec  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    per = dec.collect()
    tau = float(np.percentile(per["D4 human variants"][1], 95))
    rows = []
    for name, (surface, delta, cluster) in per.items():
        lo, hi = cluster_bootstrap(surface, delta, cluster, rng)
        rows.append({
            "source": name, "n": int(len(delta)),
            "surface_mean": float(surface.mean()),
            "delta_mean": float(delta.mean()), "delta_median": float(np.median(delta)),
            "per_unit_median": float(np.median(delta / np.maximum(surface, 1e-9))),
            "pct_not_worse": float((delta <= 0).mean() * 100),
            "pct_above_tau": float((delta > tau).mean() * 100),
            "rho": spearman(surface, delta), "rho_ci": [lo, hi],
        })
    print(f"tau = {tau:.4f}\n")
    print(f"{'source':20s} {'n':>5s} {'surface':>8s} {'dNLL':>7s} {'median':>7s} {'per unit':>9s} "
          f"{'<=0':>6s} {'>tau':>6s} {'rho':>7s} {'95% CI':>17s}")
    for r in rows:
        print(f"{r['source']:20s} {r['n']:5d} {r['surface_mean']:8.3f} {r['delta_mean']:+7.3f} "
              f"{r['delta_median']:+7.3f} {r['per_unit_median']:+9.3f} {r['pct_not_worse']:5.1f}% "
              f"{r['pct_above_tau']:5.1f}% {r['rho']:+7.3f} [{r['rho_ci'][0]:+.2f}, {r['rho_ci'][1]:+.2f}]")
    save_json({"tau": tau, "rows": rows}, "results/judge/plausibility_table.json")
    print("\ntable -> results/judge/plausibility_table.json")


if __name__ == "__main__":
    main()
