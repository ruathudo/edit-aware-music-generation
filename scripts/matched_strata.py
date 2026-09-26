"""
Localization at matched degradation, across draft sources.

    python scripts/matched_strata.py results/refiner/headline_r2

`control_strata.py` banded D2 alone, by D2's own median dNLL, and found the
two-sided judge's localization rising with degradation. That gradient cannot
distinguish two readings on its own:

  (a) generated drafts are hard to localize because most carry little damage,
      and damage of any kind is found once there is enough of it; or
  (b) generated damage is hard to localize at *any* level, because it is spread
      across the draft rather than concentrated in the changed notes.

A surprisal-based localizer finds high-dNLL drafts easier almost by
construction, so the gradient is expected under both. The discriminating test
is the same banding applied to corruption: here every source is cut at the
*same absolute* dNLL points, so D1, D2 and D3 are compared at matched damage.
Under (a) the sources converge within a band; under (b) D2 stays below D1 and
D3 in every band.

Drafts: MTC uses a fresh set (15 per melody, seed 301 -- the same draft seed
control_strata used for D2, so the D2 MTC rows reproduce its draft set); Essen
uses each sweep's own test drafts. Per band: head AUROC (3 seeds) and two-sided
judge AUROC / AP lift, with the band's mean fraction of notes changed so that
matched damage can be checked against unmatched amount of change.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import correction_cost  # noqa: E402
from src.judge import Judge, save_json  # noqa: E402
from src.masked_judge import load_masked_judge  # noqa: E402
from scripts.control_strata import drafts_for, show, stratify, tau_for  # noqa: E402

# Absolute dNLL cut points shared by every source; tau is appended per judge.
CUTS = (0.0, 0.1, 0.2, 0.4)
SWEEPS = ("d1", "d2infill", "d3")


def bands_for(deltas, tau):
    edges = [-np.inf, *CUTS, tau, np.inf]
    idx = np.arange(len(deltas))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        name = (f"dNLL <= {hi:g}" if lo == -np.inf else
                f"dNLL > tau={lo:.3f}" if hi == np.inf else f"{lo:g} < dNLL <= {hi:.3g}")
        out.append((name, idx[(deltas > lo) & (deltas <= hi)]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--extra-per-melody", type=int, default=15)
    parser.add_argument("--out", default="results/refiner/controls/matched_strata.json")
    args = parser.parse_args()
    root = Path(args.directory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = []

    for corpus in ("mtc", "essen"):
        judge = Judge.load("models/judge_ctrl.pth" if corpus == "essen" else "models/judge.pth", device)
        masked = load_masked_judge("models/masked_judge_ctrl.pth" if corpus == "essen"
                                   else "models/masked_judge.pth", device)
        tau = None
        for name in SWEEPS:
            sweep = root / f"{corpus}_{name}.json"
            config = json.loads(sweep.read_text())["config"]
            if corpus == "mtc":
                tokenizer, examples, splits = drafts_for(config, device, args.extra_per_melody,
                                                         seed=301, tag="strata")
            else:
                tokenizer, examples, splits = drafts_for(config, device)
            if tau is None:
                tau, _ = tau_for(judge, splits)
            pairs = [(list(e.draft_notes), e.reference) for e in examples]
            deltas = np.array([s.delta_nll for s in judge.score_drafts(pairs)])
            changed = np.array([correction_cost(d, r.notes) / len(r.notes) for d, r in pairs])
            bands = bands_for(deltas, tau)
            rows = stratify(examples, bands, sweep.with_suffix(""), tokenizer, masked, device)
            for row, (_, members) in zip(rows, bands):
                row["mean_fraction_changed"] = float(changed[members].mean()) if len(members) else None
            out.append({"corpus": corpus, "source": config["sources"][0], "n_drafts": len(examples),
                        "tau": tau, "bands": rows})
            show(f"{config['sources'][0]}, {corpus.upper()} ({len(examples)} drafts, tau {tau:.3f})", rows)
            for row in rows:
                if row.get("mean_fraction_changed") is not None:
                    print(f"    {row['band']:34s} fraction changed {row['mean_fraction_changed']:.3f}")

    save_json({"cuts": list(CUTS), "rows": out}, Path(args.out))
    print(f"\nmatched strata -> {args.out}")


if __name__ == "__main__":
    main()
