"""
Risk-coverage curves for the head gate: gain over copy-input as the threshold moves.

    python scripts/risk_coverage.py results/refiner/headline_r2 [--corpus MTC]

The paper frames gating as selective prediction but reports one operating point
(tau = 0.8, chosen on D1 validation). This traces the whole curve on test, per
source and corpus, from the headline run's saved checkpoints (3 seeds, decoder
fixed): at each tau, the gain in mean Correction Cost over copy-input, the
coverage (share of draft events the gate hands to the decoder) and the gate's
precision. tau = 1 is copy-input by construction and anchors every curve at 0.

The curve is descriptive, not a selection: the headline setting stays the one
chosen on validation. Its use is the D2 question -- whether *any* threshold
would have made editing generated drafts worthwhile, or whether abstention is
what every operating point converges to.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import sample_metrics  # noqa: E402
from src.judge import save_json  # noqa: E402
from scripts.judge_gate import SEEDS, build, gated_run, head_probs, load_refiner  # noqa: E402

THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
SWEEPS = ("d1", "d3", "d2infill")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--corpus", choices=["MTC", "Essen"], default=None)
    args = parser.parse_args()
    root = Path(args.directory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpora = [c for c in ("mtc", "essen") if not args.corpus or c == args.corpus.lower()]

    rows = []
    for corpus in corpora:
        for name in SWEEPS:
            sweep = root / f"{corpus}_{name}.json"
            config = json.loads(sweep.read_text())["config"]
            tokenizer, examples = build(config, "test", device)
            b3 = np.mean([sample_metrics(list(e.draft_notes), e.draft_notes, e.reference.notes)
                          ["correction_cost"] for e in examples])
            n_events = sum(len(e.draft_notes) for e in examples)
            models = {s: load_refiner(sweep.with_suffix("") / f"model_alpha0.5_seed{s}_binary.pt",
                                      tokenizer.pad_id, device) for s in SEEDS}
            probs = {s: head_probs(m, examples, tokenizer.pad_id, device) for s, m in models.items()}
            curve = []
            for thr in THRESHOLDS:
                gains, cover, prec = [], [], []
                for s in SEEDS:
                    costs, flags, hits = gated_run(models[s], tokenizer, examples, probs[s], thr, device)
                    gains.append(100 * (b3 - costs.mean()) / b3)
                    cover.append(flags / n_events)
                    prec.append(hits / flags if flags else float("nan"))
                point = {"threshold": thr, "gain_pct": statistics.mean(gains),
                         "gain_sd": statistics.stdev(gains), "per_seed": gains,
                         "seeds_positive": sum(g > 0 for g in gains),
                         "coverage": statistics.mean(cover),
                         "precision": float(np.nanmean(prec)) if any(p == p for p in prec) else None}
                curve.append(point)
                p = point["precision"]
                print(f"{corpus:5s} {config['sources'][0]:9s} tau {thr:4.2f}  gain "
                      f"{point['gain_pct']:+7.2f}% ±{point['gain_sd']:.2f} "
                      f"{point['seeds_positive']}/3  coverage {point['coverage']:.4f}  "
                      f"precision {'   -' if p is None else f'{p:.3f}'}", flush=True)
            curve.append({"threshold": 1.0, "gain_pct": 0.0, "gain_sd": 0.0, "coverage": 0.0,
                          "precision": None, "note": "copy-input, by construction"})
            rows.append({"corpus": "Essen" if corpus == "essen" else "MTC",
                         "source": config["sources"][0], "n": len(examples), "curve": curve})

    suffix = f"_{args.corpus.lower()}" if args.corpus else ""
    path = root / f"risk_coverage{suffix}.json"
    save_json({"rows": rows}, path)
    print(f"\nrisk-coverage -> {path}")


if __name__ == "__main__":
    main()
