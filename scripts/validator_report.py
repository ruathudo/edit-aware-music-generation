"""
The validator report (spec 4) -- the rebuild's falsification test.

    python scripts/validator_report.py

Scores every draft source with the judge and reports, per source, what fraction
of "corrupted" drafts are not measurably degraded. The expectation motivating the
whole redo is that D1 (uniform random perturbation, the published scheme)
frequently fails to degrade a melody, while D3 (musically grounded) reliably
does. If that expectation is wrong, the premise weakens and the plan changes --
which is exactly why this runs before any edit-aware model is trained.

Human-authored variant pairs are carried through the same table as a calibration
row. They are genuinely valid alternatives, so they *should* look undegraded; a
source that cannot be told apart from them is not producing errors.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus import duration_vocabulary  # noqa: E402
from src.data import standard_splits  # noqa: E402
from src.drafts import generate  # noqa: E402
from src.judge import Judge, save_json  # noqa: E402
from scripts.train_judge import variant_pairs  # noqa: E402


def summarize(name, scores, tau, judge):
    deltas = np.array([s.delta_nll for s in scores])
    scale = np.array([s.delta_scale_conformance or 0.0 for s in scores])
    metric = np.array([s.delta_metric_alignment or 0.0 for s in scores])
    retained = np.array([judge.is_degraded(s, tau) for s in scores])

    return {
        "source": name,
        "n": len(scores),
        "delta_nll_mean": float(deltas.mean()),
        "delta_nll_median": float(np.median(deltas)),
        "delta_nll_std": float(deltas.std()),
        "pct_not_degraded": float((deltas <= 0).mean() * 100),
        "pct_below_tau": float((deltas <= tau).mean() * 100),
        "delta_scale_conformance": float(scale.mean()),
        "delta_metric_alignment": float(metric.mean()),
        "pitch_class_distance": float(np.mean([s.pitch_class_distance for s in scores])),
        "interval_distance": float(np.mean([s.interval_distance for s in scores])),
        "retention": float(retained.mean() * 100),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default="models/judge.pth")
    parser.add_argument("--out", default="results/judge/validator_report.json")
    parser.add_argument("--max-edits", type=int, default=5)
    parser.add_argument("--per-melody", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="degradation threshold; defaults to the variant-pair 95th percentile",
    )
    args = parser.parse_args()

    judge = Judge.load(args.judge)
    splits = standard_splits()
    references = splits["heldout"]
    durations = duration_vocabulary([m for g in splits.values() for m in g])

    print(f"references: {len(references)} held-out MTC-ANN melodies")

    # Calibration first: tau is derived from what valid alternatives score, not
    # picked by hand. Anything a human variant would clear is not a degradation.
    pairs = variant_pairs(references, seed=args.seed)
    pair_scores = judge.score_drafts(pairs)
    pair_deltas = np.array([s.delta_nll for s in pair_scores])
    tau = args.tau if args.tau is not None else float(np.percentile(pair_deltas, 95))
    print(f"tau = {tau:.4f} (95th percentile of {len(pairs)} human variant pairs)\n")

    rows = [summarize("D4 human variants", pair_scores, tau, judge)]

    for source in ("D1", "D3"):
        drafts = generate(
            references,
            source,
            durations,
            seed=args.seed,
            max_edits=args.max_edits,
            per_melody=args.per_melody,
        )
        scores = judge.score_drafts([(d.notes, d.reference) for d in drafts])
        row = summarize(
            {"D1": "D1 random", "D3": "D3 musical"}[source], scores, tau, judge
        )
        row["edits_per_draft"] = float(np.mean([d.n_edits for d in drafts]))
        rows.append(row)

    header = f"{'source':20s} {'n':>6s} {'ΔNLL':>8s} {'med':>8s} {'%ΔNLL≤0':>9s} {'Δscale':>8s} {'Δmetric':>8s} {'retain%':>8s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['source']:20s} {r['n']:6d} {r['delta_nll_mean']:+8.4f} "
            f"{r['delta_nll_median']:+8.4f} {r['pct_not_degraded']:8.1f}% "
            f"{r['delta_scale_conformance']:+8.4f} {r['delta_metric_alignment']:+8.4f} "
            f"{r['retention']:7.1f}%"
        )

    save_json({"tau": tau, "rows": rows, "config": vars(args)}, args.out)
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
