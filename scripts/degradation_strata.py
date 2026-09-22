"""
Is the D2 localization task ill-posed for most drafts?

    python scripts/degradation_strata.py

96.6% of D2-infill drafts fall below tau, the 95th percentile of human-authored
variant pairs, and 17% are no less plausible than the reference. By the judge's
own calibration most of them are within the range of variation that human
variants of the same tune exhibit -- so there may be nothing wrong with them.

That matters because the ground-truth edit labels say "these notes differ from
the reference", not "these notes are wrong". For an undegraded draft those are
different claims, and asking the head to localize is asking it to identify which
notes an arbitrary alternative happened to change, with no musical signal
marking them. Aggregate precision of 0.137 might then be near the ceiling for an
ill-posed task rather than a model failure.

Spec 3.2 called the validator filter "mandatory for D2, not optional" for exactly
this reason, and the earlier runs trained on unfiltered drafts.

This stratifies the test set by dNLL and reports localization precision within
each band. If precision climbs with degradation, the aggregate was diluted by
instances that were never localizable.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import KEEP  # noqa: E402
from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.drafts import empirical_priors  # noqa: E402
from src.edit_model import IGNORE_INDEX, binarize_labels  # noqa: E402
from src.evaluation import generate_corrections_gated, sample_metrics  # noqa: E402
from src.judge import Judge, save_json  # noqa: E402
from src.refinement import make_collate  # noqa: E402
from scripts.diagnostics import build_datasets, load_generator  # noqa: E402
from scripts.train_refiner import train_one  # noqa: E402


def draft_delta_nll(judge, dataset, batch_size=64):
    """Per-example dNLL, the judge's verdict on whether the draft is degraded."""
    pairs = [(e.draft_notes, e.reference) for e in dataset.examples]
    scores = judge.score_drafts(pairs, batch_size=batch_size)
    return np.array([s.delta_nll for s in scores])


@torch.no_grad()
def per_example_localization(model, dataset, tokenizer, device, binary, threshold, batch_size=32):
    """Localization TP/FP/FN and gated Correction Cost, one row per draft."""
    collate = make_collate(tokenizer.pad_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    rows = []
    for source, labels, _, _, batch in loader:
        source = source.to(device)
        predictions = model.predict_edits(source).cpu()
        truth = binarize_labels(labels) if binary else labels
        gated = generate_corrections_gated(model, tokenizer, source, threshold=threshold)

        for row, label_row, output, example in zip(predictions, truth, gated, batch):
            keep = label_row != IGNORE_INDEX
            pred = (row[keep] != KEEP).numpy()
            actual = (label_row[keep] != KEEP).numpy()
            metrics = sample_metrics(output, example.draft_notes, example.reference.notes)
            rows.append({
                "tp": int((pred & actual).sum()),
                "fp": int((pred & ~actual).sum()),
                "fn": int((~pred & actual).sum()),
                "gated_cc": metrics["correction_cost"],
                "draft_cc": metrics["draft_cost"],
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="*", default=["D2-infill"])
    parser.add_argument("--weighting", default="capped")
    parser.add_argument("--head-type", default="binary")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--per-melody", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--surprisal", action="store_true")
    parser.add_argument("--generator", default="models/generator.pth")
    parser.add_argument("--judge", default="models/judge.pth")
    parser.add_argument("--tau", type=float, default=0.585)
    parser.add_argument("--out", default="results/refiner/degradation_strata.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = standard_splits()
    every = [m for g in splits.values() for m in g]
    tokenizer = build_tokenizer(every)
    priors = empirical_priors(every)
    generator = load_generator(args.generator, tokenizer, device)
    judge = Judge.load(args.judge)

    datasets = build_datasets(args, tokenizer, priors, generator)
    test = datasets[2]
    print(f"examples: train {len(datasets[0])} val {len(datasets[1])} test {len(test)}")

    deltas = draft_delta_nll(judge, test)
    print(f"test dNLL: mean {deltas.mean():+.4f}  "
          f"<=0 {np.mean(deltas <= 0):.1%}  >tau({args.tau}) {np.mean(deltas > args.tau):.1%}")

    model, _, label = train_one(args, args.alpha, args.seed, True, splits, tokenizer, datasets, device)
    rows = per_example_localization(
        model, test, tokenizer, device, args.head_type == "binary", args.threshold
    )

    def band(mask, name):
        sel = [r for r, m in zip(rows, mask) if m]
        if not sel:
            return None
        tp = sum(r["tp"] for r in sel); fp = sum(r["fp"] for r in sel); fn = sum(r["fn"] for r in sel)
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        lift = np.mean([(r["draft_cc"] - r["gated_cc"]) for r in sel])
        return {"band": name, "n": len(sel), "precision": precision, "recall": recall,
                "mean_delta_nll": float(np.mean([d for d, m in zip(deltas, mask) if m])),
                "gated_minus_draft_cc": float(lift)}

    quartiles = np.quantile(deltas, [0.25, 0.5, 0.75])
    bands = [
        band(deltas <= 0, "not degraded (dNLL <= 0)"),
        band((deltas > 0) & (deltas <= quartiles[1]), "mild (0 < dNLL <= median)"),
        band((deltas > quartiles[1]) & (deltas <= args.tau), "moderate (median < dNLL <= tau)"),
        band(deltas > args.tau, f"degraded (dNLL > tau={args.tau})"),
    ]

    print(f"\n{'band':32s} {'n':>6s} {'dNLL':>8s} {'locP':>7s} {'locR':>7s} {'gated-draft CC':>15s}")
    print("-" * 80)
    for b in bands:
        if b:
            print(f"{b['band']:32s} {b['n']:6d} {b['mean_delta_nll']:+8.3f} "
                  f"{b['precision']:7.3f} {b['recall']:7.3f} {b['gated_minus_draft_cc']:+15.3f}")

    save_json({"bands": [b for b in bands if b], "tau": args.tau,
               "config": vars(args)}, args.out)
    print(f"\nstrata -> {args.out}")


if __name__ == "__main__":
    main()
