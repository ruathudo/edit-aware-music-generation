"""
Is localizability driven by manifold distance, or by whether damage is point-like?

    python scripts/locality_strata.py --sources D3

Two accounts fit the D1/D2 contrast equally well:

  **H1, manifold distance.** D1 sits at 3.94 ΔNLL per unit of change and is
  localizable (precision 0.68–0.73); D2 sits at 0.9 and is not (1.5× chance).
  Off-manifold damage leaves a detectable trace.

  **H2, edit locality.** D1's edits are point-like — 95% of edited-position runs
  are a single note — while D2 replaces a contiguous span of 2–6 events. A
  per-position head may simply be suited to isolated changes and blind to
  contiguous ones, whatever the manifold distance.

Comparing sources cannot separate these, because D1 and D2 differ on both axes at
once. D3 can: its drafts are uniformly off-manifold (2.97 ΔNLL per unit) yet
mixed in locality — 73.7% of runs are single notes and 14.8% run to three or
more. Splitting D3's own test set by locality therefore holds manifold distance
roughly fixed and varies only the thing H2 cares about.

If precision is high on point-like drafts and collapses on span-like ones, the
governing variable is locality and the boundary should be described that way. If
it is flat, manifold distance is the better account.
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
from src.judge import save_json  # noqa: E402
from src.refinement import make_collate  # noqa: E402
from scripts.diagnostics import build_datasets, load_generator  # noqa: E402
from scripts.train_refiner import train_one  # noqa: E402


def longest_run(example, tokenizer) -> int:
    """Longest contiguous stretch of edited *events* in a draft.

    Measured in event space, not token space. Each event contributes two tokens
    carrying the same label, so counting token positions makes a single-note edit
    look like a run of two -- which it is not, and which would put every draft in
    the "span-like" band.

    Computed from the labels rather than the generating rule, so it reflects the
    supervision the head actually receives.
    """
    _, spans = tokenizer.encode_with_spans(example.draft_notes)
    flagged = sorted({
        event
        for event, (pitch_at, _) in enumerate(spans)
        if pitch_at < len(example.source_labels)
        and example.source_labels[pitch_at] not in (IGNORE_INDEX, KEEP)
    })
    if not flagged:
        return 0
    best = run = 1
    for a, b in zip(flagged, flagged[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return best


@torch.no_grad()
def per_example(model, dataset, tokenizer, device, binary, threshold, batch_size=32):
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
                "gain": metrics["draft_cost"] - metrics["correction_cost"],
                "run": longest_run(example, tokenizer),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="*", default=["D3"])
    parser.add_argument("--weighting", default="none")
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
    parser.add_argument("--out", default="results/refiner/locality_strata.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = standard_splits()
    every = [m for g in splits.values() for m in g]
    tokenizer = build_tokenizer(every)
    priors = empirical_priors(every)
    generator = load_generator(args.generator, tokenizer, device)

    datasets = build_datasets(args, tokenizer, priors, generator)
    print(f"examples: train {len(datasets[0])} val {len(datasets[1])} test {len(datasets[2])}")

    model, _, _ = train_one(args, args.alpha, args.seed, True, splits, tokenizer, datasets, device)
    rows = per_example(model, datasets[2], tokenizer, device,
                       args.head_type == "binary", args.threshold)

    runs = np.array([r["run"] for r in rows])
    print(f"\nlongest edited run per draft: mean {runs.mean():.2f}  "
          f"point-like (=1) {np.mean(runs == 1):.1%}  span-like (>=3) {np.mean(runs >= 3):.1%}")

    def band(mask, name):
        sel = [r for r, m in zip(rows, mask) if m]
        if not sel:
            return None
        tp = sum(r["tp"] for r in sel); fp = sum(r["fp"] for r in sel); fn = sum(r["fn"] for r in sel)
        return {
            "band": name, "n": len(sel),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "gated_gain": float(np.mean([r["gain"] for r in sel])),
        }

    bands = [b for b in (
        band(runs == 1, "point-like (1 event)"),
        band(runs == 2, "2 adjacent events"),
        band(runs >= 3, "span-like (3+ events)"),
    ) if b]

    print(f"\n{'band':26s} {'n':>6s} {'locP':>7s} {'locR':>7s} {'gated gain':>11s}")
    print("-" * 62)
    for b in bands:
        print(f"{b['band']:26s} {b['n']:6d} {b['precision']:7.3f} {b['recall']:7.3f} "
              f"{b['gated_gain']:+11.3f}")

    if len(bands) >= 2:
        lo, hi = bands[0]["precision"], bands[-1]["precision"]
        print(f"\npoint-like minus span-like precision: {lo - hi:+.3f}")
        print("A large positive gap favours H2 (locality governs); a flat profile")
        print("favours H1 (manifold distance governs).")

    save_json({"bands": bands, "sources": args.sources, "config": vars(args)}, args.out)
    print(f"\nstrata -> {args.out}")


if __name__ == "__main__":
    main()
