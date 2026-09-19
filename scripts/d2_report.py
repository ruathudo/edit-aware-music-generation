"""
Where do model-generated drafts sit, relative to random corruption and to real
musical variation?

    python scripts/d2_report.py

The validator established that D1 and D3 move a melody a short way *off* the
data manifold, costing ~100x more implausibility per unit of change than
human-authored variation does. D2 is the test of whether a generator's output
lands anywhere different: on-manifold like real variation, but worse than the
reference.

Three things can go wrong, and the sweep is designed to show which:
  - continuations too fluent -> ΔNLL ~ 0, nothing clears the filter, and D2 is
    producing valid alternatives rather than flawed drafts;
  - continuations too divergent -> surface change approaches D4's, the edit
    script stops being an "edit", and the refinement framing breaks down;
  - the useful middle -> modest surface change at a plausibility cost between
    D4's +0.04 and D1's +3.9 per unit.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import correction_cost  # noqa: E402
from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.drafts import d2_continuations, d2_infill, empirical_priors, generate  # noqa: E402
from src.judge import Judge, save_json  # noqa: E402
from src.model import LMConfig, MelodyLM  # noqa: E402
from scripts.train_judge import variant_pairs  # noqa: E402


def load_generator(path, tokenizer, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = MelodyLM(LMConfig(**blob["lm_config"]), pad_id=tokenizer.pad_id)
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()


def row(label, judge, pairs, tau):
    scores = judge.score_drafts(pairs)
    deltas = np.array([s.delta_nll for s in scores])
    cc = np.array([correction_cost(d, r.notes) for d, r in pairs])
    n = np.array([max(len(r.notes), 1) for _, r in pairs])
    surface = cc / n
    return {
        "source": label,
        "n": len(pairs),
        "delta_nll": float(deltas.mean()),
        "pct_not_degraded": float((deltas <= 0).mean() * 100),
        "surface_change": float(surface.mean()),
        "delta_nll_per_change": float(np.median(deltas / np.maximum(surface, 1e-9))),
        "retention": float(np.mean([judge.is_degraded(s, tau) for s in scores]) * 100),
        "delta_scale": float(np.mean([s.delta_scale_conformance or 0.0 for s in scores])),
        "notes_draft": float(np.mean([s.n_notes_draft for s in scores])),
        "notes_ref": float(np.mean([s.n_notes_reference for s in scores])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default="models/judge.pth")
    parser.add_argument("--generator", default="models/generator.pth")
    parser.add_argument("--checkpoint-dir", default="models/generator_checkpoints")
    parser.add_argument("--out", default="results/judge/d2_report.json")
    parser.add_argument("--temperatures", type=float, nargs="*", default=[0.8, 1.0, 1.2])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    judge = Judge.load(args.judge)
    splits = standard_splits()
    references = splits["heldout"]
    every = [m for g in splits.values() for m in g]
    tokenizer = build_tokenizer(every)
    priors = empirical_priors(every)

    pairs = variant_pairs(references, seed=args.seed)
    pair_scores = judge.score_drafts(pairs)
    tau = float(np.percentile([s.delta_nll for s in pair_scores], 95))
    print(f"references {len(references)} | tau {tau:.4f}\n")

    rows = [row("D4 human variants", judge, pairs, tau)]
    for source, label in (("D1", "D1 random"), ("D3", "D3 musical")):
        drafts = generate(references, source, priors, seed=args.seed, per_melody=1)
        rows.append(row(label, judge, [(d.notes, d.reference) for d in drafts], tau))

    # Fully trained G across temperatures, then undertrained checkpoints at T=1.0.
    conditions = [(args.generator, t, f"D2 T={t}") for t in args.temperatures]
    for path in sorted(Path(args.checkpoint_dir).glob("generator_epoch*.pth")):
        epoch = path.stem.split("epoch")[-1]
        conditions.append((str(path), 1.0, f"D2 ep{epoch} T=1.0"))

    for path, temperature, label in conditions:
        if not Path(path).exists():
            print(f"  (skipping {label}: {path} missing)")
            continue
        model = load_generator(path, tokenizer, device)
        drafts = d2_continuations(
            references, model, tokenizer, random.Random(args.seed),
            temperature=temperature, seed=args.seed,
        )
        if drafts:
            rows.append(row(label, judge, [(d.notes, d.reference) for d in drafts], tau))

    # Infilling: the variant that keeps the edit script the length a refinement
    # task implies, instead of regenerating everything after the prefix.
    for path, temperature, label in (
        [(args.generator, t, f"D2-infill T={t}") for t in args.temperatures]
        + [(str(p), 1.0, f"D2-infill ep{p.stem.split('epoch')[-1]}")
           for p in sorted(Path(args.checkpoint_dir).glob("generator_epoch*.pth"))]
    ):
        if not Path(path).exists():
            continue
        model = load_generator(path, tokenizer, device)
        drafts = d2_infill(
            references, model, tokenizer, random.Random(args.seed),
            temperature=temperature, seed=args.seed,
        )
        if drafts:
            rows.append(row(label, judge, [(d.notes, d.reference) for d in drafts], tau))

    header = (
        f"{'source':22s} {'n':>5s} {'ΔNLL':>8s} {'%≤0':>6s} {'change':>7s} "
        f"{'ΔNLL/chg':>9s} {'retain%':>8s} {'notes':>12s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['source']:22s} {r['n']:5d} {r['delta_nll']:+8.4f} {r['pct_not_degraded']:5.1f}% "
            f"{r['surface_change']:7.3f} {r['delta_nll_per_change']:+9.3f} {r['retention']:7.1f}% "
            f"{r['notes_draft']:5.1f}/{r['notes_ref']:<6.1f}"
        )

    save_json({"tau": tau, "rows": rows, "config": vars(args)}, args.out)
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
