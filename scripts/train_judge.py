"""
Train the degradation judge `J` and run its calibration controls.

    python scripts/train_judge.py

`J` trains on half of Essen; the other half is reserved for the generator `G`,
and MTC-ANN is held out whole as the validator's reference set. See
`src.data.standard_splits` for why.

The calibration step is not optional decoration. `ΔNLL` measures deviation from
the corpus distribution, not musical badness, so before it is allowed to gate any
training data it has to be shown that human-authored variants -- genuinely valid
alternatives -- score near zero, while the noise floor over held-out real
melodies is known. Those two numbers are what make every later retention figure
interpretable.
"""
import argparse
import itertools
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.judge import Judge, calibration_report, save_json, train_judge  # noqa: E402
from src.model import LMConfig  # noqa: E402


def variant_pairs(melodies, max_per_family=40, seed=0):
    """Ordered pairs of melodies from the same tune family.

    These are human-authored variants of one tune, so they are the closest thing
    to a "valid alternative" control that exists without recruiting anyone.
    """
    families = defaultdict(list)
    for melody in melodies:
        if melody.tunefamily:
            families[melody.tunefamily].append(melody)

    rng = random.Random(seed)
    pairs = []
    for members in families.values():
        if len(members) < 2:
            continue
        combos = list(itertools.combinations(members, 2))
        rng.shuffle(combos)
        for a, b in combos[:max_per_family]:
            pairs.append((a.notes, b))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="models/judge.pth")
    parser.add_argument("--report", type=str, default="results/judge/calibration.json")
    args = parser.parse_args()

    splits = standard_splits(seed=args.split_seed)
    print("splits: " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # The tokenizer is fitted to every split at once so a held-out melody can
    # never contain a pitch or duration the vocabulary lacks.
    every = [m for group in splits.values() for m in group]
    tokenizer = build_tokenizer(every)
    print(f"vocabulary: {len(tokenizer)} tokens")

    judge, history = train_judge(
        splits["judge_train"],
        splits["judge_val"],
        tokenizer,
        config=LMConfig(
            vocab_size=len(tokenizer),
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    judge.save(args.out)
    print(f"saved judge -> {args.out}")

    heldout = splits["heldout"]
    pairs = variant_pairs(heldout, seed=args.seed)
    print(f"\ncalibration: {len(heldout)} held-out melodies, {len(pairs)} variant pairs")

    report = calibration_report(judge, heldout, pairs)
    report["history"] = history
    report["config"] = vars(args)
    save_json(report, args.report)

    print(f"\nnoise floor over held-out real melodies (per-token NLL):")
    print(f"  mean {report['nll_mean']:.4f}  std {report['nll_std']:.4f}"
          f"  p05 {report['nll_p05']:.4f}  p95 {report['nll_p95']:.4f}")

    if "variant_pairs" in report:
        vp = report["variant_pairs"]
        print(f"\nhuman-authored variant pairs (the 'valid alternative' anchor):")
        print(f"  n {vp['n']}  delta_nll mean {vp['delta_nll_mean']:+.4f}"
              f"  median {vp['delta_nll_median']:+.4f}  std {vp['delta_nll_std']:.4f}")
        print(f"  fraction scored as worse than their pair: {vp['fraction_positive']:.1%}")

    print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()
