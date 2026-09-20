"""
How much of the oracle's headroom does the edit head lose to its own class weighting?

    python scripts/weighting_ablation.py --sources D1

The oracle diagnostic showed the decoder is not the bottleneck: given
ground-truth labels it removes 49% of the draft's error on D1 and 22% on
D2-infill, comfortably beating the copy-input baseline that nothing in the main
sweep could beat. Localization is the whole problem.

And the head's failure looks self-inflicted. Plain inverse-frequency weighting
pays it ~105x more for flagging an edit than for leaving a note alone, and it
duly flags nearly everything: INSERT_BEFORE recall 0.56 at precision 0.010,
KEEP recall 0.29 despite precision 0.97. The imbalance correction became a
worse imbalance.

This trains the probe under each weighting scheme and reports what the head can
actually achieve, which sets how much of the oracle ceiling is reachable.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.drafts import empirical_priors  # noqa: E402
from src.judge import save_json  # noqa: E402
from scripts.diagnostics import build_datasets, load_generator, run_probe  # noqa: E402


CONDITIONS = [
    ("none", False),
    ("sqrt", False),
    ("capped", False),
    ("inverse", False),   # the current default, for reference
    ("none", True),       # focal alone
    ("sqrt", True),       # focal plus a gentle re-weight
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="*", default=["D1"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--per-melody", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--generator", default="models/generator.pth")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = standard_splits()
    every = [m for g in splits.values() for m in g]
    tokenizer = build_tokenizer(every)
    priors = empirical_priors(every)
    generator = load_generator(args.generator, tokenizer, device)
    datasets = build_datasets(args, tokenizer, priors, generator)

    results = []
    for weighting, focal in CONDITIONS:
        scores = run_probe(args, tokenizer, datasets, device,
                           weighting=weighting, use_focal=focal, quiet=True)
        results.append({
            "weighting": weighting, "focal": focal,
            "macro_f1": scores["macro_f1"],
            "localization_f1": scores["localization"]["f1"],
            "accuracy": scores["accuracy"],
            "keep_recall": scores["per_class"]["KEEP"]["recall"],
            "keep_precision": scores["per_class"]["KEEP"]["precision"],
            "per_class": scores["per_class"],
        })

    header = (f"{'weighting':16s} {'macroF1':>8s} {'locF1':>7s} {'acc':>7s} "
              f"{'KEEP P/R':>14s}")
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        name = r["weighting"] + ("+focal" if r["focal"] else "")
        print(f"{name:16s} {r['macro_f1']:8.3f} {r['localization_f1']:7.3f} "
              f"{r['accuracy']:7.3f} {r['keep_precision']:6.3f}/{r['keep_recall']:<6.3f}")

    best = max(results, key=lambda r: r["macro_f1"])
    name = best["weighting"] + ("+focal" if best["focal"] else "")
    current = next(r for r in results if r["weighting"] == "inverse" and not r["focal"])
    print(f"\nbest: {name} at macro-F1 {best['macro_f1']:.3f} "
          f"(current default 'inverse' is {current['macro_f1']:.3f})")

    tag = "_".join(args.sources).replace("-", "")
    out = args.out or f"results/refiner/weighting_{tag}.json"
    save_json({"sources": args.sources, "results": results}, out)
    print(f"\nablation -> {out}")


if __name__ == "__main__":
    main()
