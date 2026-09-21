"""
Train the generator `G` used for D2 drafts.

    python scripts/train_generator.py

`G` trains on the half of Essen the judge never sees, so the degradation filter
cannot simply admit whatever `G` was taught to produce. Intermediate checkpoints
are saved: spec 3.2 suggests undertrained weights as a way to control how flawed
the continuations are, and severity is the whole variable of interest for D2.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.judge import save_json  # noqa: E402
from src.model import LMConfig  # noqa: E402
from src.training import train_lm  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--essen-test", type=float, default=0.0,
                        help="carve a same-corpus Essen reference set (control)")
    parser.add_argument("--out", default="models/generator.pth")
    parser.add_argument("--checkpoint-dir", default="models/generator_checkpoints")
    parser.add_argument("--checkpoint-epochs", type=int, nargs="*", default=[2, 5, 10])
    args = parser.parse_args()

    splits = standard_splits(seed=args.split_seed, essen_test=args.essen_test)
    every = [m for group in splits.values() for m in group]
    tokenizer = build_tokenizer(every)

    print("generator trains on the half of Essen the judge never sees")
    print(f"  train {len(splits['generator_train'])}  val {len(splits['generator_val'])}")

    # Deliberately shallower than the judge, and differently seeded: spec 4 asks
    # for independence in seed, depth, and data, so the filter is not just
    # scoring a sibling of itself.
    model, history = train_lm(
        splits["generator_train"],
        splits["generator_val"],
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
        label="generator",
        checkpoint_epochs=tuple(args.checkpoint_epochs),
        checkpoint_dir=args.checkpoint_dir,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "lm_config": model.config.to_dict(),
            "tokenizer_config": {
                "pitch_min": tokenizer.config.pitch_min,
                "pitch_max": tokenizer.config.pitch_max,
                "durations": list(tokenizer.config.durations),
                "ticks_per_bar": tokenizer.config.ticks_per_bar,
                "use_bar": tokenizer.config.use_bar,
            },
        },
        args.out,
    )
    print(f"saved generator -> {args.out}")
    save_json(history, "results/judge/generator_history.json")


if __name__ == "__main__":
    main()
