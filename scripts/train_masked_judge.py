"""
Train the two-sided judge on exactly the data `J` saw.

    python scripts/train_masked_judge.py                                   # main
    python scripts/train_masked_judge.py --essen-test 0.2 \\
        --out models/masked_judge_ctrl.pth                                 # same-corpus control

Same splits, split seed, size and optimizer settings as `scripts/train_judge.py`,
so the only difference from `J` is that context runs in both directions.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.masked_judge import (  # noqa: E402
    MaskedLMConfig, event_groups, save_masked_judge, train_masked_judge,
)


def encode(melodies, tokenizer, max_len=1024):
    """Same filter as `MelodyTokenDataset`, which `J` trained on: unsupported
    melodies and those longer than the positional table are dropped."""
    seqs = []
    for m in melodies:
        if not tokenizer.supports(m.notes):
            continue
        tokens, spans = tokenizer.encode_with_spans(m.notes, m.bar_starts)
        if len(tokens) > max_len:
            continue
        seqs.append((tokens, event_groups(tokens, spans, tokenizer.eos_id)))
    return seqs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--essen-test", type=float, default=0.0)
    parser.add_argument("--out", default="models/masked_judge.pth")
    args = parser.parse_args()

    splits = standard_splits(seed=args.split_seed, essen_test=args.essen_test)
    every = [m for group in splits.values() for m in group]
    tokenizer = build_tokenizer(every)
    train = encode(splits["judge_train"], tokenizer)
    val = encode(splits["judge_val"], tokenizer)
    print(f"vocabulary {len(tokenizer)}; melodies train {len(train)} val {len(val)}")

    model, history = train_masked_judge(
        train, val, tokenizer,
        MaskedLMConfig(vocab_size=len(tokenizer), d_model=args.d_model,
                       n_heads=args.n_heads, n_layers=args.n_layers),
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        rate=args.mask_rate, seed=args.seed,
    )
    save_masked_judge(model, args.out, {"args": vars(args), "history": history})
    Path(args.out).with_suffix(".history.json").write_text(json.dumps(history))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
