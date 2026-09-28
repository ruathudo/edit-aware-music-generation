"""
How good is each generator? Held-out likelihood for the checkpoints behind Figure 1a.

    python scripts/generator_quality.py

The paper orders generator settings by training stage and temperature and calls
the ordering "generator quality". That ordering should be measured by something
other than the judge that also supplies Figure 1a's x-axis. This reports each
generator's own per-token NLL on

  generator_val   the held-out Essen melodies it was early-stopped on
  references      the test references its drafts are built from (MTC-ANN, or
                  the Essen test slice for the same-corpus generator)

at temperature 1.0 -- the model's likelihood, not a sampled property. Sampling
temperature does not change a model's likelihood, so the temperature conditions
share the trained generator's row.
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import MelodyTokenDataset, build_tokenizer, make_collate, standard_splits  # noqa: E402
from src.judge import save_json  # noqa: E402
from src.model import token_nll  # noqa: E402
from scripts.train_refiner import load_generator  # noqa: E402

SETUPS = {
    "MTC": (0.0, "heldout", "models/generator.pth", "models/generator_checkpoints"),
    "Essen": (0.2, "essen_test", "models/generator_ctrl.pth", "models/generator_ctrl_checkpoints"),
}


@torch.no_grad()
def nll(model, dataset, pad_id, device):
    total = count = 0.0
    for batch in DataLoader(dataset, batch_size=64, collate_fn=make_collate(pad_id)):
        t, n = token_nll(model, batch.to(device))
        total += float(t.sum()); count += float(n.sum())
    return total / count


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for corpus, (essen_test, refs_key, final, ckpt_dir) in SETUPS.items():
        splits = standard_splits(essen_test=essen_test)
        every = [m for group in splits.values() for m in group]
        tokenizer = build_tokenizer(every)
        val = MelodyTokenDataset(splits["generator_val"], tokenizer)
        refs = MelodyTokenDataset(splits[refs_key], tokenizer)
        paths = sorted(Path(ckpt_dir).glob("generator_epoch*.pth"),
                       key=lambda p: int(p.stem.split("epoch")[-1]))
        for label, path in [(f"ep{p.stem.split('epoch')[-1]}", p) for p in paths] + [("trained", final)]:
            model = load_generator(str(path), tokenizer, device)
            row = {"corpus": corpus, "generator": label, "path": str(path),
                   "val_nll": nll(model, val, tokenizer.pad_id, device),
                   "reference_nll": nll(model, refs, tokenizer.pad_id, device)}
            rows.append(row)
            print(f"{corpus:6s} {label:8s} val NLL {row['val_nll']:.3f}  "
                  f"test-reference NLL {row['reference_nll']:.3f}", flush=True)
    path = Path("results/judge/generator_quality.json")
    save_json({"rows": rows}, path)
    print(f"\ngenerator quality -> {path}")


if __name__ == "__main__":
    main()
