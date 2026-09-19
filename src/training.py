"""
Shared language-model training loop.

The judge `J` and the generator `G` are the same architecture trained on
disjoint halves of the corpus, so they share this loop entirely. Keeping one
implementation is what makes their only differences the ones the spec actually
requires -- seed, depth, and data -- rather than incidental drift between two
copies of a training routine.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.corpus import Melody
from src.data import MelodyTokenDataset, make_collate
from src.model import LMConfig, MelodyLM, token_nll
from src.tokenizer import MelodyTokenizer


def train_lm(
    train_melodies: list[Melody],
    val_melodies: list[Melody],
    tokenizer: MelodyTokenizer,
    config: LMConfig | None = None,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 3e-4,
    patience: int = 6,
    seed: int = 1234,
    device=None,
    log_every: int = 1,
    label: str = "model",
    checkpoint_epochs: tuple[int, ...] = (),
    checkpoint_dir: str | Path | None = None,
) -> tuple[MelodyLM, dict]:
    """Train a melody LM with early stopping on validation NLL.

    `checkpoint_epochs` saves intermediate weights. That exists for the
    generator: spec 3.2 suggests undertrained checkpoints as a way to control how
    flawed the drafts are, and an undertrained `G` is the cheapest source of
    plausible-but-worse continuations.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = config or LMConfig(vocab_size=len(tokenizer))
    model = MelodyLM(config, pad_id=tokenizer.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    train_ds = MelodyTokenDataset(train_melodies, tokenizer, config.max_len)
    val_ds = MelodyTokenDataset(val_melodies, tokenizer, config.max_len)
    collate = make_collate(tokenizer.pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    print(
        f"{label}: {model.n_parameters/1e6:.2f}M params | "
        f"train {len(train_ds)} (dropped {train_ds.dropped}) | val {len(val_ds)}"
    )

    history: dict = {"train": [], "val": [], "checkpoints": {}}
    best = float("inf")
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = count = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            total, n = token_nll(model, batch)
            loss = total.sum() / n.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += float(total.sum())
            count += float(n.sum())
        train_nll = running / max(count, 1)

        model.eval()
        running = count = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                total, n = token_nll(model, batch)
                running += float(total.sum())
                count += float(n.sum())
        val_nll = running / max(count, 1)

        history["train"].append(train_nll)
        history["val"].append(val_nll)
        if log_every and epoch % log_every == 0:
            print(f"  epoch {epoch:3d}  train {train_nll:.4f}  val {val_nll:.4f}")

        if epoch in checkpoint_epochs and checkpoint_dir:
            path = Path(checkpoint_dir) / f"{label}_epoch{epoch}.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "lm_config": config.to_dict(), "epoch": epoch},
                path,
            )
            history["checkpoints"][epoch] = {"path": str(path), "val_nll": val_nll}
            print(f"    checkpoint -> {path} (val {val_nll:.4f})")

        if best - val_nll > 1e-4:
            best, stale = val_nll, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch}; best val {best:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_nll"] = best
    return model, history
