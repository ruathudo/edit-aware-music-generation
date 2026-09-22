"""
Can generated material be found at the span level, where it cannot at the note level?

    python scripts/span_detection.py

The per-position edit head sits at ~1.5x chance on generator drafts and does not
improve on drafts the judge certifies as degraded, because the damage is global:
the melody is worse as a whole while no single note carries the signal. If that
diagnosis is right, scoring *regions* should work where scoring positions did
not.

Three systems, same candidates and same metric:
  - **chance**, 1/|candidates|, usually a few hundredths of a percent;
  - **judge surprisal**, zero-shot, ranking spans by mean per-token NLL -- no
    training and no edit labels ever seen, so it measures what the judge alone
    already knows once the signal is pooled;
  - **trained detector**, a bidirectional encoder scoring span interior plus both
    boundary states.

The comparison that matters is detector against surprisal. Beating chance only
shows the task is possible; beating the zero-shot judge shows the encoder learned
something the judge did not already supply.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.drafts import d2_infill, empirical_priors  # noqa: E402
from src.judge import Judge, save_json  # noqa: E402
from src.refinement import cached_drafts  # noqa: E402
from src.span_model import (  # noqa: E402
    SpanDetector,
    SpanModelConfig,
    enumerate_spans,
    surprisal_span_scores,
)
from scripts.diagnostics import load_generator  # noqa: E402

import random  # noqa: E402


class SpanDataset(Dataset):
    """Drafts with their true infilled span and all candidate spans."""

    def __init__(self, drafts, tokenizer, lengths=(2, 3, 4, 5, 6), max_len=1024):
        self.tokenizer = tokenizer
        self.items = []
        self.dropped = 0

        for draft in drafts:
            match = re.match(r"infill@(\d+)\+(\d+)", draft.injected[0] if draft.injected else "")
            if not match:
                self.dropped += 1
                continue
            start, length = int(match.group(1)), int(match.group(2))

            notes = draft.notes
            if not tokenizer.supports(notes):
                self.dropped += 1
                continue
            tokens, spans = tokenizer.encode_with_spans(notes, draft.reference.bar_starts)
            if len(tokens) > max_len or start + length > len(spans):
                self.dropped += 1
                continue

            candidates = enumerate_spans(len(notes), lengths=lengths)
            if (start, length) not in candidates:
                self.dropped += 1
                continue

            bounds = [(spans[s][0], spans[s + L - 1][1] + 1) for s, L in candidates]
            self.items.append({
                "tokens": tokens,
                "bounds": bounds,
                "target": candidates.index((start, length)),
                "true_span": (start, length),
                "candidates": candidates,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch, pad_id):
    width = max(len(b["tokens"]) for b in batch)
    n_candidates = max(len(b["bounds"]) for b in batch)

    tokens = torch.full((len(batch), width), pad_id, dtype=torch.long)
    bounds = torch.zeros(len(batch), n_candidates, 2, dtype=torch.long)
    mask = torch.zeros(len(batch), n_candidates, dtype=torch.bool)
    target = torch.tensor([b["target"] for b in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        tokens[i, : len(item["tokens"])] = torch.tensor(item["tokens"])
        for j, (s, e) in enumerate(item["bounds"]):
            bounds[i, j] = torch.tensor([s, e])
        mask[i, : len(item["bounds"])] = True

    return tokens, bounds, mask, target, batch


def iou(a, b):
    """Event-space overlap between predicted and true span."""
    (s1, l1), (s2, l2) = a, b
    lo, hi = max(s1, s2), min(s1 + l1, s2 + l2)
    inter = max(0, hi - lo)
    union = l1 + l2 - inter
    return inter / union if union else 0.0


@torch.no_grad()
def judge_surprisal(judge, dataset, pad_id, device, batch_size=32):
    """Per-token surprisal of each draft under the judge."""
    out = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, pad_id))
    for tokens, _, _, _, _ in loader:
        tokens = tokens.to(device)
        logits = judge.model(tokens[:, :-1])
        targets = tokens[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                             ignore_index=pad_id, reduction="none").reshape(targets.shape)
        out.append(torch.cat([torch.zeros(len(tokens), 1, device=device), ce], dim=1).cpu())
    width = max(t.size(1) for t in out)
    return torch.cat([F.pad(t, (0, width - t.size(1))) for t in out], dim=0)


def evaluate(scores, dataset, indices):
    """Top-1, top-5 and mean IoU against the true span."""
    top1 = top5 = 0
    ious = []
    for row, index in zip(scores, indices):
        item = dataset.items[index]
        order = torch.argsort(row, descending=True)
        top1 += int(order[0].item() == item["target"])
        top5 += int(item["target"] in order[:5].tolist())
        ious.append(iou(item["candidates"][int(order[0])], item["true_span"]))
    n = len(indices)
    return {"top1": top1 / n, "top5": top5 / n, "iou": float(np.mean(ious)), "n": n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--per-melody", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--generator", default="models/generator.pth")
    parser.add_argument("--judge", default="models/judge.pth")
    parser.add_argument("--out", default="results/refiner/span_detection.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = standard_splits()
    every = [m for g in splits.values() for m in g]
    tokenizer = build_tokenizer(every)
    generator = load_generator(args.generator, tokenizer, device)
    judge = Judge.load(args.judge)

    def drafts_for(name, seed):
        melodies = splits[name]
        return cached_drafts(
            melodies,
            lambda m=melodies, s=seed: [
                d for r in range(args.per_melody)
                for d in d2_infill(m, generator, tokenizer, random.Random(s + r),
                                   temperature=args.temperature, seed=s + r)
            ],
            {"sources": ["D2-infill"], "temperature": args.temperature,
             "per_melody": args.per_melody, "seed": seed, "tag": name},
        )

    datasets = [
        SpanDataset(drafts_for(name, seed), tokenizer)
        for name, seed in (("judge_train", 100), ("judge_val", 200), ("heldout", 300))
    ]
    train_ds, val_ds, test_ds = datasets
    print(f"span datasets: train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}")
    mean_candidates = float(np.mean([len(i["bounds"]) for i in test_ds.items]))
    chance = 1.0 / mean_candidates
    print(f"mean candidates per draft: {mean_candidates:.0f}  ->  chance top-1 = {chance:.4%}")

    # Zero-shot judge baseline, no training at all.
    surprisal = judge_surprisal(judge, test_ds, tokenizer.pad_id, device)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, tokenizer.pad_id))
    base_scores, offset, order = [], 0, []
    for tokens, bounds, mask, _, _ in loader:
        rows = surprisal[offset : offset + len(tokens), : tokens.size(1)]
        base_scores.append(surprisal_span_scores(rows, bounds, mask))
        order.extend(range(offset, offset + len(tokens)))
        offset += len(tokens)
    width = max(s.size(1) for s in base_scores)
    base = torch.cat([F.pad(s, (0, width - s.size(1)), value=float("-inf")) for s in base_scores])
    baseline = evaluate(base, test_ds, order)
    print(f"\njudge surprisal (zero-shot): top1 {baseline['top1']:.2%}  "
          f"top5 {baseline['top5']:.2%}  IoU {baseline['iou']:.3f}  "
          f"({baseline['top1']/chance:.1f}x chance)")

    results = {"chance": chance, "mean_candidates": mean_candidates,
               "surprisal_baseline": baseline, "detector": []}

    for seed in args.seeds:
        torch.manual_seed(seed)
        model = SpanDetector(
            SpanModelConfig(vocab_size=len(tokenizer), d_model=args.d_model,
                            n_layers=args.n_layers, n_heads=args.n_heads),
            pad_id=tokenizer.pad_id,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        make = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                         collate_fn=lambda b: collate(b, tokenizer.pad_id))
        train_loader, val_loader = make(train_ds, True), make(val_ds, False)

        print(f"\n=== detector seed {seed} | {model.n_parameters/1e6:.2f}M params ===")
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for tokens, bounds, mask, target, _ in train_loader:
                logits = model(tokens.to(device), bounds.to(device), mask.to(device))
                loss = F.cross_entropy(logits, target.to(device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            total = 0.0
            with torch.no_grad():
                for tokens, bounds, mask, target, _ in val_loader:
                    logits = model(tokens.to(device), bounds.to(device), mask.to(device))
                    total += float(F.cross_entropy(logits, target.to(device)))
            val = total / max(len(val_loader), 1)
            print(f"  epoch {epoch:3d}  val {val:.4f}")
            if best - val > 1e-4:
                best, stale = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"  early stop at epoch {epoch}")
                    break
        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        scores, idx, offset = [], [], 0
        test_loader = make(test_ds, False)
        with torch.no_grad():
            for tokens, bounds, mask, _, _ in test_loader:
                logits = model(tokens.to(device), bounds.to(device), mask.to(device)).cpu()
                scores.append(logits)
                idx.extend(range(offset, offset + len(tokens)))
                offset += len(tokens)
        width = max(s.size(1) for s in scores)
        allscores = torch.cat([F.pad(s, (0, width - s.size(1)), value=float("-inf")) for s in scores])
        metrics = evaluate(allscores, test_ds, idx)
        metrics["seed"] = seed
        results["detector"].append(metrics)
        print(f"  -> top1 {metrics['top1']:.2%}  top5 {metrics['top5']:.2%}  "
              f"IoU {metrics['iou']:.3f}  ({metrics['top1']/chance:.1f}x chance)")

    if results["detector"]:
        t1 = [m["top1"] for m in results["detector"]]
        print(f"\n{'system':26s} {'top-1':>9s} {'top-5':>9s} {'IoU':>7s} {'x chance':>9s}")
        print("-" * 64)
        print(f"{'chance':26s} {chance:9.2%} {'-':>9s} {'-':>7s} {'1.0x':>9s}")
        print(f"{'judge surprisal (0-shot)':26s} {baseline['top1']:9.2%} "
              f"{baseline['top5']:9.2%} {baseline['iou']:7.3f} {baseline['top1']/chance:8.1f}x")
        print(f"{'trained detector':26s} {np.mean(t1):9.2%} "
              f"{np.mean([m['top5'] for m in results['detector']]):9.2%} "
              f"{np.mean([m['iou'] for m in results['detector']]):7.3f} "
              f"{np.mean(t1)/chance:8.1f}x")
        print(f"\nper-position head, for reference: ~1.5x chance and flat across")
        print(f"degradation bands.")

    save_json(results, args.out)
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
