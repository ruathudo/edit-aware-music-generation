"""
Two diagnostics that locate where the refinement model fails.

    python scripts/diagnostics.py --sources D2-infill
    python scripts/diagnostics.py --sources D1

The sweep established that no condition beats B3 on either draft source, while
macro-F1 sits at 0.09-0.13 even on D1 -- where the validator showed corruption
*is* detectable, at ~3.7 ΔNLL per unit of change. So the head's failure is not
explained by draft realism alone, and rebuilding the architecture without
knowing why would be guesswork.

  **probe**  -- train encoder + edit head on the edit loss *alone*, with no
    generative objective. The generative loss rewards copying, which may be
    shaping the encoder toward representations that deliberately do not
    distinguish a wrong note from a right one. If macro-F1 rises sharply here,
    that multi-task tension is the problem and the signal was available all
    along. If it stays put, the labels or the task are at fault.

  **oracle** -- hand the model ground-truth edit labels at the input and train
    generatively. This asks what the decoder would do given perfect
    localization. If Correction Cost then beats B3, localization is the
    bottleneck and the head is where to invest. If it still cannot, the
    generative formulation is the bottleneck and decoding the edit script
    directly is the answer.

Together they separate "cannot find the errors" from "cannot fix them", which
are different problems with different fixes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.drafts import empirical_priors  # noqa: E402
from src.edit_model import (  # noqa: E402
    IGNORE_INDEX,
    EditAwareTransformer,
    EditModelConfig,
    edit_class_weights,
    focal_edit_loss,
)
from src.evaluation import (  # noqa: E402
    copy_input_baseline,
    edit_prediction_scores,
    generate_corrections,
    sample_metrics,
)
from src.judge import save_json  # noqa: E402
from src.model import LMConfig, MelodyLM  # noqa: E402
from src.refinement import RefinementDataset, build_draft_corpus, cached_drafts, make_collate  # noqa: E402


def load_generator(path, tokenizer, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = MelodyLM(LMConfig(**blob["lm_config"]), pad_id=tokenizer.pad_id)
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()


def build_datasets(args, tokenizer, priors, generator):
    splits = standard_splits()
    out = []
    for name, seed in (("judge_train", 100), ("judge_val", 200), ("heldout", 300)):
        melodies = splits[name]
        if args.max_train and name == "judge_train":
            melodies = melodies[: args.max_train]
        drafts = cached_drafts(
            melodies,
            lambda m=melodies, s=seed: build_draft_corpus(
                m, tokenizer, priors, sources=tuple(args.sources),
                generator=generator, temperature=args.temperature,
                per_melody=args.per_melody, seed=s,
            ),
            {"sources": args.sources, "temperature": args.temperature,
             "per_melody": args.per_melody, "seed": seed, "tag": name},
        )
        out.append(RefinementDataset(drafts, tokenizer))
    return out


def run_probe(args, tokenizer, datasets, device, weighting=None, use_focal=None, quiet=False):
    """Train the encoder and edit head on classification alone."""
    weighting = weighting if weighting is not None else getattr(args, "weighting", "inverse")
    use_focal = use_focal if use_focal is not None else getattr(args, "focal", False)
    train_ds, val_ds, test_ds = datasets
    torch.manual_seed(args.seed)

    model = EditAwareTransformer(
        EditModelConfig(
            vocab_size=len(tokenizer), d_model=args.d_model, n_heads=args.n_heads,
            n_encoder_layers=args.n_layers, n_decoder_layers=1, use_edit_head=True,
        ),
        pad_id=tokenizer.pad_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    weights = edit_class_weights(train_ds.label_tensor(), scheme=weighting).to(device)

    collate = make_collate(tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    tag = f"{weighting}{'+focal' if use_focal else ''}"
    print(f"\n=== probe: edit head alone | weighting={tag} | {args.sources} ===")
    best, best_state, stale = float("inf"), None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for source, labels, _, _, _ in train_loader:
            source, labels = source.to(device), labels.to(device)
            logits = model.edit_head(model.encode(source))
            loss = (
                focal_edit_loss(logits, labels, class_weights=weights)
                if use_focal else
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                    weight=weights, ignore_index=IGNORE_INDEX,
                )
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        total = 0.0
        with torch.no_grad():
            for source, labels, _, _, _ in val_loader:
                source, labels = source.to(device), labels.to(device)
                logits = model.edit_head(model.encode(source))
                total += float(
                    focal_edit_loss(logits, labels, class_weights=weights)
                    if use_focal else
                    F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                        weight=weights, ignore_index=IGNORE_INDEX,
                    )
                )
        val = total / max(len(val_loader), 1)
        if not quiet:
            print(f"  epoch {epoch:3d}  val edit-loss {val:.4f}")

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

    predicted, actual = [], []
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for source, labels, _, _, _ in test_loader:
            preds = model.predict_edits(source.to(device)).cpu()
            for row, label_row in zip(preds, labels):
                keep = label_row != IGNORE_INDEX
                predicted.extend(row[keep].tolist())
                actual.extend(label_row[keep].tolist())

    scores = edit_prediction_scores(predicted, actual)
    print(f"  macro-F1 {scores['macro_f1']:.3f}   localization-F1 "
          f"{scores['localization']['f1']:.3f}   accuracy {scores['accuracy']:.3f}")
    for name, block in scores["per_class"].items():
        print(f"    {name:14s} P {block['precision']:.3f}  R {block['recall']:.3f}  "
              f"F1 {block['f1']:.3f}  support {block['support']}")
    return scores


def run_oracle(args, tokenizer, datasets, device):
    """Train generatively with ground-truth edit labels fed in at the input."""
    train_ds, val_ds, test_ds = datasets
    torch.manual_seed(args.seed)

    model = EditAwareTransformer(
        EditModelConfig(
            vocab_size=len(tokenizer), d_model=args.d_model, n_heads=args.n_heads,
            n_encoder_layers=args.n_layers, n_decoder_layers=args.n_layers,
            use_edit_head=False, use_edit_hint=True,
        ),
        pad_id=tokenizer.pad_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    collate = make_collate(tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    print(f"\n=== oracle: ground-truth edit labels at the input ({args.sources}) ===")
    best, best_state, stale = float("inf"), None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for source, labels, target_in, target_out, _ in train_loader:
            source, labels = source.to(device), labels.to(device)
            target_in, target_out = target_in.to(device), target_out.to(device)
            logits, _ = model(source, target_in, edit_hint=labels)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target_out.reshape(-1),
                ignore_index=tokenizer.pad_id,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        total = 0.0
        with torch.no_grad():
            for source, labels, target_in, target_out, _ in val_loader:
                source, labels = source.to(device), labels.to(device)
                target_in, target_out = target_in.to(device), target_out.to(device)
                logits, _ = model(source, target_in, edit_hint=labels)
                total += float(F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), target_out.reshape(-1),
                    ignore_index=tokenizer.pad_id,
                ))
        val = total / max(len(val_loader), 1)
        print(f"  epoch {epoch:3d}  val gen-loss {val:.4f}")

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

    rows = []
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for source, labels, _, _, batch in test_loader:
            source, labels = source.to(device), labels.to(device)
            memory = model.encode(source, labels)
            outputs = decode_from_memory(model, tokenizer, memory, source, device)
            for output, example in zip(outputs, batch):
                rows.append(sample_metrics(output, example.draft_notes, example.reference.notes))

    b3 = copy_input_baseline([(e.draft_notes, e.reference.notes) for e in test_ds.examples])
    cc = float(np.mean([r["correction_cost"] for r in rows]))
    fidelity = float(np.mean([r["fidelity_per_note"] for r in rows]))
    draft_cost = float(np.mean([r["draft_cost"] for r in rows]))

    print(f"  oracle CC {cc:.3f}   B3 {b3['correction_cost_mean']:.3f}   "
          f"fidelity {fidelity:.4f}   lift {(draft_cost - cc)/draft_cost:+.2%}")
    return {"cc": cc, "b3": b3["correction_cost_mean"], "fidelity": fidelity,
            "lift": (draft_cost - cc) / draft_cost,
            "costs": np.array([r["correction_cost"] for r in rows])}


@torch.no_grad()
def decode_from_memory(model, tokenizer, memory, source, device, max_new_tokens=512):
    """Greedy grammar-constrained decoding from a precomputed encoder memory."""
    from src.sampling import Grammar

    grammar = Grammar(tokenizer, device)
    batch = source.size(0)
    tokens = torch.full((batch, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(min(max_new_tokens, model.config.max_len - 1)):
        logits = model.decode(tokens, memory, source)[:, -1]
        allowed = torch.stack([grammar.next_mask(int(t)) for t in tokens[:, -1].cpu()]).to(device)
        logits = logits.masked_fill(~allowed, float("-inf"))
        nxt = logits.argmax(dim=-1)
        nxt = torch.where(finished, torch.full_like(nxt, tokenizer.pad_id), nxt)
        tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
        finished = finished | (nxt == tokenizer.eos_id)
        if bool(finished.all()):
            break

    return [tokenizer.decode(row.tolist()) for row in tokens]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="*", default=["D2-infill"])
    parser.add_argument("--which", nargs="*", default=["probe", "oracle"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--per-melody", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weighting", default="inverse",
                        choices=["none", "sqrt", "capped", "inverse"])
    parser.add_argument("--focal", action="store_true")
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
    print(f"examples: train {len(datasets[0])} val {len(datasets[1])} test {len(datasets[2])}")

    results = {"sources": args.sources}
    if "probe" in args.which:
        results["probe"] = run_probe(args, tokenizer, datasets, device)
    if "oracle" in args.which:
        oracle = run_oracle(args, tokenizer, datasets, device)
        oracle.pop("costs", None)
        results["oracle"] = oracle

    tag = "_".join(args.sources).replace("-", "")
    out = args.out or f"results/refiner/diagnostics_{tag}.json"
    save_json(results, out)
    print(f"\ndiagnostics -> {out}")


if __name__ == "__main__":
    main()
