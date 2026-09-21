"""
Train the refinement model across the alpha sweep, and evaluate it.

    python scripts/train_refiner.py --alphas 0 0.5 1.0
    python scripts/train_refiner.py --alphas 0 0.1 0.5 1.0 2.0 3.0 --seeds 0 1 2

Splits (see `src.data.standard_splits`):
  - references for training come from the Essen half `G` never trained on, so a
    D2 draft is a genuine continuation rather than `G` reciting a memorized
    melody;
  - the test set is MTC-ANN, which neither `G` nor `J` has seen.

alpha = 0 with `--no-edit-head` is baseline B1; alpha = 0 with the head present
is B2, the parameter-matched control. B3 (copy the draft unchanged) needs no
training and is computed during evaluation as the lower bound that gives
Correction Cost a scale.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import edit_labels  # noqa: E402
from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.drafts import empirical_priors  # noqa: E402
from src.edit_model import (  # noqa: E402
    IGNORE_INDEX,
    EditAwareTransformer,
    EditModelConfig,
    binarize_labels,
    edit_aware_loss,
    edit_class_weights,
)
from src.evaluation import (  # noqa: E402
    copy_input_baseline,
    edit_prediction_scores,
    generate_corrections,
    generate_corrections_gated,
    sample_metrics,
)
from src.judge import Judge, save_json  # noqa: E402
from src.model import LMConfig, MelodyLM  # noqa: E402
from src.refinement import (  # noqa: E402
    RefinementDataset,
    attach_surprisal,
    build_draft_corpus,
    cached_drafts,
    make_collate,
    surprisal_tensor,
)


def load_generator(path, tokenizer, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = MelodyLM(LMConfig(**blob["lm_config"]), pad_id=tokenizer.pad_id)
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()


def make_drafts(melodies, tokenizer, priors, generator, args, seed, tag):
    config = {
        "sources": args.sources, "temperature": args.temperature,
        "per_melody": args.per_melody, "seed": seed, "tag": tag,
    }
    return cached_drafts(
        melodies,
        lambda: build_draft_corpus(
            melodies, tokenizer, priors,
            sources=tuple(args.sources), generator=generator,
            temperature=args.temperature, per_melody=args.per_melody, seed=seed,
        ),
        config,
    )


def evaluate(model, dataset, tokenizer, device, batch_size=32, gated_thresholds=(), binary=False):
    """Autoregressive decoding, Correction Cost, fidelity, and edit-F1."""
    collate = make_collate(tokenizer.pad_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    rows, predicted_edits, actual_edits = [], [], []
    gated = {t: [] for t in gated_thresholds}

    for source, labels, _, _, batch in loader:
        source = source.to(device)
        surprisal = surprisal_tensor(batch, source.size(1), device)
        outputs = generate_corrections(model, tokenizer, source, surprisal=surprisal)

        if model.edit_head is not None:
            predictions = model.predict_edits(source, surprisal=surprisal).cpu()
            truth = binarize_labels(labels) if binary else labels
            for row, label_row in zip(predictions, truth):
                keep = label_row != IGNORE_INDEX
                predicted_edits.extend(row[keep].tolist())
                actual_edits.extend(label_row[keep].tolist())

        for output, example in zip(outputs, batch):
            rows.append(
                sample_metrics(output, example.draft_notes, example.reference.notes)
            )

        # Gated decoding: copy by default, edit only where the head is confident.
        # Threshold 1.0 reproduces B3 exactly, so this can only improve on it.
        if gated_thresholds and model.edit_head is not None:
            for threshold in gated_thresholds:
                out = generate_corrections_gated(
                    model, tokenizer, source, threshold=threshold, surprisal=surprisal
                )
                for output, example in zip(out, batch):
                    gated[threshold].append(
                        sample_metrics(output, example.draft_notes, example.reference.notes)
                    )

    summary = {
        key: float(np.mean([r[key] for r in rows]))
        for key in ("correction_cost", "correction_cost_per_note",
                    "fidelity_distance", "fidelity_per_note", "draft_cost")
    }
    summary["n"] = len(rows)
    summary["costs"] = np.array([r["correction_cost"] for r in rows])
    if predicted_edits:
        summary["edit_scores"] = edit_prediction_scores(
            predicted_edits, actual_edits, n_classes=2 if binary else 5)
    if gated:
        summary["gated"] = {
            str(t): {
                "correction_cost": float(np.mean([r["correction_cost"] for r in rs])),
                "fidelity_per_note": float(np.mean([r["fidelity_per_note"] for r in rs])),
                "costs": np.array([r["correction_cost"] for r in rs]),
            }
            for t, rs in gated.items() if rs
        }
    return summary


def train_one(args, alpha, seed, use_head, splits, tokenizer, datasets, device):
    binary = args.head_type == "binary"
    label = (f"alpha{alpha}_seed{seed}"
             + ("" if use_head else "_nohead")
             + ("_binary" if binary and use_head else ""))
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = EditModelConfig(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_layers,
        n_decoder_layers=args.n_layers,
        use_edit_head=use_head,
        use_surprisal=args.surprisal,
        n_edit_classes=2 if binary else 5,
    )
    model = EditAwareTransformer(config, pad_id=tokenizer.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    train_ds, val_ds, _ = datasets
    raw = train_ds.label_tensor()
    weights = (edit_class_weights(binarize_labels(raw) if binary else raw,
                                  n_classes=2 if binary else 5,
                                  scheme=args.weighting).to(device)
               if use_head else None)

    collate = make_collate(tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    print(f"\n=== {label} | {model.n_parameters/1e6:.2f}M params ===")

    history = {"train": [], "val": []}
    best, best_state, stale = float("inf"), None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for source, labels, target_in, target_out, batch in train_loader:
            source, labels = source.to(device), labels.to(device)
            target_in, target_out = target_in.to(device), target_out.to(device)
            surprisal = surprisal_tensor(batch, source.size(1), device)

            token_logits, edit_logits = model(source, target_in, surprisal=surprisal)
            targets_edit = binarize_labels(labels) if binary else labels
            loss, _, _ = edit_aware_loss(
                token_logits, target_out, edit_logits, targets_edit,
                tokenizer.pad_id, alpha=alpha, class_weights=weights,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for source, labels, target_in, target_out, batch in val_loader:
                source, labels = source.to(device), labels.to(device)
                target_in, target_out = target_in.to(device), target_out.to(device)
                surprisal = surprisal_tensor(batch, source.size(1), device)
                token_logits, edit_logits = model(source, target_in, surprisal=surprisal)
                targets_edit = binarize_labels(labels) if binary else labels
                loss, _, _ = edit_aware_loss(
                    token_logits, target_out, edit_logits, targets_edit,
                    tokenizer.pad_id, alpha=alpha, class_weights=weights,
                )
                val_running += float(loss)

        train_loss = running / max(len(train_loader), 1)
        val_loss = val_running / max(len(val_loader), 1)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")

        if best - val_loss > 1e-4:
            best, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  early stop at epoch {epoch}; best val {best:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--sources", nargs="*", default=["D2-infill"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--per-melody", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--generator", default="models/generator.pth")
    parser.add_argument("--out", default="results/refiner/sweep.json")
    parser.add_argument("--max-train", type=int, default=0, help="cap training melodies (0 = all)")
    parser.add_argument("--essen-test", type=float, default=0.0,
                        help="carve a same-corpus Essen reference set (control)")
    parser.add_argument("--references", default="heldout",
                        choices=["heldout", "essen_test"],
                        help="which split supplies the test references")
    parser.add_argument("--gated-thresholds", type=float, nargs="*", default=[],
                        help="evaluate gated decoding at these confidence thresholds")
    parser.add_argument("--surprisal", action="store_true",
                        help="feed the judge's per-token surprisal to the encoder")
    parser.add_argument("--judge", default="models/judge.pth")
    parser.add_argument("--head-type", default="five", choices=["five", "binary"],
                        help="spec 7 head-type axis; 'none' is the --alphas 0 B1 row")
    parser.add_argument("--weighting", default="inverse",
                        choices=["none", "sqrt", "capped", "inverse"],
                        help="edit-head class weighting; the ablation found "
                             "'inverse' halves macro-F1 on D1 against 'none'")
    args = parser.parse_args()

    # Per-sample costs are namespaced by the sweep, so two sweeps cannot
    # overwrite each other's arrays -- which silently made the D1 run's paired
    # statistics stand in for the D2 run's.
    cost_dir = Path(args.out).with_suffix("")
    cost_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = standard_splits(essen_test=args.essen_test)
    every = [m for group in splits.values() for m in group]
    tokenizer = build_tokenizer(every)
    priors = empirical_priors(every)
    generator = load_generator(args.generator, tokenizer, device)

    # References G never trained on, so D2 drafts are real continuations.
    train_refs = splits["judge_train"]
    if args.max_train:
        train_refs = train_refs[: args.max_train]
    val_refs = splits["judge_val"]
    test_refs = splits[args.references]

    print(f"references: train {len(train_refs)}  val {len(val_refs)}  test {len(test_refs)}")

    train_drafts = make_drafts(train_refs, tokenizer, priors, generator, args, 100, "train")
    val_drafts = make_drafts(val_refs, tokenizer, priors, generator, args, 200, "val")
    test_drafts = make_drafts(test_refs, tokenizer, priors, generator, args, 300, "test")

    datasets = (
        RefinementDataset(train_drafts, tokenizer),
        RefinementDataset(val_drafts, tokenizer),
        RefinementDataset(test_drafts, tokenizer),
    )
    print(f"examples: train {len(datasets[0])}  val {len(datasets[1])}  test {len(datasets[2])}")

    if args.surprisal:
        judge = Judge.load(args.judge)
        if len(judge.tokenizer) != len(tokenizer):
            raise ValueError(
                f"judge vocabulary {len(judge.tokenizer)} != refiner {len(tokenizer)}; "
                "surprisal indices would not line up"
            )
        for dataset in datasets:
            attach_surprisal(dataset, judge)
        print("attached judge surprisal to all splits")

    # B3: the lower bound. A model that cannot beat this is not correcting.
    b3 = copy_input_baseline(
        [(e.draft_notes, e.reference.notes) for e in datasets[2].examples]
    )
    print(f"\nB3 copy-input: CC {b3['correction_cost_mean']:.2f} "
          f"({b3['correction_cost_per_note']:.4f}/note)")

    results = []
    conditions = [(a, True) for a in args.alphas]
    if 0.0 in args.alphas:
        conditions.append((0.0, False))  # B1: no head at all

    for alpha, use_head in conditions:
        for seed in args.seeds:
            model, history, label = train_one(
                args, alpha, seed, use_head, splits, tokenizer, datasets, device
            )
            metrics = evaluate(model, datasets[2], tokenizer, device,
                               gated_thresholds=tuple(args.gated_thresholds),
                               binary=(args.head_type == "binary"))
            costs = metrics.pop("costs")
            gated_blocks = metrics.pop("gated", {})
            for threshold, block in gated_blocks.items():
                np.save(cost_dir / f"costs_{label}_gated{threshold}.npy", block.pop("costs"))
            if gated_blocks:
                metrics["gated"] = gated_blocks

            row = {
                "label": label, "alpha": alpha, "seed": seed,
                "use_edit_head": use_head,
                "head_type": args.head_type if use_head else "none",
                "n_parameters": model.n_parameters,
                **metrics,
                "best_val_loss": min(history["val"]),
            }
            results.append(row)
            np.save(cost_dir / f"costs_{label}.npy", costs)

            summary = (f"  -> CC {row['correction_cost']:.2f} "
                       f"({row['correction_cost_per_note']:.4f}/note)  "
                       f"fidelity {row['fidelity_per_note']:.4f}")
            if "edit_scores" in row:
                summary += (f"  macroF1 {row['edit_scores']['macro_f1']:.3f}"
                            f"  locF1 {row['edit_scores']['localization']['f1']:.3f}")
            print(summary)
            for threshold, block in sorted(row.get("gated", {}).items(), key=lambda kv: float(kv[0])):
                delta = b3["correction_cost_mean"] - block["correction_cost"]
                print(f"       gated t={threshold:>4}  CC {block['correction_cost']:.3f}  "
                      f"fidelity {block['fidelity_per_note']:.4f}  "
                      f"vs B3 {delta:+.3f}{'  BEATS B3' if delta > 0 else ''}")

    save_json({"b3_copy_input": {k: v for k, v in b3.items() if k != "costs"},
               "rows": results, "config": vars(args)}, args.out)
    np.save(cost_dir / "costs_b3.npy", b3["costs"])

    print(f"\n{'condition':24s} {'CC/note':>9s} {'fidelity':>9s} {'macroF1':>8s} {'locF1':>7s}")
    print("-" * 62)
    print(f"{'B3 copy-input':24s} {b3['correction_cost_per_note']:9.4f}")
    for row in results:
        f1 = row.get("edit_scores", {}).get("macro_f1", float("nan"))
        loc = row.get("edit_scores", {}).get("localization", {}).get("f1", float("nan"))
        print(f"{row['label']:24s} {row['correction_cost_per_note']:9.4f} "
              f"{row['fidelity_per_note']:9.4f} {f1:8.3f} {loc:7.3f}")
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
