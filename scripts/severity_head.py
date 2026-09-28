"""
The generator-quality curve, scored by a localizer independent of the judge.

    python scripts/severity_head.py

Figure 1a scores localizability with the two-sided judge, whose training data it
shares with `J` -- the model that supplies the x-axis (implausibility per unit of
change). A change `J` finds implausible is, almost by construction, one its sibling
finds surprising, so part of that curve is mechanical. Here the same drafts are
scored by edit heads trained on *another* draft source's edit labels (D1, and
separately D3; headline checkpoints, 3 seeds), applied zero-shot. They never saw
`J`'s scores or any generated draft, so a rise across generator settings is not
inherited from the x-axis.

Drafts are rebuilt exactly as `severity_curve.py` builds them (same tags, same
cache), so the rows are directly comparable with Figure 1a.
"""
import statistics
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.drafts import empirical_priors  # noqa: E402
from src.edit_model import IGNORE_INDEX  # noqa: E402
from src.judge import save_json  # noqa: E402
from src.refinement import RefinementDataset  # noqa: E402
from scripts.judge_gate import head_probs, load_refiner  # noqa: E402
from scripts.localization_report import auroc, average_precision  # noqa: E402
from scripts.severity_curve import CORPORA, conditions  # noqa: E402
from scripts.train_refiner import load_generator, make_drafts  # noqa: E402

SEEDS = (0, 1, 2)
HEADS = {"D1-trained head": "d1", "D3-trained head": "d3"}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for corpus, (essen_test, refs_key, generator, ckpt_dir, _, _) in CORPORA.items():
        splits = standard_splits(essen_test=essen_test)
        every = [m for group in splits.values() for m in group]
        tokenizer = build_tokenizer(every)
        priors = empirical_priors(every)
        refs = splits[refs_key]
        prefix = "mtc" if corpus == "MTC" else "essen"
        heads = {name: [load_refiner(Path(f"results/refiner/headline_r2/{prefix}_{src}")
                                     / f"model_alpha0.5_seed{s}_binary.pt", tokenizer.pad_id, device)
                        for s in SEEDS] for name, src in HEADS.items()}
        for label, source, gen_path, temperature, tag in conditions(generator, ckpt_dir):
            gen = load_generator(gen_path, tokenizer, device)
            args = Namespace(sources=[source], temperature=temperature, per_melody=3)
            examples = RefinementDataset(make_drafts(refs, tokenizer, priors, gen, args, 300, tag),
                                         tokenizer).examples
            row = {"corpus": corpus, "condition": label, "n_drafts": len(examples)}
            for name, models in heads.items():
                aurocs, lifts = [], []
                for model in models:
                    probs = head_probs(model, examples, tokenizer.pad_id, device)
                    scores, truth = [], []
                    for ex, p in zip(examples, probs):
                        labels = np.array(ex.source_labels)
                        keep = labels != IGNORE_INDEX
                        scores.append(p[keep]); truth.append((labels[keep] != 0).astype(int))
                    s, y = np.concatenate(scores), np.concatenate(truth)
                    aurocs.append(auroc(s, y)); lifts.append(average_precision(s, y) / y.mean())
                row[name] = {"auroc": statistics.mean(aurocs), "auroc_sd": statistics.stdev(aurocs),
                             "ap_lift": statistics.mean(lifts)}
            rows.append(row)
            print(f"{corpus:6s} {label:10s} " + "  ".join(
                f"{n}: AUROC {row[n]['auroc']:.3f} ±{row[n]['auroc_sd']:.3f} lift {row[n]['ap_lift']:.1f}x"
                for n in HEADS), flush=True)
    path = Path("results/judge/severity_head.json")
    save_json({"rows": rows}, path)
    print(f"\nseverity (independent heads) -> {path}")


if __name__ == "__main__":
    main()
