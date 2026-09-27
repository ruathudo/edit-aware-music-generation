"""
Localizability against off-manifold distance, with more than three points.

    python scripts/severity_curve.py

Figure 1 places three draft sources by the judge's median dNLL per unit of
change and reports how well the untrained two-sided judge localizes their
changed notes. Three points are consistent with "localizability follows
manifold distance" but cannot establish it. D2's severity is tunable: sampling
temperature, and undertrained generator checkpoints, move generated drafts
across the gap between D2 (~1 per unit) and corruption (~3-4). This scores those
conditions the same way, with no training:

  per unit   median of dNLL / (Correction Cost / reference length), by `J`
  AUROC      two-sided judge, per-note scores standardized within each melody,
             on the refiner's labelled positions (as zero_shot_localization.py)

Test drafts are rebuilt exactly as the headline runs build them (3 per
reference, seed 300); the conditions the headline used reuse its cache. The
draft cache is keyed on sampling settings but not on the generator checkpoint,
so every checkpoint condition gets its own tag.
"""
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import correction_cost  # noqa: E402
from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.drafts import empirical_priors  # noqa: E402
from src.edit_model import IGNORE_INDEX  # noqa: E402
from src.judge import Judge, save_json  # noqa: E402
from src.masked_judge import event_groups, load_masked_judge, score_events  # noqa: E402
from src.refinement import RefinementDataset  # noqa: E402
from scripts.localization_report import auroc, average_precision  # noqa: E402
from scripts.train_refiner import load_generator, make_drafts  # noqa: E402
from scripts.zero_shot_localization import zscore  # noqa: E402

CORPORA = {
    # corpus: (essen_test, references, generator, checkpoint dir, judge, masked judge)
    "MTC": (0.0, "heldout", "models/generator.pth", "models/generator_checkpoints",
            "models/judge.pth", "models/masked_judge.pth"),
    "Essen": (0.2, "essen_test", "models/generator_ctrl.pth", "models/generator_ctrl_checkpoints",
              "models/judge_ctrl.pth", "models/masked_judge_ctrl.pth"),
}


def conditions(generator, checkpoint_dir):
    """(label, source, generator path, temperature, cache tag)."""
    out = [("D1", "D1", generator, 1.0, "test"), ("D3", "D3", generator, 1.0, "test")]
    for t in (0.8, 1.0, 1.2):
        out.append((f"D2 T={t}", "D2-infill", generator, t, "test"))
    for path in sorted(Path(checkpoint_dir).glob("generator_epoch*.pth"),
                       key=lambda p: int(p.stem.split("epoch")[-1])):
        epoch = path.stem.split("epoch")[-1]
        out.append((f"D2 ep{epoch}", "D2-infill", str(path), 1.0, f"test-ep{epoch}"))
    return out


def two_sided_auroc(masked, tokenizer, examples, device):
    scores, truth = [], []
    for ex in examples:
        tokens, spans = tokenizer.encode_with_spans(list(ex.draft_notes), ex.reference.bar_starts)
        labels = np.array(ex.source_labels)
        keep = labels != IGNORE_INDEX
        s = score_events(masked, tokens, event_groups(tokens, spans, tokenizer.eos_id), device=device)
        scores.append(zscore(s, keep)[keep])
        truth.append((labels[keep] != 0).astype(np.int64))
    scores, truth = np.concatenate(scores), np.concatenate(truth)
    return auroc(scores, truth), average_precision(scores, truth) / truth.mean(), float(truth.mean())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for corpus, (essen_test, refs_key, generator, ckpt_dir, judge_path, masked_path) in CORPORA.items():
        splits = standard_splits(essen_test=essen_test)
        every = [m for group in splits.values() for m in group]
        tokenizer = build_tokenizer(every)
        priors = empirical_priors(every)
        judge = Judge.load(judge_path, device)
        masked = load_masked_judge(masked_path, device)
        refs = splits[refs_key]
        for label, source, gen_path, temperature, tag in conditions(generator, ckpt_dir):
            gen = load_generator(gen_path, tokenizer, device)
            args = Namespace(sources=[source], temperature=temperature, per_melody=3)
            drafts = make_drafts(refs, tokenizer, priors, gen, args, 300, tag)
            examples = RefinementDataset(drafts, tokenizer).examples
            pairs = [(list(e.draft_notes), e.reference) for e in examples]
            delta = np.array([s.delta_nll for s in judge.score_drafts(pairs)])
            surface = np.array([correction_cost(d, list(r.notes)) / max(len(r.notes), 1)
                                for d, r in pairs])
            a, lift, base = two_sided_auroc(masked, tokenizer, examples, device)
            row = {"corpus": corpus, "condition": label, "n_drafts": len(examples),
                   "surface_mean": float(surface.mean()), "delta_mean": float(delta.mean()),
                   "per_unit_median": float(np.median(delta / np.maximum(surface, 1e-9))),
                   "judge_auroc": a, "judge_ap_lift": lift, "base_rate": base}
            rows.append(row)
            print(f"{corpus:6s} {label:10s} n={len(examples):5d} changed {row['surface_mean']:.3f} "
                  f"dNLL {row['delta_mean']:+.3f} per unit {row['per_unit_median']:5.2f} "
                  f"AUROC {a:.3f} lift {lift:.1f}x", flush=True)
    path = Path("results/judge/severity_curve.json")
    save_json({"rows": rows}, path)
    print(f"\nseverity curve -> {path}")


if __name__ == "__main__":
    main()
