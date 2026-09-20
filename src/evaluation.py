"""
Evaluation for the refinement task (spec 6).

Four things the published version did not report, each answering an objection:

  - **Fidelity**, `EditDist(output, M_0)`. A model with low Correction Cost and
    high fidelity distance is not editing the draft; it is writing a fresh
    melody that happens to resemble the target. Reviewers raised this and it
    costs nothing to measure.
  - **Edit-prediction F1**, per class and macro. This is what substantiates
    "where and how"; the published model predicted a binary mask.
  - **Paired per-sample statistics** -- win rate, bootstrap CI, Wilcoxon. The
    old paper defined these in its Section 6.2 and never reported them.
  - **Musical quality** of outputs under the judge, since the validator showed
    Correction Cost and plausibility are close to orthogonal.

Decoding is autoregressive. The legacy pipeline took argmax over teacher-forced
logits, which conditions every step on the ground-truth prefix and so reports a
number no deployed model could achieve.
"""
import numpy as np
import torch

from src.align import EDIT_CLASSES, KEEP, correction_cost, edit_labels
from src.edit_model import IGNORE_INDEX, EditAwareTransformer
from src.sampling import Grammar
from src.tokenizer import MelodyTokenizer

Notes = list[tuple[int, int]]


@torch.no_grad()
def generate_corrections(
    model: EditAwareTransformer,
    tokenizer: MelodyTokenizer,
    sources: torch.Tensor,
    max_new_tokens: int = 512,
    device=None,
) -> list[Notes]:
    """Greedy, grammar-constrained autoregressive decoding.

    Constrained for the same reason sampling is: an unpaired pitch would be
    dropped by the decoder and would show up in the edit script as a deletion
    the model never chose.
    """
    device = device or next(model.parameters()).device
    model.eval()
    sources = sources.to(device)

    batch = sources.size(0)
    grammar = Grammar(tokenizer, device)
    memory = model.encode(sources)

    tokens = torch.full((batch, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    limit = min(max_new_tokens, model.config.max_len - 1)
    for _ in range(limit):
        logits = model.decode(tokens, memory, sources)[:, -1]

        allowed = torch.stack(
            [grammar.next_mask(int(t)) for t in tokens[:, -1].cpu()]
        ).to(device)
        logits = logits.masked_fill(~allowed, float("-inf"))

        nxt = logits.argmax(dim=-1)
        nxt = torch.where(finished, torch.full_like(nxt, tokenizer.pad_id), nxt)
        tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
        finished = finished | (nxt == tokenizer.eos_id)
        if bool(finished.all()):
            break

    return [tokenizer.decode(row.tolist()) for row in tokens]


def edit_prediction_scores(
    predicted: list[int],
    actual: list[int],
    n_classes: int = len(EDIT_CLASSES),
) -> dict:
    """Per-class precision/recall/F1, macro-F1, and binary localization.

    Localization collapses the five classes to KEEP against everything else,
    which is the "where" question; macro-F1 over all five is the "how".
    """
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    keep = actual != IGNORE_INDEX
    predicted, actual = predicted[keep], actual[keep]

    per_class = {}
    f1s = []
    for cls in range(n_classes):
        tp = int(((predicted == cls) & (actual == cls)).sum())
        fp = int(((predicted == cls) & (actual != cls)).sum())
        fn = int(((predicted != cls) & (actual == cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[EDIT_CLASSES[cls]] = {
            "precision": precision, "recall": recall, "f1": f1, "support": tp + fn,
        }
        if tp + fn:
            f1s.append(f1)

    edited_pred = predicted != KEEP
    edited_true = actual != KEEP
    tp = int((edited_pred & edited_true).sum())
    fp = int((edited_pred & ~edited_true).sum())
    fn = int((~edited_pred & edited_true).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "per_class": per_class,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "localization": {
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        },
        "accuracy": float((predicted == actual).mean()) if len(actual) else 0.0,
    }


def sample_metrics(output: Notes, draft: Notes, reference: Notes) -> dict:
    """Correction Cost and fidelity for one example, both length-normalized."""
    n = max(len(reference), 1)
    cost = correction_cost(output, reference)
    return {
        "correction_cost": cost,
        "correction_cost_per_note": cost / n,
        "fidelity_distance": correction_cost(output, draft),
        "fidelity_per_note": correction_cost(output, draft) / max(len(draft), 1),
        "draft_cost": correction_cost(draft, reference),
        "n_reference": len(reference),
        "n_output": len(output),
    }


def paired_comparison(baseline: np.ndarray, treatment: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """Win rate, paired delta, bootstrap CI and a Wilcoxon test.

    Spec 6.1 asks for exactly this, and the published paper defined it without
    reporting it. It is what turns an aggregate percentage into a claim that
    survives review.
    """
    baseline = np.asarray(baseline, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    deltas = baseline - treatment  # positive means the treatment is better

    rng = np.random.default_rng(seed)
    if len(deltas):
        idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
        boot = deltas[idx].mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
    else:
        low = high = float("nan")

    result = {
        "n": int(len(deltas)),
        "baseline_mean": float(baseline.mean()) if len(baseline) else float("nan"),
        "treatment_mean": float(treatment.mean()) if len(treatment) else float("nan"),
        "mean_delta": float(deltas.mean()) if len(deltas) else float("nan"),
        "median_delta": float(np.median(deltas)) if len(deltas) else float("nan"),
        "win_rate": float((deltas > 0).mean() * 100) if len(deltas) else float("nan"),
        "loss_rate": float((deltas < 0).mean() * 100) if len(deltas) else float("nan"),
        "tie_rate": float((deltas == 0).mean() * 100) if len(deltas) else float("nan"),
        "ci95": [float(low), float(high)],
    }

    try:
        from scipy.stats import wilcoxon

        if len(deltas) and np.any(deltas != 0):
            stat, p = wilcoxon(baseline, treatment)
            result["wilcoxon_p"] = float(p)
            result["wilcoxon_stat"] = float(stat)
    except ImportError:
        result["wilcoxon_p"] = None

    return result


def copy_input_baseline(pairs: list[tuple[Notes, Notes]]) -> dict:
    """Baseline B3: emit the draft unchanged.

    The lower bound that calibrates Correction Cost. A model that cannot beat it
    is not correcting anything, and without it a CC figure has no scale.
    """
    costs = np.array([correction_cost(draft, reference) for draft, reference in pairs])
    per_note = np.array(
        [correction_cost(d, r) / max(len(r), 1) for d, r in pairs]
    )
    return {
        "correction_cost_mean": float(costs.mean()) if len(costs) else float("nan"),
        "correction_cost_per_note": float(per_note.mean()) if len(per_note) else float("nan"),
        "costs": costs,
    }
