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
    surprisal: torch.Tensor | None = None,
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
    memory = model.encode(sources, surprisal=surprisal)

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


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, as the signed-rank test requires."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)

    sorted_values = values[order]
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return ranks


def wilcoxon_signed_rank(baseline: np.ndarray, treatment: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test, normal approximation.

    Implemented here rather than pulled from scipy to avoid adding a dependency
    for one test. The normal approximation with continuity and tie corrections is
    appropriate at the sample sizes involved (hundreds of test melodies); it
    would not be for a handful.
    """
    import math

    deltas = np.asarray(baseline, dtype=float) - np.asarray(treatment, dtype=float)
    nonzero = deltas[deltas != 0]
    n = len(nonzero)
    if n < 10:
        return {"n_nonzero": n, "p_value": None, "z": None,
                "note": "too few non-tied pairs for the normal approximation"}

    ranks = _average_ranks(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0].sum())

    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0

    # Tie correction over groups of equal |d|.
    _, counts = np.unique(np.abs(nonzero), return_counts=True)
    variance -= float(((counts ** 3 - counts).sum()) / 48.0)

    if variance <= 0:
        return {"n_nonzero": n, "p_value": None, "z": None, "note": "zero variance"}

    correction = 0.5 if w_plus > mean else -0.5
    z = (w_plus - mean - correction) / math.sqrt(variance)
    p = math.erfc(abs(z) / math.sqrt(2.0))

    return {"n_nonzero": n, "w_plus": w_plus, "z": float(z), "p_value": float(p)}


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

    result["wilcoxon"] = wilcoxon_signed_rank(baseline, treatment)
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


@torch.no_grad()
def generate_corrections_gated(
    model: EditAwareTransformer,
    tokenizer: MelodyTokenizer,
    sources: torch.Tensor,
    threshold: float = 0.5,
    surprisal: torch.Tensor | None = None,
    device=None,
    gate_scores: torch.Tensor | None = None,
) -> list[Notes]:
    """Copy the draft by default; regenerate only where the head is confident.

    `gate_scores`, if given, replaces the head as the gate: a (B, T) score per
    source position, read at each event's first token like the head's
    confidence. An event whose score clears `threshold` is regenerated as a
    substitution; everything else is copied. The model then serves only as the
    decoder, so it needs no edit head -- which is what lets an external,
    label-free scorer (the two-sided judge) drive the edit decisions.

    The diagnostics showed localization *precision* is the binding constraint,
    not F1. At D1's precision of 0.68 roughly a third of flagged positions are
    notes that were already correct, so every additional edit is a coin flip
    weighted against you -- which is why raising recall (via judge surprisal)
    made Correction Cost worse while improving F1, and why free-form decoding
    settled on copying.

    So the head is used as a gate rather than an auxiliary signal. Each input
    event is either copied verbatim or handed to the decoder, according to the
    predicted class and only when the model's confidence clears `threshold`.
    Raising the threshold trades recall for precision directly, which is the
    trade the measurements say is worth making.

    This is also what "where and how" should mean operationally: the head
    decides where and which operation, the decoder supplies the content.
    """
    from src.align import DELETE, INSERT_BEFORE, KEEP

    device = device or next(model.parameters()).device
    model.eval()
    sources = sources.to(device)
    batch = sources.size(0)

    grammar = Grammar(tokenizer, device)
    memory = model.encode(sources, surprisal=surprisal)

    if gate_scores is None:
        probabilities = torch.softmax(model.edit_head(memory), dim=-1).cpu()
        predicted = probabilities.argmax(dim=-1)
        confidence = 1.0 - probabilities[:, :, KEEP]  # P(this position needs an edit)
    else:
        from src.align import SUB_PITCH

        confidence = gate_scores.detach().float().cpu()
        if confidence.shape != sources.shape:
            raise ValueError(f"gate_scores {tuple(confidence.shape)} must match "
                             f"sources {tuple(sources.shape)}")
        # An external gate says only *whether* to edit, so a flagged event is
        # handed to the decoder as a substitution, as the binary head's is.
        predicted = torch.where(confidence >= threshold, SUB_PITCH, KEEP)

    # Build each sequence's action tape up front: copy a token, or let the
    # decoder write one. No model calls needed for this part.
    tapes: list[list[tuple[str, int]]] = []
    for i in range(batch):
        row = sources[i].tolist()
        events = tokenizer.decode(row)
        _, spans = tokenizer.encode_with_spans(events) if events else ([], [])

        # Map each source position back to its event, using the real token
        # stream rather than index arithmetic, which BAR and REST would break.
        starts = {}
        event_index = 0
        position = 0
        while position < len(row) and event_index < len(events):
            token = tokenizer.id_to_token.get(row[position], "")
            if token.startswith("P") or token == "<REST>":
                starts[position] = event_index
                event_index += 1
                position += 2
            else:
                position += 1

        tape: list[tuple[str, int]] = []
        position = 0
        while position < len(row):
            token_id = row[position]
            token = tokenizer.id_to_token.get(token_id, "")

            if position in starts:
                index = starts[position]
                pair = [token_id, row[position + 1]] if position + 1 < len(row) else [token_id]
                klass = int(predicted[i, position])
                confident = float(confidence[i, position]) >= threshold

                if not confident or klass == KEEP:
                    tape += [("copy", t) for t in pair]
                elif klass == DELETE:
                    pass
                elif klass == INSERT_BEFORE:
                    tape += [("free", 0), ("free", 0)]
                    tape += [("copy", t) for t in pair]
                else:  # SUB_PITCH or SUB_DUR
                    tape += [("free", 0), ("free", 0)]
                position += 2
            else:
                if token in ("<BAR>",):
                    tape.append(("copy", token_id))
                position += 1

        tape.append(("copy", tokenizer.eos_id))
        tapes.append(tape)

    # Execute the tapes in lockstep, running the decoder only where needed.
    tokens = torch.full((batch, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    cursors = [0] * batch
    longest = max((len(t) for t in tapes), default=0)

    for _ in range(min(longest, model.config.max_len - 1)):
        logits = model.decode(tokens, memory, sources)[:, -1]
        allowed = torch.stack(
            [grammar.next_mask(int(t)) for t in tokens[:, -1].cpu()]
        ).to(device)
        free_choice = logits.masked_fill(~allowed, float("-inf")).argmax(dim=-1)

        step = torch.full((batch,), tokenizer.pad_id, dtype=torch.long, device=device)
        for i in range(batch):
            if cursors[i] >= len(tapes[i]):
                continue
            action, value = tapes[i][cursors[i]]
            step[i] = value if action == "copy" else int(free_choice[i])
            cursors[i] += 1

        tokens = torch.cat([tokens, step.unsqueeze(1)], dim=1)
        if all(c >= len(t) for c, t in zip(cursors, tapes)):
            break

    return [tokenizer.decode(row.tolist()) for row in tokens]
