"""
Can an untrained judge find the errors? Causal against two-sided, per source.

    python scripts/zero_shot_localization.py results/refiner/headline_r2

No scorer here has ever seen an edit label or a corruption rule, so whatever
they localize is localizable from musical plausibility alone. Three per-note
scores, each standardized within its melody (localization is a within-melody
question; the melody's overall difficulty is not):

- **causal**: `J`'s surprisal of the event given its past (pitch + duration).
- **causal+next**: that plus the next event's surprisal. A note that breaks
  what follows is charged, by a causal model, to the *following* note; this
  folds the charge back.
- **two-sided**: the masked judge's -log p(event | everything else).

Scored on exactly the positions the refiner head is scored on, with labels
checked against the headline run's saved arrays, and set beside the trained
head's numbers. Same-corpus runs use the control judges, trained without the
Essen test slice.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.edit_model import IGNORE_INDEX  # noqa: E402
from src.judge import Judge, save_json  # noqa: E402
from src.masked_judge import event_groups, load_masked_judge, score_events  # noqa: E402
from scripts.localization_report import auroc, average_precision  # noqa: E402
from scripts.output_quality import test_examples  # noqa: E402


@torch.no_grad()
def causal_surprisal(judge, tokens):
    x = torch.tensor([tokens], dtype=torch.long, device=judge.device)
    logits = judge.model(x[:, :-1])
    ce = F.cross_entropy(logits[0], x[0, 1:], reduction="none").cpu().numpy()
    return np.concatenate([[0.0], ce])      # BOS has no prediction


def per_event(token_scores, groups, n):
    """Joint score per event, written to every token of the event."""
    out = np.zeros(n)
    joint = [token_scores[g].sum() for g in groups]
    for g, s in zip(groups, joint):
        out[g] = s
    return out, joint


def zscore(values, mask):
    v = values[mask]
    sd = v.std()
    return (values - v.mean()) / (sd if sd > 0 else 1.0)


def metrics(scores, truth):
    base = truth.mean()
    k = int(truth.sum())
    top = np.argsort(-scores, kind="stable")[:k]
    ap = average_precision(scores, truth)
    return {"ap": ap, "ap_lift": ap / base, "auroc": auroc(scores, truth),
            "precision_at_k": float(truth[top].mean()), "base_rate": float(base)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--corpus", choices=["MTC", "Essen"], default=None,
                        help="score one corpus only")
    args = parser.parse_args()
    root = Path(args.directory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = {(r["corpus"], r["source"]): r
            for r in json.loads((root / "localization.json").read_text())["rows"]}

    report = []
    for sweep in sorted(root.glob("*.json")):
        if sweep.name in ("headline.json", "localization.json", "quality.json") \
                or sweep.name.startswith("zero_shot"):
            continue
        config = json.loads(sweep.read_text())["config"]
        corpus = "Essen" if config.get("essen_test") else "MTC"
        if args.corpus and corpus != args.corpus:
            continue
        judge = Judge.load("models/judge_ctrl.pth" if corpus == "Essen" else "models/judge.pth",
                           device)
        masked = load_masked_judge("models/masked_judge_ctrl.pth" if corpus == "Essen"
                                   else "models/masked_judge.pth", device)
        tokenizer = judge.tokenizer
        examples = test_examples(config, device)

        cols = {"causal": [], "causal+next": [], "two-sided": []}
        truth_all = []
        for ex in examples:
            tokens, spans = tokenizer.encode_with_spans(list(ex.draft_notes),
                                                        ex.reference.bar_starts)
            if tokens != ex.source:
                raise SystemExit(f"{sweep}: judge tokenizer disagrees with refiner encoding")
            labels = np.array(ex.source_labels)
            keep = labels != IGNORE_INDEX
            groups = event_groups(tokens, spans, tokenizer.eos_id)

            causal, joint = per_event(causal_surprisal(judge, tokens), groups, len(tokens))
            ahead = np.zeros(len(tokens))
            for i, g in enumerate(groups):
                ahead[g] = joint[i] + (joint[i + 1] if i + 1 < len(joint) else 0.0)
            two_sided = score_events(masked, tokens, groups, device=device)

            for name, s in (("causal", causal), ("causal+next", ahead), ("two-sided", two_sided)):
                cols[name].append(zscore(s, keep)[keep])
            truth_all.append((labels[keep] != 0).astype(np.int64))

        truth = np.concatenate(truth_all)
        saved = np.load(next(sweep.with_suffix("").glob("probs_alpha*_seed0_binary.npz")))
        if not np.array_equal(saved["is_edit"].astype(np.int64), truth):
            raise SystemExit(f"{sweep}: labels do not match the refiner run's saved labels")

        row = {"corpus": corpus, "source": config["sources"][0], "n_positions": len(truth)}
        for name, parts in cols.items():
            row[name] = metrics(np.concatenate(parts), truth)
        h = head[(corpus, row["source"])]
        row["trained head"] = {"ap_lift": h["ap_lift"], "auroc": h["auroc"],
                               "precision_at_k": h["precision_at_k"]}
        report.append(row)
        print(f"scored {corpus} {row['source']}", flush=True)

    order = {"D3": 0, "D1": 1, "D2-infill": 2}
    report.sort(key=lambda r: (r["corpus"] != "MTC", order.get(r["source"], 9)))
    names = ("causal", "causal+next", "two-sided", "trained head")
    print(f"\n{'':17s}" + "".join(f"{n:>22s}" for n in names))
    print(f"{'corpus source':17s}" + "".join(f"{'lift  AUROC  P@k':>22s}" for _ in names))
    for r in report:
        cells = "".join(
            f"{r[n]['ap_lift']:>7.1f}x {r[n]['auroc']:6.3f} {r[n]['precision_at_k']:6.3f}"
            for n in names)
        print(f"{r['corpus']:6s} {r['source']:10s} " + cells)

    out = root / (f"zero_shot_{args.corpus.lower()}.json" if args.corpus else "zero_shot.json")
    save_json({"rows": report}, out)
    print(f"\nzero-shot -> {out}")


if __name__ == "__main__":
    main()
