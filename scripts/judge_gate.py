"""
Gate edits on the two-sided judge instead of the trained head (plan step 5b).

    python scripts/judge_gate.py select results/refiner/headline_r2
    python scripts/judge_gate.py test   results/refiner/headline_r2

The untrained two-sided judge localizes better than every trained head
(contribution 11). This asks whether that ranking advantage survives the gate:
does editing where the *judge* is most surprised beat editing where the head is
most confident?

No retraining. Every model comes from the headline run's saved checkpoints, so
the decoder is held fixed and only the gate changes. Conditions:

  head/EA     head P(edit), edit-aware decoder -- the current method
  judge/EA    two-sided judge score, same edit-aware decoder
  judge/B1    two-sided judge score, B1 decoder (trained with no edit labels),
              so no component of the pipeline ever saw an edit label
  combo/EA    logistic combination of judge score and head logit, fitted on
              validation positions, edit-aware decoder

The judge score is the event's -log p(event | rest), standardized within the
melody, as in `zero_shot_localization.py`.

`select` chooses each condition's threshold on **D1 validation** only, by mean
gated Correction Cost over seeds -- the same global rule as step 2 -- and never
decodes test. `test` applies those thresholds unchanged to every source and
corpus.
"""
import argparse
import json
import statistics
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.drafts import empirical_priors  # noqa: E402
from src.edit_model import IGNORE_INDEX, EditAwareTransformer, EditModelConfig  # noqa: E402
from src.evaluation import generate_corrections_gated, sample_metrics  # noqa: E402
from src.judge import save_json  # noqa: E402
from src.masked_judge import event_groups, load_masked_judge, score_events  # noqa: E402
from src.refinement import RefinementDataset  # noqa: E402
from scripts.train_refiner import load_generator, make_drafts  # noqa: E402

GRIDS = {
    "head/EA": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "judge/EA": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    "judge/B1": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    "combo/EA": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
}
SEEDS = (0, 1, 2)
BATCH = 64


def build(config, split, device):
    """Rebuild the val or test examples a headline sweep used, from the cache."""
    splits = standard_splits(essen_test=config.get("essen_test") or 0.0)
    every = [m for group in splits.values() for m in group]
    tokenizer = build_tokenizer(every)
    priors = empirical_priors(every)
    generator = load_generator(config["generator"], tokenizer, device)
    args = Namespace(sources=config["sources"], temperature=config["temperature"],
                     per_melody=config["per_melody"])
    if split == "val":
        refs, seed = splits["judge_val"], 200
    else:
        refs, seed = splits[config.get("references") or "heldout"], 300
    drafts = make_drafts(refs, tokenizer, priors, generator, args, seed, split)
    return tokenizer, RefinementDataset(drafts, tokenizer).examples


def load_refiner(path, pad_id, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = EditAwareTransformer(EditModelConfig(**blob["config"]), pad_id=pad_id)
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()


def judge_scores(masked, tokenizer, examples, device):
    """Within-melody z-scored two-sided score per source position (0 elsewhere)."""
    out = []
    for ex in examples:
        tokens, spans = tokenizer.encode_with_spans(list(ex.draft_notes), ex.reference.bar_starts)
        assert tokens == ex.source
        raw = score_events(masked, tokens, event_groups(tokens, spans, tokenizer.eos_id),
                           device=device)
        keep = np.array(ex.source_labels) != IGNORE_INDEX
        v = raw[keep]
        z = (raw - v.mean()) / (v.std() if v.std() > 0 else 1.0)
        z[~keep] = 0.0
        out.append(z)
    return out


@torch.no_grad()
def head_probs(model, examples, pad_id, device):
    out = []
    for s in range(0, len(examples), BATCH):
        chunk = examples[s:s + BATCH]
        src = pad(chunk, pad_id, device)
        p = 1.0 - model.edit_head(model.encode(src)).softmax(-1)[..., 0]
        out += [p[i, :len(e.source)].cpu().numpy() for i, e in enumerate(chunk)]
    return out


def pad(chunk, pad_id, device):
    width = max(len(e.source) for e in chunk)
    src = torch.full((len(chunk), width), pad_id, dtype=torch.long)
    for i, e in enumerate(chunk):
        src[i, :len(e.source)] = torch.tensor(e.source)
    return src.to(device)


def fit_combo(z_list, logit_list, examples):
    """Logistic regression of edit labels on (judge z, head logit), validation only."""
    x, y = [], []
    for z, lg, ex in zip(z_list, logit_list, examples):
        labels = np.array(ex.source_labels)
        keep = labels != IGNORE_INDEX
        x.append(np.stack([z[keep], lg[keep]], 1)); y.append((labels[keep] != 0).astype(float))
    x = torch.tensor(np.concatenate(x), dtype=torch.float64)
    y = torch.tensor(np.concatenate(y), dtype=torch.float64)
    w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w[:2] + w[2], y)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy()


def combo_scores(z_list, logit_list, w):
    return [1 / (1 + np.exp(-(z * w[0] + lg * w[1] + w[2]))) for z, lg in zip(z_list, logit_list)]


def gated_run(model, tokenizer, examples, scores, threshold, device):
    """Gated CC per example, plus how many events the gate opened and how many were true edits."""
    costs, flags, hits = [], 0, 0
    for s in range(0, len(examples), BATCH):
        chunk = examples[s:s + BATCH]
        src = pad(chunk, tokenizer.pad_id, device)
        g = torch.zeros(src.shape)
        for i, e in enumerate(chunk):
            g[i, :len(e.source)] = torch.tensor(scores[s + i], dtype=torch.float)
        outs = generate_corrections_gated(model, tokenizer, src, threshold=threshold,
                                          gate_scores=g)
        for out, e, sc in zip(outs, chunk, scores[s:s + len(chunk)]):
            costs.append(sample_metrics(out, e.draft_notes, e.reference.notes)["correction_cost"])
            _, spans = tokenizer.encode_with_spans(list(e.draft_notes), e.reference.bar_starts)
            labels = e.source_labels
            for p_at, _ in spans:
                if sc[p_at] >= threshold:
                    flags += 1
                    hits += int(labels[p_at] not in (0, IGNORE_INDEX))
    return np.array(costs), flags, hits


def gate_inputs(cost_dir, tokenizer, examples, masked, device, combo_w=None):
    """Per-seed score lists for every condition, and the matching decoders."""
    z = judge_scores(masked, tokenizer, examples, device)
    per_seed = {}
    for seed in SEEDS:
        ea = load_refiner(cost_dir / f"model_alpha0.5_seed{seed}_binary.pt", tokenizer.pad_id, device)
        b1 = load_refiner(cost_dir / f"model_alpha0.0_seed{seed}_nohead.pt", tokenizer.pad_id, device)
        p = head_probs(ea, examples, tokenizer.pad_id, device)
        logit = [np.log(np.clip(q, 1e-6, 1)) - np.log(np.clip(1 - q, 1e-6, 1)) for q in p]
        per_seed[seed] = {"models": {"EA": ea, "B1": b1}, "head": p, "judge": z, "logit": logit}
    return per_seed


def select(root, device, only=None, grid=None):
    """With `only` and `grid`, evaluate extra thresholds for one condition and
    merge them into the existing selection -- for when the optimum lands on a
    grid edge. Still validation-only."""
    sweep = root / "mtc_d1.json"
    config = json.loads(sweep.read_text())["config"]
    tokenizer, examples = build(config, "val", device)
    masked = load_masked_judge("models/masked_judge.pth", device)
    per_seed = gate_inputs(sweep.with_suffix(""), tokenizer, examples, masked, device)
    b3 = float(np.mean([sample_metrics(list(e.draft_notes), e.draft_notes, e.reference.notes)
                        ["correction_cost"] for e in examples]))
    print(f"D1 validation: {len(examples)} drafts, B3 CC {b3:.4f}")

    weights = {s: fit_combo(d["judge"], d["logit"], examples) for s, d in per_seed.items()}
    previous = root / "judge_gate_selection.json"
    table = json.loads(previous.read_text())["conditions"] if only else {}
    grids = {only: grid} if only else GRIDS
    for cond, grid in grids.items():
        gate, decoder = cond.split("/")
        rows = []
        for thr in grid:
            ccs = []
            for seed, d in per_seed.items():
                scores = (d["head"] if gate == "head" else d["judge"] if gate == "judge"
                          else combo_scores(d["judge"], d["logit"], weights[seed]))
                costs, _, _ = gated_run(d["models"][decoder], tokenizer, examples, scores, thr, device)
                ccs.append(costs.mean())
            gains = [100 * (b3 - c) / b3 for c in ccs]
            rows.append({"threshold": thr, "val_cc": float(np.mean(ccs)),
                         "val_gain_pct": statistics.mean(gains),
                         "val_gain_sd": statistics.stdev(gains)})
            print(f"  {cond:9s} thr {thr:4.2f}  val gain {rows[-1]['val_gain_pct']:+6.2f}% "
                  f"(sd {rows[-1]['val_gain_sd']:.2f})", flush=True)
        if only:
            rows = sorted(table[cond]["grid"] + rows, key=lambda r: r["threshold"])
        best = min(rows, key=lambda r: r["val_cc"])
        table[cond] = {"selected": best, "grid": rows}
        print(f"  -> {cond}: threshold {best['threshold']} ({best['val_gain_pct']:+.2f}%)\n")
    save_json({"val_b3_cc": b3, "conditions": table,
               "combo_weights": {str(s): w.tolist() for s, w in weights.items()}},
              root / "judge_gate_selection.json")


def test(root, device):
    selection = json.loads((root / "judge_gate_selection.json").read_text())
    report = []
    for sweep in sorted(root.glob("*.json")):
        if not sweep.name.startswith(("mtc_", "essen_")):
            continue
        config = json.loads(sweep.read_text())["config"]
        corpus = "Essen" if config.get("essen_test") else "MTC"
        masked = load_masked_judge("models/masked_judge_ctrl.pth" if corpus == "Essen"
                                   else "models/masked_judge.pth", device)
        tokenizer, examples = build(config, "test", device)
        per_seed = gate_inputs(sweep.with_suffix(""), tokenizer, examples, masked, device)
        b3_costs = np.array([sample_metrics(list(e.draft_notes), e.draft_notes, e.reference.notes)
                             ["correction_cost"] for e in examples])
        b3 = b3_costs.mean()

        row = {"corpus": corpus, "source": config["sources"][0], "n": len(examples)}
        # The selected conditions, plus the headline's own setting (head gate at
        # tau 0.8) as a consistency check: it must reproduce the headline run.
        runs = [(c, b["selected"]["threshold"]) for c, b in selection["conditions"].items()]
        runs.append(("head/EA@0.8", 0.8))
        for cond, thr in runs:
            gate, decoder = cond.split("@")[0].split("/")
            gains, wins, losses, flags, hits = [], [], [], 0, 0
            for seed, d in per_seed.items():
                if gate == "combo":
                    # Weights fitted once, on D1 validation, like every other
                    # choice here -- never refitted per source.
                    w = np.array(selection["combo_weights"][str(seed)])
                    scores = combo_scores(d["judge"], d["logit"], w)
                else:
                    scores = d["head"] if gate == "head" else d["judge"]
                costs, f, h = gated_run(d["models"][decoder], tokenizer, examples, scores, thr, device)
                gains.append(100 * (b3 - costs.mean()) / b3)
                wins.append(float(np.mean(costs < b3_costs)))
                losses.append(float(np.mean(costs > b3_costs)))
                flags += f; hits += h
            row[cond] = {"threshold": thr, "gain_pct": statistics.mean(gains),
                         "gain_sd": statistics.stdev(gains), "per_seed": gains,
                         "seeds_positive": sum(g > 0 for g in gains),
                         "win": statistics.mean(wins), "loss": statistics.mean(losses),
                         "flags_per_seed": flags / len(SEEDS),
                         "gate_precision": hits / flags if flags else float("nan")}
        report.append(row)
        print(f"scored {corpus} {row['source']}", flush=True)

    order = {"D3": 0, "D1": 1, "D2-infill": 2}
    report.sort(key=lambda r: (r["corpus"] != "MTC", order.get(r["source"], 9)))
    conds = list(selection["conditions"]) + ["head/EA@0.8"]
    print(f"\n{'corpus':6s} {'source':10s}" + "".join(f"{c:>26s}" for c in conds))
    for r in report:
        cells = "".join(
            f"{r[c]['gain_pct']:+8.2f}% ±{r[c]['gain_sd']:4.2f} {r[c]['seeds_positive']}/3 "
            f"P{r[c]['gate_precision']:5.2f}" for c in conds)
        print(f"{r['corpus']:6s} {r['source']:10s}  {cells}")
    save_json({"selection": selection, "rows": report}, root / "judge_gate.json")
    print(f"\njudge gate -> {root / 'judge_gate.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "test"])
    parser.add_argument("directory")
    parser.add_argument("--only", help="select: one condition to extend, e.g. combo/EA")
    parser.add_argument("--grid", type=float, nargs="*", help="select: extra thresholds")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.stage == "select":
        select(Path(args.directory), device, args.only, args.grid)
    else:
        test(Path(args.directory), device)


if __name__ == "__main__":
    main()
