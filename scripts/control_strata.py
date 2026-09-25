"""
The two stratified controls, re-run on 3 seeds with threshold-free localization.

    python scripts/control_strata.py results/refiner/headline_r2

Both earlier versions (`degradation_strata.py`, `locality_strata.py`) were one
seed, at the old tau=0.9 setting, and read argmax precision -- undefined for a
head that barely flags, which is what the D2 head does under the selected
setting. This version needs no training: it scores drafts with the headline
run's saved checkpoints (3 seeds) and with the two-sided judge, and reports AUROC
and AP lift per band.

**Degradation strata (D2).** "Maybe most generated drafts are not degraded, so
there is nothing to find." Drafts are banded by the judge's dNLL; if the claim
held, certified-degraded drafts (dNLL > tau) would be far more localizable. The
old degraded band had n=28, so MTC uses a *fresh* draft set (15 per melody, a
seed never used before), and Essen adds its 5k-draft test set. tau is recomputed
for each judge exactly as the validator does it: the 95th percentile of dNLL over
human tune-family variant pairs.

**Locality strata (D3).** Drafts banded by their longest run of consecutively
edited events, computed from the labels in event space.
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
from src.edit_model import IGNORE_INDEX  # noqa: E402
from src.judge import Judge, save_json  # noqa: E402
from src.masked_judge import load_masked_judge  # noqa: E402
from src.refinement import RefinementDataset  # noqa: E402
from scripts.judge_gate import head_probs, judge_scores, load_refiner  # noqa: E402
from scripts.localization_report import auroc, average_precision  # noqa: E402
from scripts.train_judge import variant_pairs  # noqa: E402
from scripts.train_refiner import load_generator, make_drafts  # noqa: E402

SEEDS = (0, 1, 2)


def drafts_for(config, device, per_melody=None, seed=300, tag="test"):
    splits = standard_splits(essen_test=config.get("essen_test") or 0.0)
    every = [m for group in splits.values() for m in group]
    tokenizer = build_tokenizer(every)
    priors = empirical_priors(every)
    generator = load_generator(config["generator"], tokenizer, device)
    args = Namespace(sources=config["sources"], temperature=config["temperature"],
                     per_melody=per_melody or config["per_melody"])
    refs = splits[config.get("references") or "heldout"]
    drafts = make_drafts(refs, tokenizer, priors, generator, args, seed, tag)
    return tokenizer, RefinementDataset(drafts, tokenizer).examples, splits


def tau_for(judge, splits):
    """The validator's rule: 95th percentile of dNLL over human variant pairs."""
    pairs = variant_pairs(splits["heldout"], seed=7)  # the validator's seed
    deltas = np.array([s.delta_nll for s in judge.score_drafts(pairs)])
    return float(np.percentile(deltas, 95)), len(pairs)


def flat(examples, per_example):
    """Concatenate per-position arrays over labelled positions, with draft index."""
    scores, truth, owner = [], [], []
    for i, (ex, s) in enumerate(zip(examples, per_example)):
        labels = np.array(ex.source_labels)
        keep = labels != IGNORE_INDEX
        scores.append(np.asarray(s)[keep]); truth.append((labels[keep] != 0).astype(int))
        owner.append(np.full(keep.sum(), i))
    return np.concatenate(scores), np.concatenate(truth), np.concatenate(owner)


def band_metrics(scores, truth, owner, members):
    mask = np.isin(owner, members)
    y, s = truth[mask], scores[mask]
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return {"auroc": auroc(s, y), "ap_lift": average_precision(s, y) / y.mean(),
            "base_rate": float(y.mean())}


def longest_run(example):
    """Longest run of consecutively edited events (EOS slot excluded)."""
    labels = [l for l in example.source_labels if l != IGNORE_INDEX]
    events = labels[:-1:2] if len(labels) % 2 == 1 else labels[::2]
    best = run = 0
    for label in events:
        run = run + 1 if label != 0 else 0
        best = max(best, run)
    return best


def stratify(examples, bands, cost_dir, tokenizer, masked, device):
    """Head (3 seeds) and two-sided judge AUROC / AP lift per band of drafts."""
    judge_flat = flat(examples, judge_scores(masked, tokenizer, examples, device))
    heads = []
    for seed in SEEDS:
        model = load_refiner(cost_dir / f"model_alpha0.5_seed{seed}_binary.pt", tokenizer.pad_id, device)
        heads.append(flat(examples, head_probs(model, examples, tokenizer.pad_id, device)))
    rows = []
    for name, members in bands:
        members = np.array(members)
        row = {"band": name, "n_drafts": int(len(members))}
        per_seed = [band_metrics(*h, members) for h in heads]
        per_seed = [m for m in per_seed if m]
        if per_seed:
            for key in ("auroc", "ap_lift"):
                vals = [m[key] for m in per_seed]
                row[f"head_{key}"] = statistics.mean(vals)
                row[f"head_{key}_sd"] = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            row["base_rate"] = per_seed[0]["base_rate"]
        j = band_metrics(*judge_flat, members)
        if j:
            row["judge_auroc"], row["judge_ap_lift"] = j["auroc"], j["ap_lift"]
        rows.append(row)
    return rows


def show(title, rows):
    print(f"\n{title}")
    print(f"  {'band':34s} {'drafts':>7s} {'base':>6s} {'head AUROC':>15s} {'head lift':>10s} "
          f"{'judge AUROC':>12s} {'judge lift':>11s}")
    for r in rows:
        if "head_auroc" not in r:
            print(f"  {r['band']:34s} {r['n_drafts']:7d}   (no edited positions)")
            continue
        print(f"  {r['band']:34s} {r['n_drafts']:7d} {r['base_rate']:6.3f} "
              f"{r['head_auroc']:7.3f} ±{r['head_auroc_sd']:5.3f} {r['head_ap_lift']:9.1f}x "
              f"{r.get('judge_auroc', float('nan')):12.3f} {r.get('judge_ap_lift', float('nan')):10.1f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--extra-per-melody", type=int, default=15)
    args = parser.parse_args()
    root = Path(args.directory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"degradation": [], "locality": []}

    # ---- degradation strata on D2
    for sweep_name, fresh in (("mtc_d2infill.json", True), ("essen_d2infill.json", False)):
        sweep = root / sweep_name
        config = json.loads(sweep.read_text())["config"]
        corpus = "Essen" if config.get("essen_test") else "MTC"
        judge = Judge.load("models/judge_ctrl.pth" if corpus == "Essen" else "models/judge.pth", device)
        masked = load_masked_judge("models/masked_judge_ctrl.pth" if corpus == "Essen"
                                   else "models/masked_judge.pth", device)
        if fresh:
            tokenizer, examples, splits = drafts_for(config, device, args.extra_per_melody,
                                                     seed=301, tag="strata")
        else:
            tokenizer, examples, splits = drafts_for(config, device)
        tau, n_pairs = tau_for(judge, splits)
        deltas = np.array([s.delta_nll for s in
                           judge.score_drafts([(list(e.draft_notes), e.reference) for e in examples])])
        median = float(np.median(deltas[deltas > 0]))
        idx = np.arange(len(examples))
        bands = [
            ("not degraded (dNLL <= 0)", idx[deltas <= 0]),
            (f"mild (0 < dNLL <= {median:.3f})", idx[(deltas > 0) & (deltas <= median)]),
            (f"moderate ({median:.3f} < dNLL <= tau)", idx[(deltas > median) & (deltas <= tau)]),
            (f"degraded (dNLL > tau={tau:.3f})", idx[deltas > tau]),
        ]
        rows = stratify(examples, bands, sweep.with_suffix(""), tokenizer, masked, device)
        block = {"corpus": corpus, "n_drafts": len(examples), "fresh_drafts": fresh,
                 "tau": tau, "tau_pairs": n_pairs, "bands": rows}
        out["degradation"].append(block)
        show(f"D2 degradation strata, {corpus} ({len(examples)} drafts"
             f"{', fresh set' if fresh else ''}; tau {tau:.3f} from {n_pairs} variant pairs)", rows)

    # ---- locality strata on D3
    for sweep_name in ("mtc_d3.json", "essen_d3.json"):
        sweep = root / sweep_name
        config = json.loads(sweep.read_text())["config"]
        corpus = "Essen" if config.get("essen_test") else "MTC"
        masked = load_masked_judge("models/masked_judge_ctrl.pth" if corpus == "Essen"
                                   else "models/masked_judge.pth", device)
        tokenizer, examples, _ = drafts_for(config, device)
        runs = np.array([longest_run(e) for e in examples])
        idx = np.arange(len(examples))
        bands = [
            ("point-like (1 event)", idx[runs == 1]),
            ("2 adjacent events", idx[runs == 2]),
            ("span-like (3+ events)", idx[runs >= 3]),
        ]
        rows = stratify(examples, bands, sweep.with_suffix(""), tokenizer, masked, device)
        out["locality"].append({"corpus": corpus, "n_drafts": len(examples),
                                "no_edit_drafts": int((runs == 0).sum()), "bands": rows})
        show(f"D3 locality strata, {corpus} ({len(examples)} drafts, "
             f"{(runs == 0).sum()} with no edited event)", rows)

    path = Path("results/refiner/controls/strata.json")
    save_json(out, path)
    print(f"\nstrata -> {path}")


if __name__ == "__main__":
    main()
