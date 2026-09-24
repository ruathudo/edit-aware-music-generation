"""
Musical quality of the refiner's outputs, not only their Correction Cost.

    python scripts/output_quality.py results/refiner/headline_r2

The validator found Correction Cost and plausibility close to orthogonal, so a
CC gain alone does not show the outputs are better *music*. This scores the
decoded outputs saved by `--save-artifacts` with the judge `J` and the
validator's musical criteria, each relative to the reference, exactly as the
validator scored drafts:

- **dNLL** -- per-token NLL of the output minus the reference's. The draft's
  dNLL is the damage; an output that repairs it moves toward 0.
- **recovered** -- (dNLL_draft - dNLL_output) / dNLL_draft: the share of the
  draft's implausibility the output removes. Negative means it added more.
- scale conformance and metric alignment, relative to the reference, and
  pitch-class / interval histogram distance to it.

Gated decoding leaves most melodies untouched, which dilutes every mean toward
the draft's. So the edited subset -- melodies where the gated output differs
from the draft -- is reported separately, with the share of those edits that
made the melody *less* plausible.

Conditions: B3 (the draft itself), B1 (no head, free decoding), EA free and EA
gated (alpha*, tau*). Same-corpus runs are scored by `judge_ctrl.pth`, the judge
retrained without the Essen test slice, so the judge never saw a reference.
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
from src.judge import Judge, save_json  # noqa: E402
from src.refinement import RefinementDataset  # noqa: E402
from scripts.train_refiner import load_generator, make_drafts  # noqa: E402

METRICS = ("delta_nll", "delta_scale_conformance", "delta_metric_alignment",
           "pitch_class_distance", "interval_distance")


def as_notes(seq):
    return [tuple(n) for n in seq]


def test_examples(config, device):
    """Rebuild the exact test examples the sweep evaluated, from the draft cache."""
    splits = standard_splits(essen_test=config.get("essen_test") or 0.0)
    every = [m for group in splits.values() for m in group]
    tokenizer = build_tokenizer(every)
    priors = empirical_priors(every)
    generator = load_generator(config["generator"], tokenizer, device)
    args = Namespace(sources=config["sources"], temperature=config["temperature"],
                     per_melody=config["per_melody"])
    refs = splits[config.get("references") or "heldout"]
    drafts = make_drafts(refs, tokenizer, priors, generator, args, 300, "test")
    # The dataset drops drafts the tokenizer cannot encode; outputs follow it.
    return RefinementDataset(drafts, tokenizer).examples


def score(judge, outputs, examples):
    scores = judge.score_drafts([(out, ex.reference) for out, ex in zip(outputs, examples)])
    return [s.to_dict() for s in scores]


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return statistics.mean(xs) if xs else float("nan")


def summarize(scored, draft_scored, changed=None):
    idx = range(len(scored)) if changed is None else [i for i in range(len(scored)) if changed[i]]
    out = {m: mean([scored[i][m] for i in idx]) for m in METRICS}
    d_draft = mean([draft_scored[i]["delta_nll"] for i in idx])
    out["recovered"] = (d_draft - out["delta_nll"]) / d_draft if d_draft else float("nan")
    out["n"] = len(idx)
    if changed is not None:
        worse = [scored[i]["delta_nll"] > draft_scored[i]["delta_nll"] for i in idx]
        out["share_less_plausible"] = float(np.mean(worse)) if worse else float("nan")
        out["draft_delta_nll"] = d_draft
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    root = Path(args.directory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    judges = {}
    order = {"D3": 0, "D1": 1, "D2-infill": 2}
    report = []

    for sweep in sorted(root.glob("*.json")):
        if sweep.name in ("headline.json", "localization.json", "quality.json"):
            continue
        blob = json.loads(sweep.read_text())
        config = blob["config"]
        corpus = "Essen" if config.get("essen_test") else "MTC"
        judge_path = "models/judge_ctrl.pth" if corpus == "Essen" else "models/judge.pth"
        judge = judges.setdefault(judge_path, Judge.load(judge_path, device))
        (tau,) = config["gated_thresholds"]
        alpha = max(config["alphas"])
        cost_dir = sweep.with_suffix("")

        examples = test_examples(config, device)
        # Outputs were written in dataset order; confirm before trusting them.
        probe = json.loads((cost_dir / f"outputs_alpha{alpha}_seed0_binary.json").read_text())
        if len(probe) != len(examples) or any(
            as_notes(r["draft"]) != list(e.draft_notes)
            or as_notes(r["reference"]) != list(e.reference.notes)
            for r, e in zip(probe, examples)
        ):
            raise SystemExit(f"{sweep}: rebuilt test set does not match saved outputs")

        draft_scored = score(judge, [e.draft_notes for e in examples], examples)
        row = {"corpus": corpus, "source": config["sources"][0], "n": len(examples),
               "judge": judge_path, "draft": summarize(draft_scored, draft_scored)}
        per_seed = {"b1_free": [], "ea_free": [], "ea_gated": [], "ea_gated_edited": []}

        for seed in config["seeds"]:
            b1 = json.loads((cost_dir / f"outputs_alpha0.0_seed{seed}_nohead.json").read_text())
            ea = json.loads((cost_dir / f"outputs_alpha{alpha}_seed{seed}_binary.json").read_text())
            gated = [as_notes(r[f"gated_{tau}"]) for r in ea]
            changed = [g != list(e.draft_notes) for g, e in zip(gated, examples)]
            per_seed["b1_free"].append(summarize(
                score(judge, [as_notes(r["free"]) for r in b1], examples), draft_scored))
            per_seed["ea_free"].append(summarize(
                score(judge, [as_notes(r["free"]) for r in ea], examples), draft_scored))
            gated_scored = score(judge, gated, examples)
            per_seed["ea_gated"].append(summarize(gated_scored, draft_scored))
            per_seed["ea_gated_edited"].append(
                summarize(gated_scored, draft_scored, changed) if any(changed) else None)

        for key, blocks in per_seed.items():
            blocks = [b for b in blocks if b]
            if not blocks:
                row[key] = None
                continue
            row[key] = {m: mean([b[m] for b in blocks]) for m in blocks[0]}
            row[key]["recovered_per_seed"] = [b["recovered"] for b in blocks]
        report.append(row)
        print(f"scored {corpus} {row['source']}", flush=True)

    report.sort(key=lambda r: (r["corpus"] != "MTC", order.get(r["source"], 9)))
    print(f"\n{'corpus':6s} {'source':10s} {'condition':14s} {'dNLL':>7s} {'recovered':>10s} "
          f"{'dScale':>7s} {'dMetric':>8s} {'pcDist':>7s} {'intDist':>8s}")
    for r in report:
        for key, name in (("draft", "B3 draft"), ("b1_free", "B1 free"),
                          ("ea_free", "EA free"), ("ea_gated", "EA gated")):
            b = r[key]
            rec = "" if key == "draft" else f"{b['recovered']:+9.1%}"
            print(f"{r['corpus']:6s} {r['source']:10s} {name:14s} {b['delta_nll']:+7.3f} {rec:>10s} "
                  f"{b['delta_scale_conformance']:+7.3f} {b['delta_metric_alignment']:+8.3f} "
                  f"{b['pitch_class_distance']:7.3f} {b['interval_distance']:8.3f}")
        e = r["ea_gated_edited"]
        if e:
            print(f"{'':18s}edited subset: n={e['n']:.0f}/seed  draft dNLL {e['draft_delta_nll']:+.3f} "
                  f"-> gated {e['delta_nll']:+.3f}  recovered {e['recovered']:+.1%}  "
                  f"less plausible {e['share_less_plausible']:.1%}")
        else:
            print(f"{'':18s}edited subset: none (gate never opened)")
        print()

    out = root / "quality.json"
    save_json({"rows": report}, out)
    print(f"quality -> {out}")


if __name__ == "__main__":
    main()
