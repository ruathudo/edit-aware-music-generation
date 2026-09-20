"""
The draft-source ablation (spec 7): does edit-awareness help more when the
errors are realistic?

    python scripts/draft_source_comparison.py \
        --sweeps results/refiner/sweep_seed0.json results/refiner/sweep_d1_seed0.json

Each sweep is reported against its *own* B3, because the draft sources differ in
how much they change the melody and so have different copy-input baselines.
Comparing raw Correction Cost across sources would mostly measure that
difference rather than anything about the models.

The quantity of interest is `lift`: how much of the draft's error the model
actually removes.

    lift = (draft_cost - model_cost) / draft_cost

Positive means the model improved the draft; zero means it reproduced it;
negative means it made things worse. A Correction Cost that looks strong in
isolation can still have lift <= 0, which is the failure the copy-input baseline
exists to expose.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import paired_comparison  # noqa: E402
from src.judge import save_json  # noqa: E402


def summarize(path: Path) -> dict:
    sweep = json.loads(path.read_text())
    directory = path.with_suffix("")
    b3_costs = np.load(directory / "costs_b3.npy") if (directory / "costs_b3.npy").exists() else None
    b3_mean = sweep["b3_copy_input"]["correction_cost_mean"]

    rows = []
    for r in sweep["rows"]:
        costs_path = directory / f"costs_{r['label']}.npy"
        costs = np.load(costs_path) if costs_path.exists() else None

        entry = {
            "label": r["label"],
            "alpha": r["alpha"],
            "use_edit_head": r["use_edit_head"],
            "cc": r["correction_cost"],
            "draft_cost": r["draft_cost"],
            "fidelity": r["fidelity_per_note"],
            "lift": (r["draft_cost"] - r["correction_cost"]) / max(r["draft_cost"], 1e-9),
            "macro_f1": r.get("edit_scores", {}).get("macro_f1"),
            "localization_f1": r.get("edit_scores", {}).get("localization", {}).get("f1"),
        }
        if b3_costs is not None and costs is not None and len(b3_costs) == len(costs):
            comparison = paired_comparison(b3_costs, costs, n_boot=2000)
            entry["beats_b3_pct"] = comparison["win_rate"]
            entry["vs_b3_p"] = comparison["wilcoxon"]["p_value"]
        rows.append(entry)

    return {"source": path.stem, "b3": b3_mean, "config": sweep["config"], "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", nargs="+", required=True)
    parser.add_argument("--out", default="results/refiner/draft_source_comparison.json")
    args = parser.parse_args()

    summaries = [summarize(Path(p)) for p in args.sweeps]

    for summary in summaries:
        sources = summary["config"].get("sources", ["?"])
        print(f"\n=== {summary['source']}  (drafts: {', '.join(sources)}) ===")
        print(f"B3 copy-input CC {summary['b3']:.3f}   "
              f"— a model must beat this to be correcting anything\n")
        header = (f"{'condition':22s} {'CC':>7s} {'lift':>8s} {'fidelity':>9s} "
                  f"{'macroF1':>8s} {'locF1':>7s} {'beatsB3':>8s}")
        print(header)
        print("-" * len(header))
        for r in summary["rows"]:
            f1 = r["macro_f1"] if r["macro_f1"] is not None else float("nan")
            loc = r["localization_f1"] if r["localization_f1"] is not None else float("nan")
            beats = r.get("beats_b3_pct", float("nan"))
            print(f"{r['label']:22s} {r['cc']:7.3f} {r['lift']:+8.2%} {r['fidelity']:9.4f} "
                  f"{f1:8.3f} {loc:7.3f} {beats:7.1f}%")

    if len(summaries) > 1:
        print("\n=== best condition per draft source ===")
        print(f"{'source':28s} {'best alpha':>11s} {'lift':>8s} {'macroF1':>8s} {'fidelity':>9s}")
        print("-" * 68)
        for summary in summaries:
            head_rows = [r for r in summary["rows"] if r["use_edit_head"]]
            if not head_rows:
                continue
            best = max(head_rows, key=lambda r: r["lift"])
            sources = ",".join(summary["config"].get("sources", ["?"]))
            f1 = best["macro_f1"] if best["macro_f1"] is not None else float("nan")
            print(f"{sources:28s} {best['alpha']:11} {best['lift']:+8.2%} "
                  f"{f1:8.3f} {best['fidelity']:9.4f}")

    save_json({"summaries": summaries}, args.out)
    print(f"\ncomparison -> {args.out}")


if __name__ == "__main__":
    main()
