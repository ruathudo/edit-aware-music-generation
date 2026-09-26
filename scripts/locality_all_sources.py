"""
Locality strata for every draft source, so D2 can be compared at matched run length.

    python scripts/locality_all_sources.py

`control_strata.py` banded only D3 by the longest run of consecutively edited
events. That showed contiguity costs localization within D3, but compared D3's
long runs against *all* of D2. This applies the same banding to D1, D2 and D3
on both corpora (headline test drafts; head AUROC over 3 seeds, two-sided judge
AUROC), so a D2 run of three or more is compared with a D3 run of three or more,
and isolated D2 changes with isolated corruption.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judge import save_json  # noqa: E402
from src.masked_judge import load_masked_judge  # noqa: E402
from scripts.control_strata import drafts_for, longest_run, show, stratify  # noqa: E402

ROOT = Path("results/refiner/headline_r2")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for corpus in ("mtc", "essen"):
        masked = load_masked_judge("models/masked_judge_ctrl.pth" if corpus == "essen"
                                   else "models/masked_judge.pth", device)
        for name in ("d1", "d2infill", "d3"):
            sweep = ROOT / f"{corpus}_{name}.json"
            config = json.loads(sweep.read_text())["config"]
            tokenizer, examples, _ = drafts_for(config, device)
            runs = np.array([longest_run(e) for e in examples])
            idx = np.arange(len(examples))
            bands = [("1 event", idx[runs == 1]), ("2 adjacent", idx[runs == 2]),
                     ("3+ adjacent", idx[runs >= 3])]
            rows = stratify(examples, bands, sweep.with_suffix(""), tokenizer, masked, device)
            show(f"{config['sources'][0]}, {corpus.upper()} locality", rows)
            out.append({"corpus": corpus, "source": config["sources"][0], "bands": rows})
    path = Path("results/refiner/controls/locality_all_sources.json")
    save_json(out, path)
    print(f"\nlocality strata -> {path}")


if __name__ == "__main__":
    main()
