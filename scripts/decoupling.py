"""
Is Correction Cost a measure of musical plausibility? A correlation, not two clusters.

    python scripts/decoupling.py

Contribution 2 claims CC and plausibility are near-orthogonal, but until now it
rested on two cluster means: human variants far apart and plausible, synthetic
drafts close and implausible. This measures the relation directly, per draft:

- **surface change**: Correction Cost / reference length, as in the validator;
- **dNLL**: the judge's per-token NLL of the draft minus the reference's.

Spearman rho within each source, over synthetic sources pooled, and over all
sources pooled. 95% intervals bootstrap over *tune families*, not drafts: each
reference contributes several drafts, D4 pairs share melodies within a family,
and none of these are independent.

Sources: D4 human variant pairs (the validator's pairs, seed 7) and the MTC test
drafts the refiner was evaluated on (D1, D2-infill, D3), all scored by `J`.
"""
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import correction_cost  # noqa: E402
from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.drafts import empirical_priors  # noqa: E402
from src.judge import Judge, save_json  # noqa: E402
from scripts.train_judge import variant_pairs  # noqa: E402
from scripts.train_refiner import load_generator, make_drafts  # noqa: E402

BOOT = 2000


def ranks(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def cluster_bootstrap(surface, delta, cluster, rng):
    """95% CI for rho, resampling whole tune families."""
    ids = np.unique(cluster)
    members = {c: np.flatnonzero(cluster == c) for c in ids}
    stats = []
    for _ in range(BOOT):
        pick = np.concatenate([members[c] for c in rng.choice(ids, len(ids))])
        stats.append(spearman(surface[pick], delta[pick]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def collect():
    """Per source: (surface change, dNLL, tune family) arrays, scored by `J`."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    judge = Judge.load("models/judge.pth", device)
    splits = standard_splits()
    every = [m for g in splits.values() for m in g]
    tokenizer = build_tokenizer(every)
    priors = empirical_priors(every)
    generator = load_generator("models/generator.pth", tokenizer, device)

    data = {}
    pairs = variant_pairs(splits["heldout"], seed=7)          # the validator's pairs
    data["D4 human variants"] = pairs              # already (variant notes, reference)
    for source in ("D1", "D2-infill", "D3"):
        args = Namespace(sources=[source], temperature=1.0, per_melody=3)
        drafts = make_drafts(splits["heldout"], tokenizer, priors, generator, args, 300, "test")
        data[source] = [(d.notes, d.reference) for d in drafts]

    per = {}
    for name, items in data.items():
        scores = judge.score_drafts(items)
        delta = np.array([s.delta_nll for s in scores])
        surface = np.array([correction_cost(list(d), list(r.notes)) / max(len(r.notes), 1)
                            for d, r in items])
        # Tune family: D4 pairs are built within families, and one conservative
        # unit for every set keeps the intervals comparable (26 families).
        cluster = np.array([r.tunefamily for _, r in items])
        per[name] = (surface, delta, cluster)
    return per


def main():
    rng = np.random.default_rng(0)
    per = collect()

    def row(name, surface, delta, cluster):
        lo, hi = cluster_bootstrap(surface, delta, cluster, rng)
        return {"set": name, "n": int(len(delta)), "n_families": int(len(np.unique(cluster))),
                "rho": spearman(surface, delta), "ci95": [lo, hi],
                "surface_median": float(np.median(surface)),
                "delta_nll_median": float(np.median(delta))}

    rows = [row(n, *v) for n, v in per.items()]
    synth = [per[s] for s in ("D1", "D2-infill", "D3")]
    rows.append(row("synthetic pooled (D1+D2+D3)",
                    *[np.concatenate([v[i] for v in synth]) for i in range(3)]))
    rows.append(row("all pooled (D1+D2+D3+D4)",
                    *[np.concatenate([v[i] for v in per.values()]) for i in range(3)]))

    # Matched distance: do human variants that change as *little* as synthetic
    # drafts look as implausible as them? Compare within the overlapping range.
    s4, d4, _ = per["D4 human variants"]
    s_syn = np.concatenate([v[0] for v in synth]); d_syn = np.concatenate([v[1] for v in synth])
    cap = float(np.percentile(s_syn, 95))
    near = s4 <= cap
    matched = {"surface_cap": cap, "d4_pairs_within_cap": int(near.sum()),
               "d4_delta_nll_median": float(np.median(d4[near])) if near.any() else float("nan"),
               "synthetic_delta_nll_median": float(np.median(d_syn[s_syn <= cap]))}

    print(f"{'set':32s} {'n':>6s} {'fams':>5s} {'rho':>7s} {'95% CI (by family)':>24s} "
          f"{'surface':>8s} {'dNLL':>7s}")
    for r in rows:
        print(f"{r['set']:32s} {r['n']:6d} {r['n_families']:5d} {r['rho']:+7.3f} "
              f"   [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]      "
              f"{r['surface_median']:8.3f} {r['delta_nll_median']:+7.3f}")
    print(f"\nmatched distance (surface <= {cap:.3f}, the synthetic 95th percentile): "
          f"{matched['d4_pairs_within_cap']} human pairs, median dNLL "
          f"{matched['d4_delta_nll_median']:+.3f} vs synthetic {matched['synthetic_delta_nll_median']:+.3f}")

    out = Path("results/judge/decoupling.json")
    save_json({"rows": rows, "matched_distance": matched}, out)
    print(f"\ndecoupling -> {out}")


if __name__ == "__main__":
    main()
