"""
Are "failed" repairs wrong, or just different? Plausibility of what the gate writes.

    python scripts/repair_plausibility.py

`repair_accuracy.py` counts a repair as correct only if it equals the reference
note exactly -- the distance-to-a-reference logic the paper argues cannot tell an
error from a valid alternative. This asks what the judges make of the repairs:

  melody level   among melodies the gate changed, J's dNLL (vs the reference) of
                 the output against that of the draft: the share of the draft's
                 implausibility removed, and how many melodies got more / less
                 plausible.
  note level     at every *mismatched* repair -- a true substitution where the
                 output differs from the reference note -- the two-sided judge's
                 surprisal of three candidates for that slot, in the output's
                 context: the draft's note, the decoder's note, and the
                 reference's note. Masking the slot makes the context identical
                 for all three, so one forward pass scores them all.

If mismatched repairs on generated drafts are as plausible as the reference
note, the decoder is repairing *differently*, not failing; if they are no better
than the draft's note, it is failing.
"""
import json
import statistics
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import align  # noqa: E402
from src.data import build_tokenizer, standard_splits  # noqa: E402
from src.drafts import empirical_priors  # noqa: E402
from src.judge import Judge, save_json  # noqa: E402
from src.masked_judge import load_masked_judge  # noqa: E402
from src.refinement import RefinementDataset  # noqa: E402
from scripts.train_refiner import load_generator, make_drafts  # noqa: E402

DIAG = 0
CONDITIONS = [
    ("MTC", "D1", "results/refiner/headline_r2/mtc_d1"),
    ("MTC", "D3", "results/refiner/headline_r2/mtc_d3"),
    ("MTC", "D2 ep2", "results/refiner/headline_ep2/mtc_d2infill"),
    ("Essen", "D1", "results/refiner/headline_r2/essen_d1"),
    ("Essen", "D3", "results/refiner/headline_r2/essen_d3"),
    ("Essen", "D2 ep2", "results/refiner/headline_ep2/essen_d2infill"),
]


def test_examples(config, device):
    """The sweep's test examples, keyed on its generator (see make_drafts)."""
    splits = standard_splits(essen_test=config.get("essen_test") or 0.0)
    every = [m for group in splits.values() for m in group]
    tokenizer = build_tokenizer(every)
    args = Namespace(sources=config["sources"], temperature=config["temperature"],
                     per_melody=config["per_melody"], generator=config["generator"])
    generator = load_generator(config["generator"], tokenizer, device)
    drafts = make_drafts(splits[config.get("references") or "heldout"], tokenizer,
                         empirical_priors(every), generator, args, 300, "test")
    return tokenizer, RefinementDataset(drafts, tokenizer).examples


@torch.no_grad()
def slot_surprisal(masked, tokenizer, notes, bar_starts, i, candidates, device):
    """-log p of each candidate note at slot i, with slot i masked in `notes`."""
    tokens, spans = tokenizer.encode_with_spans(list(notes), bar_starts)
    p, d = spans[i]
    batch = torch.tensor([tokens], dtype=torch.long)
    batch[0, [p, d]] = masked.config.mask_id
    logp = masked(batch.to(device)).log_softmax(-1)[0].cpu()
    out = []
    for cand in candidates:
        variant = list(notes); variant[i] = cand
        vt, vs = tokenizer.encode_with_spans(variant, bar_starts)
        assert vs[i] == (p, d) and len(vt) == len(tokens)
        out.append(-float(logp[p, vt[p]] + logp[d, vt[d]]))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for corpus, source, directory in CONDITIONS:
        directory = Path(directory)
        config = json.loads(directory.with_suffix(".json").read_text())["config"]
        judge = Judge.load("models/judge_ctrl.pth" if corpus == "Essen" else "models/judge.pth", device)
        masked = load_masked_judge("models/masked_judge_ctrl.pth" if corpus == "Essen"
                                   else "models/masked_judge.pth", device)
        tokenizer, examples = test_examples(config, device)
        mel = {"n": 0, "more": 0, "less": 0, "d_draft": [], "d_out": []}
        note = {"draft": [], "output": [], "ref": []}
        exact = 0
        for path in sorted(directory.glob("outputs_alpha0.5_seed*_binary.json")):
            outputs = json.loads(path.read_text())
            assert len(outputs) == len(examples)
            changed = []
            for ex, out in zip(examples, outputs):
                draft = [tuple(n) for n in out["draft"]]
                assert draft == [tuple(n) for n in ex.draft_notes], f"{path}: draft mismatch"
                gated = [tuple(n) for n in out["gated_0.8"]]
                if gated != draft and len(gated) == len(draft):
                    changed.append((ex, draft, gated))
            if not changed:
                continue
            d_draft = judge.score_drafts([(d, ex.reference) for ex, d, _ in changed])
            d_out = judge.score_drafts([(g, ex.reference) for ex, _, g in changed])
            for a, b in zip(d_draft, d_out):
                mel["n"] += 1
                mel["d_draft"].append(a.delta_nll); mel["d_out"].append(b.delta_nll)
                mel["more"] += b.delta_nll < a.delta_nll
                mel["less"] += b.delta_nll > a.delta_nll
            for ex, draft, gated in changed:
                ref = list(ex.reference.notes)
                partner = {i: j for kind, i, j in align(draft, ref)[1] if kind == DIAG}
                for i, (before, after) in enumerate(zip(draft, gated)):
                    j = partner.get(i)
                    if before == after or j is None or before == ref[j]:
                        continue                      # only true substitutions
                    if after == ref[j]:
                        exact += 1
                        continue
                    s = slot_surprisal(masked, tokenizer, gated, ex.reference.bar_starts, i,
                                       [before, after, ref[j]], device)
                    note["draft"].append(s[0]); note["output"].append(s[1]); note["ref"].append(s[2])
        dd, do = np.array(mel["d_draft"]), np.array(mel["d_out"])
        sd, so, sr = (np.array(note[k]) for k in ("draft", "output", "ref"))
        row = {"corpus": corpus, "source": source,
               "melodies_changed": mel["n"],
               "dnll_draft_mean": float(dd.mean()), "dnll_output_mean": float(do.mean()),
               "recovered": float(1 - do.mean() / dd.mean()),
               "pct_more_plausible": mel["more"] / mel["n"], "pct_less_plausible": mel["less"] / mel["n"],
               "exact_repairs": exact, "mismatched_repairs": int(len(so)),
               "surprisal_median": {"draft": float(np.median(sd)), "output": float(np.median(so)),
                                    "reference": float(np.median(sr))},
               "pct_output_beats_draft": float((so < sd).mean()),
               "pct_output_at_least_reference": float((so <= sr).mean())}
        rows.append(row)
        print(f"{corpus:5s} {source:7s} melodies {mel['n']:5d}  recovered {row['recovered']:+.2f}  "
              f"more {row['pct_more_plausible']:.2f} less {row['pct_less_plausible']:.2f} | "
              f"mismatched {len(so):5d} (exact {exact:5d})  surprisal draft/out/ref "
              f"{np.median(sd):.2f}/{np.median(so):.2f}/{np.median(sr):.2f}  "
              f"out<draft {row['pct_output_beats_draft']:.2f}  out<=ref "
              f"{row['pct_output_at_least_reference']:.2f}", flush=True)
    path = Path("results/refiner/controls/repair_plausibility.json")
    save_json({"rows": rows}, path)
    print(f"\nrepair plausibility -> {path}")


if __name__ == "__main__":
    main()
