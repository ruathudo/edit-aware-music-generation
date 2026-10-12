"""
Score the listening test against its answer key and against the judges.

    python scripts/listening_report.py [--ratings results/human_test] [--no-judge]

Reads the rating files saved by the listening page (one per rater and round),
`results/listening/items.json` and `results/listening/key.json` (from
make_listening_set.py), and writes `results/listening/report.json`.

  spot   per draft source: naturalness (1 = clearly wrong ... 5 = natural), the
         share of excerpts marked "nothing sounds wrong", and how well the marks
         find the changed notes -- per rating (hit = any mark on a changed note,
         also within one note), per note (recall, precision of marks, false-alarm
         rate on unchanged notes), and threshold-free (AUROC of the share of
         raters marking each note), the same statistic the paper reports for the
         judge and the head. The two-sided judge is scored on the same notes.
         Item naturalness is correlated with the causal judge's dNLL, which tests
         the judge as a stand-in for listeners (paper L1).
         Recall is also split by the aligner's label at each changed note (pitch,
         duration, insertion, deletion), since listeners click notes and a
         rhythmic change has no single note to click.
  pair   main: an ep2 repair that missed the reference against the reference's
         note; check: the same repair against the draft note it replaced.

All raters rated blind to the key and are pooled. Robustness is checked by
leaving each rater out in turn, and each rater's naturalness ratings are
compared with the mean of the other raters on the same excerpts. Raters are
anonymised in the report (R1..Rn in sorted order of their initials). Confidence
intervals are percentile bootstraps over items (2,000 resamples); proportions
over ratings also get an exact two-sided binomial test where a null of 0.5
makes sense.
"""
import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.localization_report import auroc  # noqa: E402

SOURCES = ["D3", "D1", "D2-ep2", "D2", "reference"]
TAU = 0.585
LABELS = ["keep", "pitch", "duration", "insert before", "delete"]   # src.align order
BOOT = 2000
OUT = Path("results/listening/report.json")


# ---------------------------------------------------------------- statistics

def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial test (sum of outcomes no likelier than k)."""
    if n == 0:
        return float("nan")
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return float(min(1.0, sum(q for q in pmf if q <= pmf[k] * (1 + 1e-9))))


def ranks(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_ci(units, stat, rng, reps=BOOT):
    """Percentile 95% CI of stat(list of units) over resampled units."""
    units = list(units)
    vals = []
    for _ in range(reps):
        sample = [units[i] for i in rng.integers(0, len(units), len(units))]
        v = stat(sample)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def krippendorff_alpha(units, metric="nominal"):
    """Krippendorff's alpha. units: list of lists of values (one list per unit)."""
    units = [u for u in units if len(u) >= 2]
    pairable = sum(len(u) for u in units)
    if pairable < 2:
        return float("nan")

    def d(a, b):
        return (a - b) ** 2 if metric == "interval" else float(a != b)

    do = sum(sum(d(a, b) for i, a in enumerate(u) for j, b in enumerate(u) if i != j) / (len(u) - 1)
             for u in units) / pairable
    values = [v for u in units for v in u]
    de = sum(d(a, b) for i, a in enumerate(values) for j, b in enumerate(values) if i != j) \
        / (pairable * (pairable - 1))
    return float(1 - do / de) if de > 0 else float("nan")


# ---------------------------------------------------------------- loading

def load_ratings(folder):
    """{(rater, round): record}; identical re-downloads collapse, conflicts fail."""
    sessions, dupes = {}, []
    for path in sorted(glob.glob(str(Path(folder) / "*.json"))):
        rec = json.loads(Path(path).read_text())
        if rec.get("study") != "folk-melody-listening" or rec.get("version") != 2:
            raise SystemExit(f"{path}: not a v2 listening-test file")
        k = (rec["rater"], rec["round"])
        if k in sessions:
            if sessions[k]["spot"] != rec["spot"] or sessions[k]["pair"] != rec["pair"]:
                raise SystemExit(f"{path}: conflicts with another file for {k}")
            dupes.append(Path(path).name)
            continue
        incomplete = [x["id"] for x in rec["spot"] + rec["pair"] if not x["complete"]]
        if incomplete:
            raise SystemExit(f"{path}: incomplete items {incomplete}")
        sessions[k] = rec
    return sessions, dupes


def judge_scores(key, items):
    """Two-sided surprisal per excerpt note, and causal dNLL per item (full melody)."""
    import torch

    from src.align import edit_labels
    from src.judge import Judge
    from src.masked_judge import event_groups, load_masked_judge, score_events
    from scripts.make_listening_set import SWEEPS
    from scripts.repair_plausibility import test_examples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    judge = Judge.load("models/judge.pth", device)
    masked = load_masked_judge("models/masked_judge.pth", device)
    tokenizer = judge.tokenizer
    out = {}
    for source, directory in SWEEPS.items():
        config = json.loads(Path(directory).with_suffix(".json").read_text())["config"]
        _, examples = test_examples(config, device)
        wanted = [(i, a) for i, a in key.items()
                  if a.get("source") == source or (source == "D1" and a.get("source") == "reference")]
        for item_id, a in wanted:
            s, e = a["window"]
            notes = [tuple(n) for n in items[item_id]["notes"]]
            if a["source"] == "reference":
                ex = next(x for x in examples if x.reference.id == a["reference_id"])
                full = list(ex.reference.notes)
            else:
                ex = next(x for x in examples if x.reference.id == a["reference_id"]
                          and [tuple(n) for n in x.draft_notes[s:e]] == notes)
                full = list(ex.draft_notes)
            assert [tuple(n) for n in full[s:e]] == notes, item_id
            tokens, spans = tokenizer.encode_with_spans(full, ex.reference.bar_starts)
            groups = event_groups(tokens, spans, tokenizer.eos_id)
            tok = score_events(masked, tokens, groups, device=device)
            per_note = [float(tok[groups[i][0]]) for i in range(s, e)]
            dnll = 0.0 if a["source"] == "reference" else \
                judge.score_drafts([(full, ex.reference)])[0].delta_nll
            labels = [0] * len(full) if a["source"] == "reference" else \
                edit_labels(full, list(ex.reference.notes))
            out[item_id] = {"surprisal": per_note, "delta_nll": float(dnll),
                            "labels": [LABELS[labels[i]] for i in range(s, e)]}
        print(f"judged {source}", flush=True)
    return out


# ---------------------------------------------------------------- analysis

def analyse(sessions, key, items, judged, raters, rng):
    spot = defaultdict(list)    # item id -> [rating]
    pair = defaultdict(list)
    for (rater, rnd), rec in sessions.items():
        if rater not in raters:
            continue
        for x in rec["spot"]:
            assert key[x["id"]]["round"] == rnd
            spot[x["id"]].append({**x, "rater": rater})
        for x in rec["pair"]:
            assert key[x["id"]]["round"] == rnd
            pair[x["id"]].append({**x, "rater": rater})

    names = anonymise({r for r, _ in sessions})
    res = {"raters": sorted(names[r] for r in raters), "sessions": sum(r in raters for r, _ in sessions),
           "spot": {}, "pair": {}}

    # ---- spot, per source
    for source in SOURCES:
        ids = [i for i in spot if key[i]["source"] == source]
        rows = []      # one per rating
        for i in ids:
            truth = set(key[i]["wrong_positions"])
            n = len(items[i]["notes"])
            near = {p + d for p in truth for d in (-1, 0, 1) if 0 <= p + d < n}
            for r in spot[i]:
                marks = set(r["marks"])
                rows.append({"item": i, "nat": r["naturalness"], "nothing": r["nothing_wrong"],
                             "n_marks": len(marks), "hit": bool(marks & truth),
                             "hit1": bool(marks & near), "tp": len(marks & truth),
                             "n_true": len(truth), "fa": len(marks - truth), "n_clean": n - len(truth)})
        by_item = defaultdict(list)
        for r in rows:
            by_item[r["item"]].append(r)
        units = list(by_item.values())

        def flat(us):
            return [r for u in us for r in u]

        def mean_of(field):
            return lambda us: float(np.mean([r[field] for r in flat(us)]))

        def ratio(num, den):
            return lambda us: (sum(r[num] for r in flat(us)) / s) if (s := sum(r[den] for r in flat(us))) else None

        def mark_precision(us):
            m = sum(r["n_marks"] for r in flat(us))
            return sum(r["tp"] for r in flat(us)) / m if m else None

        entry = {"items": len(ids), "ratings": len(rows),
                 "naturalness": mean_of("nat")(units), "naturalness_ci": boot_ci(units, mean_of("nat"), rng),
                 "nothing_wrong": mean_of("nothing")(units),
                 "nothing_wrong_ci": boot_ci(units, mean_of("nothing"), rng),
                 "marks_per_rating": mean_of("n_marks")(units),
                 "false_alarm_rate": ratio("fa", "n_clean")(units)}
        if source != "reference":
            entry.update({
                "hit": mean_of("hit")(units), "hit_ci": boot_ci(units, mean_of("hit"), rng),
                "hit_within_1": mean_of("hit1")(units),
                "recall": ratio("tp", "n_true")(units),
                "mark_precision": mark_precision(units),
                "mark_precision_ci": boot_ci(units, mark_precision, rng),
                "base_rate": sum(r["n_true"] for r in rows) / sum(r["n_true"] + r["n_clean"] for r in rows)
                if rows else None})

            # threshold-free: share of raters marking each note, pooled over items
            def pos_auroc(item_ids, score_fn):
                s, t = [], []
                for i in item_ids:
                    truth = set(key[i]["wrong_positions"])
                    sc = score_fn(i)
                    s.extend(sc); t.extend(p in truth for p in range(len(sc)))
                s, t = np.array(s, dtype=float), np.array(t)
                if t.sum() in (0, len(t)):
                    return None
                return auroc(s, t), float(average_precision(s, t) / t.mean())

            def human(i):
                n = len(items[i]["notes"])
                return [np.mean([p in r["marks"] for r in spot[i]]) for p in range(n)]

            a, lift = pos_auroc(ids, human)
            entry["human_auroc"], entry["human_ap_lift"] = a, lift
            entry["human_auroc_ci"] = boot_ci(ids, lambda s: (pos_auroc(s, human) or (None,))[0], rng)
            if judged:
                def judge_z(i):
                    v = np.array(judged[i]["surprisal"])
                    return list((v - v.mean()) / (v.std() or 1.0))
                a, lift = pos_auroc(ids, judge_z)
                entry["judge_auroc"], entry["judge_ap_lift"] = a, lift
                entry["judge_auroc_ci"] = boot_ci(ids, lambda s: (pos_auroc(s, judge_z) or (None,))[0], rng)
            if judged:
                by_type = defaultdict(lambda: [0, 0])
                for i in ids:
                    for p in key[i]["wrong_positions"]:
                        t = judged[i]["labels"][p]
                        for r in spot[i]:
                            by_type[t][0] += p in r["marks"]
                            by_type[t][1] += 1
                entry["recall_by_label"] = {t: {"marked": m, "of": n, "rate": m / n}
                                            for t, (m, n) in sorted(by_type.items())}
        res["spot"][source] = entry

    # naturalness differences from references, per item means
    item_nat = {i: float(np.mean([r["naturalness"] for r in rs])) for i, rs in spot.items()}
    ref_nat = [v for i, v in item_nat.items() if key[i]["source"] == "reference"]
    for source in SOURCES[:-1]:
        nat = [v for i, v in item_nat.items() if key[i]["source"] == source]
        diffs = [float(np.mean(rng.choice(nat, len(nat))) - np.mean(rng.choice(ref_nat, len(ref_nat))))
                 for _ in range(BOOT)]
        res["spot"][source]["naturalness_minus_reference"] = float(np.mean(nat) - np.mean(ref_nat))
        res["spot"][source]["naturalness_minus_reference_ci"] = [float(np.percentile(diffs, 2.5)),
                                                                 float(np.percentile(diffs, 97.5))]

    # rater-normalized naturalness (z within rater) -- removes scale-use differences
    per_rater = defaultdict(list)
    for i, rs in spot.items():
        for r in rs:
            per_rater[r["rater"]].append(r["naturalness"])
    stats = {k: (np.mean(v), np.std(v) or 1.0) for k, v in per_rater.items()}
    for source in SOURCES:
        z = [(r["naturalness"] - stats[r["rater"]][0]) / stats[r["rater"]][1]
             for i, rs in spot.items() if key[i]["source"] == source for r in rs]
        res["spot"][source]["naturalness_z"] = float(np.mean(z))

    # judge as a stand-in for listeners
    if judged:
        ids = sorted(item_nat)
        dn = [judged[i]["delta_nll"] for i in ids]
        nat = [item_nat[i] for i in ids]
        drafts = [k for k, i in enumerate(ids) if key[i]["source"] != "reference"]
        res["judge_vs_listeners"] = {
            "spearman_naturalness_dnll_all": spearman(nat, dn),
            "spearman_ci_all": boot_ci(list(zip(nat, dn)), lambda s: spearman(*zip(*s)), rng),
            "spearman_naturalness_dnll_drafts": spearman([nat[k] for k in drafts], [dn[k] for k in drafts]),
            "spearman_ci_drafts": boot_ci([(nat[k], dn[k]) for k in drafts],
                                          lambda s: spearman(*zip(*s)), rng),
            "naturalness_dnll_above_tau": float(np.mean([n for n, d in zip(nat, dn) if d > TAU])),
            "naturalness_dnll_0_to_tau": float(np.mean([n for n, d, i in zip(nat, dn, ids)
                                                        if 0 < d <= TAU])),
            "naturalness_dnll_le_0_drafts": float(np.mean([n for n, d, i in zip(nat, dn, ids)
                                                           if d <= 0 and key[i]["source"] != "reference"])),
            "n_above_tau": sum(d > TAU for d in dn),
            "n_0_to_tau": sum(0 < d <= TAU for d in dn),
            "n_le_0_drafts": sum(d <= 0 for d, i in zip(dn, ids) if key[i]["source"] != "reference"),
            "mean_dnll_by_source": {s: float(np.mean([judged[i]["delta_nll"] for i in ids
                                                      if key[i]["source"] == s])) for s in SOURCES},
        }

    # ---- pair
    for kind, other in (("main", "reference"), ("check", "draft")):
        ids = [i for i in pair if key[i]["kind"] == kind]
        rows = []
        for i in ids:
            a = key[i]
            sur = a["judge_surprisal"]
            judge_pref = "repair" if sur["repair"] < sur[other] else other
            for r in pair[i]:
                pick = None if r["choice"] == "none" else a["a_is"] if r["choice"] == "A" else a["b_is"]
                rows.append({"item": i, "pick": pick, "judge": judge_pref,
                             "sdiff": sur[other] - sur["repair"]})
        n = len(rows)
        rep = sum(r["pick"] == "repair" for r in rows)
        oth = sum(r["pick"] == other for r in rows)
        decided = [r for r in rows if r["pick"]]
        by_item = defaultdict(list)
        for r in rows:
            by_item[r["item"]].append(r)
        units = list(by_item.values())

        def share_repair(us):
            d = [r for u in us for r in u if r["pick"]]
            return sum(r["pick"] == "repair" for r in d) / len(d) if d else None

        majority = defaultdict(int)
        for u in units:
            score = sum((r["pick"] == "repair") - (r["pick"] == other) for r in u)
            majority["repair" if score > 0 else other if score < 0 else "tie"] += 1
        entry = {"items": len(ids), "ratings": n, "prefer_repair": rep, f"prefer_{other}": oth,
                 "no_preference": n - rep - oth,
                 "share_repair_of_decided": share_repair(units),
                 "share_repair_ci": boot_ci(units, share_repair, rng),
                 "binomial_p": binom_two_sided(rep, rep + oth),
                 "item_majority": dict(majority),
                 "judge_prefers_repair_items": sum(u[0]["judge"] == "repair" for u in units)}
        if decided:
            entry["agree_with_judge"] = float(np.mean([r["pick"] == r["judge"] for r in decided]))
            entry["agree_with_judge_p"] = binom_two_sided(
                sum(r["pick"] == r["judge"] for r in decided), len(decided))
            for side in ("repair", other):
                sub = [r for r in decided if r["judge"] == side]
                entry[f"share_repair_when_judge_prefers_{side}"] = \
                    (float(np.mean([r["pick"] == "repair" for r in sub])), len(sub)) if sub else None
            item_share = [(np.mean([r["pick"] == "repair" for r in u if r["pick"]]), u[0]["sdiff"])
                          for u in units if any(r["pick"] for r in u)]
            entry["spearman_item_share_vs_surprisal_gap"] = spearman(*zip(*item_share))
        res["pair"][kind] = entry

    # ---- agreement between raters
    res["agreement"] = {
        "naturalness_alpha_interval": krippendorff_alpha(
            [[r["naturalness"] for r in rs] for rs in spot.values()], "interval"),
        "nothing_wrong_alpha_nominal": krippendorff_alpha(
            [[int(r["nothing_wrong"]) for r in rs] for rs in spot.values()]),
        "note_marks_alpha_nominal": krippendorff_alpha(
            [[int(p in r["marks"]) for r in rs] for i, rs in spot.items()
             for p in range(len(items[i]["notes"]))]),
        "pair_choice_alpha_nominal": krippendorff_alpha(
            [[r["choice"] if r["choice"] == "none" else
              (key[i]["a_is"] if r["choice"] == "A" else key[i]["b_is"]) for r in rs]
             for i, rs in pair.items()]),
        "ratings_per_spot_item": dict(sorted(
            {n: sum(len(rs) == n for rs in spot.values()) for n in {len(rs) for rs in spot.values()}}.items())),
    }
    return res


def anonymise(raters):
    return {r: f"R{k + 1}" for k, r in enumerate(sorted(raters))}


def compact(res):
    """The numbers the paper quotes, for the leave-one-rater-out check."""
    out = {s: {k: e.get(k) for k in ("naturalness_minus_reference", "hit", "human_auroc")}
           for s, e in res["spot"].items() if s != "reference"}
    if "judge_vs_listeners" in res:
        out["rho_all"] = res["judge_vs_listeners"]["spearman_naturalness_dnll_all"]
        out["rho_drafts"] = res["judge_vs_listeners"]["spearman_naturalness_dnll_drafts"]
    m, c = res["pair"]["main"], res["pair"]["check"]
    out["main_share_repair"], out["main_p"] = m["share_repair_of_decided"], m["binomial_p"]
    out["check"] = [c["prefer_repair"], c["prefer_draft"], c["no_preference"]]
    return out


def rater_table(sessions, key, items):
    out = {}
    names = anonymise({r for r, _ in sessions})
    for rater in sorted({r for r, _ in sessions}, key=lambda r: names[r]):
        recs = [rec for (r, _), rec in sessions.items() if r == rater]
        spot = [x for rec in recs for x in rec["spot"]]
        pair = [x for rec in recs for x in rec["pair"]]
        checks = [x for x in pair if key[x["id"]]["kind"] == "check"]
        check_pass = [(key[x["id"]]["a_is"] if x["choice"] == "A" else key[x["id"]]["b_is"]
                       if x["choice"] == "B" else None) == "repair" for x in checks]
        refs = [x for x in spot if key[x["id"]]["source"] == "reference"]
        drafts = [x for x in spot if key[x["id"]]["source"] != "reference"]
        mine, others = [], []
        for x in spot:
            peers = [y["naturalness"] for (r, _), rec in sessions.items() if r != rater
                     for y in rec["spot"] if y["id"] == x["id"]]
            if peers:
                mine.append(x["naturalness"]); others.append(float(np.mean(peers)))
        out[names[rater]] = {
            "rounds": sorted(rec["round"] for rec in recs),
            "check_repair_preferred": f"{sum(check_pass)}/{len(check_pass)}",
            "reference_any_mark": f"{sum(bool(x['marks']) for x in refs)}/{len(refs)}",
            "draft_hit": f"{sum(bool(set(x['marks']) & set(key[x['id']]['wrong_positions'])) for x in drafts)}/{len(drafts)}",
            "mean_naturalness": float(np.mean([x["naturalness"] for x in spot])),
            "naturalness_rho_vs_others": spearman(mine, others),
            "naturalness_mean_abs_diff_vs_others": float(np.mean(np.abs(np.array(mine) - others))),
            "median_seconds_per_excerpt": float(np.median([x["ms"] for x in spot]) / 1000),
            "mean_plays_per_excerpt": float(np.mean([x["plays"] + x["slow_plays"] for x in spot])),
        }
    return out


def average_precision(scores, truth):
    from scripts.localization_report import average_precision as ap
    return ap(scores, truth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", default="results/human_test")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()
    items = {i["id"]: i for i in json.loads(Path("results/listening/items.json").read_text())["items"]}
    key = json.loads(Path("results/listening/key.json").read_text())["key"]
    sessions, dupes = load_ratings(args.ratings)
    judged = None if args.no_judge else judge_scores(key, items)

    raters = {r for r, _ in sessions}
    report = {
        "files": len(sessions) + len(dupes), "duplicate_files_ignored": len(dupes),
        "sessions": len(sessions),
        "coverage_by_round": {n: sorted(anonymise({r for r, _ in sessions})[r] for r, k in sessions
                                        if k == n) for n in range(1, 6)},
        "per_rater": rater_table(sessions, key, items),
        "all": analyse(sessions, key, items, judged, raters, np.random.default_rng(0)),
    }
    names = anonymise(raters)
    report["leave_one_rater_out"] = {
        names[r]: compact(analyse(sessions, key, items, judged, raters - {r}, np.random.default_rng(0)))
        for r in sorted(raters)}
    OUT.write_text(json.dumps(report, indent=1))
    print_summary(report)
    print(f"\n-> {OUT}")


def print_summary(rep):
    print(f"{rep['sessions']} sessions ({rep['files']} files, duplicates ignored: {rep['duplicate_files_ignored']})")
    for n, rs in rep["coverage_by_round"].items():
        print(f"  round {n}: {', '.join(rs)}")
    for name in ("all",):
        res = rep[name]
        print(f"\n=== {name}: raters {', '.join(res['raters'])}, {res['sessions']} sessions")
        print(f"{'source':10s} {'n':>4s} {'nat':>5s} {'Δref':>6s} {'z':>6s} {'none%':>6s} {'hit%':>6s} "
              f"{'±1%':>5s} {'rec':>5s} {'prec':>5s} {'FA':>6s} {'hAUC':>6s} {'jAUC':>6s}")
        for s, e in res["spot"].items():
            f = lambda k, fmt="{:.3f}": fmt.format(e[k]) if e.get(k) is not None else "-"  # noqa: E731
            print(f"{s:10s} {e['ratings']:4d} {e['naturalness']:5.2f} "
                  f"{f('naturalness_minus_reference', '{:+.2f}'):>6s} {e['naturalness_z']:+.2f} "
                  f"{100 * e['nothing_wrong']:6.1f} {f('hit', '{:.1%}'):>6s} {f('hit_within_1', '{:.0%}'):>5s} "
                  f"{f('recall', '{:.2f}'):>5s} {f('mark_precision', '{:.2f}'):>5s} "
                  f"{e['false_alarm_rate']:6.3f} {f('human_auroc'):>6s} {f('judge_auroc'):>6s}")
        for s, e in res["spot"].items():
            if "recall_by_label" in e:
                print(f"  {s:9s} recall by label: " + ", ".join(
                    f"{t} {v['marked']}/{v['of']}={v['rate']:.2f}" for t, v in e["recall_by_label"].items()))
        if "judge_vs_listeners" in res:
            j = res["judge_vs_listeners"]
            print(f"naturalness vs dNLL: rho all {j['spearman_naturalness_dnll_all']:+.2f} "
                  f"{[round(x, 2) for x in j['spearman_ci_all']]}, drafts "
                  f"{j['spearman_naturalness_dnll_drafts']:+.2f} {[round(x, 2) for x in j['spearman_ci_drafts']]}")
        for kind, e in res["pair"].items():
            print(f"pair {kind}: {e}")
        print("agreement:", res["agreement"])
    print("\nleave one rater out:")
    for r, e in rep["leave_one_rater_out"].items():
        src = " | ".join(f"{s} {e[s]['naturalness_minus_reference']:+.2f} {e[s]['hit']:.0%} {e[s]['human_auroc']:.2f}"
                         for s in ("D3", "D1", "D2-ep2", "D2"))
        print(f"  -{r}: {src} | rho {e.get('rho_all', float('nan')):+.2f} | main {e['main_share_repair']:.0%} "
              f"p={e['main_p']:.2f} | check {e['check']}")
    print("\nper rater:")
    for r, e in rep["per_rater"].items():
        print(f"  {r:6s} {e}")


if __name__ == "__main__":
    main()
