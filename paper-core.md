# Paper core

The definitive statement of what the paper claims, in what order, and on what evidence.
Draft sections are written from this file. `paper-framing.md` is the revision log that
got here — consult it for *why* a claim changed, not for what to claim.

Written 2026-09-25, after revision-plan steps 1–7 and 5b. Target: an 8-page venue (not
ISMIR; venue not yet named). *Since then:* drafted for AIMG 2026 (10 pages including
references); the listening study is analysed in C15 (2026-10-12).

**How to use the tiers.** Every claim carries one:

- **P1 — core.** The paper is not coherent without it. Main text, always.
- **P2 — supporting.** Answers a predictable objection or makes P1 sharper. Main text if
  space allows; otherwise one sentence plus appendix.
- **P3 — appendix.** True, measured, worth having, first to cut.

If the draft runs long, cut P3 entirely, then compress P2 to single sentences with
pointers. Never cut a P1 claim's *caveat* to save space.

**Numbers.** Canonical values are listed in §9. All are MTC-ANN references, 3 seeds,
with the one setting chosen on D1 validation (binary head, `weighting=none`, α=0.5,
τ=0.8), unless stated otherwise. **Headline figures are 5 seeds** (step 8, 2026-09-25,
`results/refiner/headline_s34/five_seed_summary.json`). Comparisons against conditions
that were run on seeds 0–2 only — transfer, the judge + head gate, output quality,
surprisal input — use the matching 3-seed values, so every comparison stays
seed-matched.

---

## 1. Thesis

> **Localize-then-fix refinement works only where errors leave a local trace — and
> model-generated drafts leave little.** On synthetic corruption, a precision-gated edit
> model improves drafts by ~6% over leaving them alone, and is the only system tested
> that improves both edit distance and musical plausibility. On drafts written by a
> generator, every detector we built — trained or untrained, per note or per span —
> localizes the generated material at no better than 2.7× chance, against 6–7× for
> corruption. Generated drafts are worse than their references, but the damage is
> diffuse: at *matched* degradation, generated changes are still localized far less well
> than corrupted ones, in every band and on both corpora. **This is a property of the
> generator's quality, not of generation (revised 2026-09-27/28):** across six generator
> settings, ordered by the generator's own held-out NLL, localizability falls steadily as
> the generator improves — for the judge and for an independent head trained only on
> corruption. Localize-then-fix gets harder exactly as generators get better. Where a
> weak generator's changes are found, they are repaired toward the rewritten context
> rather than back to the reference, which edit distance scores as no gain.

Three things the paper therefore contributes, in one line each:

1. **A measurement:** Correction Cost cannot tell an error from a valid alternative, and
   a calibrated plausibility judge can.
2. **A boundary:** where localize-then-fix applies (off-manifold errors) and where it
   does not (generator output), mapped from both sides under one configuration.
3. **A method that respects the boundary:** precision-gated editing, which improves
   drafts where errors are local and abstains where they are not.

## 2. Title

Recommended (2026-09-27):

- **The Better the Generator, the Less the Trace: When Does Localize-then-Fix Work for
  Symbolic Music Refinement?**

Previous recommendation, before the severity curve: *Generated Drafts Leave Little Local
Trace: …* — now too broad, since an undertrained generator's drafts leave as much trace
as corruption. (It also said "Localize-and-Fix"; the body uses "localize-then-fix".)

Alternatives:

- *Difference Is Not Error: Evaluating Symbolic Music Refinement Beyond Edit Distance*
- *When Is a Draft Locally Wrong? Mapping the Domain of Edit-Aware Music Refinement*

Retired: "Generated Drafts Are Not Locally Wrong" — the two-sided judge finds 2.7× chance
on D2 and 4.5× on certified-degraded D2, so "not" is false.

## 3. Story arc and page budget (8 pages)

| § | section | carries | pages |
|---|---|---|---|
| 1 | Introduction | thesis, three contributions, one figure of the boundary | 1.0 |
| 2 | Related work | MDT, RefinPaint, edit models, selective prediction | 0.6 |
| 3 | Task, drafts, representation | D1–D4, REST/BAR tokens, aligner, CC | 0.9 |
| 4 | Measuring plausibility | judge, calibration on human variants, C1–C2 | 1.0 |
| 5 | Method | encoder–decoder, edit head, gated decoding, τ=1 guarantee | 0.8 |
| 6 | Results I: where it works | C5–C7 on D1 (and D3 as control) | 1.3 |
| 7 | Results II: where it does not | C3–C4 on D2, zero-shot, transfer | 1.3 |
| 8 | Discussion and limitations | C8–C9, §6 limitations | 0.8 |
| — | Conclusion | | 0.3 |

Order rationale: measurement first (§4), because every later claim is judged with it;
the positive result (§6) before the negative (§7), so the reader has seen the method
work before seeing where it cannot.

## 4. Claims

Each claim: statement → evidence → source → caveat.

### P1 — core

**C1. A plausibility judge calibrated on human variants.** (§4)
An independently trained melody LM `J` (disjoint Essen half from the generator) scores
ΔNLL = NLL(draft) − NLL(reference). Calibration: human-authored tune-family variants —
valid alternatives by construction — score mean ΔNLL +0.054, median +0.031, 55.0%
positive: a coin flip. So the judge measures degradation, not novelty. The degradation
threshold τ is the variants' 95th percentile (0.585).
*Source:* `results/judge/validator_report.json`.
*Caveat:* ΔNLL is statistical implausibility under a small LM, not a musician's verdict
(§6 L1). *Listening study (C15):* the judge tracks listeners in direction, not closely:
excerpt naturalness correlates with ΔNLL at ρ = −0.40 over all 100 excerpts, −0.26
among drafts.

**C2. Correction Cost cannot tell an error from a valid alternative.** (§4)
Human variants sit 92% normalized edit distance from each other at +0.038 ΔNLL per unit
of change; random corruption sits 7% away at +3.94 — ~100× more implausible per unit
(rule-based: +3.14, ~80×).
Per draft, CC and ΔNLL correlate *within* a corruption process (Spearman ρ +0.52 D1,
+0.47 D3) but the relation vanishes across sources (pooled ρ = −0.06, CI [−0.11,
−0.01]); the most distant drafts are the most plausible.
**Claim:** CC can rank methods on one draft source; it cannot compare sources, and
cannot distinguish error from alternative.
*Source:* `validator_report.json`, `results/judge/decoupling.json`.
*Caveat:* within-source ρ is partly built in (edit count drives both). Matched-distance
evidence is thin (9 human pairs in the synthetic range).

**C3. Generated drafts leave little local trace.** (§7)
Under one configuration for every source, localization of edited positions
(threshold-free; AP lift over base rate / AUROC):

| | trained head | untrained two-sided judge |
|---|---|---|
| D1 random | 5.8× / 0.724 | 6.2× / 0.838 |
| D3 rule-based | 4.6× / 0.726 | 5.9× / 0.812 |
| **D2 generated** | **1.6× / 0.636** | **2.7× / 0.688** |

(Trained head: 5 seeds. Judge: one trained model.)

Essen replicates every row. The D2 head collapses to KEEP — 0 positions clear τ on any
seed — although D2 has *more* true edits (7.3% of positions) than D1 (6.0%). A span
detector (1.6× chance top-1) and heads transferred from D1/D3 (1.3–1.4×) do no better.
*Source:* `results/refiner/headline_r2/{localization,zero_shot_mtc,zero_shot_essen}.json`,
`results/refiner/transfer/`.
*Caveat:* this is an absence of evidence across the detectors tried, not impossibility.
State the detectors, and that the strongest is untrained.

**C4. Generated damage is diffuse, not merely small.** (§7) *Revised 2026-09-27.*
Banding every source at the same absolute ΔNLL cut-points, the two-sided judge localizes
D2 below D1 in every band on both corpora, by 0.08–0.16 AUROC (e.g. 0 < ΔNLL ≤ 0.1:
D1 0.830, D3 0.774, D2 0.684 on MTC), although D2 changes more notes per band. D1 is
flat at 0.83–0.86 once ΔNLL > 0; D2 climbs slowly and its most-degraded band (0.775 /
0.779) stays below D1's least-degraded positive band. Mechanism: unchanged notes after a
rewritten span also lose plausibility (+0.07/token vs +0.22 within), so damage lies at
the seam, partly on notes labelled correct.
*Source:* `results/refiner/controls/matched_strata.json` (`scripts/matched_strata.py`;
~5,300 fresh MTC drafts per source, Essen test drafts). D2 rows reproduce `strata.json`.
*Why revised:* the old C4 ("most generated drafts are different, not wrong") rested on
only 2.5–4.4% of D2 clearing τ — but only 13.0% of D1 and 9.6% of D3 clear it too. τ comes
from variants differing in 92% of notes, and ΔNLL is a per-token mean, so it is a
conservative bar for any synthetic draft. 79.5% of D2 drafts are worse than their
reference. The D2-only banding could not separate "too little damage" from "diffuse
damage", since high-ΔNLL drafts are easier for any surprisal localizer.
*Caveat:* D3 is not uniformly above D2 — on Essen D3's 0–0.1 band (0.740) is below D2's
> τ band (0.779). State the strict comparison against D1 only.

**C5. Precision-gated editing improves drafts where errors are local.** (§5, §6)
Copy each event by default; regenerate it only when the head's P(edit) ≥ τ. At τ=1 the
output is the draft verbatim, so the method's floor is copy-input by construction.
- D1: **+6.06% ± 0.58 over copy-input**, 5/5 seeds (5.87 / 6.83 / 6.34 / 5.28 / 5.96);
  same-corpus **+4.25% ± 0.46**, 5/5. Of the edits it makes, 82% are true errors; it
  improves 23.0% of melodies and worsens 6.3% (seeds 0–2).
- It beats the imitation baseline B1 by a wide margin: B1 is 24.9% *worse* than copying.
- Settings chosen on validation only; the gain is flat across them (every class
  weighting peaks at 6.3–6.5%; 84/96 settings beat copy-input on all seeds).
*Source:* `results/refiner/headline_r2/`, `results/refiner/select/selection.json`.
*Caveat:* runs are not bit-reproducible — two full runs differ by up to 1.17 pp on a
seed, 0.36 pp on a cell mean.

**C6. …and it abstains where they are not.** (§6–7)
On D2 the gate never opens: output equals the draft, +0.00% on every seed, both corpora.
This is the method respecting the boundary, not failing: C3–C4 show there is little to
act on. It is also the reason the paper's method is the head gate rather than the
stronger judge + head combination (C9).
*Caveat:* abstention requires in-distribution training. Heads trained on D1 or D3 and
applied to D2 *open* the gate and lose 1.71% and 2.01% (3/3 seeds each). The τ=1
guarantee is exact; at a working τ, no-harm is conditional.

**C7. Gated editing is the only system better than the draft on both CC and
plausibility.** (§6)
Where the gate edits (MTC D1), it removes 30.1% of the draft's implausibility and makes
2.8% of edited melodies worse; D3 48–53%, ≤1.4% worse. Free decoding recovers *more*
plausibility (40.9%) while being 24.7% worse than copying on CC: it decodes greedily
toward the LM's mode, 14.5% of its outputs score more plausible than the reference, and a
third of the melodies it changes end up more plausible *and* further from it.
An evaluation on plausibility alone would crown free decoding; on CC alone it would hide
that gated editing improves the music. Report both.
*Source:* `results/refiner/headline_r2/quality.json`.
*Caveat:* the 97.6% / 14.5% / one-third breakdowns are one seed of MTC D1. ΔNLL rewards
mode-seeking, which the 14.5% itself shows.

### P2 — supporting

**C8. Detectability follows manifold distance; training adds a source-specific share.**
(§7)
The untrained two-sided judge ranks sources D1 > D3 > D2, matching their ΔNLL per unit
of change (3.94 > 3.14 > 1.04). A trained head adds learned regularity on top, which
transfer isolates: a D1-trained head reaches +7.25% on D3 against D3's own +9.98%
(both seeds 0–2), so
~2.7 of D3's ~10 points do not transfer. *Revised 2026-09-27:* the loss is symmetric —
a D3-trained head reaches +4.80% on D1 against D1's own +6.35% (24% vs 27%). So transfer
measures source-specific learning for *every* synthetic benchmark, not D3's rules in
particular. Claim: ~a quarter of in-domain gain is specific to the corruption process,
random or rule-based alike. D3 is also intrinsically easier: a D1-trained head gains more
on D3 (+7.25%) than on D1 itself.
*Source:* `zero_shot_*.json`, `results/refiner/transfer/`.
*Caveat:* three sources is a thin basis for "follows"; say "consistent with".

**C8b. Localizability tracks the generator's quality.** (§7, Figure 1a) *New 2026-09-27.*
Six D2 conditions — T ∈ {0.8, 1.0, 1.2} with the trained generator, checkpoints after
2/5/10 epochs at T=1.0 — scored zero-shot by the two-sided judge. MTC AUROC (per unit):
T=0.8 .652 (0.86), T=1.0 .688 (1.04), T=1.2 .718 (1.21), ep10 .769 (1.41), ep5 .832
(1.70), ep2 .865 (2.01); D3 .812 (3.14), D1 .838 (3.94). Essen: .647 → .858, same order,
D1 .831. Rises steeply to ~2 per unit, then plateaus at 0.81–0.87 for every source.
*Source:* `results/judge/severity_curve.json` (`scripts/severity_curve.py`).
*Caveat:* one generator weakened, not generators of increasing strength compared (L7).
*Reviewer checks (2026-09-28):* held-out NLL orders the settings independently of `J`
(1.84 / 1.70 / 1.50 / 1.23 MTC generator; `generator_quality.json`). An independent
localizer — the D1-trained head, zero-shot — rises in the same direction (MTC 0.57 → 0.64,
one small inversion; Essen 0.61 → 0.68, monotone) but stays below D1 (0.73)
(`severity_head.json`). *Do not claim* "the 2-epoch generator's drafts are as
localizable as corruption" without "by the judge".

**C8c. Repaired, but not to the reference.** (§7, Table 7, Figure 1b) *New 2026-09-27,
revised 2026-09-28: the original title "localizable is not repairable" was wrong.*
`repair_plausibility.py`: on ep2 drafts gating removes 28% / 36% of the drafts' ΔNLL
(85–88% of changed melodies more plausible); at mismatched repairs the output note's
two-sided surprisal is 2.36 / 2.27 vs reference 3.25 / 3.90 vs draft 11.05 / 10.73, and
output ≤ reference in 64% / 68%. The decoder repairs toward the rewritten context; CC and
exact repair cannot see it. Caveat: the judge rewards typical notes. *Listening study
(C15):* listeners have no preference between the decoder's note and the reference's
(42 vs 40, 54 no preference), so they hear the repairs as different, not wrong, and
not as better. They always prefer the repair to the draft's note (30 vs 0).
Original evidence, still valid:
Refiners trained on the 2-epoch generator's drafts (headline setting, 3 seeds) gain
−0.13% ± 0.24 over copy-input on MTC (1/3 seeds) and +0.38% ± 0.30 on Essen (3/3) — a
tenth of D1's gain; B1 free −11.4% / −28.4%. Head AUROC 0.67–0.81 (judge 0.86–0.87),
gate precision 0.59–0.72 (D1 0.82). At gate-opened true substitutions the decoder writes
the reference's exact event 11% / 13% of the time (D1 59% / 58%, D3 44% / 41%). Mechanism: a rewritten span's original is one plausible continuation among
many; a corrupted note is pinned by its neighbours. Oracle agrees (49% vs 21.5%).
*Source:* `results/refiner/headline_ep2/`, `results/refiner/controls/repair_accuracy.json`
(`scripts/repair_accuracy.py`).
*Do not claim:* "gated editing works wherever errors are localizable", nor "found errors
in generated drafts are rarely repaired" — they are repaired, differently; nor that the
repairs are *better* than the reference — the judge prefers them (25 of 40 study pairs),
listeners do not (C15).

**C9. Put the judge at the gate, not in the model.** (§8)
- At the gate: combining judge surprise with the head's confidence (a two-feature
  logistic fitted on D1 validation) adds **+1.4 to +2.1 points** over the head on every
  D1/D3 cell, 3/3 seeds seed-matched (MTC D1 +7.91%, D3 +11.80%). It costs 0.4–0.5% on D2,
  where it opens the gate the head keeps shut — hence an extension, not the method.
- As input: feeding surprisal to the encoder improves localization (AUROC 0.73 → 0.82)
  with unchanged gate precision, yet loses 1.2 points (3/3 seeds) and collapses free
  decoding to −61%. The damage is in the decoder.
- The judge alone gates worse than the head (+2.8% on D1): standardized within a melody,
  it always nominates a note and cannot abstain on a clean melody.
*Source:* `headline_r2/judge_gate.json`, `results/refiner/controls/mtc_d1_surprisal.json`.
*Caveat:* judge-based gates looked optimistic on validation (the two-sided judge was
early-stopped on the validation references); test numbers are clean.

**C10. An untrained judge localizes better than every trained head.** (§7)
Two-sided judge vs trained head (5 seeds), AUROC: D1 0.838 vs 0.724, D3 0.812 vs
0.726, D2 0.688 vs 0.636, both corpora. The head is shown at its best (standardizing its scores lowers
them). A note's plausibility given both sides is a stronger error signal than anything
the supervised head learned.
*Caveat:* each judge is one trained model; heads are 3 seeds.

**C11. An upper bound.** (§6)
Given ground-truth edit locations the same decoder removes 49.3% of D1's error and 21.5%
of D2's. The head gate recovers ~13% of the D1 headroom. D2's oracle result also shows
the generated spans are *correctable* once located — the bottleneck is finding them.
*Source:* `results/refiner/diagnostics_{D1,D2infill}.json` (single run, same test sets).

**C12. The imitation baseline converges to copying.** (§6)
Plain seq2seq refinement is worse than copying the draft (−24.9% on D1, 5 seeds). Adding an
edit-aware auxiliary loss on D2 improves on it by 6.3% (α=1.0, 3/3 seeds) — but at that
optimum fidelity is 0.0009 per note and CC equals copy-input's (4.726 vs 4.709): the gain
is convergence to copying. **Always report a copy-input baseline and fidelity.**
*Source:* `results/refiner/seeds_d2*.json`.
*Framing note:* do not present this as reproducing a prior published result — the
earlier version of this work was not published, and a double-blind venue forbids citing
it. Present the α-weighted objective as a natural baseline and copy collapse as the
finding.

**C13. Contiguity matters, but does not explain D2.** (§7)
Within D3, AUROC falls with run length (head 0.80 → 0.71; judge 0.94 → 0.75), plausibly
because masked neighbours stay visible as corrupted context. But D3's contiguous band
still beats D2 (0.71 vs 0.64). Gating, which uses only the confident tail, is
insensitive to run length.
*Run-length matched (2026-09-27):* runs of 3+, judge D2 0.68 vs D3 0.75 (Essen 0.68 vs
0.72), head 0.64 vs 0.71; drafts with no two adjacent changes, judge D2 0.75 vs D1 0.84,
D3 0.94 (Essen 0.73 / 0.83 / 0.92). D2 is worse at every matched run length.
*Source:* `results/refiner/controls/strata.json`, `locality_all_sources.json`
(`scripts/locality_all_sources.py`).

**C14. The confound checks.** (§7, one paragraph)
- Same-corpus control (judge, generator and references all Essen): both halves
  replicate; D2 localization equally weak on both corpora.
- Capacity: localization precision flat from 1.6M to 42.7M parameters; gated beats
  copy-input in 9/9 size configurations [earlier τ=0.9 setting, ranked 2nd of 96 on
  validation].
- Weighting and threshold: flat landscape (C5).

**C15. Listening study: listeners agree with the judge at the extremes, not on random
corruption.** (§7 "Listening study", one table; feeds C1, C3, C8c, L1)
*Added 2026-10-12.* Main text: the paper has reserved space for it.

*Design* (`scripts/make_listening_set.py`). MTC test material, synthesized playback at
100 bpm (70 bpm slow option). Five balanced rounds, each with 20 excerpts and 10 pairs.
- **Spot task:** 100 excerpts of at most 20 notes, 20 per source: D3, D1, D2
  (trained generator), D2-ep2 (2-epoch generator), and untouched references as catch
  trials. Each excerpt window contains every changed note, and each comes from a
  different tune. Raters click the notes that sound wrong (or tick "nothing sounds
  wrong") and rate naturalness from 1 (clearly wrong) to 5 (natural).
- **Pair task:** two versions of an excerpt that differ at one highlighted note. In the
  40 main pairs, an ep2 gate repair that missed the reference is set against the
  reference's note, both in the output's context. The 10 check pairs set the same kind
  of repair against the draft note it replaced; the judge prefers the repair in all 10.

*Sample.* 17 sessions from 6 raters, most with musical training, all rating blind to
the key, pooled. Each excerpt was rated 3–4 times: 340 excerpt ratings and 170 pair
ratings in all. CIs are 95% bootstraps over items; raters are too few to resample.
*Robustness:* leaving out any one rater moves no conclusion:

| measure | range over the six leave-one-out runs |
|---|---|
| changed notes found: D3 / D1 / ep2 / D2 | 75–81 / 23–27 / 23–35 / 18–21% |
| D2 listener AUROC | 0.53–0.54 |
| ρ(naturalness, ΔNLL) | −0.35 to −0.45 |
| repair share, repair vs reference | 49–53% |
| draft note chosen, repair vs draft | never |

Raters differ mainly in how they use the scale (mean naturalness 3.0–4.0). Each rater's
naturalness correlates with the other raters' mean on the same excerpts at ρ = 0.43–0.67,
so none is an outlier.
*Source:* `results/listening/report.json` (`scripts/listening_report.py`). Ratings are in
`results/human_test/`, gitignored because they carry raters' names; the report
anonymises them (R1–R6).

| source | naturalness (1–5) | vs reference | "nothing wrong" | hit | mark precision (× base) | listener AUROC | judge AUROC, same notes |
|---|---|---|---|---|---|---|---|
| D3 rule-based | 2.56 | −1.43 [−1.89, −0.92] | 21% | **78%** | 0.81 (6.1×) | **0.77** | 0.89 |
| D1 random | 3.65 | −0.38 [−0.79, +0.03] | 54% | 25% | 0.45 (4.2×) | 0.59 | 0.78 |
| D2-ep2 | 3.43 | −0.59 [−1.00, −0.15] | 57% | 28% | 0.63 (3.2×) | 0.62 | 0.83 |
| D2 generated | 3.66 | −0.38 [−0.79, +0.01] | 62% | **19%** | 0.42 (2.5×) | **0.54** [0.49, 0.60] | 0.73 |
| reference | 4.06 | — | 82% | — | — | — | — |

"Hit" is the share of ratings with at least one mark on a changed note; widening it to
within one note adds at most 10 points. On unchanged notes the false-alarm rate is
1.5–1.9% for every source, references included, so the sources differ in what listeners
find, not in how freely they click. The listener AUROC treats the share of raters
marking a note as its score, the same threshold-free statistic as C3.

What it shows:

1. **Listeners cannot locate a good generator's changes (supports C3 and the thesis).**
   On D2 the hit rate is 19% and the listener AUROC is 0.54, with a CI that touches 0.5.
   Listeners find 8% of changed pitches, against 68% for D3. The judge ranks D2 lowest
   on the same notes too. D2-ep2 sits above D2 on every detection measure, as C8b
   predicts, but the CIs overlap.
2. **Rule-based errors are the only ones listeners reliably hear.** D3 is the only source
   with a clear naturalness drop, and listeners mark 78% of D3 excerpts at precision
   0.81.
3. **Random corruption is nearly as quiet to listeners as generated changes — a
   divergence from the judge.** Listeners find D1 at 25% with AUROC 0.59; the judge
   scores the same notes at 0.78. Their naturalness gap matches D2's, both CIs reaching
   0. The judge ranks D1 first on the full test set (C8) and listeners rank it with D2.
   Two reasons, both measured:
   - **Rhythm changes have no note to click.** Listeners mark 5% of D1's changed
     durations and insertion points, and 3–15% on the other sources. 59% of D1's
     changed-note ratings fall on these, against D3's changes, which are mostly
     pitches, marked at 68%.
   - **The D1 excerpts are mild.** Each window had to contain every edit, which
     selected D1 drafts with few edits: mean ΔNLL 0.16, against 0.30 on D1's full test
     set and 0.23 for the D2 excerpts.

   Even D1's random *pitch* changes are found only 15% of the time; they are drawn from
   the corpus pitch prior, so they are often in key. Where listeners do mark D1, they
   are 4.2× above base rate against D2's 2.5×, so the D1 > D2 ordering survives, weakly.
4. **The judge tracks listeners in direction (bears on C1 and L1).** Mean excerpt
   naturalness against the judge's ΔNLL gives ρ = −0.40 [−0.55, −0.21] over all 100
   excerpts and −0.26 [−0.47, −0.05] over the 80 drafts. Mean naturalness by ΔNLL
   band:

   | judge ΔNLL band | excerpts | mean naturalness |
   |---|---|---|
   | > τ | 5 | 1.95 |
   | 0 to τ | 68 | 3.35 |
   | ≤ 0 (drafts) | 7 | 4.04 |
   | references | — | 4.06 |

   The judge is a proxy, not a substitute.
5. **Repairs sound different, not wrong (supports C8c).**
   - Main pairs: repair 42, reference 40, no preference 54. The repair takes 51% of
     decided ratings [38%, 64%], binomial p = 0.91. Item majorities split 16 / 17 with
     7 ties.
   - The judge prefers the repair on 25 of 40 pairs. Listeners agree with it 55% of the
     time, p = 0.44, and their item-level preference barely follows the judge's
     surprisal gap (ρ = 0.21). The judge's preference for typical notes is not
     something listeners share.
   - Check pairs: repair 30, draft **0**, no preference 4, p = 2×10⁻⁹. On ep2 drafts the
     notes the gate replaces are ones listeners hear as worse.
   - No rater ever chose a draft note, so all six pass the attention check. One rater
     chose "no preference" on 2 of 6 checks.
6. **Agreement between raters is moderate.** Krippendorff's α: naturalness 0.45
   (interval), note marks 0.38, "nothing wrong" 0.34, pair choice 0.22. The pair figure
   is low, as expected when most main pairs have no audible winner.

*How to use it in the paper.*
- State the working assumption only as far as it held. Listeners agree with the judge
  that D3 is audible and D2 is not, that repairs beat draft notes, and on the direction
  of naturalness. They part from it on random corruption, and on whether repairs beat
  the reference.
- The D1 divergence does not touch C5–C7, which are claims about edit distance and the
  judge. It does narrow "local trace" to *a trace a detector can find*, which is not the
  same as one a listener can hear (L8).
*Caveat:* six raters, 3–4 ratings per item and 20 items per source is a small study. The
CIs above, not the point estimates, are the result. The task needs no training, so the
paper need not dwell on rater background.

### P3 — appendix, first to cut

- **Head-type ablation.** A five-class head (where *and* how) matches a binary one
  (seed-matched +0.10 ± 0.36 points of gain over copy-input — equivalently +0.003 ± 0.011
  in Correction Cost — at τ=0.9); INSERT_BEFORE F1 0.000. Justifies the binary head. One
  sentence in §5 suffices.
- **Size ablation table** (9 configurations × 3 seeds).
- **Span-level detector** (1.6× chance) — one row in the C3 discussion is enough.
- **Label-free pipeline:** judge gate with a decoder trained without edit labels beats
  copy-input on D1 (+2.4%) and D3 (+6–7%), 3/3 seeds. Relevant to future work on
  unlabelled drafts.
- **Gate fixes pitch, not rhythm:** D1 metric misalignment unchanged by gating.
- **Representation measurements:** REST/BAR tokens, 100% representability, onset-shift
  removed (answers the representation objection; may compress to one sentence + table).

## 5. What the paper must not claim

1. "Generated drafts cannot be localized" → *far less localizable* (2.7× vs 6–7×).
2. "Generated drafts are not locally wrong" → retired title; see C4.
3. "Where and how" → the paper localizes; typing adds nothing (P3).
4. "Edit-awareness lowers Correction Cost" → not reliably on D1 (2/3 seeds); its value
   is a usable localizer for the gate.
5. "Gating cannot hurt" → only at τ=1 (C6 caveat).
6. "CC and plausibility are orthogonal" → coupled within a source, decoupled across (C2).
7. "Contiguity is irrelevant" / "D2 is explained by span length" → neither (C13).
8. "D3 is circular" / "a quarter of D3's gain is learning its rules" → the transfer
   loss is symmetric (24% D3→D1, 27% D1→D3); it is a property of synthetic benchmarks
   in general (C8).
9. "Raising recall hurts because precision is binding" → the decoder degrades (C9).
10. "Edit-awareness hurts on random corruption" → an artifact of class weighting.
11. "Most generated drafts are different, not wrong" → at matched degradation D2 is
   still far less localizable; the damage is diffuse, not absent (C4). Do not use τ
   exceedance as evidence that a draft is error-free.
12. "Generated drafts are hard to localize" (unqualified) → drafts from a *good*
   generator are; an undertrained generator's are as localizable as corruption (C8b).
13. "A listening study confirms the judge" (unqualified) → listeners agree on D3 vs D2,
   on draft-vs-repair, and on the direction of naturalness (ρ = −0.40); they do not
   agree on D1, and do not share the judge's preference for repairs over the reference
   (C15).
14. "Listeners prefer the repairs" → to the draft's note, yes (30–0); to the reference's
   note, no (42–40, 54 no preference) (C15).
15. "Generated drafts sound as good as their references" → the naturalness gap
   (−0.38 [−0.79, +0.01]) is not resolved by 20 items per source; it is the same size
   as D1's. Absence of a significant gap is not evidence of no damage (C4, C15).
16. "Listeners can hear random corruption" / "D1's gain is audible" → listeners find
   25% of D1 excerpts' changes and mark 5% of rhythmic ones; C5–C7 are claims in edit
   distance and the judge, not in perception (C15, L8).

Also avoid: any human-in-the-loop or user-utility claim beyond one motivating sentence;
"reproduces the published result" (C12 framing note).

## 6. Limitations to state

- **L1.** ΔNLL is statistical implausibility under a small LM, calibrated against human
  variants and checked against listeners (C15): it tracks their naturalness ratings in
  direction (ρ = −0.40), but loosely. It also rewards typical continuations (C7), which
  listeners do not share (C15, point 5).
- **L2.** Two folk corpora (MTC-ANN, Essen); monophonic; the style claim is narrow.
- **L3.** The negative result is bounded by the detectors tried — including the
  strongest we could build without labels.
- **L4.** The judge scores MTC with a 0.1365 nats/token domain gap; the same-corpus
  control rules out a confound, not the sensitivity.
- **L5.** Runs are not bit-reproducible (C5 caveat); results are reported over seeds and
  one full re-run.
- **L6.** D2 drafts are not filtered by the judge; C4 (matched strata) is the argument
  for why filtering would not change the localizability conclusion.
- **L7.** The generator-quality gradient weakens one generator; that stronger generators
  leave still less trace is an extrapolation (C8b).
- **L8.** The listening study is small: six raters with 3–4 ratings per item and 20
  items per source. Its click-a-note interface finds pitch errors more
  easily than rhythmic ones, and its excerpt selection made the D1 items mild. "Local
  trace" in this paper means a trace a detector can find; listeners hear less of it on
  random corruption (C15).

## 7. Anticipated objections (short form)

| objection | answer |
|---|---|
| Your method only works on synthetic corruption. | Yes; that is the finding, mapped from both sides (C3, C6). |
| A better detector would find D2's errors. | The strongest we built is untrained, beats every trained head, and reaches 2.7× on D2 vs 6–7× elsewhere (C3, C10). |
| Models too small / badly tuned. | Same machinery gets 5.8× on D1; flat across 26× parameters and 96 settings (C5, C14). |
| D2 drafts just carry less damage. | At matched ΔNLL they are still far less localizable than D1, on both corpora (C4). |
| +6% is small. | It is over copy-input, which the imitation baseline loses to by 25% (C5, C12). |
| CC tracks plausibility in your own table. | Within a source; not across (C2). |
| ΔNLL isn't musical correctness. | Conceded; calibrated on human variants and checked against listeners, who agree in direction (ρ = −0.40) but not on everything (C1, C15, L1). |
| Do listeners hear any of this? | They hear rule-based errors (78%) and not a good generator's changes (19%, AUROC 0.54). They prefer repairs to draft notes 30–0 and are indifferent between repairs and the reference. They also miss most random corruption, so the D1 result is a detector-level claim (C15, L8). |
| D3 is circular. | Partly; transfer measures how much (C8). |

## 8. Dropped from the original spec (state in §3 or appendix)

Data-size ablation; mixed-source training; D3→D4 transfer (D4 is a calibration control,
not a refinement target); weighted CC variant; per-operation CC breakdown; scale-matched
B2 (the binary head is 514 parameters, 0.009%, so parameter matching is moot); the judge
degradation filter on D2 (L6); human-in-the-loop framing.

## 9. Canonical numbers

MTC, 3 seeds, validation-selected setting, unless noted. Essen in brackets.

| quantity | value | source |
|---|---|---|
| D4 variants: ΔNLL mean / median / % positive | +0.054 / +0.031 / 55.0% | validator_report |
| τ (95th pct of variant ΔNLL) | 0.585 | validator_report |
| ΔNLL per unit change: D4 / D1 / D3 / D2 (test drafts) | 0.038 / 3.94 / 3.14 / 1.04 | plausibility_table |
| ΔNLL ≤ 0 / > τ: D4, D1, D3, D2 | 45.0/5.0, 5.2/13.0, 4.2/9.6, 20.5/2.6 % | plausibility_table |
| Spearman ρ(surface, ΔNLL): D1 / D3 / D2 / D4 / pooled | +.52 / +.47 / +.20 / +.14 / −.06 | decoupling |
| Gated vs copy-input, 5 seeds: D1 / D3 / D2 | +6.06 ± 0.58 / +9.81 ± 0.46 / 0.00% (+4.25 / +7.38 / 0.00); all D1/D3 cells 5/5 | five_seed_summary |
| Gated vs copy-input, seeds 0–2 (for matched comparisons) | +6.35 / +9.98 / 0.00% (+4.47 / +7.30 / 0.00) | headline_r2 |
| B1 (imitation) vs copy-input, D1 | −24.9% (5 seeds) | five_seed_summary |
| Gate precision D1 / D3 | 0.823 / 0.904 | localization |
| Head AP lift / AUROC, 5 seeds: D1 / D3 / D2 | 5.8× .724 / 4.6× .726 / 1.6× .636 | five_seed_summary |
| Two-sided judge AP lift / AUROC: D1 / D3 / D2 | 6.2× .838 / 5.9× .812 / 2.7× .688 | zero_shot_mtc |
| Degraded D2: fraction / judge AUROC | 2.5% (4.4%) / .775 (.779) | controls/strata |
| Transfer: D1→D3 / D3→D1 / D1→D2 / D3→D2 | +7.25 / +4.80 / −1.71 / −2.01% | transfer |
| Judge + head gate: D1 / D3 / D2 | +7.91 / +11.80 / −0.39% (+6.52 / +8.71 / −0.46) | judge_gate |
| Surprisal input: AUROC / gated | .820 / +5.14% | controls |
| Plausibility recovered on edited melodies, D1 / D3 | 30.1% / 47.5% | quality |
| Oracle lift D1 / D2 | 49.3% / 21.5% (single run) | diagnostics |
| Run-to-run variation | ≤ 1.17 pp per seed, ≤ 0.36 pp per cell | headline vs headline_r2 |
| Listening study: raters / sessions / ratings per excerpt | 6 / 17 / 3–4 | listening/report |
| Listener naturalness: D3 / D1 / ep2 / D2 / reference | 2.56 / 3.65 / 3.43 / 3.66 / 4.06 | listening/report |
| Listener hit rate: D3 / D1 / ep2 / D2 | 78 / 25 / 28 / 19% | listening/report |
| Listener AUROC: D3 / D1 / ep2 / D2; judge on same notes | .77 / .59 / .62 / .54; .89 / .78 / .83 / .73 | listening/report |
| Naturalness vs ΔNLL: ρ all / drafts | −.40 / −.26 | listening/report |
| Pairs, repair vs reference: repair / reference / none | 42 / 40 / 54 (51% of decided, p = .91) | listening/report |
| Pairs, repair vs draft: repair / draft / none | 30 / 0 / 4 | listening/report |

## 10. Open items before and during drafting

- ~~**Step 8**~~ — done 2026-09-25: D1 +6.06% (5/5 seeds), from +6.35% on 3.
- **Step 10** — related work. Position against MDT (its degradation-*detection* task is
  exactly where C8's circularity measurement applies — word carefully), RefinPaint
  (where-to-edit), sequence-editing models (LaserTagger / EditNTS / copy mechanisms),
  selective prediction / abstention, SWE-bench-style patch evaluation as the cross-domain
  analogue of "difference is not error".
- **Figure 1** (proposed): x-axis = draft source ordered by ΔNLL per unit change, two
  y-series — two-sided-judge AUROC and gated gain over copy-input — with D2 at the floor.
  It is the thesis in one picture.
- **Determinism** — set `torch.use_deterministic_algorithms` for any new runs.
- **Listening study (C15), in the paper since 2026-10-12** (§7 with Table VIII; abstract,
  §1, §4 and §8 aligned):
  - The abstract clause reads "In a blinded listening study, listeners locate rule-based
    errors but not a generator's changes, and hear those repairs as different, not
    wrong." Never "and a small listening study agrees", which overstates the result.
  - The paper's working assumption, that listeners agree with the judge, fails on D1;
    the paper says so in §7 and §8.
  - Rerun `scripts/listening_report.py` if more rounds arrive. It reads every file in
    `results/human_test/` and fails on conflicting duplicates.
