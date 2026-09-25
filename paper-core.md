# Paper core

The definitive statement of what the paper claims, in what order, and on what evidence.
Draft sections are written from this file. `paper-framing.md` is the revision log that
got here — consult it for *why* a claim changed, not for what to claim.

Written 2026-09-25, after revision-plan steps 1–7 and 5b. Target: an 8-page venue (not
ISMIR; venue not yet named).

**How to use the tiers.** Every claim carries one:

- **P1 — core.** The paper is not coherent without it. Main text, always.
- **P2 — supporting.** Answers a predictable objection or makes P1 sharper. Main text if
  space allows; otherwise one sentence plus appendix.
- **P3 — appendix.** True, measured, worth having, first to cut.

If the draft runs long, cut P3 entirely, then compress P2 to single sentences with
pointers. Never cut a P1 claim's *caveat* to save space.

**Numbers.** Canonical values are listed in §9. All are MTC-ANN references, 3 seeds,
with the one setting chosen on D1 validation (binary head, `weighting=none`, α=0.5,
τ=0.8), unless stated otherwise. Figures marked **[S8]** will be updated to 5 seeds when
step 8 runs; the framing does not depend on them.

---

## 1. Thesis

> **Localize-then-fix refinement works only where errors leave a local trace — and
> model-generated drafts leave little.** On synthetic corruption, a precision-gated edit
> model improves drafts by ~6% over leaving them alone, and is the only system tested
> that improves both edit distance and musical plausibility. On drafts written by a
> generator, every detector we built — trained or untrained, per note or per span —
> localizes the generated material at no better than 2.7× chance, against 6–7× for
> corruption. Where a generated draft is genuinely worse, the damage *can* be found;
> most generated drafts are simply different, not wrong.

Three things the paper therefore contributes, in one line each:

1. **A measurement:** Correction Cost cannot tell an error from a valid alternative, and
   a calibrated plausibility judge can.
2. **A boundary:** where localize-then-fix applies (off-manifold errors) and where it
   does not (generator output), mapped from both sides under one configuration.
3. **A method that respects the boundary:** precision-gated editing, which improves
   drafts where errors are local and abstains where they are not.

## 2. Title

Recommended:

- **Generated Drafts Leave Little Local Trace: When Does Localize-and-Fix Work for
  Symbolic Music Refinement?**

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
(§6 L1).

**C2. Correction Cost cannot tell an error from a valid alternative.** (§4)
Human variants sit 92% normalized edit distance from each other at +0.038 ΔNLL per unit
of change; random corruption sits 7% away at +3.94 — ~100× more implausible per unit.
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
| D1 random | 5.8× / 0.727 | 6.2× / 0.838 |
| D3 rule-based | 5.0× / 0.743 | 5.9× / 0.812 |
| **D2 generated** | **1.6× / 0.635** | **2.7× / 0.688** |

Essen replicates every row. The D2 head collapses to KEEP — 0 positions clear τ on any
seed — although D2 has *more* true edits (7.3% of positions) than D1 (6.0%). A span
detector (1.6× chance top-1) and heads transferred from D1/D3 (1.3–1.4×) do no better.
*Source:* `results/refiner/headline_r2/{localization,zero_shot_mtc,zero_shot_essen}.json`,
`results/refiner/transfer/`.
*Caveat:* this is an absence of evidence across the detectors tried, not impossibility.
State the detectors, and that the strongest is untrained.

**C4. Most generated drafts are different, not wrong.** (§7)
Only 2.5–4.4% of D2 drafts are certified-degraded (ΔNLL > τ). On those, the two-sided
judge localizes at AUROC 0.78 / 4.5× — near D1 — while the head stays flat (~0.64)
across every degradation band. So where a generator makes a genuine error, it is
findable; the rest is divergence that edit labels mislabel as error ("difference is not
error").
*Source:* `results/refiner/controls/strata.json` (fresh 5,339-draft MTC set + 5,057
Essen; n = 131 / 222 degraded).
*Caveat:* bands are defined by `J`'s ΔNLL and scored by the two-sided judge, trained on
the same data, so part of the rise is built in. The head's flatness has no such link.

**C5. Precision-gated editing improves drafts where errors are local.** (§5, §6)
Copy each event by default; regenerate it only when the head's P(edit) ≥ τ. At τ=1 the
output is the draft verbatim, so the method's floor is copy-input by construction.
- D1: **+6.35% over copy-input** [S8] (per seed 5.87 / 6.83 / 6.34; win 23.0%, loss 6.3%);
  same-corpus **+4.47%**. Of the edits it makes, 82% are true errors.
- It beats the imitation baseline B1 by a wide margin: B1 is 24.7% *worse* than copying.
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
of change (3.94 > 2.97 > 1.16). A trained head adds learned regularity on top, which
transfer isolates: a D1-trained head reaches +7.25% on D3 against D3's own +9.98%, so
**~2.7 of D3's ~10 points come from learning the injection rules**. D3 is therefore a
partly circular benchmark — report it, as a control, with that number in the caption.
*Source:* `zero_shot_*.json`, `results/refiner/transfer/`.
*Caveat:* three sources is a thin basis for "follows"; say "consistent with".

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
Two-sided judge vs trained head, AUROC: D1 0.838 vs 0.727, D3 0.812 vs 0.743, D2 0.688
vs 0.635, both corpora. The head is shown at its best (standardizing its scores lowers
them). A note's plausibility given both sides is a stronger error signal than anything
the supervised head learned.
*Caveat:* each judge is one trained model; heads are 3 seeds.

**C11. An upper bound.** (§6)
Given ground-truth edit locations the same decoder removes 49.3% of D1's error and 21.5%
of D2's. The head gate recovers ~13% of the D1 headroom. D2's oracle result also shows
the generated spans are *correctable* once located — the bottleneck is finding them.
*Source:* `results/refiner/diagnostics_{D1,D2infill}.json` (single run, same test sets).

**C12. The imitation baseline converges to copying.** (§6)
Plain seq2seq refinement is worse than copying the draft (−24.7% on D1). Adding an
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
*Source:* `results/refiner/controls/strata.json`.

**C14. The confound checks.** (§7, one paragraph)
- Same-corpus control (judge, generator and references all Essen): both halves
  replicate; D2 localization equally weak on both corpora.
- Capacity: localization precision flat from 1.6M to 42.7M parameters; gated beats
  copy-input in 9/9 size configurations [earlier τ=0.9 setting, ranked 2nd of 96 on
  validation].
- Weighting and threshold: flat landscape (C5).

### P3 — appendix, first to cut

- **Head-type ablation.** A five-class head (where *and* how) matches a binary one
  (seed-matched +0.003 ± 0.011); INSERT_BEFORE F1 0.000. Justifies the binary head. One
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
8. "D3 is circular" without qualification → ~2.7 of ~10 points (C8).
9. "Raising recall hurts because precision is binding" → the decoder degrades (C9).
10. "Edit-awareness hurts on random corruption" → an artifact of class weighting.

Also avoid: any human-in-the-loop or user-utility claim beyond one motivating sentence;
"reproduces the published result" (C12 framing note).

## 6. Limitations to state

- **L1.** ΔNLL is statistical implausibility under a small LM; calibrated against human
  variants, but not a listening study. It also rewards typical continuations (C7).
- **L2.** Two folk corpora (MTC-ANN, Essen); monophonic; the style claim is narrow.
- **L3.** The negative result is bounded by the detectors tried — including the
  strongest we could build without labels.
- **L4.** The judge scores MTC with a 0.1365 nats/token domain gap; the same-corpus
  control rules out a confound, not the sensitivity.
- **L5.** Runs are not bit-reproducible (C5 caveat); results are reported over seeds and
  one full re-run.
- **L6.** D2 drafts are not filtered by the judge; C4 is the argument for why filtering
  would not change the localizability conclusion.

## 7. Anticipated objections (short form)

| objection | answer |
|---|---|
| Your method only works on synthetic corruption. | Yes; that is the finding, mapped from both sides (C3, C6). |
| A better detector would find D2's errors. | The strongest we built is untrained, beats every trained head, and reaches 2.7× on D2 vs 6–7× elsewhere (C3, C10). |
| Models too small / badly tuned. | Same machinery gets 5.8× on D1; flat across 26× parameters and 96 settings (C5, C14). |
| D2 drafts aren't really degraded. | Mostly true, and the point (C4). |
| +6% is small. | It is over copy-input, which the imitation baseline loses to by 25% (C5, C12). |
| CC tracks plausibility in your own table. | Within a source; not across (C2). |
| ΔNLL isn't musical correctness. | Conceded; calibrated on human variants (C1, L1). |
| D3 is circular. | Partly; transfer measures how much (C8). |

## 8. Dropped from the original spec (state in §3 or appendix)

Data-size ablation; mixed-source training; D3→D4 transfer (D4 is a calibration control,
not a refinement target); weighted CC variant; per-operation CC breakdown; scale-matched
B2 (the head is 1,285 parameters, 0.02%, so parameter matching is moot); the judge
degradation filter on D2 (L6); human-in-the-loop framing.

## 9. Canonical numbers

MTC, 3 seeds, validation-selected setting, unless noted. Essen in brackets.

| quantity | value | source |
|---|---|---|
| D4 variants: ΔNLL mean / median / % positive | +0.054 / +0.031 / 55.0% | validator_report |
| τ (95th pct of variant ΔNLL) | 0.585 | validator_report |
| ΔNLL per unit change: D4 / D1 / D3 / D2 (T=1.0) | 0.038 / 3.94 / 2.97 / 1.16 | validator, d2_report |
| Spearman ρ(surface, ΔNLL): D1 / D3 / D2 / D4 / pooled | +.52 / +.47 / +.20 / +.14 / −.06 | decoupling |
| Gated vs copy-input: D1 / D3 / D2 | +6.35% [S8] / +9.98% / 0.00% (+4.47 / +7.30 / 0.00) | headline_r2 |
| B1 (imitation) vs copy-input, D1 | −24.7% | headline_r2 |
| Gate precision D1 / D3 | 0.823 / 0.904 | localization |
| Head AP lift / AUROC: D1 / D3 / D2 | 5.8× .727 / 5.0× .743 / 1.6× .635 | localization |
| Two-sided judge AP lift / AUROC: D1 / D3 / D2 | 6.2× .838 / 5.9× .812 / 2.7× .688 | zero_shot_mtc |
| Degraded D2: fraction / judge AUROC | 2.5% (4.4%) / .775 (.779) | controls/strata |
| Transfer: D1→D3 / D3→D1 / D1→D2 / D3→D2 | +7.25 / +4.80 / −1.71 / −2.01% | transfer |
| Judge + head gate: D1 / D3 / D2 | +7.91 / +11.80 / −0.39% (+6.52 / +8.71 / −0.46) | judge_gate |
| Surprisal input: AUROC / gated | .820 / +5.14% | controls |
| Plausibility recovered on edited melodies, D1 / D3 | 30.1% / 47.5% | quality |
| Oracle lift D1 / D2 | 49.3% / 21.5% (single run) | diagnostics |
| Run-to-run variation | ≤ 1.17 pp per seed, ≤ 0.36 pp per cell | headline vs headline_r2 |

## 10. Open items before and during drafting

- **Step 8** — 5 seeds on the head-gate headline, when the user says go. Updates [S8].
- **Step 10** — related work. Position against MDT (its degradation-*detection* task is
  exactly where C8's circularity measurement applies — word carefully), RefinPaint
  (where-to-edit), sequence-editing models (LaserTagger / EditNTS / copy mechanisms),
  selective prediction / abstention, SWE-bench-style patch evaluation as the cross-domain
  analogue of "difference is not error".
- **Figure 1** (proposed): x-axis = draft source ordered by ΔNLL per unit change, two
  y-series — two-sided-judge AUROC and gated gain over copy-input — with D2 at the floor.
  It is the thesis in one picture.
- **Determinism** — set `torch.use_deterministic_algorithms` for any new runs.
