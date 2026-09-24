# Paper framing plan

Companion to `edit-aware-revision-spec.md`. The spec says what to *build*; this says
what to *claim*, and which measurement backs each claim.

Rewritten 2026-09-22 after span-level detection failed. The earlier version expected
span-level detection to rescue the realistic-draft setting; it did not, and the thesis
below is the one the evidence supports. Updated 2026-09-23 with the within-D3 locality
stratification, which rules out edit contiguity as the account of the boundary
(contribution 4). Numbers reconciled against the result files 2026-09-23 (§9 step 1),
and the headline re-run under validation-selected settings 2026-09-24 (§9 steps 2–3).
The framing itself is still to be revised (§9 step 9).

**Reporting conventions.** Unless stated otherwise: MTC-ANN references, three seeds,
and **one setting for every source and corpus — binary head, `weighting=none`, α=0.5,
τ=0.8 — selected on D1 *validation* data** (`scripts/select_settings.py`; test decoded
at τ\* only). Canonical numbers are `results/refiner/headline_r2/`. "Gated vs B3" is
(CC_B3 − CC_gated) / CC_B3 per seed, averaged over seeds; spread is the sample sd of the
per-seed percentages, in percentage points. Localization is compared threshold-free —
AP lift (average precision ÷ base rate) and AUROC of the head's P(edit) — because
argmax precision is undefined for a head that barely flags. **Gate precision** is
precision among the positions the gate actually edits.

Runs are not bit-reproducible: cuDNN determinism is not set, so an identical seed gives
a slightly different model. Two full runs of the headline differ by up to 1.17 pp on one
seed and 0.36 pp on a cell mean (`headline/` vs `headline_r2/`); state this.

---

## 1. Thesis

> **Apparent progress on symbolic draft refinement is largely a function of how the
> drafts were made.** Across three sources under identical machinery, localizability
> collapses as drafts become more realistic — AUROC 0.74 for rule-based degradation,
> 0.73 for random corruption, 0.64 for generator output — and measured performance
> falls with it, +9.98% to +6.35% to +0.00%. Generated drafts are the realistic case, and they are
> **not locally wrong**: the material the generator wrote cannot be identified per note,
> per span, by a trained detector, or by the judge's own surprisal. *Localize-then-fix*
> is the wrong formulation for that class of draft. Where errors genuinely are local,
> the same machinery works and precision-gated editing beats copy-input by 6%.

The paper's subject is therefore **the domain of validity of the localize-then-fix
formulation**, mapped from both sides. That subsumes the ISMIR reviewers' P1 objection
instead of conceding it: random corruption is indeed a poor model of revision, and the
obvious fix — use a generator — produces drafts nothing can localize.

## 2. Contributions, in the order they should appear

1. **A calibrated degradation validator.** An independently trained melody LM plus
   musical criteria, anchored on human-authored tune-family variants. The anchor is what
   makes it credible: genuinely valid alternatives score ΔNLL ≈ 0 (mean +0.054, median
   +0.031, 55.0% positive — a coin flip; `validator_report.json`, n=1,012 pairs), so the judge measures degradation rather than
   novelty.

2. **Correction Cost and musical plausibility are near-orthogonal.** Human variants of
   one tune sit ~92% normalized edit distance apart at +0.038 ΔNLL per unit of change;
   synthetic corruption sits ~7% away at +3.94 — roughly 100× more implausible per unit
   changed. Corruption moves a melody *off* the data manifold; real variation moves it a
   long way *along* it.

3. **Localizability collapses as drafts become more realistic — and apparent method
   performance falls with it.** Three draft sources, same machinery, three seeds each:

   | source | errors come from | ΔNLL/unit | edit rate | AP lift | AUROC | gate flags / P | gated vs B3 |
   |---|---|---|---|---|---|---|---|
   | D3 rule-based | rules we wrote | 2.97 | 8.2% | 5.0× | 0.743 ± 0.038 | 1,968 / 0.904 | +9.98% ± 0.30 pp |
   | D1 random | unsystematic noise | 3.94 | 6.0% | 5.8× | 0.727 ± 0.005 | 1,489 / 0.823 | +6.35% ± 0.48 pp |
   | D2 generated | a real generator | 1.16 | 7.3% | 1.6× | 0.635 ± 0.003 | **0** / — | +0.00% ± 0.00 pp |

   Same-corpus (Essen references): D3 4.3× / 0.718 / +7.30%, D1 5.8× / 0.732 / +4.47%,
   D2 1.8× / 0.654 / +0.00%. Every D1 and D3 cell is 3/3 seeds on both corpora.

   **This is now one configuration end to end**, which the earlier table was not (it
   gave D2 a different head and weighting). Under it the D2 head **collapses to KEEP**:
   0 of ~52k positions clear τ on any seed, and argmax flags number 0–16 per seed
   against D1's ~2,300 — although D2 has *more* true edits (7.3% of positions against
   6.0%). Same loss, more positives, and the head finds nothing worth flagging. This
   is cleaner evidence than the old precision 0.137, because nothing about the method
   differs between the sources.

   D2 is not information-free: AUROC 0.64 and AP lift 1.6–1.8× show a weak ranking
   signal, the same ~1.5× the earlier measurements found. It is too weak for any
   position to be worth editing.

   ΔNLL/unit: D1/D3 from `validator_report.json` (n≈1,080); D2 from the `D2-infill
   T=1.0` row of `d2_report.json` (n=352), the temperature the refiner's drafts used.

   **D3 is a negative control, not a result.** Its edits come from the same rule family
   a detector can learn — out-of-scale, unsingable leap, cadential disruption — so it
   reports the best number in the paper precisely because the benchmark is circular.
   Report it in the ablation, keep it out of headline claims, and put the circularity in
   the caption. Spec §3 proposed rule-based degradation as the *fix* for D1's unrealism;
   the result is that it is worse in a different way — more realistic-sounding errors,
   less valid as a benchmark. Anyone about to build one needs to know that.

   D3 also bears on the two obvious accounts of the boundary. It is *less* off-manifold
   than D1 (2.97 against 3.94) and *less* point-like (mean edited run 1.50 against
   1.05), yet **no less localizable**. **Revised 2026-09-24:** earlier versions said
   "more localizable", from argmax precision. Threshold-free the two are level (AUROC
   0.743 vs 0.727 MTC, 0.718 vs 0.732 Essen; AP lift is lower for D3, 5.0× vs 5.8×).
   D3 wins only in the **confident tail** — gate precision 0.904 against 0.823 — and
   that is where its larger gated gain comes from. So D3 does not *falsify* manifold
   distance; it shows that a source 25% less off-manifold is just as localizable, which
   is evidence against manifold distance as the *sole* governor, not a refutation. The
   locality account is refuted separately (contribution 4). So neither manifold distance
   nor edit locality governs on its own. The working hypothesis — held as a hypothesis, since two
   earlier accounts were already falsified — is that what matters is whether the error
   has a **learnable regularity**: off-manifold is necessary but not sufficient, and the
   deviation must also be systematic enough to train a detector for. D3's errors are
   rule-shaped, D1's are unsystematic, D2's are in-distribution by construction.

   **The D1 result stays the headline** (+6.35%, replicated same-corpus at +4.47%, 9/9
   size configurations) precisely because random corruption is not something we designed a
   detector for.

4. **The boundary is not about contiguity.** The obvious alternative account of
   contribution 3 is mechanical rather than epistemic: a per-position head suits isolated
   changes and is blind to contiguous ones, and D2 happens to replace spans of 2–6 events
   while D1 alters single notes. Comparing sources cannot decide this, because D1 and D2
   differ in manifold distance *and* locality at once. D3 can, being uniformly
   off-manifold but mixed in locality; splitting its own test set by the longest edited
   run per draft holds the first fixed and varies the second.

   | band | n | loc P | loc R | gated gain (CC) |
   |---|---|---|---|---|
   | point-like (1 event) | 544 | **0.825** | 0.427 | +0.441 |
   | 2 adjacent events | 262 | 0.806 | 0.251 | +0.405 |
   | span-like (3+ events) | 188 | **0.805** | 0.210 | +0.356 |

   **Precision is flat — +0.019 end to end — so contiguity is not what breaks
   localization.** And the comparison that settles it: D3's span-like band is contiguous
   in exactly the way D2's infilled spans are, yet holds precision **0.805**, while D2's
   head under a matched setting flags nothing at all (AUROC 0.64 against D3's 0.74).
   *These strata were measured at the earlier τ=0.9 / argmax-precision setting; §9 step
   6 re-runs them under the validation-selected setting, threshold-free.* What contiguity does cost is *recall*, which halves (0.427 → 0.210) as the
   head flags part of a long run and misses the rest — the same recall/precision
   dissociation as contribution 9, and precision is the binding constraint.

   This is worth reporting as a result and not only as a control. It says the failure is
   a property of the *drafts*, not an artifact of per-position output granularity — which
   is the same conclusion the span-detector null reached (contribution 5) from the
   opposite direction, by changing the granularity and getting nothing. One seed, D3
   only; runs measured in event space, since each event carries two like-labelled tokens
   and a token-space run would classify every single-note edit as span-like.

5. **Generated material is not localizable, at any granularity tried.** Three
   independent measurements on D2-infill drafts:

   | method | result |
   |---|---|
   | per-position edit head | ~1.5× chance, **flat across degradation bands** (1.56× on ΔNLL ≤ 0, 1.66× on judge-certified degraded) |
   | judge surprisal pooled over spans, zero-shot | 1.2× chance |
   | trained span detector, interior + both boundary states, 3 seeds | 1.6× chance (0.75% ± 0.19% top-1 against 0.48% chance) |

   **The decisive control: the same architecture, size, recipe and setting reaches AP
   lift 5.8× and AUROC 0.73 on D1, against 1.6× and 0.64 on D2** — on both corpora. The difference is in the drafts, not the method — this is
   not a capacity or tuning failure. The span task was also given an unfair advantage
   (candidates restricted to the true span-length range, which a deployed system would
   not know) and still failed.

   A tempting alternative explanation was tested and rejected: that most drafts simply
   are not degraded, so there is nothing to find. Certified-degraded drafts are no more
   localizable than undegraded ones.

   Why: the damage is distributed. Around a spliced span the *unchanged* notes also
   become dearer under the judge (+0.0677) alongside the span itself (+0.2162). The
   generator writes a locally coherent passage; what is wrong is how it sits in the
   piece.

6. **The published result reproduces, and is an artifact.** On D2-infill drafts the
   U-shaped α curve replicates: the CC reduction over B1 peaks at α=0.5–1.0 at
   **+6.3%** (α=1.0, per seed 11.6 / 3.5 / 3.9%, 3/3 positive; published: 6.3%), and
   turns negative at α=3.0. But at that optimum fidelity is 0.0009 per note and CC is
   4.726 against copy-input's 4.709: the curve measures convergence to copying. On D1 —
   the analogue of the published random-corruption protocol — the gain over B1 does
   **not** reproduce reliably (α=0.5: +6.3 / +19.3 / −7.2%, 2/3 seeds), so "reproduces"
   must be stated for D2 only. (An earlier version quoted 5.85%, which is the
   size-ablation mean pasted in error.) Visible only with a copy-input baseline and a fidelity
   metric, neither of which the original evaluation had. **Prescription: always report
   B3 and fidelity alongside CC.**

7. **Precision-gated decoding, where errors are local.** Copy by default; hand an event
   to the decoder only when the head's confidence clears τ. **At τ=1 the output is the
   draft verbatim, so the method provably cannot do worse than leaving the draft
   alone.** **+6.35% over copy-input on D1** (per seed 5.87 / 6.83 / 6.34%; win 23.0%,
   loss 6.3%), with the binary head and settings chosen on validation; replicated
   same-corpus at **+4.47%** (4.10 / 4.75 / 4.55%) on a 5× larger test set. A second full
   run gave +5.99% and +4.55%. Among the positions the gate edits, 82% are true errors.
   Validation selection over 96 settings found the gain flat — every weighting peaks at
   6.3–6.5% on validation, 84/96 settings beat copy-input on all seeds — so the result
   is not a product of tuning. Report per-seed values, not a pooled Wilcoxon p, which
   treats melodies × seeds as independent. On D2 it returns exactly 0.00% — no position
   clears τ on any seed, so it abstains, which is correct given AUROC 0.64.

8. **The gain holds across capacity, and the gate bounds the damage.** Nine
   configurations, 2/4/8 layers × d_model 128/256/512, 1.6M to 42.7M parameters (26×),
   three seeds each. **Gated editing beats copy-input in 9/9 configurations on every
   seed**, +3.78% to +6.99%, mean +5.85%. *(Run at the earlier none / 0.5 / 0.9 setting,
   which validation ranks 2nd of 96, 0.07 pp behind the selected one; not re-run.)*

   Two things in that sweep matter more than the headline. **Localization precision is
   flat across the whole range** — 0.655 to 0.719, correlation with log-parameters
   −0.58, i.e. if anything it *falls* with scale. Localization is information-limited,
   not capacity-limited. That is a third independent argument against "your models are
   too small," alongside the D1/D2 same-architecture control: precision does not improve
   with 26× more parameters even where the task *is* solvable.

   And **the gate bounds the damage**. Free decoding ranges over CC 3.42–9.04 and is
   worse than doing nothing in **9/9** configurations; gated stays within 2.77–2.87 and
   beats doing nothing in 9/9. At 1.6M parameters free decoding is three times worse
   than leaving the draft alone, while gated is still ahead of it. That is the τ=1
   guarantee doing real work, and for a deployable tool the graceful degradation is
   arguably worth more than the 6%.

10. **Gated editing is the only condition that improves both Correction Cost and
    plausibility — and free decoding shows why CC alone cannot be trusted.** The
    decoded outputs scored by the judge (`scripts/output_quality.py`; ΔNLL is output
    minus reference, "recovered" is the share of the draft's ΔNLL removed):

    | MTC | CC vs copy-input | ΔNLL | recovered | on edited melodies |
    |---|---|---|---|---|
    | D1 draft (B3) | 0 | +0.298 | — | — |
    | D1 B1 free | −24.7% | +0.176 | +40.9% | — |
    | D1 EA gated | **+6.35%** | +0.250 | +16.3% | **+30.1%**, 2.8% made less plausible |
    | D3 EA gated | +9.98% | +0.166 | +38.8% | +47.5%, 1.4% less plausible |
    | D2 B1 free | −3.0% | +0.135 | +2.7% | — |
    | D2 EA gated | +0.00% | +0.139 | 0 | gate never opens |

    Essen replicates every row (D1 gated edited subset +31.1%, 2.8% less plausible;
    D3 +53.3%, 0.7%; D2 free +5.8%).

    **The CC gain is not bought with plausibility.** Where the gate edits, the judge
    finds the melody more plausible in 97.6% of cases (MTC D1, seed 0) and removes 30%
    of the draft's implausibility. The overall +16% is diluted by the half of melodies
    the gate leaves alone.

    **Free decoding "wins" on plausibility while losing on CC — the dissociation of
    contribution 2, reproduced on model outputs.** It recovers 41% of the implausibility
    against gated's 16%, yet is 25% *worse* than copying on CC. It decodes greedily, so
    it regenerates toward the language model's mode: 14.5% of its outputs are *more*
    plausible than the reference itself (drafts: 5.2%), and a third of the melodies it
    changes end up both more plausible and further from the reference. A plausibility-
    only evaluation would crown free decoding; a CC-only evaluation hides that gated
    editing improves the music. **Report both; the claim for gated editing is that it is
    the only condition that beats the draft on both axes.**

    Two smaller findings. D2's implausibility resists even regeneration (free decoding
    recovers 3–6%), consistent with damage that is contextual rather than located. And
    gated editing leaves D1's metric misalignment untouched (Δ −0.117 against the
    draft's −0.119, where free decoding reaches −0.071): the gate acts on pitch errors,
    matching the head-type finding that duration substitutions are poorly predicted.

    Caveat: the judge is a language model and so shares free decoding's preference for
    typical continuations; the 14.5%-above-reference figure is itself evidence that ΔNLL
    rewards mode-seeking. The breakdown percentages (97.6%, 14.5%, a third) are from one
    seed of MTC D1.

9. **Precision, not recall, is the binding constraint — with an oracle bound.** Given
   ground-truth labels the model removes 49.3% of the draft's error (D1); the real head
   recovers 12.2%. Raising *recall* (judge surprisal, 0.245 → 0.314) made CC *worse*,
   because at precision ~0.68 a third of flagged positions are notes that were already
   correct. The surprisal comparison is **single-seed** (`D1_surp0.json`); §9 step 6.

## 3. Framing moves

**M1 — The boundary is the paper, not the method.** Lead with the validator and the
orthogonality finding, then the negative result, then the positive one as the
demonstration that the diagnosis is actionable where the formulation applies. The +6% is
evidence that the map is right, not the headline.

**M2 — "Difference is not error."** Edit labels derived by aligning a draft to a
reference encode *difference*, not *error*. Whenever the draft is a valid alternative,
the supervision marks arbitrary divergence. This transfers well beyond music — any
reference-derived edit supervision in code or text has the same hazard.

**M3 — Abstention is selective prediction.** "The gate declines when no edit is
confidently warranted" is a *feature* for a creative tool, where a wrong edit is worse
than no edit. With the τ=1 guarantee, bounded downside plus abstention is a deployable
property and more honest than claiming uniform improvement.

**M4 — Say plainly what the field should stop doing.** Two warnings, in order of how
easy they are to walk into. Rule-based degradation produces a circular benchmark: the
detector learns the injection rules, and the resulting number says nothing about musical
error. Generator-infilled drafts avoid that but are unlocalizable, so they are not a
drop-in fix either. Anyone wanting realistic drafts needs a different task formulation
(holistic rewriting, span-free) or human-annotated error locations.

**M5 — Report the number you distrust.** D3 gives the paper's best result and is in the
git history; reporting D1 and D2 while dropping it would be exactly the selective
reporting reviewers look for. Include it, demote it, and explain why it should not be
believed. That is more persuasive than omission and costs nothing.

## 4. Claim discipline

Five things the evidence does **not** support, each of which was claimed at some point
while this work was being done:

- **Not "where and how".** Binary matches 5-class (seed-matched +0.003 ± 0.011 on MTC,
  +0.012 ± 0.020 same-corpus, 2/3 seeds each way). Per-class F1: INSERT_BEFORE **0.000**,
  DELETE 0.107, SUB_DUR 0.194. Claim **"where"** — precision-gated *localization* — and
  report the typed head as an ablation that did not pay off. Recommend the binary head.
- **Not "edit-awareness lowers Correction Cost".** On D1 it does **not** reliably improve
  what free decoding produces (CC vs B1 at α=1.0 straddles zero, 2/3 seeds, sd 0.404
  CC). What it does supply is a usable localizer — head precision 0.65–0.69, localization
  F1 0.361 ± 0.005 — which gating then exploits. Do **not** present this as "improves
  where the model looks" against α=0: at α=0 the head receives no gradient, so that
  comparison is against a randomly initialized head (precision 0.06). Earlier versions
  quoted macro-F1 0.092 → 0.356; macro-F1 averages in KEEP's F1 of ~0.96 and should
  not be reported for localization.
- **Not "edit-awareness hurts on random corruption".** An artifact of inverse-frequency
  class weighting, which paid the head ~105× more for flagging than for leaving a note
  alone. Fixed with `weighting=none` on D1.
- **Not "span-level detection recovers what per-position missed".** It does not: 1.6×
  chance, barely above the untrained judge's 1.2×.
- **Not "the per-position head is blind to contiguous edits".** Tested within D3, where
  manifold distance is held roughly fixed: precision is 0.825 on point-like drafts and
  0.805 on span-like ones, a gap of +0.019. Recall does fall with run length; precision,
  the quantity gating depends on, does not. Do not explain D2 by span length.

## 5. Title (closes P2)

Name the boundary:

- *Generated Drafts Are Not Locally Wrong: Limits of Localize-and-Fix in Symbolic Music Refinement*
- *When Is a Draft Locally Wrong? Mapping the Domain of Edit-Aware Music Refinement*
- *Difference Is Not Error: Synthetic Corruption, Generated Drafts, and What Can Be Localized*

Any of these disposes of the human-in-the-loop framing for free. Collaborative utility
gets one sentence as motivation, never as a demonstrated result.

## 6. Caveats to state openly, not bury

- **ΔNLL is statistical implausibility under a small LM, not a musical judgement.** The
  human-variant calibration is the strongest defence available without a user study and
  should be presented as a limitation, in the open.
- Two folk corpora only; the style claim is narrow and should say so.
- The negative result is **an absence of evidence from the methods tried**, not a proof
  of impossibility. State the three methods, the D1 control that rules out capacity, and
  the unfair advantage given to the span task — then let the reader judge.
- `J` is trained on Essen and scores MTC in the headline numbers (0.1365 nats/token
  domain gap). The same-corpus control rules out the confound: both halves replicate,
  and D2 localization is equally weak on both corpora (AP lift 1.6× MTC, 1.8× Essen;
  AUROC 0.635 and 0.654). An earlier version claimed the dissociation was *stronger* same-corpus
  (0.220 → 0.137); the 0.220 came from a single-seed surprisal-input run and the claim
  is withdrawn.

## 7. Anticipated reviewer objections

| objection | answer |
|---|---|
| "Your method only works on the synthetic setting you criticise." | Yes — and that is the finding, stated as the title. The boundary is mapped from both sides, and the negative side is measured three independent ways. |
| "Your detector is just weak / your models are too small." | Three answers. The same architecture, size, recipe and setting reaches AP lift 5.8× on D1 against 1.6× on D2 — the difference is the drafts. Localization precision is flat from 1.6M to 42.7M parameters (corr with log-params −0.58), so it is information-limited, not capacity-limited. And the span task was given the true span-length prior and still failed. |
| "Your per-position head just can't represent multi-note edits." | Tested within D3, holding manifold distance roughly fixed: precision 0.825 point-like against 0.805 span-like (+0.019). D3's span-like band is contiguous exactly as D2's infills are and still scores 0.805, while D2's head under a matched setting flags nothing. Granularity was also varied directly, in the span detector, with the same null. |
| "Maybe the drafts aren't degraded, so there is nothing to find." | Tested and rejected: precision is flat across degradation bands; judge-certified degraded drafts are no more localizable. |
| "+6% is small." | It is over *copy-input*. The published +6.3% was over an imitation baseline itself 21.4% worse than copying. |
| "ΔNLL isn't musical correctness." | Conceded explicitly; calibrated against human variants; see §6. |
| "Thin evaluation." | 3 seeds with mean ± std, B1/B2/B3, fidelity, edit-F1, paired stats with bootstrap CI and Wilcoxon, same-corpus control, size ablation. |

## 8. Rejection reasons → status

| | status |
|---|---|
| P1 random corruption ≠ revision | **answered** — validator, orthogonality, and the boundary result |
| P2 human-in-the-loop framing | **retitle** per §5; no experiments needed |
| P3 "where and how" | **answered by narrowing** — claim "where" (§4) |
| P4 representation contradictions | **fixed** — REST/BAR, measured; onset shift removed from the vocabulary |
| P5 thin evaluation | **answered** — §7 final row, plus 9 size configurations × 3 seeds |
| P6 no citations in introduction | **outstanding** — pure writing, and an explicit rejection reason |

## 9. Revision plan

Worked one step at a time, each reviewed before the next. Steps 2–8 may change numbers
quoted above; update §1–§7 as each lands. Target venue is no longer ISMIR and the page
budget is 8 pages, so the contribution list need not be cut to three — but it should
still be ordered by what carries the thesis.

1. **Reconcile reported numbers against result files.** *Done 2026-09-23.* Docs only,
   no reruns. Fixed: the thesis/source table mixed MTC and Essen runs and 5-class and
   binary heads; D3 +9.85% was the τ=0.7 result; the same-corpus "stronger
   dissociation" rested on a surprisal-input run (withdrawn); the α-curve "5.85%" was
   the size-ablation mean; D2 ΔNLL/unit was the wrong temperature; D4's fraction
   positive came from a different file; the "where the model looks" comparison was
   against an untrained head. Two facts confirmed and still to be stated in the paper:
   **D2 training and evaluation data are not judge-filtered** (spec §3.2 made the filter
   mandatory), and **the class weighting was selected on the test set**.

2. ~~**Remove test-set selection**~~ *Done 2026-09-24.* Validation picks none / 0.5 /
   0.8; the old test-selected none / 0.5 / 0.9 ranks 2nd of 96, 0.07 pp behind, so the
   earlier selection did not inflate results. Headline re-run twice (`headline/`,
   `headline_r2/`; the second saves checkpoints, P(edit) and outputs). Found on the way:
   runs are not bit-reproducible — consider `torch.use_deterministic_algorithms` for
   anything run from here on. Original description: `weighting_ablation.py` picks the weighting
   by test macro-F1, and α=0.5 and τ=0.9 were chosen after seeing test results. Select
   all three on the validation split, by localization precision for τ and the gated CC
   for α, then rerun the headline gated evaluation (D1, D2, D3; MTC and same-corpus).
   Comes first because every later step inherits these settings.

3. ~~**D2 with a binary head**~~ *Done 2026-09-24, inside step 2's headline.* The D2 head
   collapses to KEEP under the global setting; localization is now compared
   threshold-free (`scripts/localization_report.py`).

4. ~~**Musical quality of outputs.**~~ *Done 2026-09-24 — contribution 10, `quality.json`.* Judge NLL, scale conformance and interval distance
   for B3 / B1 / free / gated / oracle outputs on D1, D2, D3. Needed because
   contribution 2 argues CC cannot carry an evaluation, while the positive result is
   currently CC-only. Check whether refiner checkpoints were kept; if not, rerun.

5. **Measure circularity rather than assert it.** (a) Zero-shot judge localization on
   D1 and D3 — the judge never saw either corruption rule. (b) Transfer: train on D1,
   test on D3, and the reverse. If D3's advantage disappears under transfer while the
   zero-shot judge does not favour it, the gap is the learnable injection regularity,
   quantified. This is the test the "learnable regularity" hypothesis currently lacks.

6. **Strengthen thin controls.** Degradation strata: 3 seeds, and enough drafts that
   the degraded band is not n=28. Locality strata: 3 seeds. Surprisal/recall
   comparison (contribution 9): 3 seeds.

7. **Decoupling statistic.** "Near-orthogonal" currently rests on two clusters, not a
   correlation. Compute rank correlation between CC and ΔNLL within and across
   sources, or retreat to "decoupled".

8. **5 seeds on the headline** (seed 0 was an outlier once).

9. **Framing rewrite.** The three-point "trend" is not monotone on the paper's own
   realism measure (ΔNLL/unit orders D1 > D3 > D2; precision orders D3 > D1 > D2), so
   present D1-vs-D2 as the contrast and D3 as the regularity control. "Not locally
   wrong" conflicts with the D2 oracle (+21.5% lift given locations) — use "leave no
   local trace" / "not locally detectable". State D2 is unfiltered and why. State the
   spec items deliberately dropped: data-size ablation, mixed-source training, D3→D4
   transfer, weighted CC, per-operation CC breakdown, and scale-matched B2 (the head is
   1,285 parameters, 0.02%).

10. **Related work (P6).** McLeod et al. (MDT, ISMIR 2020) has its own degradation-
    *detection* task, so the D3 circularity point applies to it directly — that is the
    stated delta, worded carefully. RefinPaint (Ramoneda et al., ISMIR 2024), Ji, Yang &
    Luo (2023), Lakh provenance, Vaswani et al.

11. *(Parallel, optional)* `scripts/` have no test coverage — that is how a broken-
    argparse bug reached a launch.
