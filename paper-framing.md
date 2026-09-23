# Paper framing plan

Companion to `edit-aware-revision-spec.md`. The spec says what to *build*; this says
what to *claim*, and which measurement backs each claim.

Rewritten 2026-09-22 after span-level detection failed. The earlier version expected
span-level detection to rescue the realistic-draft setting; it did not, and the thesis
below is the one the evidence supports.

---

## 1. Thesis

> **Apparent progress on symbolic draft refinement is largely a function of how the
> drafts were made.** Across three sources under identical machinery, localizability
> collapses as drafts become more realistic — rule-based degradation 0.776 precision,
> random corruption 0.728, generator output 0.137 — and measured performance falls with
> it, +9.85% to +6.04% to +0.00%. Generated drafts are the realistic case, and they are
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
   +0.031, 53.3% positive — a coin flip), so the judge measures degradation rather than
   novelty.

2. **Correction Cost and musical plausibility are near-orthogonal.** Human variants of
   one tune sit ~92% normalized edit distance apart at +0.038 ΔNLL per unit of change;
   synthetic corruption sits ~7% away at +3.94 — roughly 100× more implausible per unit
   changed. Corruption moves a melody *off* the data manifold; real variation moves it a
   long way *along* it.

3. **Localizability collapses as drafts become more realistic — and apparent method
   performance falls with it.** Three draft sources, same machinery, three seeds each:

   | source | errors come from | ΔNLL/unit | loc precision | gated vs B3 |
   |---|---|---|---|---|
   | D3 rule-based | rules we wrote | 2.97 | 0.776 | +9.85% |
   | D1 random | unsystematic noise | 3.94 | 0.728 | +6.04% |
   | D2 generated | a real generator | 0.90 | 0.137 | +0.00% |

   **D3 is a negative control, not a result.** Its edits come from the same rule family
   a detector can learn — out-of-scale, unsingable leap, cadential disruption — so it
   reports the best number in the paper precisely because the benchmark is circular.
   Report it in the ablation, keep it out of headline claims, and put the circularity in
   the caption. Spec §3 proposed rule-based degradation as the *fix* for D1's unrealism;
   the result is that it is worse in a different way — more realistic-sounding errors,
   less valid as a benchmark. Anyone about to build one needs to know that.

   D3 also earns its place by falsifying the two obvious accounts of the boundary. It is
   *less* off-manifold than D1 (2.97 against 3.94) and *less* point-like (mean edited run
   1.50 against 1.05), yet *more* localizable. So neither manifold distance nor edit
   locality governs on its own. The working hypothesis — held as a hypothesis, since two
   earlier accounts were already falsified — is that what matters is whether the error
   has a **learnable regularity**: off-manifold is necessary but not sufficient, and the
   deviation must also be systematic enough to train a detector for. D3's errors are
   rule-shaped, D1's are unsystematic, D2's are in-distribution by construction.

   **The D1 result stays the headline** (+6.04%, replicated same-corpus, 9/9 size
   configurations) precisely because random corruption is not something we designed a
   detector for.

4. **Generated material is not localizable, at any granularity tried.** Three
   independent measurements on D2-infill drafts:

   | method | result |
   |---|---|
   | per-position edit head | ~1.5× chance, **flat across degradation bands** (1.56× on ΔNLL ≤ 0, 1.66× on judge-certified degraded) |
   | judge surprisal pooled over spans, zero-shot | 1.2× chance |
   | trained span detector, interior + both boundary states, 3 seeds | 1.6× chance (0.75% ± 0.19% top-1 against 0.48% chance) |

   **The decisive control: the same architecture, size and recipe reaches localization
   precision 0.728 on D1.** The difference is in the drafts, not the method — this is
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

5. **The published result reproduces, and is an artifact.** The U-shaped α curve
   replicates with its minimum at α=1.0 and a 5.85% CC reduction (published: 6.3%), but
   at that optimum fidelity is 0.0007 and CC equals copy-input: the curve measures
   convergence to copying. Visible only with a copy-input baseline and a fidelity
   metric, neither of which the original evaluation had. **Prescription: always report
   B3 and fidelity alongside CC.**

6. **Precision-gated decoding, where errors are local.** Copy by default; hand an event
   to the decoder only when the head's confidence clears τ. **At τ=1 the output is the
   draft verbatim, so the method provably cannot do worse than leaving the draft
   alone.** +6.04% ± 0.014 over copy-input on D1 (3/3 seeds, p=4.2e-52), replicated
   same-corpus at +5.21% ± 0.020 on a 5× larger test set. On D2 it returns ≈0.00% — it
   abstains, which is correct given precision 0.137.

7. **The gain holds across capacity, and the gate bounds the damage.** Nine
   configurations, 2/4/8 layers × d_model 128/256/512, 1.6M to 42.7M parameters (26×),
   three seeds each. **Gated editing beats copy-input in 9/9 configurations on every
   seed**, +3.78% to +6.99%, mean +5.85%.

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

8. **Precision, not recall, is the binding constraint — with an oracle bound.** Given
   ground-truth labels the model removes 49.3% of the draft's error (D1); the real head
   recovers 12.2%. Raising *recall* (judge surprisal, 0.245 → 0.314) made CC *worse*,
   because at precision 0.678 a third of flagged positions are notes that were already
   correct.

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

Four things the evidence does **not** support, each of which was claimed at some point
while this work was being done:

- **Not "where and how".** Binary matches 5-class (seed-matched +0.003 ± 0.011 on MTC,
  +0.012 ± 0.020 same-corpus, 2/3 seeds each way). Per-class F1: INSERT_BEFORE **0.000**,
  DELETE 0.107, SUB_DUR 0.194. Claim **"where"** — precision-gated *localization* — and
  report the typed head as an ablation that did not pay off. Recommend the binary head.
- **Not "edit-awareness lowers Correction Cost".** It reliably improves *where the model
  looks* (macro-F1 0.092 → 0.356, 3/3 seeds, ±0.004) and does **not** reliably improve
  what it produces (CC vs B1 straddles zero, 2/3 seeds, ±0.404). Report the dissociation.
- **Not "edit-awareness hurts on random corruption".** An artifact of inverse-frequency
  class weighting, which paid the head ~105× more for flagging than for leaving a note
  alone. Fixed with `weighting=none` on D1.
- **Not "span-level detection recovers what per-position missed".** It does not: 1.6×
  chance, barely above the untrained judge's 1.2×.

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
  domain gap). The same-corpus control rules out the confound, and the dissociation is
  *stronger* there (D2 precision 0.220 → 0.137), which is the theory's own prediction.

## 7. Anticipated reviewer objections

| objection | answer |
|---|---|
| "Your method only works on the synthetic setting you criticise." | Yes — and that is the finding, stated as the title. The boundary is mapped from both sides, and the negative side is measured three independent ways. |
| "Your detector is just weak / your models are too small." | Three answers. The same architecture, size and recipe reaches precision 0.728 on D1 — the difference is the drafts. Localization precision is flat from 1.6M to 42.7M parameters (corr with log-params −0.58), so it is information-limited, not capacity-limited. And the span task was given the true span-length prior and still failed. |
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

## 9. Remaining work

1. ~~Model-size ablation~~ — done, 9/9 configurations; see contribution 6.
2. **P6 related work** — McLeod et al. (MDT, ISMIR 2020) and Ramoneda et al. (RefinPaint,
   ISMIR 2024) need explicit positioning with a stated delta; plus Ji, Yang & Luo (2023),
   Lakh provenance, Vaswani et al. Zero citations in the introduction was a stated
   rejection reason.
3. ~~D3 refinement sweep~~ — done; it falsified both accounts of the boundary and is
   reframed as a negative control (contribution 3).
4. 5 seeds on the headline (seed 0 was an outlier once).
5. `scripts/` have no test coverage — that is how a broken-argparse bug reached a launch.
