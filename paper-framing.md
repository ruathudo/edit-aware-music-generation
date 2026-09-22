# Paper framing plan

Companion to `edit-aware-revision-spec.md`. The spec says what to *build*; this says
what to *claim*, and which measurement backs each claim. Written 2026-09-22.

The degradation stratification has now run. The hypothesis it tested — that D2
localization fails because most drafts are not actually degraded — is **not
supported**, and §2.6 records what replaced it.

---

## 1. Thesis

> Synthetic corruption makes symbolic music refinement look **easier** than it is;
> model-generated drafts make it look **harder** — for opposite reasons. Corruption
> produces off-manifold errors that are detectable but unrepresentative. Generators
> produce on-manifold alternatives that are representative but frequently are not
> errors at all.

This subsumes the ISMIR reviewers' P1 objection rather than conceding it: random
corruption is indeed a poor model of revision, *and* the obvious fix — use a
generator — relocates the problem instead of solving it.

## 2. Contributions, in the order they should appear

1. **A calibrated degradation validator.** An independently trained melody LM plus
   musical criteria, anchored on human-authored tune-family variants. The anchor is
   what makes it credible: genuinely valid alternatives score ΔNLL ≈ 0 (mean +0.054,
   median +0.031, 53.3% positive — a coin flip), so the judge measures degradation
   rather than novelty.
2. **Correction Cost and musical plausibility are near-orthogonal.** Human variants of
   one tune sit ~92% normalized edit distance apart at +0.038 ΔNLL per unit of change;
   synthetic corruption sits ~7% away at +3.94 — roughly 100× more implausible per unit
   changed. Corruption moves a melody *off* the data manifold; real variation moves it
   a long way *along* it.
3. **The published result reproduces, and is an artifact.** The U-shaped α curve
   replicates with its minimum at α=1.0 and a 5.85% CC reduction (published: 6.3%), but
   at that optimum fidelity is 0.0007 and CC equals copy-input: the curve measures
   convergence to copying. Visible only with a copy-input baseline and a fidelity
   metric, neither of which the original evaluation had. **This doubles as a
   methodological prescription: always report B3 and fidelity alongside CC.**
4. **Localization precision is the binding constraint, with an oracle bound.** Given
   ground-truth labels the model removes 49.3% of the draft's error (D1); the real head
   recovers 12.2%. Raising *recall* (via judge surprisal: 0.245 → 0.314) made CC
   *worse*, because at precision 0.678 a third of flagged positions are notes that were
   already correct.
5. **Precision-gated decoding.** Copy by default; hand an event to the decoder only
   when the head's confidence clears τ. **At τ=1 the output is the draft verbatim, so
   the method provably cannot do worse than leaving the draft alone.** +6.04% ± 0.014
   over copy-input on D1 (3/3 seeds, p=4.2e-52), replicated same-corpus at +5.21% ±
   0.020 on a 5× larger test set.
6. **On generator drafts, degradation is global, not local.** The draft is measurably
   less plausible as a whole, yet no individual position carries enough signal for a
   per-position head to find it. Localization precision sits at 1.3–1.7× chance and is
   **flat across degradation bands** — 1.56× on drafts with ΔNLL ≤ 0 and 1.66× on those
   the judge certifies as degraded, with no trend (`scripts/degradation_strata.py`).

   The tempting explanation — that most drafts simply are not degraded, so there is
   nothing to find — was tested and rejected: certified-degraded drafts are no more
   localizable than undegraded ones. Two measurements point instead at distribution.
   In a D2 draft the *unchanged* notes also become more expensive under the judge
   (+0.0677) alongside the infilled span (+0.2162), so the damage is spread through
   context rather than concentrated on the notes that changed. The generator writes a
   locally coherent span; what is wrong is how it fits the piece.

   **Consequence: per-position edit supervision is the wrong tool for this error class.**
   Span-level or global detection is the direction. State this as a contribution, not a
   limitation — it is what the next person needs.

   Supporting context, still true and worth reporting: 96.6% of generator drafts fall
   below τ, the 95th percentile of human-authored variants, and 17% are no less
   plausible than the reference. Using a generator does not solve the realism problem,
   it relocates it.

## 3. The four framing moves

**M1 — Lead with methodology, not the model.** The validator and the orthogonality
finding are the novel, reusable pieces. The +6% gated result is real but modest; it is
the *demonstration* that the diagnosis is actionable, not the headline.

**M2 — State the label-semantics point as a general warning.** Edit labels derived by
aligning a draft to a reference encode **difference, not error**. Whenever the draft is
a valid alternative, the supervision marks arbitrary divergence. This applies to any
reference-based edit supervision — code, text, or music — and is probably the single
most transferable sentence in the paper.

**M3 — Reframe abstention as selective prediction.** "The gate declines when no edit is
confidently warranted" is a *feature* for a creative tool, where a wrong edit is worse
than no edit. Pair it with the τ=1 guarantee: bounded downside plus abstention is a
deployable property, and more honest than claiming uniform improvement.

**M4 — Make the unsolved part an explicit contribution, not a limitation buried in
§7.** The next person needs two things: that generator drafts are mostly within human
variation, and that their degradation is global rather than per-position, so
per-position supervision cannot reach it.

## 4. Claim discipline

Three things the evidence does **not** support, which earlier drafts of this work
claimed and which must stay out:

- **Not "where and how".** The head-type ablation is decisive: binary matches 5-class
  (seed-matched difference +0.003 ± 0.011 on MTC, +0.012 ± 0.020 same-corpus, 2/3 seeds
  each way). Per-class F1: INSERT_BEFORE **0.000**, DELETE 0.107, SUB_DUR 0.194. Claim
  **"where"** — precision-gated *localization* — and report the typed head as an
  ablation that did not pay off. Recommend the simpler binary head.
- **Not "edit-awareness lowers Correction Cost".** It reliably improves *where the
  model looks* (macro-F1 0.092 → 0.356, 3/3 seeds, ±0.004) and does **not** reliably
  improve what it produces (CC vs B1 straddles zero, 2/3 seeds, ±0.404). Report the
  dissociation; it is a finding, not a weak result.
- **Not "edit-awareness hurts on random corruption".** That was an artifact of
  inverse-frequency class weighting, which paid the head ~105× more for flagging than
  for leaving a note alone. Fixed by `weighting=none` on D1.

## 5. Title and reframing (closes P2)

Name the question, not the method:

- *When Is a Draft Wrong? Calibrating Synthetic Corruption for Symbolic Music Refinement*
- *Difference Is Not Error: Rethinking Synthetic Corruption for Symbolic Draft Refinement*

Either disposes of the human-in-the-loop framing for free. Collaborative utility gets
one sentence as motivation, never as a demonstrated result.

## 6. Caveats to state prominently, not bury

- **ΔNLL is statistical implausibility under a small LM, not a musical judgement.** The
  human-variant calibration is the strongest available defence without a user study and
  should be presented as such — as a limitation, in the open.
- Two folk corpora only; the style claim is narrow and should say so.
- `J` is trained on Essen and scores MTC in the headline numbers (0.1365 nats/token
  domain gap). The same-corpus control rules out the confound — and the dissociation is
  *stronger* there, D2 precision falling 0.220 → 0.137, which is the theory's own
  prediction since `G` is an Essen model whose spans blend into Essen references.

## 7. Anticipated reviewer objections

| objection | answer |
|---|---|
| "Your method only works on the synthetic setting you criticise." | Correct, and the paper says why. On generator drafts the degradation is global rather than per-position: precision sits at 1.3–1.7× chance and does not improve even on judge-certified degraded drafts. A per-position head cannot address that error class, and the gate abstains rather than damaging the melody. The oracle shows 21.5% is available to a method that *can* localize, so the gap is quantified rather than a silent failure. |
| "+6% is small." | It is over *copy-input*. The published +6.3% was over an imitation baseline itself 21.4% worse than copying. |
| "ΔNLL isn't musical correctness." | Conceded explicitly; calibrated against human variants; see §6. |
| "One model size." | Size ablation over 2/4/8 layers × 128/256/512 (running). |
| "Thin evaluation." | 3 seeds with mean ± std, B1/B2/B3, fidelity, edit-F1, paired stats with bootstrap CI and Wilcoxon, same-corpus control. |

## 8. Rejection reasons → status

| | status |
|---|---|
| P1 random corruption ≠ revision | **answered** — validator, orthogonality, same-corpus control |
| P2 human-in-the-loop framing | **retitle** per §5; no experiments needed |
| P3 "where and how" | **answered by narrowing** — claim "where" (§4) |
| P4 representation contradictions | **fixed** — REST/BAR, measured; onset shift removed from the vocabulary |
| P5 thin evaluation | **answered** — §7 row 5, plus the size ablation |
| P6 no citations in introduction | **outstanding** — pure writing, and an explicit rejection reason |

## 9. Remaining work

1. ~~degradation stratification~~ — done; hypothesis rejected, §2.6 rewritten.
2. Model-size ablation (running).
3. **P6 related work** — McLeod et al. (MDT, ISMIR 2020) and Ramoneda et al.
   (RefinPaint, ISMIR 2024) need explicit positioning with a stated delta; plus Ji,
   Yang & Luo (2023), Lakh provenance, Vaswani et al. Zero citations in the
   introduction was a stated rejection reason.
4. D3 refinement sweep to complete the draft-source axis.
5. 5 seeds on the headline (seed 0 was an outlier once).
6. `scripts/` have no test coverage — that is how a broken-argparse bug reached a launch.
