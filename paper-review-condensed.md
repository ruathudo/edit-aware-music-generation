# Reviewer report: "The Better the Generator, the Less the Trace" (paper-condense.md)

*Written 2026-09-28 as an external reviewer would, for an 8–10 page IEEE conference.
Caveat on the reviewer: I helped build this paper, so I know the evidence better than
a real reviewer would and may be biased toward it. I have tried to review only what is
on the page.*

---

## Summary

The paper asks when *localize-then-fix* refinement of symbolic melodies works. Using
monophonic folk melodies (MTC test, Essen training), it (1) calibrates a small language
model as a plausibility judge on human variants of the same tune and argues that edit
distance to a reference cannot distinguish error from valid alternative; (2) compares
random corruption, rule-based degradation and generator-rewritten spans, finding that a
well-trained generator's changes are hard to localize even at matched degradation and
matched contiguity, and that localizability rises as the generator is weakened; (3)
proposes precision-gated editing, which copies each note unless an edit head is
confident, improving randomly corrupted drafts by ~6% over the unchanged draft and
abstaining on generated drafts; and (4) shows that even localizable generated errors are
rarely repaired exactly.

## Strengths

1. **Unusually careful experimental hygiene.** One setting is selected on validation
   data and applied unchanged everywhere. Every result is replicated same-corpus.
   Copy-input is the reference baseline throughout, rather than a weaker learned
   baseline. The paper reports per-seed consistency rather than pooled p-values. Few
   papers in this area do all of this.
2. **Negative results are taken seriously and decomposed.** The §7 funnel (not the
   detector, not the amount of damage, not contiguity, then generator quality, then
   repair) is the paper's best part. Each alternative explanation gets its own control.
3. **The copy-input finding is useful to the field.** A seq2seq refiner that is 25%
   worse than doing nothing, and an auxiliary loss that "wins" against B1 while copying,
   are cautionary results that other refinement papers should hear.
4. **The generator-quality gradient is a genuinely interesting observation.** It is
   also stated with an appropriate limitation (one generator, degraded).
5. **Honest scoping.** The limitations paragraph pre-empts most obvious objections, and
   the paper retracts its own stronger claims (e.g. transfer affects every benchmark,
   not only the rule-based one).

## Weaknesses

### Major

**M1. Both axes of the central figure come from the same family of models.** The x-axis
of Figure 1a (implausibility per unit of change) is computed by `J`. The y-axis
(localization AUROC) is computed by a masked language model trained on the same data.
A change that `J` finds implausible per note is, almost by construction, one the masked
model finds surprising at that note. So the monotone rise in Figure 1a is partly
mechanical, and it cannot on its own support "the better the generator, the less
trace". The matched-degradation bands (Table 5) are also defined by `J` and scored by
its sibling. The paper acknowledges a milder version of this in the full draft, but the
condensed version drops the caveat. The claim would be much stronger with a localizer
independent of `J`'s training data along the curve. The cheapest option: the D1-trained
head applied zero-shot to all six generator conditions, since it learned from edit
labels, not from `J`.

**M2. "Generator quality" is never measured independently.** The paper orders
generators by training epoch and temperature, then by `J`'s per-unit implausibility,
and calls this quality. Neither the generators' own held-out likelihood nor any human
judgement is reported. The title's causal phrasing ("the better the generator") needs
at least the checkpoints' validation NLL. A small listening test would be better.

**M3. The repair metric contradicts the paper's own first contribution.** Table 6
scores a repair as correct only if it *exactly* matches the reference note. But §4
argues at length that distance to a reference cannot distinguish an error from a valid
alternative. A decoder that replaces a generator-written note with a *different
plausible* note is counted as a failure. The "11% vs 59%" contrast may therefore
partly measure underdetermination of the *target*, not failure to repair. That is in
fact the paper's interpretation, but the paper should show it. Report the plausibility
(ΔNLL) of the repaired notes, or of the gated outputs on the 2-epoch drafts. If those
repairs make the music more plausible, the story changes from "rarely repaired" to
"repaired differently".

**M4. No human evaluation anywhere.** Every musical claim — plausibility, "error",
"better music", generator quality — rests on one 4-layer LM and its sibling. The paper
says so, but for a paper whose thesis is about what counts as an error, this is the
central gap. Even a small expert annotation would help: have a musician mark which
notes sound wrong in about 50 D1 and D2 drafts, then check the judge's localization
against those marks. That would also address M1.

**M5. External validity of the "generated drafts".** D2 splices a 2–6 note generated
span into a human melody. That is still a synthetic construction, and the paper's own
argument is that synthetic constructions mislead. No draft comes from a real generation
workflow, such as a full generated melody later revised by a person. The conclusions
about "refining generated music" should be phrased as conclusions about *span-infilled*
drafts. Alternatively, the paper could argue why span infilling is representative.

### Moderate

**M6. The method's positive result lives where the paper says it matters least.** The
~6% gain is on random corruption, which §4 calls unrealistic. On the realistic source the
method does nothing, and on the intermediate one (2-epoch generator) it gains
essentially nothing. The bounded-downside argument is sound, but a reader may conclude
that the method is a safe baseline rather than a contribution. Consider presenting it
explicitly as the *instrument* that makes the analysis possible, with contribution 3
ranked last.

**M7. Small evidence bases behind general statements.**
- "*Every* synthetic benchmark we built" rewards learning its process by about a
  quarter. That rests on two benchmarks and two transfer numbers from three seeds.
- The oracle headroom and the "49% vs 21.5%" repairability comparison come from a
  single run under an earlier configuration, yet the repair argument leans on them.
- The test set is 360 melodies in 26 families, and uncertainty is reported only across
  seeds. Family-level bootstrap intervals on the headline gains would be more
  informative.

**M8. Magnitudes.** A 6% relative reduction in CC on D1 is roughly 0.18 fewer edits per
melody (B3 CC ≈ 2.98). Stating the absolute effect, and the win and loss rates per
melody, would help readers judge practical significance.

### Minor

- **Inconsistent "genuinely wrong" figure.** §6 says 82% of gate-opened events were
  genuinely wrong (flag-level precision). Table 6 implies 88% (76% true substitutions
  plus 12% unmatched events, whose fix is deletion). Define one quantity and use it
  in both places.
- **Hidden normalization difference.** Judge scores are standardized within each
  melody; head probabilities are not. The full draft notes that this favours the head;
  the condensed version dropped that caveat.
- **Mixed seed counts.** Five seeds (Table 2) and three seeds (Tables 3, 6, transfer,
  judge gates) are mixed. The caption convention is correct, but the Intro quotes both
  without saying which is which.
- **Figure 1b** mixes five-seed and three-seed points. Say so in the legend, not only
  in the caption.
- **"Two orders of magnitude"** comes from dividing by a median near zero (0.04). Say
  "over 25× that of generated drafts and far above human variants", or report
  differences instead.
- **Abstract density.** Seven numeric claims in one paragraph; cut to three.
- **Related work** is thin for a music venue. Consider music inpainting and interactive
  composition systems. Also cite GEC's precision-weighted evaluation (F0.5), which is
  the direct precedent for "precision is the binding constraint". Verify each citation
  before adding it.
- **Header note** says the main text is ≈ 5,600 words; it is ≈ 4,700.
- **B2** is referenced in §6 but its row now lives only in the supplement.

## Questions for the authors

1. Does the D1-trained head, applied zero-shot, show the same rise across the six
   generator conditions as the judge does? (M1)
2. What are the held-out NLLs of the 2/5/10-epoch and final generators? (M2)
3. Are the 2-epoch gated outputs more plausible than the drafts, even where they do not
   match the reference? (M3)
4. Why is span infilling a representative model of real generated drafts? (M5)
5. What are the family-level bootstrap intervals for the D1 gains? (M7)

## Scores and recommendation

| criterion | score | comment |
|---|---|---|
| Soundness | 3 / 4 | Careful protocol; the central gradient is partly circular (M1), and the repair metric sits in tension with the paper's own argument (M3) |
| Novelty | 3 / 4 | Generator-quality gradient and repair decomposition are new for music refinement; the edit-distance critique is known in NLP |
| Significance | 2.5 / 4 | Useful cautionary findings; small scale, folk only, no humans |
| Clarity | 3 / 4 | Well organized; numerically dense |
| Reproducibility | 4 / 4 | Scripts, committed results, seeds, re-run variance |

**Recommendation: weak accept (borderline).** Confidence: moderate.

As an *analysis* paper, it is scientifically sound in its protocol and more honest than
most. Its main claims are stated with appropriate hedges, and the controls are real. It
is not yet convincing on the one claim that now carries the title. The generator-quality
gradient is measured with instruments that share training data with the quantity on
the other axis, it has no independent measure of generator quality, and it has no human
check. At a mid-tier IEEE venue I would lean accept if M1–M3 were addressed, all of
which the authors could do in a few days. At a top venue, M4 (no human evaluation) and
M5 (synthetic drafts only) would likely lead to rejection. The method contribution alone
(M6) would not carry the paper.

---

## Author-side action list (not part of the review)

Ordered by value per effort. Items 1–3 need no training and use saved artifacts.

1. **M1:** score the six generator conditions with the D1-trained head, zero-shot.
   Add the result to Figure 1a as a second series.
2. **M3:** compute ΔNLL of the ep2 gated outputs against the drafts, and against the
   references on the changed notes only. Report plausibility of the repairs beside
   exact repair in Table 6.
3. **M2:** evaluate held-out NLL for each generator checkpoint and give it in S5 and in
   the Figure 1 caption.
4. **M7:** add family-level bootstrap CIs for the headline gains.
5. **Minor fixes:** the 82%/88% definition, the header word count, the standardization
   caveat, the abstract density, the "two orders of magnitude" wording.
6. **M4 (costly, highest value for acceptance):** a small expert annotation of wrong
   notes in about 50 D1 and 50 D2 drafts.
7. **M5:** add a paragraph arguing representativeness, or rephrase claims to
   "span-infilled drafts".
