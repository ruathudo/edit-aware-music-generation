# The Better the Generator, the Less the Trace: When Does Localize-then-Fix Work for Symbolic Music Refinement?

*Condensed version of `paper-draft.md` for an 8–10 page IEEE conference submission.
Main text ≈ 4,900 words, six tables, one figure. Everything trimmed is kept in the
**Supplementary material** at the end of this file, as blocks that can go into a
supplement if the venue allows one or be dropped if it does not. `paper-draft.md`
remains the full version; `paper-core.md` remains the claims document. Numbers are
identical to the full draft.*

---

## Abstract

Refining a draft melody — changing the few notes that are wrong and keeping the rest —
is naturally framed as *localize-then-fix*, typically trained on synthetically
corrupted melodies and evaluated by edit distance to the original. We test both halves
of that recipe on monophonic folk melodies. A melody language model, calibrated against
human variants of the same tune, shows that edit distance cannot tell an error from a
valid alternative: human variants differ in 92% of their notes yet are judged equally
plausible, while synthetic corruption is two orders of magnitude more implausible per
unit of change. Comparing random, rule-based and generator-written drafts, we find that
localize-then-fix works where changes leave a local trace, and that how much trace a
generator leaves depends on how good it is. An untrained judge that scores each note in
context finds corrupted notes at 6–7 times chance but a trained generator's at 2.7
times, even at matched degradation, and across six generator settings localizability
falls steadily as the generator's own held-out likelihood improves. Where a weak
generator's changes are found and edited, the edits make the music more plausible but
rarely restore the reference, so edit distance scores them as no gain. A precision-gated
refiner, which
copies each note unless confident it is wrong, improves randomly corrupted drafts by
6.1% on every seed, is the only system tested that improves them in both edit distance
and plausibility, and abstains on generated drafts, where plain sequence-to-sequence
refinement does harm.

## 1. Introduction

Generative models of symbolic music now produce plausible melodies on demand, but a
draft is rarely final. Someone — a composer, or a second model — must decide which notes
to change and which to keep. This *refinement* problem differs from generation in one
respect: most of the draft is already right, and a system that rewrites it wholesale has
not refined anything. The natural formulation is *localize-then-fix*: find the wrong
notes and change only those, as in models that locate and correct degradations in
symbolic excerpts [McLeod2020] or pick out the weak notes of a composition for
resampling [Ramoneda2024].

The standard recipe rests on two assumptions. Training data is made by *corrupting*
clean melodies, so synthetic corruption is assumed to resemble the errors real drafts
contain; and evaluation measures edit distance to the original, so distance to a
reference is assumed to measure improvement. This paper tests both.

**Edit distance cannot tell an error from an alternative.** We calibrate a melody
language model as a plausibility judge against human-authored variants of the same folk
tune — valid alternatives by construction. The judge rates them as neither better nor
worse than one another, though they sit 92% apart in normalized edit distance; synthetic
corruption sits 7% away and is two orders of magnitude more implausible per unit of
change. Edit distance can rank methods on one draft source, but cannot compare sources
or say whether a change was a correction (§4).

**The better the generator, the less trace it leaves.** We compare random corruption,
rule-based musical degradation, and spans rewritten by a trained generator under one
model, one recipe and one setting chosen on validation data. An untrained two-sided
judge ranks corrupted notes at 6–7× chance but the generator's at 2.7×, and the gap
holds at matched degradation and matched contiguity: the damage is real but diffuse,
lying in how a rewritten passage meets its context. This is a property of the
generator, not of generation: across six generator settings, ordered by the generator's
own held-out likelihood, localizability falls steadily as the generator improves, for
the judge and for a detector trained only on corruption (§7; Figure 1).
Localize-then-fix gets harder exactly as generators get better.

**A method that respects the boundary.** A plain sequence-to-sequence refiner is 25%
*worse* than returning the draft: it rewrites notes that were right. We instead gate
each note — copied unless the model is confident it needs changing, the reject option of
selective classification [Chow1970, Geifman2017] applied per note — so the method's
floor is the draft by construction. On random corruption it improves on the draft by
6.1% on every seed, and is the only system we test that improves both edit distance and
judged plausibility (§6). On generated drafts it declines to act, and no other threshold
or gate we tried does better. On a weak generator's drafts the gate opens and makes the
music more plausible, yet gains nothing in edit distance: its repairs match the
reference 11–13% of the time (58–59% for corrupted notes) but are, by the judge, mostly
as plausible as it. Edit distance cannot see such repairs — our first finding, turned on
our own evaluation.

We make three contributions:

1. **A measurement.** A plausibility judge calibrated on human variants, and evidence
   that edit distance to a reference cannot distinguish an error from a valid
   alternative and should be reported alongside plausibility.
2. **A boundary.** A controlled comparison showing that localize-then-fix applies where
   changes leave a local trace, and that a generator's drafts leave less of it the
   better the generator, with matched-degradation, matched-contiguity, transfer and
   same-corpus controls. Transfer also shows that about a quarter of a synthetic
   benchmark's in-domain score is specific to the process that made its errors.
3. **A method, and its limit.** Precision-gated editing, whose downside is bounded by
   construction, which improves drafts where errors are local and abstains where they
   are not — and a decomposition showing that found errors in generated drafts are
   repaired toward the rewritten context rather than back to the reference, which edit
   distance scores as no gain.

We use monophonic folk melodies from the Meertens Tune Collections [MTC] and the Essen
Folksong Collection [Essen]. Plausibility is judged by a language model, not by
listeners; we return to this limitation in §8.

**Figure 1** (`figures/figure1.pdf`). Draft conditions placed by the judge's median
implausibility per unit of change (Table 1); the dashed line marks human variants.
(a) AUROC of an untrained two-sided judge for the changed notes (0.5 is chance), for
random (D1) and rule-based (D3) corruption and six generator settings — temperature
0.8/1.0/1.2, and checkpoints after 2, 5 and 10 epochs — on MTC and Essen. D2 is the
trained generator at T = 1.0. (b) Gated editing's improvement over the unchanged draft
for the headline sources (five seeds) and, hollow, for the 2-epoch generator's drafts
(three seeds); MTC. Localizability rises steeply with implausibility; by this judge an
undertrained generator's drafts are as localizable as corruption, and gated editing makes
them more plausible without moving them closer to the reference.

## 2. Related Work

**Symbolic music correction.** The MIDI Degradation Toolkit [McLeod2020] applies
rule-defined degradations and poses detection, classification, localization and
correction tasks; our random (D1) and rule-based (D3) drafts belong to this family, and
we measure how much of such a benchmark's gain is learned from its corruption process
(§7). RefinPaint [Ramoneda2024] uses a feedback model to find a composition's weak
elements and an inpainting model to resample them; we ask the prior question of *when*
such localization is possible. Our generated drafts come from span infilling, in the
line of models that condition on arbitrary parts of a score [Huang2017, Thickstun2023].

**Edit-based transduction.** When input and output overlap heavily, predicting edits
beats regenerating: tagging models label tokens keep, delete or insert [Malmi2019,
Omelianchuk2020], others iterate insertion and deletion [Gu2019] or copy from the input
[Gu2016]. Our gated decoder combines the two: a head decides *whether* to edit and a
decoder *what* to write, so copying is the default rather than a learned behaviour.

**Synthetic errors and reference-based evaluation.** Grammatical error detection is
framed as sequence labeling [Rei2016] and leans on synthetic errors [Grundkiewicz2019,
Kiyono2019], whose realism matters: errors generated by a model trained on real learner
errors improve detectors, and readers struggle to tell them from real ones [Kasewa2018]. Our drafts span the same range, and show a sharp
consequence of realism in music. Single-reference scoring undercounts valid
alternatives in error correction [Bryant2015]; we quantify the musical analogue.

**Plausibility and selective prediction.** Our judge is a neural instance of modelling
melodic expectation as information content [Pearce2005], with masked per-note scoring as
in pseudo-log-likelihood [Salazar2020]; likelihood-maximizing decoding favours typical
but degenerate output [Holtzman2020], which we observe in free decoding. Declining to
act when uncertain is the reject option [Chow1970], developed for deep networks as
selective classification [Geifman2017]; our gate applies it per note.

## 3. Task, drafts and representation

**Task.** A *draft* $M_0$ is a monophonic melody that differs from a *reference* $M^*$
in some notes. A refiner receives $M_0$ alone and outputs $\hat M$; it succeeds to the
extent that $\hat M$ is closer to $M^*$ than $M_0$ was.

**Corpora and splits.** The Essen collection (8,469 melodies) supplies training data. It
is split in half: one half trains the plausibility judge (§4), the other the generator
that writes D2 drafts, so the judge never scores material it could have memorized from
the generator's training set. Refiners train on drafts of the judge's half (3,812
melodies; 423 for validation). The test set is MTC-ANN [MTC]: 360 melodies grouped by
specialists into 26 *tune families*, melodies believed to descend from a common
ancestor. No model sees MTC during training, so the main results are cross-corpus; a
same-corpus replication holds out 20% of Essen and retrains judge and generator on the
rest.

**Draft sources** (three drafts per test reference):

- **D1, random corruption:** one to five operations — pitch replacement 40%, duration
  change 30%, deletion 15%, insertion 15% — with replacement values drawn from the
  corpus's own marginals.
- **D2, generated:** a span of 2–6 consecutive notes rewritten by a three-layer
  transformer language model, sampling at temperature 1.0 from the preceding notes, after
  which the reference resumes; the splice is not smoothed. §7 also draws D2 drafts at
  temperatures 0.8 and 1.2, and from the generator's checkpoints after 2, 5 and 10
  epochs. Unqualified, "D2" is the trained generator at 1.0.
- **D3, rule-based degradation:** one to five applications of six music-theoretic
  violations (out-of-scale pitch, leap over 14 semitones, strong note moved off the
  beat, disrupted final cadence, rhythm flattened, figure collapsed to one pitch).
- **D4, human variants:** 1,012 pairs from the same tune family, valid alternatives by
  construction. They calibrate the judge only; family members differ in metre and
  length, so they are variants of a tune, not drafts, and we do not refine them.

**Representation and cost.** An event is a pitch (or rest) and a duration in ticks at
24 per quarter note, exact for both corpora; it becomes two tokens, with a bar token
where the annotated metre begins a bar (121 tokens in all). Two melodies are aligned by
Wagner–Fischer distance over events with unit costs for insertion, deletion, pitch and
duration substitution; *Correction Cost* (CC) is the minimal cost. The alignment also
labels each draft event *keep* or *edit*, with an end slot for notes missing at the end.

**Evaluation.** The primary measure is the relative reduction in mean CC over returning
the draft unchanged (*copy-input*, B3), so positive values mean the refiner improved on
doing nothing. We also report *fidelity* (CC between output and draft) and plausibility
(§4). Headline results are means over five seeds, secondary comparisons over three, all
under one configuration selected on validation data (§5).

## 4. Measuring plausibility

**The judge.** $J$ is a decoder-only transformer language model over the tokens of §3
(four layers, width 256), trained on the judge's half of Essen. It scores a draft by its
degradation relative to the reference in mean per-token negative log-likelihood,
$\Delta\mathrm{NLL}(M_0) = \mathrm{NLL}_J(M_0) - \mathrm{NLL}_J(M^*)$; positive means less
plausible. Because the measure is paired, the offset between Essen and MTC cancels;
its effect on the judge's sensitivity does not, which the same-corpus replication
controls.

**Calibration on human variants.** A judge that penalized any departure from the
reference would call every alternative an error. On the D4 pairs, ΔNLL has mean +0.054,
and 45.0% of pairs score at or below zero: the judge is close to indifferent between
variants that differ in 92% of their notes. It measures degradation, not difference. We
set a degradation threshold $\tau = 0.585$, the variants' 95th percentile, used only to
band drafts for analysis.

**Table 1.** Draft sources under the judge (MTC). *Per unit* is the median of ΔNLL over
the fraction of notes changed; ρ is Spearman's correlation of fraction changed with
ΔNLL, 95% interval from resampling tune families.

| source | mean CC | fraction changed | ΔNLL mean | per unit | ΔNLL ≤ 0 | ρ (95% CI) |
|---|---|---|---|---|---|---|
| D4 human variants | 45.7 | 92.3% | +0.054 | +0.04 | 45.0% | +0.14 [+0.04, +0.24] |
| D1 random | 2.98 | 6.6% | +0.298 | +3.94 | 5.2% | +0.52 [+0.46, +0.59] |
| D3 rule-based | 4.11 | 9.1% | +0.271 | +3.14 | 4.2% | +0.47 [+0.41, +0.52] |
| D2 generated | 4.71 | 10.6% | +0.139 | +1.04 | 20.5% | +0.20 [+0.11, +0.30] |
| all pooled | | | | | | −0.06 [−0.11, −0.01] |

**Corruption is damage, but of an unrealistic kind.** Random corruption does degrade:
95% of D1 drafts are less plausible than their references. But it changes 7–9% of a
melody at 3–4 nats per unit of change, two orders of magnitude more than a human variant.
Corruption moves a melody a short distance *off* the distribution of plausible
melodies; human variation moves it a long way *along* it. Generated drafts sit between,
at about 1 nat per unit: most (79.5%) are less plausible than their reference, a fifth
are not.

**Edit distance cannot tell an error from an alternative.** Within one corruption
process, more change means more damage (ρ ≈ 0.5, partly by construction). Across sources
the relation vanishes and slightly reverses (ρ = −0.06): the melodies furthest from their
references — the human variants — are the most plausible. Edit distance can therefore
rank methods on one draft source, but cannot compare sources or say whether a change was
a correction. We report both measures throughout.

## 5. Method

**Refiner.** An encoder–decoder transformer (three plus three layers, width 256; 5.86M
parameters) reads the draft and writes the refined melody, trained with teacher forcing
on the reference. An encoder is needed because deciding whether a note is wrong requires
the notes after it, and because insertions and deletions make draft and reference differ
in length.

**Edit head.** A linear layer on the encoder output predicts, per draft event, whether
it needs editing, trained jointly: $\mathcal{L} = \mathcal{L}_{\text{gen}} + \alpha\,
\mathcal{L}_{\text{edit}}$. A five-way head that also predicts the edit type performs no
better (+0.10 ± 0.36 points, seed-matched) and never predicts an insertion correctly, so
we use the binary head.

**Gated decoding.** Each draft event is copied verbatim unless the head's probability
that it needs editing reaches a threshold $\tau$; then the decoder writes a replacement,
conditioned on everything decoded so far. With an unreachable threshold the output *is*
the draft, so the method's worst setting is doing nothing; lowering $\tau$ trades
coverage for risk, as in selective classification [Geifman2017]. The gate substitutes
events; it does not insert or delete.

**Baselines.** B1 is the same model without a head, decoded freely: plain
sequence-to-sequence imitation. B2 keeps the head but sets $\alpha = 0$, isolating the
loss from the parameters. B3 returns the draft. B3 and fidelity are essential: in a
preliminary sweep, an auxiliary edit loss looked like a clear gain over B1 (+6.3%, three
of three seeds) while its output was nearly identical to the draft and no better than
copying it.

**One setting, chosen on validation.** Class weighting, $\alpha$ and $\tau$ are selected
once, on D1 validation drafts only (96 settings, three seeds each), and applied
unchanged to every source and corpus: no class weighting, $\alpha = 0.5$, $\tau = 0.8$.
The landscape is flat — 84 of 96 settings beat copy-input on every seed. Models use
AdamW (learning rate $3 \times 10^{-4}$) with early stopping.

## 6. Results I: where localize-then-fix works

**Table 2.** Improvement in mean CC over copy-input (%), mean ± s.d. over five seeds.
Essen columns are the same-corpus replication.

| | D1 MTC | D1 Essen | D3 MTC | D3 Essen | D2 MTC | D2 Essen |
|---|---|---|---|---|---|---|
| B1 imitation, free | −24.9 ± 11.6 | −92.2 ± 7.1 | +3.9 ± 1.9 | −22.3 ± 5.3 | −3.6 ± 1.3 | −27.1 ± 4.2 |
| edit-aware, free | −17.8 ± 4.6 | −92.4 ± 10.8 | +4.0 ± 1.2 | −22.7 ± 4.2 | −4.2 ± 2.1 | −25.6 ± 5.4 |
| **edit-aware, gated** | **+6.06 ± 0.58** | **+4.25 ± 0.46** | **+9.81 ± 0.46** | **+7.38 ± 0.27** | 0.00 | 0.00 |
| gated: seeds > copy | 5/5 | 5/5 | 5/5 | 5/5 | abstains | abstains |

**Gated editing improves drafts where errors are local.** On random corruption gated
decoding reduces CC by 6.06% on MTC and 4.25% same-corpus, on every seed (Table 2). It changes
about 1.5% of notes, and 82% of the events it hands to the decoder were genuinely wrong.
(Here "wrong" is flag-level precision; counting only events the decoder actually
changed, 88% were wrong.) The result is not delicate: every threshold from 0.2 to 0.95 beats copy-input on every
seed, on both corruption sources and corpora, and so do all nine model sizes we tried
(1.6M–42.7M parameters, at τ = 0.9).

**Free decoding is worse than doing nothing.** Every freely decoded model is worse than
the draft on random corruption — B1 by 25% on MTC and over 90% same-corpus — because it
rewrites notes that were right. The edit loss does not rescue free decoding: with the
head trained or present but untrained (B2), free decoding stays within the seed noise of
B1. The loss's contribution is a head confident enough to gate with. An oracle given the true edit positions removes about half of D1's
CC (49.3%, single run), so gating recovers about an eighth of the headroom.

**Table 3.** Plausibility of outputs, MTC, seeds 0–2. *Recovered* is the share of the
draft's ΔNLL removed; for gating, the share over melodies it changed, and how many of
those it made less plausible, is in brackets.

| | D1: CC vs copy | D1: ΔNLL recovered | D3: CC vs copy | D3: ΔNLL recovered |
|---|---|---|---|---|
| B1, free | −24.7% | +40.9% | +4.3% | +48.8% |
| edit-aware, free | −17.7% | +42.5% | +3.7% | +51.0% |
| **edit-aware, gated** | **+6.35%** | +16.3% [+30.1%; 2.8% worse] | **+9.98%** | +38.8% [+47.5%; 1.4% worse] |

**Better music, not only closer to the reference.** On the melodies it changes (Table
3), gated editing removes 30% of the draft's implausibility on D1 and 48% on D3, making fewer than
3% less plausible. Free decoding recovers *more* plausibility while being 25% worse on
CC: it drifts toward the model's most typical continuation, and 14.5% of its D1 outputs
outscore the reference itself. An evaluation on plausibility alone would prefer free
decoding; one on CC alone would hide that gating improves the music. On random
corruption, gated editing is the only system better than the draft on both.

**D3 and D2.** Gated editing gains most on rule-based degradation (9.81%), and a refiner
trained only on random corruption improves D3 by more than it improves D1, so part of
this is intrinsic; part is learned from D3's rules (§7). On D2 the gate never opens.
Free decoding loses 3.6% and 27.1%, and no threshold helps: at 0.2 the gate opens at
precision 0.17 and loses about 1% on every seed.

## 7. Results II: where it does not

On generated drafts the gate never opens. We rule out three explanations, show what
does govern localizability, and show that localizing is not enough.

**Table 4.** Localization of the notes that differ from the reference: AP lift (average
precision over base rate; 1.0× is chance) / AUROC. MTC (Essen). The judges are never
trained on edit labels.

| detector | D1 random | D3 rule-based | D2 generated |
|---|---|---|---|
| trained head | 5.8× / .724 (5.6× / .727) | 4.6× / .726 (4.3× / .717) | 1.6× / .636 (1.8× / .658) |
| causal judge | 4.1× / .775 (4.9× / .774) | 4.7× / .768 (5.1× / .756) | 1.4× / .561 (1.4× / .564) |
| two-sided judge | **6.2× / .838** (7.0× / .831) | **5.9× / .812** (6.1× / .790) | **2.7× / .688** (2.7× / .683) |

**Not for want of a detector.** The head trained on D2 localizes at 1.6× chance (Table 4) and
never becomes confident enough to open the gate, although D2 drafts contain *more*
changed positions than D1 (7.3% against 6.0%). A span-level detector does no better
(1.6× at top-1), nor do heads transferred from D1 or D3 (1.3–1.4×). The strongest
localizer was never trained on edits: a two-sided judge — a masked language model of
the same size and data as $J$, scoring each note given all the others — beats the
trained head on every source and corpus, yet reaches only 2.7× on D2.

**Not less damage.** Generated drafts carry less damage (mean ΔNLL 0.14 against 0.30),
and any surprisal-based localizer finds high-ΔNLL drafts easier. Table 5 therefore bands
*every* source at the same absolute ΔNLL cut-points (about 5,300 fresh MTC drafts per
source, and the Essen test drafts).

**Table 5.** Localization at matched degradation: two-sided judge AUROC, drafts banded by
$J$'s ΔNLL at shared cut-points (four of six bands shown; all in Supplement S1). MTC
(Essen).

| ΔNLL band | D1 random | D3 rule-based | D2 generated |
|---|---|---|---|
| ≤ 0 | .753 (.722) | .638 (.618) | .606 (.579) |
| 0 – 0.1 | .830 (.826) | .774 (.740) | .684 (.670) |
| 0.2 – 0.4 | .855 (.837) | .825 (.816) | .730 (.724) |
| > τ | .861 (.855) | .850 (.838) | .775 (.779) |

In every band and on both corpora, generated changes are localized worse than random
corruption, by 0.08–0.16 AUROC, although D2 changes *more* notes per band. Even D2's
most-degraded band falls below random corruption's least-degraded positive band. The
trace goes to the seam: after a rewritten span, the *unchanged* notes that follow also
lose plausibility (+0.07 per token, against +0.22 within the span). A causal model
charges that cost to the wrong notes; a two-sided judge, seeing both sides, doubles D2
localization (1.4× to 2.7×) but cannot recover what spreads beyond the changed notes.

**Not contiguity.** Generated changes come in runs of 2–6 notes, and contiguity does cost
localization — within D3, the judge's AUROC falls from 0.94 for isolated changes to 0.75
for runs of three or more. But at matched run length D2 is still worse: 0.68 against
0.75 for runs of three or more, and, where no two changed notes are adjacent, 0.75
against 0.84 (D1) and 0.94 (D3). Essen agrees throughout (Supplement S2).

**The better the generator, the less it leaves.** Figure 1a varies the generator itself.
Localizability rises steadily as its changes become less plausible — AUROC 0.65 at
temperature 0.8, 0.69 for D2, 0.72 at 1.2, then 0.77, 0.83 and 0.87 for the checkpoints
after 10, 5 and 2 epochs (Essen: 0.65 to 0.86, same order). The ordering is the
generator's quality by its own measure: held-out NLL falls from 1.84 (2 epochs) through
1.70 and 1.50 to 1.23 (trained). Because the judge shares training data with $J$, which
supplies the x-axis, part of its rise is built in; an independent detector — the head
trained on random corruption's labels, applied zero-shot — rises in the same direction
(0.57 to 0.64; Essen 0.61 to 0.68, monotone) but more shallowly, staying below random
corruption (0.73) where the judge puts the 2-epoch drafts above it (0.87 against 0.84).
What makes D2 hard to localize is not that a generator wrote it, but that a good one did.
(Judge scores are standardized within each melody and head probabilities are not;
standardizing the head lowers its AUROC, so this favours the head.)

**Repaired, but not to the reference.** Refiners trained on the 2-epoch generator's
drafts under the same setting gain only −0.13% ± 0.24 over copy-input on MTC and +0.38% ± 0.30
same-corpus — a tenth of their gain on random corruption. Two things fall short. The
trained head localizes these drafts less well than the judge (AUROC 0.67–0.81 against
0.86–0.87), so the gate opens at precision 0.59–0.72 rather than 0.82. And where the gate
finds a changed note, the decoder rarely writes the reference's note (Table 6) — but it
is not failing. Gating removes 28–36% of these drafts' implausibility, and at repairs
that miss the reference the two-sided judge rates the decoder's note far above the
draft's (median surprisal 2.3–2.4 against 10.7–11.1) and at least as plausible as the
reference's own note in 64–68% of cases.

**Table 6.** What the decoder writes where the gate opens (seeds 0–2). *True
substitution*: the draft event differs from its aligned reference event. *Exact repair*:
share of true substitutions where the output equals the reference event. *≥ reference*:
share of the other (mismatched) repairs the two-sided judge rates at least as plausible
as the reference note. *Recovered*: share of changed melodies' ΔNLL removed. MTC (Essen).

| drafts | true substitution | exact repair | mismatched ≥ reference | recovered |
|---|---|---|---|---|
| D1 random | 76% (69%) | **59%** (58%) | 49% (49%) | 30% (31%) |
| D3 rule-based | 94% (89%) | **44%** (41%) | 39% (40%) | 48% (53%) |
| D2, 2-epoch generator | 67% (65%) | **11%** (13%) | **64%** (68%) | 28% (36%) |

A corrupted note is pinned down by the intact notes around it. Inside a rewritten span
the neighbours were rewritten too, so the reference's note is no longer the best fit and
the decoder writes one that fits them; the span's original is one plausible continuation
among many. The oracle agrees: given the true positions it removes 49% of D1's CC but
21.5% of D2's. Edit distance — which §4 showed cannot tell an error from an alternative —
scores these repairs as no improvement. The judge rewards typical notes and may favour
the repairs for that reason alone; whether listeners do is the question for our
listening study.

**What training adds.** Transfer between sources isolates what a trained head learns
about its own corruption process (seeds 0–2). A D1 head reaches +7.25% on D3 against
+9.98% for a D3 head; a D3 head reaches +4.80% on D1 against +6.35%. The loss is about a
quarter in both directions, so *every* synthetic benchmark we built rewards learning its
own corruption process by about a quarter of its in-domain score. Transfer also bounds
the gate's safety: heads trained on D1 or D3 and applied to D2 open the gate and make
drafts 1.7–2.0% worse on every seed. Abstention is the behaviour of a head trained on
generated drafts, not of any head.

**Where the judge belongs.** The judge localizes better than any head, but helps only at
the gate (Supplement S3). Fed to the encoder it improves localization yet worsens gated
editing (+6.35% to +5.14%); alone at the gate it cannot abstain, because a within-melody
score always nominates some note (+2.8% on D1). Combined with the head's confidence it
adds 1.4–2.1 points on every corrupted cell and seed, at a cost of 0.4–0.5% on generated
drafts.

## 8. Discussion and limitations

**Recommendations.** *Report copy-input and fidelity:* a refiner measured only against
another learned baseline can look successful while copying its input. *Report
plausibility beside edit distance:* the two disagree across sources and systems, and
either alone would have misled us. *Test synthetic benchmarks by transfer:* about a
quarter of in-domain gain is specific to the corruption process. *Report the generator's
quality with any result on generated drafts:* a weak generator's drafts are far easier to
localize, so results on them will not transfer to a stronger one. *Do not score repairs
by exact match:* on generated drafts, repairs that miss the reference are mostly as
plausible as it.
Refining the output of good generators likely needs a formulation that does not start
from per-note error labels — revision scored by plausibility over whole passages, or
error locations marked by people. The same hazard applies wherever edit supervision is
derived from a reference, in text and code as well as music.

**Limitations.** *Plausibility is a model's judgement:* ΔNLL is implausibility under a
small language model, calibrated on human variants but not validated by listeners, and
it rewards typical continuations — which bears directly on the repair finding. *[TODO:
listening study with three musically trained raters; replace this sentence with its
results.]* *Shared training data:* the refiners train on the
judge's half of Essen, which flatters free decoding most and gated editing less.
*One generator, degraded:* the quality gradient weakens one generator rather than
comparing independently built ones; that stronger generators leave still less trace is
an extrapolation. *Two corpora, one texture:* both are monophonic European folk song.
*Detectors tried:* a stronger detector may exist, but at matched degradation the same
judge finds corruption far better, so the shortfall lies in where the damage is, not
how much. *The gate only substitutes*, although 30% of random-corruption operations are
insertions or deletions. *Runs are not bit-reproducible* on GPU; a full second run of
the main experiment differs by at most 0.36 points per cell.

## 9. Conclusion

Localize-then-fix works on the drafts it is usually tested on, and we have argued that
this is largely a property of those drafts. Synthetic corruption damages a melody in a
way unusual enough to be found note by note, and a refiner that copies by default and
edits only where confident improves such drafts on every seed without making the music
less plausible. A good generator's drafts are different in kind: worse than their
references, but diffusely, in how a rewritten passage meets its context. The same
generator early in training leaves far clearer traces, so how much trace remains
measures how good the generator is. And where such changes are found and edited, the
refiner repairs toward the rewritten context rather than back to the reference — more
plausible music that edit distance scores as no gain. Two
measurement lessons carry beyond music: edit distance to a reference should be reported
beside plausibility and beside the score of returning the draft, and a synthetic
benchmark should be tested by transfer.

---

## References

Entries as verified in `paper-draft.md` (2026-09-25), limited to those cited here.

- **[Bryant2015]** Christopher Bryant and Hwee Tou Ng. 2015. How Far are We from Fully
  Automatic High Quality Grammatical Error Correction? In *Proceedings of ACL-IJCNLP
  2015*, pages 697–707. https://aclanthology.org/P15-1068/
- **[Chow1970]** C. K. Chow. 1970. On Optimum Recognition Error and Reject Tradeoff.
  *IEEE Transactions on Information Theory*, 16(1):41–46.
  https://ieeexplore.ieee.org/document/1054406/
- **[Geifman2017]** Yonatan Geifman and Ran El-Yaniv. 2017. Selective Classification for
  Deep Neural Networks. In *Advances in Neural Information Processing Systems 30*, pages
  4878–4887. https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks
- **[Grundkiewicz2019]** Roman Grundkiewicz, Marcin Junczys-Dowmunt, and Kenneth Heafield.
  2019. Neural Grammatical Error Correction Systems with Unsupervised Pre-training on
  Synthetic Data. In *Proceedings of the 14th Workshop on Innovative Use of NLP for
  Building Educational Applications (BEA)*, pages 252–263.
- **[Gu2016]** Jiatao Gu, Zhengdong Lu, Hang Li, and Victor O.K. Li. 2016. Incorporating
  Copying Mechanism in Sequence-to-Sequence Learning. In *Proceedings of ACL 2016*.
- **[Gu2019]** Jiatao Gu, Changhan Wang, and Junbo Zhao. 2019. Levenshtein Transformer. In
  *Advances in Neural Information Processing Systems 32*.
  https://papers.nips.cc/paper/9297-levenshtein-transformer
- **[Holtzman2020]** Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020.
  The Curious Case of Neural Text Degeneration. In *ICLR 2020*.
  https://openreview.net/forum?id=rygGQyrFvH
- **[Huang2017]** Cheng-Zhi Anna Huang, Tim Cooijmans, Adam Roberts, Aaron Courville, and
  Douglas Eck. 2017. Counterpoint by Convolution. In *Proceedings of ISMIR 2017*.
  https://archives.ismir.net/ismir2017/paper/000187.pdf
- **[Kasewa2018]** Sudhanshu Kasewa, Pontus Stenetorp, and Sebastian Riedel. 2018. Wronging
  a Right: Generating Better Errors to Improve Grammatical Error Detection. In
  *Proceedings of EMNLP 2018*, pages 4977–4983. https://aclanthology.org/D18-1541/
- **[Kiyono2019]** Shun Kiyono, Jun Suzuki, Masato Mita, Tomoya Mizumoto, and Kentaro Inui.
  2019. An Empirical Study of Incorporating Pseudo Data into Grammatical Error
  Correction. In *Proceedings of EMNLP-IJCNLP 2019*, pages 1236–1242.
  https://aclanthology.org/D19-1119/
- **[Malmi2019]** Eric Malmi, Sebastian Krause, Sascha Rothe, Daniil Mirylenka, and
  Aliaksei Severyn. 2019. Encode, Tag, Realize: High-Precision Text Editing. In
  *Proceedings of EMNLP-IJCNLP 2019*. https://aclanthology.org/D19-1510/
- **[McLeod2020]** Andrew McLeod, James Owers, and Kazuyoshi Yoshii. 2020. The MIDI
  Degradation Toolkit: Symbolic Music Augmentation and Correction. In *Proceedings of
  ISMIR 2020*. https://arxiv.org/abs/2010.00059
- **[Omelianchuk2020]** Kostiantyn Omelianchuk, Vitaliy Atrasevych, Artem Chernodub, and
  Oleksandr Skurzhanskyi. 2020. GECToR – Grammatical Error Correction: Tag, Not
  Rewrite. In *Proceedings of the 15th Workshop on Innovative Use of NLP for Building
  Educational Applications (BEA)*, pages 163–170. https://aclanthology.org/2020.bea-1.16/
- **[Pearce2005]** Marcus T. Pearce. 2005. *The Construction and Evaluation of Statistical
  Models of Melodic Structure in Music Perception and Composition.* PhD thesis, City
  University, London.
- **[Ramoneda2024]** Pedro Ramoneda, Martín Rocamora, and Taketo Akama. 2024. Music
  Proofreading with RefinPaint: Where and How to Modify Compositions given Context. In
  *Proceedings of ISMIR 2024*. https://arxiv.org/abs/2407.09099
- **[Rei2016]** Marek Rei and Helen Yannakoudakis. 2016. Compositional Sequence Labeling
  Models for Error Detection in Learner Writing. In *Proceedings of ACL 2016*.
  https://aclanthology.org/P16-1112/
- **[Salazar2020]** Julian Salazar, Davis Liang, Toan Q. Nguyen, and Katrin Kirchhoff.
  2020. Masked Language Model Scoring. In *Proceedings of ACL 2020*.
  https://aclanthology.org/2020.acl-main.240/
- **[Thickstun2023]** John Thickstun, David Hall, Chris Donahue, and Percy Liang. 2023.
  Anticipatory Music Transformer. arXiv:2306.08620. https://arxiv.org/abs/2306.08620

Data:

- **[MTC]** The Meertens Tune Collections; MTC-ANN: 360 melodies in 26 tune families.
  https://www.liederenbank.nl/mtc/ — cite the MTC-ANN documentation and the MTCFeatures
  Zenodo record (https://zenodo.org/records/3551003); CC BY-NC-SA.
- **[Essen]** Helmut Schaffrath. 1995. *The Essen Folksong Collection in Kern Format.*
  D. Huron (ed.). Menlo Park, CA: Center for Computer Assisted Research in the
  Humanities.

---

## Supplementary material

*Blocks trimmed from the main text. Each stands alone; move them into a supplement if
the venue allows one, otherwise drop them. Numbering is independent of the main text.*

### S1. Matched degradation, all bands

Two-sided judge AUROC for the notes that differ from the reference, drafts banded by
$J$'s ΔNLL at shared cut-points. MTC (Essen); τ = 0.585 (MTC), 0.503 (same-corpus judge).

| ΔNLL band | D1 random | D3 rule-based | D2 generated |
|---|---|---|---|
| ≤ 0 | .753 (.722) | .638 (.618) | .606 (.579) |
| 0 – 0.1 | .830 (.826) | .774 (.740) | .684 (.670) |
| 0.1 – 0.2 | .844 (.834) | .807 (.801) | .727 (.728) |
| 0.2 – 0.4 | .855 (.837) | .825 (.816) | .730 (.724) |
| 0.4 – τ | .845 (.831) | .838 (.831) | .748 (.738) |
| > τ | .861 (.855) | .850 (.838) | .775 (.779) |

D2 drafts change more notes than D1 drafts in every band (8.8–13.1% against 3.9–9.4% on
MTC). For random corruption, localization is nearly flat once a draft is worse at all
(0.83–0.86); for generated drafts it climbs slowly and never reaches that level. The
trained D2 head varies little across bands (0.62–0.67), so the gradient is the judge's.
On Essen, D3's 0–0.1 band (0.740) lies below D2's > τ band (0.779), so the strict
"below every positive band" comparison holds against D1 only.
(`results/refiner/controls/matched_strata.json`, `scripts/matched_strata.py`.)

### S2. Contiguity at matched run length

Drafts banded by their longest run of adjacent changed events; two-sided judge AUROC.

| longest run | D1 random | D3 rule-based | D2 generated |
|---|---|---|---|
| 1 event | .839 (.831) | .938 (.915) | .746 (.728) |
| 3+ events | — (n = 7) (.834) | .751 (.723) | .680 (.677) |

The trained head shows the same ordering for runs of 3+ (D2 0.64, D3 0.71 on MTC).
(`results/refiner/controls/locality_all_sources.json`.)

### S3. Using the judge as a gate

Gated editing's improvement over copy-input (%), decoder fixed; seeds 0–2; thresholds
and combination weights chosen on D1 validation only. MTC (Essen).

| gate | D1 random | D3 rule-based | D2 generated |
|---|---|---|---|
| trained head (τ = 0.8) | +6.35 (+4.47) | +9.98 (+7.30) | 0.00 (0.00) |
| two-sided judge (z ≥ 3) | +2.81 (+2.86) | +7.96 (+6.25) | −0.47 (−1.17) |
| judge, imitation decoder (no edit labels) | +2.43 (+2.43) | +7.29 (+6.17) | −0.16 (−1.14) |
| judge + head | **+7.91** (**+6.52**) | **+11.80** (**+8.71**) | −0.39 (−0.46) |

As an encoder input, judge surprisal improves localization (AUROC 0.73 → 0.82 on MTC D1)
but lowers gated gain (+6.35% → +5.14%, three of three seeds) and collapses free decoding
(−17.7% → −61.0%). The judge was early-stopped on the melodies the validation drafts are
made from, so judge gates looked better on validation (+9.8% on D1) than on test. The
judge + imitation-decoder variant needs no edit labels in training, and is the setting
that would apply to unlabelled drafts. (`results/refiner/headline_r2/judge_gate.json`.)

### S4. Risk–coverage of the gate

Threshold swept on test, as a description, not a selection (3 seeds). Gated editing
beats copy-input on every seed at every τ from 0.2 to 0.95: D1 +5.0 to +6.4% (MTC),
+2.7 to +4.6% (Essen); D3 +8.9 to +10.3% and +6.5 to +7.3%. On MTC D1, τ = 0.2 opens the
gate on 3.2% of events at precision 0.50, τ = 0.95 on 1.0% at 0.90. On D2 every τ from
0.2 to 0.7 is at or below copy-input (MTC −0.97% to 0.00; Essen −1.14% to −0.01%, 0/3
seeds each), and from 0.8 up the gate stays shut.
(`results/refiner/headline_r2/risk_coverage.json`.)

### S5. The generator-quality curve

Zero-shot localization of each generator setting's drafts, with D1 and D3: the two-sided
judge, and the D1-trained head (3 seeds; trained on edit labels only, never on $J$).
*Gen. NLL*: the generator's own held-out per-token NLL (temperature does not change it).

| condition | gen. NLL (MTC / Essen) | MTC per unit | judge AUROC MTC (Essen) | D1-head AUROC MTC (Essen) |
|---|---|---|---|---|
| D2, T = 0.8 | 1.23 / 1.32 | 0.86 | .652 (.647) | .569 (.610) |
| D2, T = 1.0 | 1.23 / 1.32 | 1.04 | .688 (.683) | .584 (.621) |
| D2, T = 1.2 | 1.23 / 1.32 | 1.21 | .718 (.715) | .598 (.635) |
| D2, 10 epochs | 1.50 / 1.59 | 1.41 | .769 (.764) | .590 (.636) |
| D2, 5 epochs | 1.70 / 1.74 | 1.70 | .832 (.823) | .604 (.646) |
| D2, 2 epochs | 1.84 / 1.91 | 2.01 | .865 (.858) | .641 (.683) |
| D3 rule-based | — | 3.14 | .812 (.790) | .697 (.707) |
| D1 random | — | 3.94 | .838 (.831) | .727 (.732) |

The D3-trained head, applied the same way, is nearly flat across generator settings
(0.57–0.59 MTC, 0.62–0.64 Essen). (`results/judge/generator_quality.json`,
`results/judge/severity_head.json`.)

The 2-epoch refiners: head AUROC 0.67–0.76 (MTC) and 0.73–0.81 (Essen); gate precision
0.59–0.69 and 0.66–0.72; free decoding (B1) −11.4% ± 5.1 and −28.4% ± 1.3.
(`results/judge/severity_curve.json`, `results/refiner/headline_ep2/`.)

### S6. Plausibility details

The variants' median ΔNLL is +0.031; resampling which pairs are drawn moves τ by about
0.02. Because the pairs come from the test corpus, τ is used only for analysis bands.
τ is a conservative bar for synthetic drafts — calibrated on variants differing in 92%
of their notes, with ΔNLL a per-token mean — so only 13.0% of D1 drafts clear it, 9.6%
of D3 and 2.6% of D2; it certifies degradation and never declares a draft error-free.
At matched distance the contrast between error and alternative is starkest but thin: the
nine variant pairs that differ as little as synthetic drafts have median ΔNLL 0.000,
against +0.18 for synthetic drafts at the same distance. A third of the melodies free
decoding changes end up both more plausible and further from the reference (one seed);
5.2% of D1 drafts, and 14.5% of free-decoded D1 outputs, outscore the reference.

### S7. Method and representation details

The vocabulary has 121 tokens: 76 pitches, 40 durations, rest, bar and three special
tokens. Bars are structure, not events: an edit never inserts or deletes a barline, and
an onset shift is a duration change. Edit labels collapse five types (keep, pitch
substitution, duration substitution, insert-before, delete); the five-way head's
comparison was measured at τ = 0.9. The selection grid is four class weightings × four
α × six τ, three seeds each; every weighting reaches within 0.2 points of the best.
Batch size 32, up to 40 epochs. Gated decoding is greedy under a mask enforcing
pitch/duration alternation, and bar tokens are always copied. The oracle is a separately
trained model of the same architecture with true edit positions as input, run once
under an earlier configuration; it removes 49.3% of D1's and 21.5% of D2's CC. Gated
editing improves 22.4% of D1 melodies and worsens 6.5%; free decoding changes about
three times as many notes. The D2 generator is a three-layer transformer; D1 is uniform
corruption of the kind used in prior work [McLeod2020].

### S8. Additional controls

*B2 (head, α = 0, free decoding):* −18.8 ± 6.8 (D1 MTC), −90.1 ± 9.6 (D1 Essen),
+5.3 ± 3.0 (D3 MTC), −21.8 ± 4.9 (D3 Essen), −5.7 ± 1.1 (D2 MTC), −26.1 ± 3.5 (D2 Essen);
within the seed noise of B1 everywhere. *Five seeds versus a re-run:* two complete runs
of the main experiment differ by at most 0.36 points in any cell mean. *Transfer
precision:* heads trained on D1 or D3 open the gate on D2 at precision 0.27 and 0.21.
*D3 intrinsic ease:* a D1-trained head gains +7.25% on D3, more than its own +6.35% on D1.

### S9. Further limitations

The gate repairs pitch far more than rhythm: metric misalignment in D1 drafts is
essentially unchanged by gating, while free decoding reduces it. Generated drafts are
not filtered to those the judge certifies as degraded; S1 suggests filtering would not
change the conclusion, but it would change D2's error distribution. The negative result
is bounded by the detectors tried: trained per-note and per-span detectors, transfer
from other sources, and two untrained judges.

---
