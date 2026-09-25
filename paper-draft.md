# Paper draft

Written section by section from `paper-core.md` (claims, tiers, canonical numbers). Each
section notes the claims it carries and its page budget. Headline numbers are 5 seeds;
comparisons with conditions run on 3 seeds use the matching 3 (see `paper-core.md`).

**Working title:** Generated Drafts Leave Little Local Trace: When Does Localize-and-Fix
Work for Symbolic Music Refinement?

Status: §1–§6 drafted (2026-09-25). Other sections are placeholders.

---

## Abstract

*(Written last.)*

## 1. Introduction

*Budget 1.0 page; this draft is ~820 words. Carries the thesis and C1–C7; P2 claims
appear only as pointers.*

Generative models of symbolic music now produce plausible melodies on demand, but a
draft is rarely final. Someone — a composer, or a second model — has to decide which
notes to change and which to keep. This *refinement* problem differs from generation in
one important respect: most of the draft is already right, and a system that rewrites it
wholesale has not refined anything. The natural formulation is therefore
*localize-then-fix*: find the notes that are wrong, and change only those. Recent work
has pursued exactly this, with models that locate and correct degradations in symbolic
excerpts [McLeod2020] and feedback models that pick out the weak notes of a composition
for resampling [Ramoneda2024].

The standard recipe for training and evaluating such systems rests on two assumptions.
Training data is made by *corrupting* clean melodies — replacing, shifting, deleting or
inserting notes — and a model learns to undo the damage. Evaluation measures the edit
distance between the model's output and the original melody. The first assumption is
that synthetic corruption resembles the errors real drafts contain; the second, that
edit distance to a reference measures improvement. This paper tests both, and finds
that each holds only on one side of a boundary that matters.

**Edit distance cannot tell an error from an alternative.** We train an independent
melody language model as a plausibility judge and calibrate it against human-authored
variants of the same folk tune — valid alternatives by construction. The judge scores
them as neither better nor worse than one another, as it should. But the variants sit
92% of the way apart in normalized edit distance, while synthetic corruption sits 7%
away and is roughly 100× more implausible per unit of change. Across draft sources,
distance and plausibility are unrelated; the drafts furthest from their references are
the most plausible. Edit distance can rank methods on a single draft source, but it
cannot compare sources, and it cannot say whether a change was a correction (§4).

**Generated drafts leave little local trace.** We then compare three draft sources
under one model, one training recipe and one setting chosen on validation data: random
corruption, rule-based musical degradation, and spans rewritten by a trained generator.
The errors in corrupted drafts are readily found: a per-note detector ranks them at
5.8× chance, and an untrained two-sided judge — which scores each note given everything
around it — at 6–7×. Generator-written material is another matter. The best detector we
built, that same untrained judge, finds it at 2.7× chance; the trained detector finds
almost nothing and flags no position confidently enough to act on. The reason is not
that the detectors are weak — the untrained judge outperforms every trained one — but
that most generated drafts are not wrong. Only 2.5–4.4% are measurably worse than their
reference, and on those the judge's localization approaches that for corruption. The
rest are *different*, and edit labels derived from a reference mistake difference for
error (§7).

**A method that respects the boundary.** Where errors are local, localize-then-fix
works, provided copying is the default. A plain sequence-to-sequence refiner is 25%
*worse* than simply returning the draft: it rewrites notes that were right. We instead
gate each note — it is copied unless the model is confident it needs changing, the reject
option of selective classification [Chow1970, Geifman2017] applied per note. At the
strictest threshold the output is the draft itself, so the method's floor is the draft
by construction. On random corruption, gated editing improves on the draft by 6.1%
in edit distance on every one of five seeds, and it is the only system we test that improves on the draft in
both edit distance *and* judged plausibility (§6). On generated drafts it declines to
edit — which, given the evidence above, is the correct behaviour rather than a failure.

In summary, we make three contributions:

1. **A measurement.** A plausibility judge calibrated on human variants, and evidence
   that edit distance to a reference — the field's standard metric — cannot distinguish
   an error from a valid alternative, and should be reported alongside plausibility.
2. **A boundary.** A controlled comparison of random, rule-based and generated drafts
   showing that localize-then-fix applies where errors leave a local trace and not where
   a generator wrote the draft, with evidence from trained and untrained detectors,
   transfer between sources, and a same-corpus replication. It also measures how much of
   a rule-based benchmark's score comes from learning its own rules.
3. **A method.** Precision-gated editing, whose downside is bounded by construction,
   which improves drafts where errors are local and abstains where they are not.

Our experiments use monophonic folk melodies from the Meertens Tune Collections [MTC]
and the Essen Folksong Collection [Essen]. Plausibility is judged by a language model,
not by listeners; we treat this as the central limitation and return to it in §8.

**Figure 1** *(proposed)*. Three draft sources ordered by how far their changes move a
melody off the data distribution (judged implausibility per unit of change). Left axis:
how well an untrained two-sided judge localizes the changed notes (AUROC). Right axis:
gated editing's improvement over the draft. Corruption (random, rule-based) is
localizable and improvable; generator-written drafts are neither, and the method
abstains.

## 2. Related Work

*Budget 0.6 page; this draft is ~680 words, so expect to trim ~15%. Trim order: the
edit-model list in ¶2 to two citations, then the evaluation sentence in ¶4.*

**Symbolic music correction.** The closest precedent is the MIDI Degradation Toolkit
(MDTK) [McLeod2020], which applies rule-defined degradations to symbolic excerpts and
poses four tasks of increasing difficulty — detecting, classifying, locating and
correcting the error — aimed at post-processing automatic transcription. Our random
(D1) and music-theoretic (D3) draft sources belong to this family. What we add is a
measurement of how much a detector trained on such data learns the injection process
itself rather than musical error: under transfer, roughly a quarter of our rule-based
benchmark's gain comes from learning its rules (§7). RefinPaint [Ramoneda2024] is closest
in method: a feedback model identifies the weaker elements of a composition and guides
an inpainting model to resample them, addressing where and how to modify a piece given
its context. We ask the prior question of *when* such localization is possible, and
answer it on drafts written by a generator, where it largely is not. Our D2 drafts are
produced by span infilling, in the line of models that condition on arbitrary subsets of
a score [Huang2017, Thickstun2023]; our two-sided judge scores each note from exactly
such conditionals. See [Ji2023] for a survey of symbolic generation.

**Edit-based sequence transduction.** When input and output overlap heavily, predicting
edits is more precise and cheaper than regenerating the sequence. Tagging models label
each input token keep, delete or insert [Malmi2019, Omelianchuk2020]; others emit edit
programs [Dong2019], span-level edits [Stahlberg2020], or iterate insertion and deletion
[Gu2019]; copy mechanisms let a decoder reuse its input directly [Gu2016, See2017]. Our
gated decoder combines the two families: a per-event head decides *whether* to edit, and
a decoder supplies *what* to write only where the gate opens, so that copying is the
default rather than a behaviour the model must learn. We find that without such a
default, a sequence-to-sequence refiner converges on copying the draft (§6).

**Error detection and the realism of synthetic errors.** Grammatical error detection is
framed as sequence labeling [Rei2016], and correction systems lean heavily on synthetic
errors for pre-training [Grundkiewicz2019, Kiyono2019]. How realistic those errors are
matters: errors generated by a sequence-to-sequence model trained on real learner
errors improve detectors beyond the previous state of the art, and human readers
struggle to tell them from real errors [Kasewa2018]. Our drafts span the same range — random, rule-based and model-generated —
and show a sharp consequence of realism in music: the more a draft resembles the output
of a real generator, the less local trace its differences leave. Reference-based
evaluation compounds the problem. Single-reference scoring in grammatical error
correction undercounts valid alternatives [Bryant2015], and code-repair benchmarks judge
a patch by whether tests pass rather than by its distance to a reference diff
[Jimenez2024]. We quantify the musical analogue: edit distance to a reference cannot
tell an error from a valid alternative (§4).

**Plausibility and its limits.** Melodic expectation has long been modelled as the
information content of a note under a statistical model trained on a corpus [Pearce2005];
our judge is a neural instance of this idea, calibrated against human variants of the
same tune. Scoring each note with a masked, bidirectional model follows pseudo-log-
likelihood scoring of text [Salazar2020]. Likelihood is an imperfect proxy for quality:
it can be largely independent of sample quality [Theis2016], and maximization-based
decoding produces high-likelihood but degenerate text [Holtzman2020] — which is the
behaviour we observe in free decoding, whose outputs sometimes outscore the reference
itself (§6). Evaluating symbolic generation beyond likelihood remains open
[Yang2020]; we report edit distance and plausibility side by side for that reason.

**Selective prediction.** Declining to act when uncertain is the reject option of
classification [Chow1970], developed for deep networks as selective classification
[Geifman2017]. Our gate applies it per note: an event is changed only when the model is
confident, and at the strictest threshold the output is the draft itself, which bounds
the method's downside by construction.

## 3. Task, drafts and representation

*Budget 0.9 page; this draft is ~760 words plus Table 1. Carries the definitions every
later section uses. P3 detail (representation measurements) is compressed to one
sentence.*

**Task.** A *draft* $M_0$ is a monophonic melody that differs from a *reference* $M^*$
in some notes. A refiner receives $M_0$ alone and outputs a melody $\hat M$; it succeeds
to the extent that $\hat M$ is closer to $M^*$ than $M_0$ was, without rewriting the
notes that were already right.

**Corpora and splits.** References come from two collections of monophonic folk song.
The Essen Folksong Collection [Essen] (8,469 melodies, 54.5 notes on average) supplies
training data; it is split in half at random, one half for the plausibility judge of §4
and the other for the generator that writes D2 drafts, so that the judge never scores
material it could have memorized from the generator's training set. Refiners are trained
on drafts of the judge's half (3,812 melodies) and validated on a further 423. The test
set is MTC-ANN, the annotated corpus of the Meertens Tune Collections [MTC]: 360 melodies
(49.0 notes on average) grouped by folk-song specialists into 26 *tune families* —
melodies believed to descend from a common ancestor through oral transmission. MTC-ANN
is seen by no model during training, which makes every main result cross-corpus; a
same-corpus control, which holds out 20% of Essen as the test set and retrains the judge
and generator on the rest, is reported in §7.

**Draft sources.** Each test reference yields three drafts per source.

- **D1, random corruption.** One to five operations, drawn as pitch replacement (40%),
  duration change (30%), deletion (15%) and insertion (15%). Replacement pitches and
  durations are sampled from the corpus's own marginal distributions, so each edit
  individually uses plausible material. This is uniform corruption of the kind used to
  build training data in prior work [McLeod2020], and our main control.
- **D2, generated.** A span of 2–6 consecutive notes is rewritten by a generator — a
  three-layer transformer language model trained on the other half of Essen — sampling
  at temperature 1.0 from the preceding notes only, after which the reference resumes.
  The splice is not smoothed: a passage that fails to rejoin its continuation is a
  realistic draft flaw. We use span rewriting rather than free continuation, which
  changes 89% of a melody and stops describing a draft at all.
- **D3, rule-based musical degradation.** One to five applications of six rules, each
  a violation music theory names: an out-of-scale pitch given the annotated key, a leap
  beyond 14 semitones, a structurally strong note moved off the beat, a disrupted final
  cadence, a varied rhythm flattened to uniform durations, and a figure collapsed to a
  repeated pitch.
- **D4, human variants.** Pairs of melodies from the same tune family (1,012 pairs, at
  most 40 per family). These are valid alternatives by construction. They are used only
  to calibrate the judge (§4) — family members differ in metre and length, so they are
  variants of a tune, not a draft and its revision, and we do not refine them.

**Table 1** *(proposed)*. Draft sources on the MTC test set (three drafts per
reference; D4 is the 1,012 variant pairs). Fraction changed is Correction Cost divided
by reference length.

| source | how the draft is made | mean Correction Cost | fraction of notes changed |
|---|---|---|---|
| D1 random | 1–5 random edits | 2.98 | 6.6% |
| D2 generated | 2–6 notes rewritten by a generator | 4.71 | 10.6% |
| D3 rule-based | 1–5 music-theoretic violations | 4.11 | 9.1% |
| D4 human variants | a different member of the same tune family | 45.7 | 92.3% |

**Representation.** A melody is a sequence of events, each a pitch (or an explicit rest)
and a duration in integer ticks at 24 per quarter note, which represents every duration
in both corpora exactly. Each event becomes two tokens, and a bar token is inserted where
the annotated metre begins a new bar, which bounds how far a single duration error can
shift later notes. The vocabulary has 121 tokens: 76 pitches, 40 durations, rest, bar,
and three special tokens. Bars are structure, not events: an edit never inserts or
deletes a barline. Onset shifts are not a separate operation; under this encoding they
are a duration change.

**Alignment and Correction Cost.** Two melodies are aligned by Wagner–Fischer edit
distance over events, with unit costs for insertion, deletion, pitch substitution and
duration substitution, so an event differing in both attributes costs 2. *Correction
Cost* (CC) is the minimal total cost. The same alignment supplies training labels: each
draft event is labelled *keep* or *edit* (collapsed from five types — keep, pitch
substitution, duration substitution, insert-before and delete — which §5 shows add
nothing), and an extra end-of-sequence slot receives labels for notes missing after the
last draft event.

**Evaluation.** Our primary measure is the relative reduction in mean CC over returning
the draft unchanged (*copy-input*, B3): $(\overline{\mathrm{CC}}(M_0, M^*) -
\overline{\mathrm{CC}}(\hat M, M^*)) \,/\, \overline{\mathrm{CC}}(M_0, M^*)$, with means
taken over the test drafts, so that positive values mean the refiner improved on doing
nothing. We also report *fidelity*, the CC between output and draft,
which distinguishes a refiner that edits from one that regenerates; per-melody win and
loss rates against copy-input; and, in §4, plausibility. Every result is the mean over
five training seeds for the headline and three for secondary comparisons, with the seed
standard deviation, under one configuration
selected on validation data (§5).

## 4. Measuring plausibility

*Budget 1.0 page; this draft is ~770 words plus Table 2. Carries C1 and C2. All numbers
from `results/judge/plausibility_table.json`, computed on the drafts of Table 1.*

Edit distance says how far a draft is from its reference, not whether it is worse. To
ask the second question without a listening study, we use a language model as a judge
and calibrate it against melodies that are known to be acceptable.

**The judge.** $J$ is a decoder-only transformer language model over the token sequence
of §3 (four layers, width 256), trained on the judge's half of Essen with early stopping
on held-out Essen melodies. It shares no training data with the generator that writes D2
drafts, which is trained on the other half and differs in depth; a judge trained on the
generator's data could simply recognize its output. $J$ scores a melody by its mean
per-token negative log-likelihood — per token, because insertions and deletions change
length — and a draft by its degradation relative to its reference,

$$\Delta\mathrm{NLL}(M_0) = \mathrm{NLL}_J(M_0) - \mathrm{NLL}_J(M^*).$$

Positive values mean the draft is less plausible than its reference. Both melodies are
given barlines inferred from the reference's metre, so neither benefits from annotation
the other lacks. Because the difference is paired, the offset between $J$'s training
corpus and MTC — $J$ fits MTC 0.137 nats per token worse than Essen — cancels; its effect
on the judge's *sensitivity* does not, and §7 rules it out with a same-corpus replication.

**Calibration on human variants.** A judge that penalized any departure from the
reference would call every alternative an error, and would be useless for our question.
We therefore score the D4 pairs: two members of one tune family, each a valid melody in
its own right. Their ΔNLL has mean +0.054 and median +0.031, and 45.0% of pairs score at
or below zero — the judge is close to indifferent between variants that differ in 92% of
their notes. It measures degradation, not difference. We set the degradation threshold
$\tau$ to the 95th percentile of the variants' ΔNLL, $\tau = 0.585$: a draft above $\tau$
is less plausible than 95% of human alternatives are. (Resampling which variant pairs are
drawn moves $\tau$ by about 0.02.)

**Table 2** *(proposed)*. Draft sources under the judge (MTC). *Per unit* is the median
of ΔNLL divided by the fraction of notes changed; ρ is Spearman's correlation between
fraction changed and ΔNLL across drafts, with a 95% interval from resampling tune
families.

| source | fraction changed | ΔNLL mean | ΔNLL per unit | ΔNLL ≤ 0 | ΔNLL > τ | ρ (95% CI) |
|---|---|---|---|---|---|---|
| D4 human variants | 92.3% | +0.054 | +0.04 | 45.0% | 5.0% | +0.14 [+0.04, +0.24] |
| D1 random | 6.6% | +0.298 | +3.94 | 5.2% | 13.0% | +0.52 [+0.46, +0.59] |
| D3 rule-based | 9.1% | +0.271 | +3.14 | 4.2% | 9.6% | +0.47 [+0.41, +0.52] |
| D2 generated | 10.6% | +0.139 | +1.04 | 20.5% | 2.6% | +0.20 [+0.11, +0.30] |
| all pooled | | | | | | −0.06 [−0.11, −0.01] |

**Corruption is damage, but of an unrealistic kind.** A common worry about random
corruption is that a replaced note may simply produce a valid alternative, so that the
model learns to undo harmless changes. Table 2 does not support that worry: 95% of D1
drafts are less plausible than their references. The problem is the opposite one. Random
and rule-based corruption change 7–9% of a melody at a ΔNLL of about 3–4 per unit of
change — **80–100× more per unit than a human variant**. Corruption moves a melody a
short distance *off* the distribution of plausible melodies; human variation moves it a
long way *along* it. Generated drafts sit between the two, at about 1 nat per unit, and
are the only synthetic source that is often no worse than its reference (20.5%) and
rarely clearly degraded (2.6% above $\tau$).

**Edit distance cannot tell an error from an alternative.** Within a single corruption
process, more change does mean more damage: fraction changed and ΔNLL correlate at
ρ ≈ 0.5 for D1 and D3, partly by construction, since the number of injected edits drives
both. Across sources the relation disappears and slightly reverses (ρ = −0.06 pooled): the
drafts furthest from their references are the most plausible. The contrast is starkest at
matched distance, though the evidence is thin — the nine variant pairs that differ as
little as synthetic drafts do have median ΔNLL 0.000, against +0.18 for synthetic drafts
at the same distance.

Edit distance to a reference is therefore a usable way to rank methods *on one draft
source*, where it tracks damage. It cannot compare draft sources, and it cannot say
whether a change was a correction or a valid alternative. We report both measures
throughout: Correction Cost for comparability with prior work, and ΔNLL for whether the
music got better. ΔNLL is itself a statistical judgement, not a musician's; we return to
this limitation in §8.

## 5. Method

*Budget 0.8 page; this draft is ~640 words. Carries the method behind C5–C6, the
selection protocol, and the head-type ablation (P3) in one sentence.*

**Refiner.** The refiner is an encoder–decoder transformer (three encoder and three
decoder layers, width 256, four heads; 5.86M parameters). The encoder reads the draft;
the decoder writes the refined melody autoregressively, trained with teacher forcing on
the reference. We use an encoder–decoder rather than a decoder-only model for two
reasons: deciding whether a note is wrong requires seeing the notes after it, and drafts
and references differ in length whenever a note has been inserted or deleted, so they
cannot be aligned position by position.

**Edit head.** A linear layer on the encoder's output predicts, at every draft event,
whether that event needs editing, supervised by the alignment labels of §3. The model is
trained on

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + \alpha\,\mathcal{L}_{\text{edit}},$$

where $\mathcal{L}_{\text{gen}}$ is cross-entropy over the reference's tokens and
$\mathcal{L}_{\text{edit}}$ is cross-entropy over the keep/edit labels at the draft's
positions. The head adds 514 parameters. A five-way head that also predicts the edit's
type performs no better than this binary one (a seed-matched difference of
+0.10 ± 0.36 points in improvement over the draft, measured at τ = 0.9), and never
predicts an insertion correctly, so we use the binary head throughout.

**Gated decoding.** Free decoding — letting the decoder rewrite the whole melody — is the
standard way to use such a model, and it is where the problem lies: most of a draft is
already right, and each note the decoder rewrites is another chance to get one wrong. We
instead make copying the default. For each draft event, the head gives a probability
$p$ that the event needs editing. If $p < \tau$, the event is copied verbatim; if
$p \geq \tau$, the decoder writes a replacement event in its place, conditioned on
everything decoded so far. Bar tokens are always copied. Decoding is greedy, under a mask
that enforces the alternation of pitch and duration tokens.

This construction has a useful property. With a threshold no probability can reach, the
gate never opens and the output *is* the draft: gated decoding contains copy-input as a
special case, so its worst setting is doing nothing. Lowering $\tau$ trades coverage for
risk, exactly as in selective classification [Geifman2017], and the head's precision in
its confident range — not its overall accuracy — decides whether each additional edit
helps. In this form the gate substitutes events; it neither inserts nor deletes, which
limits what it can repair (§8).

**Baselines.** B1 is the same encoder–decoder without a head ($\alpha = 0$), decoded
freely: plain sequence-to-sequence imitation. B2 keeps the head but trains with
$\alpha = 0$, so it has exactly the refiner's parameters and differs only in the loss;
since the head is 0.009% of the model, parameter count is not a confound in any case. B3
returns the draft unchanged. We also report the edit-aware model decoded freely, which
separates what the loss changes from what the gate changes.

**One setting, chosen on validation.** The loss weight $\alpha$, the threshold $\tau$ and
the class weighting of $\mathcal{L}_{\text{edit}}$ interact, and choosing them by looking
at test results would inflate every number that follows. We therefore select them once,
on D1 validation drafts only, over a grid of four class weightings, four values of
$\alpha$ and six of $\tau$ (96 settings, three seeds each), by lowest mean gated Correction
Cost. The chosen setting — no class weighting, $\alpha = 0.5$, $\tau = 0.8$ — is then
applied unchanged to every draft source and both corpora. The landscape is flat: every
class weighting reaches within 0.2 points of the best, and 84 of the 96 settings improve
on copy-input for every seed, so no result below depends on a lucky choice. Models are
trained with AdamW (learning rate $3 \times 10^{-4}$, batch 32) for up to 40 epochs with
early stopping on validation loss. Runs are not bit-reproducible on GPU; two complete
re-runs of the main experiment differ by at most 0.36 points in any cell mean.

## 6. Results I: where localize-then-fix works

*Budget 1.3 pages; this draft is ~900 words plus Tables 3–4. Carries C5–C7, C11, C12,
and D3 as a control. Table 3 is 5 seeds (`headline_r2` + `headline_s34`); Table 4 is
seeds 0–2, matching the output-quality scoring.*

**Table 3** *(proposed)*. Improvement in mean Correction Cost over returning the draft
(copy-input), in percent; mean ± standard deviation over five seeds. Positive is better.
The last row gives how many seeds improve on copy-input.

| | D1 MTC | D1 Essen | D3 MTC | D3 Essen | D2 MTC | D2 Essen |
|---|---|---|---|---|---|---|
| B1 imitation, free decoding | −24.9 ± 11.6 | −92.2 ± 7.1 | +3.9 ± 1.9 | −22.3 ± 5.3 | −3.6 ± 1.3 | −27.1 ± 4.2 |
| B2 head, α = 0, free | −18.8 ± 6.8 | −90.1 ± 9.6 | +5.3 ± 3.0 | −21.8 ± 4.9 | −5.7 ± 1.1 | −26.1 ± 3.5 |
| edit-aware, free decoding | −17.8 ± 4.6 | −92.4 ± 10.8 | +4.0 ± 1.2 | −22.7 ± 4.2 | −4.2 ± 2.1 | −25.6 ± 5.4 |
| **edit-aware, gated** | **+6.06 ± 0.58** | **+4.25 ± 0.46** | **+9.81 ± 0.46** | **+7.38 ± 0.27** | 0.00 ± 0.00 | 0.00 ± 0.00 |
| gated: seeds > copy-input | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 (abstains) | 0/5 (abstains) |

**Gated editing improves drafts where errors are local.** On random corruption, gated
decoding reduces Correction Cost by 6.06% relative to the draft on MTC and by 4.25% in
the same-corpus setting, on every one of five seeds (Table 3). It improves 22.4% of
melodies and worsens 6.5%, leaving the rest untouched, and it changes about 1.5% of
notes, a third as many as free decoding. Of the events the gate hands to the decoder,
82% were genuinely wrong. The same setting, chosen on D1 validation data, transfers
unchanged to the other sources.

**Free decoding is worse than doing nothing.** Every freely decoded model is worse than
returning the draft on random corruption, on every seed: the imitation baseline B1 by
25% on MTC, and by more than 90% in the same-corpus setting. It rewrites notes that
were right. The one exception is MTC D3, discussed below. The
edit-aware loss does not rescue free decoding: with the head trained ($\alpha = 0.5$)
or present but untrained (B2), free decoding stays within the seed noise of B1 on every
cell. The gain in the last row comes from making copying the default, not from the
auxiliary loss alone; the loss's contribution is a head confident enough to gate with.

This also explains how an edit-aware objective can appear to work when it does not.
Measured only against B1, an auxiliary edit loss can look like a clear gain — in a
preliminary sweep on generated drafts, +6.3% over B1 on three of three seeds — while its
output is nearly identical to the draft (0.0009 notes changed per note) and no better
than copying it. The model had learned to copy. Without a copy-input baseline and a
fidelity measure, that is invisible; we report both throughout.

**How much is left.** An oracle that is told which events are wrong, and lets the same
decoder rewrite exactly those, removes 49.3% of the draft's Correction Cost on D1
(single run). Gated editing recovers about 12% of that headroom. The gap lies mainly in
localization: given the right positions, the same decoder's replacements remove most of
the error.

**Table 4** *(proposed)*. Plausibility of outputs on MTC (seeds 0–2). *Recovered* is the
share of the draft's ΔNLL that the output removes; for gated editing, the share over the
melodies the gate actually changed, and how many of those it made less plausible, is in
brackets.

| | D1: CC vs copy | D1: ΔNLL recovered | D3: CC vs copy | D3: ΔNLL recovered |
|---|---|---|---|---|
| draft (copy-input) | 0 | — (ΔNLL +0.298) | 0 | — (ΔNLL +0.271) |
| B1, free decoding | −24.7% | +40.9% | +4.3% | +48.8% |
| edit-aware, free | −17.7% | +42.5% | +3.7% | +51.0% |
| **edit-aware, gated** | **+6.35%** | +16.3% [+30.1%; 2.8% worse] | **+9.98%** | +38.8% [+47.5%; 1.4% worse] |

**Better music, not only closer to the reference.** A Correction Cost gain could come
from edits the judge finds implausible. It does not: on the melodies it changes, gated
editing removes 30% of the draft's implausibility on D1 and 48% on D3, and makes fewer
than 3% of them less plausible (Table 4). The overall figures are lower only because the
gate leaves most melodies alone. Free decoding recovers *more* plausibility — 41% on
D1 — while being 25% worse than copying on Correction Cost. The two measures disagree
because free decoding is greedy and drifts toward the model's most typical continuation:
14.5% of its D1 outputs score as more plausible than the reference itself (5.2% of
drafts do), and a third of the melodies it changes end up both more plausible and
further from the reference (one seed). This is the dissociation of §4 again, now in
model outputs. An evaluation on plausibility alone would prefer free decoding; one on
Correction Cost alone would hide that gated editing improves the music. Gated editing is
the only system that improves on the draft on both.

**D3 is easier, and partly for a bad reason.** Gated editing's largest gains are on
rule-based degradation — 9.81% on MTC and 7.38% same-corpus, every seed — and D3 is the
only source on which free decoding also beats copying on MTC. This should not be read as
the method working better on more musical errors. A refiner trained on D1 alone reaches
+7.25% on D3 drafts (seeds 0–2) against +9.98% for one trained on D3 itself: about 2.7
points of D3's gain come from learning the six rules that produced it, not from detecting
implausible music. §7 separates the two.

**Generated drafts.** On D2 the gate never opens: the output equals the draft on every
seed, and gated editing neither helps nor harms. Free decoding, by contrast, loses 3.6%
and 27.1%. §7 shows why abstaining is the right response.

## 7. Results II: where it does not

*(Pending — C3, C4, C8, C10, C13, C14. Budget 1.3 pages.)*

## 8. Discussion and limitations

*(Pending — C9; limitations L1–L6. Budget 0.8 page.)*

## 9. Conclusion

*(Pending. Budget 0.3 page.)*

---

## References

Every entry below was checked against the publisher or archive (2026-09-25). Keys are
the in-text citation keys. Data sources (MTC, Essen) are listed for §3.

- **[Bryant2015]** Christopher Bryant and Hwee Tou Ng. 2015. How Far are We from Fully
  Automatic High Quality Grammatical Error Correction? In *Proceedings of ACL-IJCNLP
  2015*, pages 697–707. https://aclanthology.org/P15-1068/
- **[Chow1970]** C. K. Chow. 1970. On Optimum Recognition Error and Reject Tradeoff.
  *IEEE Transactions on Information Theory*, 16(1):41–46.
  https://ieeexplore.ieee.org/document/1054406/
- **[Dong2019]** Yue Dong, Zichao Li, Mehdi Rezagholizadeh, and Jackie Chi Kit Cheung.
  2019. EditNTS: An Neural Programmer-Interpreter Model for Sentence Simplification
  through Explicit Editing. In *Proceedings of ACL 2019*, pages 3393–3402.
  https://aclanthology.org/P19-1331/
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
- **[Ji2023]** Shulei Ji, Xinyu Yang, and Jing Luo. 2023. A Survey on Deep Learning for
  Symbolic Music Generation: Representations, Algorithms, Evaluations, and Challenges.
  *ACM Computing Surveys*, 56(1), Article 7, 1–39. https://doi.org/10.1145/3597493
- **[Jimenez2024]** Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei,
  Ofir Press, and Karthik Narasimhan. 2024. SWE-bench: Can Language Models Resolve
  Real-world GitHub Issues? In *ICLR 2024*. https://arxiv.org/abs/2310.06770
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
- **[See2017]** Abigail See, Peter J. Liu, and Christopher D. Manning. 2017. Get To The
  Point: Summarization with Pointer-Generator Networks. In *Proceedings of ACL 2017*.
- **[Stahlberg2020]** Felix Stahlberg and Shankar Kumar. 2020. Seq2Edits: Sequence
  Transduction Using Span-level Edit Operations. In *Proceedings of EMNLP 2020*, pages
  5147–5159. https://aclanthology.org/2020.emnlp-main.418/
- **[Theis2016]** Lucas Theis, Aäron van den Oord, and Matthias Bethge. 2016. A Note on the
  Evaluation of Generative Models. In *ICLR 2016*. https://arxiv.org/abs/1511.01844
- **[Thickstun2023]** John Thickstun, David Hall, Chris Donahue, and Percy Liang. 2023.
  Anticipatory Music Transformer. arXiv:2306.08620. https://arxiv.org/abs/2306.08620
- **[Yang2020]** Li-Chia Yang and Alexander Lerch. 2020. On the Evaluation of Generative
  Models in Music. *Neural Computing and Applications*, 32:4773–4784.
  https://doi.org/10.1007/s00521-018-3849-7

Data (for §3):

- **[MTC]** The Meertens Tune Collections; MTC-ANN: 360 melodies in 26 tune families.
  https://www.liederenbank.nl/mtc/ — cite the MTC-ANN documentation and the MTCFeatures
  Zenodo record (https://zenodo.org/records/3551003); CC BY-NC-SA.
- **[Essen]** Helmut Schaffrath. 1995. *The Essen Folksong Collection in Kern Format.*
  D. Huron (ed.). Menlo Park, CA: Center for Computer Assisted Research in the
  Humanities.

*Still to verify before submission* (search confirmed title, venue and first authors,
but these details were filled from memory): full author lists of Malmi2019 (after
Krause), Huang2017 (after Cooijmans), and Jimenez2024 (after Press); Gu2016 and See2017
bibliographic details and page ranges. Also choose the exact MTC citation (paper vs
documentation) when writing §3.
