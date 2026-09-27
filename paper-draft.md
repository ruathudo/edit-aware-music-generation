# Paper draft

Written section by section from `paper-core.md` (claims, tiers, canonical numbers). Each
section notes the claims it carries and its page budget. Headline numbers are 5 seeds;
comparisons with conditions run on 3 seeds use the matching 3 (see `paper-core.md`).

**Working title:** Generated Drafts Leave Little Local Trace: When Does Localize-and-Fix
Work for Symbolic Music Refinement?

Status: full draft, all sections (2026-09-26). Consistency pass done: no must-not-claim
statement; superseded numbers absent; 3-seed figures only in seed-matched contexts; all
table, figure and section references resolve.

---

## Abstract

*~235 words.*

Refining a draft melody — changing the few notes that are wrong and keeping the rest —
is naturally framed as *localize-then-fix*, and such systems are typically trained on
synthetically corrupted melodies and evaluated by their edit distance to the original.
We test both halves of that recipe on monophonic folk melodies. Using a melody language
model, calibrated against human variants of the same tune, as a plausibility judge, we
show that edit distance cannot tell an error from a valid alternative: human variants
differ in 92% of their notes yet are judged equally plausible, while synthetic corruption
is two orders of magnitude more implausible per unit of change. Comparing random, rule-based and
generator-written drafts under a single configuration, we find that localize-then-fix
works where errors leave a local trace, and not where a generator wrote the draft. The
strongest detector we built — an untrained judge that scores each note given everything
around it — finds corrupted notes at 6–7 times chance but generated ones at 2.7 times,
and the gap persists when drafts are matched for how much they degrade the melody:
generated damage is real, but diffuse. A
precision-gated refiner, which copies each note unless it is confident the note is wrong,
improves randomly corrupted drafts by 6.1% over leaving them unchanged on every seed, is
the only system tested that improves them in both edit distance and plausibility, and abstains
on generated drafts, where a plain sequence-to-sequence refiner does harm. A transfer
test further shows that about a quarter of the gain on a synthetic benchmark, random or
rule-based alike, does not survive a change of corruption process.

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
away and is two orders of magnitude more implausible per unit of change. Across draft
sources, distance and plausibility are unrelated; the melodies furthest from their
references — the human variants — are the most plausible. Edit distance can rank methods on a single draft source, but it
cannot compare sources, and it cannot say whether a change was a correction (§4).

**Generated drafts leave little local trace.** We then compare three draft sources
under one model, one training recipe and one setting chosen on validation data: random
corruption, rule-based musical degradation, and spans rewritten by a trained generator.
The errors in corrupted drafts are readily found: a per-note detector ranks them at
5.8× chance, and an untrained two-sided judge — which scores each note given everything
around it — at 6–7×. Generator-written material is another matter. The best detector we
built, that same untrained judge, finds it at 2.7× chance; the trained detector finds
nothing confidently enough to act on. Generated drafts are not simply less damaged: most
are less plausible than their references, and when corrupted and generated drafts are
compared at *matched* degradation, generated changes are still localized far less well,
in every band and on both corpora. Their damage is real but diffuse — spread across notes
that each look plausible, and partly onto the unchanged notes around them — so labels
derived from the reference point at notes the draft gives little reason to suspect
(§7; Figure 1).

**A method that respects the boundary.** Where errors are local, localize-then-fix
works, provided copying is the default. A plain sequence-to-sequence refiner is 25%
*worse* than simply returning the draft: it rewrites notes that were right. We instead
gate each note — it is copied unless the model is confident it needs changing, the reject
option of selective classification [Chow1970, Geifman2017] applied per note. At the
strictest threshold the output is the draft itself, so the method's floor is the draft
by construction. On random corruption, gated editing improves on the draft by 6.1%
in edit distance on every one of five seeds, and it is the only system we test that
improves these drafts in both edit distance *and* judged plausibility (§6). On generated
drafts it declines to edit. That forgoes nothing any gate we tried could recover —
every alternative gate makes these drafts worse (§7, §8) — and it is safe where free
decoding does harm.

In summary, we make three contributions:

1. **A measurement.** A plausibility judge calibrated on human variants, and evidence
   that edit distance to a reference — the field's standard metric — cannot distinguish
   an error from a valid alternative, and should be reported alongside plausibility.
2. **A boundary.** A controlled comparison of random, rule-based and generated drafts
   showing that localize-then-fix applies where errors leave a local trace and not where
   a generator wrote the draft, with evidence from trained and untrained detectors,
   transfer between sources, and a same-corpus replication. Transfer also shows that
   about a quarter of a synthetic benchmark's in-domain score is specific to the process
   that generated its errors.
3. **A method.** Precision-gated editing, whose downside is bounded by construction,
   which improves drafts where errors are local and abstains where they are not.

Our experiments use monophonic folk melodies from the Meertens Tune Collections [MTC]
and the Essen Folksong Collection [Essen]. Plausibility is judged by a language model,
not by listeners; we treat this as the central limitation and return to it in §8.

**Figure 1** (`figures/figure1.pdf`, `scripts/figure1.py`). Draft conditions placed by
how far their changes move a melody off the data distribution: the judge's median
implausibility per unit of change (Table 2); the dashed line marks human variants of the
same tune. (a) How well an untrained two-sided judge localizes the changed notes (AUROC;
0.5 is chance), for random (D1) and rule-based (D3) corruption and for six generator
settings — sampling temperature 0.8/1.0/1.2 with the trained generator, and checkpoints
after 2, 5 and 10 epochs — on MTC and, same-corpus, Essen. D2 is the trained generator at
T = 1.0. (b) Gated editing's improvement in Correction Cost over returning the draft
unchanged, for the three headline sources (Table 3; MTC; error bars ±1 s.d. over five
seeds, mostly smaller than the markers). Localizability of generator-written changes
rises steeply with their implausibility, and drafts from an undertrained generator are
as localizable as corruption; the trained generator's drafts sit at the bottom of the
curve, where the gate abstains.

## 2. Related Work

*Budget 0.6 page; this draft is ~680 words, so expect to trim ~15%. Trim order: the
edit-model list in ¶2 to two citations, then the evaluation sentence in ¶4.*

**Symbolic music correction.** The closest precedent is the MIDI Degradation Toolkit
(MDTK) [McLeod2020], which applies rule-defined degradations to symbolic excerpts and
poses four tasks of increasing difficulty — detecting, classifying, locating and
correcting the error — aimed at post-processing automatic transcription. Our random
(D1) and music-theoretic (D3) draft sources belong to this family. What we add is a
measurement of how much a detector trained on such data learns the corruption process
itself rather than musical error: under transfer between sources, about a quarter of the
in-domain gain does not carry over, for random and rule-based corruption alike (§7). RefinPaint [Ramoneda2024] is closest
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
drawn moves $\tau$ by about 0.02.) Because the variant pairs come from the test corpus,
$\tau$ is used only to band drafts for analysis, never to train or select a model.

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
change — **two orders of magnitude more per unit than a human variant** (whose median
is near zero, so the exact ratio is unstable). Corruption moves a melody a
short distance *off* the distribution of plausible melodies; human variation moves it a
long way *along* it. Generated drafts sit between the two, at about 1 nat per unit:
most (79.5%) are less plausible than their reference, but a fifth are not, and few clear
$\tau$. That last figure should not be over-read. $\tau$ is calibrated on variants that
differ in 92% of their notes, and ΔNLL is a per-token mean, so it is a conservative bar
for drafts that change a tenth as much: only 13.0% of D1 drafts clear it either. We use
$\tau$ to certify degradation, never to declare a draft error-free.

**Edit distance cannot tell an error from an alternative.** Within a single corruption
process, more change does mean more damage: fraction changed and ΔNLL correlate at
ρ ≈ 0.5 for D1 and D3, partly by construction, since the number of injected edits drives
both. Across sources the relation disappears and slightly reverses (ρ = −0.06 pooled): the
melodies furthest from their references, the human variants, are the most plausible. The contrast is starkest at
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

**How much is left.** An oracle — a model of the same architecture, trained and tested
with the true edit positions given as input — removes 49.3% of the draft's Correction
Cost on D1 (a single run, under an earlier configuration). Gated editing's 6.1% is about
an eighth of that headroom. The gap lies mainly in localization: given the right
positions, a decoder of this size removes about half of the error.

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
Correction Cost alone would hide that gated editing improves the music. On random
corruption, gated editing is the only system that improves on the draft on both; on MTC
D3, where every system beats copying, free decoding does too.

**D3 is easier, but its in-domain gain overstates it.** Gated editing's largest gains are
on rule-based degradation — 9.81% on MTC and 7.38% same-corpus, every seed — and D3 is
the only source on which free decoding also beats copying on MTC. Part of this is
intrinsic: a refiner trained only on random corruption improves D3 drafts by more than
it improves the drafts it was trained on. Part is not: about a quarter of the in-domain
gain disappears under transfer, as it does for D1 (§7).

**Generated drafts.** On D2 the gate never opens: the output equals the draft on every
seed, and gated editing neither helps nor harms. Free decoding, by contrast, loses 3.6%
and 27.1%. §7 shows why abstaining is the right response.

## 7. Results II: where it does not

*Budget 1.3 pages; this draft is ~930 words plus Tables 5–6. Carries C3, C4, C8, C10,
C13, C14. Trained heads are 5 seeds; judges are one trained model each; transfer and
strata are seeds 0–2.*

On generated drafts the gate never opens. This section asks why: whether the generated
material can be found at all, whether a better detector would find it, and what
distinguishes the drafts that can be corrected from those that cannot.

**Table 5** *(proposed)*. Localization of the notes that differ from the reference,
MTC (Essen in brackets). Each cell gives AP lift — average precision divided by the base
rate of edited positions, so 1.0× is chance — and AUROC. The judges are never trained on
edit labels.

| detector | D1 random (6.0% edited) | D3 rule-based (8.2%) | D2 generated (7.3%) |
|---|---|---|---|
| trained head | 5.8× / .724 (5.6× / .727) | 4.6× / .726 (4.3× / .717) | 1.6× / .636 (1.8× / .658) |
| causal judge, untrained | 4.1× / .775 (4.9× / .774) | 4.7× / .768 (5.1× / .756) | 1.4× / .561 (1.4× / .564) |
| two-sided judge, untrained | **6.2× / .838** (7.0× / .831) | **5.9× / .812** (6.1× / .790) | **2.7× / .688** (2.7× / .683) |

**Generated drafts leave little local trace.** Corrupted notes are readily found: the
trained head ranks them at 4.6–5.8× chance (Table 5). Generator-written notes are not.
The head trained on D2 localizes them at 1.6× chance and, at the selected setting, never
becomes confident enough to open the gate — although D2 drafts contain *more* changed
positions than D1 drafts do (7.3% against 6.0%). The same loss, more positive examples,
and the head finds nothing worth acting on. Other detectors do no better: a span-level
detector trained to find the rewritten span directly reaches 1.6× chance at top-1 even
when told the span's length range, and heads trained on D1 or D3 localize D2's changes at
1.3–1.4×.

**It is not for want of a detector.** The strongest localizer we built was not trained
on edits at all. A two-sided judge — a masked language model of the same size and
training data as $J$, which scores each note given every other note in the melody —
outperforms the trained head on every source and both corpora. The comparison favours
the head: standardizing its scores within each melody, as the judge's are, lowers its
AUROC. Even this judge reaches only 2.7× chance on D2, against 6–7× on corruption. Two
further controls bear on this, less directly. On random corruption, localization
precision is flat across a 26-fold range of model sizes (1.6M to 42.7M parameters, at the
earlier τ = 0.9); we did not repeat the ablation on D2, so it shows only that capacity is
not what limits the head where localization works. And given the true positions, the
oracle model removes 21.5% of a generated draft's Correction Cost (single run):
generated spans are correctable once found. Finding them is the problem.

**Not less damage, but diffuse damage.** One explanation would make the result
uninteresting: generated drafts may simply carry less damage than corrupted ones (mean
ΔNLL 0.14 against 0.30), and damage of any kind may be findable once there is enough of
it. Banding D2 alone cannot test this, since any surprisal-based localizer finds
high-ΔNLL drafts easier almost by construction. Table 6 therefore bands *every* source at
the same absolute ΔNLL cut-points, on fresh sets of about 5,300 MTC drafts per source
and on the Essen test drafts, so that corrupted and generated drafts are compared at
matched degradation.

**Table 6** *(proposed)*. Localization at matched degradation: AUROC of the untrained
two-sided judge for the notes that differ from the reference, with drafts banded by the
causal judge's ΔNLL at the same cut-points for every source. MTC (Essen in brackets);
τ is the threshold of §4 (0.585 for MTC, 0.503 for the same-corpus judge).

| ΔNLL band | D1 random | D3 rule-based | D2 generated |
|---|---|---|---|
| ≤ 0 | .753 (.722) | .638 (.618) | .606 (.579) |
| 0 – 0.1 | .830 (.826) | .774 (.740) | .684 (.670) |
| 0.1 – 0.2 | .844 (.834) | .807 (.801) | .727 (.728) |
| 0.2 – 0.4 | .855 (.837) | .825 (.816) | .730 (.724) |
| 0.4 – τ | .845 (.831) | .838 (.831) | .748 (.738) |
| > τ | .861 (.855) | .850 (.838) | .775 (.779) |

The explanation fails. In every band and on both corpora, generated changes are
localized worse than corrupted ones — by 0.08–0.16 AUROC against D1 — although D2 drafts
change *more* notes than D1 drafts in the same band (8.8–13.1% against 3.9–9.4% on MTC).
Even D2's most-degraded band is localized below the least-degraded positive band of
random corruption. For random corruption, localization is nearly flat once a draft is
worse at all (0.83–0.86); for generated drafts it climbs slowly with damage and never
reaches that level.
The same damage, written by a generator, leaves less trace on the notes that were
changed. (The trained D2 head varies little across bands, 0.62–0.67; the gradient is
the judge's.)

Where the trace goes is visible in how generated damage is distributed. After a
rewritten span, the *unchanged* notes that follow also become less plausible under the
causal judge (+0.07 per token, against +0.22 within the span). The span's own notes are
individually plausible; what is implausible is how the span meets its context, and the
cost of that falls partly on notes the reference-derived labels call correct. A causal
model charges it to the wrong notes; a two-sided judge, which sees both sides of the
seam, doubles D2 localization, from 1.4× to 2.7×, but cannot recover what was spread
beyond the changed notes.

**What is detectable, and what is learned.** Across the three sources, the untrained
judge's localization follows how far the changes move a melody off the data
distribution: D1 (ΔNLL of 3.94 per unit of change) > D3 (3.14) > D2 (1.04), and the judge
ranks the sources in this order on both corpora — consistent with, though three points
cannot establish, manifold distance governing what is detectable in principle. Trained
heads add a source-specific component on top, which transfer isolates (seeds 0–2). A
head trained on D1 reaches +7.25% on D3 drafts, against +9.98% for a head trained on D3
itself; in the other direction, a D3 head reaches +4.80% on D1 against +6.35%. The loss is
about a quarter in both directions (27% and 24%). Rule-based corruption is therefore not
unusually circular: *every* synthetic benchmark we built rewards learning its own
corruption process by about a quarter of its in-domain score, and an in-domain number
alone cannot show how much. The component that transfers is what the refiner learned
about implausible music. Transfer also bounds the gate's safety. Heads trained on D1 or D3 and applied to D2
*do* open the gate, at a precision of 0.27 and 0.21, and make drafts worse by 1.7% and
2.0% on every seed. Abstention on generated drafts is the behaviour of a head trained on
them; it is not guaranteed for a head trained on something else.

**Contiguity costs localization, but does not explain D2.** Generated changes come in
runs of 2–6 notes, while random corruption mostly alters single notes, and contiguous
errors are harder to rank: within D3, the judge's AUROC falls from 0.94 for isolated
changes to 0.75 for runs of three or more (Essen: 0.92 to 0.72), plausibly because
corrupted neighbours stay visible as context. At matched run length, however, generated
changes are still localized worse. For runs of three or more the judge reaches 0.68 on
D2 against 0.75 on D3 (Essen: 0.68 against 0.72), and the trained head 0.64 against
0.71. In drafts where no two changed notes are adjacent, so that contiguity cannot
matter, the judge reaches 0.75 on D2 against 0.84 on D1 and 0.94 on D3 (Essen: 0.73,
0.83, 0.92).

**Confounds.** The main results are cross-corpus — the judge and generator are trained on
Essen and tested on MTC — but every result above replicates when judge, generator and
test melodies all come from Essen (the Essen columns of Table 3 and the bracketed
values of Tables 5 and 6). The selected setting is not fragile (§5), and gated editing
beats copy-input on D1 in all nine model sizes tested.

## 8. Discussion and limitations

*This draft is ~740 words (budget was 0.8 page; the length limit has been relaxed).
Carries C9, the recommendations that follow from C2–C8, and limitations L1–L6.
Judge-gate and surprisal figures are seeds 0–2, matched against the head gate on the
same seeds (+6.35% on MTC D1).*

**Where the judge belongs.** The untrained two-sided judge localizes better than any
trained head (Table 5), which suggests using it directly. How it is used matters.
*As an input feature* — the judge's per-note surprisal fed to the refiner's encoder — it
improves localization (AUROC 0.73 → 0.82 on D1) yet makes gated editing *worse*
(+6.35% → +5.14%, three of three seeds) and collapses free decoding (−17.7% → −61.0%): the
loss is in what the decoder writes, not in where it looks. *As the gate on its own*, it is
worse than the trained head (+2.8% on D1), because its scores are standardized within
each melody: it always nominates some note as the most surprising, and so cannot decline
to edit a melody that has nothing wrong with it. The head's probability is absolute, and
can. *Combined at the gate* — a two-feature logistic model of judge surprise and head
confidence, fitted once on D1 validation data — the two are complementary, and gated
editing improves by a further 1.4 to 2.1 points on every D1 and D3 cell on every seed
(MTC D1 +7.91%, D3 +11.80%; Essen D1 +6.52%, D3 +8.71%). The combination has a cost: on
generated drafts the judge opens a gate the head keeps shut, and results fall by
0.4–0.5%. We therefore keep the head alone as the method — its abstention on generated
drafts is part of the paper's evidence — and report the combination as the natural next
step for drafts known to contain local errors. Two notes of caution: the judge was
early-stopped on the melodies our validation drafts are made from, so judge-based gates
looked better on validation (+9.8%) than on test; and a variant trained without edit
labels — the judge as gate, the imitation model as decoder, with only the gate's
threshold chosen on validation — still improves on the draft
(+2.4% on D1, +6–7% on D3, every seed), which is the setting that would apply to drafts
nobody has labelled.

**What this means for refinement research.** Four recommendations follow.
*Report copy-input and fidelity.* A refiner measured only against another learned
baseline can look successful while copying its input (§6).
*Report plausibility alongside edit distance.* The two disagree across draft sources
(§4) and across systems (§6); either alone would have led us to a wrong conclusion.
*Test synthetic benchmarks by transfer.* A detector trained on synthetic errors partly
learns the process that made them — here about a quarter of its in-domain gain, whether
that process was random or rule-based. Training on one corruption process and testing on
another measures how much (§7).
*Do not expect localize-then-fix to carry over to generated drafts.* Generated drafts
are worse than their references, but not at the notes that differ from them: the damage
is diffuse, and labels derived from the reference point at notes the draft gives little
reason to suspect, even at matched degradation (Table 6). Refining generated music likely needs either a
formulation that does not start from per-note error labels — holistic revision scored
by plausibility, for example — or error locations annotated by people. The same hazard
applies wherever edit supervision is derived from a reference, in text and code as well
as music.

**Limitations.**
*Plausibility is a model's judgement.* ΔNLL is statistical implausibility under a small
language model, calibrated against human variants but not validated by listeners; it
also rewards typical continuations, as free decoding's outputs show (§6). Every
plausibility claim here is a claim about this judge.
*The judge and the refiners share training data.* The judge is kept independent of the
generator, but the refiners are trained on drafts of the judge's half of Essen, so the
judge is predisposed to find their output plausible. This mostly flatters free decoding,
which rewrites many notes toward what both models learned. Gated editing changes few
notes and is less exposed, but its plausibility gains may be inflated too, to a smaller
degree.
*Two corpora, one texture.* Both collections are monophonic European folk song. Whether
the boundary falls in the same place for polyphony, other styles, or larger generators
is open.
*The negative result is bounded by the detectors tried.* We tested trained per-note and
per-span detectors, transfer from other sources, and two untrained judges, the strongest
of which we could build without labels. A stronger detector may exist. Table 6 bounds
what it could gain from damage alone: at matched degradation the same judge localizes
corruption far better, so the shortfall is in where the damage lies, not how much of it
there is.
*The gate only substitutes.* Gated decoding replaces events but never inserts or deletes
them, although 30% of random-corruption operations are insertions or deletions; and it
repairs pitch far more than rhythm — metric misalignment in D1 drafts is essentially
unchanged by gating, while free decoding reduces it. Both bound what the method can
recover.
*Generated drafts are unfiltered.* We did not restrict D2 to drafts the judge certifies
as degraded. Table 6 suggests filtering would not change the conclusion — D2's most
degraded band is still localized below every positive band of random corruption — but it
would change D2's error distribution.
*Runs are not bit-reproducible* on GPU. We report five seeds for the headline and a full
second run of the main experiment, which differs by at most 0.36 points per cell.

## 9. Conclusion

*~250 words.*

Localize-then-fix is the natural way to refine a draft, and on the drafts it is usually
tested on, it works. We have argued that this is largely a property of those drafts.
Synthetic corruption, random or rule-based, damages a melody in a way that is unusual
enough to be found note by note, and a refiner that copies by default and edits only
where it is confident improves such drafts on every seed, without making the music less
plausible. Drafts written by a generator are different in kind. They are worse than
their references, but their damage is diffuse: at the same level of degradation, the
notes a generator changed are far harder to find than the notes corruption changed,
because the implausibility lies in how a rewritten passage meets its context rather than
in any note of it. On such drafts the safe behaviour for a localize-then-fix system is to
decline to act, and a gated refiner does.

Two measurement lessons carry beyond this setting. Edit distance to a reference ranks
methods on one kind of draft but cannot compare kinds, or say whether a change was a
correction; it should be reported beside a plausibility measure, and beside the score of
simply returning the draft. And a benchmark built from synthetic errors rewards
learning how they were made; a transfer test measures by how much.

Refining generated music will likely need a different starting point: a formulation
that does not assume the damage lies at the notes that differ from a reference —
revision scored by plausibility over whole passages, or error locations marked by
musicians. The judge-and-head gate of §8 is not that step: it improves corrupted drafts
further, and one variant needs no edit labels in training, but on generated drafts it
too does harm.

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
