# Paper (full text)

Rewritten section by section from the Overleaf version (itself drafted from
`paper-draft.md`; claims and numbers per `paper-core.md`). Sections not yet rewritten
here are as on Overleaf. Done: abstract and §1–§5 with the method figure (applied to
Overleaf); §6 and §7 with Figure 2 (applied to Overleaf); the §7 listening study with
Table VIII and its follow-ups in the abstract, §1, §4 and §8 (applied to Overleaf
2026-10-12). Not yet on Overleaf: the full reviews of §8 and §9 (2026-10-12).

**Title:** The Better the Generator, the Less the Trace: When Does Localize-then-Fix Work
for Symbolic Music Refinement?

---

## Abstract

Refining a draft melody means changing the few notes that are wrong and keeping the
rest. It is naturally framed as *localize-then-fix*: find the faulty notes, then repair
them. Such systems are usually trained on synthetically corrupted melodies and evaluated
by their edit distance to the original. We test both halves of this recipe on monophonic
folk melodies. Using a language model calibrated on human variants of the same tune as a
plausibility judge, we show that edit distance cannot tell an error from a valid
alternative. We then compare drafts made by random corruption, by music-theoretic rules
and by a trained generator. Localize-then-fix works where changes leave a local trace,
and how much trace a generator leaves depends on how good it is: the better the
generator, the harder its changes are to find. Where a weak generator's changes are
found, the repairs make the music more plausible but rarely restore the original, so
edit distance scores them as no gain. In a blinded listening study, listeners locate
rule-based errors but not a generator's changes, and hear those repairs as different, not
wrong. Finally, we propose a precision-gated refiner that
copies each note unless it is confident the note is wrong. It improves corrupted drafts
by about 6% on every seed, and it declines to edit generated drafts, where a standard
sequence-to-sequence refiner does harm.

## 1. Introduction

Generative models of symbolic music now produce plausible melodies on demand, but a
draft is rarely final. Someone, whether a composer or a second model, has to decide
which notes to change and which to keep. This *refinement* problem differs from
generation in one important respect: most of the draft is already right, and a system
that rewrites it wholesale has not refined anything. The natural formulation is
therefore *localize-then-fix*: find the notes that are wrong, and change only those.
Recent work has pursued this with models that locate and correct degradations in
symbolic excerpts [McLeod2020], and with feedback models that pick out the weak notes of
a composition for resampling [Ramoneda2024].

The standard recipe for building such systems rests on two assumptions. Training data is
made by *corrupting* clean melodies, replacing, deleting or inserting notes, and a model
learns to undo the damage. Evaluation measures the edit distance between the model's
output and the original melody. The first assumption is that synthetic corruption
resembles the errors real drafts contain. The second is that being closer to the
original means being better. Neither has been tested directly, and both matter more as
the drafts to be refined increasingly come from generative models rather than from
hand-made corruption.

This paper asks three questions:

1. **Does edit distance to a reference measure improvement?** A melody can differ from
   its reference because it is wrong, or because it is a different but equally valid
   way to sing the tune. If edit distance cannot separate the two, it cannot tell us
   whether a refiner helped.
2. **When does localize-then-fix work?** If corrupted notes can be found but a
   generator's mistakes cannot, results on synthetic benchmarks will not carry over to
   the drafts that refinement is ultimately for. We ask what governs whether the
   changed notes in a draft can be located at all.
3. **How should a refiner behave where it cannot localize?** A system that edits
   wherever it is unsure risks damaging notes that were right. We ask whether a refiner
   can be built to improve drafts where errors are findable, and to do no harm where
   they are not.

To answer them, we work with monophonic folk melodies, where folk-song collections
supply something rare: human variants of the same tune, which serve as known-valid
alternatives. We calibrate a plausibility judge on these variants, then compare drafts
from three sources (random corruption, music-theoretic degradation, and passages
rewritten by a trained generator) under one model, one training recipe and one setting
chosen on validation data. We further vary how well-trained the generator is, which
lets us ask whether localizability depends on generation as such or on the generator's
quality. Plausibility is measured by a language model rather than by listeners, and a
model's judgement is not a musician's. We therefore check the two conclusions that rest
most on it in a blinded listening study (§7, Table VIII). Listeners hear rule-based
errors but rarely a trained generator's changes. They hear repairs that miss the
reference as no worse than the reference's own note. They also rarely hear random
corruption, so what we report about it rests on edit distance and the judge.

Our contributions are:

1. **A measurement.** A plausibility judge calibrated on human variants, and evidence
   that edit distance to a reference cannot distinguish an error from a valid
   alternative. We argue it should be reported alongside plausibility and alongside the
   score of simply returning the draft unchanged.
2. **A boundary.** A controlled comparison showing that localize-then-fix applies where
   changes leave a local trace, and that a generator's changes leave less of it the
   better the generator is. The finding holds across trained and untrained detectors,
   under controls for how much damage a draft carries, and on a second corpus. A
   transfer test also shows that part of any synthetic benchmark's score reflects the
   process that made its errors rather than musical error as such.
3. **A method and its limit.** A precision-gated refiner that copies each note by
   default and edits only where it is confident. It improves drafts where errors are
   local, abstains where they are not, and by construction can do no worse than
   returning the draft. On generated drafts, even the changes it does find are repaired
   toward the rewritten context rather than back to the reference, which edit distance
   cannot credit.

The rest of the paper is organized as follows. §2 reviews related work. §3 defines the
task, the draft sources and the evaluation. §4 introduces the plausibility judge and
addresses the first question. §5 describes the refiner and gated decoding. §6 and §7
report where localize-then-fix works and where it does not, addressing the second and
third questions. §8 discusses implications and limitations, and §9 concludes.

## 2. Related Work

**Symbolic music correction.** The closest precedent is the MIDI Degradation Toolkit
(MDTK) [McLeod2020], which applies rule-defined degradations to symbolic excerpts and
poses four tasks of increasing difficulty: detecting, classifying, locating and
correcting the error. Our random and music-theoretic draft sources belong to this
family. What we add is a measure of how much a detector trained on such data learns the
corruption process itself rather than musical error (§7). RefinPaint [Ramoneda2024] is
closest in method: a feedback model identifies the weaker elements of a composition and
guides an inpainting model to resample them. We ask the prior question of *when* such
localization is possible, and find that on generated drafts it depends on how good the
generator is. Our generated drafts are produced by span infilling, in the line of models
that condition on arbitrary subsets of a score [Huang2017, Thickstun2023]; see [Ji2023]
for a survey of symbolic generation.

**Edit-based sequence transduction and selective prediction.** When input and output
overlap heavily, predicting edits is more precise than regenerating the sequence.
Tagging models label each input token keep, delete or insert [Malmi2019,
Omelianchuk2020], and copy mechanisms let a decoder reuse its input directly [See2017].
Our gated decoder combines the two: a per-note head decides *whether* to edit, and a
decoder supplies *what* to write only where the gate opens, so copying is the default
rather than a behaviour the model must learn. Declining to edit when unsure is the reject
option of classification [Chow1970], developed for deep networks as selective
classification [Geifman2017], here applied note by note.

**Synthetic errors and reference-based evaluation.** Grammatical error detection is
framed as sequence labeling [Rei2016], and correction systems rely on synthetic errors
for pre-training [Kiyono2019]. The realism of those errors matters: errors generated by a
model trained on real learner errors improve detectors beyond the previous state of the
art, and readers struggle to tell them from real errors [Kasewa2018]. Our draft sources span the same range, from random to model-generated, and
show a consequence of realism specific to refinement: the better the generator that
wrote a draft, the less local trace its changes leave. Reference-based evaluation adds a
second problem. Single-reference scoring in grammatical error correction undercounts
valid alternatives [Bryant2015], and code-repair benchmarks judge a patch by whether
tests pass rather than by its distance to a reference diff [Jimenez2024]. We quantify the
musical analogue (§4).

**Plausibility and its limits.** Melodic expectation has long been modelled as the
information content of a note under a statistical model of a corpus [Pearce2005]. Our
judge is a neural instance of this idea, calibrated against human variants of the same
tune, and scoring each note with a masked, bidirectional model follows pseudo-log-
likelihood scoring of text [Salazar2020]. Likelihood is an imperfect proxy for quality
[Theis2016], and maximization-based decoding produces high-likelihood but degenerate
text [Holtzman2020], a behaviour we also observe in free decoding (§6). Since evaluating
symbolic generation beyond likelihood remains open [Yang2020], we report edit distance
and plausibility side by side, and check the judge against listeners (§7).

## 3. Task, Drafts and Representation

**Task.** A *draft* $M_0$ is a monophonic melody that differs from a *reference* $M^*$ in
some notes. A refiner receives $M_0$ alone and outputs a melody $\hat M$. It should change
the notes that are wrong and keep the rest. The standard score is how much closer
$\hat M$ is to $M^*$ than $M_0$ was; §4 asks whether that score measures improvement,
which is why we report it alongside plausibility.

**Corpora and splits.** References come from two collections of monophonic folk song.
The Essen Folksong Collection [Essen] (8,469 melodies, 54.5 notes on average) supplies all
training data. It is split at random into two halves, one for the plausibility judge of
§4 and one for the generator that writes D2 drafts, so that the judge never scores
material the generator could have memorized. Refiners are trained on drafts of the
judge's half (3,812 melodies, with 423 more for validation). The test set is MTC-ANN, the
annotated corpus of the Meertens Tune Collections [MTC]: 360 melodies (49.0 notes on
average) in 26 *tune families*, groups of melodies believed to descend from a common
ancestor through oral transmission. No model sees MTC-ANN in training, so every main
result is cross-corpus. A same-corpus control (20% of Essen held out as test, judge and
generator retrained on the rest) is reported alongside every result from §6 on.

**Draft sources.** Drafts are made in four ways (Table I). Each test reference yields
three drafts per source.

- **D1, random corruption.** One to five operations, drawn as pitch replacement (40%),
  duration change (30%), deletion (15%) and insertion (15%). New pitches and durations
  are drawn from the corpus's own distributions, so each edit uses material typical of
  the corpus even though where it lands is random. This is the random corruption used to
  build training data in prior work [McLeod2020], and our main control.
- **D2, generated.** A span of 2–6 consecutive notes is rewritten by a generator, a
  three-layer transformer language model trained on the other half of Essen. It samples
  at temperature 1.0 from the notes before the span only, and the reference resumes after
  it. The splice is not smoothed: a passage that fails to rejoin its continuation is a
  realistic draft flaw. Rewriting a short span changes 10.6% of a melody on average.
  Letting the generator continue freely from a prefix instead rewrites 89%, as far apart
  as two variants of a tune, which leaves no draft to refine. To vary how good the
  generator is, §7 also draws D2 drafts at temperatures 0.8 and 1.2, and from the same
  generator's checkpoints after 2, 5 and 10 epochs of training (at temperature 1.0).
  Unqualified, "D2" means the fully trained generator at temperature 1.0.
- **D3, rule-based musical degradation.** One to five applications of six rules, each a
  violation music theory names: an out-of-scale pitch given the annotated key, a leap
  beyond 14 semitones, a structurally strong note moved off the beat, a disrupted final
  cadence, a varied rhythm flattened to uniform durations, and a figure collapsed to a
  repeated pitch.
- **D4, human variants.** Pairs of melodies from the same tune family (1,012 pairs, at
  most 40 per family). Both members are attested versions of the tune, so neither is an
  error. They are used only to calibrate the judge (§4). Family members differ in metre
  and length, so they are variants of a tune, not a draft and its revision, and we do not
  refine them.

**Table I.** Draft sources on the MTC test set (three drafts per reference; D4 is the
1,012 variant pairs). Changed is Correction Cost divided by reference length, roughly the
share of notes changed.

| source | how the draft is made | mean CC | changed |
|---|---|---|---|
| D1 random | 1–5 random edits | 2.98 | 6.6% |
| D2 generated | 2–6 notes regenerated | 4.71 | 10.6% |
| D3 rule-based | 1–5 rule violations | 4.11 | 9.1% |
| D4 human variants | another version of the tune | 45.7 | 92.3% |

**Representation.** A melody is a sequence of events, each a pitch (or an explicit rest)
and a duration in integer ticks at 24 per quarter note, which represents every duration
in both corpora exactly. Each event becomes two tokens, a pitch or rest token and a
duration token, and a bar token is inserted where the annotated metre begins a new bar
(Table II), which bounds how far a duration error can shift later notes. The vocabulary
has 121 tokens: 76 pitches, 40 durations, rest, bar and three special tokens. Bars are
structure, not events: an edit never inserts or deletes a barline, and a shifted onset is
a duration change.

**Table II.** The opening of an MTC melody (4/4, starting on an upbeat) as tokens, and a
draft of it with two edits (bold). Durations are in ticks, 24 to the quarter note; `BOS`
and `EOS` are not shown. The draft's Correction Cost is 2.

| note | G4 | | G4 | A4 | B4 | B4 | | B4 | rest | D5 | D5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | P67 | BAR | P67 | P69 | P71 | P71 | BAR | P71 | REST | P74 | P74 |
| | D24 | | D24 | D24 | D36 | D12 | | D48 | D24 | D18 | D6 |
| draft | P67 | BAR | P67 | **P72** | P71 | P71 | BAR | P71 | REST | P74 | P74 |
| | D24 | | D24 | D24 | D36 | D12 | | **D24** | D24 | D18 | D6 |
| label | keep | | keep | edit | keep | keep | | edit | keep | keep | keep |

*(Reference is MTC-ANN `NLB070535_01`, tokenized by the paper's tokenizer. The draft is
illustrative: one pitch substitution, A4 → C5, and one duration substitution, a half note
shortened to a quarter. LaTeX label `tab:example`.)*

**Alignment and Correction Cost.** Two melodies are aligned with the Wagner–Fischer
algorithm [Wagner1974] over events, a simplified form of the musical edit distance of
Mongeau and Sankoff [Mongeau1990] without its fragmentation and consolidation operations.
Insertion, deletion, pitch substitution and duration substitution each cost 1, so an
event differing in both pitch and duration costs 2. *Correction Cost* (CC) is the minimal
total cost. The same alignment supplies the refiner's training labels: each draft event
is labelled *keep* or *edit* (Table II), collapsed from five types that §5 shows add
nothing, and an extra end-of-sequence slot receives labels for notes missing after the
last draft event.

**Evaluation.** Our primary measure is the relative reduction in mean CC over returning
the draft unchanged (*copy-input*, B3):
$(\overline{\mathrm{CC}}(M_0, M^*) - \overline{\mathrm{CC}}(\hat M, M^*)) \,/\,
\overline{\mathrm{CC}}(M_0, M^*)$, so that positive values mean the refiner improved on
doing nothing. We also report *fidelity*, the CC between output and draft, which
distinguishes a refiner that edits from one that regenerates; per-melody win and loss
rates; and plausibility under the judge of §4. Results are means over five training
seeds for the headline and three for secondary comparisons, under one configuration
selected on validation data (§5).

## 4. Measuring Plausibility

Edit distance says how far a draft is from its reference, not whether it is worse. This
section takes up the first question, whether edit distance to a reference measures
improvement. Answering it needs a second measure, of how plausible a melody is, that does
not depend on the reference. We use a language model as a judge and calibrate it against
melodies known to be acceptable. A model's judgement is not a listener's. §7 (Table VIII)
checks the two conclusions that rest most on it with listeners, whose naturalness ratings
follow the judge in direction but loosely.

**The judge.** $J$ is a decoder-only transformer language model over the token sequence
of §3 (four layers, width 256), trained on the judge's half of Essen with early stopping
on held-out Essen melodies. It shares no training data with the generator that writes D2
drafts, which is trained on the other half and has three layers rather than four; a judge
trained on the generator's data could simply recognize its output. $J$ scores a melody by
its mean negative log-likelihood (NLL) per token, per token because insertions and
deletions change length, and scores a draft by how much less plausible it is than its
reference:

$$\Delta\mathrm{NLL}(M_0) = \mathrm{NLL}_J(M_0) - \mathrm{NLL}_J(M^*). \qquad (1)$$

Positive values mean the draft is less plausible than its reference. Both melodies are
given barlines from the reference's metre, so neither benefits from annotation the other
lacks. $J$ fits MTC 0.137 nats per token worse than Essen. Because ΔNLL is a paired
difference this offset cancels, but the gap may still make $J$ more sensitive to change on
MTC than on its own corpus; the same-corpus control of §3 removes the gap, and every
judge-based result from §6 on holds under it. $J$ scores whole melodies; to score
individual notes, §7 adds a two-sided judge that sees the notes on both sides.

**Calibration on human variants.** A judge that penalized any departure from the
reference would call every alternative an error, and would be useless for our question.
We therefore score the D4 pairs, two members of one tune family, each an attested version
of the tune. Their ΔNLL has mean +0.054 and median +0.031, and 45.0% of pairs score at or
below zero: the judge is close to indifferent between versions that differ in 92% of their
notes. It measures degradation, not difference. We set a degradation threshold $\tau$ at
the 95th percentile of the variants' ΔNLL, $\tau = 0.585$, so a draft above $\tau$ is less
plausible than 95% of human alternatives are. (Resampling which variant pairs are drawn
moves $\tau$ by about 0.02.) Because the variant pairs come from the test corpus, $\tau$ is
used only to group drafts for analysis, never to train or select a model.

**Table III.** Draft sources under the judge (MTC). *Per unit* is the median of ΔNLL
divided by the fraction of notes changed (Table I); $\rho$ is Spearman's correlation
between fraction changed and ΔNLL across drafts, with a 95% interval from resampling tune
families.

| source | ΔNLL mean | per unit | ΔNLL ≤ 0 | ΔNLL > τ | ρ [95% CI] |
|---|---|---|---|---|---|
| D4 human variants | +0.054 | +0.04 | 45.0% | 5.0% | +0.14 [+0.04, +0.24] |
| D1 random | +0.298 | +3.94 | 5.2% | 13.0% | +0.52 [+0.46, +0.59] |
| D3 rule-based | +0.271 | +3.14 | 4.2% | 9.6% | +0.47 [+0.41, +0.52] |
| D2 generated | +0.139 | +1.04 | 20.5% | 2.6% | +0.20 [+0.11, +0.30] |
| all pooled | | | | | −0.06 [−0.11, −0.01] |

**Corruption is damage, but of an unrealistic kind.** A common worry about random
corruption is that a replaced note may simply produce a valid alternative. Table III does
not support that worry: 95% of D1 drafts are less plausible than their references. The
problem is the opposite one. Random and rule-based corruption change 7–9% of a melody at a
ΔNLL of 3–4 per unit of change, roughly two orders of magnitude more per unit than a human
variant (whose per-unit median is near zero, so the exact ratio is unstable). Corruption
moves a melody a short distance *off* the distribution of plausible melodies; human
variation moves it a long way *along* it. Generated drafts sit between the two, at about
1 per unit, and 79.5% are less plausible than their reference. Because $\tau$ is
calibrated on variants that differ in 92% of their notes, it is a conservative bar for
drafts that change a tenth as much: only 13.0% of D1 drafts clear it, although 95% are
degraded. We use it to certify degradation, never to declare a draft error-free.

**Edit distance cannot tell an error from an alternative.** Within a single corruption
process, more change does mean more damage: fraction changed and ΔNLL correlate at
$\rho \approx 0.5$ for D1 and D3, partly by construction, since the number of injected
edits drives both. Across sources the relation disappears and slightly reverses
($\rho = -0.06$ pooled): the melodies furthest from their references, the human variants,
are the most plausible. The contrast is starkest at matched distance, though the evidence
is thin: the nine variant pairs that differ in fewer than 20% of their notes have median
ΔNLL 0.000, against +0.18 for synthetic drafts in the same range.

So the answer to the first question is: only within one draft source. Edit distance to a
reference can rank methods on one source, but it cannot compare sources, or say whether a
change was a correction or a valid alternative. We therefore report both Correction Cost,
for comparability with prior work, and ΔNLL, for whether the music became more plausible.
ΔNLL has its own bias: it rewards typical notes, so it can favour a refiner that writes
the most likely note over the reference's (§6). §8 returns to its limits.

## 5. Method

**Figure 1 (placed at the top of §5).** `figures/refiner.tex` (TikZ, `\input` from
`main.tex`; preview `figures/refiner.png`). One column wide. Five draft events (the
opening of Table II's draft) pass through the encoder to the edit head, which shows a
probability per event; four are copied straight to the output, and the one above the
threshold (C5, $p = .93$) goes to the decoder, which writes A4. Blue is the model, orange
the edited event.

![Figure 1](figures/refiner.png)

**Caption.** Gated decoding on the draft of Table II. The edit head gives each event a
probability $p$ that it needs editing. Events with $p < \theta$ are copied; only where
$p \ge \theta$ does the decoder write a replacement (here C5 becomes A4). Probabilities are
illustrative; bar tokens are always copied.

**Refiner.** The refiner (Fig. 1) is an encoder–decoder transformer (three encoder and
three decoder layers, width 256, four heads; 5.86M parameters). The encoder reads the
whole draft; the decoder writes the refined melody one token at a time, attending to the
encoded draft, and is trained with teacher forcing on the reference. An encoder–decoder
suits the task for two reasons: deciding whether a note is wrong requires seeing the notes
after it, and a draft and its reference differ in length whenever a note has been
inserted or deleted, so they cannot be aligned position by position.

**Edit head.** A linear layer on the encoder's output gives, for every draft event, a
probability $p$ that the event needs editing. It is supervised by the keep/edit labels of
§3, and the model is trained on

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + \alpha\,\mathcal{L}_{\text{edit}}, \qquad (2)$$

where $\mathcal{L}_{\text{gen}}$ is cross-entropy over the reference's tokens and
$\mathcal{L}_{\text{edit}}$ is cross-entropy over the labels. The head adds 514
parameters. A five-way head that also predicts the type of edit does no better
($+0.10 \pm 0.36$ points, seed-matched) and never predicts an insertion correctly, so we
use the binary head throughout.

**Gated decoding.** The usual way to use such a model is free decoding, in which the
decoder rewrites the whole melody. But most of a draft is already right, and every note
the decoder rewrites is another chance to get one wrong. We instead make copying the
default (Fig. 1). An event with $p < \theta$ is copied verbatim. An event with
$p \ge \theta$ is handed to the decoder, which writes a replacement event in its place,
conditioned on the encoded draft and on everything output so far. Bar tokens are always
copied. Decoding is greedy, under a mask that enforces the alternation of pitch and
duration tokens.

This construction has a useful property. With a threshold no probability can reach, the
gate never opens and the output *is* the draft: gated decoding contains copy-input as a
special case, so its worst setting is doing nothing. Lowering $\theta$ trades coverage
for risk, as in selective classification [Geifman2017], and what decides whether each
additional edit helps is the head's precision among its most confident predictions, not
its overall accuracy. In this form the gate only substitutes events; it neither inserts
nor deletes, which limits what it can repair (§8).

**Baselines.** B1 is the same encoder–decoder without the head, decoded freely: plain
sequence-to-sequence imitation. B2 keeps the head but trains with $\alpha = 0$, so it has
exactly the refiner's parameters and differs only in the loss. B3 returns the draft
unchanged (copy-input). We also report the edit-aware model decoded freely, which
separates what the loss changes from what the gate changes. B3 and fidelity are not
optional: in a preliminary sweep on generated drafts, an auxiliary edit loss beat B1 by
6.3% on every seed only because it had learned to copy the draft (0.0009 notes changed
per note), which nothing else would have shown.

**One setting, chosen on validation.** The loss weight $\alpha$, the threshold $\theta$
and the weighting of the rare edit class in $\mathcal{L}_{\text{edit}}$ interact, and
choosing them on test results would inflate every number that follows. We select them
once, on D1 validation drafts only, over four class weightings, four values of $\alpha$
and six of $\theta$ (96 settings, three seeds each), by lowest mean gated Correction Cost.
The chosen setting (no class weighting, $\alpha = 0.5$, $\theta = 0.8$) is applied
unchanged to every draft source and both corpora. The landscape is flat: every class
weighting comes within 0.2 points of the best, and 84 of the 96 settings improve on
copy-input on every seed. Models are trained with AdamW (learning rate $3 \times 10^{-4}$,
batch 32) for up to 40 epochs, with early stopping on validation loss. Runs are not
bit-reproducible on GPU; two complete re-runs differ by at most 0.36 points in any cell
mean.

## 6. Results I: Where Localize-then-Fix Works

Our main results are cross-corpus (judge and generator trained on Essen, tested on MTC),
so every experiment is repeated same-corpus, with judge, generator and test melodies all
from Essen. These are the bracketed values in every table; each conclusion below holds on
both.

**Table IV** (now single-column; replaces the full-width version). Improvement in mean
Correction Cost over returning the draft (copy-input), %, mean over five seeds; positive
is better. MTC (Essen in brackets). Gated editing improves on copy-input on all five seeds
in every D1 and D3 cell (s.d. 0.27–0.58); on D2 its gate never opens.

| system | D1 random | D3 rule-based | D2 generated |
|---|---|---|---|
| B1 imitation, free | −24.9 (−92.2) | +3.9 (−22.3) | −3.6 (−27.1) |
| B2 (α = 0), free | −18.8 (−90.1) | +5.3 (−21.8) | −5.7 (−26.1) |
| edit-aware, free | −17.8 (−92.4) | +4.0 (−22.7) | −4.2 (−25.6) |
| **edit-aware, gated** | **+6.06 (+4.25)** | **+9.81 (+7.38)** | 0.00 (0.00) |

**Gated editing improves drafts where errors are local.** On random corruption, gated
decoding reduces Correction Cost by 6.06% relative to the draft on MTC (±0.58) and by
4.25% same-corpus (±0.46), on every one of five seeds (Table IV). It improves 22.4% of
melodies and worsens 6.5%, leaving the rest untouched, and changes about 1.5% of notes, a
third as many as free decoding. Of the events the gate hands to the decoder, 82% were
genuinely wrong. The setting chosen on D1 validation transfers unchanged to the other
sources.

**The operating point is not delicate.** Sweeping the threshold on test, as a
description rather than a selection, gated editing beats copy-input on every seed at
every threshold from 0.2 to 0.95, on both corruption sources and both corpora (D1: +5.0
to +6.4% on MTC, +2.7 to +4.6% on Essen; D3: +8.9 to +10.3% and +6.5 to +7.3%). Nor does
it depend on model size: it beats copy-input on D1 on every seed in all nine
configurations we tried, from 1.6M to 42.7M parameters (at the earlier θ = 0.9), while
free decoding is worse than copying in all nine.

**Free decoding is worse than doing nothing.** On random corruption every freely decoded
model is worse than returning the draft, on every seed: the imitation baseline B1 by 25%
on MTC and by more than 90% same-corpus. It rewrites notes that were right. The
edit-aware loss does not rescue it: with the head trained (α = 0.5) or present but
untrained (B2), free decoding stays within the seed noise of B1 on every cell. The gain of
gated editing comes from making copying the default; the loss's contribution is a head
confident enough to gate with. The one cell where free decoding beats copying is MTC D3,
discussed below.

**How much is left.** An oracle, the same architecture trained and tested with the true
edit positions as input, removes 49.3% of the draft's Correction Cost on D1 (single run,
earlier configuration). Gated editing's 6.1% is about an eighth of that headroom, and the
gap lies mainly in localization.

**Better music, not only closer to the reference.** A Correction Cost gain could come
from edits the judge finds implausible. It does not. On the melodies it changes, gated
editing removes 30% of the draft's ΔNLL on D1 and 48% on D3, and makes fewer than 3% of
them less plausible (seeds 0–2; Table VII). Free decoding removes *more*,
41% on D1, while being 25% worse than copying on Correction Cost. The two measures
disagree because free decoding is greedy and drifts toward the model's most typical
continuation: 14.5% of its D1 outputs score as more plausible than the reference itself
(against 5.2% of drafts), and a third of the melodies it changes end up both more
plausible and further from the reference (one seed). This is the dissociation of §4
again, now in model outputs. An evaluation on plausibility alone would prefer free
decoding; one on Correction Cost alone would hide that gated editing improves the music.
On random corruption, gated editing is the only system better than the draft on both.

**D3 is easier, but its in-domain gain overstates it.** Gated editing's largest gains are
on rule-based degradation, 9.81% on MTC and 7.38% same-corpus, on every seed. Part of this
is intrinsic: a head trained only on random corruption improves D3 drafts by +7.25%, more
than it improves the drafts it was trained on. Part is not: a head trained on D3 itself
reaches +9.98% (seeds 0–2). Transfer loses about a quarter of the in-domain gain in both
directions (D1 to D3, 27%; D3 to D1, +4.80% against +6.35%, 24%), so rule-based corruption
is not unusually circular: *every* synthetic benchmark we built rewards learning its own
corruption process by about a quarter of its in-domain score.

**Generated drafts.** On D2 the gate never opens: the output equals the draft on every
seed, and gated editing neither helps nor harms, while free decoding loses 3.6% and
27.1%. No other threshold would have helped: lowering it to 0.2 opens the gate on 3% of
events at a precision of 0.17 and makes drafts worse on every seed (−1.0% on MTC, −1.1% on
Essen), and from 0.5 upward the gate barely opens. Abstention is learned, not guaranteed:
heads trained on D1 or D3 and applied to D2 *do* open the gate, at a precision of 0.27
and 0.21, and make drafts worse by 1.7% and 2.0% on every seed. §7 shows why no operating
point exists.

*Removed from §6: the output-quality table (old Table V). Its gated "recovered" values
are the repair table's last column (§7), and its free-decoding values (41% on D1) are in
the text above. Moved into §6 from §7: the transfer results ("What training adds"), split
between the D3 paragraph and the generated-drafts paragraph.*

## 7. Results II: Where It Does Not

On generated drafts the gate never opens. This section asks why. It rules out a weak
detector, too little damage and contiguous changes as the explanation, shows that
localizability follows how good the generator is, and then shows that localizing a change
is not enough to restore the reference. A listening study checks these conclusions
against listeners.

**Table V** (full width, unchanged). Localization of the notes that differ from the
reference, MTC (Essen in brackets). Each cell gives AP lift (average precision divided by
the base rate of edited positions, so 1.0× is chance) and AUROC. The judges are never
trained on edit labels.

| detector | D1 random (6.0% edited) | D3 rule-based (8.2%) | D2 generated (7.3%) |
|---|---|---|---|
| trained head | 5.8× / .724 (5.6× / .727) | 4.6× / .726 (4.3× / .717) | 1.6× / .636 (1.8× / .658) |
| causal judge, untrained | 4.1× / .775 (4.9× / .774) | 4.7× / .768 (5.1× / .756) | 1.4× / .561 (1.4× / .564) |
| two-sided judge, untrained | **6.2× / .838** (7.0× / .831) | **5.9× / .812** (6.1× / .790) | **2.7× / .688** (2.7× / .683) |

**A trained generator's drafts leave little local trace.** Corrupted notes are readily
found: the trained head ranks them at 4.6–5.8× chance (Table V). Generator-written notes
are not. The head trained on D2 localizes them at 1.6× chance and, at the selected
setting, never becomes confident enough to open the gate, although D2 drafts contain
*more* changed positions than D1 drafts (7.3% against 6.0%). The same loss and more
positive examples, and the head finds nothing worth acting on. Other trained detectors do
no better: a span-level detector trained to find the rewritten span directly reaches 1.6×
chance at top-1 even when told the span's length range, and heads trained on D1 or D3
localize D2's changes at 1.3–1.4×.

**It is not for want of a detector.** The strongest localizer we built was not trained on
edits at all. A two-sided judge, a masked language model of the same size and training
data as $J$ that scores each note given every other note in the melody, outperforms the
trained head on every source and both corpora. The comparison favours the head:
standardizing its scores within each melody, as the judge's are, lowers its AUROC. Even
this judge reaches only 2.7× chance on D2, against 6–7× on corruption. Nor is capacity
the limit where localization works: on random corruption, localization precision is flat
across a 26-fold range of model sizes (at the earlier θ = 0.9; we did not repeat this on
D2). And the changes are correctable once found: given the true positions, the oracle
removes 21.5% of a generated draft's Correction Cost (single run). Finding them is the
first problem.

**Not less damage, but diffuse damage.** One explanation would make the result
uninteresting: generated drafts may simply carry less damage than corrupted ones (mean
ΔNLL 0.14 against 0.30), and damage of any kind may be findable once there is enough of
it. Banding D2 alone cannot test this, since any surprisal-based localizer finds
high-ΔNLL drafts easier almost by construction. Table VI therefore bands *every* source
at the same absolute ΔNLL cut-points, on fresh sets of about 5,300 MTC drafts per source
and on the Essen test drafts, so that corrupted and generated drafts are compared at
matched degradation.

**Table VI** (one column, unchanged). Localization at matched degradation: AUROC of the
untrained two-sided judge for the notes that differ from the reference, with drafts
banded by the causal judge's ΔNLL at the same cut-points for every source. MTC (Essen in
brackets); τ is the threshold of §4 (0.585 for MTC, 0.503 for the same-corpus judge).

| ΔNLL band | D1 random | D3 rule-based | D2 generated |
|---|---|---|---|
| ≤ 0 | .753 (.722) | .638 (.618) | .606 (.579) |
| 0 – 0.1 | .830 (.826) | .774 (.740) | .684 (.670) |
| 0.1 – 0.2 | .844 (.834) | .807 (.801) | .727 (.728) |
| 0.2 – 0.4 | .855 (.837) | .825 (.816) | .730 (.724) |
| 0.4 – τ | .845 (.831) | .838 (.831) | .748 (.738) |
| > τ | .861 (.855) | .850 (.838) | .775 (.779) |

The explanation fails. In every band and on both corpora, generated changes are localized
worse than corrupted ones, by 0.08–0.16 AUROC against D1, although D2 drafts change
*more* notes than D1 drafts in the same band (8.8–13.1% against 3.9–9.4% on MTC). Even
D2's most degraded band is localized below the least degraded positive band of random
corruption. For random corruption, localization is nearly flat once a draft is worse at
all (0.83–0.86); for generated drafts it climbs slowly with damage and never reaches that
level. The same damage, written by a generator, leaves less trace on the notes that were
changed. (The trained D2 head varies little across bands, 0.62–0.67; the gradient is the
judge's.)

Where the trace goes is visible in how generated damage is distributed. After a rewritten
span, the *unchanged* notes that follow also become less plausible under the causal judge
(+0.07 per token, against +0.22 within the span). The span's own notes are individually
plausible; what is implausible is how the span meets its context, and the cost of that
falls partly on notes the reference-derived labels call correct. A causal model charges
it to the wrong notes. A two-sided judge, which sees both sides of the seam, doubles D2
localization, from 1.4× to 2.7×, but cannot recover what was spread beyond the changed
notes.

**Contiguity costs localization, but does not explain D2.** Generated changes come in
runs of 2–6 notes, while random corruption mostly alters single notes, and contiguous
errors are harder to rank: within D3, the judge's AUROC falls from 0.94 for isolated
changes to 0.75 for runs of three or more (Essen: 0.92 to 0.72). At matched run length,
however, generated changes are still localized worse. For runs of three or more the judge
reaches 0.68 on D2 against 0.75 on D3 (Essen: 0.68 against 0.72), and the trained head
0.64 against 0.71. Where no two changed notes are adjacent, the judge reaches 0.75 on D2
against 0.84 on D1 and 0.94 on D3 (Essen: 0.73, 0.83, 0.92).

**Figure 2** (one column). `figures/figure2.pdf`, from `scripts/figure2.py`. A dot plot,
one row per draft condition, MTC and Essen side by side. Rows: *Corrupted* (random D1,
rule-based D3); *Generator: training* (2, 5, 10 epochs, fully trained D2, all at
T = 1.0); *Generator: temperature* (T = 1.2, 1.0 = D2, 0.8, fully trained). x: AUROC of
the untrained two-sided judge, 0.5 = chance; the shaded band marks the corrupted drafts.
The former panel (b), gated gain per source, repeated Table IV and was dropped.

![Figure 2](figures/figure2.png)

**Caption.** How well an untrained judge, which scores each note given its context, finds
the notes that differ from the reference (AUROC; 0.5 is chance, 1 is perfect), on MTC
and, same-corpus, Essen. Drafts are made by random (D1) or rule-based (D3) corruption, or
by a generator varied in training length (at T = 1.0) and sampling temperature (fully
trained); D2 is the fully trained generator at T = 1.0. The shaded band marks the
corrupted drafts. The better trained the generator, the harder its changes are to find.

**The better the generator, the less it leaves.** Everything so far compares one
generator with corruption. Fig. 2 varies the generator itself: three sampling
temperatures, and three checkpoints saved early in its training, each scored by the
untrained judge exactly as D2 is. Localizability rises steadily as the generator gets
worse: AUROC 0.65 at temperature 0.8, 0.69 for D2 and 0.72 at temperature 1.2, then
0.77, 0.83 and 0.87 for the checkpoints after 10, 5 and 2 epochs (Essen: 0.65 to 0.86, in
the same order). This ordering is the generator's quality by its own measure: held-out
NLL falls from 1.84 after 2 epochs to 1.70, 1.50 and 1.23 for the trained model
(same-corpus generator: 1.91 to 1.32). Two cautions apply. The judge shares training data
with $J$, so part of its agreement with $J$'s measure of how implausible the changes are
is built in. A detector independent of both, the head trained on random corruption's edit
labels and applied zero-shot, shows the same direction (0.57 at temperature 0.8 rising to
0.64 for the 2-epoch generator; Essen 0.61 to 0.68), but a much shallower slope: for it,
even the weakest generator's drafts stay well below random corruption (0.73). By the
judge, the 2-epoch generator's drafts are localized *better* than random corruption
(0.87 against 0.84). What makes D2 hard to localize is therefore not that a generator
wrote it, but that a good one did; how close a weak generator comes to corruption depends
on the detector. Across all eight conditions, localizability rises steeply with how
implausible the changes are per unit of change, up to about 2 per unit, and then levels
off at 0.81–0.87, where both corruption sources and the undertrained generator sit.

**Table VII** (now one column; was full width). What the decoder writes where the gate
opens. *Exact*: share of gate-opened true substitutions where the output equals the
reference's note. *≥ ref.*: share of the remaining, mismatched repairs that the two-sided
judge rates at least as plausible as the reference's note. *Recovered*: share of the
changed melodies' ΔNLL removed. Seeds 0–2, MTC (Essen in brackets).

| drafts | exact | ≥ ref. | recovered |
|---|---|---|---|
| D1 random | **59%** (58%) | 49% (49%) | 30% (31%) |
| D3 rule-based | **44%** (41%) | 39% (40%) | 48% (53%) |
| D2, 2-epoch generator | **11%** (13%) | **64%** (68%) | 28% (36%) |

**Repaired, but not to the reference.** If localizability were the whole story, gated
editing should work on the 2-epoch generator's drafts. By edit distance it barely does.
Refiners trained on those drafts under the same setting gain −0.13% ± 0.24 on copy-input
on MTC (one seed of three positive) and +0.38% ± 0.30 same-corpus (three of three), a
tenth of what the same method gains on random corruption, while free decoding loses 11%
and 28%. Two things fall short. The trained head localizes these drafts less well than
the judge does (AUROC 0.67–0.81 across seeds and corpora, against 0.86–0.87), so the gate
opens at a precision of 0.59–0.72, against 0.82 on D1. And where the gate does find a
changed note, the decoder rarely writes the reference's note (Table VII): 11–13% of the
time, against 58–59% on random corruption and 41–44% on rule-based. That is not the same
as failing. By the judge, gated editing removes 28% (MTC) and 36% (Essen) of these drafts'
implausibility, and makes 85–88% of the melodies it changes more plausible. At the
repairs that miss the reference, the two-sided judge finds the decoder's note far more
plausible than the draft's (median surprisal 2.3–2.4 against 10.7–11.1) and at least as
plausible as the reference's own note in 64–68% of cases (median 3.3–3.9). Inside a
rewritten span the reference's note is no longer the best fit: its neighbours were
rewritten too, and the decoder writes a note that fits them. A corrupted note is pinned
down by the intact notes around it; a rewritten span's original is one plausible
continuation among many, and the draft does not say which one the reference used. The
oracle agrees: given the true positions, it removes 49% of D1's Correction Cost but only
21.5% of D2's. Edit distance to the reference, the measure §4 showed cannot tell an error
from an alternative, scores these repairs as no improvement. The judge rewards typical
notes, and may prefer them for that reason alone; whether listeners hear them as wrong
is tested next.

**Listening study.** Two of this section's conclusions rest on the judge: that a
generator's changes leave less trace than corruption, and that repairs which miss the
reference are not wrong. We checked both with listeners. Six raters took a blinded
listening test on MTC test material: 17 sessions over five balanced rounds, so each item
was rated three or four times.

The test had two tasks:
- **Excerpts.** Raters heard 100 excerpts of at most 20 notes: 20 each from D3, D1, D2,
  the 2-epoch generator, and untouched references. Each excerpt contains all of its
  draft's changes and comes from a different tune. Raters clicked the notes that
  sounded wrong, or none, and rated naturalness from 1 (clearly wrong) to 5 (natural).
- **Pairs.** Raters heard two versions of an excerpt that differ at one highlighted note
  and chose the better one or no preference. In 40 pairs, a gated repair on a 2-epoch
  draft that missed the reference was set against the reference's note in the same
  context. In 10 check pairs, the repair was set against the draft's note it replaced.

**Table VIII** (one column, new; `tab:listening`). Listening study, excerpt task. *Nat.*:
mean naturalness, 1 (clearly wrong) to 5 (natural). *None*: share of ratings that marked
nothing. *Found*: share of ratings that marked at least one changed note. *AUROC*: the
share of raters marking each note, used as a score for the changed notes, beside the
untrained two-sided judge on the same notes. 68 ratings per row, 20 excerpts.

| excerpts | nat. | none | found | AUROC listeners / judge |
|---|---|---|---|---|
| D3 rule-based | 2.56 | 21% | **78%** | .77 / .89 |
| D1 random | 3.65 | 54% | 25% | .59 / .78 |
| D2, 2-epoch generator | 3.43 | 57% | 28% | .62 / .83 |
| D2 generated | 3.66 | 62% | **19%** | .54 / .73 |
| reference | 4.06 | 82% | — | — |

**Caption.** Listening study, excerpt task (6 raters, 3–4 ratings per excerpt). The judge
column scores the same notes the listeners heard.

Listeners agree with the judge at both ends of the range (Table VIII):
- **Rule-based errors are heard.** Listeners find D3's errors in 78% of ratings,
  including 68% of its changed pitches, and rate those excerpts 1.4 points less natural
  than the references.
- **A trained generator's changes are not.** Listeners find them in 19% of ratings and
  mark 8% of its changed pitches. Their AUROC of 0.54 [0.49, 0.60] is barely above
  chance, and the judge also ranks these notes lowest.
- **The 2-epoch generator falls between,** as Fig. 2 predicts, though the intervals
  overlap.
- **The differences are in what listeners find, not how freely they click.** Marks on
  unchanged notes run at 1.5–1.9% for every row, references included.

Across all 100 excerpts, naturalness falls as the judge's ΔNLL rises (Spearman
ρ = −0.40 [−0.55, −0.21]; −0.26 among drafts alone):
- the five excerpts above τ average 1.95;
- the seven drafts the judge scores no worse than their reference average 4.04;
- the references average 4.06.

The judge is a coarse stand-in for listeners, but it points the same way.

**Random corruption is quiet to the ear.** Here listeners and the judge part. Listeners
find D1's changes in only 25% of ratings, at AUROC 0.59 against the judge's 0.78, and
rate D1 excerpts as natural as D2's. Two things contribute:
- **Rhythmic changes have no note to click.** Most of D1's changes alter durations or
  drop a note. Listeners attribute these to a note in about 5% of cases, against 68% for
  D3's pitch changes.
- **The D1 excerpts are mild.** Requiring an excerpt to contain all its changes selected
  D1 drafts with few edits (mean ΔNLL 0.16, against 0.30 over the test set).

Even D1's changed pitches are found only 15% of the time; they are drawn from the
corpus's own pitch distribution and are often in key. The trace that makes random
corruption easy for a detector is therefore not one listeners readily hear. Our claims
about random corruption concern edit distance and the judge, not perception.

**Repairs are heard as different, not wrong.** Between a repair that missed the
reference and the reference's own note, listeners had no preference:
- **Repair against reference.** 42 ratings chose the repair, 40 the reference, and 54
  neither. That is 51% of decided ratings for the repair [38%, 64%], binomial p = 0.91.
- **They do not share the judge's preference.** The judge prefers the repair in 25 of
  the 40 pairs; listeners agree with it 55% of the time.
- **Against the draft's note, the repair always wins:** 30 ratings to 0, with 4 no
  preference. So the notes the gate replaces are heard as worse, and what it writes is
  heard as no worse than the reference: different, not wrong, and not better.

Leaving out any one rater moves the found rates by at most 7 points and keeps D2's
AUROC at 0.53–0.54. It also keeps the repair-against-reference split between 49% and 53%.
Agreement between raters is moderate (Krippendorff's α = 0.45 for naturalness).

**Using the judge.** The untrained two-sided judge localizes better than any trained head
(Table V), and how it is used matters. Fed to the refiner's encoder as a per-note input,
it improves localization (AUROC 0.73 to 0.82 on MTC D1) yet makes gated editing worse
(+6.35% to +5.14%, every seed): the loss is in what the decoder writes, not in where it
looks. Alone at the gate it is worse than the head (+2.81% on MTC D1), because its scores
are standardized within each melody and it cannot decline to edit a melody with nothing
wrong. Combined with the head in a two-feature gate fitted on D1 validation data, it adds
1.4 to 2.1 points on every D1 and D3 cell on every seed (+7.91% on MTC D1, +11.80% on D3),
at a cost of 0.4–0.5% on generated drafts, where the judge opens a gate the head keeps
shut. We therefore keep the head alone as the method, because its abstention on generated
drafts is part of the evidence, and offer the combination for drafts known to contain
local errors.

*Removed from §7: the gate-variants table (old Table IX; its numbers are in "Using the
judge"), the transfer paragraph "What training adds" (moved to §6), and Figure 2's
panel (b). Added: the listening study and Table VIII. The §8 paragraph "Where the judge belongs" repeated "Using the judge" and was
folded into it. Table numbers after §5: IV headline, V localization, VI matched
degradation, VII repair, VIII listening.*

## 8. Discussion and Limitations

*Reviewed 2026-10-12. Changes from Overleaf:*
- *Recommendations.* There are now five. The repair sentence becomes "Score repairs by
  plausibility". The generator recommendation now says generated damage is diffuse,
  instead of "not at the notes that differ", which read as the retired "not locally
  wrong". "Will not transfer" is softened to "should not be assumed to hold". The
  transfer item now names a refiner's gain rather than a detector's.
- *Beyond localize-then-fix.* A new paragraph adds that a refiner should be able to
  decline, but only in-distribution, and says which hazard is meant.
- *What a local trace is.* A new paragraph separates what a detector finds from what
  listeners hear.
- *Limitations.*
  - "The listening study is small" is new: it covers L8 and must-not-claim #15.
  - The detector-bound limitation no longer says Table VI "bounds" the gain.
  - Limitations are reordered with perception first.

**What this means for refinement research.** Five recommendations follow:
- *Compare against copying the draft, and report fidelity.* A refiner measured only
  against another learned baseline can look successful while doing worse than leaving
  the draft alone: our imitation baseline is 25% worse than copy-input (§6).
- *Report plausibility alongside edit distance.* The two disagree across draft sources
  (§4) and across systems (§6).
- *Score repairs by plausibility, not exact match.* Repairs that miss the reference are
  mostly as plausible as it by the judge (Table VII), and listeners hear them as no worse
  (Table VIII). Edit distance scores them as no gain.
- *Test synthetic benchmarks by transfer.* A refiner trained on random corruption and
  tested on rule-based degradation, or the reverse, loses about a quarter of its
  in-domain gain (§6). That share comes from learning the process that made the errors,
  not from refining.
- *Report the generator's quality.* A good generator's drafts are worse than their
  references, but the damage is diffuse rather than concentrated at the notes that
  differ, even at matched degradation (Table VI). The weaker the generator, the more
  trace it leaves (Fig. 2). Results on a weak generator's drafts should not be assumed
  to hold for a stronger one.

**Beyond localize-then-fix.** Refining the output of good generators likely needs a
formulation that does not start from per-note error labels: holistic revision scored by
plausibility, for example, or error locations annotated by people. The hazard is general.
Wherever edit supervision is derived from a reference that the draft does not determine,
the labels mark differences, not errors. Where per-note editing is used, the refiner
should be able to decline. Gated editing abstains on generated drafts, where free
decoding does harm. But abstention is learned, not guaranteed: heads trained on
corruption open the gate on generated drafts and make them worse (§6).

**What a local trace is.** In this paper a local trace is one a detector can find.
Listeners find a narrower set (Table VIII):
- they hear most of the rule-based pitch errors (68%), which break the key or the contour;
- they hear few changed durations from any source (3–12%);
- they hear few of random corruption's new pitches (15%), which are drawn from the
  corpus's own distribution and are often in key.

Over the full test sets, a detector finds random corruption at least as readily as
rule-based errors (Table V).
On the contrast the thesis rests on, the two agree: the judge and listeners both locate a
trained generator's changes worst of all sources. Whether a trace that a detector finds
but listeners miss is worth repairing depends on what refinement is for, and edit
distance cannot settle that either.

**Limitations.**

*Plausibility is a model's judgement.* ΔNLL is statistical implausibility under a small
language model. It is calibrated against human variants and checked against six
listeners (Table VIII). Listeners follow it in direction (ρ = −0.40 between naturalness
and ΔNLL) and at the extremes: they hear rule-based errors and not a trained generator's
changes. But they follow it loosely, and part from it in two places:
- *Random corruption.* Listeners rarely hear it. Our results on random corruption are
  claims about edit distance and the judge, not about what listeners hear.
- *Repairs.* The judge rewards typical continuations, as free decoding's outputs show
  (§6), and prefers the decoder's repairs to the reference's note. Listeners find the
  repairs no worse, not better.

The judge remains the primary measure: a study of six raters and 20 excerpts per source
supports it but cannot replace it.

*The listening study is small.* Each excerpt received three or four ratings. That
separates rule-based errors from a generator's changes, and repairs from the draft's
notes. It does not resolve every comparison:
- A generated draft's naturalness gap to the references (−0.38 points [−0.79, +0.01]) is
  as large as random corruption's. It is no evidence that generated drafts sound as good
  as their references.
- Clicking a note favours pitch errors over rhythmic ones.
- Requiring each excerpt to contain all of its draft's changes made the
  random-corruption excerpts mild (§7).

*The judge and the refiners share training data.* The judge is kept independent of the
generator, but the refiners are trained on drafts of the judge's half of Essen, so the
judge is predisposed to find their output plausible. This mostly flatters free decoding,
which rewrites many notes toward what both models learned. Gated editing changes few
notes and is less exposed, but its plausibility gains may be inflated too, to a smaller
degree.

*One generator, degraded.* The generator-quality gradient is measured by weakening one
generator (earlier checkpoints, other temperatures), not by comparing independently built
generators of increasing strength. That generators stronger than ours leave still less
trace is an extrapolation along the curve, not a measurement.

*The negative result is bounded by the detectors tried.* We tested:
- trained per-note and per-span detectors;
- transfer from other sources;
- two untrained judges, the stronger of which is the best localizer we could build
  without labels.

A stronger detector may exist. Table VI shows what it would have to overcome: at matched
degradation the same judge localizes corruption far better. So the shortfall lies in
where the damage is, not in how much of it there is.

*Generated drafts are unfiltered.* We did not restrict D2 to drafts the judge certifies
as degraded. Table VI suggests filtering would not change the conclusion, since D2's most
degraded band is still localized below every positive band of random corruption. It
would, however, change D2's error distribution.

*The gate only substitutes.* Gated decoding replaces events but never inserts or deletes
them, although 30% of random-corruption operations are insertions or deletions. It also
repairs pitch far more than rhythm: metric misalignment in D1 drafts is essentially
unchanged by gating, while free decoding reduces it. Both bound what the method can
recover.

*Two corpora, one texture.* Both collections are monophonic European folk song. Whether
the boundary falls in the same place for polyphony, other styles, or larger generators is
open.

*Runs are not bit-reproducible* on GPU. We report five seeds for the headline and a full
second run of the main experiment, which differs by at most 0.36 points per cell.

## 9. Conclusion

*Reviewed 2026-10-12. Changes from Overleaf:*
- *"Because the implausibility lies in how a rewritten passage meets its context rather
  than in any note of it"* contradicted D2's 2.7× localization and C4's own evidence.
  It now reads: much of the implausibility lies at the seam, partly on notes that were
  never changed.
- *"How much trace remains is a measure of how good the generator is"* overstated a
  trend measured by weakening one generator. It is replaced by the title's claim, "the
  better the generator, the less the trace". "Far clearer traces" becomes "clearer".
- *Listeners* are added to the repair sentence. They prefer the repairs to the draft's
  notes and hear them as no worse than the reference's. A third measurement lesson is
  added: a trace a detector finds need not be audible.
- *"A gated refiner does"* now says "trained on such drafts", since abstention is
  learned, not guaranteed (§6).
- *"One variant needs no edit labels in training"* is cut. That variant left the paper
  with the gate-variants table. "It too does harm" now says by how little.

Localize-then-fix is the natural way to refine a draft, and on the drafts it is usually
tested on, it works. We have argued that this is largely a property of those drafts.
Synthetic corruption, random or rule-based, damages a melody in a way that is unusual
enough to be found note by note. A refiner that copies by default and edits only where it
is confident improves such drafts on every seed, and makes the music more plausible, not
less.

Drafts written by a good generator are different in kind. They are worse than their
references, but their damage is diffuse. At the same level of degradation, the notes
such a generator changed are far harder to find than the notes corruption changed. Much
of the implausibility lies where a rewritten passage meets its context, partly on notes
that were never changed. The same generator early in training leaves clearer traces:
the better the generator, the less the trace.

Where a weak generator's changes are found, the refiner repairs them toward the
rewritten context rather than back to the reference. Listeners prefer these repairs to
the draft's notes and hear them as no worse than the reference's own, yet edit distance
scores them as no gain. On a good generator's drafts the safe behaviour for a
localize-then-fix system is to decline to act, and a gated refiner trained on such
drafts does.

Three measurement lessons carry beyond this setting:
- **Edit distance needs company.** Edit distance to a reference ranks methods on one
  kind of draft but cannot compare kinds, or say whether a change was a correction.
  Report it beside a plausibility measure and the score of returning the draft.
- **Synthetic benchmarks teach their own construction.** A benchmark built from
  synthetic errors rewards learning how they were made; a transfer test measures by how
  much.
- **What a detector finds need not be heard.** Random corruption is easy to find and
  hard to hear, so results on it are claims about edit distance and the judge, not about
  perception.

Refining the output of good generators will likely need a different starting point: a
formulation that does not assume the damage lies at the notes that differ from a
reference. Revision scored by plausibility over whole passages is one option. Error
locations marked by musicians are another. Combining the judge with the head at the gate
(§7) is not that step: it improves corrupted drafts further, but on generated drafts it
makes them slightly worse.

---

*Notes for the next pass:*

- *Listening study: §7 written from the results (2026-10-12; `paper-core.md` C15,
  `results/listening/report.json`). The working assumption held except on random
  corruption, which listeners rarely hear.*
  - *Done: the introduction, §4 and §8 are aligned with the study, and their red TODOs
    and §7.x pointers now point to Table VIII (`tab:listening`).*
  - *Abstract: one clause added after the repair sentence ("In a blinded listening
    study, listeners locate rule-based errors but not a generator's changes, and hear
    those repairs as different, not wrong."). Not "and a small listening study agrees".*
- *§8 and §9 reviewed (2026-10-12), not yet on Overleaf. The judge's domain gap (L4)
  stays in §4 only, since the same-corpus control answers it there. Every section has
  now had one pass; next is applying §8–§9 to Overleaf and a whole-paper read for
  length and repetition (§8's recommendations and §9's lessons overlap by design).*
- *Removed from the introduction, and still reported in later sections: 92% vs 7%
  distance, two orders of magnitude, 6–7× vs 2.7× chance, the 25% loss of free decoding,
  −0.1%/+0.4%, and 11–13% vs 58–59% exact repair.*
- *The abstract keeps one number (≈6%). It can be dropped if you want none.*
- *§3 adds two `ref.bib` entries, `Wagner1974` (the alignment algorithm) and
  `Mongeau1990` (musical edit distance), and a new Table II (`tab:example`), so the
  later tables are renumbered by one. Local build: 9.64 pages, up from about 9.4.*
- *§4 on Overleaf: Table III is now single-column (the "fraction changed" column is in
  Table I). The §7.x pointer in its opening paragraph carries a red TODO, like the
  introduction's, until the listening-test subsection has a label. Local build: 9.67
  pages.*
- *§5: the method diagram becomes Figure 1 and the results figure Figure 2 (made by
  `scripts/figure2.py`; the `fig:main` label is unchanged). The gate threshold is renamed
  $\theta$ everywhere (§5, §6, §7 and the former gate table) so that $\tau$ means only the degradation
  threshold of §4. `main.tex` gains `tikz` with `arrows.meta`. Local build: 9.78 pages.*
- *§6–§7: the results tables were consolidated (old Tables V and IX dropped, headline and
  repair tables made one column, Figure 2 reduced to one panel). Local build: 8.82 pages,
  which leaves room for the listening-study paragraph and its table.*
