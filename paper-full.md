# Paper (full text)

Rewritten section by section from the Overleaf version (itself drafted from
`paper-draft.md`; claims and numbers per `paper-core.md`). Sections not yet rewritten
here are as on Overleaf. Done: abstract, §1, §2, §3, §4, §5 with the method figure, and
the results figure (all applied to Overleaf).

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
edit distance scores them as no gain. Finally, we propose a precision-gated refiner that
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
model's judgement is not a musician's. We therefore test the two conclusions that rest
most on it in a blinded listening study with three musically trained raters (§7.x): that
corrupted notes are easier to hear than a generator's, and that repairs which miss the
reference are not heard as wrong.

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
melodies known to be acceptable. A model's judgement is not a listener's, so §7.x checks
the two conclusions that rest most on it with listeners.

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

## Figure 2 (placed in §7)

`figures/figure1.pdf`, from `scripts/figure1.py` (also writes the panels separately as
`figure1a` and `figure1b`). Full text width, placed at the §7 paragraph "The better the
generator, the less it leaves", where D1–D3 are already defined.

- **(a) Can the changed notes be found?** Dot plot, one row per draft condition, MTC and
  Essen side by side. Rows: *Corrupted* (random D1, rule-based D3); *Generator:
  training* (2, 5, 10 epochs, fully trained D2, all at T = 1.0); *Generator:
  temperature* (T = 1.2, 1.0 = D2, 0.8, fully trained). x: AUROC of the untrained
  two-sided judge, 0.5 = chance. The shaded band marks the corrupted drafts.
- **(b) Does gated editing improve the draft?** Bars, MTC, five seeds: D1 +6.1%,
  D3 +9.8%, D2 0.0%.

Dropped from the earlier version: the implausibility-per-unit x-axis (its numbers stay
in the judge table of §4, now Table III, and §7), the human-variants line, seed counts and the abstention note on the
chart, and the 2-epoch generator's gain in (b) (reported in §7).

**Caption.** (a) How well an untrained judge, which scores each note given its context,
finds the notes that differ from the reference (AUROC; 0.5 is chance, 1 is perfect), on
MTC and, same-corpus, Essen. Drafts are made by random (D1) or rule-based (D3)
corruption, or by a generator varied in training length (at T = 1.0) and in sampling
temperature (fully trained); D2 is the fully trained generator at T = 1.0. The shaded
band marks the corrupted drafts. The better trained the generator, the harder its
changes are to find. (b) Gated editing's reduction in Correction Cost relative to
returning the draft unchanged, MTC, mean ± s.d. over five seeds. Every seed improves on
both corrupted sources; on generated drafts the gate never opens, so the output is the
draft.

---

*Notes for the next pass:*

- *Listening test (TODO until results are in).* The introduction sentence above holds
  whatever the outcome, because it states only what is tested. Where the results go:
  - *§7.x, a short subsection after "Repaired, but not to the reference".* Design (two
    tasks: wrong-note spotting with naturalness ratings across D1, D3, D2, the 2-epoch
    generator and untouched references; paired preference between a mismatched repair
    and the reference's note, with a repair-vs-draft attention check), then results.
  - *§8 limitation "Plausibility is a model's judgement".* Rewrite as: the judge is
    supported by a small listening study but remains the primary measure; three raters
    is too few to replace it.
  - *Working assumption for drafting: listeners agree with the judge.* They spot
    corrupted notes more often than generated ones, and do not reliably prefer the
    reference's note over a mismatched repair. If so, the abstract can add one clause:
    "and a small listening study agrees". If not, that clause is dropped, §7.x reports
    the disagreement, and the §8 limitation is strengthened rather than softened.
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
- *§5: the method diagram becomes Figure 1 and the results figure Figure 2 (still made by
  `scripts/figure1.py`; the `fig:main` label is unchanged). The gate threshold is renamed
  $\theta$ everywhere (§5, §6, §7, Table IX) so that $\tau$ means only the degradation
  threshold of §4. `main.tex` gains `tikz` with `arrows.meta`. Local build: 9.78 pages,
  so the §6–§8 rewrites need to give back about 0.1 page to keep the 0.5 page for the
  listening study.*
