# Paper (full text)

Rewritten section by section from the Overleaf version (itself drafted from
`paper-draft.md`; claims and numbers per `paper-core.md`). Sections not yet rewritten
here are as on Overleaf. Done: abstract, §1, §2, Figure 1 (all applied to Overleaf).

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

## Figure 1 (placed in §7)

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
in Table 2 and §7), the human-variants line, seed counts and the abstention note on the
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
