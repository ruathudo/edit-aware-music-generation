# Paper draft

Written section by section from `paper-core.md` (claims, tiers, canonical numbers). Each
section notes the claims it carries and its page budget. Numbers marked **[S8]** update
when the 5-seed headline lands.

**Working title:** Generated Drafts Leave Little Local Trace: When Does Localize-and-Fix
Work for Symbolic Music Refinement?

Status: §1–§3 drafted (2026-09-25). Other sections are placeholders.

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
by construction. On random corruption, gated editing improves on the draft by 6.35%
[S8] in edit distance, and it is the only system we test that improves on the draft in
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

**Table 1** *(proposed)*. Draft sources on MTC. Mean Correction Cost is over the test
drafts (3 per reference); fraction changed is Correction Cost divided by reference length,
from the validator's scoring sample.

| source | how the draft is made | mean Correction Cost | fraction of notes changed |
|---|---|---|---|
| D1 random | 1–5 random edits | 2.98 | 6.7% |
| D2 generated | 2–6 notes rewritten by a generator | 4.71 | 10.7% |
| D3 rule-based | 1–5 music-theoretic violations | 4.11 | 9.3% |
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
three training seeds [S8] with the seed standard deviation, under one configuration
selected on validation data (§5).

## 4. Measuring plausibility

*(Pending — C1, C2. Budget 1.0 page.)*

## 5. Method

*(Pending — encoder–decoder, edit head, gated decoding, τ=1 guarantee; head-type
ablation in one sentence. Budget 0.8 page.)*

## 6. Results I: where localize-then-fix works

*(Pending — C5, C6, C7, C11, C12; D3 as a control. Budget 1.3 pages.)*

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
