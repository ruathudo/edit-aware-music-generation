# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research pipeline for training a transformer to correct/complete corrupted monophonic melodies. Melodies are randomly "corrupted" with edit operations (pitch replace, duration change, note delete/insert), and a `MelodyTransformer` is trained to reconstruct the original from the corrupted version. Two training modes are compared: a **baseline** (plain cross-entropy over the whole sequence) and an **edit-aware** variant (loss upweighted at the positions that were actually edited), controlled by an `alpha` weighting factor.

## Rebuild in progress

The paper this implements was rejected at ISMIR 2026 on execution. `edit-aware-revision-spec.md` (repo root) is the governing plan for *what to build*; **`paper-framing.md` records what to *claim*, which measurement backs each claim, and three claims the evidence does not support** (it is not "where and how"; edit-awareness does not reliably lower CC; it does not hurt on random corruption). Read both before writing anything up. **Since 2026-09-25, `paper-core.md` is the definitive claims document — thesis, tiered claims (P1/P2/P3) with evidence and sources, must-not-claim list, canonical numbers. Draft from it; `paper-framing.md` is now the revision log explaining how each claim got there.** **The spec supersedes `src/train.py`'s design** — treat `src/train.py` as the legacy pipeline being replaced piece by piece, not as the target architecture.

Agreed build order: aligner → tokenizer/representation → judge `J` → validator report → generator `G` → training sweep. The validator (spec §4) comes before any edit-aware model because it is the only component that can falsify the rebuild's premise.

Two corpus/layout decisions already settled, which the spec does not record:
- **Corpus moved to Meertens Tune Collections (primary) + Essen Folksong Collection (cross-corpus).** MTC's *tune family* annotations supply the spec's D4 human-variant set in-corpus, so D4 stopped being an external dependency.
- **Code lives under `src/`, artifacts stay at root.** The spec's proposed `data/` and `models/` (for loaders and model code) collide with the existing gitignored artifact directories of the same names.

Done so far: `src/align.py`, `src/tokenizer.py`, `src/corpus.py`, `src/model.py`, `src/data.py`, `src/judge.py`, `src/drafts.py`, `src/training.py`, `src/sampling.py`, `src/edit_model.py`, `src/refinement.py`, `src/evaluation.py`, and their tests (185, stdlib `unittest`). Nothing in `src/train.py` or `main.py` has been migrated onto them yet — see "Migration status" below.

### The validator result, and what it changes

**The spec's expected motivating finding did not replicate, and a stronger one took its place.** Spec §2 predicted that a large fraction of uniformly random perturbations would prove not to be degradations. They are: only 4.6% of D1 drafts score `ΔNLL ≤ 0`. Reported by `scripts/validator_report.py`:

| source | surface change | ΔNLL | ΔNLL per unit change | %ΔNLL≤0 | retention |
|---|---|---|---|---|---|
| D4 human variants | 0.923 | +0.054 | **+0.038** | 45.0% | 4.4% |
| D1 random | 0.067 | +0.303 | **+3.940** | 4.6% | 12.6% |
| D1 random (uniform prior) | 0.068 | +0.468 | +6.393 | 2.6% | 32.6% |
| D3 musical | 0.093 | +0.267 | **+2.968** | 3.8% | 8.3% |

The finding is that **Correction Cost and musical plausibility are close to orthogonal**. Human-authored variants of one tune sit ~92% normalized edit distance apart yet are statistically indistinguishable in plausibility (45% of pairs score *negative*). Synthetic corruption sits ~7% away and is ~100× more implausible per unit changed. Synthetic corruption moves a melody a short distance *off* the data manifold; genuine musical variation moves it a long distance *along* it.

This is a problem for the paper's own headline metric, not only for its corruption protocol, and it substantiates P5's thin-evaluation objection with evidence rather than assertion.

Consequences for the plan:
- **D2 is promoted from optional to scientifically central.** A generator's output is on-manifold by construction, so D2 is the only draft source that can produce a plausible-but-worse melody. D1 and D3 both sit off-manifold (3.94 and 2.97 per unit), so D3 is only marginally more human-like than random. This is confirmed below.
- **CC alone cannot carry the evaluation.** The musical-quality metrics in spec §6.1 are now load-bearing rather than due diligence.
- **D4 is a calibration control, not a refinement target.** Family members differ in metre and length (one pair: 3/4 versus 6/8, 64 versus 30 events; within-family normalized CC 0.833 against 1.268 across families). They are variants of a tune, not a draft and its revision, so they must not be framed as a draft-refinement evaluation set.

Caveats to state in the paper: `ΔNLL` is statistical implausibility under a small LM, not a judgement that a musician would call something an error; and `J` is trained on Essen but scores MTC, so it may be more sensitive to any deviation than a same-corpus judge would be. The paired design means the corpus offset cancels, but the sensitivity does not.

### D2, and why infilling replaced prefix continuation

`scripts/d2_report.py`, generator `G` trained on the Essen half `J` never sees (val NLL 1.228, three undertrained checkpoints kept).

Prefix continuation confirmed the manifold hypothesis — every D2 condition costs +0.02 to +0.90 ΔNLL per unit change, in D4's regime rather than D1/D3's — but changed ~89% of the melody, as much as two different settings of one tune. Spec §3.2 predicted exactly this: the edit script stops describing an edit. **Infilling** regenerates a short interior span and keeps the rest of the reference.

| source | surface change | ΔNLL | ΔNLL per unit change | retention |
|---|---|---|---|---|
| D4 human variants | 0.923 | +0.054 | +0.038 | 4.4% |
| D2 continuation T=0.8 | 0.891 | +0.021 | +0.021 | 2.5% |
| D2 continuation ep2 | 0.981 | +0.829 | +0.899 | 65.8% |
| **D2-infill T=0.8** | **0.106** | +0.129 | **+0.900** | 2.0% |
| **D2-infill ep2** | **0.112** | +0.259 | **+1.999** | 8.1% |
| D3 musical | 0.094 | +0.262 | +2.999 | 5.9% |
| D1 random | 0.070 | +0.298 | +3.697 | 13.6% |

**D2-infill is the only source that is simultaneously local enough for the edit framing and on-manifold enough to be a plausible draft**, and severity is tunable across the whole intermediate band by temperature and checkpoint. Use it as the primary training source; keep D1 and D3 as comparison conditions.

Two practical notes. The splice point is deliberately not smoothed — a passage that fails to rejoin its continuation is a realistic draft flaw. And D2-infill retention under the human-calibrated `tau` is low (2–8%), which is the training-set-shrinkage risk spec §8 flagged; mitigate by generating many drafts per reference, by using undertrained checkpoints, and by drawing references from Essen rather than only the 360 MTC melodies.

### The refinement model, and copy collapse

`src/edit_model.py` is encoder-decoder, not the legacy decoder-only stack. Spec §5.1 puts the edit head on input positions, and more fundamentally, deciding whether note *i* is an error requires seeing what follows it — a causal model reading the draft has only the prefix. The legacy design also aligned `corrupted[:, :-1]` against `clean[:, 1:]` position by position, which silently assumes draft and reference share a length; insertions and deletions break that.

Baselines: **B1** = no edit head; **B2** = head present with α=0, so architecture and parameter count are identical and only the loss differs (a tighter control than rescaling width); **B3** = copy the draft unchanged.

**The headline problem is copy collapse.** On D2-infill drafts the imitation baseline learns to emit the draft verbatim — fidelity distance 0.0067/note, CC/note 0.1102 against B3's 0.1056, i.e. marginally *worse* than doing nothing. With realistic drafts only ~3 of 48 notes need changing, so copying gets ~94% of tokens right and is a strong local optimum. The fidelity metric was added to catch a model that *ignores the draft and regenerates*; it caught the opposite failure, a model that ignores the *correction*.

Raising α does not break the collapse — it deepens it. Fidelity falls toward zero and CC converges onto B3 exactly. `lift = (draft_cost − model_cost) / draft_cost` is the fraction of the draft's error actually removed; positive means the model improved the draft.

**D2-infill drafts (realistic), MTC-ANN test, one seed:**

| α | CC | lift | fidelity/note | macro-F1 | beats B3 |
|---|---|---|---|---|---|
| 0.0 | 4.876 | −3.55% | 0.0066 | 0.082 | 1.4% |
| 0.1 | 4.781 | −1.54% | 0.0033 | 0.104 | 0.8% |
| **1.0** | **4.715** | **−0.14%** | **0.0007** | 0.096 | 0.5% |
| 3.0 | 5.935 | −26.05% | 0.0150 | 0.124 | 0.4% |
| B1 (no head) | 4.975 | −5.65% | 0.0079 | — | 1.1% |
| *B3 copy-input* | *4.709* | *0* | *0* | — | — |

**D1 drafts (random corruption), same protocol:**

| α | CC | lift | fidelity/note | macro-F1 | beats B3 |
|---|---|---|---|---|---|
| 0.0 | 3.501 | −17.46% | 0.0482 | 0.094 | 26.1% |
| **0.1** | **3.452** | **−15.81%** | 0.0481 | 0.111 | 27.4% |
| 1.0 | 3.829 | −28.46% | 0.0451 | 0.121 | 18.8% |
| 2.0 | 6.964 | −133.64% | 0.0953 | 0.130 | 10.8% |
| *B3 copy-input* | *2.981* | *0* | *0* | — | — |

#### Three seeds change one of these conclusions

The single-seed reading above understated the effect. With three seeds (`scripts/seed_aggregate.py`, mean ± std, seed-matched against the same seed's B1):

**D2-infill**, B1 CC 5.053 ± 0.244, B3 4.709:

| α | CC | vs B1 | seeds positive | vs B3 |
|---|---|---|---|---|
| 0.0 (B2) | 4.993 ± 0.120 | +0.059 ± 0.305 (+1.2%) | 1/3 — straddles 0 | −0.285 |
| 0.1 | 4.770 ± 0.033 | +0.283 ± 0.273 (+5.6%) | **3/3** | −0.061 |
| 0.5 | 4.727 ± 0.019 | +0.326 ± 0.252 (+6.4%) | **3/3** | −0.018 |
| **1.0** | **4.726 ± 0.020** | **+0.327 ± 0.252 (+6.5%)** | **3/3** | −0.017 |
| 2.0 | 4.950 ± 0.303 | +0.103 ± 0.068 (+2.0%) | **3/3** | −0.241 |
| 3.0 | 5.331 ± 0.705 | −0.278 ± 0.467 | 1/3 — straddles 0 | −0.622 |

**Edit-awareness does reliably beat the imitation baseline on realistic drafts** — 3/3 seeds positive for α ∈ [0.1, 2.0], peaking at **+6.5%**, which is close to the published 6.3%. And **B2 (head present, α=0) straddles zero at 1/3 seeds**, so the gain comes from the auxiliary *loss*, not from the head's parameters. The parameter-matched control works exactly as intended.

**D1 random corruption**, B1 CC 3.702 ± 0.512, B3 2.981:

| α | vs B1 | seeds positive |
|---|---|---|
| 0.1 | +0.182 ± 0.466 | 1/3 — straddles 0 |
| 0.5 | −0.267 ± 0.282 | 1/3 |
| 1.0 | −1.151 ± 0.855 | **0/3** |
| 2.0 | −2.436 ± 1.181 | **0/3** |

On random corruption edit-awareness **does not help and actively hurts** at α ≥ 0.5. Note α=0.1's pooled Wilcoxon reports p = 1.45×10⁻⁸ while its per-seed deltas are −0.074, **+0.719**, −0.099 — one seed produced the entire effect. That is the concrete case for why a pooled within-run p-value cannot stand in for multi-seed reproducibility.

**So the draft-source ablation answers spec §7 in the affirmative, and opposite to the copy-collapse reading alone: edit-awareness helps more when the errors are realistic.** The mechanism is visible in fidelity, which moves in opposite directions: on D2 raising α drives fidelity 0.0092 → 0.0009 (*less* editing), on D1 it drives 0.0539 → 0.0851 (*more* editing). The auxiliary loss calibrates the model's **edit propensity** to match the label distribution. On D2 the labels are almost all KEEP, so it teaches restraint, and restraint is nearly optimal. On D1 there are more true edits, so it encourages editing — and the model's edits are bad, so CC worsens.

**Edit-awareness regulates how much the model edits, not how well.** That is a defensible mechanistic claim, and it is consistent with everything else: no condition on either source beats B3, and macro-F1 stays at 0.09–0.13 throughout.

Caveat on magnitude: seed 0 is an outlier (+0.617 against +0.173 and +0.190 for seeds 1 and 2), so the *direction* is reproducible but the effect size is not well pinned by three seeds. More seeds would tighten it.

Three further things to take from the single-seed detail below:

1. **No condition, on either draft source, achieves positive lift.** The model never learns to correct. On D2 it copies; on D1 it edits (fidelity 0.048, ~2 notes per melody) but its edits are net harmful. The task is not being done, and CC read in isolation conceals that completely.
2. **Draft realism changes the failure mode, as predicted.** On D1 the model engages — it beats B3 on 27% of melodies. On D2 it disengages almost entirely — 0.5%. That is consistent with the validator's manifold result: corruption leaves off-manifold traces a model can detect, whereas an infilled span is locally plausible everywhere. Edit-awareness does not rescue either regime, and large α actively harms both.
3. **The edit head barely localizes anywhere** — macro-F1 0.08–0.14 throughout, against a localization F1 of ~0.15. "Where and how" is architecturally expressible now but not yet learned.

#### Diagnostics: the decoder is fine, the head is the bottleneck

`scripts/diagnostics.py` runs two experiments that separate "cannot find the errors" from "cannot fix them". **They reverse the architectural conclusion below** — keep that paragraph only as a record of what the sweep alone suggested.

**Oracle** — ground-truth edit labels fed in at the encoder input, then train generatively:

| source | oracle CC | B3 | lift |
|---|---|---|---|
| D1 | 1.510 | 2.981 | **+49.3%** |
| D2-infill | 3.695 | 4.709 | **+21.5%** |

Given perfect localization the model removes half the draft's error on D1 and a fifth on D2, beating the copy-input baseline that nothing in the entire sweep could beat. **So the generative encoder-decoder formulation is not the problem, and it should not be replaced.**

**Probe** — edit head trained on the edit loss alone: macro-F1 0.153 (D1), 0.141 (D2), against 0.12/0.11 jointly. Barely moves, so multi-task tension is not the problem either. The head simply cannot localize.

**And the cause is partly self-inflicted.** Inverse-frequency weighting — chosen citing spec §5.1 without checking the ratio it produced — pays the head **105× more** for flagging an edit than for leaving a note alone:

| class | precision | recall | weight |
|---|---|---|---|
| KEEP | 0.966 | 0.292 | 0.016 |
| INSERT_BEFORE | 0.010 | 0.560 | 1.725 |

The head learned to flag nearly everything: accuracy 0.29. `src/edit_model.py` now offers `none`/`sqrt`/`capped`/`inverse` weighting plus `focal_edit_loss`, with `scripts/weighting_ablation.py` to choose among them. The trade-off is two-sided — light weighting collapses to predicting KEEP everywhere (localization F1 0.000), heavy weighting flags everything.

**Consequence for the plan:** keep the architecture, fix the head. The oracle also supplies something the paper needed — a principled **upper bound**. "Perfect localization yields +49% lift; our head recovers X% of it" is a far stronger framing than a bare CC comparison, and it converts the weak macro-F1 from an embarrassment into a quantified gap.

The paragraph below predates these diagnostics and is retained for the record:

The most likely cause is the architecture, not the idea. Free-form seq2seq regeneration is a poor fit for sparse-edit refinement: the model must learn to copy ~94% of tokens *and* make a few targeted changes, and copying alone is a strong local optimum. Established editing models use explicit copy/pointer mechanisms or decode edit operations directly rather than regenerating the sequence. Since the aligner already produces typed operations and the head already predicts them, **predicting and applying the edit script** is the natural next formulation and a far better-posed problem.

Caveats: one seed, ~20 epochs, 5.9M parameters. These numbers show the task is not being learned in this setup; they do not establish that edit-awareness cannot help in a setup where the base task is learnable.

### Gated decoding: the first thing to beat copy-input

The diagnostics identified localization **precision**, not F1, as the binding constraint. At D1's precision of 0.678, a third of flagged positions are notes that were already correct, so each additional edit is a losing bet. That is why judge surprisal improved F1 (recall 0.245 → 0.314) while making CC *worse* (3.371 → 4.624), and why free-form decoding always settled on copying.

`generate_corrections_gated` turns the head into a gate: each input event is copied verbatim or handed to the decoder, and only when confidence clears a threshold. **At threshold 1.0 the output is the draft verbatim, so this reproduces B3 exactly and can only improve on it** — B3 stops being an unbeaten baseline and becomes a special case.

**D1, α=0.5, threshold 0.9, three seeds:**

| | CC |
|---|---|
| B1 imitation baseline | 3.793 |
| free decoding (α=0.5) | 3.497 |
| B3 copy-input | 2.981 |
| **gated (α=0.5, τ=0.9)** | **2.801** |
| oracle (perfect localization) | 1.510 |

**+0.180 ± 0.014 CC vs B3 = +6.04%, 3/3 seeds (5.59 / 6.00 / 6.52%).** (A pooled Wilcoxon over melodies × seeds gives p = 4.2×10⁻⁵², but that is not a reproducibility statistic — report the per-seed values.) Win 23.9%, loss 8.5%, tie 67.6% — it improves a quarter of melodies, damages under a tenth, and leaves two-thirds untouched. It recovers **12.2%** of the B3→oracle headroom.

For contrast: the published result was +6.3% over an imitation baseline that is itself **21.4% worse than simply copying the draft**. A comparable percentage measured against a far stronger reference.

**D2-infill: gating declines to act, and is right to.** At τ=0.9 it returns **exactly +0.000 ± 0.000** on all three seeds — no edit is confident enough to be worth making at precision 0.22, so it degenerates to copying. At τ=0.7 it goes slightly negative. That dissociation is the cleanest statement of the thesis yet:

- **D1** (off-manifold corruption, precision 0.684) → gating converts localization into a real 6% gain over copy-input.
- **D2** (on-manifold drafts, precision 0.137) → no threshold makes editing worthwhile, and the gate correctly refuses.

A useful control falls out too: with α=0 the head exists but receives no edit loss, and gating on its untrained predictions is catastrophic (CC 29.2 against B3's 2.98). The gain comes from the auxiliary loss, not from the gating machinery.

### Same-corpus control: the result is not a corpus-mismatch artifact

The headline numbers are cross-corpus — `J` and `G` trained on Essen, references from MTC-ANN — and `J` fits MTC **0.1365 nats/token** worse than its own corpus (1.3298 against 1.1933), a gap comparable to D2's entire ΔNLL signal. Two measurements rule the confound out.

**Span-level.** If `J` were discounting `G`'s Essen-flavoured spans, those spans would score *cheaper* than the MTC notes they replaced. They score **+0.2162 dearer**, with only 35.5% coming out cheaper. (The unchanged notes also get +0.0677 dearer — splicing shifts the context for everything downstream, so the un-smoothed discontinuity propagates.)

**Full same-corpus rerun.** `--essen-test 0.2` carves a held-out Essen slice; `J` and `G` are retrained on the reduced pool and the references come from Essen too, so the gap is zero by construction. Gated decoding, α=0.5, τ=0.9, three seeds:

| | MTC refs (cross-corpus) | Essen refs (same-corpus) |
|---|---|---|
| D1 vs B3 | +6.04% ± 0.014 | **+5.21% ± 0.020** |
| D1 seeds positive | 3/3 | **3/3** |
| D1 localization precision | 0.684 | **0.728** |
| D2 vs B3 | +0.00% ± 0.000 | **−0.03% ± 0.002** |
| D2 seeds positive | 0/3 | 1/3 |
| D2 localization precision | 0.137 | **0.137** |
| n (test drafts) | 1,080 | ~5,060 |

Both halves replicate on a test set nearly five times larger.

**Correction (2026-09-23): the dissociation is *not* stronger same-corpus.** An earlier version of this section claimed D2 precision fell 0.220 → 0.137 once the corpus gap was removed, "the theory's own prediction". The 0.220 came from `D2-infill_surp0.json` — a single-seed α=1.0 run with judge-surprisal input — not from the gated baseline. Under the matched configuration D2 precision is **0.137 ± 0.010 (MTC) and 0.137 ± 0.006 (Essen)**: identical. The control does what a control needs to — both halves replicate, so the corpus mismatch is not driving the result — but it supplies no additional evidence for the theory.

A control also falls out of the same runs: with α=0 the head receives no edit loss, and gating on its untrained predictions is catastrophic (−1005% against B3). The auxiliary loss, not the gating machinery, is what makes the gate usable.

### Head-type ablation: typing adds nothing, so the claim must narrow

Spec §7's head-type axis (none / binary / 5-class), α=0.5, τ=0.9, three seeds:

| references | head | loc P | loc R | loc F1 | gated vs B3 | seeds+ |
|---|---|---|---|---|---|---|
| MTC (n=1080) | 5-class | 0.684 | 0.244 | 0.360 | +6.04% ± 0.014 | 3/3 |
| MTC | **binary** | 0.689 | 0.232 | 0.347 | **+5.93% ± 0.010** | 3/3 |
| Essen (n=5082) | 5-class | 0.728 | 0.188 | 0.296 | +5.21% ± 0.020 | 3/3 |
| Essen | **binary** | 0.687 | 0.188 | 0.293 | **+4.80% ± 0.006** | 3/3 |

Seed-matched, 5-class minus binary: **+0.003 ± 0.011** (MTC) and **+0.012 ± 0.020** (Essen), **2/3 seeds** each way. Indistinguishable on both, on a test set of 1,080 and one of 5,082.

**The five-class head earns nothing over a binary one.** That follows directly from its per-class scores — INSERT_BEFORE at F1 **0.000**, DELETE at **0.107**, SUB_DUR at 0.194. Only pitch substitution is predicted at all, so collapsing the classes loses nothing that was working.

**Consequence for the paper: the abstract must claim "where", not "where and how".** The supported contribution is *precision-gated localization* — knowing where to edit, used as a gate, beats copy-input by ~5-6%. This resolves P3 by correcting the overclaim rather than fulfilling it, which is the outcome the evidence supports and is better made by the authors than by a reviewer.

Two things make this a better paper rather than a worse one. The recommended method gets **simpler** — a binary head matches a five-way one. And "typed edit prediction does not work at this scale, with insertions unpredictable outright" is a reusable negative finding for anyone building edit-aware music models; the oracle quantifies what typing would be worth if it did work.

### Span-level detection also fails: generated material is not localizable

The global-not-local finding prescribed span-level detection. It was built
(`src/span_model.py`, `scripts/span_detection.py`) and it does not work.

| system | top-1 | top-5 | × chance |
|---|---|---|---|
| chance (210 candidates/draft) | 0.48% | — | 1.0× |
| judge surprisal pooled over spans, zero-shot | 0.56% | 2.44% | **1.2×** |
| trained span detector, 3 seeds | 0.75% ± 0.19% | 3.51% | **1.6×** |
| *per-position head, for reference* | — | — | *~1.5×* |

Changing granularity changed nothing, and the trained detector barely beats the untrained judge — so the encoder learned almost nothing the judge did not already supply. The span task was additionally given the true span-length range as a candidate prior, which a deployed system would not have; making it easier did not help.

**The decisive control is that the same architecture, size and recipe reaches localization precision 0.68–0.73 on D1 (0.689 MTC binary, 0.728 same-corpus 5-class).** The difference lies in the drafts, not in the method — this is not a capacity or tuning failure.

Three independent measurements now agree that generator-written material cannot be localized: the per-position head (~1.5× chance, flat across degradation bands), zero-shot pooled surprisal (1.2×), and the trained span detector (1.6×). **Localize-then-fix is the wrong formulation for model-generated drafts.** See `paper-framing.md` for what the paper claims as a result.

### Model-size ablation: the gain holds, and the gate bounds the damage

Nine configurations (2/4/8 layers × d_model 128/256/512), 1.6M–42.7M parameters, three seeds each. **Gated editing beats copy-input in 9/9 on every seed**, +3.78% to +6.99%, mean +5.85%.

| | free decoding CC | gated CC |
|---|---|---|
| range across 9 configs | 3.42 – 9.04 (2.65×) | **2.77 – 2.87 (1.03×)** |
| worse than B3 (2.981) | **9/9 configurations** | 0/9 |

Two secondary findings matter more than the headline:

- **Localization precision is flat across a 26× parameter range** — 0.655 to 0.719, correlation with log-parameters −0.58. Localization is *information*-limited, not capacity-limited. This is a third independent argument that the D2 failure is not about model size, alongside the D1/D2 same-architecture control.
- **The gate bounds the damage.** Free decoding is worse than doing nothing in every configuration; at 1.6M parameters it is three times worse. Gated stays ahead of B3 throughout. That is the τ=1 guarantee doing real work, and the graceful degradation may matter more for deployment than the 6%.

### D3, and the benchmark-validity spectrum

D3 (rule-based musical degradation) is the **best-performing source and a negative control**, not a result: loc precision 0.776, gated +9.69% (per seed 9.07 / 10.33 / 9.66%; win 36.9% / loss 4.5%) at τ=0.9. An earlier version quoted +9.85% ± 0.040, win 38.6% / loss 6.0% — that is the τ=0.7 result, not comparable with D1's τ=0.9 numbers. Its edits come from the same rule family a detector can learn, so it reports the paper's best number precisely because the benchmark is circular.

| source | errors come from | ΔNLL/unit | loc precision | gated vs B3 |
|---|---|---|---|---|
| D3 rule-based | rules we wrote | 2.97 | 0.776 | +9.69% |
| D1 random | unsystematic noise | 3.94 | 0.689 | +5.93% |
| D2 generated | a real generator | 1.16 | 0.137 | +0.00% |

MTC references, α=0.5, τ=0.9, three seeds. D1/D3: binary head, `weighting=none`. **D2 differs**: 5-class head, `weighting=capped` (no binary-head D2 run exists yet). ΔNLL/unit for D2 is the T=1.0 infill row, matching the drafts the refiner actually trained on; 0.90 is the T=0.8 row.

**Localizability collapses as drafts get more realistic, and apparent method performance falls with it.** That trend needs all three points; D1/D2 alone is a contrast, not a curve.

D3 also falsified both accounts of the boundary: it is *less* off-manifold than D1 (2.97 vs 3.94) and *less* point-like (mean run 1.50 vs 1.05) yet *more* localizable. The working hypothesis is that **learnable regularity** governs — off-manifold is necessary but not sufficient, and the deviation must be systematic enough to train a detector for. The locality account is then tested directly within D3 below, and falsified.

Report D3 in the ablation, keep it out of headline claims, state the circularity in the caption. **D1 stays the headline** because random corruption is not something we designed a detector for.

### Locality does not govern: H2 falsified within D3

The D1/D2 contrast admitted two accounts, and comparing sources cannot separate them because D1 and D2 differ on both axes at once. **H1**: off-manifold damage is detectable (D1 at 3.94 ΔNLL/unit localizable, D2 at 0.90 not). **H2**: a per-position head suits isolated changes and is blind to contiguous ones (D1's edits are near-all single notes, D2 replaces spans of 2–6).

D3 separates them — uniformly off-manifold, but mixed in locality. `scripts/locality_strata.py` splits D3's own test set by the longest edited run per draft (measured in *event* space; each event carries two tokens with the same label, so a token-space run makes every single-note edit look span-like). α=0.5, τ=0.9, binary head, one seed. Longest run per draft: mean 1.58, 51.3% point-like, 17.7% span-like.

| band | n | loc P | loc R | gated gain (CC) |
|---|---|---|---|---|
| point-like (1 event) | 544 | **0.825** | 0.427 | +0.441 |
| 2 adjacent events | 262 | 0.806 | 0.251 | +0.405 |
| span-like (3+ events) | 188 | **0.805** | 0.210 | +0.356 |

**Precision is flat — +0.019 across the whole range — so H2 is falsified.** Contiguity does not cost the head its precision, which is the quantity gating actually depends on.

The decisive comparison is with D2: D3's span-like band is contiguous in exactly the way D2's infilled spans are, yet holds precision **0.805 against D2's 0.137**. Contiguity cannot be what breaks D2.

What locality *does* cost is **recall**, which halves (0.427 → 0.210): the head flags part of a long run and misses the rest. Gated gain falls with it (+0.441 → +0.356), and understates the shortfall — span-like drafts carry more damage, so the same absolute gain is a smaller fraction of it. This is the recall/precision dissociation already on record: recall responds to properties of the edit, precision does not, and precision is the binding constraint.

So **manifold distance survives where locality does not**, though D3 itself already showed manifold distance is insufficient alone (less off-manifold than D1, more localizable). Learnable regularity remains the working hypothesis. Caveats: one seed, D3 only, and the bands are sized 544/262/188.

### Validation-selected headline (2026-09-24) — supersedes the numbers above

Every headline number above inherited α, τ and the class weighting from looking at test results. `scripts/select_settings.py` now picks **one global setting on D1 validation** (runs with `--eval-splits val` never decode test): **binary head, `weighting=none`, α=0.5, τ=0.8**. The old none / 0.5 / 0.9 ranks 2nd of 96, 0.07 pp behind, so the earlier selection did not inflate anything. The landscape is flat: every weighting peaks at 6.3–6.5%, and 84/96 settings beat copy-input on all seeds.

`scripts/run_headline.sh none 0.5 0.8` applies it to every source and both corpora (`results/refiner/headline_r2/`, canonical; `headline/` is a first run without artifacts). `scripts/headline_report.py` and `scripts/localization_report.py` summarize.

| corpus | source | AP lift | AUROC | gate flags / precision | gated vs B3 |
|---|---|---|---|---|---|
| MTC | D3 | 5.0× | 0.743 | 1,968 / 0.904 | +9.98% ± 0.30 |
| MTC | D1 | 5.8× | 0.727 | 1,489 / 0.823 | **+6.35% ± 0.48** |
| MTC | D2 | 1.6× | 0.635 | **0** / — | +0.00% |
| Essen | D3 | 4.3× | 0.718 | 5,972 / 0.865 | +7.30% ± 0.32 |
| Essen | D1 | 5.8× | 0.732 | 6,366 / 0.788 | +4.47% ± 0.34 |
| Essen | D2 | 1.8× | 0.654 | **0** / — | +0.00% |

Three things changed in the story:

- **D2's head collapses to KEEP under the global setting**, even though D2 has more true edits than D1 (7.3% vs 6.0% of positions). With nothing about the method differing between sources, this replaces the old D2 precision 0.137, which came from a different head and weighting.
- **Localization is compared threshold-free** (AP ÷ base rate, AUROC), because argmax precision is undefined for a head that barely flags. D2 keeps a weak ranking signal (AUROC ~0.64) that is too weak to act on.
- **D3 is not more localizable than D1 threshold-free** — AUROC level, AP lift lower. It wins only in the confident tail (gate precision 0.90 vs 0.82). "D3 falsifies manifold distance" weakens to "evidence against manifold distance as the sole governor".

Runs are **not bit-reproducible** (cuDNN determinism unset): two full runs differ by up to 1.17 pp on a seed, 0.36 pp on a cell mean. `--save-artifacts` saves checkpoints (gitignored, ~23 MB each), per-position P(edit) (committed), and decoded outputs (gitignored, large) — later analyses should read those rather than retrain.

### Output quality: the CC gain is not bought with plausibility (2026-09-24)

`scripts/output_quality.py` scores the saved outputs with `J` (same-corpus runs with `judge_ctrl.pth`). ΔNLL is output minus reference; "recovered" is the share of the draft's ΔNLL removed. MTC:

| | CC vs copy | ΔNLL | recovered |
|---|---|---|---|
| D1 draft | 0 | +0.298 | — |
| D1 B1 free (greedy) | −24.7% | +0.176 | +40.9% |
| D1 EA gated | +6.35% | +0.250 | +16.3% overall, **+30.1% on edited melodies (2.8% made worse)** |
| D3 EA gated | +9.98% | +0.166 | +47.5% on edited melodies (1.4% worse) |
| D2 free / gated | −3.0% / 0 | +0.135 / +0.139 | +2.7% / 0 |

Essen replicates. **Gated editing is the only condition better than the draft on both axes.** Free decoding recovers more plausibility but is worse than copying on CC: it decodes greedily toward the LM's mode, 14.5% of its outputs score *more* plausible than the reference, and a third of the melodies it changes become more plausible *and* further from the reference. That is the CC/plausibility dissociation reproduced on outputs, and a warning that ΔNLL rewards mode-seeking. D2's implausibility resists even regeneration (3–6%). The gate leaves D1's metric misalignment untouched — it fixes pitch, not duration.

### Zero-shot and transfer localization (2026-09-24)

`src/masked_judge.py` is a **two-sided judge**: same size and training half as `J`, but whole events (pitch + duration) are masked and predicted from both sides, so a note is scored as −log p(note | everything else). `J` is causal and charges a note that breaks what follows to the *following* notes. Trained by `scripts/train_masked_judge.py` (`models/masked_judge.pth`, `masked_judge_ctrl.pth` for the same-corpus control; it drops melodies over 1,024 tokens, as `J`'s dataset does).

`scripts/zero_shot_localization.py` — AP lift / AUROC, same positions as the refiner head, MTC (Essen):

| | causal `J` | two-sided judge | trained head |
|---|---|---|---|
| D3 | 4.7× / 0.768 | **5.9× / 0.812** (6.1× / 0.790) | 5.0× / 0.743 |
| D1 | 4.1× / 0.775 | **6.2× / 0.838** (7.0× / 0.831) | 5.8× / 0.727 |
| D2 | 1.4× / 0.561 | **2.7× / 0.688** (2.7× / 0.683) | 1.6× / 0.635 |

An **untrained judge beats every trained head**. D2 is **not unlocalizable** — 2.7× chance — but far behind; "not locally wrong" must soften to "little local trace". The rule-free ordering D1 > D3 > D2 matches manifold distance.

`scripts/run_transfer.sh` (`--test-sources` on `train_refiner.py`): D1→D3 +7.25% (vs D3 in-domain +9.98%), D3→D1 +4.80% (vs +6.35%). So ~2.7 points of D3's gain are learned injection regularity — **D3 is partly, not wholly, circular**. D1→D2 and D3→D2 open the gate and *lose* 1.7–2.0% on all seeds: the no-harm property needs in-distribution training.

### Judge + head gating (2026-09-25)

`generate_corrections_gated(..., gate_scores=...)` lets an external per-position score drive the gate; the model then needs no edit head. `scripts/judge_gate.py select|test` compares gates from the headline checkpoints with the decoder fixed; thresholds and the combination's logistic weights come from D1 validation only. Gated vs copy-input:

| | head τ=0.8 (headline) | judge z≥3 | judge, B1 decoder | **judge + head** |
|---|---|---|---|---|
| MTC D1 | +6.35% | +2.81% | +2.43% | **+7.91%** |
| Essen D1 | +4.47% | +2.86% | +2.43% | **+6.52%** |
| MTC D3 | +9.98% | +7.96% | +7.29% | **+11.80%** |
| D2 (MTC / Essen) | 0 / 0 | −0.47 / −1.17 | −0.16 / −1.14 | −0.39 / −0.46 |

Combination beats the head by 1.4–2.1 pp, 3/3 seeds seed-matched, on every D1/D3 cell; the judge alone is *worse* than the head (it cannot abstain on a whole melody). The combination loses ~0.4% on D2, where the head abstains. Judge-based gates look optimistic on validation because the two-sided judge was early-stopped on `judge_val`; test numbers are clean.

### Thin controls re-run on 3 seeds (2026-09-25)

`scripts/control_strata.py` re-runs both strata from the headline checkpoints, threshold-free, with the two-sided judge alongside the head. All three earlier conclusions changed:

- **Degradation strata (D2)** — fresh 5,339-draft MTC set + Essen test; τ recomputed per judge with the validator's pair seed (7), giving 0.585 exactly. The head stays flat across bands (AUROC ~0.64), but the **judge climbs to AUROC 0.78 on certified-degraded drafts** (n=131 / 222). "Nothing to find" is partly true: genuinely degraded generated drafts are localizable; they are 2.5–4.4% of D2.
- **Locality strata (D3)** — AUROC falls with run length (head 0.80 → 0.71, judge 0.94 → 0.75). The earlier "flat" result read argmax precision, the confident tail, which is flat. D3's contiguous band still beats D2 (0.71 vs 0.64), so contiguity explains part of D2's gap, not all.
- **Surprisal as encoder input** (`results/refiner/controls/mtc_d1_surprisal.json`) — localization improves (AUROC 0.73 → 0.82) with unchanged gate precision, yet gated CC drops +6.35% → +5.14% on 3/3 seeds and free decoding collapses to −61%. The loss is in the decoder, not the gate. Put the judge at the gate, not in the model.

### Decoupling statistic (2026-09-25)

`scripts/decoupling.py` — per-draft Spearman ρ between surface change (CC / reference length) and ΔNLL, MTC, CI by bootstrapping tune families: D1 **+0.52**, D3 **+0.47**, D2 +0.20, D4 +0.14, **all pooled −0.06** [−0.11, −0.01]. "Near-orthogonal" was too strong: within a corruption process CC tracks damage (partly by construction — edit count drives both); across sources the relation vanishes and slightly inverts. Claim: CC can rank methods on one draft source, but cannot compare sources or tell an error from a valid alternative. Matched-distance comparison is thin (9 human pairs inside the synthetic range: median ΔNLL 0.000 vs +0.184).

### Five-seed headline (2026-09-25)

Seeds 3–4 (`SEEDS="3 4" scripts/run_headline.sh none 0.5 0.8 results/refiner/headline_s34`) merged with seeds 0–2 in `headline_s34/five_seed_summary.json`. Head-gated vs copy-input: **MTC D1 +6.06% ± 0.58**, D3 +9.81%, D2 0.00%; Essen D1 +4.25%, D3 +7.38%, D2 0.00%. Every D1/D3 cell positive on 5/5 seeds; D2 abstains on all 5; B1 −24.9%. Headline figures quote 5 seeds; comparisons with conditions run only on seeds 0–2 (transfer, judge + head gate, quality, surprisal) use the matching 3-seed values.

### Matched-degradation strata (2026-09-27) — C4 revised

`scripts/matched_strata.py` bands D1, D2 and D3 at the *same absolute* ΔNLL cut-points
(`results/refiner/controls/matched_strata.json`). Two-sided judge AUROC, MTC (Essen):

| ΔNLL band | D1 | D3 | D2 |
|---|---|---|---|
| ≤ 0 | .753 (.722) | .638 (.618) | .606 (.579) |
| 0–0.1 | .830 (.826) | .774 (.740) | .684 (.670) |
| 0.2–0.4 | .855 (.837) | .825 (.816) | .730 (.724) |
| > τ | .861 (.855) | .850 (.838) | .775 (.779) |

D2 is below D1 in every band on both corpora, by 0.08–0.16, while changing *more* notes
per band. **"Most generated drafts are different, not wrong" is retracted**: it rested on
only 2.5–4.4% of D2 clearing τ, but only 13.0% of D1 clears it either — τ comes from
variants differing in 92% of notes, so it is a conservative bar for any synthetic draft.
The claim now is that generated damage is **diffuse, not small**. Also revised: the
transfer loss is symmetric (D1→D3 −27%, D3→D1 −24%), so it measures source-specific
learning in every synthetic benchmark, not D3's rules in particular. `paper-core.md`
C4/C8 and must-not-claim #8/#11 carry both revisions.

### Getting the data

Corpora are **not committed** — `data/` is gitignored and neither collection permits redistribution.

```bash
python -m src.corpus fetch          # MTC-ANN + ESSEN from Zenodo, MD5-checked
python -m src.corpus                # print the dataset table

python scripts/train_judge.py       # train J + run its calibration controls
python scripts/validator_report.py  # the D1/D3/D4 degradation table
python scripts/train_generator.py   # train G (+ undertrained checkpoints) for D2
python scripts/d2_report.py         # where D2 sits vs D1/D3/D4

python scripts/train_refiner.py --alphas 0 0.1 0.5 1.0 2.0 3.0   # the sweep
python scripts/refiner_analysis.py results/refiner/sweep_seed0.json
python scripts/draft_source_comparison.py --sweeps <sweep-a> <sweep-b>
```

Licensing, which matters for the paper: the MTCFeatures data is **CC BY-NC-SA 3.0** (attribution, non-commercial, share-alike) and should be cited to both the MTC and the Zenodo record. Essen feature values there derive from CCARH's **kern, whose own terms are "licensed (not sold)" with notable restrictions — so use the Zenodo feature sequences rather than redistributing kern.

## Environment & setup

- Python 3.12, managed with `uv` (`pyproject.toml` + `uv.lock`); a `requirements.txt` is also present for pip-based setups.
- Activate the existing venv with `source .venv/bin/activate`, or use `uv run ...`.
- `export TF_ENABLE_ONEDNN_OPTS=0` before running anything — TensorFlow is a dependency (legacy TFRecord path) and this avoids oneDNN numerical noise.

## Commands

All commands go through `main.py`.

```bash
# Generate edit scripts for a dataset (needed once per dataset before training/eval)
python main.py sample --dataset train --max-edits 5
python main.py sample --dataset val --max-edits 5
python main.py sample --dataset test --max-edits 5

# Train
python main.py train --model baseline
python main.py train --model edit_aware --alpha 0.5

# Evaluate (loads models/<model>.pth)
python main.py evaluate --model baseline --dataset test
python main.py evaluate --model edit_aware_alpha_0.5 --dataset test

# Compare baseline vs. edit-aware on a random test batch, save JSON to results/samples/
python main.py analyze --alpha 2.0

# Render a 4-panel piano-roll PNG for one sample from an analyze JSON
python main.py visualize --file results/samples/analyze_alpha_2.0.json --sample 0
```

`run.sh` runs the full sweep: trains baseline + edit-aware models at alpha ∈ {0.1, 0.5, 1.0, 2.0, 3.0}, then evaluates all of them on the test set. It assumes `main.py`'s default hyperparameters (100 epochs, batch size 32, lr 1e-4, d_model 256, 4 heads, 4 layers, patience 5, min-improvement 1e-4).

Rebuild modules have tests (stdlib `unittest`, no pytest dependency):

```bash
python -m unittest discover -s tests -v
```

The legacy pipeline has no test coverage — verify changes there by running the relevant `main.py` subcommand.

## Architecture

### Rebuild modules (`src/align.py`, `src/tokenizer.py`)

- **`src/align.py`** — Wagner-Fischer alignment over note tuples. Single source of truth for edit-script derivation, Correction Cost, fidelity, and the 5-class per-position labels (`KEEP`/`SUB_PITCH`/`SUB_DUR`/`INSERT_BEFORE`/`DELETE`). Corpus-independent but schema-dependent, so the cost vector is a `CostVector` parameter rather than inlined — that also provides the spec's weighted-CC variant for free. `correction_cost()` is verified against the legacy `minimal_edit_distance()` across 300 random pairs, so the rebuild provably preserves the published metric.
- **`src/tokenizer.py`** — `MelodyTokenizer` with explicit REST and BAR tokens. `frames_to_events()` replaces `pitch_seq_to_notes()`; `encode_with_spans()` returns each event's token indices and is what downstream code must use to place masks or labels. Pass `bar_starts` from the corpus whenever it is known; the `ticks_per_bar` fallback is only correct for fixed-grid sources.
- **`src/corpus.py`** — MTC / Essen loading. `Melody` carries `events` plus the annotations D3 and the judge need (`scaledegree`, `beat`, `beatstrength`, `phrase_ix`, `phrase_end`, `timesignature`, `tonic`, `mode`); `melody.notes` is the `(pitch, duration)` view the aligner consumes. Durations are integer ticks at `TICKS_PER_QUARTER = 24`, which represents every duration in both corpora exactly (verified across 462k notes). `split_by_tunefamily()` is the splitter to use — see below.

Three design decisions in these modules that the spec leaves open, resolved here and needing to be stated in the paper:

1. **Bars are token-level structure, not events.** The note sequence handed to the aligner stays `(pitch, duration)`, so edit scripts never contain bar operations (asking a composer to "insert a barline" is not an edit). Bar positions come from the source metre, not cumulative duration, which is what actually bounds the onset cascade.
2. **A note differing in both pitch and duration has no 5-class label.** The full script keeps both ops; the single-position label falls back to `SUB_PITCH` (configurable via `both_differ`). When several ops land on one position, first-in-canonical-order wins.
3. **Trailing insertions attach to an EOS slot.** `edit_labels()` returns `len(source) + 1` labels — insertions past the end of the source have no input position otherwise, and the tokenizer emits EOS anyway.

Three further decisions in `src/corpus.py`, each with a measured justification:

4. **`restduration_frac[i]` is the rest *following* note i**, confirmed against the corpus's own `nextisrest` flag (4876 co-occurrences, versus zero for the preceding note). Attaching it to the wrong side shifts every rest one position and drops bar accuracy from ~98% to 76%/71% — while every round-trip test still passes. `test_detected_bars_hold_the_duration_their_metre_implies` is the regression guard, and it works by checking the representation against real musical structure rather than against itself.
5. **A bar begins where the beat count stops advancing** (`beats[i] <= beats[i-1]`, not `<`): two onsets cannot share a metric position within one bar, so an equal beat also means a new bar. Reading bars from annotated metre rather than cumulative duration is what handles pickup bars and the 30.6% of MTC-ANN melodies that change metre mid-piece.
6. **Splits are by tune family, never by melody.** Family members are variants of one another, so a random split would leak a near-duplicate of a test melody into training, and would violate the judge's required independence from the generator even with disjoint melody ids.

Corpus as loaded:

| | MTC-ANN | ESSEN |
|---|---|---|
| melodies | 360 | 8,469 |
| events / melody (mean) | 49.0 | 54.5 |
| tune families | 26 | 0 |
| melodies with a variant | 360 (all) | 0 |
| rest fraction | 1.8% | 3.7% |
| distinct durations | 16 | 40 |
| free metre (no barlines) | 13 | 1,035 |

Both roughly double the old 4-bar task's 25 events per melody, which is part of answering P5's toy-task objection.

Measured on 20k melodies of the *legacy* corpus, justifying the representation change:

| | legacy | rest-aware + full duration vocab |
|---|---|---|
| melodies fully representable | 54.2% | 100% |
| duration-token coverage | 96.6% | 100% |
| melodies with rhythm intact | 21.7% | 100% |
| vocabulary size | 99 | 157 |

Retention collapses far faster than coverage because a *single* out-of-vocabulary duration rejects an entire melody. Widening durations from `(1,2,3,4,6,8)` to `1..64` costs 58 embedding rows and recovers ~46% of the corpus; dropping rests was compressing time in 78% of melodies. Rests are 18.6% of events.

### Migration status

`src/train.py` and `main.py` still use the legacy vocabulary, `pitch_seq_to_notes()`, `minimal_edit_distance()`, and `build_edit_mask()`. Nothing has been ported yet — deliberately, since the model (5-class head) and corpus are both about to change, and migrating now would mean rewriting the dataset, mask building, and model together against a corpus that is being replaced. When porting, `build_edit_mask`'s `1 + 2*t` arithmetic must be replaced by `encode_with_spans()`; REST and BAR void the alternating-pair assumption it relies on.

### Legacy pipeline

- **`src/train.py`** — the entire ML pipeline in one module (tokenization, edit simulation, dataset, model, training loops, evaluation). Originally refactored out of `training.ipynb`; that notebook and `data_processing.ipynb` / `data_analysis.ipynb` are the design history for the data format and are useful for understanding *why* something works this way, but `src/train.py` is the source of truth for behavior.
- **`src/audit.py`** — post-hoc inspection: `analyze()` loads a trained baseline + edit-aware pair, runs both on one random test batch, prints/saves a comparison; `visual_result()` renders piano-roll images from that JSON. Invoked via `main.py analyze` / `main.py visualize`. The file's `if __name__ == "__main__"` block (`validate_val_dataloader_consistency()`, a debug script that checks `MelodyDataset` seeding is stable across sessions) is not wired into `main.py` and is itself broken: it calls `convert_to_lists()`, but that function's body (src/audit.py:194-200) is missing its `def` line — it's dead code stranded after `visual_result()`'s `return`, so the name doesn't exist at module scope and calling it raises `NameError`. Irrelevant to the documented CLI commands; only matters if someone runs `python -m src.audit` directly.
- **`main.py`** — argparse CLI wiring the above together; also owns model/loss save-load conventions (`models/<name>.pth`, `logs/<name>_loss_train_val.json`, `logs/eval_results_<model>_<dataset>.json`).

### Data / tokenization pipeline

- Raw data is per-note MIDI pitch sequences stored as `.npy` arrays (`data/{train,val,test}_data.npy`), derived from TFRecord piano-roll files under `data/{train,validation,test}/`. Frame value `128` = hold (extends previous note's duration), `129` = rest. `pitch_seq_to_notes()` collapses these into `(pitch, duration)` note tuples.
- Vocabulary is built at import time in `src/train.py`: `<PAD>/<BOS>/<EOS>` + one token per MIDI pitch (21–108) + one token per allowed duration (`[1,2,3,4,6,8]`). A note becomes two tokens, `[Pitch, Duration]`; a sequence is `<BOS> P d P d ... <EOS>`.
- **Edit simulation** (`simulate_edits`, used by `MelodyDataset.__getitem__`) corrupts a clean note sequence with a random number of edits (`replace_pitch`, `change_duration`, `delete_note`, `insert_note`) and returns the corrupted notes plus an edit script describing what changed. This is done on-the-fly per `__getitem__` call, seeded deterministically from `BASE_SEED + idx + epoch * 100000` (via `dataset.set_epoch(epoch)`, called each epoch in the training loops) — this makes corruption reproducible across runs/sessions for a given epoch, not random noise you need to worry about when debugging.
- **Edit script tuples encode the reverse operation**, not the forward corruption, so `build_edit_mask()`/downstream code can locate the affected span in the *corrupted* sequence: `("replace_pitch", t, old_pitch)`, `("change_duration", t, old_duration)`, `("insert_note", t, note)` (reverses a `delete_note` corruption — the note to insert back), `("delete_note", t)` (reverses an `insert_note` corruption — just a position, no note payload needed). Position `t` is a note index into the *corrupted* note list, not the original.
- `build_edit_mask()` turns an edit script into a binary per-token mask aligned to the (BOS-prefixed) token sequence — this mask is what the edit-aware loss upweights. Note index `t` maps to token indices `1 + 2*t` (pitch) and `1 + 2*t + 1` (duration); for `insert_note` (i.e. a corruption-side deletion), there's no token at `t` to mark, so it approximates by flagging the *following* note's two tokens instead (a documented simplification, see the inline comment "simple & stable").
- Separately, `sample_edit_script()` / `main.py sample` generates and persists edit scripts to `data/{split}_edit_scripts.json` — a static, precomputed alternative to the on-the-fly simulation above (used for inspection/analysis, not by the live `MelodyDataset`).
- Evaluation metric throughout is **melody edit distance**: `minimal_edit_distance()` is a Levenshtein-style DP over notes (not tokens) where substitution cost is 0/1/2 depending on whether pitch and/or duration differ.

**Known broken state — `main.py sample` crashes.** `sample_edit_script()` (src/train.py:180) calls `sample_position(origin_notes, edit_type, used_positions)` without the `rng` argument, so `sample_position()` falls back to its default `rng=None` — then unconditionally calls `rng.choice(valid_positions)` whenever any valid position exists, raising `AttributeError: 'NoneType' object has no attribute 'choice'`. This fires on essentially any real melody (confirmed by direct invocation), so `python main.py sample` cannot currently produce `data/{split}_edit_scripts.json`. Contrast with `simulate_edits()` (the on-the-fly path actually used by training/eval), which correctly threads its local `rng = random.Random(seed)` through every `sample_position()` call and is unaffected. Fix: pass `rng` through in `sample_edit_script()` (e.g. a `random.Random()` instance), mirroring `simulate_edits()`.

### Model

`MelodyTransformer` (`src/train.py`) is a decoder-only causal transformer (`nn.TransformerDecoder` used in self-attention-only mode: `decoder(x, x, tgt_mask=causal_mask)`) over the token vocabulary, producing next-token logits for the clean sequence given the corrupted one (`inputs = corrupted[:, :-1]`, `targets = clean[:, 1:]`).

**Known broken state:** the model was originally meant to have a second `edit_head` output (predicting per-token edit likelihood) alongside `token_logits`, matching the `edit_weighted_loss` function and the `train_edit_aware` / `audit.analyze` call sites, which do `logits, edit_logits = model(inputs)`. After the `edit_head` branch was merged, `MelodyTransformer.forward()` only returns `token_logits` (the edit head is commented out at src/train.py:643-644), so three call sites currently crash:
  - `train_edit_aware()`'s training loop (src/train.py:955-970) never calls the model at all — it references an undefined name `logits` directly (`NameError`).
  - `train_edit_aware()`'s validation loop (src/train.py:996) does `logits, edit_logits = model(inputs)`, which fails to unpack since `forward()` returns a single tensor, not a tuple.
  - `audit.analyze()` (src/audit.py:64-65) has the same two-value unpacking bug for both the baseline and edit-aware model calls.

  `train_baseline()` and `evaluate_model()`, which call `model(inputs)` expecting a single tensor, are unaffected. If asked to fix edit-aware training, the fix needs to reconcile the model's return signature with these call sites (either restore the edit head and its return value, or switch these call sites to `weighted_edit_ce_loss`, which only needs `token_logits` and the ground-truth `edit_mask` — no model-predicted edit logits — and is already correctly used, once the missing `model(inputs)` call is added, elsewhere in the same training loop).

### Directories

- `data/` — gitignored; raw TFRecords, derived `.npy` arrays, generated edit-script JSONs.
- `models/` — gitignored; saved `.pth` checkpoints, named `baseline.pth` / `edit_aware_alpha_<alpha>.pth`.
- `logs/` — gitignored; training loss curves and evaluation result JSONs.
- `results/samples/` — output of `main.py analyze` / `visualize`.
