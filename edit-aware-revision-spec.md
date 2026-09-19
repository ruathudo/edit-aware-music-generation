# Edit-Aware Symbolic Music: Revision & Implementation Spec

**Purpose of this document.** A self-contained brief for a coding agent rebuilding the experimental pipeline of the paper *Learning Symbolic Music Through Edit-Aware Framework* (rejected at ISMIR 2026). It states what went wrong, what to build, and what results should look like. Section 1 is context; Sections 2–7 are the work.

**Design constraint driving every decision below: no human annotation, no user study, no recruitment.** Every credibility claim must be earned automatically.

---

## 1. What went wrong

### 1.1 The rejection in one line

All three reviewers found the idea worthwhile (novelty and reusable-insight scores were mostly positive). The paper was rejected on execution — specifically on a gap between what it claimed and what it demonstrated.

### 1.2 The five real problems

**P1 — Random corruption does not model revision.** Training data was built by applying 1–5 uniformly sampled edits (40% pitch / 30% duration / 15% delete / 15% insert) to ground-truth melodies. Every reviewer objected: a randomly substituted pitch in a *monophonic* melody with no harmonic context may produce a perfectly valid alternative line, not an error. The learned task is therefore "undo random noise," not "correct a draft." This was the meta-reviewer's headline objection and is the single reason the paper failed.

**P2 — "Human-in-the-loop" framing with no human anywhere.** The abstract, introduction and evaluation section all claim collaborative utility and criticize existing metrics for failing to capture it. No composer looked at any output, and nothing validated that flagged positions correspond to edits a person would make. The framing wrote a check the experiments did not cash.

**P3 — "Where and how" promised, only "where" delivered.** The abstract states the model predicts where corrections occur *and how they transform* the draft. The edit head outputs a binary mask — pure localization. The conclusion admits this; the abstract does not. Worse, the paper's central criticism of imitation learning (that it forces the model to implicitly infer *which* correction to apply) applies unchanged to its own method, since the generative head still does exactly that.

**P4 — Representation contradiction and under-specification.** Section 3 presents Onset Shifting as part of the contributed edit vocabulary; Section 5 drops it because the (P, D) encoding makes onsets implicit. Rests are never mentioned though the source melodies contain them. A single duration error cascades an onset shift through every subsequent note. The auxiliary head's attachment point, output dimensionality, and loss instantiation in Eq. 3 are never specified.

**P5 — Thin evaluation.** One metric (Correction Cost), one model size (256-dim, 4-layer, 4-head), one baseline. No fidelity check (does the model edit the input or ignore it and regenerate?), no musical quality metric, no scale or data ablations. The paired per-sample analysis is defined in Section 6.2 and never reported. A 6.3% CC reduction on a 4-bar toy task at one configuration is not enough evidence for a methodological claim.

**P6 — Related work.** Zero citations in the Introduction. Missing and named by reviewers: McLeod et al., *MIDI Degradation Toolkit* (ISMIR 2020) — near-identical corruption-and-correction methodology; Ramoneda et al., *RefinPaint* (ISMIR 2024) — proofreading, where-and-how to modify; Ji, Yang & Luo (2023) survey; SWE-bench / code-patch generation as an analogy; AMT-correction literature; Lakh MIDI provenance of the dataset; Vaswani et al. for the architecture itself.

---

## 2. Strategy for the redo

Three moves, all automatic:

1. **Replace the draft distribution.** Drafts come from a real generative model and from music-theoretically grounded degradation, not uniform random edits. Random corruption is retained *as a comparison condition*, not as the training regime.
2. **Prove degradation instead of assuming it.** An independently trained held-out language model acts as an automatic judge. Every draft source is scored; only measurably degraded drafts enter training and evaluation. This answers the reviewers' central question with evidence rather than assertion.
3. **Drop the human-in-the-loop framing.** Retitle the task *draft refinement* / *symbolic proofreading* / *post-generation correction*. This removes the most-repeated objection at the cost of one rewrite and zero experiments. Collaborative utility can be stated as motivation in one sentence; it must not be claimed as a demonstrated result.

Plus two model-side fixes: a multi-class edit-type head (closes P3) and a repaired token representation (closes P4).

**The strongest new result is likely free.** If the validator in Section 4 shows that a large fraction of uniformly random perturbations do not measurably degrade the melody, that is an empirical confirmation of the reviewers' concern, a motivation for the entire redo, and a reusable finding for anyone doing synthetic-corruption work in symbolic music. Build the validator early and report it prominently.

---

## 3. Phase 1 — Data pipeline

### 3.1 Base corpus

Keep the 4 Bars Monophonic Melodies Dataset (Zenodo 13369389). Add its provenance explicitly: it derives from Lakh MIDI, so these are popular-song vocal lines — cite both, and use that fact (see D3 below).

Optional second corpus, worth the effort if time permits: **Essen Folksong Collection** (clean, well-documented, symbolic, freely available, and not Lakh-derived) as a cross-corpus generalization check.

### 3.2 Four draft sources

Implement each as a module behind a common interface producing `(M_0, M_star, edit_script, source_tag)`.

**D1 — Random perturbation (legacy baseline).** Reimplement the existing scheme exactly as published. Its role is now to be the *control* that the validator exposes as weak. Do not delete it.

**D2 — Model-generated drafts (primary training source).**
- Train a generator `G` on the training split: same tokenizer, autoregressive, small (this is not the contribution).
- Produce drafts by **prefix continuation**: feed the first *k* notes of a reference (sample `k` uniformly from roughly 25–50% of the sequence), sample the remainder from `G`.
- Sweep sampling temperature (e.g. 0.8 / 1.0 / 1.2) and optionally use undertrained checkpoints, to control error severity.
- **Known risk:** a continuation may be a valid alternative melody rather than a flawed one — the same objection P1 in new clothes. This is exactly why the validator filter (Section 4) is mandatory for D2, not optional. Keep only pairs the judge scores as degraded relative to the reference.
- Honest framing of what D2 gives you: *refinement of real generator output toward a target*, which is the deployment scenario the paper always claimed.

**D3 — Musically grounded degradation (rule-based, no annotation).** Errors chosen because music theory says they are errors, not because a RNG chose them:
- *Out-of-scale substitution* — move a pitch to a non-diatonic degree given the inferred key (Krumhansl-Schmuckler or simple pitch-class histogram key-finding).
- *Out-of-range / unsingable leap* — introduce intervals beyond a vocal-plausible threshold (e.g. > major 9th) or push notes outside the melody's established tessitura. Justified by the Lakh-vocal-line provenance.
- *Metric misalignment* — alter a duration so a structurally important note (downbeat, phrase start) lands off the strong beat.
- *Cadential disruption* — alter the final note away from tonic/dominant resolution.
- *Rhythmic monotony / collapse* — flatten a varied rhythmic figure to uniform durations.
- *Repeated-note collapse* — replace a melodic figure with a static repeated pitch.

Each rule emits a typed, localized ground-truth edit script. Sample 1–5 per melody as before, so severity is comparable to D1.

**D4 — Human-authored variants (evaluation only, optional but high value).** thesession.org publishes multiple human-submitted *settings* of the same tune in ABC — genuine human-authored variants of a shared melody, obtained by download rather than recruitment. Align setting pairs with Wagner-Fischer to get real human edit scripts. Use strictly as a held-out eval set; do not train on it. Note: this requires a download step outside the pipeline, and licensing/attribution should be checked before redistribution.

### 3.3 Representation fixes (P4)

- Add an explicit **REST** token. Real melodies have rests; the current scheme cannot express them.
- Add **BAR** delimiter tokens. These bound the onset-cascade problem: a duration error no longer silently shifts every subsequent note to the end of the sequence.
- **Decide on onset shift and be consistent.** Recommended: keep the (P, D) encoding, and *remove* onset shift from the contributed vocabulary in Section 3, stating plainly that it is subsumed by duration modification under this encoding. The alternative (explicit onset tokens) is more expressive but changes every downstream component. Pick one; the contradiction, not the choice, is what reviewers punished.
- Document the tokenizer in a table: token classes, vocabulary sizes, special tokens, sequence-length statistics.

### 3.4 Edit script derivation

All ground-truth edit scripts are derived by **Wagner-Fischer alignment** of `M_0` to `M_star` over note tuples, with the existing canonicalization rules (order by onset; merge consecutive ops on the same attribute; discard redundant edits; attribute modifications before deletions). For D1 and D3 the injected script is known, but derive it via alignment anyway and assert agreement — that assertion is a free correctness test on the aligner.

---

## 4. Phase 2 — Automatic degradation validator

This component is what replaces the user study. Build it before the main experiments.

**The judge `J`.** A melody language model trained independently of the generator `G`: different random seed, different architecture depth, and a *disjoint data split* from `G`'s training data. Independence matters — if `J` and `G` share weights or data the filter is circular.

**Per-draft measurements.**
- `ΔNLL = NLL_J(M_0) − NLL_J(M_star)` — the primary degradation signal. Positive means the draft is less plausible than the reference.
- Scale-conformance rate (fraction of notes diatonic to inferred key).
- Pitch-class histogram distance to the reference.
- Interval-distribution distance; maximum interval.
- Metric-accent alignment score (do structurally salient notes land on strong beats?).

**Two uses.**

1. **Filter.** Admit a pair to training/eval only if it clears a degradation threshold (e.g. `ΔNLL > τ`, τ chosen from the reference distribution, plus at least one musical criterion degraded). Report retention rate per source.
2. **Report — this is a headline table.** For each of D1/D2/D3, the fraction of "corrupted" drafts that are *not* measurably degraded. The expectation is that D1 retention is low and D3 retention is high. That contrast is the empirical justification for the whole redo.

**Table to produce:**

| Draft source | Mean ΔNLL | % with ΔNLL ≤ 0 | Δ scale conformance | Retention after filter |
|---|---|---|---|---|
| D1 random | | | | |
| D2 model | | | | |
| D3 musical | | | | |
| D4 human variants | | | | |

Be prepared for D1 to look worse than expected and say so. A finding that "a widely used synthetic-corruption protocol frequently fails to produce degradations" is a contribution, not an embarrassment.

---

## 5. Phase 3 — Model changes

### 5.1 Multi-class edit head (closes P3)

Replace the binary mask with **per-position classification over 5 classes**: `KEEP`, `SUB_PITCH`, `SUB_DUR`, `INSERT_BEFORE`, `DELETE`. Labels come from the Wagner-Fischer alignment of Section 3.4.

The model now predicts *where* and *what kind of* correction — which is what the abstract always claimed.

**Specify precisely (reviewers explicitly asked; put this in the paper):**
- Attachment point: linear projection on the final hidden states at **encoder/input positions** (positions of `M_0`), not decoder output positions.
- Output dimensionality: `hidden_dim → 5`.
- Loss: class-weighted cross-entropy. `KEEP` dominates heavily (typically >85% of positions) — use inverse-frequency weighting or focal loss and state which.
- Retain the binary-mask head as an ablation condition so the multi-class gain is measurable.

### 5.2 Objective

```
L = L_gen + α · L_edit

L_gen  = cross-entropy over the musical token vocabulary (unchanged)
L_edit = class-weighted cross-entropy over 5 edit classes at input positions
α      ∈ {0, 0.1, 0.5, 1.0, 2.0, 3.0}   (α = 0 is the baseline)
```

Write Eq. 3 in the paper with these instantiations named inline. The previous version left both terms abstract, which reviewers flagged.

### 5.3 Baselines

Three, not one:
- **B1** — imitation-only Transformer (the published baseline).
- **B2** — imitation-only, but **scale-matched** to the edit-aware model's total parameter count, so gains cannot be attributed to the extra head's parameters.
- **B3** — copy-input baseline (output `M_0` unchanged). This lower bound calibrates Correction Cost and exposes whether the models help at all.

---

## 6. Phase 4 — Evaluation suite

### 6.1 Metrics

**Correction Cost (CC)** — unchanged; minimal edit script from output to reference via Wagner-Fischer. Also report the **weighted** variant with a configurable cost vector `w` so the weighting machinery the paper describes is actually exercised.

**Fidelity — `EditDist(output, M_0)`.** Directly answers "did the model edit the draft or ignore it and regenerate?" A model with low CC and very high fidelity distance is not editing; it is producing a fresh melody that happens to resemble the target. Reviewers raised this; it costs nothing to measure.

**Musical quality of the output** — the same suite `J` uses in Section 4, applied to model outputs: NLL under `J`, scale conformance, pitch-class and interval distances to the reference.

**Edit-prediction accuracy** — precision / recall / F1 for localization (`KEEP` vs. everything else) and macro-F1 over the 5 classes. This is the metric that substantiates "where and how," and the old paper had nothing equivalent.

**Edit-type breakdown of CC** — as before, per operation type.

**Paired per-sample analysis** — promised in Section 6.2 of the old paper and never reported. Deliver: win rate (fraction of test melodies where edit-aware beats baseline), mean and median paired delta, bootstrap 95% CI, and a Wilcoxon signed-rank test. This is what turns a 6% aggregate improvement into a defensible claim.

### 6.2 Reporting

Every table reports mean ± std over **≥3 seeds**. Single-run numbers will not survive review a second time. Fix and log all seeds.

---

## 7. Phase 5 — Ablations

| Axis | Values | Question answered |
|---|---|---|
| α | 0, 0.1, 0.5, 1.0, 2.0, 3.0 | Does the U-shaped curve replicate on realistic drafts? |
| Draft source | D1 / D2 / D3 / mixed | Does edit-awareness help more when errors are real? |
| Head type | none / binary / 5-class | Does predicting *how* beat predicting only *where*? |
| Model size | 2/4/8 layers × dim 128/256/512 | Does the gain survive scale? (P5) |
| Data size | 10k / 50k / 125k | Is the gain a small-data artifact? |
| Eval transfer | train D3 → eval D4 | Does it transfer to human-authored edits? |

The draft-source and head-type axes are the scientifically interesting ones; the rest are the due diligence reviewers asked for.

---

## 8. Expected outcomes, including failure modes

**Likely to hold.**
- The validator shows a substantial fraction of D1 random perturbations are not degradations. This becomes the motivating result.
- The 5-class head beats the binary mask on CC and gives a directly reportable localization/type F1.
- The α curve stays U-shaped with an interior optimum, since the mechanism (auxiliary signal as guided attention, over-weighted signal crowding out generation) is not data-dependent.
- Insertions remain the hardest operation. Predicting *that* something is missing is genuinely harder than flagging that something present is wrong.

**Plausible and must be planned for.**
- **The edit-aware gain shrinks on D2/D3 relative to D1.** Realistic errors are less mechanically invertible than random ones, so the headline percentage may drop below 6.3%. If this happens: do not hide it. Report it as a finding — the published gain was partly an artifact of an easy synthetic task — and shift the headline to edit-type F1 and the validator analysis. A smaller, honest, well-measured effect on a defensible task is a stronger paper than a larger effect on a task reviewers already rejected.
- **The generator drafts may be too good or too different.** If `G`'s continuations are fluent, few clear the degradation filter and the training set shrinks; if they diverge wildly from the reference, edit scripts become long and the "edit" framing breaks down. Mitigate by tuning temperature and checkpoint, and by tracking the distribution of edit-script lengths — target a distribution comparable to the original 1–5 edits.
- **Class imbalance swamps the multi-class head.** If macro-F1 collapses to predicting `KEEP` everywhere, revisit loss weighting before revisiting the architecture.

**Decision rule.** If D3-trained models show a clear, statistically significant CC improvement with meaningful edit-type F1, the modeling contribution stands and the paper is a resubmission. If the effect largely vanishes on realistic drafts, pivot the paper to the negative/methodological result — *synthetic corruption protocols in symbolic music do not produce human-meaningful errors, here is a validator and a grounded alternative* — which is publishable and, given the reviewer comments, arguably more useful to the community.

---

## 9. Paper-side changes (no experiments required)

- **Reframe.** Remove "human-in-the-loop" from title, abstract, introduction and evaluation framing. Use *draft refinement*, *symbolic proofreading*, or *post-generation correction*. Collaborative motivation gets one sentence, flagged as motivation, never as a demonstrated result.
- **Align the abstract with delivery.** With the 5-class head, "where and how" becomes accurate — keep it. Drop the overstated qualitative contrast with imitation learning (P3), or restate it precisely: the auxiliary signal supplies typed localization that imitation learning leaves implicit, without claiming it removes implicit inference entirely.
- **Rebuild related work.** Add and position explicitly against McLeod et al. (MDT, ISMIR 2020) and Ramoneda et al. (RefinPaint, ISMIR 2024) — a reviewer already flagged near-duplicate methodology, so pre-empt it with a stated delta. Add Ji, Yang & Luo (2023); HITL music generation; AMT correction; SWE-bench patch generation as a cross-domain analogy; Lakh MIDI provenance; Vaswani et al. Put citations in the Introduction — currently it has none.
- **Structure.** Move Figure 1 earlier and enlarge it. Drop Table 1 or Figure 3 (they show the same data). Give Section 7 headed subsections matching the analysis terms defined in Section 6.2. Fix the stray "1" typo at line 288. Replace jargon ("high-bandwidth supervision signals") with what it denotes.
- **Specify the head.** Attachment point, output dimensionality, loss instantiation, class weighting — as in Section 5.1.

---

## 10. Engineering notes for the agent

- **Repo layout:** `data/` (corpus loaders, tokenizer), `drafts/` (D1–D4 modules behind one interface), `judge/` (validator `J` + metrics), `models/` (generator, baselines, edit-aware), `eval/` (CC, fidelity, quality, edit-F1, paired stats), `experiments/` (configs), `results/`.
- **Config-driven.** One YAML per experiment; no hardcoded hyperparameters. Every run writes its resolved config alongside its results.
- **Determinism.** Seed everything (Python, NumPy, framework, dataloader workers). Retain the existing deterministic on-the-fly synthesis so all models see identical sequences. Log seeds in results.
- **Cache draft generation.** D2 requires sampling from `G` for the full corpus — expensive. Generate once, persist to disk with a content hash of the generating config, reuse.
- **Test the aligner first.** Wagner-Fischer over note tuples underpins ground-truth labels, CC, and fidelity. Unit-test it against the known injected scripts of D1/D3 before anything else runs; a bug there silently corrupts every number in the paper.
- **Run order:** aligner tests → tokenizer + representation fixes → judge `J` → validator report (Section 4) → generator `G` → draft caches → main training sweep → ablations.
- Note that the validator report can and should be produced before any edit-aware model is trained. If it comes back saying random corruption was fine all along, the plan changes and you will want to know that on day three rather than day thirty.
