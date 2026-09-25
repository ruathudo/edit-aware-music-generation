# Paper framing plan

> **Superseded for drafting by `paper-core.md` (2026-09-25).** This file is the revision
> log: it records how each claim was measured, revised or withdrawn. Draft from the core.

Companion to `edit-aware-revision-spec.md`. The spec says what to *build*; this says
what to *claim*, and which measurement backs each claim.

Rewritten 2026-09-22 after span-level detection failed. The earlier version expected
span-level detection to rescue the realistic-draft setting; it did not, and the thesis
below is the one the evidence supports. Updated 2026-09-23 with the within-D3 locality
stratification, which rules out edit contiguity as the account of the boundary
(contribution 4). Numbers reconciled against the result files 2026-09-23 (§9 step 1),
and the headline re-run under validation-selected settings 2026-09-24 (§9 steps 2–3).
The framing itself is still to be revised (§9 step 9).

**Reporting conventions.** Unless stated otherwise: MTC-ANN references, three seeds,
and **one setting for every source and corpus — binary head, `weighting=none`, α=0.5,
τ=0.8 — selected on D1 *validation* data** (`scripts/select_settings.py`; test decoded
at τ\* only). Canonical numbers are `results/refiner/headline_r2/`. "Gated vs B3" is
(CC_B3 − CC_gated) / CC_B3 per seed, averaged over seeds; spread is the sample sd of the
per-seed percentages, in percentage points. Localization is compared threshold-free —
AP lift (average precision ÷ base rate) and AUROC of the head's P(edit) — because
argmax precision is undefined for a head that barely flags. **Gate precision** is
precision among the positions the gate actually edits.

Runs are not bit-reproducible: cuDNN determinism is not set, so an identical seed gives
a slightly different model. Two full runs of the headline differ by up to 1.17 pp on one
seed and 0.36 pp on a cell mean (`headline/` vs `headline_r2/`); state this.

---

## 1. Thesis

> **Apparent progress on symbolic draft refinement is largely a function of how the
> drafts were made.** Across three sources under identical machinery, localizability
> collapses as drafts become more realistic — AUROC 0.74 for rule-based degradation,
> 0.73 for random corruption, 0.64 for generator output — and measured performance
> falls with it, +9.98% to +6.35% to +0.00%. Generated drafts are the realistic case, and they
> leave **little local trace**: the best detector we have — an untrained two-sided judge —
> finds generator-written notes at 2.7× chance, against 6–7× for corruption, and no
> trained detector does better. *Localize-then-fix* is the wrong formulation for that
> class of draft. Where errors genuinely are local,
> the same machinery works and precision-gated editing beats copy-input by 6%.
>
> *Revised 2026-09-24 (§9 step 5): "not locally wrong / cannot be identified" is too
> strong. A two-sided judge localizes D2 at 2.7× chance (AUROC 0.69) — far below D1 and
> D3, but not nothing. The thesis is a large gap, not an absence.*

The paper's subject is therefore **the domain of validity of the localize-then-fix
formulation**, mapped from both sides. That subsumes the ISMIR reviewers' P1 objection
instead of conceding it: random corruption is indeed a poor model of revision, and the
obvious fix — use a generator — produces drafts nothing can localize.

## 2. Contributions, in the order they should appear

1. **A calibrated degradation validator.** An independently trained melody LM plus
   musical criteria, anchored on human-authored tune-family variants. The anchor is what
   makes it credible: genuinely valid alternatives score ΔNLL ≈ 0 (mean +0.054, median
   +0.031, 55.0% positive — a coin flip; `validator_report.json`, n=1,012 pairs), so the judge measures degradation rather than
   novelty.

2. **Correction Cost is not a measure of musical plausibility: coupled within a
   corruption process, decoupled across sources.** *(Revised 2026-09-25, §9 step 7 —
   was "near-orthogonal", which rested on two cluster means.)* Human variants of one tune
   sit ~92% normalized edit distance apart at +0.038 ΔNLL per unit of change; synthetic
   corruption sits ~7% away at +3.94 — roughly 100× more implausible per unit changed.
   Corruption moves a melody *off* the data manifold; real variation moves it a long way
   *along* it.

   Per draft, Spearman ρ between surface change and ΔNLL (`scripts/decoupling.py`; MTC,
   scored by `J`; 95% CI by bootstrapping tune families, since drafts and pairs from one
   family are not independent):

   | set | n | ρ | 95% CI |
   |---|---|---|---|
   | D1 random | 1,080 | **+0.52** | [+0.46, +0.59] |
   | D3 rule-based | 1,061 | **+0.47** | [+0.41, +0.52] |
   | D2 generated | 1,064 | +0.20 | [+0.11, +0.30] |
   | D4 human variants | 1,012 | +0.14 | [+0.04, +0.25] |
   | **all sources pooled** | 4,217 | **−0.06** | [−0.11, −0.01] |

   **Within one corruption process, more change means more damage** (ρ ≈ 0.5) — partly by
   construction, since the number of injected edits drives both. **Across sources the
   relation vanishes and slightly inverts:** the drafts furthest from their reference,
   human variants, are the most plausible. At matched distance the contrast is stark but
   thin — the 9 human pairs within the synthetic drafts' range (surface ≤ 0.20) have
   median ΔNLL 0.000 against +0.184 for synthetic drafts.

   So the defensible claim is narrower than "orthogonal": **CC can rank methods on a fixed
   draft source, but cannot compare across draft sources, and cannot distinguish an error
   from a valid alternative.** That is exactly the use the paper's evaluation needs to
   warn against, and it is why contribution 10 reports plausibility alongside CC.

3. **Localizability collapses as drafts become more realistic — and apparent method
   performance falls with it.** Three draft sources, same machinery, three seeds each:

   | source | errors come from | ΔNLL/unit | edit rate | AP lift | AUROC | gate flags / P | gated vs B3 |
   |---|---|---|---|---|---|---|---|
   | D3 rule-based | rules we wrote | 2.97 | 8.2% | 5.0× | 0.743 ± 0.038 | 1,968 / 0.904 | +9.98% ± 0.30 pp |
   | D1 random | unsystematic noise | 3.94 | 6.0% | 5.8× | 0.727 ± 0.005 | 1,489 / 0.823 | +6.35% ± 0.48 pp |
   | D2 generated | a real generator | 1.16 | 7.3% | 1.6× | 0.635 ± 0.003 | **0** / — | +0.00% ± 0.00 pp |

   Same-corpus (Essen references): D3 4.3× / 0.718 / +7.30%, D1 5.8× / 0.732 / +4.47%,
   D2 1.8× / 0.654 / +0.00%. Every D1 and D3 cell is 3/3 seeds on both corpora.

   **This is now one configuration end to end**, which the earlier table was not (it
   gave D2 a different head and weighting). Under it the D2 head **collapses to KEEP**:
   0 of ~52k positions clear τ on any seed, and argmax flags number 0–16 per seed
   against D1's ~2,300 — although D2 has *more* true edits (7.3% of positions against
   6.0%). Same loss, more positives, and the head finds nothing worth flagging. This
   is cleaner evidence than the old precision 0.137, because nothing about the method
   differs between the sources.

   D2 is not information-free: AUROC 0.64 and AP lift 1.6–1.8× show a weak ranking
   signal, the same ~1.5× the earlier measurements found. It is too weak for any
   position to be worth editing.

   ΔNLL/unit: D1/D3 from `validator_report.json` (n≈1,080); D2 from the `D2-infill
   T=1.0` row of `d2_report.json` (n=352), the temperature the refiner's drafts used.

   **D3 is a negative control, not a result.** Its edits come from the same rule family
   a detector can learn — out-of-scale, unsingable leap, cadential disruption — so it
   reports the best number in the paper precisely because the benchmark is circular.
   Report it in the ablation, keep it out of headline claims, and put the circularity in
   the caption. Spec §3 proposed rule-based degradation as the *fix* for D1's unrealism;
   the result is that it is worse in a different way — more realistic-sounding errors,
   less valid as a benchmark. Anyone about to build one needs to know that.

   *Revised 2026-09-24:* the circularity is **partial and now measured**, not total. D3's
   errors are genuinely implausible notes — an untrained judge finds them nearly as
   well as D1's — and a head trained on D1 alone reaches +7.25% on D3. Of D3's +9.98%,
   about **2.7 points** come from learning the injection rules. So the warning becomes:
   rule-based degradation inflates a benchmark by roughly a quarter of its reported gain,
   and only a transfer test reveals how much.

   D3 also bears on the two obvious accounts of the boundary. It is *less* off-manifold
   than D1 (2.97 against 3.94) and *less* point-like (mean edited run 1.50 against
   1.05), yet **no less localizable**. **Revised 2026-09-24:** earlier versions said
   "more localizable", from argmax precision. Threshold-free the two are level (AUROC
   0.743 vs 0.727 MTC, 0.718 vs 0.732 Essen; AP lift is lower for D3, 5.0× vs 5.8×).
   D3 wins only in the **confident tail** — gate precision 0.904 against 0.823 — and
   that is where its larger gated gain comes from. So D3 does not *falsify* manifold
   distance; it shows that a source 25% less off-manifold is just as localizable, which
   is evidence against manifold distance as the *sole* governor, not a refutation. The
   locality account is refuted separately (contribution 4). So neither manifold distance
   nor edit locality governs on its own. The working hypothesis — held as a hypothesis, since two
   earlier accounts were already falsified — is that what matters is whether the error
   has a **learnable regularity**: off-manifold is necessary but not sufficient, and the
   deviation must also be systematic enough to train a detector for. D3's errors are
   rule-shaped, D1's are unsystematic, D2's are in-distribution by construction.

   **Resolved by §9 step 5 (2026-09-24): both accounts hold, and they separate.** A
   detector that never saw a label or a rule — the untrained two-sided judge — ranks
   the sources **exactly in manifold-distance order**: D1 (ΔNLL/unit 3.94) AP lift
   6.2× / 7.0× (MTC / Essen), D3 (2.97) 5.9× / 6.1×, D2 (1.16) 2.7× / 2.7×. So what is
   detectable *in principle* is governed by how far off-manifold the error is. A trained
   head adds a source-specific component on top — **learnable regularity** — and that
   is exactly what fails to transfer (contribution 11). D3's lead over D1 appears only
   with training, and only in-domain. Three sources is a thin basis for "monotone"; state
   it as consistent with, not proof of.

   **The D1 result stays the headline** (+6.35%, replicated same-corpus at +4.47%, 9/9
   size configurations) precisely because random corruption is not something we designed a
   detector for.

4. **Contiguity matters, but it does not account for D2.** *(Revised 2026-09-25, §9
   step 6 — was "The boundary is not about contiguity".)* The obvious alternative account
   of contribution 3 is mechanical rather than epistemic: a per-position scorer suits
   isolated changes and struggles with contiguous ones, and D2 replaces spans of 2–6
   events while D1 alters single notes. Comparing sources cannot decide this, because D1
   and D2 differ in manifold distance *and* locality at once. D3 can, being uniformly
   off-manifold but mixed in locality; splitting its own test set by the longest edited
   run per draft holds the first fixed and varies the second
   (`scripts/control_strata.py`, headline checkpoints, 3 seeds, threshold-free):

   | D3, MTC (Essen) | drafts | head AUROC | two-sided judge AUROC |
   |---|---|---|---|
   | point-like (1 event) | 501 (2,340) | 0.800 ± 0.046 (0.762) | 0.938 (0.915) |
   | 2 adjacent events | 227 (1,204) | 0.726 ± 0.040 (0.703) | 0.806 (0.788) |
   | span-like (3+ events) | 333 (1,446) | 0.714 ± 0.034 (0.701) | 0.751 (0.723) |
   | *D2 overall, for comparison* | 1,064 (5,057) | *0.635 (0.654)* | *0.688 (0.683)* |

   **Contiguous errors are harder to rank** — AUROC falls 0.09 for the head and 0.19 for
   the judge from single notes to runs of three or more. The judge's larger drop fits a
   simple mechanism: when one event is masked, its edited neighbours stay visible as
   context and make it look consistent. **But D3's span-like band still localizes better
   than D2** (head 0.714 vs 0.635, judge 0.751 vs 0.688, both corpora), so contiguity
   explains part of D2's gap and not all of it.

   The earlier version of this contribution said the opposite — "precision is flat,
   contiguity is not what breaks localization" — from one seed at τ=0.9, reading argmax
   precision. That measure is the head's *confident tail*, and it is indeed flat
   (0.825 → 0.805); the full ranking is not. Report both: gating, which acts only on the
   confident tail, is insensitive to contiguity; localization as a whole is not.

5. **Generated material is far less localizable, by every method tried.**
   *(Heading revised 2026-09-24 — was "not localizable"; the two-sided judge row below
   is why.)* Measurements on D2-infill drafts:

   | method | result |
   |---|---|
   | per-position edit head | ~1.5× chance, **flat across degradation bands** (1.56× on ΔNLL ≤ 0, 1.66× on judge-certified degraded) |
   | judge surprisal pooled over spans, zero-shot | 1.2× chance |
   | trained span detector, interior + both boundary states, 3 seeds | 1.6× chance (0.75% ± 0.19% top-1 against 0.48% chance) |
   | causal judge, per note, zero-shot | 1.4× AP lift, AUROC 0.56 |
   | **two-sided judge, per note, zero-shot** | **2.7× AP lift, AUROC 0.69** (both corpora) — the best detector found, and still less than half D1's 6.2–7.0× |
   | heads trained on D1 or D3, transferred | 1.3–1.4×, AUROC 0.58 |

   **The decisive control: the same architecture, size, recipe and setting reaches AP
   lift 5.8× and AUROC 0.73 on D1, against 1.6× and 0.64 on D2** — on both corpora. The difference is in the drafts, not the method — this is
   not a capacity or tuning failure. The span task was also given an unfair advantage
   (candidates restricted to the true span-length range, which a deployed system would
   not know) and still failed.

   **"Most generated drafts are not degraded, so there is nothing to find" — partly
   true.** *(Revised 2026-09-25, §9 step 6; the earlier version said "tested and
   rejected" from one seed, n=28 degraded drafts.)* Re-run on 3 seeds with a fresh set
   of 5,339 MTC drafts plus 5,057 Essen drafts, τ recomputed per judge exactly as the
   validator does (0.585 MTC; 0.503 for the control judge):

   | D2 band (by judge ΔNLL), MTC (Essen) | drafts | head AUROC | two-sided judge AUROC / lift |
   |---|---|---|---|
   | not degraded (ΔNLL ≤ 0) | 1,080 (937) | 0.640 (0.657) | 0.606 / 1.6× (0.579 / 1.4×) |
   | mild | 2,130 (2,060) | 0.615 (0.643) | 0.692 / 2.6× (0.676 / 2.4×) |
   | moderate (≤ τ) | 1,998 (1,838) | 0.642 (0.658) | 0.733 / 3.5× (0.726 / 3.6×) |
   | **degraded (> τ)** | **131 (222)** | 0.656 (0.631) | **0.775 / 4.6× (0.779 / 4.4×)** |

   The **head is flat** across bands, as before — it does not exploit degradation. The
   **judge climbs steadily**, and on certified-degraded drafts reaches D1-like
   localization. So where generation produced a genuine error, the error is findable;
   such drafts are 2.5–4.4% of D2, and the rest is mostly *difference*, not error (M2).
   Caveat: the bands are defined by the causal judge's ΔNLL and scored by the two-sided
   judge, trained on the same data, so part of the rise is built in. The head's
   flatness carries no such link.

   Why: the damage is distributed. Around a spliced span the *unchanged* notes also
   become dearer under the judge (+0.0677) alongside the span itself (+0.2162). The
   generator writes a locally coherent passage; what is wrong is how it sits in the
   piece.

6. **The published result reproduces, and is an artifact.** On D2-infill drafts the
   U-shaped α curve replicates: the CC reduction over B1 peaks at α=0.5–1.0 at
   **+6.3%** (α=1.0, per seed 11.6 / 3.5 / 3.9%, 3/3 positive; published: 6.3%), and
   turns negative at α=3.0. But at that optimum fidelity is 0.0009 per note and CC is
   4.726 against copy-input's 4.709: the curve measures convergence to copying. On D1 —
   the analogue of the published random-corruption protocol — the gain over B1 does
   **not** reproduce reliably (α=0.5: +6.3 / +19.3 / −7.2%, 2/3 seeds), so "reproduces"
   must be stated for D2 only. (An earlier version quoted 5.85%, which is the
   size-ablation mean pasted in error.) Visible only with a copy-input baseline and a fidelity
   metric, neither of which the original evaluation had. **Prescription: always report
   B3 and fidelity alongside CC.**

7. **Precision-gated decoding, where errors are local.** Copy by default; hand an event
   to the decoder only when the head's confidence clears τ. **At τ=1 the output is the
   draft verbatim, so the method provably cannot do worse than leaving the draft
   alone.** **+6.35% over copy-input on D1** (per seed 5.87 / 6.83 / 6.34%; win 23.0%,
   loss 6.3%), with the binary head and settings chosen on validation; replicated
   same-corpus at **+4.47%** (4.10 / 4.75 / 4.55%) on a 5× larger test set. A second full
   run gave +5.99% and +4.55%. Among the positions the gate edits, 82% are true errors.
   Validation selection over 96 settings found the gain flat — every weighting peaks at
   6.3–6.5% on validation, 84/96 settings beat copy-input on all seeds — so the result
   is not a product of tuning. Report per-seed values, not a pooled Wilcoxon p, which
   treats melodies × seeds as independent. On D2 it returns exactly 0.00% — no position
   clears τ on any seed, so it abstains, which is correct given AUROC 0.64.

8. **The gain holds across capacity, and the gate bounds the damage.** Nine
   configurations, 2/4/8 layers × d_model 128/256/512, 1.6M to 42.7M parameters (26×),
   three seeds each. **Gated editing beats copy-input in 9/9 configurations on every
   seed**, +3.78% to +6.99%, mean +5.85%. *(Run at the earlier none / 0.5 / 0.9 setting,
   which validation ranks 2nd of 96, 0.07 pp behind the selected one; not re-run.)*

   Two things in that sweep matter more than the headline. **Localization precision is
   flat across the whole range** — 0.655 to 0.719, correlation with log-parameters
   −0.58, i.e. if anything it *falls* with scale. Localization is information-limited,
   not capacity-limited. That is a third independent argument against "your models are
   too small," alongside the D1/D2 same-architecture control: precision does not improve
   with 26× more parameters even where the task *is* solvable.

   And **the gate bounds the damage**. Free decoding ranges over CC 3.42–9.04 and is
   worse than doing nothing in **9/9** configurations; gated stays within 2.77–2.87 and
   beats doing nothing in 9/9. At 1.6M parameters free decoding is three times worse
   than leaving the draft alone, while gated is still ahead of it. That is the τ=1
   guarantee doing real work, and for a deployable tool the graceful degradation is
   arguably worth more than the 6%.

9. **An oracle bound, and where the judge belongs.** Given ground-truth labels the model
   removes 49.3% of the draft's error (D1); the real head recovers 12.2%.

   *Revised 2026-09-25 (§9 step 6).* Feeding the judge's surprisal to the encoder as an
   input feature — re-run on 3 seeds under the selected setting
   (`results/refiner/controls/`):

   | MTC D1 | AUROC | AP lift | gate precision | gated vs B3 | free decoding |
   |---|---|---|---|---|---|
   | headline | 0.727 | 5.8× | 0.823 | **+6.35%** | −17.7% |
   | surprisal input | **0.820** | **7.3×** | 0.819 | +5.14% | −61.0% |

   Worse on all three seeds (−0.25 / −0.56 / −2.83 pp; one seed carries most of it). The
   direction of the old single-seed claim replicates; **its explanation does not.** The
   old account was "raising recall adds wrong edits, because precision is binding" —
   but here localization *improves* and gate precision is *unchanged*. The loss is in the
   corrections: free decoding collapses from −18% to −61%, so surprisal in the encoder
   degrades what the decoder writes. With contribution 12 this gives a clean design
   rule: **use the judge to decide *where* (at the gate: +1.6 pp), not as model input
   (−1.2 pp).**

10. **Gated editing is the only condition that improves both Correction Cost and
    plausibility — and free decoding shows why CC alone cannot be trusted.** The
    decoded outputs scored by the judge (`scripts/output_quality.py`; ΔNLL is output
    minus reference, "recovered" is the share of the draft's ΔNLL removed):

    | MTC | CC vs copy-input | ΔNLL | recovered | on edited melodies |
    |---|---|---|---|---|
    | D1 draft (B3) | 0 | +0.298 | — | — |
    | D1 B1 free | −24.7% | +0.176 | +40.9% | — |
    | D1 EA gated | **+6.35%** | +0.250 | +16.3% | **+30.1%**, 2.8% made less plausible |
    | D3 EA gated | +9.98% | +0.166 | +38.8% | +47.5%, 1.4% less plausible |
    | D2 B1 free | −3.0% | +0.135 | +2.7% | — |
    | D2 EA gated | +0.00% | +0.139 | 0 | gate never opens |

    Essen replicates every row (D1 gated edited subset +31.1%, 2.8% less plausible;
    D3 +53.3%, 0.7%; D2 free +5.8%).

    **The CC gain is not bought with plausibility.** Where the gate edits, the judge
    finds the melody more plausible in 97.6% of cases (MTC D1, seed 0) and removes 30%
    of the draft's implausibility. The overall +16% is diluted by the half of melodies
    the gate leaves alone.

    **Free decoding "wins" on plausibility while losing on CC — the dissociation of
    contribution 2, reproduced on model outputs.** It recovers 41% of the implausibility
    against gated's 16%, yet is 25% *worse* than copying on CC. It decodes greedily, so
    it regenerates toward the language model's mode: 14.5% of its outputs are *more*
    plausible than the reference itself (drafts: 5.2%), and a third of the melodies it
    changes end up both more plausible and further from the reference. A plausibility-
    only evaluation would crown free decoding; a CC-only evaluation hides that gated
    editing improves the music. **Report both; the claim for gated editing is that it is
    the only condition that beats the draft on both axes.**

    Two smaller findings. D2's implausibility resists even regeneration (free decoding
    recovers 3–6%), consistent with damage that is contextual rather than located. And
    gated editing leaves D1's metric misalignment untouched (Δ −0.117 against the
    draft's −0.119, where free decoding reaches −0.071): the gate acts on pitch errors,
    matching the head-type finding that duration substitutions are poorly predicted.

    Caveat: the judge is a language model and so shares free decoding's preference for
    typical continuations; the 14.5%-above-reference figure is itself evidence that ΔNLL
    rewards mode-seeking. The breakdown percentages (97.6%, 14.5%, a third) are from one
    seed of MTC D1.

11. **An untrained two-sided judge localizes better than any trained head — and
    transfer separates what is detectable from what is learned.** Zero-shot scorers
    never see a label or a corruption rule (`scripts/zero_shot_localization.py`; AP
    lift / AUROC on the same positions as the head, labels checked against its saved
    arrays; same-corpus rows use judges retrained without the Essen test slice):

    | MTC (Essen) | causal judge | two-sided judge | trained head (in-domain) |
    |---|---|---|---|
    | D3 | 4.7× / 0.768 (5.1× / 0.756) | **5.9× / 0.812** (6.1× / 0.790) | 5.0× / 0.743 (4.3× / 0.718) |
    | D1 | 4.1× / 0.775 (4.9× / 0.774) | **6.2× / 0.838** (7.0× / 0.831) | 5.8× / 0.727 (5.8× / 0.732) |
    | D2 | 1.4× / 0.561 (1.4× / 0.564) | **2.7× / 0.688** (2.7× / 0.683) | 1.6× / 0.635 (1.8× / 0.654) |

    The two-sided judge wins on every source and both corpora. The comparison favours
    the head, not the judge: within-melody standardization — which the zero-shot scores
    use — *lowers* the head's numbers (D1 AUROC 0.727 → 0.701), so the head is shown at
    its best. Two-sided context matters most on D2 (1.4× → 2.7×), confirming that a
    generated span's damage lies partly in how it fails to rejoin what follows, which a
    causal judge charges to the wrong notes.

    **Transfer (MTC, 3 seeds, validation-selected setting):**

    | train → test | AP lift | AUROC | gate precision | gated vs B3 | in-domain on that test set |
    |---|---|---|---|---|---|
    | D1 → D3 | 4.5× | 0.697 | 0.861 | **+7.25%** (7.30 / 6.95 / 7.51) | D3: +9.98% |
    | D3 → D1 | 4.0× | 0.649 | 0.778 | +4.80% (3.57 / 5.81 / 5.03) | D1: +6.35% |
    | D1 → D2 | 1.4× | 0.580 | 0.269 | **−1.71%** (−1.94 / −1.52 / −1.68) | D2: +0.00% |
    | D3 → D2 | 1.3× | 0.577 | 0.211 | **−2.01%** (−2.89 / −1.30 / −1.84) | D2: +0.00% |

    Read together:
    - **Detectability follows manifold distance.** The rule-free judge orders D1 > D3 >
      D2, matching ΔNLL/unit (3.94 > 2.97 > 1.16).
    - **Training adds a source-specific share, and D3's is larger.** The D1-trained head
      keeps 90% of D3's in-domain AP lift on D3; the D3-trained head keeps 69% of D1's
      on D1. The in-domain minus transferred gated gain — **2.7 points on D3**, 1.6 on
      D1 — is the part that is learned regularity rather than plausibility. That is the
      quantified form of the circularity warning.
    - **Mis-transferred heads do harm on D2** (M3).
    - **An obvious next method:** gate on the two-sided judge instead of, or with, the
      trained head. It needs no labels, which is what real drafts lack. Not yet tested.

    Caveats: each judge is a single trained model (heads are 3 seeds); both two-sided
    judges hit the 60-epoch cap still improving, as `J` did — longer training could only
    strengthen them; transfer was run on MTC only.

12. **Adding the label-free judge to the gate raises the gain by 1.4–2.1 points — at a
    small, consistent cost on generated drafts.** Plan step 5b
    (`scripts/judge_gate.py`). The decoder is held fixed (the α=0.5 edit-aware model from
    the headline checkpoints); only the gate changes. Each gate's threshold was chosen on
    D1 validation only, and the combination's weights (logistic on judge z and head
    logit) were fitted once, on D1 validation, and applied unchanged everywhere. The head
    gate at τ=0.8 reproduces the headline exactly on every cell, which validates the new
    gating path.

    | gated vs copy-input | head (headline, τ=0.8) | judge alone (z≥3) | judge alone, B1 decoder | **judge + head** |
    |---|---|---|---|---|
    | MTC D3 | +9.98% | +7.96% | +7.29% | **+11.80%** ± 0.38 |
    | MTC D1 | +6.35% | +2.81% | +2.43% | **+7.91%** ± 1.12 |
    | MTC D2 | +0.00% | −0.47% | −0.16% | −0.39% ± 0.07 |
    | Essen D3 | +7.30% | +6.25% | +6.17% | **+8.71%** ± 0.36 |
    | Essen D1 | +4.47% | +2.86% | +2.43% | **+6.52%** ± 0.34 |
    | Essen D2 | +0.00% | −1.17% | −1.14% | −0.46% ± 0.02 |

    Seed-matched, combination minus head: **+1.56 pp (MTC D1), +2.05 (Essen D1), +1.81
    (MTC D3), +1.42 (Essen D3), 3/3 seeds each**; −0.39 and −0.46 on D2, 0/3.

    - **The two signals are complementary.** The judge's score is standardized within
      each melody, so it always nominates a "most surprising" note and cannot abstain on
      a clean melody; the head's probability is absolute and can. Fitted weights are
      stable across seeds (≈0.92 per judge sd, ≈0.50 per head logit), and the judge
      carries the larger share.
    - **The judge alone is worse than the head** despite ranking better (contribution
      11): at the gate what matters is precision in the confident tail, and the judge's
      is lower (0.58 vs 0.81 on MTC D1).
    - **A pipeline with no edit labels anywhere** — judge gate, B1 decoder — beats
      copy-input on D1 and D3 in both corpora, 3/3 seeds (+2.4% D1, +6–7% D3). Modest,
      but it is the variant that applies to drafts nobody has labelled.
    - **The cost: the judge opens the gate on D2** (621 flags per seed, precision 0.44)
      and loses 0.4–0.5%, where the head alone abstains. Even at 0.44 precision the
      regenerated notes do not recover the reference. The combination buys ~1.5–2 points
      on off-manifold drafts with ~0.4 on generated ones; a deployment that cannot tell
      its drafts apart has to weigh that.

    **Decision (2026-09-25): the head gate is the paper's method; the combination is an
    extension.** The thesis is about where localize-then-fix applies, and the head's
    correct abstention on D2 is some of the cleanest evidence for it. So the headline
    stays head-gated (+6.35% MTC D1); judge + head is reported as "+1.4–2.1 points
    available where drafts are known to be off-manifold, at ~0.4% risk on generated
    drafts", and as the natural next step. Do not lead with +7.91%.

    Caveat — **validation optimism for judge-based gates.** Validation gains shrank on
    test for every judge-based condition (combination +9.84% → +7.91% / +6.52%; judge
    alone +5.33% → +2.8%) but not for the head (+6.09% → +6.37%). The two-sided judge was
    early-stopped on `judge_val`, the references the validation drafts are built from, so
    its scores are slightly better on validation than on test. Test numbers are clean
    (test was never used); the selected thresholds may be slightly off-optimum.

## 3. Framing moves

**M1 — The boundary is the paper, not the method.** Lead with the validator and the
orthogonality finding, then the negative result, then the positive one as the
demonstration that the diagnosis is actionable where the formulation applies. The +6% is
evidence that the map is right, not the headline.

**M2 — "Difference is not error."** Edit labels derived by aligning a draft to a
reference encode *difference*, not *error*. Whenever the draft is a valid alternative,
the supervision marks arbitrary divergence. This transfers well beyond music — any
reference-derived edit supervision in code or text has the same hazard.

**M3 — Abstention is selective prediction.** "The gate declines when no edit is
confidently warranted" is a *feature* for a creative tool, where a wrong edit is worse
than no edit. With the τ=1 guarantee, bounded downside plus abstention is a deployable
property and more honest than claiming uniform improvement.

*Qualified 2026-09-24:* abstention is only as good as the head's calibration on the
drafts it meets. Heads trained on D1 or D3 and applied to D2 **open** the gate (348 and
293 positions, precision 0.27 and 0.21) and make drafts **worse**: −1.71% and −2.01%
against copy-input, 3/3 seeds each. D2's own head correctly abstains. The τ=1 guarantee
is real, but at a working τ the no-harm property requires in-distribution training.

**M4 — Say plainly what the field should stop doing.** Two warnings, in order of how
easy they are to walk into. Rule-based degradation produces a partly circular benchmark:
the detector learns the injection rules, and a transfer test shows about a quarter of
the reported gain (2.7 of 10 points) comes from that. Generator-infilled drafts avoid it
but are barely localizable, so they are not a drop-in fix either. Anyone wanting realistic drafts needs a different task formulation
(holistic rewriting, span-free) or human-annotated error locations.

**M5 — Report the number you distrust.** D3 gives the paper's best result and is in the
git history; reporting D1 and D2 while dropping it would be exactly the selective
reporting reviewers look for. Include it, demote it, and explain why it should not be
believed. That is more persuasive than omission and costs nothing.

## 4. Claim discipline

Ten things the evidence does **not** support, each of which was claimed at some point
while this work was being done:

- **Not "where and how".** Binary matches 5-class (seed-matched +0.003 ± 0.011 on MTC,
  +0.012 ± 0.020 same-corpus, 2/3 seeds each way). Per-class F1: INSERT_BEFORE **0.000**,
  DELETE 0.107, SUB_DUR 0.194. Claim **"where"** — precision-gated *localization* — and
  report the typed head as an ablation that did not pay off. Recommend the binary head.
- **Not "edit-awareness lowers Correction Cost".** On D1 it does **not** reliably improve
  what free decoding produces (CC vs B1 at α=1.0 straddles zero, 2/3 seeds, sd 0.404
  CC). What it does supply is a usable localizer — head precision 0.65–0.69, localization
  F1 0.361 ± 0.005 — which gating then exploits. Do **not** present this as "improves
  where the model looks" against α=0: at α=0 the head receives no gradient, so that
  comparison is against a randomly initialized head (precision 0.06). Earlier versions
  quoted macro-F1 0.092 → 0.356; macro-F1 averages in KEEP's F1 of ~0.96 and should
  not be reported for localization.
- **Not "edit-awareness hurts on random corruption".** An artifact of inverse-frequency
  class weighting, which paid the head ~105× more for flagging than for leaving a note
  alone. Fixed with `weighting=none` on D1.
- **Not "span-level detection recovers what per-position missed".** It does not: 1.6×
  chance, barely above the untrained judge's 1.2×.
- **Not "contiguity is irrelevant", and not "D2 is explained by span length".**
  Contiguous errors are harder to rank (D3 AUROC 0.80 → 0.71), but D3's contiguous band
  still beats D2. *(Revised 2026-09-25; the earlier bullet claimed contiguity did not
  matter, from argmax precision.)*
- **Not "CC and plausibility are orthogonal".** Within a corruption source they
  correlate (ρ ≈ 0.5); only across sources do they decouple (ρ = −0.06).
- **Not "raising recall hurts because precision is binding".** Surprisal input improves
  localization with unchanged gate precision and still loses — the decoder degrades.
- **Not "generated drafts cannot be localized".** A zero-shot two-sided judge reaches
  2.7× chance on D2. Say "far less localizable" and give the numbers.
- **Not "the D3 benchmark is circular" without qualification.** Transfer puts the
  rule-learned share at ~2.7 of D3's ~10 points; the rest is genuine implausibility.
- **Not "gating cannot hurt".** Only at τ=1. Mis-transferred heads lose 1.7–2.0% on D2.

## 5. Title (closes P2)

Name the boundary:

- *Generated Drafts Are Not Locally Wrong: Limits of Localize-and-Fix in Symbolic Music Refinement*
- *When Is a Draft Locally Wrong? Mapping the Domain of Edit-Aware Music Refinement*
- *Difference Is Not Error: Synthetic Corruption, Generated Drafts, and What Can Be Localized*

Any of these disposes of the human-in-the-loop framing for free. Collaborative utility
gets one sentence as motivation, never as a demonstrated result.

## 6. Caveats to state openly, not bury

- **ΔNLL is statistical implausibility under a small LM, not a musical judgement.** The
  human-variant calibration is the strongest defence available without a user study and
  should be presented as a limitation, in the open.
- Two folk corpora only; the style claim is narrow and should say so.
- The negative result is **an absence of evidence from the methods tried**, not a proof
  of impossibility. State the three methods, the D1 control that rules out capacity, and
  the unfair advantage given to the span task — then let the reader judge.
- `J` is trained on Essen and scores MTC in the headline numbers (0.1365 nats/token
  domain gap). The same-corpus control rules out the confound: both halves replicate,
  and D2 localization is equally weak on both corpora (AP lift 1.6× MTC, 1.8× Essen;
  AUROC 0.635 and 0.654). An earlier version claimed the dissociation was *stronger* same-corpus
  (0.220 → 0.137); the 0.220 came from a single-seed surprisal-input run and the claim
  is withdrawn.

## 7. Anticipated reviewer objections

| objection | answer |
|---|---|
| "Your method only works on the synthetic setting you criticise." | Yes — and that is the finding, stated as the title. The boundary is mapped from both sides, and the negative side is measured three independent ways. |
| "Your detector is just weak / your models are too small." | Three answers. The same architecture, size, recipe and setting reaches AP lift 5.8× on D1 against 1.6× on D2 — the difference is the drafts. Localization precision is flat from 1.6M to 42.7M parameters (corr with log-params −0.58), so it is information-limited, not capacity-limited. And the span task was given the true span-length prior and still failed. |
| "Your per-position head just can't represent multi-note edits." | Partly: within D3, AUROC falls from 0.80 (single notes) to 0.71 (runs of 3+). But D3's contiguous band still beats D2 (0.71 vs 0.64), and gating — which uses only the confident tail — is insensitive to run length. *(Revised 2026-09-25; the earlier answer follows.)* |
| *(earlier answer)* | Tested within D3, holding manifold distance roughly fixed: precision 0.825 point-like against 0.805 span-like (+0.019). D3's span-like band is contiguous exactly as D2's infills are and still scores 0.805, while D2's head under a matched setting flags nothing. Granularity was also varied directly, in the span detector, with the same null. |
| "Maybe the drafts aren't degraded, so there is nothing to find." | Partly right, and we say so: the two-sided judge localizes the 2.5–4.4% of generated drafts that are certified-degraded at AUROC 0.78, near D1. The rest are mostly difference, not error. Our trained head does not exploit this (flat across bands, 3 seeds). |
| "A better detector would find D2's errors." | We built the strongest one available without labels — a two-sided judge seeing both sides of every note — and it reaches 2.7× chance on D2 against 6–7× on corruption. It is also better than every trained head on every source, so D2's gap is not for want of a stronger detector. |
| "D3 is circular, so why report it?" | It is partly circular, and we measured how much: a D1-trained head reaches +7.25% on D3 against D3's own +9.98%. |
| "+6% is small." | It is over *copy-input*. The published +6.3% was over an imitation baseline itself 21.4% worse than copying. |
| "Your own table shows CC tracks plausibility (ρ ≈ 0.5)." | Within one corruption process, yes — more injected edits, more damage. Across sources it vanishes (ρ = −0.06), and the most distant drafts are the most plausible. CC ranks methods on one source; it cannot compare sources or tell error from alternative. |
| "ΔNLL isn't musical correctness." | Conceded explicitly; calibrated against human variants; see §6. |
| "Thin evaluation." | 3 seeds with mean ± std, B1/B2/B3, fidelity, edit-F1, paired stats with bootstrap CI and Wilcoxon, same-corpus control, size ablation. |

## 8. Rejection reasons → status

| | status |
|---|---|
| P1 random corruption ≠ revision | **answered** — validator, orthogonality, and the boundary result |
| P2 human-in-the-loop framing | **retitle** per §5; no experiments needed |
| P3 "where and how" | **answered by narrowing** — claim "where" (§4) |
| P4 representation contradictions | **fixed** — REST/BAR, measured; onset shift removed from the vocabulary |
| P5 thin evaluation | **answered** — §7 final row, plus 9 size configurations × 3 seeds |
| P6 no citations in introduction | **outstanding** — pure writing, and an explicit rejection reason |

## 9. Revision plan

Worked one step at a time, each reviewed before the next. Steps 2–8 may change numbers
quoted above; update §1–§7 as each lands. Target venue is no longer ISMIR and the page
budget is 8 pages, so the contribution list need not be cut to three — but it should
still be ordered by what carries the thesis.

1. **Reconcile reported numbers against result files.** *Done 2026-09-23.* Docs only,
   no reruns. Fixed: the thesis/source table mixed MTC and Essen runs and 5-class and
   binary heads; D3 +9.85% was the τ=0.7 result; the same-corpus "stronger
   dissociation" rested on a surprisal-input run (withdrawn); the α-curve "5.85%" was
   the size-ablation mean; D2 ΔNLL/unit was the wrong temperature; D4's fraction
   positive came from a different file; the "where the model looks" comparison was
   against an untrained head. Two facts confirmed and still to be stated in the paper:
   **D2 training and evaluation data are not judge-filtered** (spec §3.2 made the filter
   mandatory), and **the class weighting was selected on the test set**.

2. ~~**Remove test-set selection**~~ *Done 2026-09-24.* Validation picks none / 0.5 /
   0.8; the old test-selected none / 0.5 / 0.9 ranks 2nd of 96, 0.07 pp behind, so the
   earlier selection did not inflate results. Headline re-run twice (`headline/`,
   `headline_r2/`; the second saves checkpoints, P(edit) and outputs). Found on the way:
   runs are not bit-reproducible — consider `torch.use_deterministic_algorithms` for
   anything run from here on. Original description: `weighting_ablation.py` picks the weighting
   by test macro-F1, and α=0.5 and τ=0.9 were chosen after seeing test results. Select
   all three on the validation split, by localization precision for τ and the gated CC
   for α, then rerun the headline gated evaluation (D1, D2, D3; MTC and same-corpus).
   Comes first because every later step inherits these settings.

3. ~~**D2 with a binary head**~~ *Done 2026-09-24, inside step 2's headline.* The D2 head
   collapses to KEEP under the global setting; localization is now compared
   threshold-free (`scripts/localization_report.py`).

4. ~~**Musical quality of outputs.**~~ *Done 2026-09-24 — contribution 10, `quality.json`.* Judge NLL, scale conformance and interval distance
   for B3 / B1 / free / gated / oracle outputs on D1, D2, D3. Needed because
   contribution 2 argues CC cannot carry an evaluation, while the positive result is
   currently CC-only. Check whether refiner checkpoints were kept; if not, rerun.

5. ~~**Measure circularity rather than assert it.**~~ *Done 2026-09-24 — contribution
   11; extended with a two-sided judge (`src/masked_judge.py`).* Original: (a) Zero-shot judge localization on
   D1 and D3 — the judge never saw either corruption rule. (b) Transfer: train on D1,
   test on D3, and the reverse. If D3's advantage disappears under transfer while the
   zero-shot judge does not favour it, the gap is the learnable injection regularity,
   quantified. This is the test the "learnable regularity" hypothesis currently lacks.

5b. ~~**Judge-gated editing.**~~ *Done 2026-09-25 — contribution 12.* Original: Gate
   on the untrained two-sided judge's per-note score, alone and combined with the
   trained head. It localizes better than every head (contribution 11) and needs no
   labels. Select its threshold on validation exactly as in step 2. If it beats the
   head-gated +6.35% on D1 — or opens a gain on D2 — it becomes the method.

6. ~~**Strengthen thin controls.**~~ *Done 2026-09-25 — contributions 4, 5 and 9 revised;
   `scripts/control_strata.py`, `results/refiner/controls/`.* Original: Degradation strata: 3 seeds, and enough drafts that
   the degraded band is not n=28. Locality strata: 3 seeds. Surprisal/recall
   comparison (contribution 9): 3 seeds.

7. ~~**Decoupling statistic.**~~ *Done 2026-09-25 — contribution 2 revised;
   `results/judge/decoupling.json`.* Original: "Near-orthogonal" currently rests on two clusters, not a
   correlation. Compute rank correlation between CC and ΔNLL within and across
   sources, or retreat to "decoupled".

8. ~~**5 seeds on the headline**~~ *Done 2026-09-25, head gate only:* D1 +6.06% ± 0.58, 5/5 seeds (was +6.35% on 3); Essen +4.25%; D3 +9.81% / +7.38%; D2 abstains on all 5. (Original: seed 0 was an outlier once.)

9. ~~**Framing rewrite.**~~ *Done 2026-09-25 — superseded by `paper-core.md`, which is now
   the file to draft from; this file stays as the revision log.* Original: The three-point "trend" is not monotone on the paper's own
   realism measure (ΔNLL/unit orders D1 > D3 > D2; precision orders D3 > D1 > D2), so
   present D1-vs-D2 as the contrast and D3 as the regularity control. "Not locally
   wrong" conflicts with the D2 oracle (+21.5% lift given locations) — use "leave no
   local trace" / "not locally detectable". State D2 is unfiltered and why. State the
   spec items deliberately dropped: data-size ablation, mixed-source training, D3→D4
   transfer, weighted CC, per-operation CC breakdown, and scale-matched B2 (the head is
   1,285 parameters, 0.02%).

10. **Related work (P6).** McLeod et al. (MDT, ISMIR 2020) has its own degradation-
    *detection* task, so the D3 circularity point applies to it directly — that is the
    stated delta, worded carefully. RefinPaint (Ramoneda et al., ISMIR 2024), Ji, Yang &
    Luo (2023), Lakh provenance, Vaswani et al.

11. *(Parallel, optional)* `scripts/` have no test coverage — that is how a broken-
    argparse bug reached a launch.
