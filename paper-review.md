# Review of paper-draft.md (2026-09-26)

Scope: logic first, then comprehensiveness, flow, voice. Issues are ranked by how much
damage a reviewer could do with them. Numbers checked against `paper-core.md` and the
result JSONs.

---

## A. Logic

### A1. "Most generated drafts are not wrong" is not supported by the τ statistic — **critical** — ✅ done 2026-09-27

The thesis sentence (Abstract, §1, §7, §9) rests on "only 2.5–4.4% of D2 drafts exceed τ".
But by the same criterion **87% of D1 drafts and 90% of D3 drafts are also "not measurably
worse"** (Table 2: 13.0% and 9.6% above τ). The paper calls those errors. So exceeding τ
cannot be the test for "wrong", and the argument does not separate D2 from corruption.

The reason: τ is the 95th percentile of *human variants that differ in 92% of their
notes*. ΔNLL is a per-token mean, so a draft that changes 7–10% of notes is diluted
roughly tenfold. The comparison with τ is biased toward "not degraded" for every
synthetic source.

What the paper's own data actually say about D2:
- 79.5% of D2 drafts are *worse* than their reference (ΔNLL > 0), against 55% for human
  pairs; the mean ΔNLL is 2.6× the human mean.
- Per unit of change, D2 is **~27× more implausible than human variation** (1.04 vs 0.04)
  and only 3–4× less than corruption. On a ratio scale D2 sits nearer corruption than
  humans. (Figure 1's caption says "nearer the human variants", which only holds on its
  linear axis. That sentence is mine; it needs the same fix.)

The defensible claim is the earlier one in `paper-framing.md`: **generated drafts are
measurably worse on average, but the damage is diffuse, with little local trace**. That
matches every localization number and doesn't need τ to carry weight. Rewrite
"different, not wrong" as "rarely worse by more than human variation, and not locally
so". Keep the Table 6 gradient, but as evidence that local traces appear only when damage
is concentrated, not as evidence that the rest contain no error.

### A2. Table 6 needs a D1 control, or it can't discriminate — **high** — ✅ done: D2 below D1 in every band, both corpora (`matched_strata.json`)

Localization rising with ΔNLL band is close to guaranteed for any surprisal-based
localizer: high-ΔNLL drafts contain high-surprisal notes by definition. The draft
concedes part of this ("built in") in a parenthesis. The test that would make Table 6
informative is to **band D1 the same way**:
- If D1's low-ΔNLL drafts are still localized at ~0.8 while D2's sit at ~0.6, the
  difference is the *kind* of change, not its amount. That supports the "local trace"
  framing directly.
- If D1 also falls to ~0.6 in its low bands, the gradient is generic and says nothing
  specific about generated drafts.

This is cheap: `scripts/control_strata.py` from saved artifacts, no training.

### A3. The transfer test is symmetric, so it does not isolate rule-learning — **high** — ✅ reframed

D1→D3 loses 2.73 of 9.98 points (27%). D3→D1 loses 1.55 of 6.35 (**24%**). By the
paper's own reasoning, D1's benchmark is then "inflated by a quarter" too. The transfer
gap measures **source-specific learning in general** (domain shift), not D3's rules in
particular. That undercuts contribution 2's "measures how much of a rule-based
benchmark's score comes from learning its own rules", the last sentence of the Abstract,
§6's "D3 is easier, and partly for a bad reason", and §8's third recommendation.

Two ways out:
- **Reframe (recommended):** "every synthetic benchmark's in-domain gain includes ~25%
  that does not transfer, whether the corruption is ruled or random". That is a more
  general finding, and it still justifies the transfer-test recommendation.
- **Keep the D3 claim:** you'd need an asymmetry argument, e.g. normalizing by how
  different the sources are. With 24% vs 27% I don't think one exists.

### A4. "The only system that improves both CC and plausibility" is false on D3 — **high** — ✅ done

Table 4: B1 free decoding on D3 gives CC **+4.3%** and ΔNLL recovered **+48.8%**. Both
positive. §6's closing sentence ("Gated editing is the only system that improves on the
draft on both") and the Abstract's sentence (whose scope is ambiguous) must be limited to
D1.

### A5. The oracle is not "the same decoder", and 49% is not "most" — **medium** — ✅ done

`scripts/diagnostics.py`'s oracle is a *separately trained* model with ground-truth
labels fed at the encoder input. It was also trained under the pre-selection
configuration. §6 says "lets the same decoder rewrite exactly those" and "the same
decoder's replacements remove most of the error". Say "a model of the same architecture,
given the true positions as input", and "about half". The ~12% headroom figure mixes
that single oracle run with the 5-seed gated number; say so, or use paper-core's ~13%
consistently.

### A6. The judge is not independent of the refiner — **medium** — ✅ done

§4 argues carefully that `J` must not share training data with the generator. But the
refiners are trained on drafts of **the judge's half** of Essen (§3). The decoder learns
the same distribution `J` models, so `J` is predisposed to rate refiner output as
plausible. That especially affects the free-decoding claim (14.5% of outputs "more
plausible than the reference") and Table 4 generally. At minimum, state it in §8
Limitations. Better: note that the gated edits are few, so they are less exposed than
free decoding. The confound then *favours* free decoding, and the "gated improves both"
conclusion survives it.

### A7. "Abstention is the correct behaviour" overclaims — **medium** — ✅ softened to "safe", tied to the alternative gates; the D2 τ sweep is folded into B2

The gate stays shut on D2 because the D2-trained head's confidence never reaches 0.8.
That is the designed safety behaviour, not a judgement that nothing could be fixed: the
oracle removes 21.5% of D2's CC given true positions. §1's "the correct behaviour rather
than a failure" and §9's "the right behaviour" should become "the safe behaviour". To
defend "right", add the evidence: **show that no threshold beats copy-input on D2**
(sweep τ on D2 validation, or plot risk–coverage). Without it, a reviewer can say the
selection just happened to land above D2's confidence range.

### A8. The §9 closing contradicts §8 — **medium** — ✅ done

§9 calls the judge + head gate "a first step" toward refining generated music. §8 reports
that it *loses* 0.4–0.5% on generated drafts, and the judge-only gate loses 0.5–1.2%.
Either drop the sentence or say what it is a first step toward: label-free refinement of
*locally damaged* drafts.

### A9. "Drafts furthest from their references are the most plausible" names the wrong thing — **low** — ✅ done

The pooled ρ = −0.06 is driven by D4, which §3 says are *not* drafts. It's a between-group
composition effect. §1 and §4 should say "the melodies furthest from their references
— human variants — are the most plausible". Also be wary of the "80–100×" ratio: its
denominator is a median near zero (0.038, with 45% of pairs negative), so the ratio is
fragile. "Two orders of magnitude" is safer, or report the difference.

### A10. Smaller logic points — ✅ done

- **Model-size ablation (§7)** was run on D1, under τ = 0.9. It shows capacity doesn't
  limit D1 localization; for D2 it is only suggestive. Say so, or cut it from the D2
  argument.
- **Contiguity (§7)** compares D3 runs of 3+ against *all* of D2 (spans 2–6). Match the
  run lengths, or note the mismatch.
- **"Uses no edit labels anywhere" (§8)**: the z ≥ 3 threshold was selected on D1
  validation CC, which needs references. Say "no edit labels in training".
- **Intro "the trained detector finds almost nothing"**: AUROC 0.636 / 1.6× is weak, not
  nothing. §7's phrasing ("finds nothing worth acting on") is the accurate one; use it in
  §1 too.
- **2.6% (§4) vs 2.5–4.4% (§1, §7)** come from different draft sets. A reader will see a
  contradiction; name the sets once.
- **τ is calibrated on MTC, the test corpus.** It's used only for analysis bands, never
  for selection. One sentence in §4 saying so pre-empts the question.

---

## B. Comprehensiveness

1. ✅ **Done 2026-09-27** — six generator settings; it changed the thesis (generator quality), and the ep2 refiner follow-up found the repair gap. **Turn three points into a curve (highest value).** "Three points cannot establish"
   manifold distance is the draft's own caveat, and the data to fix it exist. D2 at
   T = 0.8 and the undertrained ep2 checkpoint sit at 0.90 and 2.0 ΔNLL/unit, filling the
   gap between D2 and D3. Zero-shot two-sided-judge localization on those drafts needs no
   training. Five or six points on Figure 1(a) would turn a suggestive ordering into a
   trend.
2. ✅ **Done** (`scripts/risk_coverage.py`; reported as a §6 paragraph, not a figure). **Risk–coverage curve for the gate.** The paper leans on selective prediction but
   shows a single operating point. A curve of CC gain against τ for D1/D3/D2 is the
   standard figure for that framing, and it also settles A7.
3. **A qualitative example.** A music venue will expect one melody shown as reference,
   D1, D2 and D3 drafts, and gated output, with the gate's flags. Nothing in the draft
   shows actual music.
4. **Precision-oriented evaluation in GEC.** GEC standardized on F0.5 because a false
   correction costs more than a missed one, which is exactly the finding that precision is
   the binding constraint. Cite the M² scorer (Dahlmeier & Ng, NAACL 2012). Verify the
   entry before adding it, as with the other references.
5. **Missing detail a reproducer needs:** generator size and validation NLL, D3's rule
   proportions, training-set draft counts per source, and how many drafts per reference
   are used in training (it's stated for test only).

---

## C. Flow — ✅ C1–C5 done 2026-09-27 (C1 was already a pointer after the transfer reframe)

1. **Transfer appears twice.** +7.25 vs +9.98 is in both §6 ("D3 is easier…") and §7
   ("What is detectable…"). Keep one full statement in §7 and a forward pointer in §6.
2. **§8 "Where the judge belongs" is a results section.** It introduces about a dozen new
   numbers with no table. Move it to the end of §7 as "Using the judge", with a small
   table (head / judge / judge + head × D1, D3, D2), and keep only the interpretation in
   §8.
3. **Same-corpus replication is announced three times** (§3, §4, §7 "Confounds"), and the
   Confounds paragraph sits after the argument it protects. Move it to the start of §6,
   directly after Table 3, where the Essen columns first appear.
4. **§6 teaches the copy-collapse lesson before the reader needs it.** The "+6.3% over B1
   while copying" paragraph breaks the run from "gated works" to "free decoding hurts" to
   "plausibility". It fits better in §5 under Baselines, as the reason B3 and fidelity
   exist.
5. **The Intro's second bold paragraph carries the whole §7 argument in one breath** (five
   numbers, three detector types, the degraded-band result). After fixing A1, cut it to
   the claim plus one number, and let Figure 1 carry the rest.

---

## D. Voice and consistency

- **Title vs body:** the title says "Localize-and-Fix"; the body says "localize-then-fix"
  throughout. Pick one.
- **Hedging is uneven:** careful in §7 ("consistent with, though three points cannot
  establish"), assertive in §1 and §9 ("the reason is…", "the right behaviour"). The
  Intro and Conclusion should be no stronger than §7.
- **Bold paragraph leads:** effective, but there are 20+. Keep them in §4, §6 and §7;
  drop them in §5 and §8, where they fragment prose that is already short.
- **Awkward sentences:** "the few that are are found" (§9). The Abstract's second
  sentence nests a participial phrase inside a long subject ("Using a melody language
  model, calibrated against …, as a plausibility judge, we show …"); split it.
- **Mixed seed counts in one paragraph:** §6 ¶1 pairs 5-seed CC with 3-seed win/loss
  (paper-core has 23.0% / 6.3% for seeds 0–2; the draft's 22.4% / 6.5% needs a stated
  source) and 3-seed gate precision. State the seed count once per paragraph.
- **Drafting scaffolding** ("*Budget …*", the "*(proposed)*" table tags, the status
  header) must be stripped before conversion. Not an issue yet, just don't forget it.

---

## Suggested order of work

1. A1 + A2 together: rewrite the thesis sentence to "not locally wrong", and run the D1
   banding control to back it.
2. A3: reframe transfer as a general benchmark effect (Abstract, §1 contribution 2, §6,
   §7, §8).
3. A4–A8: sentence-level fixes, about an hour.
4. B1 and B2: two cheap analyses that strengthen the figures.
5. Flow moves (C1–C4), then the length trim, which the moves will partly accomplish.
