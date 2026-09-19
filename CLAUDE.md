# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research pipeline for training a transformer to correct/complete corrupted monophonic melodies. Melodies are randomly "corrupted" with edit operations (pitch replace, duration change, note delete/insert), and a `MelodyTransformer` is trained to reconstruct the original from the corrupted version. Two training modes are compared: a **baseline** (plain cross-entropy over the whole sequence) and an **edit-aware** variant (loss upweighted at the positions that were actually edited), controlled by an `alpha` weighting factor.

## Rebuild in progress

The paper this implements was rejected at ISMIR 2026 on execution, and `edit-aware-revision-spec.md` (repo root) is the governing plan for a rebuild. **The spec supersedes `src/train.py`'s design** — treat `src/train.py` as the legacy pipeline being replaced piece by piece, not as the target architecture.

Agreed build order: aligner → tokenizer/representation → judge `J` → validator report → generator `G` → training sweep. The validator (spec §4) comes before any edit-aware model because it is the only component that can falsify the rebuild's premise.

Two corpus/layout decisions already settled, which the spec does not record:
- **Corpus moved to Meertens Tune Collections (primary) + Essen Folksong Collection (cross-corpus).** MTC's *tune family* annotations supply the spec's D4 human-variant set in-corpus, so D4 stopped being an external dependency.
- **Code lives under `src/`, artifacts stay at root.** The spec's proposed `data/` and `models/` (for loaders and model code) collide with the existing gitignored artifact directories of the same names.

Done so far: `src/align.py`, `src/tokenizer.py`, `src/corpus.py`, `src/model.py`, `src/data.py`, `src/judge.py`, `src/drafts.py`, `src/training.py`, `src/sampling.py`, and their tests (141, stdlib `unittest`). Nothing in `src/train.py` or `main.py` has been migrated onto them yet — see "Migration status" below.

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

### Getting the data

Corpora are **not committed** — `data/` is gitignored and neither collection permits redistribution.

```bash
python -m src.corpus fetch          # MTC-ANN + ESSEN from Zenodo, MD5-checked
python -m src.corpus                # print the dataset table

python scripts/train_judge.py       # train J + run its calibration controls
python scripts/validator_report.py  # the D1/D3/D4 degradation table
python scripts/train_generator.py   # train G (+ undertrained checkpoints) for D2
python scripts/d2_report.py         # where D2 sits vs D1/D3/D4
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
