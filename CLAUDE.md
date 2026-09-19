# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research pipeline for training a transformer to correct/complete corrupted monophonic melodies. Melodies are randomly "corrupted" with edit operations (pitch replace, duration change, note delete/insert), and a `MelodyTransformer` is trained to reconstruct the original from the corrupted version. Two training modes are compared: a **baseline** (plain cross-entropy over the whole sequence) and an **edit-aware** variant (loss upweighted at the positions that were actually edited), controlled by an `alpha` weighting factor.

## Rebuild in progress

The paper this implements was rejected at ISMIR 2026 on execution, and `edit-aware-revision-spec.md` (repo root) is the governing plan for a rebuild. **The spec supersedes `src/train.py`'s design** — treat `src/train.py` as the legacy pipeline being replaced piece by piece, not as the target architecture.

Agreed build order: aligner → tokenizer/representation → judge `J` → validator report → generator `G` → training sweep. The validator (spec §4) comes before any edit-aware model because it is the only component that can falsify the rebuild's premise.

Two corpus/layout decisions already settled, which the spec does not record:
- **Corpus moves to Meertens Tune Collections (primary) + Essen Folksong Collection (cross-corpus).** Not yet downloaded — `data/` still holds only the legacy 4-bar arrays. MTC's *tune family* annotations supply the spec's D4 human-variant set in-corpus, so D4 stops being an external dependency.
- **Code lives under `src/`, artifacts stay at root.** The spec's proposed `data/` and `models/` (for loaders and model code) collide with the existing gitignored artifact directories of the same names.

Done so far: `src/align.py`, `src/tokenizer.py`, and their tests. Nothing in `src/train.py` or `main.py` has been migrated onto them yet — see "Migration status" below.

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
- **`src/tokenizer.py`** — `MelodyTokenizer` with explicit REST and BAR tokens. `frames_to_events()` replaces `pitch_seq_to_notes()`; `encode_with_spans()` returns each event's token indices and is what downstream code must use to place masks or labels.

Three design decisions in these modules that the spec leaves open, resolved here and needing to be stated in the paper:

1. **Bars are token-level structure, not events.** The note sequence handed to the aligner stays `(pitch, duration)`, so edit scripts never contain bar operations (asking a composer to "insert a barline" is not an edit). Bar positions come from the source metre, not cumulative duration, which is what actually bounds the onset cascade.
2. **A note differing in both pitch and duration has no 5-class label.** The full script keeps both ops; the single-position label falls back to `SUB_PITCH` (configurable via `both_differ`). When several ops land on one position, first-in-canonical-order wins.
3. **Trailing insertions attach to an EOS slot.** `edit_labels()` returns `len(source) + 1` labels — insertions past the end of the source have no input position otherwise, and the tokenizer emits EOS anyway.

Measured on 20k training melodies, justifying the representation change:

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
