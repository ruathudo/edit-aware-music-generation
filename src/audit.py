import json
import random
import torch
from src.train import (
    create_dataloaders,
    create_model_and_optimizer,
    tokens_to_notes,
    minimal_edit_distance,
    id_to_token,
    device,
)


def _load_model(model_name, model_path, d_model=256, n_heads=4, n_layers=4):
    """Load a trained model from disk."""
    model, _ = create_model_and_optimizer(
        model_name=model_name,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    state = torch.load(f"models/{model_path}.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _tokens_to_readable(token_list):
    """Convert a flat token id list to a list of (pitch, duration) note tuples."""
    return tokens_to_notes(token_list, id_to_token)


def analyze(alpha=2, d_model=256, n_heads=4, n_layers=4, seed=42):
    """
    Load trained baseline and edit-aware models, draw one random batch of 5
    samples from the test dataloader and run inference for both models.

    Prints, for each sample:
      - corrupted tokens (model input)
      - edit script
      - target (clean) tokens
      - baseline predicted tokens
      - edit-aware predicted tokens
    """
    # ---- dataloaders ----
    test_dataloader = create_dataloaders("test", batch_size=5)

    # Pick a random batch
    # rng = random.Random(seed)
    all_batches = list(test_dataloader)
    batch = random.choice(all_batches)
    corrupted, clean, edit_masks, edit_scripts = batch

    # ---- models ----
    baseline_model = _load_model("baseline", "baseline", d_model, n_heads, n_layers)
    edit_aware_path = f"edit_aware_alpha_{alpha}"
    edit_aware_model = _load_model("edit_aware", edit_aware_path, d_model, n_heads, n_layers)

    # ---- inference ----
    corrupted = corrupted.to(device)
    inputs = corrupted[:, :-1]

    with torch.no_grad():
        baseline_logits, _ = baseline_model(inputs)
        edit_aware_logits, _ = edit_aware_model(inputs)

    baseline_preds = torch.argmax(baseline_logits, dim=-1).cpu().tolist()
    edit_aware_preds = torch.argmax(edit_aware_logits, dim=-1).cpu().tolist()
    corrupted_tokens = corrupted.cpu().tolist()
    clean_tokens = clean.tolist()

    # ---- collect results ----
    results = []
    separator = "=" * 60
    for i in range(len(corrupted_tokens)):
        print(separator)
        print(f"Sample {i + 1}")
        print(separator)

        corr_notes = _tokens_to_readable(corrupted_tokens[i])
        target_notes = _tokens_to_readable(clean_tokens[i])
        baseline_notes = _tokens_to_readable(baseline_preds[i])
        edit_aware_notes = _tokens_to_readable(edit_aware_preds[i])

        baseline_cost, _ = minimal_edit_distance(baseline_notes, target_notes)
        edit_aware_cost, _ = minimal_edit_distance(edit_aware_notes, target_notes)

        print(f"  Edit script       : {list(edit_scripts[i])}")
        print(f"  Corrupted         : {corr_notes}")
        print(f"  Target (clean)    : {target_notes}")
        print(f"  Baseline pred     : {baseline_notes}")
        print(f"  Baseline cost     : {baseline_cost}")
        print(f"  EditAware pred    : {edit_aware_notes}")
        print(f"  EditAware cost    : {edit_aware_cost}")

        results.append({
            "sample": i + 1,
            "edit_script": [list(e) for e in edit_scripts[i]],
            "corrupted": corr_notes,
            "target": target_notes,
            "baseline_pred": baseline_notes,
            "baseline_edit_cost": baseline_cost,
            "edit_aware_pred": edit_aware_notes,
            "edit_aware_edit_cost": edit_aware_cost,
        })

    print(separator)

    # ---- save to JSON ----
    output_path = f"results/samples/analyze_alpha_{alpha}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

def _draw_piano_roll(ax, notes, title, highlight_indices=None):
    """Draw a piano roll for a list of [pitch, duration] notes on a given axis."""
    import matplotlib.patches as mpatches

    x = 0
    for idx, (pitch, duration) in enumerate(notes):
        color = "tomato" if highlight_indices and idx in highlight_indices else "steelblue"
        rect = mpatches.FancyBboxPatch(
            (x, pitch - 0.4), duration, 0.8,
            boxstyle="round,pad=0.05",
            linewidth=0.5,
            edgecolor="black",
            facecolor=color,
        )
        ax.add_patch(rect)
        x += duration

    if notes:
        pitches = [p for p, _ in notes]
        total_dur = sum(d for _, d in notes)
        ax.set_xlim(-0.5, total_dur + 0.5)
        ax.set_ylim(min(pitches) - 3, max(pitches) + 3)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(20, 110)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Beats")
    ax.set_ylabel("MIDI Pitch")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def visual_result(json_path, sample_index=0):
    """
    Visualize one sample from an analyzed JSON file as a 4-panel piano-roll image.

    Parameters
    ----------
    json_path   : path to the analyzed JSON file produced by `analyze`
    sample_index: 0-based index of the sample to visualize
    """
    import os
    import matplotlib.pyplot as plt

    with open(json_path, "r") as f:
        results = json.load(f)

    sample = results[sample_index]

    panels = [
        ("Corrupted (Input)",                                         sample["corrupted"]),
        ("Target (Clean)",                                            sample["target"]),
        (f"Baseline Pred  [edit cost: {sample['baseline_edit_cost']}]",  sample["baseline_pred"]),
        (f"Edit-Aware Pred [edit cost: {sample['edit_aware_edit_cost']}]", sample["edit_aware_pred"]),
    ]

    edit_script_str = str(sample["edit_script"])
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    fig.suptitle(
        f"Sample {sample['sample']}  |  Edit script: {edit_script_str}",
        fontsize=8,
        wrap=True,
    )

    for ax, (title, notes) in zip(axes, panels):
        _draw_piano_roll(ax, notes, title)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_dir = os.path.dirname(json_path)
    output_path = os.path.join(output_dir, f"{base_name}_sample{sample['sample']}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Score image saved to {output_path}")
    return output_path



    """Recursively convert tuples to lists for JSON serialization."""
    if isinstance(obj, tuple):
        return [convert_to_lists(item) for item in obj]
    elif isinstance(obj, list):
        return [convert_to_lists(item) for item in obj]
    else:
        return obj

def validate_val_dataloader_consistency():
    """Validate that val_dataloader produces identical samples across sessions for the same epoch."""
    output_file = "val_samples_epoch_0.json"

    # Create val dataloader
    val_dataloader = create_dataloaders("val", batch_size=1)

    # Set epoch to 0
    val_dataloader.dataset.set_epoch(0)

    samples = []
    for i, (corrupted, clean, edit_mask, edit_script) in enumerate(val_dataloader):
        if i >= 5:  # Check first 5 samples
            break
        samples.append({
            'idx': i,
            'corrupted': corrupted.tolist(),
            'clean': clean.tolist(),
            'edit_script': convert_to_lists(edit_script[0])  # convert all tuples to lists
        })

    # Check if file exists (from previous session)
    try:
        with open(output_file, 'r') as f:
            saved_samples = json.load(f)
        print("Loaded saved samples from previous session.")

        # Compare
        all_match = True
        for i, (new_sample, saved_sample) in enumerate(zip(samples, saved_samples)):
            if new_sample != saved_sample:
                print(f"Sample {i} does not match!")
                print(f"New: {new_sample}")
                print(f"Saved: {saved_sample}")
                all_match = False
            else:
                print(f"Sample {i} matches.")

        if all_match:
            print("All samples match across sessions!")
        else:
            print("Samples do not match across sessions.")

    except FileNotFoundError:
        # Save samples for next session
        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"Saved {len(samples)} samples to {output_file} for validation in next session.")

if __name__ == "__main__":
    validate_val_dataloader_consistency()