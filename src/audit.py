import json
from src.train import create_dataloaders

def convert_to_lists(obj):
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