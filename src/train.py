"""
Training module for the Music Edit Learning project.
Refactored from training.ipynb with minimized global variables.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import json


# ============ Constants at top of file ============
BASE_SEED = 42
PITCH_MIN = 21
PITCH_MAX = 108
DURATIONS = [1, 2, 3, 4, 6, 8]
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>"]
EDIT_TYPES = ["replace_pitch", "change_duration", "delete_note", "insert_note"]
INSERT_COST = 1
DELETE_COST = 1

# Set random seeds for reproducibility
torch.manual_seed(BASE_SEED)
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)

# Device initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
test_path = "data/test_data.npy"
train_path = "data/train_data.npy"
val_path = "data/val_data.npy"

# Build vocabulary
pitch_tokens = [f"P{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
duration_tokens = [f"D{d}" for d in DURATIONS]
vocab = SPECIAL_TOKENS + pitch_tokens + duration_tokens
token_to_id = {tok: i for i, tok in enumerate(vocab)}
id_to_token = {i: tok for tok, i in token_to_id.items()}
PAD_ID = token_to_id["<PAD>"]
BOS_ID = token_to_id["<BOS>"]
EOS_ID = token_to_id["<EOS>"]


# ============ Parsing and Data Processing ============
# def parse_seq_example(example_proto):
#     """Parse TFRecord example into pitch sequence."""
#     context_features = {}
#     sequence_features = {
#         "pitch_seq": tf.io.VarLenFeature(dtype=tf.int64),
#     }
#     _, sequence = tf.io.parse_single_sequence_example(
#         example_proto,
#         context_features=context_features,
#         sequence_features=sequence_features
#     )
#     pitch_seq = tf.sparse.to_dense(sequence["pitch_seq"])
#     pitch_seq = tf.reshape(pitch_seq, [-1])
#     return pitch_seq


def pitch_seq_to_notes(pitch_seq):
    """Convert pitch sequence to list of (pitch, duration) notes."""
    notes = []
    i = 0
    while i < len(pitch_seq):
        token = pitch_seq[i]
        if token == 128:  # hold token
            i += 1
            continue
        if token == 129:  # rest token
            i += 1
            continue
        
        pitch = int(token)
        duration = 1
        j = i + 1
        while j < len(pitch_seq) and pitch_seq[j] == 128:
            duration += 1
            j += 1
        notes.append((pitch, duration))
        i = j
    return notes


def melody_is_valid(notes):
    """Check if all note durations are valid."""
    return all(d in DURATIONS for _, d in notes)


def notes_to_tokens(notes):
    """Convert notes to token sequence."""
    tokens = [BOS_ID]

    for pitch, dur in notes:
        tokens.append(token_to_id[f"P{pitch}"])
        tokens.append(token_to_id[f"D{dur}"])

    tokens.append(EOS_ID)

    return tokens


def tokens_to_notes(tokens, id_to_token_dict):
    """Convert token sequence back to notes."""
    notes = []
    i = 0
    while i < len(tokens):
        tok = id_to_token_dict[tokens[i]]
        if tok == "<BOS>" or tok == "<EOS>" or tok == "<PAD>":
            i += 1
            continue
        if tok.startswith("P"):
            pitch = int(tok[1:])
            if i + 1 < len(tokens):
                dur_tok = id_to_token_dict[tokens[i + 1]]
                if dur_tok.startswith("D"):
                    duration = int(dur_tok[1:])
                    notes.append((pitch, duration))
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    return notes


# ============ Edit Simulation ============
def sample_edit_type():
    """Sample an edit type from EDIT_TYPES."""
    return np.random.choice(EDIT_TYPES, p=[0.4, 0.3, 0.15, 0.15])


def sample_position( melody, edit_type, used_positions=None, rng = None):
    """Sample a valid position for a given edit type, avoiding used positions."""
    if used_positions is None:
        used_positions = set()
    
    if edit_type == "insert_note":
        # Can insert at the end or between existing notes
        valid_positions = [i for i in range(1, len(melody)) if i not in used_positions]
        if not valid_positions:
            # If all positions are used, allow inserting at end
            return len(melody)
        return rng.choice(valid_positions)
    
    # For other edit types, can only edit existing notes
    valid_positions = [i for i in range(len(melody)) if i not in used_positions]
    if not valid_positions:
        return None  # No valid positions available
    return rng.choice(valid_positions)


def sample_duration(exclude_duration):
    """Sample a duration from DURATIONS, excluding exclude_duration."""
    new_durations = [d for d in DURATIONS if d != exclude_duration]
    if not new_durations:
        return exclude_duration
    return random.choice(new_durations)


def sample_insert_note():
    """Sample a random note to insert."""
    pitch = random.randint(PITCH_MIN, PITCH_MAX)
    duration = random.choice(DURATIONS)
    return (pitch, duration)



def sample_edit_script(origin_notes, max_edits=5):
    """ Sample a random edit script for given note sequences. 
    This will not modify the original notes, but will return a list of edit operations and index of affected notes. 
    The edit script can be used to generate corrupted notes and also to create edit masks for training.
    Ensures no overlapping positions for different edit operations.
    """
    edit_script = []
    n_edits = random.randint(1, max_edits)
    used_positions = set()
    
    for _ in range(n_edits):
        edit_type = sample_edit_type()
        
        if edit_type == "insert_note":
            t = sample_position(origin_notes, edit_type, used_positions)
            if t is not None:
                used_positions.add(t)
                note = sample_insert_note()
                edit_script.append(("delete_note", t))
        elif len(origin_notes) == 0:
            continue
        else:
            t = sample_position(origin_notes, edit_type, used_positions)
            if t is not None:
                used_positions.add(t)
                if edit_type == "replace_pitch":
                    old_p, d = origin_notes[t]
                    new_p = old_p
                    while new_p == old_p:
                        new_p = old_p + random.choice([-2, -1, 1, 2])
                        new_p = max(PITCH_MIN, min(PITCH_MAX, new_p))
                    edit_script.append(("replace_pitch", t, old_p))

                elif edit_type == "change_duration":
                    p, old_d = origin_notes[t]
                    new_d = sample_duration(old_d)
                    edit_script.append(("change_duration", t, old_d))

                elif edit_type == "delete_note":
                    note = origin_notes[t]
                    edit_script.append(("insert_note", t, note))
    
    return edit_script


def simulate_edits(melody, max_edits=5, seed=None):
    rng = random.Random(seed)   # LOCAL RNG

    melody_corrupt = copy.deepcopy(melody)
    edit_script = []

    used_positions = set()
    n_edits = rng.randint(1, max_edits)

    for _ in range(n_edits):
        edit_type = rng.choice(EDIT_TYPES)

        if edit_type == "insert_note":
            t = sample_position(melody_corrupt, edit_type, used_positions, rng)
            pitch = rng.randint(PITCH_MIN, PITCH_MAX)
            duration = rng.choice(DURATIONS)
            note = (pitch, duration)

            melody_corrupt.insert(t, note)
            edit_script.append(("delete_note", t))

        elif len(melody_corrupt) == 0:
            continue

        else:
            t = sample_position(melody_corrupt, edit_type, used_positions, rng)

            if edit_type == "replace_pitch":
                old_p, d = melody_corrupt[t]
                new_p = old_p
                while new_p == old_p:
                    new_p = old_p + rng.choice([-2, -1, 1, 2])
                    new_p = max(PITCH_MIN, min(PITCH_MAX, new_p))

                melody_corrupt[t] = (new_p, d)
                edit_script.append(("replace_pitch", t, old_p))

            elif edit_type == "change_duration":
                p, old_d = melody_corrupt[t]
                new_d = rng.choice([d for d in DURATIONS if d != old_d])
                melody_corrupt[t] = (p, new_d)
                edit_script.append(("change_duration", t, old_d))

            elif edit_type == "delete_note":
                note = melody_corrupt.pop(t)
                edit_script.append(("insert_note", t, note))

    return melody_corrupt, edit_script


def build_edit_mask(
    corrupted_tokens,
    edit_script,
    pad_id,
):
    """
    Build a binary edit mask aligned with corrupted_tokens.
    The first token is BOS and the last token is EOS, 
    so note t starts at index 1 + 2*t for pitch and 1 + 2*t + 1 for duration.
    Args:
        corrupted_tokens: List[int] (already padded)
        edit_script: list of edit operations
        pad_id: PAD token id

    Returns:
        mask: List[int] same length as corrupted_tokens
    """
    L = len(corrupted_tokens)
    mask = [0] * L

    for edit in edit_script:
        etype = edit[0]

        if etype == "replace_pitch":
            t = edit[1]
            pitch_idx = 1 + 2 * t
            if pitch_idx < L:
                mask[pitch_idx] = 1

        elif etype == "change_duration":
            t = edit[1]
            dur_idx = 1 + 2 * t + 1
            if dur_idx < L:
                mask[dur_idx] = 1

        elif etype == "delete_note":
            t = edit[1]
            pitch_idx = 1 + 2 * t
            dur_idx = pitch_idx + 1
            if pitch_idx < L:
                mask[pitch_idx] = 1
            if dur_idx < L:
                mask[dur_idx] = 1

        elif etype == "insert_note":
            t = edit[1]
            # insertion affects *neighboring context*
            # mark next note as editable (simple & stable)
            pitch_idx = 1 + 2 * t
            if pitch_idx < L:
                mask[pitch_idx] = 1
                if pitch_idx + 1 < L:
                    mask[pitch_idx + 1] = 1

    # Never penalize PAD tokens
    for i in range(L):
        if corrupted_tokens[i] == pad_id:
            mask[i] = 0

    return mask


# ============ Dataset ============
def collate_fn(batch):
    """Collate function for DataLoader."""
    corrupted, clean, edit_scripts = zip(*batch)

    corrupted = pad_sequence(corrupted, batch_first=True, padding_value=PAD_ID)
    clean = pad_sequence(clean, batch_first=True, padding_value=PAD_ID)

    max_len = max(corrupted.size(1), clean.size(1))
    # pad to max_len if necessary
    if corrupted.size(1) < max_len:
        pad_size = max_len - corrupted.size(1)
        corrupted = F.pad(corrupted, (0, pad_size), value=PAD_ID)
    if clean.size(1) < max_len:
        pad_size = max_len - clean.size(1)
        clean = F.pad(clean, (0, pad_size), value=PAD_ID)

    edit_masks = []
    for tokens, script in zip(corrupted, edit_scripts):
        mask = build_edit_mask(tokens.tolist(), script, PAD_ID)
        edit_masks.append(mask)

    edit_masks = torch.tensor(edit_masks, dtype=torch.float)

    return corrupted, clean, edit_masks, edit_scripts


class MelodyDataset(Dataset):
    """Dataset for melody sequences from TFRecord files."""
    def __init__(self, data_path):
        loaded_data = np.load(data_path)
        # self.dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(tfrecord_path))
        self.dataset = torch.from_numpy(loaded_data)
        self.samples = []
        
        for pitch_seq in self.dataset:
            notes = pitch_seq_to_notes(pitch_seq.numpy())
            if len(notes) > 4 and melody_is_valid(notes):
                self.samples.append(notes)

        self.epoch = 0  # For deterministic edit simulation
        
    def set_epoch(self, epoch):
        """Set epoch number for deterministic edit simulation."""
        self.epoch = epoch

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seed = BASE_SEED + idx + self.epoch * 100000
        original_notes = self.samples[idx]
        corrupted_notes, edit_script = simulate_edits(original_notes, seed=seed)

        corrupted_tokens = notes_to_tokens(corrupted_notes)
        original_tokens = notes_to_tokens(original_notes)

        return (
            torch.tensor(corrupted_tokens, dtype=torch.long),
            torch.tensor(original_tokens, dtype=torch.long),
            edit_script
        )


# ============ Edit Distance and Evaluation ============
def note_replace_cost(n1, n2):
    """Calculate replacement cost between two notes."""
    cost = 0
    if n1[0] != n2[0]:
        cost += 1
    if n1[1] != n2[1]:
        cost += 1
    return cost


def minimal_edit_distance(pred_notes, ref_notes, DELETE_COST=1, INSERT_COST=1):
    """Calculate minimal edit distance between predicted and reference notes."""
    N = len(pred_notes)
    M = len(ref_notes)
    
    # 1. Forward DP Pass
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        dp[i][0] = i * DELETE_COST
    for j in range(1, M + 1):
        dp[0][j] = j * INSERT_COST
        
    def note_replace_cost(n1, n2):
        # We assume cost is additive: 1 for pitch change + 1 for duration change
        p1, d1 = n1
        p2, d2 = n2
        return (p1 != p2) + (d1 != d2)
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            delete = dp[i - 1][j] + DELETE_COST
            insert = dp[i][j - 1] + INSERT_COST
            replace = dp[i - 1][j - 1] + note_replace_cost(pred_notes[i - 1], ref_notes[j - 1])
            dp[i][j] = min(delete, insert, replace)
    
    # 2. Backtracking (Traceback)
    i, j = N, M
    edits = {
        "insert": 0,
        "delete": 0,
        "replace_pitch": 0,
        "replace_duration": 0
    }
    
    while i > 0 or j > 0:
        current = dp[i][j]
        
        # We must explicitly calculate the cost of the diagonal path
        if i > 0 and j > 0:
            p, d = pred_notes[i - 1]
            rp, rd = ref_notes[j - 1]
            diag_cost = dp[i - 1][j - 1] + note_replace_cost(pred_notes[i - 1], ref_notes[j - 1])
        else:
            diag_cost = float('inf') # Impossible if we hit a wall

        # PRIORITIZE DIAGONAL (Match / Substitution) if it yields the minimum path
        if i > 0 and j > 0 and current == diag_cost:
            if p != rp:
                edits["replace_pitch"] += 1
            if d != rd:
                edits["replace_duration"] += 1
            i -= 1
            j -= 1
        # Then check Deletion
        elif i > 0 and current == dp[i - 1][j] + DELETE_COST:
            edits["delete"] += 1
            i -= 1
        # Then check Insertion
        elif j > 0 and current == dp[i][j - 1] + INSERT_COST:
            edits["insert"] += 1
            j -= 1
            
    return dp[N][M], edits

# def minimal_edit_distance(pred_notes, ref_notes):
#     """Calculate minimal edit distance between predicted and reference notes."""
#     N = len(pred_notes)
#     M = len(ref_notes)
    
#     dp = [[0] * (M + 1) for _ in range(N + 1)]
    
#     for i in range(1, N + 1):
#         dp[i][0] = i * DELETE_COST
#     for j in range(1, M + 1):
#         dp[0][j] = j * INSERT_COST
    
#     for i in range(1, N + 1):
#         for j in range(1, M + 1):
#             delete = dp[i - 1][j] + DELETE_COST
#             insert = dp[i][j - 1] + INSERT_COST
#             replace = dp[i - 1][j - 1] + note_replace_cost(
#                 pred_notes[i - 1],
#                 ref_notes[j - 1]
#             )
#             dp[i][j] = min(delete, insert, replace)
    
#     i, j = N, M
#     edits = {
#         "insert": 0,
#         "delete": 0,
#         "replace_pitch": 0,
#         "replace_duration": 0
#     }
    
#     while i > 0 or j > 0:
#         current = dp[i][j]
        
#         if i > 0 and current == dp[i - 1][j] + DELETE_COST:
#             edits["delete"] += 1
#             i -= 1
#         elif j > 0 and current == dp[i][j - 1] + INSERT_COST:
#             edits["insert"] += 1
#             j -= 1
#         else:
#             p, d = pred_notes[i - 1]
#             rp, rd = ref_notes[j - 1]
#             if p != rp:
#                 edits["replace_pitch"] += 1
#             if d != rd:
#                 edits["replace_duration"] += 1
#             i -= 1
#             j -= 1
    
#     return dp[N][M], edits


def calculate_edit_cost(pred_tokens, ref_tokens):
    """Calculate total edit cost for predictions."""
    total_cost = 0
    for p, r in zip(pred_tokens, ref_tokens):
        pred_notes = tokens_to_notes(p, id_to_token)
        ref_notes = tokens_to_notes(r, id_to_token)
        cost, _ = minimal_edit_distance(pred_notes, ref_notes)
        total_cost += cost
    return total_cost


def paired_edit_improvement(baseline_costs, editaware_costs):
    """Calculate paired statistics comparing baseline and edit-aware models."""
    deltas = baseline_costs[:, 0] - editaware_costs[:, 0]
    stats = {
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "std_delta": float(deltas.std()),
        "pct_improved": float((deltas > 0).mean() * 100),
        "pct_worse": float((deltas < 0).mean() * 100),
        "pct_equal": float((deltas == 0).mean() * 100),
        "baseline_mean_cost": float(np.mean(baseline_costs)),
        "editaware_mean_cost": float(np.mean(editaware_costs)),
    }
    return stats


def compare_cost_per_note(baseline_costs, editaware_costs):
    """Compare cost per note between two models."""
    baseline_norm = baseline_costs[:, 1]
    editaware_norm = editaware_costs[:, 1]
    deltas = baseline_norm - editaware_norm
    return {
        "baseline_mean": float(baseline_norm.mean()),
        "editaware_mean": float(editaware_norm.mean()),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "pct_improved": float((deltas > 0).mean() * 100),
    }


def evaluate_model(model, dataloader, device_arg=None):
    """Evaluate model on a dataloader."""
    if device_arg is None:
        device_arg = device
    
    model.eval()
    total_cost = 0
    stats = defaultdict(int)
    sample_costs = []
    
    for corrupted, clean, _, _ in dataloader:
        corrupted = corrupted.to(device_arg)
        clean = clean.to(device_arg)
        
        with torch.no_grad():
            inputs = corrupted[:, :-1]
            targets = clean[:, 1:]
            
            logits, _ = model(inputs)
            pred_tokens = torch.argmax(logits, dim=-1).cpu().numpy()
            ref_tokens = targets.cpu().numpy()
            
            for p, r in zip(pred_tokens, ref_tokens):
                pred_notes = tokens_to_notes(p, id_to_token)
                ref_notes = tokens_to_notes(r, id_to_token)
                cost, edit_counts = minimal_edit_distance(pred_notes, ref_notes)
                costs_per_note = cost / max(len(ref_notes), 1)
                sample_costs.append([cost, costs_per_note])
                total_cost += cost
                
                for k, v in edit_counts.items():
                    stats[k] += v
    
    return total_cost, stats, np.array(sample_costs)


# ============ Model Architecture ============
def causal_mask(size, device_arg):
    """Create a causal mask for autoregressive generation."""
    return torch.triu(
        torch.ones(size, size, device=device_arg), diagonal=1
    ).bool()


class MelodyTransformer(nn.Module):
    """Transformer model for melody sequence correction."""
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        use_edit_head=False
    ):
        super().__init__()

        self.use_edit_head = use_edit_head

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(512, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        # token prediction head (baseline)
        self.fc = nn.Linear(d_model, vocab_size)

        # optional edit head
        if self.use_edit_head:
            self.edit_head = nn.Linear(d_model, 1)

    def forward(self, x):

        B, T = x.shape

        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        mask = causal_mask(T, x.device)

        x = self.embed(x) + self.pos_embed(pos)

        h = self.decoder(x, x, tgt_mask=mask)

        token_logits = self.fc(h)

        if not self.use_edit_head:
            return token_logits, None

        edit_logits = self.edit_head(h).squeeze(-1)

        return token_logits, edit_logits


# ============ Training Functions ============
def train_baseline(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device_arg=None,
    epochs=10,
    patience=5,
    min_improvement=1e-3
):
    """Train baseline model with early stopping."""
    if device_arg is None:
        device_arg = device
    
    print(f"Training on device: {device_arg}")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    baseline_loss = defaultdict(list)
    
    for epoch in range(epochs):
        train_dataloader.dataset.set_epoch(epoch)  # For deterministic edit simulation
        val_dataloader.dataset.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        
        for corrupted, clean, _, _ in train_dataloader:
            corrupted = corrupted.to(device_arg)
            clean = clean.to(device_arg)
            
            inputs = corrupted[:, :-1]
            targets = clean[:, 1:]
            
            logits, _ = model(inputs)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"[Baseline] Epoch {epoch+1}: loss = {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_edit_cost = 0.0
        
        with torch.no_grad():
            for corrupted, clean, _, _ in val_dataloader:
                corrupted = corrupted.to(device_arg)
                clean = clean.to(device_arg)
                
                inputs = corrupted[:, :-1]
                targets = clean[:, 1:]
                
                logits, _ = model(inputs)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
                
                pred_tokens = torch.argmax(logits, dim=-1).cpu().numpy()
                ref_tokens = targets.cpu().numpy()
                
                val_loss += loss.item()
                val_edit_cost += calculate_edit_cost(pred_tokens, ref_tokens)
        
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_edit_cost = val_edit_cost / len(val_dataloader)
        print(f"[Validation] Epoch {epoch+1}: loss = {avg_val_loss:.4f}, edit_cost = {avg_val_edit_cost:.2f}")
        
        baseline_loss['train'].append(avg_loss)
        baseline_loss['val'].append(avg_val_loss)
        baseline_loss['edit_cost'].append(avg_val_edit_cost)
        
        if best_val_loss - avg_val_loss > min_improvement:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"Validation loss improved to {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! Validation loss did not improve for {patience} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f}")
                
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print("Restored best model weights.")
                break
    
    return baseline_loss


def visualize_loss(loss_dict):
    """Visualize training and validation loss."""
    epochs = range(1, len(loss_dict['train']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_dict['train'], label='Training Loss')
    plt.plot(epochs, loss_dict['val'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()


def generate_edit_scripts(data_path, output_path, max_edits=5):
    """Generate and save edit scripts for a dataset."""
    print(f"Loading data from {data_path}")
    loaded_data = np.load(data_path)
    dataset = torch.from_numpy(loaded_data)
    
    samples = []
    for pitch_seq in dataset:
        notes = pitch_seq_to_notes(pitch_seq.numpy())
        if len(notes) > 4 and melody_is_valid(notes):
            samples.append(notes)
    
    print(f"Found {len(samples)} valid samples")
    
    edit_scripts = []
    for i, notes in enumerate(samples):
        if i % 1000 == 0:
            print(f"Generating edit script {i}/{len(samples)}")
        edit_script = sample_edit_script(notes, max_edits=max_edits)
        edit_scripts.append(edit_script)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(edit_scripts, f, indent=2)
    
    print(f"Saved {len(edit_scripts)} edit scripts to {output_path}")
    return edit_scripts


def create_dataloaders(data_name, batch_size=32):
    """Create train, validation, and test dataloaders."""
    if data_name == "train":
        data_path = train_path
    elif data_name == "val":
        data_path = val_path
    elif data_name == "test":
        data_path = test_path
    else:
        raise ValueError(f"Unknown data_name: {data_name}")

    dataset = MelodyDataset(data_path)
    print(f"{data_name.capitalize()} Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(data_name == "train"), collate_fn=collate_fn, num_workers=0)
    
    return dataloader


def create_model_and_optimizer(model_name="baseline", d_model=256, n_heads=4, n_layers=4, lr=0.0001):
    """Create a model and optimizer."""
    if model_name == "baseline":
        use_edit_head = False
    else:  # "edit-aware"
        use_edit_head = True

    model = MelodyTransformer(vocab_size=len(vocab), d_model=d_model, n_heads=n_heads, n_layers=n_layers, use_edit_head=use_edit_head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


# ============ Edit-Aware Training ============
def edit_weighted_loss(
    token_logits,      # (B, T, V)
    target_tokens,     # (B, T)
    edit_logits,       # (B, T)
    edit_masks,        # (B, T)
    corrupted_tokens,  # (B, T)
    pad_id = PAD_ID,
    alpha=1.0
):
    """
    Multi-task edit-aware loss.

    token_logits: model token predictions
    target_tokens: original tokens as target
    edit_logits: model edit predictions
    edit_masks: ground truth edit mask (0/1)
    corrupted_tokens: input tokens

    Returns:
        scalar loss
    """

    B, T, V = token_logits.shape

    # --------------------------------------------------
    # Token prediction loss
    # --------------------------------------------------

    token_loss = F.cross_entropy(
        token_logits.reshape(B * T, V),
        target_tokens.reshape(B * T),
        ignore_index=pad_id
    )

    # --------------------------------------------------
    # Edit prediction loss
    # --------------------------------------------------

    # valid tokens (exclude padding)
    valid_mask = corrupted_tokens != pad_id   # (B, T)

    # select valid positions
    edit_logits_valid = edit_logits[valid_mask]
    edit_targets_valid = edit_masks.float()[valid_mask]

    edit_loss = F.binary_cross_entropy_with_logits(
        edit_logits_valid,
        edit_targets_valid
    )

    # --------------------------------------------------
    # Combined loss
    # --------------------------------------------------

    loss = token_loss + alpha * edit_loss

    return loss

def weighted_edit_ce_loss(
    logits,        # (B, T, V)
    targets,       # (B, T)
    edit_mask,     # (B, T)
    pad_id,
    alpha=0.7
):
    B, T, V = logits.shape

    ce = F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        ignore_index=pad_id,
        reduction="none"
    ).reshape(B, T)

    # weights: emphasize edited regions
    weights = 1.0 + alpha * edit_mask.float()

    # valid tokens only
    valid = (targets != pad_id).float()

    loss = (ce * weights * valid).sum() / valid.sum()

    return loss

def train_edit_aware(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device_arg=None,
    alpha=1.0,
    epochs=10,
    patience=5,
    min_improvement=1e-3
):
    """Train edit-aware model with weighted loss on edited positions."""
    if device_arg is None:
        device_arg = device
    
    print(f"Training on device: {device_arg}")
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    edit_aware_loss = defaultdict(list)

    for epoch in range(epochs):
        train_dataloader.dataset.set_epoch(epoch)  # For deterministic edit simulation
        val_dataloader.dataset.set_epoch(epoch)

        total_loss = 0.0
        model.train()

        for corrupted, clean, edit_mask, _ in train_dataloader:
            corrupted = corrupted.to(device_arg)
            clean = clean.to(device_arg)
            edit_mask = edit_mask.to(device_arg)

            inputs = corrupted[:, :-1]
            targets = clean[:, 1:]
            mask = edit_mask[:, 1:]

            logits, edit_logits = model(inputs)

            loss = edit_weighted_loss(
                logits,
                targets,
                edit_logits,
                mask,
                inputs,
                pad_id=PAD_ID,
                alpha=alpha,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"[Edit-aware α={alpha}] Epoch {epoch+1}: loss = {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_edit_cost = 0.0
        
        with torch.no_grad():
            for corrupted, clean, edit_mask, _ in val_dataloader:
                corrupted = corrupted.to(device_arg)
                clean = clean.to(device_arg)
                edit_mask = edit_mask.to(device_arg)

                inputs = corrupted[:, :-1]
                targets = clean[:, 1:]
                mask = edit_mask[:, 1:]

                logits, edit_logits = model(inputs)

                loss = edit_weighted_loss(
                    logits,
                    targets,
                    edit_logits,
                    mask,
                    inputs,
                    pad_id=PAD_ID,
                    alpha=alpha,
                )

                pred_tokens = torch.argmax(logits, dim=-1).cpu().numpy()
                ref_tokens = targets.cpu().numpy()

                val_loss += loss.item()
                val_edit_cost += calculate_edit_cost(pred_tokens, ref_tokens)

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_edit_cost = val_edit_cost / len(val_dataloader)
        print(f"[Validation] Epoch {epoch+1}: loss = {avg_val_loss:.4f}, edit_cost = {avg_val_edit_cost:.2f}")

        edit_aware_loss['train'].append(avg_loss)
        edit_aware_loss['val'].append(avg_val_loss)
        edit_aware_loss['edit_cost'].append(avg_val_edit_cost)

        # Early stopping logic
        if best_val_loss - avg_val_loss > min_improvement:
            # Significant improvement detected
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"Validation loss improved to {avg_val_loss:.4f}")
        else:
            # No significant improvement
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! Validation loss did not improve for {patience} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f}")
                
                # Restore best model
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print("Restored best model weights.")
                break
    
    return edit_aware_loss
