"""
Handles generation and loading of (histogram, expression) training pairs.

Each sample in the dataset is:
  - encoder_input : float32 tensor of shape (n_bins,)
                    normalised histogram (bin fractions summing to 1)
  - decoder_input : int64 tensor of shape (seq_len,)
                    prefix token ids with <BOS> prepended
  - decoder_target: int64 tensor of shape (seq_len,)
                    prefix token ids with <EOS> appended
  - constants      : float32 tensor of shape (max_constants,)
                    actual values of numeric leaf tokens (padded with 0s)

The split between encoder_input / decoder_input / decoder_target
follows the standard teacher-forcing training setup:
  - At step t, the decoder sees tokens 0..t (from decoder_input)
  - At step t, the target is token t+1 (from decoder_target)
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.expression_generator import generate_expression
from data.histogram_generator import expression_to_histogram, histogram_to_float
from data.tokenizer import Tokenizer


MAX_CONSTANTS = 8 # Max no of numeric constant positions in an expr. More than this, then discard!
MAX_SEQ_LEN = 30 # Max token seq len including <BOS>/<EOS>


def _extract_constants(tokens):
    """
    Finds the numeric literal values from a token list and replace them with <CONST>

    Returns
    -------
    cleaned_tokens : list of str
        Original tokens with integer literals replaced by <CONST>.
    const_values : list of float
        Ordered list of the replaced literal values.
    """
    from data.tokenizer import INTEGER_TOKENS
    cleaned = []
    const_values = []
    int_set = set(INTEGER_TOKENS)
    for tok in tokens:
        if tok in int_set:
            const_values.append(float(tok))
            cleaned.append("<CONST>")
        else:
            cleaned.append(tok)
    return cleaned, const_values


def generate_single_sample(tokenizer, n_bins=50, total_count=500,
                            n_ops=None, rng_py=None, rng_np=None):
    """
    Generate one complete training sample.

    Returns a dict with keys: encoder_input, decoder_input,
    decoder_target, constants, raw_tokens.
    Returns None if generation fails.
    """
    if rng_py is None:
        rng_py = random.Random()
    if rng_np is None:
        rng_np = np.random.default_rng()

    prefix_tokens, normalised_expr = generate_expression(
        n_ops=n_ops, rng=rng_py
    )
    if prefix_tokens is None:
        return None

    if len(prefix_tokens) + 2 > MAX_SEQ_LEN:   # +2 for BOS/EOS
        return None

    histogram, _, _ = expression_to_histogram(
        normalised_expr, n_bins=n_bins,
        total_count=total_count, rng=rng_np
    )
    enc_input = histogram_to_float(histogram)   # normalised fractions

    cleaned_tokens, const_values = _extract_constants(prefix_tokens)

    if len(const_values) > MAX_CONSTANTS:
        return None

    # decoder_input:  <BOS> token1 token2 ...
    # decoder_target:        token1 token2 ... <EOS>
    full_ids = tokenizer.wrap(cleaned_tokens)       # BOS + tokens + EOS
    dec_input = full_ids[:-1]                       # drop last
    dec_target = full_ids[1:]                       # drop first

    # Pad constant values to MAX_CONSTANTS
    padded_consts = const_values + [0.0] * (MAX_CONSTANTS - len(const_values))

    return {
        "encoder_input": torch.tensor(enc_input, dtype=torch.float32),
        "decoder_input": torch.tensor(dec_input, dtype=torch.long),
        "decoder_target": torch.tensor(dec_target, dtype=torch.long),
        "constants": torch.tensor(padded_consts, dtype=torch.float32),
        "raw_tokens": prefix_tokens,     # kept for debugging / plotting
    }


class SymbolicDataset(Dataset):
    """
    PyTorch Dataset that either loads pre-generated samples from disk
    or generates them on-the-fly.

    Parameters
    ----------
    samples : list of dicts
        Each dict has keys: encoder_input, decoder_input,
        decoder_target, constants.
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    Custom collate function for variable-length decoder sequences.
    Pads decoder_input and decoder_target to the length of the longest
    sequence in the batch.
    """
    tokenizer = Tokenizer()
    pad_id = tokenizer.pad_id

    enc_inputs = torch.stack([s["encoder_input"] for s in batch])      # (B, n_bins)
    constants = torch.stack([s["constants"] for s in batch])            # (B, MAX_CONSTANTS)

    max_len = max(s["decoder_input"].size(0) for s in batch)
    dec_inputs = []
    dec_targets = []
    for s in batch:
        length = s["decoder_input"].size(0)
        pad_needed = max_len - length
        dec_inputs.append(
            torch.cat([s["decoder_input"],
                       torch.full((pad_needed,), pad_id, dtype=torch.long)])
        )
        dec_targets.append(
            torch.cat([s["decoder_target"],
                       torch.full((pad_needed,), pad_id, dtype=torch.long)])
        )

    return {
        "encoder_input": enc_inputs,
        "decoder_input": torch.stack(dec_inputs),
        "decoder_target": torch.stack(dec_targets),
        "constants": constants,
    }


def generate_dataset(n_samples, n_bins=50, total_count=500, seed=42, verbose=True):
    """
    Generate a full dataset of (histogram, expression) pairs.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_bins : int
        Histogram resolution.
    total_count : int
        Total counts per histogram.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress every 10% of samples.

    Returns
    -------
    SymbolicDataset
    """
    tokenizer = Tokenizer()
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    samples = []

    while len(samples) < n_samples:
        sample = generate_single_sample(
            tokenizer, n_bins=n_bins,
            total_count=total_count,
            rng_py=rng_py, rng_np=rng_np
        )
        if sample is not None:
            samples.append(sample)
            if verbose and len(samples) % 100 == 0:
                print(f"  Generated {len(samples)}/{n_samples} samples")

    if verbose:
        print(f"Generated {len(samples)} samples")

    return SymbolicDataset(samples)


def save_dataset(dataset, path):
    """Save a SymbolicDataset to disk as a JSON-serialisable list."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    records = []
    for s in dataset.samples:
        records.append({
            "encoder_input": s["encoder_input"].tolist(),
            "decoder_input": s["decoder_input"].tolist(),
            "decoder_target": s["decoder_target"].tolist(),
            "constants": s["constants"].tolist(),
            "raw_tokens": s.get("raw_tokens", []),
        })
    with open(path, "w") as f:
        json.dump(records, f)
    print(f"Saved {len(records)} samples to {path}")


def load_dataset(path):
    """Load a SymbolicDataset saved by save_dataset()."""
    with open(path) as f:
        records = json.load(f)
    samples = []
    for r in records:
        samples.append({
            "encoder_input": torch.tensor(r["encoder_input"], dtype=torch.float32),
            "decoder_input": torch.tensor(r["decoder_input"], dtype=torch.long),
            "decoder_target": torch.tensor(r["decoder_target"], dtype=torch.long),
            "constants": torch.tensor(r["constants"], dtype=torch.float32),
            "raw_tokens": r.get("raw_tokens", []),
        })
    return SymbolicDataset(samples)


def get_dataloader(dataset, batch_size=64, shuffle=True, num_workers=0):
    """Wrap a SymbolicDataset in a PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )