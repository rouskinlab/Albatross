#!/usr/bin/env python3
"""
Dependency map generation for Albatross.

For a sequence of length N, runs 3N+1 forward passes (wild type + 3 substitutions
at each site). Computes log2 odds shifts and takes M[i,j] = max |S| over mutation
and target bases, with diagonal zeroed. Maps are directed and not symmetrized.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rinalmo.pretrained import get_pretrained_model

try:
    from albatross.training import IRESPretrainingWrapper

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

NUC_TABLE = {"A": 0, "C": 1, "G": 2, "U": 3}
ACGU_IDXS = [5, 6, 7, 8]  # A, C, G, T indices in RiNALMo output


class RNADataset(Dataset):
    """Simple dataset for RNA sequences."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def mutate_sequence(seq: str) -> pd.DataFrame:
    """Generate all single nucleotide mutations of the sequence."""
    seq = seq.upper().replace("T", "U")
    mutated_sequences = {"seq": [], "mutation_pos": [], "nuc": [], "var_nt_idx": []}

    mutated_sequences["seq"].append(seq)
    mutated_sequences["mutation_pos"].append(-1)
    mutated_sequences["nuc"].append("real sequence")
    mutated_sequences["var_nt_idx"].append(-1)

    for i in range(len(seq)):
        for nuc in ["A", "C", "G", "U"]:
            if nuc != seq[i]:
                mutated_sequences["seq"].append(seq[:i] + nuc + seq[i + 1 :])
                mutated_sequences["mutation_pos"].append(i)
                mutated_sequences["nuc"].append(nuc)
                mutated_sequences["var_nt_idx"].append(NUC_TABLE[nuc])

    return pd.DataFrame(mutated_sequences)


def load_model_and_alphabet(checkpoint_path=None, model_name="giga-v1", device="cpu"):
    """Load model and alphabet from either checkpoint or pretrained weights."""
    if checkpoint_path:
        if not LIGHTNING_AVAILABLE:
            raise ImportError(
                "Lightning is required to load checkpoints. "
                "Please install lightning.pytorch"
            )

        print(f"Loading model from checkpoint: {checkpoint_path}")
        wrapper = IRESPretrainingWrapper.load_from_checkpoint(
            checkpoint_path, map_location=device
        )
        model = wrapper.model

        from rinalmo.data.alphabet import Alphabet

        alphabet = Alphabet(**model.config["alphabet"])
        print("Successfully loaded model from checkpoint")
    else:
        print(f"Loading pretrained model: {model_name}")
        model, alphabet = get_pretrained_model(model_name=model_name)

    model = model.to(device=device)
    model.eval()
    return model, alphabet


def compute_dependency_map(
    seq: str,
    model,
    alphabet,
    device: str,
    batch_size: int = 64,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Compute raw dependency map for a sequence using SNP effect analysis."""
    print(f"Computing dependency map for sequence of length {len(seq)}...")

    dataset = mutate_sequence(seq)
    print(f"Generated {len(dataset)} mutations")

    def collate_fn(batch):
        tokenized_batch = torch.tensor(
            alphabet.batch_tokenize(batch), dtype=torch.int64, device=device
        )
        return tokenized_batch

    rna_dataset = RNADataset(list(dataset["seq"].values))
    data_loader = DataLoader(
        rna_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    print(f"Running inference on {len(data_loader)} batches...")
    output_arrays = []
    for batch_tokens in data_loader:
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(batch_tokens)["logits"].cpu().to(torch.float32)
        output_probs = torch.nn.functional.softmax(outputs, dim=-1)[:, :, ACGU_IDXS]
        output_arrays.append(output_probs)

    snp_reconstruct = torch.concat(output_arrays, axis=0).numpy()
    print(f"Model inference output shape: {snp_reconstruct.shape}")

    # Remove CLS and EOS tokens
    snp_reconstruct = snp_reconstruct[:, 1:-1, :]

    # Normalize
    snp_reconstruct = snp_reconstruct + epsilon
    snp_reconstruct = snp_reconstruct / snp_reconstruct.sum(axis=-1)[:, :, np.newaxis]

    # Compute SNP effects (log2 odds shift)
    seq_len = snp_reconstruct.shape[1]
    snp_effect = np.zeros((seq_len, seq_len, 4, 4))
    reference_probs = snp_reconstruct[dataset[dataset["nuc"] == "real sequence"].index[0]]

    snp_effect[
        dataset.iloc[1:]["mutation_pos"].values,
        :,
        dataset.iloc[1:]["var_nt_idx"].values,
        :,
    ] = (
        np.log2(snp_reconstruct[1:])
        - np.log2(1 - snp_reconstruct[1:])
        - np.log2(reference_probs)
        + np.log2(1 - reference_probs)
    )

    dep_map = np.max(np.abs(snp_effect), axis=(2, 3))
    dep_map[np.arange(dep_map.shape[0]), np.arange(dep_map.shape[0])] = 0

    print(f"Dependency map shape: {dep_map.shape}")
    print(f"Dependency map range: {np.min(dep_map):.4f} to {np.max(dep_map):.4f}")

    return dep_map


def main():
    parser = argparse.ArgumentParser(
        description="Compute RNA dependency maps using Albatross / RiNALMo"
    )
    parser.add_argument("--fasta", type=str, help="Input FASTA file")
    parser.add_argument("--sequence", type=str, help="Input sequence string")
    parser.add_argument("--out", type=str, default="depmap", help="Output prefix")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--checkpoint", type=str, help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument(
        "--model_name",
        type=str,
        default="giga-v1",
        help="Pretrained model name (used if --checkpoint not provided)",
    )

    args = parser.parse_args()

    if args.fasta:
        with open(args.fasta, "r") as f:
            lines = f.readlines()
            sequence = "".join(
                [line.strip() for line in lines if not line.startswith(">")]
            )
    elif args.sequence:
        sequence = args.sequence
    else:
        parser.error("Provide --sequence or --fasta")

    sequence = sequence.upper().replace("T", "U")
    print(f"Input sequence length: {len(sequence)}")
    print(f"Using device: {args.device}")

    model, alphabet = load_model_and_alphabet(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=args.device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {total_params:,} parameters")

    dep_map = compute_dependency_map(
        sequence, model, alphabet, args.device, args.batch_size
    )

    output_dir = os.path.dirname(args.out) if os.path.dirname(args.out) else "."
    os.makedirs(output_dir, exist_ok=True)

    npz_output = f"{args.out}.npz"
    np.savez(
        npz_output,
        dependency_map=dep_map,
        sequence=sequence,
        sequence_length=len(sequence),
        checkpoint_path=args.checkpoint if args.checkpoint else "pretrained",
        model_name=args.model_name,
    )
    print(f"Saved dependency map to {npz_output}")


if __name__ == "__main__":
    main()
