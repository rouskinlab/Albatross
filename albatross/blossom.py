#!/usr/bin/env python3
"""
Blossom matching and pseudoknot removal for Albatross.

Implements Algorithm 1 steps 6–7:
  6. Maximum-weight matching (Edmonds' Blossom via NetworkX)
  7. Greedy pseudoknot removal (longest span first)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np


def prepare_weighted_map(raw_map: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Combines raw scores with mask and enforces symmetry."""
    # 1. Apply Mask (if provided)
    if mask is not None:
        weighted_map = raw_map * mask
    else:
        weighted_map = raw_map.copy()

    # 2. Symmetrize (Take Max) - Crucial for RNA contact prediction
    # Ensures w[i,j] == w[j,i]
    return np.maximum(weighted_map, weighted_map.T)


def remove_pseudoknots_from_matrix(binary_matrix: np.ndarray) -> np.ndarray:
    """
    Remove pseudoknots (crossing base pairs) from a binary pairing matrix.

    Uses a greedy algorithm: prioritizes longer-span pairs (which correspond to
    longer helices / more stable stems) and removes pairs that cross with
    already-accepted pairs.

    Args:
        binary_matrix: NxN symmetric binary matrix where 1 indicates a base pair.

    Returns:
        NxN symmetric binary matrix with all pseudoknots removed (nested structure).
    """
    N = binary_matrix.shape[0]

    # Extract pairs from upper triangle (0-based indexing)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if binary_matrix[i, j] > 0:
                pairs.append((i, j))

    if not pairs:
        return np.zeros_like(binary_matrix)

    # Sort by span length (longer spans first = prioritize longer helices)
    sorted_pairs = sorted(pairs, key=lambda p: p[1] - p[0], reverse=True)

    accepted = []
    for pair in sorted_pairs:
        a1, b1 = pair
        is_crossing = False
        for a2, b2 in accepted:
            # Two pairs cross if one opens inside the other but doesn't close inside
            if (a1 < a2 < b1 < b2) or (a2 < a1 < b2 < b1):
                is_crossing = True
                break
        if not is_crossing:
            accepted.append(pair)

    # Reconstruct binary matrix
    clean_matrix = np.zeros_like(binary_matrix)
    for i, j in accepted:
        clean_matrix[i, j] = 1
        clean_matrix[j, i] = 1

    removed = len(pairs) - len(accepted)
    if removed > 0:
        print(
            f"Pseudoknot removal: removed {removed} crossing pairs "
            f"({len(accepted)} pairs remaining)"
        )

    return clean_matrix


class GraphSanitizer:
    """
    Sanitizes constraints using Maximum Weight Matching (Blossom Algorithm).

    Resolves conflicts globally by selecting the set of edges that maximizes
    total weight while enforcing 1-to-1 pairing.

    NOTE: By default this ALLOWS pseudoknots (crossing edges).
    Set remove_pseudoknots=True to strip them via greedy post-processing.
    """

    def __init__(self, remove_pseudoknots: bool = False):
        """
        Args:
            remove_pseudoknots: If True, apply greedy pseudoknot removal after
                matching to ensure a fully nested (non-crossing) structure.
        """
        self.remove_pseudoknots = remove_pseudoknots

    def sanitize(self, raw_map: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        # Pre-process inputs
        S = prepare_weighted_map(raw_map, mask)
        N = S.shape[0]

        # Build the Graph
        # Only add edges with positive weight to keep graph sparse and fast
        G = nx.Graph()
        rows, cols = np.where(np.triu(S) > 0)

        for r, c in zip(rows, cols):
            G.add_edge(r, c, weight=S[r, c])

        # Run Edmonds' Blossom Algorithm (Max Weight Matching)
        # maxcardinality=False allows the algorithm to skip weak pairs
        # rather than forcing a full matching.
        matching = nx.max_weight_matching(G, maxcardinality=False)

        # Reconstruct Binary Matrix
        clean_matrix = np.zeros_like(S)
        for u, v in matching:
            clean_matrix[u, v] = 1
            clean_matrix[v, u] = 1

        # Optionally remove pseudoknots (crossing pairs)
        if self.remove_pseudoknots:
            clean_matrix = remove_pseudoknots_from_matrix(clean_matrix)

        return clean_matrix


def apply_blossom(
    raw_map: np.ndarray,
    mask: np.ndarray = None,
    remove_pseudoknots: bool = True,
) -> np.ndarray:
    """
    Run Blossom matching (and optional pseudoknot removal) on a map.

    Args:
        raw_map: Normalized dependency map (or raw scores).
        mask: Optional filtered mask from HeuristicPixel.
        remove_pseudoknots: If True (paper default), remove crossing pairs.

    Returns:
        Binary N×N pairing matrix.
    """
    return GraphSanitizer(remove_pseudoknots=remove_pseudoknots).sanitize(
        raw_map, mask=mask
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply Blossom matching and pseudoknot removal"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input NPZ with filtered_map and/or dependency_map",
    )
    parser.add_argument("--output", type=str, required=True, help="Output NPZ path")
    parser.add_argument(
        "--keep-pseudoknots",
        action="store_true",
        help="Keep crossing pairs (default: remove them)",
    )

    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)

    if "filtered_map" in data:
        mask = data["filtered_map"]
        raw = mask
        if "dependency_map" in data:
            raw = np.clip(data["dependency_map"] / 20.0, 0.0, 1.0)
        elif "depmap" in data:
            raw = np.clip(data["depmap"] / 20.0, 0.0, 1.0)
    elif "dependency_map" in data:
        raw = np.clip(data["dependency_map"] / 20.0, 0.0, 1.0)
        mask = None
    elif "depmap" in data:
        raw = np.clip(data["depmap"] / 20.0, 0.0, 1.0)
        mask = None
    else:
        raise KeyError(
            f"No map found in {args.input}. Keys: {list(data.keys())}"
        )

    binary = apply_blossom(
        raw, mask=mask, remove_pseudoknots=not args.keep_pseudoknots
    )

    n_pairs = int(np.sum(binary) / 2)
    print(f"Binary map: {n_pairs} base pairs")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {"binary_map": binary, "n_base_pairs": n_pairs}
    if "sequence" in data:
        payload["sequence"] = data["sequence"]

    np.savez_compressed(out, **payload)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
