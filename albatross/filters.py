#!/usr/bin/env python3
"""
Heuristic pixel filters for Albatross binary structure prediction.

Implements Algorithm 1 steps 1–5:
  1. Normalize: clip(M / 20, 0, 1)
  2. Symmetrize: max(M, M^T)
  3. Threshold τ
  4. Diagonal-band exclusion γ
  5. Anti-diagonal connectivity α

Paper defaults: α=3, τ=0.11, γ=4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


class HeuristicPixel:
    """
    Heuristic pixel-based denoising method (formerly AntiDiagonalFilter).
    Applies filtering based on:
    - Tau (presence threshold)
    - Alpha (minimum diagonal run length)
    - Epsilon (gap tolerance in diagonal runs)
    - Gamma (diagonal band exclusion)
    """

    def denoise(
        self,
        map_matrix: np.ndarray,
        alpha: int = 3,
        epsilon: int = 0,
        tau: float = 0.0,
        gamma: int = 0,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply heuristic filtering.

        Args:
            map_matrix: Input dependency map (already normalized to [0, 1]).
            alpha: Minimum length of a stem (diagonal run).
            epsilon: Gap tolerance (max distance to look for supporters).
            tau: Presence threshold (values < tau are ignored).
            gamma: Diagonal band exclusion radius (exclude if |i-j| <= gamma).

        Returns:
            Filtered dependency map.
        """
        # 0. Symmetrize the Dependency Map
        # Take the maximum between e[i,j] and e[j,i] to ensure symmetry
        # This achieves one unique score per pair of positions
        map_matrix = np.maximum(map_matrix, map_matrix.T)

        # 1. Apply Tau (Presence Threshold)
        # We work with a boolean mask for structural logic, but return values based on original map
        B = map_matrix >= tau
        H, W = map_matrix.shape

        # 2. Apply Gamma FIRST (Diagonal Band Filter)
        # This must be done BEFORE alpha filtering to avoid finding stems that cross the diagonal band
        if gamma > 0:
            rr, cc = np.ogrid[:H, :W]
            band_mask = np.abs(rr - cc) <= gamma
            B[band_mask] = False

        keep = np.zeros((H, W), dtype=bool)

        # 3. Apply Alpha & Epsilon (Anti-Diagonal Filter)
        if epsilon == 0:
            # Fast path for epsilon == 0 (no gap tolerance)
            # Directly scan anti-diagonals for runs without supporters logic
            max_k = H + W - 2
            for k in range(max_k + 1):
                r = min(k, H - 1)
                c = k - r
                current_run_coords = []

                while r >= 0 and c < W:
                    if B[r, c]:
                        current_run_coords.append((r, c))
                    else:
                        # End of run, check if it's long enough
                        if len(current_run_coords) >= alpha:
                            for rr, cc in current_run_coords:
                                keep[rr, cc] = True
                        current_run_coords = []
                    r -= 1
                    c += 1

                # Check final run
                if len(current_run_coords) >= alpha:
                    for rr, cc in current_run_coords:
                        keep[rr, cc] = True
        else:
            # General case for epsilon > 0 (with gap tolerance)
            for s in range(H + W - 1):
                coords = self._anti_diag_coords(s, H, W)
                L = len(coords)

                if L < alpha:
                    continue

                # Check presence along the diagonal with epsilon tolerance
                q = np.zeros(L, dtype=bool)
                supporters_per_index = [[] for _ in range(L)]

                for t in range(L):
                    r, c = coords[t]
                    supporters = self._get_supporters(B, r, c, epsilon)
                    supporters_per_index[t] = supporters
                    q[t] = len(supporters) > 0

                # Find runs of sufficient length (alpha)
                t = 0
                while t < L:
                    if not q[t]:
                        t += 1
                        continue

                    start = t
                    while t < L and q[t]:
                        t += 1
                    end = t - 1  # inclusive

                    if end - start + 1 >= alpha:
                        # Mark supported pixels to keep
                        for u in range(start, end + 1):
                            for rs, cs in supporters_per_index[u]:
                                keep[rs, cs] = True

        # 4. Apply mask to original matrix (preserve values where kept, zero elsewhere)
        return np.where(keep, map_matrix, 0.0)

    @staticmethod
    def _anti_diag_coords(s: int, H: int, W: int) -> list:
        """Get coordinates along anti-diagonal index s (where r + c = s)."""
        r_start = min(s, H - 1)
        c_start = s - r_start
        L = min(r_start + 1, W - c_start)
        return [(r_start - t, c_start + t) for t in range(L)]

    @staticmethod
    def _get_supporters(B: np.ndarray, r: int, c: int, epsilon: int) -> list:
        """Return list of supporting (r,c) coords within epsilon radius."""
        H, W = B.shape
        supporters = []
        if 0 <= r < H and 0 <= c < W and B[r, c]:
            supporters.append((r, c))

        # Check epsilon neighborhood (cross shape)
        for d in range(1, epsilon + 1):
            if r - d >= 0 and B[r - d, c]:
                supporters.append((r - d, c))
            if r + d < H and B[r + d, c]:
                supporters.append((r + d, c))
            if c - d >= 0 and B[r, c - d]:
                supporters.append((r, c - d))
            if c + d < W and B[r, c + d]:
                supporters.append((r, c + d))

        return supporters


def apply_filters(
    dep_map: np.ndarray,
    alpha: int = 3,
    tau: float = 0.11,
    gamma: int = 4,
    reference_max: float = 20.0,
    epsilon: int = 0,
) -> np.ndarray:
    """
    Normalize and apply HeuristicPixel filters (Algorithm 1 steps 1–5).

    Args:
        dep_map: Raw N×N dependency map.
        alpha: Minimum stem length (default 3).
        tau: Presence threshold (default 0.11).
        gamma: Diagonal band exclusion (default 4).
        reference_max: Normalization divisor (default 20).
        epsilon: Gap tolerance (paper uses 0).

    Returns:
        Filtered (weighted) map used as the Blossom mask.
    """
    normalized = np.clip(dep_map / reference_max, 0.0, 1.0)
    return HeuristicPixel().denoise(
        normalized, alpha=alpha, epsilon=epsilon, tau=tau, gamma=gamma
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply HeuristicPixel filters to a raw dependency map"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input NPZ with dependency_map (or depmap / arr_0)",
    )
    parser.add_argument("--output", type=str, required=True, help="Output NPZ path")
    parser.add_argument("--alpha", type=int, default=3, help="α: minimum stem length")
    parser.add_argument("--tau", type=float, default=0.11, help="τ: presence threshold")
    parser.add_argument("--gamma", type=int, default=4, help="γ: diagonal band exclusion")
    parser.add_argument(
        "--reference-max", type=float, default=20.0, help="Normalization divisor"
    )
    parser.add_argument("--epsilon", type=int, default=0, help="ε: gap tolerance")

    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if "dependency_map" in data:
        dep_map = data["dependency_map"]
    elif "depmap" in data:
        dep_map = data["depmap"]
    elif "arr_0" in data:
        dep_map = data["arr_0"]
    else:
        raise KeyError(
            f"No dependency map found in {args.input}. "
            f"Keys: {list(data.keys())}"
        )

    filtered = apply_filters(
        dep_map,
        alpha=args.alpha,
        tau=args.tau,
        gamma=args.gamma,
        reference_max=args.reference_max,
        epsilon=args.epsilon,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "filtered_map": filtered,
        "alpha": args.alpha,
        "tau": args.tau,
        "gamma": args.gamma,
        "reference_max": args.reference_max,
        "epsilon": args.epsilon,
    }
    if "sequence" in data:
        payload["sequence"] = data["sequence"]

    np.savez_compressed(out, **payload)
    n_kept = int(np.sum(filtered > 0))
    print(f"Filtered map: {n_kept} non-zero pixels (out of {filtered.size})")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
