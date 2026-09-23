#!/usr/bin/env python3
"""
Structure comparison metrics for Albatross.

Compares predicted and reference structures as binary pairing matrices over the
upper triangle. Reports precision, recall, and F1:

  P = TP / (TP + FP)
  R = TP / (TP + FN)
  F1 = 2PR / (P + R)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def compute_f1(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float | int]:
    """Upper-triangle precision, recall, and F1 for binary pairing matrices."""
    pred_bin = (pred > 0).astype(np.uint8)
    truth_bin = (truth > 0).astype(np.uint8)
    upper = np.triu_indices(pred_bin.shape[0], k=1)
    pred_vals = pred_bin[upper]
    truth_vals = truth_bin[upper]
    tp = int(np.sum((pred_vals == 1) & (truth_vals == 1)))
    fp = int(np.sum((pred_vals == 1) & (truth_vals == 0)))
    fn = int(np.sum((pred_vals == 0) & (truth_vals == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_base_pairs": int(np.triu(truth_bin, k=1).sum()),
        "predicted_base_pairs": int(np.triu(pred_bin, k=1).sum()),
    }


def _load_matrix(path: str, preferred_keys=("binary_map", "adj", "dependency_map")) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    for key in preferred_keys:
        if key in data:
            return np.asarray(data[key])
    # Fall back to first array
    key = list(data.keys())[0]
    return np.asarray(data[key])


def main():
    parser = argparse.ArgumentParser(
        description="Compute precision, recall, and F1 for binary pairing matrices"
    )
    parser.add_argument(
        "--pred", type=str, required=True, help="Predicted binary map NPZ"
    )
    parser.add_argument(
        "--truth", type=str, required=True, help="Ground-truth adjacency NPZ"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional JSON output path"
    )

    args = parser.parse_args()

    pred = _load_matrix(args.pred)
    truth = _load_matrix(args.truth, preferred_keys=("adj", "binary_map", "dependency_map"))

    if pred.shape != truth.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs truth {truth.shape}"
        )

    metrics = compute_f1(pred, truth)

    print(f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print(f"Precision={metrics['precision']:.4f}")
    print(f"Recall={metrics['recall']:.4f}")
    print(f"F1={metrics['f1']:.4f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
