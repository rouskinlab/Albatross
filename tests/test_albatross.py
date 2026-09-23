"""Tests for Albatross masking, filters, Blossom, metrics, and CLIs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from albatross.blossom import GraphSanitizer, apply_blossom, remove_pseudoknots_from_matrix
from albatross.filters import HeuristicPixel, apply_filters
from albatross.metrics import compute_f1
from albatross.training import IRESDataset
from rinalmo.data.alphabet import Alphabet


# ---------------------------------------------------------------------------
# Masking (historical independent 15% draws)
# ---------------------------------------------------------------------------

def test_masking_produces_labels_and_corruption():
    alphabet = Alphabet()
    seq = "ACGTACGTACGTACGTACGT"  # 20 nt
    ds = IRESDataset([seq], alphabet, mask_ratio=0.15, max_length=64)
    item = ds[0]
    labels = item["labels"]
    inputs = item["input_ids"]

    # Some positions contribute to loss
    n_loss = int((labels != -100).sum())
    assert n_loss >= 1

    # CLS / EOS never contribute to loss
    assert labels[0].item() == -100
    assert labels[-1].item() == -100

    # Sequence length = CLS + seq + EOS
    assert len(inputs) == len(seq) + 2


def test_masking_independent_draws_can_diverge():
    """Corruption positions and loss positions are separate randperms."""
    alphabet = Alphabet()
    # Long enough that independent 15% draws usually disagree
    seq = "A" * 200
    ds = IRESDataset([seq], alphabet, mask_ratio=0.15, max_length=512)

    torch.manual_seed(0)
    diverged = False
    for _ in range(20):
        item = ds[0]
        labels = item["labels"]
        inputs = item["input_ids"]
        loss_pos = set((labels != -100).nonzero(as_tuple=True)[0].tolist())
        # Positions that were corrupted (mask or random) relative to original
        # Reconstruct original tokens
        original = [alphabet.cls_idx] + [alphabet.get_idx(c) for c in seq] + [alphabet.eos_idx]
        original = torch.tensor(original)
        corrupted = set((inputs != original).nonzero(as_tuple=True)[0].tolist())
        if loss_pos != corrupted and loss_pos and corrupted:
            diverged = True
            break
    # Not guaranteed every seed diverges, but over 20 trials it should
    assert diverged


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def test_apply_filters_normalization_and_defaults():
    # Build a raw map with a clear stem of length 3 off-diagonal
    N = 20
    raw = np.zeros((N, N), dtype=float)
    # High scores for stem positions (i,j) with |i-j| > 4 and run length >= 3
    for k in range(3):
        i, j = 2 + k, 15 - k
        raw[i, j] = 15.0
        raw[j, i] = 15.0

    filtered = apply_filters(raw, alpha=3, tau=0.11, gamma=4, reference_max=20.0)
    assert filtered.shape == (N, N)
    # Stem should survive
    assert filtered[2, 15] > 0
    # Diagonal band should be zero
    assert filtered[5, 5] == 0
    assert filtered[5, 6] == 0


def test_heuristic_pixel_drops_short_runs():
    N = 16
    m = np.zeros((N, N), dtype=float)
    # Single isolated pair far from diagonal — too short for alpha=3
    m[2, 12] = 1.0
    m[12, 2] = 1.0
    out = HeuristicPixel().denoise(m, alpha=3, epsilon=0, tau=0.1, gamma=4)
    assert np.sum(out) == 0


# ---------------------------------------------------------------------------
# Blossom + pseudoknots
# ---------------------------------------------------------------------------

def test_blossom_one_to_one_pairing():
    N = 10
    raw = np.zeros((N, N), dtype=float)
    # Two competing edges from node 0
    raw[0, 5] = 0.9
    raw[5, 0] = 0.9
    raw[0, 7] = 0.5
    raw[7, 0] = 0.5
    # Independent pair
    raw[2, 8] = 0.8
    raw[8, 2] = 0.8

    binary = apply_blossom(raw, remove_pseudoknots=False)
    # Node 0 pairs with at most one partner
    assert binary[0].sum() == 1
    assert binary[0, 5] == 1  # higher weight wins
    assert binary[2, 8] == 1


def test_pseudoknot_removal():
    N = 12
    binary = np.zeros((N, N), dtype=float)
    # Nested: (1,10) and (3,8)
    binary[1, 10] = binary[10, 1] = 1
    binary[3, 8] = binary[8, 3] = 1
    # Crossing: (2,6) crosses (3,8)? 2<3<6<8 — yes crossing
    binary[2, 6] = binary[6, 2] = 1

    clean = remove_pseudoknots_from_matrix(binary)
    # Longest span (1,10) kept; (3,8) nested inside it kept;
    # (2,6) crosses (3,8) — one of the crossing pair is dropped
    pairs = [
        (i, j)
        for i in range(N)
        for j in range(i + 1, N)
        if clean[i, j] > 0
    ]
    # No two pairs should cross
    for (a1, b1) in pairs:
        for (a2, b2) in pairs:
            if (a1, b1) >= (a2, b2):
                continue
            assert not ((a1 < a2 < b1 < b2) or (a2 < a1 < b2 < b1))


def test_graph_sanitizer_with_mask():
    N = 8
    raw = np.ones((N, N), dtype=float) * 0.5
    np.fill_diagonal(raw, 0)
    mask = np.zeros((N, N), dtype=float)
    mask[1, 6] = mask[6, 1] = 0.8
    mask[2, 5] = mask[5, 2] = 0.7

    binary = GraphSanitizer(remove_pseudoknots=True).sanitize(raw, mask=mask)
    assert binary[1, 6] == 1 or binary[2, 5] == 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_compute_f1_perfect():
    truth = np.zeros((6, 6))
    truth[0, 5] = truth[5, 0] = 1
    truth[1, 4] = truth[4, 1] = 1
    m = compute_f1(truth, truth)
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_compute_f1_partial():
    truth = np.zeros((6, 6))
    truth[0, 5] = truth[5, 0] = 1
    truth[1, 4] = truth[4, 1] = 1

    pred = np.zeros((6, 6))
    pred[0, 5] = pred[5, 0] = 1  # TP
    pred[2, 3] = pred[3, 2] = 1  # FP

    m = compute_f1(pred, truth)
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Dependency map math with stub model
# ---------------------------------------------------------------------------

class _StubModel:
    """Returns logits that prefer the input base at each position."""

    def __call__(self, tokens):
        # tokens: (B, L) with CLS/EOS; vocab has ACGU at indices 5,6,7,8
        B, L = tokens.shape
        logits = torch.zeros(B, L, 22)
        for b in range(B):
            for i in range(L):
                t = int(tokens[b, i])
                if t in (5, 6, 7, 8):
                    logits[b, i, t] = 5.0
                else:
                    logits[b, i, 5:9] = 1.0
        return {"logits": logits}


class _StubAlphabet:
    def batch_tokenize(self, seqs):
        # Encode U->T style: A=5,C=6,G=7,T/U=8, with CLS=0 EOS=2
        table = {"A": 5, "C": 6, "G": 7, "U": 8, "T": 8}
        max_len = max(len(s) for s in seqs) + 2
        batch = []
        for s in seqs:
            enc = [0] + [table[c] for c in s.upper().replace("T", "U")] + [2]
            enc += [1] * (max_len - len(enc))
            batch.append(enc)
        return batch


def test_dependency_map_with_stub():
    from albatross.dependency_map import compute_dependency_map

    seq = "ACGUACGU"
    model = _StubModel()
    alphabet = _StubAlphabet()
    dep = compute_dependency_map(seq, model, alphabet, device="cpu", batch_size=16)
    assert dep.shape == (len(seq), len(seq))
    assert np.allclose(np.diag(dep), 0)
    assert np.isfinite(dep).all()


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module",
    [
        "albatross.training",
        "albatross.dependency_map",
        "albatross.filters",
        "albatross.blossom",
        "albatross.metrics",
    ],
)
def test_cli_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower() or "Usage" in result.stdout or len(result.stdout) > 0


def test_filters_and_blossom_cli_roundtrip(tmp_path):
    N = 16
    raw = np.zeros((N, N), dtype=float)
    for k in range(4):
        i, j = 1 + k, 14 - k
        raw[i, j] = raw[j, i] = 12.0

    inp = tmp_path / "raw.npz"
    np.savez(inp, dependency_map=raw, sequence="A" * N)

    filtered = tmp_path / "filtered.npz"
    binary = tmp_path / "binary.npz"

    r1 = subprocess.run(
        [
            sys.executable, "-m", "albatross.filters",
            "--input", str(inp),
            "--output", str(filtered),
            "--alpha", "3", "--tau", "0.11", "--gamma", "4",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r1.returncode == 0, r1.stderr

    r2 = subprocess.run(
        [
            sys.executable, "-m", "albatross.blossom",
            "--input", str(filtered),
            "--output", str(binary),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 0, r2.stderr
    assert binary.exists()
    data = np.load(binary)
    assert "binary_map" in data


def test_metrics_cli(tmp_path):
    truth = np.zeros((6, 6))
    truth[0, 5] = truth[5, 0] = 1
    pred = truth.copy()

    tpath = tmp_path / "truth.npz"
    ppath = tmp_path / "pred.npz"
    opath = tmp_path / "metrics.json"
    np.savez(tpath, adj=truth)
    np.savez(ppath, binary_map=pred)

    r = subprocess.run(
        [
            sys.executable, "-m", "albatross.metrics",
            "--pred", str(ppath),
            "--truth", str(tpath),
            "--output", str(opath),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    metrics = json.loads(opath.read_text())
    assert metrics["f1"] == 1.0
