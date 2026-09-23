# Albatross

Code accompanying **An RNA language model trained on sequence alone reveals IRES structures** (Rouskin Lab, Harvard Medical School).

Albatross continues pretraining of [RiNALMo](https://github.com/lbcb-sci/RiNALMo) (giga, 650M) on IRES sequences, then converts model log-odds into dependency maps and nested secondary structures.

**Dependency maps and predicted structures for 75,229 full-length IRESes:** [https://albatrossrna.org/](https://albatrossrna.org/)

## Repository layout

```
albatross/
  training.py         # Continued MLM pretraining
  dependency_map.py   # Raw dependency maps (3N+1 forwards, log₂ odds)
  filters.py          # HeuristicPixel filters (α, τ, γ)
  blossom.py          # Edmonds' Blossom + greedy pseudoknot removal
  metrics.py          # Upper-triangle precision / recall / F1
rinalmo/              # Minimal RiNALMo model, tokenizer, and weight loader
```

Each file maps to a STAR Methods section of the manuscript.

| Module | Methods section |
|--------|-----------------|
| `training.py` | Model Training |
| `dependency_map.py` | Dependency Map Generation |
| `filters.py` | Binary Structure Filtering (Algorithm 1, steps 1–5) |
| `blossom.py` | Binary Structure Filtering (Algorithm 1, steps 6–7) |
| `metrics.py` | Structure comparison |

## Installation

For filtering, Blossom matching, metrics, CLI inspection, and tests, a GPU is
not required:

```bash
git clone https://github.com/rouskinlab/Albatross.git
cd Albatross
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

Training and dependency-map inference use the 650M-parameter RiNALMo giga
model and require a Linux NVIDIA CUDA environment with `flash-attn`:

```bash
conda env create -f environment.yml
conda activate albatross

# If flash-attn was not built successfully during environment creation:
python -m pip install flash-attn==2.3.2 --no-build-isolation
python -m pip install -e .
```

No repository path is hard-coded. Commands may be run from any clone location
after the package is installed.

## Quick start

Paper defaults for binary filtering: **α=3, τ=0.11, γ=4**, then Blossom with pseudoknot removal.

### 1. Continued pretraining

```bash
python -m albatross.training data/sequences.txt \
  --pretrained_weights weights/rinalmo_giga_pretrained.pt \
  --lr 5e-6 --batch_size 8 --warmup_steps 10000 \
  --max_epochs 10 --seed 42 --output_dir outputs/
```

Official RiNALMo giga weights can also be downloaded automatically via `rinalmo.pretrained.get_pretrained_model("giga-v1")`. The Albatross checkpoint released with the paper is the 50k run at epoch 5, step 39,096.

### 2. Dependency map

```bash
python -m albatross.dependency_map \
  --sequence AUGC... \
  --checkpoint path/to/albatross.ckpt \
  --out results/example
```

Writes `results/example.npz` with key `dependency_map` (raw, directed, not symmetrized).

### 3. Filters (HeuristicPixel)

```bash
python -m albatross.filters \
  --input results/example.npz \
  --output results/example_filtered.npz \
  --alpha 3 --tau 0.11 --gamma 4
```

### 4. Blossom matching + pseudoknot removal

```bash
python -m albatross.blossom \
  --input results/example_filtered.npz \
  --output results/example_binary.npz
```

### 5. Metrics

```bash
python -m albatross.metrics \
  --pred results/example_binary.npz \
  --truth path/to/ground_truth.npz
```

## Data and weights

This repository ships **code only**. Large assets are external:

| Asset | Where |
|-------|--------|
| 75,229 dependency maps & structures | [albatrossrna.org](https://albatrossrna.org/) |
| Official RiNALMo giga weights | Auto-download via `get_pretrained_model("giga-v1")` |
| Albatross fine-tuned checkpoint | Released with the manuscript / Zenodo |
| Training & evaluation sequences | Released with the manuscript / Zenodo |

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Tests cover masking, filtering, Blossom/pseudoknot removal, metrics, and CLI entry points. They do not require a GPU or model weights.

The release is also tested by building a wheel from a fresh clone, installing
it into a new virtual environment, running every CLI from outside the source
tree, and completing a synthetic dependency-map filtering → Blossom → F1
round trip.

## Citation

If you use this code, please cite the Albatross manuscript and the original RiNALMo paper. See `NOTICE` for license attribution.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`. RiNALMo model parameters are CC BY 4.0.
