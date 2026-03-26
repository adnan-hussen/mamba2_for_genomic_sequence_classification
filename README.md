# 🧬 RC-Equivariant Mamba-2 for Genomic Sequence Classification

A PyTorch/TensorFlow implementation of a **reverse-complement (RC) equivariant Mamba-2** model for DNA sequence classification, benchmarked on the [Genomic Benchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) `demo_human_or_worm` dataset.

---

## 📋 Overview

This notebook trains a structured state space model (SSM) — based on the **Mamba-2 (SSD)** architecture — to classify genomic DNA sequences. The model incorporates **reverse-complement symmetry** by averaging predictions from both the forward strand and its reverse complement, making it biologically aware of DNA's double-stranded nature.

| Property | Value |
|---|---|
| Task | Binary classification (human vs. worm DNA) |
| Architecture | RC-equivariant Mamba-2 |
| Dataset | `demo_human_or_worm` (75,000 training sequences) |
| Final Accuracy | ~**95.2%** (5 epochs) |
| Framework | TensorFlow / Keras + mamba-ssm |

---

## 🏗️ Architecture

```
Input DNA Sequence
       │
  [Tokenizer]          A→1, C→2, G→3, T→4, N→0
       │
  ┌────┴────────────────────┐
  │ Forward Branch          │  RC Branch (reverse complement)
  │  Embedding              │   Embedding
  │  Mamba2Block × n_layers │   Mamba2Block × n_layers
  │  GlobalAvgPool          │   GlobalAvgPool
  └────────────┬────────────┘
               │  (feat_fwd + feat_rc) / 2
          LayerNorm
          Dense (softmax)
               │
          Class Probabilities
```

### Mamba2Block

Each block implements a simplified **SSD (Selective State Space Dual)** recurrence:

1. Input projection → split into `x` and `z` gating branches
2. Depthwise Conv1D + SiLU activation
3. Simplified selective recurrence (`A = 0.9`)
4. Gated output via `x * SiLU(z)`
5. Output projection back to `d_model`

---

## ⚙️ Installation

```bash
# Install PyTorch (CUDA 12.1)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install Mamba-2 SSM kernels
pip install causal-conv1d mamba-ssm --no-build-isolation

# Install Genomic Benchmarks
pip install genomic-benchmarks
```

> **Note:** A CUDA-capable GPU is required to build `causal-conv1d` and `mamba-ssm`.

---

## 🚀 Usage

Run all cells in the notebook, or execute the training script directly:

```python
prepare_and_train()
```

This will:
1. Download the `demo_human_or_worm` dataset via `genomic_benchmarks`
2. Tokenize raw DNA strings into integer tensors
3. Build and compile the `RCMamba2Model`
4. Train for 5 epochs with the Adam optimizer

---

## 📊 Training Results

| Epoch | Accuracy | Loss |
|-------|----------|------|
| 1 | 79.30% | 0.4097 |
| 2 | 94.25% | 0.1511 |
| 3 | 94.69% | 0.1387 |
| 4 | 94.91% | 0.1321 |
| 5 | **95.16%** | **0.1265** |

Training time: ~25s (epoch 1), ~14s/epoch thereafter on GPU.

---

## 🧪 Model Configuration

```python
RCMamba2Model(
    vocab_size = 5,      # A, C, G, T, N
    d_model    = 64,     # embedding / hidden dimension
    n_layers   = 2,      # number of Mamba2 blocks
    n_classes  = 2       # human or worm
)
```

---

## 📁 Project Structure

```
├── notebook.ipynb          # Main Kaggle notebook
├── README.md               # This file
```

---

## 🔬 Background

- **Mamba / Mamba-2**: Selective state space models that achieve linear-time sequence modeling. See [Gu & Dao, 2023](https://arxiv.org/abs/2312.00752) and [Dao & Gu, 2024](https://arxiv.org/abs/2405.21060).
- **RC-Equivariance**: DNA is double-stranded; a model that processes both strands symmetrically is more biologically principled and empirically stronger.
- **Genomic Benchmarks**: A standardized suite of classification tasks for benchmarking sequence models on genomic data.

---

## 📦 Dependencies

| Package | Version |
|---|---|
| `torch` | 2.4.0 (cu121) |
| `causal-conv1d` | 1.6.0 |
| `mamba-ssm` | 2.3.0 |
| `tensorflow` / `keras` | ≥2.x |
| `genomic-benchmarks` | 1.0.0 |
| `numpy` | ≥1.17 |
| `pandas` | ≥1.1.4 |

---

## 📄 License

MIT License. See `LICENSE` for details.
