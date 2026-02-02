# ICLR Reproduce: The Price of Amortized Inference in Sparse Autoencoders

This repository contains code to reproduce the experiments in the paper **"The Price of Amortized Inference in Sparse Autoencoders"** (ICLR). It includes:

- **§4.3 Training dynamics** — NMSE, dead/dense latents, splitting, absorption, amortization gap across checkpoints.
- **§5 Main comparison** — Fully / semi / non-amortized inference across SAE variants (Standard, Top-K, Gated, **BatchTopK**, **Matryoshka BatchTopK**).
- **Loca-SAE (Loca-Amortization SAE)** — Training and evaluation of the local-amortization SAE method.

---

## Repository structure

```
ICLR-Reproduce/
├── README.md
├── requirements.txt
├── configs/                    # YAML configs for collection & evaluation
├── sae_amortization/           # Core evaluation (run_exp5, index_checkpoints, collect_activations, ...)
├── local_eval/                 # Streaming activations, SAE loading, inference modes
├── loca_sae/                   # Loca-SAE training pipeline (config, data, models, train_sae, convert)
└── scripts/                    # Entry scripts
    ├── run_new_baselines.py    # Single SAE: three modes (amortized / semi / non)
    ├── run_loc_amt_eval.py     # Evaluate Loca-SAE (amortized only)
    └── run_semi_step_sweep.py  # Semi-amortized step sweep
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/ICLR-Reproduce.git
cd ICLR-Reproduce
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** Python ≥3.10, PyTorch ≥2.1, transformers, datasets, scikit-learn, tqdm, pyyaml, pandas, safetensors, pyarrow.

---

## Data & models

- **Language models:** EleutherAI/pythia-160m-deduped (layer 8), google/gemma-2-2b (layer 12).
- **Corpus:** monology/pile-uncopyrighted (streaming) or ag_news for collection.
- **SAE checkpoints:** You need Standard / Top-K / BatchTopK / Matryoshka BatchTopK checkpoints (train or download from SAEBench / your own). Place them in directories and point `sae_checkpoints` in config to those roots.

---

## Reproducing experiments

**All commands below are run from the repository root `ICLR-Reproduce/`.**

### 1) §4.3 — Training dynamics (batch evaluation over checkpoints)

**A. Collect activations**

Create a YAML config (e.g. `configs/collect_pythia.yaml`) with `output_dir`, `model.name`, `model.layer_index`, and `dataset` (see `configs/collect_agnews.yaml`). Then:

```bash
python -m sae_amortization.collect_activations --config configs/collect_pythia.yaml
```

This writes `output_dir/acts_layer{L}.npy` (e.g. `outputs/pythia160m_L8/acts_layer8.npy`).

**B. Build checkpoint index**

Use the same or another config that includes `output_dir` and `sae_checkpoints` with variant names and `roots` (paths to folders containing `.pt` / `.safetensors`):

```yaml
output_dir: "outputs/pythia160m_L8"
sae_checkpoints:
  standard:
    name: "standard"
    roots: ["path/to/standard_sae"]
  topk:
    name: "topk"
    roots: ["path/to/topk_sae"]
  batch_topk:
    name: "batch_topk"
    roots: ["batch_topk/pythia-160m"]
  matryoshka_batch_topk:
    name: "matryoshka_batch_topk"
    roots: ["matryoshka_batch_topk/pythia-160m"]
```

Then:

```bash
python -m sae_amortization.index_checkpoints --config configs/eval_three_modes_example.yaml
```

This creates `output_dir/checkpoint_index.parquet`.

**C. Run evaluation (amortized only or all three modes)**

To evaluate all indexed checkpoints with **amortized**, **semi_amortized**, and **non_amortized** (rebuttal-style):

```yaml
# In the same config
eval_modes:
  - amortized
  - semi_amortized
  - non_amortized
semi_steps: 30
non_steps: 200
target_dense_semi_non: 0.1
```

Run:

```bash
python -m sae_amortization.run_exp5 --config configs/eval_three_modes_example.yaml
```

Output: `output_dir/metrics.parquet` (one row per checkpoint × mode, with NMSE, dead_rate, dense@0.1/0.2, F1@1, F1@2, dF1, absorption, amort_gap_*).

**D. Visualize**

```bash
python -m sae_amortization.visualize --config configs/eval_three_modes_example.yaml
```

Plots are saved under `output_dir/fig_*.png`.

---

### 2) §5 — Single SAE, three modes (streaming activations)

If you have **one** BatchTopK or Matryoshka BatchTopK checkpoint and want amortized / semi / non metrics **without** building an index:

1. Set `SAE_PATHS` in `local_eval/config.py` for `(model_name, sae_variant)`, or pass `--sae_path`.
2. Run from repo root:

```bash
python scripts/run_new_baselines.py --model_name pythia-160m-deduped --sae_variant batch_topk --out_dir new_baseline_results
```

This streams activations, loads the SAE, runs three modes (λ calibrated to match amortized density), and writes `new_baseline_results/{model_name}_{sae_variant}_3modes.json`.

---

### 3) Semi-amortized step sweep

Vary the number of ISTA steps starting from the amortized code:

```bash
python scripts/run_semi_step_sweep.py --model_name pythia-160m-deduped --steps 5,10,15,20,25,30 --out_dir semi_sweep_results
```

Uses BatchTopK SAE from `SAE_PATHS[(model_name, "batch_topk")]` unless `--sae_path` is given. Output: `semi_sweep_results/{model_name}_batch_topk_semi_sweep.json`.

---

### 4) Loca-SAE training and evaluation

**A. Train Loca-SAE**

From repo root:

```bash
# Pythia-160M (output: loca_sae/loc-amt-sae/pythia-160m/ by default inside the package)
python -m loca_sae.train_pythia160m
# Or Gemma-2-2B
python -m loca_sae.train_gemma2_2b
```

To change the output directory, edit `output_dir` in `loca_sae/train_pythia160m.py` or `loca_sae/train_gemma2_2b.py`. Training writes `standard_sae.pt`, `multi_encoder_sae.pt`, `loc_angular_meta.pt`, and `config.json`.

**B. Convert multi-encoder to single ae.pt**

Edit `sae_dir` in `loca_sae/convert_multi_to_saebench.py` to point to your trained folder (e.g. `loca_sae/loc-amt-sae/pythia-160m`), then:

```bash
python -m loca_sae.convert_multi_to_saebench
```

This creates `ae.pt` (W_enc, W_dec) in that folder.

**C. Evaluate Loca-SAE**

Set default paths in `scripts/run_loc_amt_eval.py` (`DEFAULT_LOC_AMT_SAE_PATHS`) to your `ae.pt`, or pass `--sae_path`:

```bash
python scripts/run_loc_amt_eval.py --model_name pythia-160m-deduped --out_dir loc_amt_eval_results
```

Writes `loc_amt_eval_results/{model_name}_loc_amt_sae_metrics.json` (same 8 metrics, amortized only).

---

## Config examples

- **configs/collect_agnews.yaml** — Collect activations (dataset, model, output_dir).
- **configs/eval_three_modes_example.yaml** — Full evaluation with multiple variants and `eval_modes: [amortized, semi_amortized, non_amortized]`. Adjust `sae_checkpoints.roots` to your checkpoint paths.

---

## Metrics (unified protocol)

All evaluation uses the same protocol:

- **NMSE** — Normalized MSE of reconstruction (Z @ D^T vs X).
- **Dead rate** — Fraction of latents with firing rate ≤ 1e-6.
- **Dense rate @0.1 / @0.2** — Fraction of latents with firing rate ≥ 0.1 / 0.2.
- **F1@1, F1@2, ΔF1** — k-sparse probe F1 (correlation-based feature order, balanced LR).
- **Absorption rate** — On positive examples, fraction where main latent did not fire but aux latents did (correlation-based main/aux).
- **Amortization gap** (optional) — L(z_amort) − L(z_opt) with ISTA-200; reported only for amortized mode when enabled in config.

---

## License

MIT. See the paper and original repo for attribution.
