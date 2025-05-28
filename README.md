> ⚠️ **Work in progress** – This repository is under active development. APIs, configs, and model logic may change without notice.

# 🧠 Structural Damage Modeling with Spectral MMVAE & Latent Diffusion

This repository contains modular deep learning pipelines for **structural damage detection and data synthesis** from multimodal inputs such as time-frequency spectrograms and crack masks.

It combines two complementary approaches:

1. **Spectral MMVAE** – A Multimodal Variational Autoencoder for realistic data synthesis.
2. **Multi-Modal Latent Diffusion (MLD)** – A generative diffusion-based model for realistic data synthesis.
Both try to archieve the same goal. Learn, in a multimodal way, to generate synthetic vabriational data and corresponding damage scenarios as pairs based on real world data from a masonry arch bridge.

---

## 📁 Repository Structure

```text
DataSynthSHM/
├── configs/                # OmegaConf YAMLs for model configs
├── scripts/                # SLURM cluster launch scripts
├── src/
│   ├── bridge_data/        # Shared preprocessing, transforms, IO
│   ├── diff_pt/            # PyTorch latent diffusion model
│   │   ├── model.py        # Full diffusion model architecture
│   │   ├── train.py        # Training loop and checkpointing
│   │   ├── run.py          # CLI entry point
│   │   ├── io.py, losses.py, utils.py, vis.py
│   │   └── tests/          # Evaluation & reconstruction scripts
│   └── mmvae_tf/           # TensorFlow-based Spectral MMVAE
│       ├── model.py        # Encoders, decoders, and MoE logic
│       ├── train.py        # MMVAE training logic and loop
│       ├── run.py          # CLI entry point
│       ├── losses.py, utils.py, env.py, io.py, vis.py
│       └── tests/          # VAE test & visualization utilities
├── cache/                  # Cached .npy/.pkl data
├── results_diff/           # Diffusion model logs/outputs
├── results_mmvae/          # MMVAE model logs/outputs
├── logs/                   # Shared training logs
└── setup.py                # Editable install
```

---

## 🧰 Features

* Multimodal learning from spectrograms and crack masks
* Modular, extensible design for model components
* Latent diffusion model with conditional sampling
* Framework-hybrid: TensorFlow (MMVAE), PyTorch (Diffusion)
* Compatible with multi-GPU training (MirroredStrategy/DDP)
* Config-driven (OmegaConf-based YAMLs)

---

## 1⃣ Spectral MMVAE (TensorFlow)\$1

**Module tests (example):**

```bash
python -m mmvae_tf.tests.test_MoE_MMVAE
```

---

## 2⃣ Latent Diffusion Model (PyTorch)\$1

**Module tests (example):**

```bash
python -m diff_pt.tests.test_reconstruction --ckpt_dir results_diff --n_segments 1
```

---

## 🗂 Data Format

Raw input:

```
Data/
├── Data/Test_*/Accel_*.csv        # Time-series accelerometer input
├── Labels/Test_*/mask_*.png       # Crack masks
```

Cached and normalized data is stored in `cache/` based on config-defined segment duration and STFT parameters.

---

## 🧪 Package Installation & Usage

Install the project as an **editable package** from the repo root:

```bash
pip install -e .
```

This enables:

* Running model code using `python -m mmvae_tf.run` or `diff_pt.run`
* Cross-folder imports between `bridge_data`, `diff_pt`, and `mmvae_tf`
* Reusable modules without reinstalling after edits

> ⚠️ This is required both locally and on the ETH Euler cluster.

---

## 🚀 Running on ETH Euler Cluster

### ✅ Upload Your Code

Use `rsync` to upload your local folder to `/cluster/scratch/<username>/`:

```bash
rsync -av --progress ./DataSynthSHM/ <username>@euler.ethz.ch:/cluster/scratch/<username>/DataSynthSHM/
```

### ✅ Launch a Job with SLURM

```bash
cd /cluster/scratch/<username>/DataSynthSHM
sbatch scripts/diff_run.slurm         # or run_vae.slurm for MMVAE
```

### ✅ Monitor Your Job

```bash
squeue -u $USER                       # show your running jobs
scontrol show job <job_id>           # detailed info
seff <job_id>                        # efficiency summary
```

### ✅ Debug Output

```bash
tail -f diff_run.out           # live stdout from job
cat diff_run.err               # print error output
```

See the full cheatsheet in `Euler_Cluster_cheatsheet.txt` for more.

---

## 🛠 Requirements

* Python ≥ 3.10
* TensorFlow ≥ 2.15
* PyTorch ≥ 2.0
* NumPy, SciPy, OpenCV, Plotly, UMAP, Matplotlib, tqdm

Install with:

```bash
pip install -r requirements.txt
```

---

## 🧠 Author

Developed by **Simon Scandella**
MSc ETH Zürich – Structural Health Monitoring & Machine Learning

---

## 📌 Note

This repository is under active development. Expect changes in model structure, training logic, and evaluation over time.
