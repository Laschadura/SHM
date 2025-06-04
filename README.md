> ⚠️ **Work in progress** – This repository is under active development. APIs, configs, and model logic may change without notice.

# 🧠 Structural Damage Modeling with Spectral MMVAE & Latent Diffusion

This repository contains modular deep learning pipelines for **structural damage detection and data synthesis** using multimodal input data like accelerometer spectrograms and spatial crack masks.

It combines two complementary approaches:

1. **Spectral MMVAE** – A Multimodal Variational Autoencoder for realistic data synthesis.
2. **Multi-Modal Latent Diffusion (MLD)** – A generative diffusion-based model for realistic data synthesis.

Both models aim to learn, in a multimodal way, to generate synthetic vibrational data and corresponding damage scenarios as pairs, using real-world measurements from a masonry arch bridge.

> 📌 This project is an extension of my semester thesis at ETH Zürich.
> 📚 The dataset and experimental foundation were provided by [Liu et al., 2024](https://doi.org/10.1016/j.engstruct.2024.118466), *“Full life-cycle vibration-based monitoring of a full-scale masonry arch bridge with increasing levels of damage”*, Engineering Structures 315.

---

## 📁 Repository Structure

This repository is organized as follows:

```text
DataSynthSHM/
├── configs/                # OmegaConf YAMLs for model configs
├── scripts/                # SLURM cluster launch scripts
├── src/
│   ├── bridge_data/        # Shared preprocessing, transforms, IO
│   ├── diff_pt/            # PyTorch latent diffusion model
│   └── mmvae_tf/           # TensorFlow-based Spectral MMVAE
```

Additional folders that are generated locally or used during training (not tracked in Git):

```text
Data/                       # Raw accelerometer CSVs (not included)
Labels/                     # Raw label masks (not included)
cache/                      # Cached spectrograms, masks, etc.
results_diff/               # Diffusion model outputs and logs
results_mmvae/              # MMVAE model outputs and logs
logs/                       # Training logs
```

These folders are ignored via `.gitignore` to reduce repo size and preserve data privacy. To run the pipeline, users must place raw data in the expected structure and allow the pipeline to regenerate cached features.

---

## 🧰 Features

* Multimodal learning from spectrograms and crack masks
* Modular, extensible design for model components
* Latent diffusion model with conditional sampling
* Framework-hybrid: TensorFlow (MMVAE), PyTorch (Diffusion)
* Compatible with multi-GPU training (MirroredStrategy/DDP)
* Config-driven (OmegaConf-based YAMLs)

---

## 1⃣ Spectral MMVAE (TensorFlow)

**Module tests (example):**

```bash
python -m mmvae_tf.tests.test_MoE_MMVAE
```

---

## 2⃣ Latent Diffusion Model (PyTorch)

**Module tests (example):**

```bash
python -m diff_pt.tests.test_reconstruction --ckpt_dir results_diff --n_segments 1
```

---

## 🗂 Data Format

Raw input:

```
Data/
├── Data/Test_*/accel_*.csv        # Time-series accelerometer data of 12 sensors
├── Labels/Test_*/label_*.png      # Images of the damage on the bridge
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
MSc ETH Zürich

---

## 📄 License

This repository is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute the code with attribution.

---

## 📌 Note

This repository is under active development. Expect changes in model structure, training logic, and evaluation over time.
