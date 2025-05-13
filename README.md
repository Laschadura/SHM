# 🧠 Structural Damage Modeling with Spectral MMVAE & Latent Diffusion

This repository contains two complementary deep learning pipelines for **structural damage detection and synthesis** using multimodal input:

1. **Spectral MMVAE** — A Multimodal Variational Autoencoder for reconstructing and analyzing real-world bridge damage.
2. **Multi-Modal Latent Diffusion (MLD)** — A generative model for synthesizing realistic damage states in a shared latent space.

---

## 📆 Repository Structure

```
Euler_MMVAE/
├── scripts/
│   ├── vae_generator.py         # Spectral MMVAE training
│   ├── diffusion_model.py       # Latent diffusion training
│   ├── test_diffusion.py        # Evaluation & synthesis for MLD
│   ├── Gen_VAE_Test.py          # Evaluation for MMVAE
│   ├── data_loader.py           # Shared: I/O, augmentation, ISTFT, caching
│   ├── losses.py                # Shared: waveform, mask, and phase losses
│   ├── custom_distributions.py  # MMVAE-specific: JS divergence and latent mixing
│
├── cache/                       # Preprocessed .npy: spec, masks, wave, etc.
├── data/                        # Raw Test_* folders (spectrograms + labels)
├── results_mmvae/               # Saved MMVAE models and logs
├── results_diff/                # Diffusion models, reconstructions, plots
├── logs/                        # Training curves, beta schedules, metrics
└── README.md
```

---

## 🔧 Key Features

* ✅ Multimodal learning (spectrograms & spatial masks)
* ✅ Jensen-Shannon divergence & latent regularization
* ✅ Waveform reconstruction from spectrogram via ISTFT
* ✅ Score-based latent diffusion with conditioned sampling
* ✅ Modular training + eval scripts for both models
* ✅ Data augmentation (SpecAug, pink noise, sign flip, time shift)
* ✅ Multi-GPU compatible (TF MirroredStrategy / PyTorch DDP-ready)

---

## 1⃣ Spectral MMVAE

### 📚 Description

A Variational Autoencoder that learns a **shared latent space** from time-frequency spectrograms and binary crack masks. It allows semi-supervised learning of damage progression, using JS divergence and reconstruction losses across modalities.

### 🔥 Training

```bash
python scripts/vae_generator.py
```

Includes:

* Waveform L1 + SI-L1 loss
* Multi-resolution STFT (MRSTFT) loss
* Phase gradient + Laplacian
* Binary mask loss (Dice + BCE)
* Damage supervision via global mask mean

### 📊 Evaluation

```bash
python scripts/Gen_VAE_Test.py
```

Visualizes:

* Reconstruction comparison
* Interpolation
* Latent UMAPs

---

## 2⃣ Latent Diffusion Model (MLD)

### 📚 Description

Autoencoders compress spectrogram and mask inputs into latent codes, then a **score-based diffusion model** learns to denoise and reconstruct samples. It supports conditional generation (e.g., generate mask from spec).

### 🔥 Training

Train autoencoders and the diffusion model:

```bash
python scripts/diffusion_model.py
```

Set flags:

```python
train_AE = True
dm_mode  = "scratch"  # or "continue"
```

### 📊 Evaluation

```bash
python scripts/test_diffusion.py --ckpt_dir results_diff --batch 32 --samples 8
```

Visualizes:

* Time-series vs reconstruction overlays
* Mask prediction and upsampling
* PSD metrics and waveform plots
* Latent space UMAP & FID
* Sample interpolation

---

## 📁 Data Preparation

Expected directory:

```
data/
├── Data/
│   ├── Test_01/*.csv       # Accelerometer waveforms
├── Labels/
│   ├── Test_01/*.png       # Crack masks (RGB, 512x1536)
```

Preprocessed data is cached as:

* `cache/spectrograms.npy`
* `cache/masks.npy`
* `cache/segments.npy`
* `cache/test_ids.npy`

---

## 🧪 Shared Losses

Implemented in `losses.py`:

* `waveform_l1_loss`, `waveform_si_l1_loss`
* `multi_channel_mrstft_loss`
* `gradient_loss_phase_only`, `laplacian_loss_phase_only`
* `magnitude_l1_loss`
* `custom_mask_loss` (Dice, BCE, optional Focal)

---

## 🛠 Requirements

* Python ≥ 3.10
* TensorFlow ≥ 2.15
* PyTorch ≥ 2.0
* NumPy, SciPy, Matplotlib, OpenCV, Plotly
* UMAP, tqdm, scikit-learn

Use your own `requirements.txt` to install all packages:

```bash
pip install -r requirements.txt
```

---

## 📊 Logging

* Training logs: `logs/beta_tracking.csv`, training curves
* Models: `.keras` (TF) and `.pt` (Torch) saved in respective results folders

---

## 🧠 Authors & Contact

Developed by **Simon Scandella**


This READ ME is a Draft
