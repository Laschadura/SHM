# test_diffusion.py
"""
Extended evaluation / synthesis script for the **Multi‑Modal Latent Diffusion**
model.  Added end‑to‑end *round‑trip* tests that

1. convert decoded **spectrograms ➜ time‑series** using
   `data_loader.inverse_spectrogram()`
2. convert decoded **32×96 masks ➜ 256×768** using
   `data_loader.mask_recon()`

Run from the project root:
    $ python test_diffusion.py  --ckpt_dir results_diff --batch 32 --samples 8
"""

from __future__ import annotations
import os
import argparse
import platform
from pathlib import Path

import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# working directory (cluster vs local) -------------------------------------
# ---------------------------------------------------------------------------
if platform.system() == "Windows":
    os.chdir("c:/SP-Master-Local/SP_DamageLocalization-MasonryArchBridge_SimonScandella/ProbabilisticApproach/Euler_MMVAE")
else:
    os.chdir("/cluster/scratch/scansimo/Euler_MMVAE")
print("✅ Script has started executing")

# ---------------------------------------------------------------------------
# local modules -------------------------------------------------------------
# ---------------------------------------------------------------------------
import diffusion_model as dm        # your training script
import data_loader                  # STFT ↔︎ time‑series, mask resize

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Helper metrics ------------------------------------------------------------
# ---------------------------------------------------------------------------

def dice_iou_scores(target: torch.Tensor, pred: torch.Tensor, thr: float = 0.5):
    """Dice & IoU for binary masks [N,1,H,W] – expects torch tensors."""
    pred_bin   = (pred > thr).float()
    target_bin = (target > thr).float()
    inter = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3))
    dice  = (2.0 * inter + 1e-8) / (union + 1e-8)
    iou   = (inter + 1e-8) / (union - inter + 1e-8)
    return dice.cpu().numpy(), iou.cpu().numpy()


def latent_fid(real: np.ndarray, fake: np.ndarray) -> float:
    """Lightweight Fréchet distance in latent space (μ/Σ only)."""
    mu_r, mu_f = real.mean(0), fake.mean(0)
    cov_r, cov_f = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    eps = 1e-6 * np.eye(cov_r.shape[0])
    covmean = torch.from_numpy(cov_r + eps) @ torch.from_numpy(cov_f + eps)
    covmean = torch.linalg.matrix_power(covmean, 1//2).numpy()
    return float(np.sum((mu_r - mu_f) ** 2) + np.trace(cov_r + cov_f - 2 * covmean))

# ---------------------------------------------------------------------------
# NEW  –   round‑trip utilities --------------------------------------------
# ---------------------------------------------------------------------------

# ------------------------------------------------------------------
# robust wrapper: converts (N, 2C, F, T)  *or*  (N, 2C, T, F)
#                 to      (N, F,  T, 2C) before inverse-STFT
# ------------------------------------------------------------------
def spectrograms_to_timeseries(spec_batch: torch.Tensor,
                               *, fs=200, nperseg=256, noverlap=224):
    """
    Accepts a tensor from the spectrogram decoder in either layout:
        (N, 2C, F, T)   or   (N, 2C, T, F)
    Returns reconstructed windows: (N, time_len, C)
    """
    spec_np = spec_batch.detach().cpu().numpy()          # (N, 2C, ?, ?)
    print("🔎  decoder out:", spec_np.shape)       # NEW ①

    # identify which spatial axis is the frequency axis
    freq_bins_expected = nperseg // 2 + 1                # 129 for n_fft=256
    _, _, d2, d3 = spec_np.shape

    if d2 == freq_bins_expected:
        # layout is (N, 2C, F, T)
        spec_np = spec_np.transpose(0, 2, 3, 1)          # → (N, F, T, 2C)
    elif d3 == freq_bins_expected:
        # layout is (N, 2C, T, F)  ⇨ swap last two axes
        spec_np = spec_np.transpose(0, 3, 2, 1)          # → (N, F, T, 2C)
    else:
        raise ValueError(
            f"Neither spatial axis equals expected freq-bins "
            f"{freq_bins_expected} (got {d2} × {d3}).")
    
    print("🔎  to inverse :", spec_np.shape)       # NEW ②

    time_len = int(fs * 4.0)        # 4-s windows everywhere in pipeline
    return data_loader.inverse_spectrogram(
        spec_np, time_length=time_len,
        fs=fs, nperseg=nperseg, noverlap=noverlap)




def masks_to_fullsize(mask_batch: torch.Tensor, *, target=(256, 768)):
    """Convert 32×96 float masks ➜ full 256×768."""
    mk_np = mask_batch.detach().cpu().numpy().squeeze(1)   # (N,H,W)
    recon = data_loader.mask_recon(mk_np, target_size=target)
    return recon                                           # (N,256,768)

# ---------------------------------------------------------------------------
# Evaluation ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def evaluate(mld: dm.MultiModalLatentDiffusion,
             loader: DataLoader,
             device: torch.device,
             *,
             gen_samples: int = 23,
             out_dir: str = "Diff_eval",
             segments: np.ndarray | None = None,
             test_ids: np.ndarray | None = None,
             heats: np.ndarray | None = None):
    """
    End-to-end evaluation.

    • reconstruction metrics (spectrogram + mask)
    • latent-FID on unconditional samples
    • visual comparison (1 repr. segment per unique test-id)
    """
    os.makedirs(out_dir, exist_ok=True)
    ae_spec, ae_mask = (mld.autoencoders["spec"].eval(),
                        mld.autoencoders["mask"].eval())
    

    # ─────────────────── pass through all batches once ────────────────────
    rec_s_loss, rec_m_loss, dices, ious = [], [], [], []
    lat_real, idx_real, id_real = [], [], []       # keep indices for rep. pick

    with torch.no_grad():
        start_idx = 0
        for spec, mask, ids in loader:             # ids = (batch,)
            B = spec.size(0)
            spec, mask = spec.to(device), mask.to(device)

            # ---- AE reconstruction ---------------------------------------
            rec_spec, z_s = ae_spec(spec)
            rec_mask, z_m = ae_mask(mask)

            rec_s_loss.append(dm.complex_spectrogram_loss(spec, rec_spec).item())
            rec_m_loss.append(dm.custom_mask_loss(mask, rec_mask).item())
            d, i = dice_iou_scores(mask, rec_mask)
            dices.extend(d);  ious.extend(i)

            # ---- store latents & bookkeeping -----------------------------
            lat_real.append(torch.cat([z_s, z_m], 1).cpu().numpy())
            idx_real.extend(range(start_idx, start_idx + B))
            id_real .extend(ids.cpu().numpy())
            start_idx += B

    lat_real = np.concatenate(lat_real, 0)         # (N, latent_dim)

    # ─────────────────── choose 1 segment per test-id ─────────────────────
    id2idx = defaultdict(list)
    for i, tid in enumerate(id_real):
        id2idx[int(tid)].append(i)

    chosen_indices = []
    seg_idx_to_tid = {}

    for tid, lst in id2idx.items():
        Z = lat_real[lst]                          # (n_k, d)
        mu = Z.mean(0, keepdims=True)              # (1, d)
        dist = ((Z - mu)**2).sum(1)                # L2^2
        best_idx = lst[int(dist.argmin())]
        chosen_indices.append(best_idx)
        seg_idx_to_tid[best_idx] = tid     
    
    # sort for reproducible order, crop to gen_samples
    chosen_indices = sorted(chosen_indices)[:gen_samples]

    # ─────────────────── visual recon (real → recon) ──────────────────────
    spec_real = loader.dataset.tensors[0][chosen_indices].float().to(device)

    with torch.no_grad():
        spec_recon, _ = ae_spec(spec_real)

    ts_recon = spectrograms_to_timeseries(spec_recon)     # (M,T,C), M<=gen_samples

    # ─────────────────── unconditional samples (fake) ─────────────────────
    with torch.no_grad():
        fake = mld.sample(batch_size=gen_samples)

    # latent-FID (same as before)
    z_fake = []
    for i in range(gen_samples):
        z_s = ae_spec.encoder(fake["spec"][i:i+1].to(device))
        z_m = ae_mask.encoder(fake["mask"][i:i+1].to(device))
        z_fake.append(torch.cat([z_s, z_m], 1).detach().cpu().numpy())
    z_fake = np.concatenate(z_fake, 0)

    # ─────────────────── summary metrics ──────────────────────────────────
    metrics = {
        "rec_spec_loss": np.mean(rec_s_loss),
        "rec_mask_loss": np.mean(rec_m_loss),
        "dice":          np.mean(dices),
        "iou":           np.mean(ious),
        "latent_FID":    latent_fid(lat_real, z_fake)
    }
    print("\n".join([f"{k:>15}: {v:.4f}" for k, v in metrics.items()]))

    # ─────────────────── plotting per chosen segment ──────────────────────
    for plot_id, seg_idx in enumerate(chosen_indices):
        spec_sample = spec_recon[plot_id].cpu().numpy()      # (2C,F,T)
        ts_sample   = ts_recon  [plot_id]                    # (T,C)
        ts_orig     = segments  [seg_idx] if segments is not None else None
        test_id = seg_idx_to_tid[seg_idx]
        mask_fake   = fake["mask"][plot_id,0].cpu().numpy()

        # 1) overlay all-channels ------------------------------------------------
        if ts_orig is not None:
            t = np.linspace(0, 4, ts_orig.shape[0])
            plt.figure(figsize=(15,4))
            for ch in range(ts_orig.shape[1]):
                scale = ts_orig[:,ch].std() / (ts_sample[:,ch].std() + 1e-8)
                plt.plot(t, ts_orig[:,ch],            color="royalblue", lw=.8, alpha=.6)
                plt.plot(t, ts_sample[:,ch]*scale,    color="crimson",   lw=.8, alpha=.6)
            plt.title(f"Test ID {test_id} • Original vs Recon (all ch.)")
            plt.xlabel("Time [s]"); plt.ylabel("Amplitude"); plt.grid(alpha=.25)
            plt.legend(["original", "recon"], framealpha=.7)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/testid_{test_id}_ts_overlay.png", dpi=300)
            plt.close()

        # 2) per-channel sub-plots ----------------------------------------------
        if ts_orig is not None:
            fig, axs = plt.subplots(6,2, figsize=(14,10))
            t = np.linspace(0,4, ts_sample.shape[0])
            for ch in range(ts_sample.shape[1]):
                ax = axs[ch//2, ch%2]
                scale = ts_orig[:,ch].std() / (ts_sample[:,ch].std() + 1e-8)
                ax.plot(t, ts_orig[:,ch],            color="royalblue", lw=.7, alpha=.7)
                ax.plot(t, ts_sample[:,ch]*scale,    color="crimson",   lw=.7, alpha=.7)
                ax.set_title(f"Ch {ch+1}")
                ax.grid(alpha=.3)
            fig.legend(["original","recon"], loc="upper right", framealpha=.7)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/testid_{test_id}_ts_subplots.png", dpi=300)
            plt.close()

        # 3) decoder spectrograms (first 4 channels) -----------------------------
        plt.figure(figsize=(12,6))
        for j in range(min(4, spec_sample.shape[0]//2)):
            plt.subplot(2,2,j+1)
            plt.imshow(spec_sample[2*j], origin="lower", aspect="auto", cmap="viridis")
            plt.title(f"Ch {j+1} log-mag")
            plt.colorbar()
        plt.suptitle(f"Decoder output • Test ID {test_id}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/testid_{test_id}_spectrograms.png", dpi=300)
        plt.close()

        # 4) mask comparison -----------------------------------------------------
        if heats is not None:
            # use heatmaps[test_id] as the true low-res mask
            lowres_gt  = heats[test_id]               # (32,96,1)
            lowres_rec = mask_fake                    # (32,96)
            highres_gt  = data_loader.mask_recon(lowres_gt.transpose(2, 0, 1))[0]  # → (256,768)
            highres_rec = data_loader.mask_recon(lowres_rec[None])[0]             # → (256,768)

            fig, ax = plt.subplots(2, 2, figsize=(12, 8))
            ax[0, 0].imshow(lowres_gt[..., 0], cmap="gray");  ax[0, 0].set_title("Original (low-res)")
            ax[0, 1].imshow(lowres_rec,     cmap="gray");     ax[0, 1].set_title("Reconstructed (low-res)")
            ax[1, 0].imshow(highres_gt,     cmap="gray");     ax[1, 0].set_title("Original upsampled")
            ax[1, 1].imshow(highres_rec,    cmap="gray");     ax[1, 1].set_title("Reconstructed upsampled")
            for a in ax.ravel(): a.axis("off")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/testid_{test_id}_mask_comparison.png", dpi=300)
            plt.close()





# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="results_diff")
    p.add_argument("--batch",    type=int, default=32)
    p.add_argument("--samples",  type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fixed STFT settings
    segment_duration = 4.0
    nperseg, noverlap = 256, 224
    latent_dim = 256

    # ── load cached data ────────────────────────────────────────────────────
    tag       = f"{segment_duration:.2f}s_{nperseg}_{noverlap}"
    cache_dir = "cache"
    spec_path = os.path.join(cache_dir, f"specs_{tag}.npy")
    seg_path  = os.path.join(cache_dir, f"segments_{tag}.npy")   # ← NEW
    heat_path = os.path.join(cache_dir, f"masks_{tag}.npy")
    ids_path  = os.path.join(cache_dir, f"segIDs_{tag}.npy")

    print("📂 Loading pre-computed cache from:", cache_dir)
    accel_dict, binary_masks, heats, segments, specs, ids = data_loader.load_data(
        segment_duration=segment_duration,
        nperseg=nperseg,
        noverlap=noverlap,
        sample_rate=200,
        recompute=False,
        cache_dir=cache_dir
    )

    masks = np.stack([heats[i] for i in ids], 0)

    # transpose for PyTorch
    specs = specs.transpose(0, 3, 1, 2)          # (N,C,H,W)
    masks = masks.transpose(0, 3, 1, 2)          # (N,1,H,W)

    # ── rebuild model skeleton & load weights ───────────────────────────────
    channels = specs.shape[1] // 2
    F, T     = specs.shape[2], specs.shape[3]
    Hm, Wm   = masks.shape[2:]

    specAE = dm.SpectrogramAutoencoder(latent_dim, channels, F, T).to(device)
    maskAE = dm.MaskAutoencoder(latent_dim, (Hm, Wm)).to(device)
    mld    = dm.MultiModalLatentDiffusion(specAE, maskAE, latent_dim, ["spec", "mask"], device)
    mld.load_autoencoders(Path(args.ckpt_dir) / "autoencoders")
    mld.load_diffusion_model(Path(args.ckpt_dir) / "diffusion" / "final_diffusion_model.pt")

    # ── torch dataset / loader ──────────────────────────────────────────────
    loader = DataLoader(
        TensorDataset(torch.from_numpy(specs).float(),
                      torch.from_numpy(masks).float(),
                      torch.from_numpy(ids)),
        batch_size=args.batch, shuffle=False, num_workers=4)

    # ── run evaluation ──────────────────────────────────────────────────────
    evaluate(mld,
            loader,
            device,
            gen_samples=23,
            out_dir="Diff_eval",
            segments=segments,       
            test_ids=ids,                
            heats=heats)                 



if __name__ == "__main__":
    main()
