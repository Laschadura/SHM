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
from pathlib import Path  

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
from data_loader import inspect_frequency_content

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Helper metrics and utils --------------------------------------------------
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

def dice_score(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Computes the Dice score between two binary or soft masks.

    Args:
        gt (np.ndarray): ground-truth mask
        pred (np.ndarray): predicted mask
        eps (float): epsilon for numerical stability

    Returns:
        float: Dice coefficient
    """
    gt_flat   = gt.flatten()
    pred_flat = pred.flatten()
    intersection = np.sum(gt_flat * pred_flat)
    return (2.0 * intersection + eps) / (np.sum(gt_flat) + np.sum(pred_flat) + eps)

def damage_amount(mask: np.ndarray) -> float:
    """
    Computes total 'damage amount' as sum of pixel values in a (soft) mask.

    Args:
        mask (np.ndarray): predicted or ground-truth mask

    Returns:
        float: Total damage estimate
    """
    return float(mask.sum()) / mask.size  # normalize by area

def latent_fid(real: np.ndarray, fake: np.ndarray) -> float:
    """Lightweight Fréchet distance in latent space (μ/Σ only)."""
    mu_r, mu_f = real.mean(0), fake.mean(0)
    cov_r, cov_f = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    eps = 1e-6 * np.eye(cov_r.shape[0])
    covmean = torch.from_numpy(cov_r + eps) @ torch.from_numpy(cov_f + eps)
    covmean = torch.linalg.matrix_power(covmean, 1//2).numpy()
    return float(np.sum((mu_r - mu_f) ** 2) + np.trace(cov_r + cov_f - 2 * covmean))

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
def evaluate_synthesis(
        mld: dm.MultiModalLatentDiffusion,
        loader: DataLoader,
        device: torch.device,
        *,
        pair_ids: list[int],          # e.g. [20,21,22,25]
        steps_per_pair: int = 3,      # Nr. of α-values *between* each pair
        out_dir: str = "results_diff",
        segments: np.ndarray | None = None,
        heats:    np.ndarray | None = None):

    import matplotlib.pyplot as plt
    from collections import defaultdict
    from pathlib import Path


    root_dir  = Path(out_dir)
    synth_dir = root_dir / "synthesis"
    synth_dir.mkdir(parents=True, exist_ok=True)

    ae_spec, ae_mask = mld.autoencoders["spec"].eval(), mld.autoencoders["mask"].eval()

    lat_spec, lat_mask, id_all = [], [], []
    with torch.no_grad():
        for spec, mask, ids in loader:
            z_s = ae_spec.encoder(spec.to(device))
            z_m = ae_mask.encoder(mask.to(device))
            lat_spec.append(z_s.cpu())
            lat_mask.append(z_m.cpu())
            id_all.extend(ids.cpu().numpy())

    lat_spec = torch.cat(lat_spec, 0)
    lat_mask = torch.cat(lat_mask, 0)
    joint_lat = torch.cat([lat_spec, lat_mask], 1).numpy()
    id_all    = np.asarray(id_all, dtype=int)

    id2idx = defaultdict(list)
    for i, tid in enumerate(id_all):
        id2idx[tid].append(i)

    rep_idx = [ max(idxs, key=lambda x: 0) for idxs in id2idx.values() ]
    reps_real = loader.dataset.tensors[0][rep_idx].float().to(device)
    reps_ts   = spectrograms_to_timeseries(ae_spec(reps_real)[0])
    centroid_ts = { int(id_all[i]): reps_ts[j][:,0] for j,i in enumerate(rep_idx) }

    centroid_lat = { tid: joint_lat[id_all == tid].mean(0) for tid in np.unique(id_all) }

    hops   = [(pair_ids[i], pair_ids[i+1]) for i in range(len(pair_ids)-1)]
    alphas = np.linspace(0,1, steps_per_pair+2, endpoint=True)[1:-1]

    syn_lat, lab_A, lab_B, lab_alpha = [], [], [], []
    for A,B in hops:
        zA = torch.tensor(centroid_lat[A], device=device).float()
        zB = torch.tensor(centroid_lat[B], device=device).float()
        for a in alphas:
            z_mid = a*zA + (1-a)*zB
            z_ref = z_mid[None].clone()
            for t in range(mld.noise_scheduler.num_timesteps-5,
                           mld.noise_scheduler.num_timesteps):
                tt = torch.full((1,), t, device=device, dtype=torch.long)
                z_ref = mld.noise_scheduler.p_sample(mld.diffusion_model, z_ref, tt)
            syn_lat.append(z_ref.squeeze(0))
            lab_A.append(A); lab_B.append(B); lab_alpha.append(float(a))

    syn_lat = torch.stack(syn_lat).float()
    syn = mld.decode_modalities(syn_lat)
    ts_syn  = spectrograms_to_timeseries(syn["spec"])
    mk_up   = masks_to_fullsize(syn["mask"])

    for tid, trace in centroid_ts.items():
        rms = trace.std()
        centroid_ts[tid] = trace / (rms + 1e-8)

    ts_syn_norm = ts_syn / (ts_syn.std(axis=1, keepdims=True) + 1e-8)

    max_vis = min(6, ts_syn.shape[0])
    for k in range(max_vis):
        A,B,a = lab_A[k], lab_B[k], lab_alpha[k]
        fig = plt.figure(figsize=(10,5)); gs = fig.add_gridspec(2,4,width_ratios=[2,1,1,1])

        ax = fig.add_subplot(gs[:,0])
        t  = np.linspace(0, 4, ts_syn.shape[1])
        ax.plot(t, centroid_ts[A], color="#1f77b4", lw=.8, alpha=.6, label=f"T{A}")
        ax.plot(t, centroid_ts[B], color="#ff7f0e", lw=.8, alpha=.6, label=f"T{B}")
        ax.plot(t, ts_syn_norm[k,:,0], color="#d62728", lw=.9, label="synthetic")
        ax.set(title=f"α = {a:.2f}   •   T{A} → T{B}", xlabel="t [s]", ylabel="norm. amp")
        ax.legend(framealpha=.8, fontsize=8)
        ax.grid(alpha=.3)

        fig.add_subplot(gs[0,1]).imshow(syn["mask"][k,0].cpu(), cmap="gray"); plt.axis("off"); plt.title("32×96")
        fig.add_subplot(gs[1,1]).imshow(mk_up[k], cmap="gray"); plt.axis("off"); plt.title("256×768")

        if heats is not None:
            fig.add_subplot(gs[0,2]).imshow(heats[A][...,0], cmap="gray"); plt.axis("off"); plt.title(f"T{A} GT", fontsize=8)
            fig.add_subplot(gs[1,2]).imshow(heats[B][...,0], cmap="gray"); plt.axis("off"); plt.title(f"T{B} GT", fontsize=8)

        # quantitative scores
        mask_pred = syn["mask"][k,0].cpu().numpy()
        gt_mask_A = heats[A][...,0] if heats is not None else None
        gt_mask_B = heats[B][...,0] if heats is not None else None

        if gt_mask_A is not None and gt_mask_B is not None:
            alpha = lab_alpha[k]
            gt_interp = (1-alpha)*gt_mask_A + alpha*gt_mask_B
            dice = dice_score(gt_interp, mask_pred)
            dmg_true = damage_amount(gt_interp)
            dmg_pred = damage_amount(mask_pred)
            err = abs(dmg_pred - dmg_true)
            fig.suptitle(f"Dice: {dice:.3f}  |  Δdamage: {err:.4f}", fontsize=10)

        fig.tight_layout()
        fig.savefig(synth_dir/f"quickcheck_{k}.png", dpi=150); plt.close(fig)



def evaluate_reconstruction(mld: dm.MultiModalLatentDiffusion,
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
    root_dir   = Path(out_dir)
    recon_dir  = root_dir / "reconstruction"
    recon_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # dedicated sub-folders
    # ------------------------------------------------------------------
    ts_dir   = recon_dir / "ts"
    psd_dir  = recon_dir / "psd"
    spec_dir = recon_dir / "spec"
    mask_dir = recon_dir / "masks"

    for d in (ts_dir, psd_dir, spec_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)



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

    if gen_samples == 0 or len(chosen_indices) == 0:
        print("✅ Skipping visual + sample reconstructions (gen_samples=0 or empty).")
        print(f"rec_spec_loss: {np.mean(rec_s_loss):.4f}")
        print(f"rec_mask_loss: {np.mean(rec_m_loss):.4f}")
        print(f"dice:          {np.mean(dices):.4f}")
        print(f"iou:           {np.mean(ious):.4f}")
        return


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
        z_s    = ae_spec.encoder(fake["spec"][i:i+1].to(device))
        z_m    = ae_mask.encoder(fake["mask"][i:i+1].to(device))
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
            plt.savefig(ts_dir / f"testid_{test_id}_ts_overlay.png", dpi=300)
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
            plt.savefig(ts_dir / f"testid_{test_id}_ts_subplots.png", dpi=300)
            plt.close()

        # 3) frequency content comparison (PSD original vs reconstruction) ---------
        if ts_orig is not None:
            fs = 200
            ts_o = ts_orig[None]          # (1,T,C)
            ts_r = ts_sample[None]

            f,  psd_orig  = inspect_frequency_content(ts_o, fs=fs, avg_over_segments=False)
            _,  psd_recon = inspect_frequency_content(ts_r, fs=fs, avg_over_segments=False)
            psd_orig, psd_recon = psd_orig[0], psd_recon[0]     # (freq, C)

            # -------- per-channel shaded plots -----------------------------------
            C     = psd_orig.shape[1]
            rows  = int(np.ceil(C / 2))
            fig, axs = plt.subplots(rows, 2, figsize=(14, 4*rows), squeeze=False)
            axs = axs.ravel()

            for ch in range(C):
                ax = axs[ch]
                ax.semilogy(f, psd_orig[:, ch], color="royalblue", lw=.8, label="orig"  if ch==0 else None)
                ax.semilogy(f, psd_recon[:, ch], color="crimson",   lw=.8, label="recon" if ch==0 else None)

                # shaded differences
                ax.fill_between(f, psd_orig[:, ch], psd_recon[:, ch],
                                where=psd_recon[:, ch] > psd_orig[:, ch],
                                color="crimson",  alpha=.25)
                ax.fill_between(f, psd_orig[:, ch], psd_recon[:, ch],
                                where=psd_recon[:, ch] < psd_orig[:, ch],
                                color="royalblue", alpha=.25)

                ax.set_title(f"Ch {ch+1}")
                ax.set_xlabel("Frequency [Hz]")
                ax.set_ylabel("PSD [V²/Hz]")
                ax.grid(alpha=.3)

            # hide unused axes if channel count is odd
            for k in range(C, len(axs)):
                axs[k].axis("off")

            fig.legend(loc="upper right", framealpha=.8)
            fig.suptitle(f"Test {test_id} • PSD comparison")
            plt.tight_layout()
            plt.savefig(psd_dir / f"testid_{test_id}_psd_subplots.png", dpi=300)
            plt.close()

            # -------- mean PSD with 95 % spread ----------------------------------
            psd_o_mean = psd_orig.mean(1)         # (freq,)
            psd_r_mean = psd_recon.mean(1)

            lo_o, hi_o = np.percentile(psd_orig,  [2.5, 97.5], axis=1)
            lo_r, hi_r = np.percentile(psd_recon, [2.5, 97.5], axis=1)

            plt.figure(figsize=(8,4))
            plt.semilogy(f, psd_o_mean, label="orig (mean)",  color="royalblue")
            plt.semilogy(f, psd_r_mean, label="recon (mean)", color="crimson")
            plt.fill_between(f, lo_o, hi_o, color="royalblue", alpha=.15)
            plt.fill_between(f, lo_r, hi_r, color="crimson",  alpha=.15)

            plt.title(f"Test {test_id} • PSD averaged over {C} channels")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("PSD [V²/Hz]")
            plt.grid(alpha=.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(psd_dir / f"testid_{test_id}_psd_mean.png", dpi=300)
            plt.close()



        # 4) decoder spectrograms (first 4 channels) -----------------------------
        plt.figure(figsize=(12,6))
        for j in range(min(4, spec_sample.shape[0]//2)):
            plt.subplot(2,2,j+1)
            plt.imshow(spec_sample[2*j], origin="lower", aspect="auto", cmap="viridis")
            plt.title(f"Ch {j+1} log-mag")
            plt.colorbar()
        plt.suptitle(f"Decoder output • Test ID {test_id}")
        plt.tight_layout()
        plt.savefig(spec_dir / f"testid_{test_id}_spectrograms.png", dpi=300)
        plt.close()

        # 4b) visualize raw decoder mask (no upsampling)
        raw_mask_lowres = rec_mask[plot_id, 0].cpu().numpy()  # (32,96)

        plt.figure(figsize=(4, 3))
        plt.imshow(raw_mask_lowres, cmap="gray")
        plt.title(f"Raw Decoder Output • Test ID {test_id}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(mask_dir / f"testid_{test_id}_mask_lowres_raw.png", dpi=200)
        plt.close()

        # 5) mask comparison + print raw values ---------------------------------------
        if heats is not None:
            # --- define masks first ---
            lowres_gt  = heats[test_id]               # (32,96,1)
            lowres_rec = mask_fake                    # (32,96)

            lowres_gt_vals  = lowres_gt[..., 0]       # squeeze channel
            lowres_rec_vals = lowres_rec

            # --- print raw values ---
            print(f"\n🧾 Raw values of low-res masks for Test ID {test_id}:")
            print("Original low-res (GT) mask slice [16:20, 40:50]:")
            print(np.array_str(lowres_gt_vals[16:20, 40:50], precision=3, suppress_small=True))

            print("Reconstructed low-res mask slice [16:20, 40:50]:")
            print(np.array_str(lowres_rec_vals[16:20, 40:50], precision=3, suppress_small=True))

            # --- upsample and visualize ---
            highres_gt  = data_loader.mask_recon(lowres_gt.transpose(2, 0, 1))[0]  # → (256,768)
            highres_rec = data_loader.mask_recon(lowres_rec[None])[0]             # → (256,768)

            fig, ax = plt.subplots(2, 2, figsize=(12, 8))
            ax[0, 0].imshow(lowres_gt_vals, cmap="gray");  ax[0, 0].set_title("Original (low-res)")
            ax[0, 1].imshow(lowres_rec,     cmap="gray");  ax[0, 1].set_title("Reconstructed (low-res)")
            ax[1, 0].imshow(highres_gt,     cmap="gray");  ax[1, 0].set_title("Original upsampled")
            ax[1, 1].imshow(highres_rec,    cmap="gray");  ax[1, 1].set_title("Reconstructed upsampled")
            for a in ax.ravel(): a.axis("off")
            plt.tight_layout()
            plt.savefig(mask_dir / f"testid_{test_id}_mask_comparison.png", dpi=300)
            plt.close()



# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="results_diff")
    p.add_argument("--batch",    type=int, default=32)
    p.add_argument("--samples",  type=int, default=8)
    p.add_argument("--no_diffusion", action="store_true",
               help="Only run encoder/decoder reconstruction; skip diffusion sampling.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fixed STFT settings
    segment_duration = 4.0
    nperseg, noverlap = 256, 224
    latent_dim = 256

    # ── load cached data ────────────────────────────────────────────────────
    cache_dir = "cache"

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
    
    # ------------------------------------------------------------------
    # Which test-IDs to interpolate and how many interior points
    # ------------------------------------------------------------------
    PAIR_IDS       = [20, 21, 22, 25]   # chain order matters
    STEPS_PER_PAIR = 3                  # α-values between each pair


    # ── run evaluation ──────────────────────────────────────────────────────
    # evaluate_synthesis(
    #     mld, loader, device,
    #     pair_ids=PAIR_IDS,
    #     steps_per_pair=STEPS_PER_PAIR,
    #     out_dir="Diff_eval",
    #     segments=segments,
    #     heats=heats)
    
    # ── run evaluation ──────────────────────────────────────────────────────
    if args.no_diffusion:
        print("✅ Running reconstruction without diffusion...")
        evaluate_reconstruction(mld,
                loader,
                device,
                gen_samples=0,              # skip generation
                out_dir="Diff_eval",
                segments=segments,       
                test_ids=ids,                
                heats=heats)
    else:
        evaluate_reconstruction(mld,
                loader,
                device,
                gen_samples=args.samples,
                out_dir="Diff_eval",
                segments=segments,       
                test_ids=ids,                
                heats=heats)
 




if __name__ == "__main__":
    main()

