import os
import numpy as np
import torch

from diff_pt.model import SpectrogramAutoencoder, MaskAutoencoder, MultiModalLatentDiffusion, DifferentiableISTFT
from diff_pt.train import train_autoencoders
from diff_pt.vis import save_plotly_loss_curve, save_visualizations_and_metrics, visualize_training_history
from diff_pt.io import (
    create_torch_dataset,
    save_autoencoders,
    load_autoencoders,
    save_diffusion_model,
    load_diffusion_model,
)

from bridge_data.loader import load_data

def main():
    # ─── Mode Control ───────────────────────────────────────────────
    train_AE        = True                 # whether to train AE
    ae_epochs      = 600                   # epochs for autoencoder training
    patience_ae     = 250                     # patience for autoencoder training
    recompute_data  = False                   # whether to recompute cache
    dm_mode         = "scratch"              # "scratch" or "continue"
    dm_epochs       = 1                  # extra epochs for diffusion
    learning_rate_dm = 5e-4
    learning_rate_ae = 2e-4
    diff_ckpt       = "results_diff/diffusion/final_diffusion_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results_diff/autoencoders", exist_ok=True)
    os.makedirs("results_diff/diffusion", exist_ok=True)

    # ─── Data Parameters ────────────────────────────────────────────
    fs = 200
    segment_duration = 4.0
    nperseg = 256
    noverlap = 224

    latent_dim = 384
    batch_size = 200

    tag = f"{segment_duration:.2f}s_{nperseg}_{noverlap}"
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    # ─── Load or compute features ───────────────────────────────────
    (accel_dict, binary_masks, heatmaps,
    segments,  spectrograms, test_ids,
    segment_metadata, seg_stats) = load_data(
            segment_duration = segment_duration,
            nperseg          = nperseg,
            noverlap         = noverlap,
            sample_rate      = fs,
            recompute        = recompute_data,
            cache_dir        = cache_dir)

    # post-process into channel-first arrays
    spectrograms  = spectrograms.transpose(0, 3, 1, 2)            # (N, 2C, F, T)
    mask_segments = np.stack([heatmaps[tid] for tid in test_ids], axis=0)
    mask_segments = mask_segments.transpose(0, 3, 1, 2)          # (N, 1, 32, 96)

    # ───   Train / val split  ───────────────────────────────────────
    N       = spectrograms.shape[0]
    perm    = np.random.permutation(N)
    split   = int(0.8 * N)
    tr_idx, va_idx = perm[:split], perm[split:]

    tr_spec,    va_spec   = spectrograms[tr_idx],       spectrograms[va_idx]
    tr_mask,    va_mask   = mask_segments[tr_idx],     mask_segments[va_idx]
    tr_seg,     va_seg    = segments[tr_idx],          segments[va_idx]
    tr_ids,     va_ids    = test_ids[tr_idx],          test_ids[va_idx]
    tr_stats,va_stats  = np.array(seg_stats, dtype=object)[tr_idx], np.array(seg_stats, dtype=object)[va_idx]

    # ───   GLOBAL μ,σ  (no leakage/train_ds only used) ─────────────────────────────────
    C = tr_spec.shape[1] // 2  # 2C channels total → split into C mag + C phase

    # Separate
    logmag_train = tr_spec[:, :C]     # (N, C, F, T)
    phase_train  = tr_spec[:, C:]     # (N, C, F, T)

    # Channel-wise stats for log(|S|)
    mu_logmag  = logmag_train.mean(axis=(0, 2, 3), keepdims=True)       # shape: (1, C, 1, 1)
    sigma_logmag = logmag_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8

    tr_spec = np.concatenate([logmag_train, phase_train], axis=1)  # keep raw
    
    mu_for_loader  = mu_logmag[0]      # shape (C, 1, 1)
    sigma_for_loader = sigma_logmag[0]  # shape (C, 1, 1)

    np.savez(os.path.join(cache_dir, "spec_norm_magonly.npz"),
         mu=mu_for_loader.astype(np.float32),
         sigma=sigma_for_loader.astype(np.float32))

    # ─── 4.  DataLoaders  ────────────────────────────────────────────

    train_loader = create_torch_dataset(tr_spec, tr_mask, tr_seg, tr_ids,
                                        tr_stats, mu_for_loader, sigma_for_loader,
                                        batch_size=batch_size, shuffle=True)

    val_loader   = create_torch_dataset(va_spec, va_mask, va_seg, va_ids,
                                        va_stats, mu_for_loader, sigma_for_loader,
                                        batch_size=batch_size, shuffle=False)


    freq_bins, time_bins = tr_spec.shape[2:]
    channels = tr_spec.shape[1] // 2
    mask_height, mask_width = mask_segments.shape[-2:]

    # ─── Build Model ────────────────────────────────────────────────
    spec_autoencoder = SpectrogramAutoencoder(latent_dim, channels, freq_bins, time_bins).to(device)
    mask_autoencoder = MaskAutoencoder(latent_dim, (mask_height, mask_width)).to(device)

    spec_autoencoder.istft = DifferentiableISTFT(nperseg=nperseg, noverlap=noverlap).to(device)

    # torch.autograd.set_detect_anomaly(True) 

    mu_tensor  = torch.tensor(mu_for_loader, dtype=torch.float32).to(device)
    std_tensor = torch.tensor(sigma_for_loader, dtype=torch.float32).to(device)


    mld = MultiModalLatentDiffusion(
        spec_autoencoder,
        mask_autoencoder,
        latent_dim,
        ["spec", "mask"],
        device,
        mu_spec=mu_tensor,
        sig_spec=std_tensor
    )

    mld.istft = DifferentiableISTFT(nperseg=nperseg, noverlap=noverlap).to(device)

    if train_AE:
        print("🚀 Training Autoencoders...")
        spec_history, mask_history = train_autoencoders(
            spec_autoencoder, mask_autoencoder,
            train_loader, val_loader,
            device, epochs=ae_epochs,
            lr=learning_rate_ae, patience=patience_ae,
            cache_dir=cache_dir,
        )
        save_plotly_loss_curve(spec_history, "results_diff/autoencoders/spec_loss.html", title="Spectrogram AE Loss")
        save_plotly_loss_curve(mask_history, "results_diff/autoencoders/mask_loss.html", title="Mask AE Loss")

        load_autoencoders(mld.autoencoders, device)

    else:
        print("🔄 Loading pretrained autoencoders...")
        load_autoencoders(mld.autoencoders, device)



    # ─── Diffusion training control ─────────────────────────────────
    if dm_mode == "continue":
        if os.path.exists(diff_ckpt):
            print("🔄 Continuing diffusion training from checkpoint.")
            load_diffusion_model(mld.diffusion_model, device, diff_ckpt)
        else:
            print("❌ No diffusion checkpoint found – aborting continue mode.")
            return
    elif dm_mode == "scratch":
        print("🆕 Training diffusion model from scratch.")
    else:
        raise ValueError(f"Invalid dm_mode: {dm_mode}. Choose 'scratch' or 'continue'.")

    print(f"🚀 Training diffusion model for {dm_epochs} epoch(s).")
    history = mld.train_diffusion_model(
        train_dataloader = train_loader,
        val_dataloader   = val_loader,
        num_epochs       = dm_epochs,
        learning_rate    = learning_rate_dm,
        save_dir         = "results_diff/diffusion"
    )

    save_diffusion_model(mld.diffusion_model, diff_ckpt)

    visualize_training_history(history, save_path="results_diff/diffusion/training_curve.png")
    save_visualizations_and_metrics(mld, train_loader, val_loader, training_metrics=history, output_dir="results_diff")

    print("✅ All training, evaluation, and synthesis complete.")

if __name__ == "__main__":
  main()