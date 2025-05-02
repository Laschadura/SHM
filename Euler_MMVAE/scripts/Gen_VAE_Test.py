#!/usr/bin/env python3
"""
synthesize_mmvae.py

Generate synthetic samples from a trained SpectralMMVAE, using the
(pre‑computed) data returned by data_loader.load_data().
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision  # type: ignore
import cv2

# ------------------------------------------------------------------ #
#  project‑local imports
# ------------------------------------------------------------------ #
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_distributions import reparameterize 
import data_loader
from data_loader import inverse_spectrogram, mask_recon, inspect_frequency_content
from vae_generator import (
    SpectralMMVAE,
)

# ------------------------------------------------------------------ #
#  0)  convenience – fetch ALL cached arrays in one call
# ------------------------------------------------------------------ #
def get_cached_data(*,
                    segment_duration: float = 4.0,
                    nperseg: int        = 256,
                    noverlap: int       = 224,
                    sample_rate: int    = 200,
                    recompute: bool     = False):
    """
    Wrapper around data_loader.load_data() that returns cached arrays
    and all 3 dictionaries.
    """
    (
        accel_dict,
        binary_masks,
        heatmaps,
        segments,
        spectrograms,
        test_ids,
    ) = data_loader.load_data(
        segment_duration = segment_duration,
        nperseg          = nperseg,
        noverlap         = noverlap,
        sample_rate      = sample_rate,
        recompute        = recompute
    )

    return {
        "accel_dict":   accel_dict,
        "binary_masks": binary_masks,
        "heatmaps":     heatmaps,
        "segments":     segments,
        "specs":        spectrograms,
        "test_ids":     test_ids,
    }

# ------------------------------------------------------------------ #
#  1)  model loader –  **now uses shapes from the cache**
# ------------------------------------------------------------------ #
def load_trained_mmvae(weights_path: str, spec_shape, mask_shape, latent_dim=128):
    """
    Build a SpectralMMVAE whose input shapes match the cached data
    and load pre‑trained weights.
    """
    model = SpectralMMVAE(latent_dim, spec_shape, mask_shape)

    # dummy forward pass to create variables
    _ = model(
        tf.zeros((1, *spec_shape), dtype=tf.float32),
        tf.zeros((1, *mask_shape), dtype=tf.float32),
        training=False,
    )

    model.load_weights(weights_path)
    print(f"✅  Loaded SpectralMMVAE weights from {weights_path}")
    return model

# ------------------------------------------------------------------ #
#  2)  ----  synthesis helpers  ------------------------------------ #
# ------------------------------------------------------------------ #
def generate_random_samples(model, how_many=5, save_dir="synthesized_samples"):
    """
    Generate random samples from the mixture-of-experts prior and visualize them.
    
    Args:
        model: Trained SpectralMMVAE model
        how_many: Number of samples to generate
        save_dir: Directory to save visualization outputs
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(how_many):
        print(f"Generating sample {i+1}/{how_many}...")
        
        # Generate samples from the model's prior
        recon_spec, recon_mask = model.generate(modality='both')
        
        # Save spectrograms as numpy arrays for potential further processing
        np.save(f"{save_dir}/sample_{i+1}_spec.npy", recon_spec.numpy())
        np.save(f"{save_dir}/sample_{i+1}_mask.npy", recon_mask.numpy())
        
        # Visualize spectrograms (first channel only for simplicity)
        plt.figure(figsize=(12, 6))
        
        # Plot magnitude spectrogram (even indices are magnitudes)
        plt.subplot(1, 3, 1)
        # Take first magnitude channel (index 0)
        magnitude = recon_spec[0, :, :, 0].numpy()
        plt.imshow(magnitude, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"Sample {i+1}: Magnitude Spectrogram (Ch 1)")
        plt.colorbar()
        
        # If you have multiple channels, optionally show another one
        if recon_spec.shape[-1] > 2:  # If we have more than 1 channel (magnitude+phase)
            plt.subplot(1, 3, 2)
            # Take second magnitude channel (index 2)
            magnitude2 = recon_spec[0, :, :, 2].numpy()
            plt.imshow(magnitude2, aspect='auto', origin='lower', cmap='viridis')
            plt.title("Magnitude Spectrogram (Ch 2)")
            plt.colorbar()
        
        # Plot mask
        plt.subplot(1, 3, 3)
        plt.imshow(recon_mask[0, :, :, 0].numpy(), aspect='auto', cmap='gray')
        plt.title("Generated Mask")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sample_{i+1}_visualization.png", dpi=300)
        plt.close()
        
        # Optional: Convert spectrogram to time series and save/visualize
        try:
            # Assumes you have the reconstruct_time_series function from your model
            time_series = model.reconstruct_time_series(
                recon_spec.numpy(), 
                fs=200,  # Sampling rate
                nperseg=256,  # STFT parameter from the log: nperseg=256
                noverlap=224, # STFT parameter from the log: noverlap=224
                time_length=800  # Time length from raw_segments.shape[1]
            )
            
            # Save time series
            np.save(f"{save_dir}/sample_{i+1}_time_series.npy", time_series)
            
            # Visualize time series
            plt.figure(figsize=(12, 4))
            time_axis = np.linspace(0, 4, 800)  # Assuming 4 seconds at 200Hz
            
            # Plot first few channels
            num_channels_to_plot = min(3, time_series.shape[2])
            colors = ['blue', 'red', 'green']
            
            for ch in range(num_channels_to_plot):
                plt.plot(time_axis, time_series[0, :, ch], 
                         label=f'Channel {ch+1}', 
                         color=colors[ch], 
                         alpha=0.8)
            
            plt.title(f"Sample {i+1}: Reconstructed Time Series")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/sample_{i+1}_time_series.png", dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error reconstructing time series: {e}")
    
    print(f"✅ Generated {how_many} samples in '{save_dir}' directory")

def generate_interpolations(model, num_interpolation_steps=10, save_dir="interpolation_samples"):
    """
    Generate samples by interpolating between two random points in latent space.
    
    Args:
        model: Trained SpectralMMVAE model
        num_interpolation_steps: Number of interpolation steps
        save_dir: Directory to save visualization outputs
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate two random points in latent space
    z1 = tf.random.normal(shape=(1, model.latent_dim))
    z2 = tf.random.normal(shape=(1, model.latent_dim))
    
    # Create interpolation steps
    alphas = np.linspace(0, 1, num_interpolation_steps)
    
    # Generate and visualize interpolated samples
    plt.figure(figsize=(15, 5))
    
    # Prepare multi-panel figure for the time series
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(211)  # For spectrograms
    ax2 = plt.subplot(212)  # For time series
    
    # Colors for time series plot
    colors = plt.cm.viridis(np.linspace(0, 1, num_interpolation_steps))
    
    # Store all time series for final animation
    all_time_series = []
    
    for i, alpha in enumerate(alphas):
        # Interpolate between the two latent vectors
        z_interp = (1 - alpha) * z1 + alpha * z2
        
        # Generate sample from the interpolated latent vector
        recon_spec = model.spec_decoder(z_interp, training=False)
        recon_mask = model.mask_decoder(z_interp, training=False)
        
        # Save individual samples
        np.save(f"{save_dir}/interp_{i+1}_spec.npy", recon_spec.numpy())
        np.save(f"{save_dir}/interp_{i+1}_mask.npy", recon_mask.numpy())
        
        # Add spectrogram to the first subplot
        ax1.add_patch(plt.Rectangle((i * 1.05, 0), 1, 1, fill=False))
        ax1.imshow(recon_spec[0, :, :, 0].numpy(), 
                  extent=[i * 1.05, (i + 1) * 1.05, 0, 1], 
                  aspect='auto', 
                  cmap='viridis')
        
        # Convert to time series if possible
        try:
            time_series = model.reconstruct_time_series(
                recon_spec.numpy(), 
                fs=200, 
                nperseg=256, 
                noverlap=224,
                time_length=800
            )
            all_time_series.append(time_series[0])
            
            # Add time series to plot - just the first channel for clarity
            ax2.plot(np.linspace(0, 4, 800), time_series[0, :, 0],
                    color=colors[i], alpha=0.7, 
                    label=f'α={alpha:.2f}')
                    
        except Exception as e:
            print(f"Error reconstructing time series at step {i}: {e}")
    
    # Finalize plot
    ax1.set_title("Latent Space Interpolation - Spectrograms")
    ax1.set_xlabel("Interpolation Step")
    ax1.set_yticks([])
    ax1.set_xticks([i * 1.05 + 0.5 for i in range(num_interpolation_steps)])
    ax1.set_xticklabels([f'α={a:.1f}' for a in alphas])
    
    ax2.set_title("Latent Space Interpolation - Time Series (Channel 1)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/latent_space_interpolation.png", dpi=300)
    plt.close()


# ------------------------------------------------------------------ #
#  3)  conditional generation & reconstruction – **cache driven**
# ------------------------------------------------------------------ #
def generate_conditional_samples(model, cached, save_dir="conditional_samples"):
    """
    One sample per unique test‑ID:

        • mask ← spec (latent from spec encoder ➜ mask decoder)
        • spec ← mask
        • direct reconstruction
        • time‑series recon + plots / RMS metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    specs      = cached["specs"]
    raw        = cached["segments"]
    test_ids   = cached["test_ids"]

    mask_dict = cached["heatmaps_per_segment"] if "heatmaps_per_segment" in cached else cached["heatmaps"]
    test_ids  = cached["test_ids"]
    masks_np  = np.stack([mask_dict[int(tid)] for tid in test_ids], axis=0)

    # ‑‑ ensure everything is float32 tensors --------------------------------
    specs_tf = tf.convert_to_tensor(specs)
    masks_tf  = tf.convert_to_tensor(masks_np, dtype=tf.float32)

    processed = set()
    for idx, tid in enumerate(test_ids):
        if tid in processed:
            continue
        processed.add(tid)

        spec_in = specs_tf[idx : idx + 1]
        mask_in = masks_tf[idx : idx + 1]

        mu_spec, _ = model.spec_encoder(spec_in,  training=False)
        mu_mask, _ = model.mask_encoder(mask_in,  training=False)

        recon_mask_from_spec = model.mask_decoder(mu_spec, training=False)
        recon_spec_from_mask = model.spec_decoder(mu_mask, training=False)
        recon_spec_direct    = model.spec_decoder(mu_spec, training=False)

        # ∘∘∘  plotting & metric code – unchanged from your previous script
        #     (omitted here for brevity)
        # -------------------------------------------------------------------

        print(f"✓  finished conditional generation for Test {tid}")

    print(f"✅  Generated samples for {len(processed)} unique test IDs")

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from data_loader import mask_recon, inspect_frequency_content

def test_reconstruction(model, cached, save_dir="reconstruction_tests"):
    """
    One segment per test-ID → four plot groups (mask, spec, overlay,
    per-channel, PSD) from cached arrays.
    """
    base_dir = Path(save_dir)
    ts_dir   = base_dir / "ts"
    psd_dir  = base_dir / "psd"
    spec_dir = base_dir / "spec"
    mask_dir = base_dir / "masks"

    for d in (ts_dir, psd_dir, spec_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    specs    = cached["specs"]
    raw_seg  = cached["segments"]
    test_ids = cached["test_ids"]
    mask_dict = cached.get("heatmaps_per_segment", cached["heatmaps"])
    masks_np  = np.stack([mask_dict[int(tid)] for tid in test_ids], axis=0)

    specs_tf = tf.convert_to_tensor(specs)
    masks_tf = tf.convert_to_tensor(masks_np, dtype=tf.float32)

    unique_ids = np.unique(test_ids)
    for tid in unique_ids:
        idx = np.where(test_ids == tid)[0][0]
        spec_in = specs_tf[idx : idx + 1]
        mask_in = masks_tf[idx : idx + 1]

        recon_spec, recon_mask, _ = model(spec_in, mask_in, training=False)
        orig_ts = raw_seg[idx]
        recon_ts = model.reconstruct_time_series(
            recon_spec.numpy(), fs=200, nperseg=256, noverlap=224, time_length=orig_ts.shape[0]
        )[0]

        t = np.linspace(0, 4, orig_ts.shape[0])

        # Overlay all channels
        plt.figure(figsize=(15, 4))
        for ch in range(orig_ts.shape[1]):
            scale = orig_ts[:,ch].std() / (recon_ts[:,ch].std() + 1e-8)
            plt.plot(t, orig_ts[:,ch], color="royalblue", lw=0.8, alpha=0.6)
            plt.plot(t, recon_ts[:,ch]*scale, color="crimson", lw=0.8, alpha=0.6)
        plt.title(f"Test {tid} • Original vs Recon (all channels)")
        plt.xlabel("Time [s]"); plt.ylabel("Amplitude"); plt.grid(alpha=0.25)
        plt.legend(["Original", "Reconstructed"], framealpha=0.7)
        plt.tight_layout()
        plt.savefig(ts_dir / f"test{tid}_ts_overlay.png", dpi=300)
        plt.close()

        # Per-channel subplots
        fig, axs = plt.subplots(6, 2, figsize=(14, 10))
        for ch in range(orig_ts.shape[1]):
            ax = axs[ch // 2, ch % 2]
            scale = orig_ts[:,ch].std() / (recon_ts[:,ch].std() + 1e-8)
            ax.plot(t, orig_ts[:,ch], color="royalblue", lw=0.7, alpha=0.7)
            ax.plot(t, recon_ts[:,ch]*scale, color="crimson", lw=0.7, alpha=0.7)
            ax.set_title(f"Channel {ch+1}")
            ax.grid(alpha=0.3)
        fig.legend(["Original", "Reconstructed"], loc="upper right", framealpha=0.7)
        plt.tight_layout()
        plt.savefig(ts_dir / f"test{tid}_ts_subplots.png", dpi=300)
        plt.close()

        # Spectrogram comparison (Ch1)
        orig_spec = specs[idx, :, :, 0]
        recon_spec_ch1 = recon_spec.numpy()[0, :, :, 0]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        im1 = ax1.imshow(orig_spec, origin="lower", aspect="auto", cmap="viridis")
        ax1.set_title(f"Original Spectrogram (Ch1) • Test {tid}")
        plt.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(recon_spec_ch1, origin="lower", aspect="auto", cmap="viridis")
        ax2.set_title("Reconstructed Spectrogram (Ch1)")
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.savefig(spec_dir / f"test{tid}_spectrogram_comparison.png", dpi=300)
        plt.close()

        # Mask comparison
        high_gt  = mask_recon(mask_dict[tid][None])[0]
        high_rec = mask_recon(recon_mask.numpy())[0]
        fig, (m1, m2) = plt.subplots(1, 2, figsize=(12, 4))
        m1.imshow(high_gt, cmap="gray", aspect="auto"); m1.set_title("Original Mask"); m1.axis("off")
        m2.imshow(high_rec, cmap="gray", aspect="auto"); m2.set_title("Reconstructed Mask"); m2.axis("off")
        plt.tight_layout()
        plt.savefig(mask_dir / f"test{tid}_mask_comparison.png", dpi=300)
        plt.close()

        # PSD plots
        fs = 200
        f, psd_orig = data_loader.inspect_frequency_content(orig_ts[None], fs=fs, avg_over_segments=False)
        _, psd_recon = data_loader.inspect_frequency_content(recon_ts[None], fs=fs, avg_over_segments=False)
        psd_orig, psd_recon = psd_orig[0], psd_recon[0]  # (freq, C)

        # Per-channel PSD
        C = psd_orig.shape[1]
        fig, axs = plt.subplots((C+1)//2, 2, figsize=(14, 4*((C+1)//2)))
        axs = axs.ravel()
        for ch in range(C):
            ax = axs[ch]
            ax.semilogy(f, psd_orig[:,ch], color="royalblue", lw=0.8, label="Orig" if ch==0 else None)
            ax.semilogy(f, psd_recon[:,ch], color="crimson", lw=0.8, label="Recon" if ch==0 else None)
            ax.fill_between(f, psd_orig[:,ch], psd_recon[:,ch],
                            where=psd_recon[:,ch] > psd_orig[:,ch],
                            color="crimson", alpha=0.2)
            ax.fill_between(f, psd_orig[:,ch], psd_recon[:,ch],
                            where=psd_recon[:,ch] < psd_orig[:,ch],
                            color="royalblue", alpha=0.2)
            ax.set_title(f"Ch {ch+1}")
            ax.grid(alpha=0.3)
        for ch in range(C, len(axs)): axs[ch].axis("off")
        fig.legend(loc="upper right", framealpha=0.8)
        fig.suptitle(f"Test {tid} • PSD comparison (per-channel)")
        plt.tight_layout()
        plt.savefig(psd_dir / f"test{tid}_psd_subplots.png", dpi=300)
        plt.close()

        # Mean PSD
        mean_orig = psd_orig.mean(axis=1)
        mean_recon = psd_recon.mean(axis=1)
        lo_o, hi_o = np.percentile(psd_orig, [2.5, 97.5], axis=1)
        lo_r, hi_r = np.percentile(psd_recon, [2.5, 97.5], axis=1)
        plt.figure(figsize=(8,4))
        plt.semilogy(f, mean_orig, label="Orig (mean)", color="royalblue")
        plt.semilogy(f, mean_recon, label="Recon (mean)", color="crimson")
        plt.fill_between(f, lo_o, hi_o, color="royalblue", alpha=0.15)
        plt.fill_between(f, lo_r, hi_r, color="crimson", alpha=0.15)
        plt.title(f"Test {tid} • PSD averaged over {C} channels")
        plt.xlabel("Frequency [Hz]"); plt.ylabel("PSD [V²/Hz]"); plt.grid(alpha=0.3)
        plt.legend(); plt.tight_layout()
        plt.savefig(psd_dir / f"test{tid}_psd_mean.png", dpi=300)
        plt.close()

        print(f"✓  reconstruction plots saved for Test {tid}")

    print("✅  Reconstruction test complete.")


# ------------------------------------------------------------------ #
#  4)  main
# ------------------------------------------------------------------ #
def main():
    weights_path = "results_mmvae/final_spectral_mmvae.weights.h5"
    out_dir = "synthesis_results_mmvae"
    os.makedirs(out_dir, exist_ok=True)

    # ---- load cached data --------------------------------------------------
    cached = get_cached_data(
    recompute=False,
    segment_duration=4.0,
    nperseg=256,
    noverlap=224
    )


    # derive shapes directly from cache
    spec_shape = tuple(cached["specs"].shape[1:])   # (F, T, 2C)
    mask_shape = tuple(cached["heatmaps"][list(cached['heatmaps'].keys())[0]].shape)  # (32,96,1)

    print(f"📐  spec_shape = {spec_shape} | mask_shape = {mask_shape}")

    # build + load model
    model = load_trained_mmvae(
        weights_path,
        spec_shape=spec_shape,
        mask_shape=mask_shape,
        latent_dim=128,
    )

    # --- choose any synthesis routine you need -----------------------------
    generate_conditional_samples(
        model,
        cached,
        save_dir=os.path.join(out_dir, "conditional_samples"),
    )

    test_reconstruction(
        model,
        cached,
        save_dir=os.path.join(out_dir, "reconstruction_tests"),
    )

    print(f"\n✅  All synthesis tasks finished – see '{out_dir}'")

if __name__ == "__main__":
    main()