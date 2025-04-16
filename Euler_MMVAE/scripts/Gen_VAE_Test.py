#!/usr/bin/env python3
"""
synthesize_mmvae.py

A focused script to generate synthetic samples from a trained SpectralMMVAE model.
This script provides more granular control over the synthesis process and additional
visualization options compared to the basic run_trained_mmvae.py functionality.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision #type: ignore
import sys
from scipy import signal
import cv2

# -------------------------------------------------------------------
# 1) Import your classes (assuming they're in your project structure)
# -------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Parent directory
from custom_distributions import (
    compute_js_divergence,
    reparameterize,
    compute_mixture_prior,
    compute_kl_divergence,
)
import data_loader
from data_loader import mask_recon

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vae_generator import (
    SpectrogramEncoder,
    MaskEncoder,
    SpectrogramDecoder,
    MaskDecoder,
    SpectralMMVAE,
    compute_complex_spectrogram,
    inverse_spectrogram
)

# -------------------------------------------------------------------
# 2) Define synthesis functions
# -------------------------------------------------------------------
def load_trained_mmvae(weights_path, latent_dim=128):
    """
    Build a SpectralMMVAE with the same shape config used in training,
    then load saved weights.
    """
    # Must match EXACT spec_shape & mask_shape from your training:
    spec_shape = (129, 18, 24)   # Matching the log: freq_bins=129, time_bins=18, channels=24
    mask_shape = (32, 96, 1)     # Matching the mask shape from the log

    # Build model
    model = SpectralMMVAE(latent_dim, spec_shape, mask_shape)
    
    # Dummy forward pass to build weights
    _ = model(
        tf.zeros((1, *spec_shape), dtype=tf.float32),
        tf.zeros((1, *mask_shape), dtype=tf.float32),
        training=False
    )
    
    # Load weights
    model.load_weights(weights_path)
    print(f"✅ Loaded trained SpectralMMVAE weights from: {weights_path}")
    return model

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
    
    # Optional: Create a video of the interpolation if you have many steps
    if len(all_time_series) >= 10 and cv2.__version__:  # Only if OpenCV is properly installed
        try:
            create_interpolation_video(all_time_series, f"{save_dir}/interpolation_video.mp4")
        except Exception as e:
            print(f"Could not create interpolation video: {e}")
    
    print(f"✅ Generated {num_interpolation_steps} interpolated samples in '{save_dir}' directory")

def create_interpolation_video(time_series_list, output_path, fps=10, duration=10):
    """
    Create a video showcasing the interpolation through the time series.
    
    Args:
        time_series_list: List of time series arrays
        output_path: Path to save the video
        fps: Frames per second
        duration: Duration in seconds
    """
    # Determine video parameters
    frame_count = len(time_series_list)
    width, height = 800, 400  # Video dimensions
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Time axis for plotting
    time_axis = np.linspace(0, 4, time_series_list[0].shape[0])
    
    # Find global min and max for consistent y-axis
    all_data = np.concatenate(time_series_list)
    global_min = np.min(all_data)
    global_max = np.max(all_data)
    y_range = global_max - global_min
    y_min = global_min - 0.1 * y_range
    y_max = global_max + 0.1 * y_range
    
    for ts in time_series_list:
        # Create a matplotlib figure for each frame
        fig = plt.figure(figsize=(8, 4), dpi=100)
        
        # Plot all channels
        num_channels = min(3, ts.shape[1])  # Plot at most 3 channels
        colors = ['blue', 'red', 'green']
        
        for ch in range(num_channels):
            plt.plot(time_axis, ts[:, ch], color=colors[ch], 
                     label=f'Channel {ch+1}', alpha=0.8)
        
        plt.title("Interpolated Time Series")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.ylim(y_min, y_max)  # Consistent y-axis scaling
        plt.legend()
        plt.tight_layout()
        
        # Convert matplotlib figure to opencv image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.resize(img, (width, height))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Add frame to video, multiple times for slower transitions
        repeats = int(fps * duration / frame_count)
        for _ in range(repeats):
            video.write(img)
        
        plt.close(fig)
    
    # Release video writer
    video.release()
    print(f"✅ Created interpolation video: {output_path}")

def generate_conditional_samples(model, conditioning_data=None, save_dir="conditional_samples"):
    """
    Generate samples conditioned on real data - one sample for each unique test ID.
    For each test ID:
    1. Generate masks from spectrograms
    2. Generate spectrograms from masks
    3. Reconstruct time series from generated spectrograms
    
    Args:
        model: Trained SpectralMMVAE model
        conditioning_data: Optional data for conditioning
        save_dir: Directory to save outputs
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # If no conditioning data provided, load from dataset
    if conditioning_data is None:
        try:
            # Load ALL available data
            accel_dict, binary_masks, heatmaps = data_loader.load_data()
            
            # Get ALL test IDs
            test_ids = list(accel_dict.keys())
            
            # Process these samples
            from vae_generator import segment_and_transform
            raw_segments, mask_segments, test_ids = segment_and_transform(
                {k: accel_dict[k] for k in test_ids},
                {k: heatmaps[k] for k in test_ids},
                segment_duration=4.0
            )
            
            # Convert to spectrograms
            spec_features = compute_complex_spectrogram(
                raw_segments, fs=200, nperseg=256, noverlap=224
            )
            
            # Store the binary masks for visualization
            binary_mask_segments = []
            for t_id in test_ids:
                binary_mask_segments.append(binary_masks[t_id])
            
            conditioning_data = {
                'specs': spec_features,
                'masks': mask_segments,
                'raw': raw_segments,
                'binary_masks': binary_mask_segments,
                'test_ids': test_ids
            }
            print(f"✅ Loaded {len(raw_segments)} real samples for conditioning")
            
        except Exception as e:
            print(f"❌ Failed to load conditioning data: {e}")
            print("Proceeding with random generation instead")
            generate_random_samples(model, how_many=3, save_dir=save_dir)
            return
    
    # Track processed test IDs to ensure one sample per unique ID
    processed_test_ids = set()
    
    for i in range(len(conditioning_data['specs'])):
        # Get current test ID
        current_test_id = conditioning_data['test_ids'][i]
        
        # Skip if already processed
        if current_test_id in processed_test_ids:
            continue
            
        # Mark this test ID as processed
        processed_test_ids.add(current_test_id)
        
        print(f"Generating sample for test ID {current_test_id}...")
        
        # Extract conditioning inputs
        spec_in = tf.convert_to_tensor(conditioning_data['specs'][i:i+1])
        mask_in = tf.convert_to_tensor(conditioning_data['masks'][i:i+1])
        
        # Get the latent representation from each modality
        mu_spec, _ = model.spec_encoder(spec_in, training=False)
        mu_mask, _ = model.mask_encoder(mask_in, training=False)
        
        # Generate cross-modal reconstructions
        # 1. From spectrogram to mask
        recon_mask_from_spec = model.mask_decoder(mu_spec, training=False)
        
        # 2. From mask to spectrogram
        recon_spec_from_mask = model.spec_decoder(mu_mask, training=False)
        
        # 3. Direct reconstruction of spectrogram
        recon_spec_direct = model.spec_decoder(mu_spec, training=False)
        
        # Plot masks comparison
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(mask_in[0, :, :, 0].numpy(), aspect='auto', cmap='gray')
        plt.title(f"Original Mask (Test {current_test_id})")
        plt.subplot(1, 2, 2)
        plt.imshow(recon_mask_from_spec[0, :, :, 0].numpy(), aspect='auto', cmap='gray')
        plt.title("Generated Mask (from Spec)")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/conditional_sample_{current_test_id}_masks_comparison.png", dpi=300)
        plt.close()
        
        # Reconstruct time series from both the original and generated spectrograms
        try:
            # Get original time series
            original_ts = conditioning_data['raw'][i]
            time_axis = np.linspace(0, 4, original_ts.shape[0])
            
            # Reconstructed time series from spectrogram-to-mask-to-spectrogram
            recon_ts_from_mask = model.reconstruct_time_series(
                recon_spec_from_mask.numpy(),
                fs=200, nperseg=256, noverlap=224,
                time_length=original_ts.shape[0]
            )
            
            # Reconstructed time series directly from spectrogram
            recon_ts_direct = model.reconstruct_time_series(
                recon_spec_direct.numpy(),
                fs=200, nperseg=256, noverlap=224,
                time_length=original_ts.shape[0]
            )
            
            # Calculate RMS errors
            num_channels = original_ts.shape[1]
            rms_errors_from_mask = np.zeros(num_channels)
            rms_errors_direct = np.zeros(num_channels)
            
            for ch in range(num_channels):
                original = original_ts[:, ch]
                recon_from_mask = recon_ts_from_mask[0, :, ch]
                recon_direct = recon_ts_direct[0, :, ch]
                
                rms_errors_from_mask[ch] = np.sqrt(np.mean((original - recon_from_mask)**2))
                rms_errors_direct[ch] = np.sqrt(np.mean((original - recon_direct)**2))
            
            mean_rms_error_from_mask = np.mean(rms_errors_from_mask)
            mean_rms_error_direct = np.mean(rms_errors_direct)
            
            # Create separate plots for each channel
            for ch in range(num_channels):
                plt.figure(figsize=(15, 6))
                plt.title(f"Time Series Comparison (Test ID: {current_test_id}, Channel {ch+1})")
                plt.plot(time_axis, original_ts[:, ch], 'b-', linewidth=1.5, alpha=0.8, label='Original')
                plt.plot(time_axis, recon_ts_direct[0, :, ch], 'r-', linewidth=1.5, alpha=0.8, 
                         label=f'Reconstructed (Direct) - RMS Error: {rms_errors_direct[ch]:.4f}')
                
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{save_dir}/ts_comparison_{current_test_id}_ch{ch+1}.png", dpi=300)
                plt.close()
            
            # Multi-channel overview plot
            plt.figure(figsize=(20, 15))
            plt.suptitle(f"Time Series Comparison (Test ID: {current_test_id}) - Mean RMS Error: {mean_rms_error_direct:.4f}", fontsize=16)
            
            # Determine grid layout
            cols = 3  # Number of columns in the grid
            rows = int(np.ceil(num_channels / cols))
            
            for ch in range(num_channels):
                plt.subplot(rows, cols, ch+1)
                
                # Plot original and reconstructed time series
                plt.plot(time_axis, original_ts[:, ch], 'b-', linewidth=1.5, alpha=0.8, label='Original')
                plt.plot(time_axis, recon_ts_direct[0, :, ch], 'r-', linewidth=1.5, alpha=0.8, label='Reconstructed')
                
                plt.title(f"Channel {ch+1} - RMS Error: {rms_errors_direct[ch]:.4f}")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.grid(True, alpha=0.3)
                
                # Only add legend to first subplot to save space
                if ch == 0:
                    plt.legend()
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for suptitle
            plt.savefig(f"{save_dir}/all_channels_{current_test_id}.png", dpi=300)
            plt.close()
            
            # Save metadata
            with open(f"{save_dir}/conditional_{current_test_id}_metrics.txt", "w") as f:
                f.write(f"Test ID: {current_test_id}\n")
                f.write(f"Direct Reconstruction Mean RMS Error: {mean_rms_error_direct:.6f}\n")
                f.write(f"From-Mask Reconstruction Mean RMS Error: {mean_rms_error_from_mask:.6f}\n")
                f.write("\nPer-channel RMS Errors (Direct):\n")
                for ch in range(num_channels):
                    f.write(f"Channel {ch+1} RMS Error: {rms_errors_direct[ch]:.6f}\n")
            
        except Exception as e:
            print(f"Error processing test ID {current_test_id}: {e}")
    
    print(f"✅ Generated samples for {len(processed_test_ids)} unique test IDs")

def test_reconstruction(model, test_data=None, save_dir="reconstruction_tests"):
    """
    Test reconstruction capabilities by feeding one time-series segment per unique test ID
    (rather than all segments). This ensures we only reconstruct once per ID.
    """

    os.makedirs(save_dir, exist_ok=True)

    # 1) If no test data, load from dataset
    if test_data is None:
        try:
            accel_dict, binary_masks, heatmaps = data_loader.load_data()

            # Grab a few test IDs
            chosen_ids = list(accel_dict.keys())[:3]

            from vae_generator import segment_and_transform
            raw_segments, mask_segments, test_ids = segment_and_transform(
                {k: accel_dict[k] for k in chosen_ids},
                {k: heatmaps[k] for k in chosen_ids},
                segment_duration=4.0
            )
            spec_features = compute_complex_spectrogram(
                raw_segments, fs=200, nperseg=256, noverlap=224
            )
            test_data = {
                'specs': spec_features,
                'masks': mask_segments,
                'raw': raw_segments,
                'test_ids': test_ids
            }
            print(f"✅ Loaded {len(raw_segments)} segments for reconstruction testing")

        except Exception as e:
            print(f"❌ Failed to load test data: {e}")
            return

    # 2) Identify unique test IDs in test_data
    unique_ids = np.unique(test_data['test_ids'])
    print(f"Found {len(unique_ids)} unique IDs in the test data.")

    # 3) For each test ID, pick exactly one segment index
    processed_count = 0
    for test_id in unique_ids:
        # Find all indices for this test_id
        indices_for_id = np.where(test_data['test_ids'] == test_id)[0]
        if len(indices_for_id) == 0:
            continue

        # Just pick the first occurrence
        idx = indices_for_id[0]

        spec_in = tf.convert_to_tensor(test_data['specs'][idx:idx+1])
        mask_in = tf.convert_to_tensor(test_data['masks'][idx:idx+1])

        # Reconstruct
        recon_results = model(spec_in, mask_in, training=False)
        recon_spec = recon_results[0]
        recon_mask = recon_results[1]

        # Original time series
        original_ts = test_data['raw'][idx]
        time_axis = np.linspace(0, 4, original_ts.shape[0])

        # Reconstruct the time series from the model's spectrogram
        recon_ts = model.reconstruct_time_series(
            recon_spec.numpy(),
            fs=200, nperseg=256, noverlap=224,
            time_length=original_ts.shape[0]
        )

        # Calculate errors
        num_channels = original_ts.shape[1]
        rms_errors = np.zeros(num_channels)
        for ch in range(num_channels):
            original = original_ts[:, ch]
            reconstructed = recon_ts[0, :, ch]
            rms_errors[ch] = np.sqrt(np.mean((original - reconstructed)**2))
        mean_rms_error = np.mean(rms_errors)

        # --- Plot 1: Masks (coarse 32×96) ---
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(mask_in[0, :, :, 0].numpy(), aspect='auto', cmap='gray', vmin=0, vmax=1)
        plt.title(f"Original Mask (Test {test_id})")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(recon_mask[0, :, :, 0].numpy(), aspect='auto', cmap='gray', vmin=0, vmax=1)
        plt.title("Reconstructed Mask")
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f"{save_dir}/test_{test_id}_masks.png", dpi=300)
        plt.close()

        # --- NEW: Upsample and plot side-by-side in 'hot' colormap ---
        orig_coarse_np = mask_in[0, :, :, 0].numpy()[None, ...]     # (1,32,96)
        recon_coarse_np = recon_mask[0, :, :, 0].numpy()[None, ...] # (1,32,96)

        upsampled_orig = mask_recon(orig_coarse_np, target_size=(256, 768), interpolation=1)  # cv2.INTER_LINEAR=1
        upsampled_recon = mask_recon(recon_coarse_np, target_size=(256, 768), interpolation=1)

        # Plot the first slice from each
        upsampled_orig_2d = upsampled_orig[0]   # shape (256,768)
        upsampled_recon_2d = upsampled_recon[0] # shape (256,768)

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(upsampled_orig_2d, cmap='hot', vmin=0, vmax=1)
        plt.title(f"Upsampled Original Mask (Test {test_id})")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(upsampled_recon_2d, cmap='hot', vmin=0, vmax=1)
        plt.title("Upsampled Reconstructed Mask")
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f"{save_dir}/test_{test_id}_masks_upsampled.png", dpi=300)
        plt.close()
        # --- end upsample plot ---

        # --- Plot 2: Spectrograms (first channel) ---
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(spec_in[0, :, :, 0].numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"Original Spectrogram (Ch 1) - Test {test_id}")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(recon_spec[0, :, :, 0].numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.title("Reconstructed Spectrogram (Ch 1)")
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f"{save_dir}/test_{test_id}_specs.png", dpi=300)
        plt.close()

        # --- Plot 3: All channels overlay (blue=original, red=reconstructed) ---
        plt.figure(figsize=(12, 6))
        for ch in range(num_channels):
            plt.plot(time_axis, original_ts[:, ch], 'b-', alpha=0.4)
            plt.plot(time_axis, recon_ts[0, :, ch], 'r-', alpha=0.4)

        plt.title(f"Test {test_id} - All Channels Overlaid")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.legend(["Original (all ch)", "Reconstructed (all ch)"])
        plt.tight_layout()
        plt.savefig(f"{save_dir}/test_{test_id}_time_series_overlay.png", dpi=300)
        plt.close()

        # --- Plot 4: Per-channel subplots ---
        plt.figure(figsize=(20, 15))
        plt.suptitle(f"Test {test_id} - Time Series Comparison (Mean RMS Error: {mean_rms_error:.4f})",
                     fontsize=16)
        cols = 3
        rows = int(np.ceil(num_channels / cols))

        for ch in range(num_channels):
            plt.subplot(rows, cols, ch+1)
            plt.plot(time_axis, original_ts[:, ch], 'b-', linewidth=1.5, alpha=0.8, label='Original')
            plt.plot(time_axis, recon_ts[0, :, ch], 'r-', linewidth=1.5, alpha=0.8, label='Reconstructed')
            plt.title(f"Channel {ch+1} - RMS: {rms_errors[ch]:.4f}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)
            if ch == 0:
                plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{save_dir}/test_{test_id}_time_series_detail.png", dpi=300)
        plt.close()

        # --- Write out RMS error metrics ---
        with open(f"{save_dir}/test_{test_id}_metrics.txt", "w") as f:
            f.write(f"Test ID: {test_id}\n")
            f.write(f"Index used: {idx}\n")
            f.write(f"Mean RMS Error: {mean_rms_error:.6f}\n")
            for ch in range(num_channels):
                f.write(f"Channel {ch+1} RMS Error: {rms_errors[ch]:.6f}\n")

        processed_count += 1
        print(f"Done reconstruction for Test ID {test_id}")

    print(f"\n✅ Completed single-segment reconstruction tests for {processed_count} test IDs.")

# -------------------------------------------------------------------
# 3) Main function
# -------------------------------------------------------------------
def main():
    # Adjust these paths to match your project structure
    # Based on your logs, model was saved at early stopping
    weights_path = "results_mmvae/final_spectral_mmvae.weights.h5"
    output_dir = "synthesis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load trained model
    model = load_trained_mmvae(weights_path=weights_path, latent_dim=128)
    
    # Synthesis options - uncomment the ones you want to use
    
    # # 1. Generate random samples from the prior
    # generate_random_samples(
    #     model, 
    #     how_many=5, 
    #     save_dir=f"{output_dir}/random_samples"
    # )
    
    # # 2. Generate interpolations in latent space
    # generate_interpolations(
    #     model, 
    #     num_interpolation_steps=8, 
    #     save_dir=f"{output_dir}/interpolations"
    # )
    
    # 3. Generate conditional samples (if data available)
    generate_conditional_samples(
        model, 
        conditioning_data=None,  # Will try to load from your dataset
        save_dir=f"{output_dir}/conditional_samples"
    )

    test_reconstruction(
        model, 
        test_data=None,  # Will try to load from your dataset
        save_dir=f"{output_dir}/reconstruction_tests"
    )
    
    print(f"\n✅ All synthesis tasks completed. Results saved in '{output_dir}'")
    
if __name__ == "__main__":
    main()