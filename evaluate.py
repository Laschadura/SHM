import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import umap
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from scipy.stats import ks_2samp
from scipy.signal import welch
from sklearn.decomposition import PCA

# Import your real data loader
import data_loader

data_gen_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DataGenMethods")
if data_gen_path not in sys.path:
    sys.path.append(data_gen_path)

# Ensure the output directory exists
output_dir = "evaluation_plots"
os.makedirs(output_dir, exist_ok=True)

##########################
# 1. Data Loading
##########################

def load_real_data():
    """Loads and segments real data into 5s impact events."""
    accel_dict, mask_dict = data_loader.load_data()
    signals, masks, test_ids = [], [], []

    for test_id, all_ts in accel_dict.items():
        mask_data = mask_dict[test_id]

        for ts_raw in all_ts:
            segment_size = 1000  # 5s at 200Hz
            for i in range(0, ts_raw.shape[0] - segment_size + 1, segment_size):
                segment = ts_raw[i:i+segment_size, :]
                signals.append(segment)
                masks.append(mask_data)
                test_ids.append(test_id)

    # Debug: Print loaded test IDs
    print(f"[DEBUG] Loaded {len(test_ids)} real test IDs. First 10: {test_ids[:10]}")
    masks_array = np.stack(masks)
    print(f"[DEBUG] Shape Masks: {masks_array.shape}")
    return np.stack(signals), masks_array, np.array(test_ids)

def load_synthetic_data(synthetic_folder):
    """Loads synthetic 5s segments and masks."""
    ts_files = sorted(glob.glob(os.path.join(synthetic_folder, "gen_ts_*.npy")))
    mask_files = sorted(glob.glob(os.path.join(synthetic_folder, "gen_mask_*.npy")))

    if not ts_files or not mask_files:
        raise FileNotFoundError(f"No synthetic sample files found in folder: {synthetic_folder}")

    signals, masks, test_ids = [], [], []

    for ts_file, mask_file in zip(ts_files, mask_files):
        ts = np.load(ts_file).squeeze(0)  # Shape: (1000, 12)
        m = np.load(mask_file)  # Shape: (256, 768)

        test_id = int(ts_file.split("_")[-1].split(".")[0])  # Extract test ID from filename
        signals.append(ts)
        masks.append(m)
        test_ids.append(test_id)

    # Debug: Print loaded synthetic test IDs
    print(f"[DEBUG] Loaded {len(test_ids)} synthetic test IDs. First 10: {test_ids[:10]}")
    masks_array = np.stack(masks)
    print(f"[DEBUG] Shape Masks: {masks_array.shape}")
    return np.stack(signals), masks_array, np.array(test_ids)


##########################
# 2. Feature Computation
##########################

def compute_features(data, sample_rate=200):
    """Computes FFT, RMS, and PSD features for time-series data."""
    fft_features, rms_features, psd_features = [], [], []

    for sample in data:
        fft_vals = np.abs(np.fft.rfft(sample, axis=0))  # Shape: (freq_bins, num_channels)
        fft_feature = np.linalg.norm(fft_vals, axis=1)  # Reduce across channels

        rms_feature = np.sqrt(np.mean(sample**2, axis=1))  # Per time step

        psd_list = []
        for ch in range(sample.shape[1]):
            f, pxx = welch(sample[:, ch], fs=sample_rate)
            psd_list.append(pxx)
        psd_feature = np.mean(np.array(psd_list), axis=0)  # Mean across channels

        fft_features.append(fft_feature)
        rms_features.append(rms_feature)
        psd_features.append(psd_feature)

    return np.array(fft_features), np.array(rms_features), np.array(psd_features)

def compute_mask_features(masks):
    """Flattens masks for UMAP analysis."""
    return np.array([mask.flatten() for mask in masks])

##########################
# 3. UMAP Visualization
##########################

def plot_umap_3d_combined(real_features, real_labels, synthetic_features, synthetic_labels, title, filename):
    """Plots real & synthetic data in a single UMAP 3D scatter plot."""
    reducer = umap.UMAP(n_components=3, random_state=42)
    
    # Combine real and synthetic features
    combined_features = np.vstack([real_features, synthetic_features])
    combined_labels = np.hstack([real_labels, synthetic_labels])
    
    # Apply UMAP reduction
    embedding = reducer.fit_transform(combined_features)

    # Assign different color gradients for real (reds) vs. synthetic (blues)
    colors = np.hstack([
        np.linspace(0.2, 1.0, len(real_labels)),  # Real data (progressing from light red to dark red)
        np.linspace(0.2, 1.0, len(synthetic_labels))  # Synthetic data (progressing from light blue to dark blue)
    ])
    color_scale = np.hstack([
        np.full(len(real_labels), "Reds"),  
        np.full(len(synthetic_labels), "Blues")  
    ])

    # Create a single 3D scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=embedding[:len(real_labels), 0],
        y=embedding[:len(real_labels), 1],
        z=embedding[:len(real_labels), 2],
        mode='markers',
        marker=dict(
            size=4,
            color=colors[:len(real_labels)],
            colorscale="Reds",  # Real data: Red gradient
            opacity=0.7
        ),
        name="Real Data"
    ))

    fig.add_trace(go.Scatter3d(
        x=embedding[len(real_labels):, 0],
        y=embedding[len(real_labels):, 1],
        z=embedding[len(real_labels):, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=colors[len(real_labels):],
            colorscale="Blues",  # Synthetic data: Blue gradient
            opacity=0.7
        ),
        name="Synthetic Data"
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3"
        ),
        title=title
    )

    pio.write_html(fig, file=os.path.join(output_dir, filename), auto_open=False)
    print(f"Saved interactive combined UMAP plot: {filename}")

def check_damage_correlation_combined(real_features, real_labels, synthetic_features, synthetic_labels, dataset_name):
    """Computes Spearman correlation between UMAP components and damage progression for real & synthetic data."""
    reducer = umap.UMAP(n_components=3, random_state=42)
    
    # Combine real and synthetic features
    combined_features = np.vstack([real_features, synthetic_features])
    combined_labels = np.hstack([real_labels, synthetic_labels])
    
    # Apply UMAP
    embedding = reducer.fit_transform(combined_features)

    # Compute correlation separately for real and synthetic data
    real_corr_x, _ = spearmanr(embedding[:len(real_labels), 0], real_labels)
    real_corr_y, _ = spearmanr(embedding[:len(real_labels), 1], real_labels)
    real_corr_z, _ = spearmanr(embedding[:len(real_labels), 2], real_labels)

    synthetic_corr_x, _ = spearmanr(embedding[len(real_labels):, 0], synthetic_labels)
    synthetic_corr_y, _ = spearmanr(embedding[len(real_labels):, 1], synthetic_labels)
    synthetic_corr_z, _ = spearmanr(embedding[len(real_labels):, 2], synthetic_labels)

    print(f"\n[INFO] Spearman Correlation of {dataset_name} UMAP with Damage Progression:")
    print(f"    Real Data - UMAP 1: {real_corr_x:.4f}, UMAP 2: {real_corr_y:.4f}, UMAP 3: {real_corr_z:.4f}")
    print(f"    Synthetic Data - UMAP 1: {synthetic_corr_x:.4f}, UMAP 2: {synthetic_corr_y:.4f}, UMAP 3: {synthetic_corr_z:.4f}\n")

    return (real_corr_x, real_corr_y, real_corr_z), (synthetic_corr_x, synthetic_corr_y, synthetic_corr_z)

def plot_3d_damage_mask(real_masks, synthetic_masks, output_dir="evaluation_plots"):
    """Generates two separate 3D surface plots for real and synthetic damage masks, ensuring correct elevation order and solid color."""

    def process_mask(mask_samples, invert=False):
        """Sums up all masks across test IDs, removes zero-damage pixels, and optionally inverts elevation ordering."""
        
        # If mask_samples has an extra singleton dimension, remove it
        if mask_samples.ndim == 4 and mask_samples.shape[1] == 1:
            mask_samples = mask_samples.squeeze(1)  # Removes the second dimension, making it (100, 256, 768)

        # If the mask is 3D (test_ids, H, W), sum across the first dimension to aggregate all test IDs
        if mask_samples.ndim == 3:
            mask_samples = np.sum(mask_samples, axis=0)

        # Ensure the mask is 2D before proceeding
        if mask_samples.ndim != 2:
            raise ValueError(f"Expected a 2D mask, but got shape {mask_samples.shape}")

        # **Invert the mask (flip vertically) if needed**
        if invert:
            mask_samples = np.flipud(mask_samples)

        H, W = mask_samples.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))

        # Flatten arrays
        x_flat, y_flat, z_flat = x.ravel(), y.ravel(), mask_samples.ravel()

        # Filter out zero-damage pixels
        nonzero_idx = np.where(z_flat > 0)
        return x_flat[nonzero_idx], y_flat[nonzero_idx], z_flat[nonzero_idx]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process real and synthetic masks
    x_real, y_real, z_real = process_mask(real_masks, invert=True)  # Flip real damage mask to correct elevation
    x_synthetic, y_synthetic, z_synthetic = process_mask(synthetic_masks, invert=False)

    # Create a grid for plotting
    def create_grid(x, y, z):
        """Reconstructs a structured grid from scattered (x, y, z) points for surface plotting."""
        H = max(y) + 1
        W = max(x) + 1
        grid = np.zeros((H, W))  # Initialize empty grid

        for i in range(len(x)):
            grid[int(y[i]), int(x[i])] = z[i]  # Assign values at respective locations

        return grid

    # Convert scattered points back into a 2D grid
    real_z_grid = create_grid(x_real.astype(int), y_real.astype(int), z_real)
    synthetic_z_grid = create_grid(x_synthetic.astype(int), y_synthetic.astype(int), z_synthetic)

    # **Plot for Real Damage Mask with SOLID COLOR**
    fig_real = go.Figure()
    fig_real.add_trace(go.Surface(z=real_z_grid, surfacecolor=np.ones_like(real_z_grid) * 0.8, colorscale=[[0, 'red'], [1, 'red']], showscale=False))
    fig_real.update_layout(title="Real Damage Mask 3D",
                           scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Damage Level"),
                           margin=dict(l=0, r=0, b=0, t=40),
                           height=600, width=800)

    # Save HTML
    real_filename = os.path.join(output_dir, "3D_Real_Damage_Mask.html")
    pio.write_html(fig_real, file=real_filename, auto_open=False)
    print(f"Saved 3D real damage mask plot: {real_filename}")

    # **Plot for Synthetic Damage Mask with SOLID COLOR**
    fig_synthetic = go.Figure()
    fig_synthetic.add_trace(go.Surface(z=synthetic_z_grid, surfacecolor=np.ones_like(synthetic_z_grid) * 0.8, colorscale=[[0, 'blue'], [1, 'blue']], showscale=False))
    fig_synthetic.update_layout(title="Synthetic Damage Mask 3D",
                                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Damage Level"),
                                margin=dict(l=0, r=0, b=0, t=40),
                                height=600, width=800)

    # Save HTML
    synthetic_filename = os.path.join(output_dir, "3D_Synthetic_Damage_Mask.html")
    pio.write_html(fig_synthetic, file=synthetic_filename, auto_open=False)
    print(f"Saved 3D synthetic damage mask plot: {synthetic_filename}")

def print_sample_masks(real_masks, synthetic_masks):
    """Prints two example 2D masks (one real, one synthetic) in the terminal."""
    
    # Ensure masks are properly shaped (remove singleton dimensions if necessary)
    if real_masks.ndim == 4 and real_masks.shape[1] == 1:
        real_masks = real_masks.squeeze(1)  # (N, 256, 768)
    if synthetic_masks.ndim == 4 and synthetic_masks.shape[1] == 1:
        synthetic_masks = synthetic_masks.squeeze(1)

    # Select two random test cases (one from real, one from synthetic)
    real_sample_idx = np.random.randint(0, real_masks.shape[0])
    synthetic_sample_idx = np.random.randint(0, synthetic_masks.shape[0])

    real_mask_sample = real_masks[real_sample_idx]
    synthetic_mask_sample = synthetic_masks[synthetic_sample_idx]

    # Print to the terminal
    print("\n========== Sample Real Mask ==========")
    np.set_printoptions(threshold=50, linewidth=150)  # Controls output format
    print(real_mask_sample)

    print("\n========== Sample Synthetic Mask ==========")
    print(synthetic_mask_sample)

    # Reset print options to default (avoid affecting other outputs)
    np.set_printoptions(threshold=1000, linewidth=75)

##########################
# 4. Evaluation Pipeline
##########################

def main(args):
    # Load real & synthetic data
    real_signals, real_masks, real_ids = load_real_data()
    synthetic_signals, synthetic_masks, synthetic_ids = load_synthetic_data(args.synthetic_folder)

    # Debug: Print test ID distribution
    print(f"\n[DEBUG] Real Data - Unique Test IDs: {np.unique(real_ids)}")
    print(f"[DEBUG] Synthetic Data - Unique Test IDs: {np.unique(synthetic_ids)}\n")

    # Ensure same sample size
    min_samples = min(real_signals.shape[0], synthetic_signals.shape[0])
    real_signals, real_masks, real_ids = real_signals[:min_samples], real_masks[:min_samples], real_ids[:min_samples]
    synthetic_signals, synthetic_masks, synthetic_ids = synthetic_signals[:min_samples], synthetic_masks[:min_samples], synthetic_ids[:min_samples]

    # Compute features
    real_fft, real_rms, real_psd = compute_features(real_signals)
    synthetic_fft, synthetic_rms, synthetic_psd = compute_features(synthetic_signals)

    real_mask_features = compute_mask_features(real_masks)
    synthetic_mask_features = compute_mask_features(synthetic_masks)

    # Compute reconstruction error (MSE)
    mse = np.mean((real_signals - synthetic_signals) ** 2)
    print(f"Mean Squared Reconstruction Error: {mse:.4f}")

    # KS Test for real vs. synthetic distributions
    ks_stat, ks_p = ks_2samp(real_signals.flatten(), synthetic_signals.flatten())
    print(f"KS Test Statistic: {ks_stat:.4f}, p-value: {ks_p:.4f}")

    # UMAP Visualization
    plot_umap_3d_combined(
        real_fft, real_ids, synthetic_fft, synthetic_ids,
        title="UMAP 3D of FFT Features (Real vs. Synthetic)",
        filename="UMAP_FFT_Combined.html"
    )

    plot_umap_3d_combined(
        real_rms, real_ids, synthetic_rms, synthetic_ids,
        title="UMAP 3D of RMS Features (Real vs. Synthetic)",
        filename="UMAP_RMS_Combined.html"
    )

    plot_umap_3d_combined(
        real_psd, real_ids, synthetic_psd, synthetic_ids,
        title="UMAP 3D of PSD Features (Real vs. Synthetic)",
        filename="UMAP_PSD_Combined.html"
    )

    plot_umap_3d_combined(
        real_mask_features, real_ids, synthetic_mask_features, synthetic_ids,
        title="UMAP 3D of Mask Features (Real vs. Synthetic)",
        filename="UMAP_Mask_Combined.html"
    )

    print("\nChecking if VAE captures damage progression...\n")

    check_damage_correlation_combined(real_fft, real_ids, synthetic_fft, synthetic_ids, "FFT Features")
    check_damage_correlation_combined(real_rms, real_ids, synthetic_rms, synthetic_ids, "RMS Features")
    check_damage_correlation_combined(real_psd, real_ids, synthetic_psd, synthetic_ids, "PSD Features")
    check_damage_correlation_combined(real_mask_features, real_ids, synthetic_mask_features, synthetic_ids, "Mask Features")

    print_sample_masks(real_masks, synthetic_masks)


    plot_3d_damage_mask(real_masks, synthetic_masks)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script for Synthetic Data")
    parser.add_argument("--synthetic_folder", type=str, required=True, help="Folder containing synthetic data")
    args = parser.parse_args()
    main(args)
