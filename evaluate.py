import os, sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import your real data loader
import data_loader

data_gen_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DataGenMethods")
if data_gen_path not in sys.path:
    sys.path.append(data_gen_path)

##########################
# 1. Data Loading
##########################
def load_real_data():
    """
    Load held-out real test data using your data_loader module.
    Aggregates all samples from the returned dictionary.
    
    Returns:
      - signals: NumPy array of shape [num_samples, 12000, 12] for the time series data.
      - masks: NumPy array of shape [num_samples, H, W] for the corresponding masks.
    """
    accel_dict, mask_dict = data_loader.load_data()
    signals = []
    masks = []
    # Iterate over all tests in the dictionary.
    for test_id, samples in accel_dict.items():
        # Assume that each sample in the test uses the same mask from mask_dict.
        for sample in samples:
            signals.append(sample)
            masks.append(mask_dict[test_id])
    signals = np.stack(signals)
    masks = np.stack(masks)
    return signals, masks



def load_synthetic_data(synthetic_folder):
    """
    Load synthetic data from the specified folder.
    This function searches for pairs of .npy files:
      - synthetic_sample_*_acc.npy for time series data,
      - synthetic_sample_*_mask.npy for corresponding mask data.
    
    It then aggregates them into NumPy arrays.
    
    Returns:
      signals: NumPy array of shape [num_samples, 12000, 12] (or whatever shape your data has).
      masks: NumPy array of shape [num_samples, H, W] for the corresponding masks.
      metadata: None (or you can populate this if needed).
    """

    # Find all accelerometer files and mask files
    ts_files = sorted(glob.glob(os.path.join(synthetic_folder, "synthetic_sample_*_acc.npy")))
    mask_files = sorted(glob.glob(os.path.join(synthetic_folder, "synthetic_sample_*_mask.npy")))
    
    if not ts_files or not mask_files:
        raise FileNotFoundError("No synthetic sample files found in folder: " + synthetic_folder)
    
    signals = []
    masks = []
    
    # Assuming files are named so that their sorted order matches (e.g., sample_0, sample_1, ...)
    for ts_file, mask_file in zip(ts_files, mask_files):
        ts = np.load(ts_file)
        m = np.load(mask_file)
        signals.append(ts)
        masks.append(m)
    
    signals = np.stack(signals)
    masks = np.stack(masks)
    metadata = None  # You can add metadata if needed.
    return signals, masks, metadata


##########################
# 2. Reconstruction Error Analysis
##########################
def calculate_reconstruction_error(real_data, synthetic_data):
    """
    Calculate the Mean Squared Error (MSE) between real and synthetic data.
    If the number of samples differs, randomly subsample from the larger array
    so that both have the same number of samples.
    """
    n_real = real_data.shape[0]
    n_syn = synthetic_data.shape[0]
    if n_real > n_syn:
        idx = np.random.choice(n_real, n_syn, replace=False)
        real_data = real_data[idx]
    elif n_syn > n_real:
        idx = np.random.choice(n_syn, n_real, replace=False)
        synthetic_data = synthetic_data[idx]
    
    mse = np.mean((real_data - synthetic_data)**2)
    return mse


##########################
# 3. Statistical Similarity Tests
##########################
def run_ks_test(real_data, synthetic_data):
    """
    Flatten the data and run the Kolmogorov–Smirnov test to compare the overall distributions.
    """
    real_flat = real_data.flatten()
    synthetic_flat = synthetic_data.flatten()
    statistic, p_value = ks_2samp(real_flat, synthetic_flat)
    return statistic, p_value

def spectral_analysis(signal, fs=200.0):
    """
    Compute the Fourier transform of a 1D signal to analyze its frequency content.
    
    Args:
      signal: 1D numpy array.
      fs: Sampling frequency in Hz.
    
    Returns:
      freqs: Frequency bins in Hz.
      fft_vals: Magnitude of the FFT.
    """
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal), d=1/fs)
    return freqs, fft_vals


def plot_spectral_comparison(real_signal, synthetic_signal, sample_idx=0):
    """
    Plot the frequency spectra of a pair of real and synthetic signals for all channels.
    
    For clarity:
      - All real channels are plotted in a consistent dark navy color.
      - All synthetic channels are plotted in a consistent red color.
    
    Args:
      real_signal: NumPy array of shape [time_steps, channels].
      synthetic_signal: NumPy array of shape [time_steps, channels].
      sample_idx: The sample index (for title display).
    """
    num_channels = real_signal.shape[1]
    plt.figure(figsize=(10, 5))
    
    # Plot real channels in navy color.
    for ch in range(num_channels):
        r_ch = real_signal[:, ch]
        freqs, fft_real = spectral_analysis(r_ch)
        if ch == 0:
            plt.plot(freqs, fft_real, label='Real', color='navy', alpha=0.7)
        else:
            plt.plot(freqs, fft_real, color='navy', alpha=0.7)
    
    # Plot synthetic channels in red color.
    for ch in range(num_channels):
        s_ch = synthetic_signal[:, ch]
        freqs, fft_synthetic = spectral_analysis(s_ch)
        if ch == 0:
            plt.plot(freqs, fft_synthetic, label='Synthetic', color='red', alpha=0.7)
        else:
            plt.plot(freqs, fft_synthetic, color='red', alpha=0.7)
    
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title(f"Spectral Analysis Comparison (Sample {sample_idx})")
    plt.legend()
    plt.show()


##########################
# 4. Latent Space Evaluation and Optional Interpolation
##########################
def latent_space_visualization(latent_real, latent_synthetic, method="pca"):
    """
    Visualize the latent space of real and synthetic samples using PCA or t-SNE.
    """
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2)
    else:
        raise ValueError(f"Unknown latent visualization method: {method}")
    
    combined = np.concatenate([latent_real, latent_synthetic], axis=0)
    reduced = reducer.fit_transform(combined)
    n_real = latent_real.shape[0]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:n_real, 0], reduced[:n_real, 1], label='Real', alpha=0.7)
    plt.scatter(reduced[n_real:, 0], reduced[n_real:, 1], label='Synthetic', alpha=0.7)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"Latent Space Visualization using {method.upper()}")
    plt.legend()
    plt.show()


def interpolation_experiment(encoder, decoder, real_data, num_steps=10):
    """
    Perform an interpolation experiment between two real samples:
      - Encode two real samples into the latent space.
      - Interpolate between the latent vectors.
      - Decode the intermediate latent vectors to generate interpolated samples.
      
    Note:
      - The decoder is assumed to return a tuple: (reconstructed time series, reconstructed mask).
      - For the time series, we plot the first channel.
      - For the mask, we use imshow to display the binary image.
    """
    sample1 = real_data[0]
    sample2 = real_data[1]
    
    # Encode the two samples.
    z1 = encoder(sample1[np.newaxis, :])
    z2 = encoder(sample2[np.newaxis, :])
    
    interpolated_ts = []
    interpolated_masks = []
    alphas = np.linspace(0, 1, num_steps)
    
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        # decoder now returns a tuple (recon_ts, recon_mask)
        recon_ts, recon_mask = decoder(z_interp)
        interpolated_ts.append(recon_ts.numpy().squeeze(0))
        interpolated_masks.append(recon_mask.numpy().squeeze(0))
    
    # Create a colormap that assigns a distinct color to each alpha.
    cmap = plt.get_cmap("viridis", num_steps)
    
    plt.figure(figsize=(12, 6))
    for i, ts in enumerate(interpolated_ts):
        # Now each line gets its own label, so the legend will list all alpha values.
        plt.plot(ts[:, 0], color=cmap(i), label=f"alpha={alphas[i]:.2f}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude (Channel 1)")
    plt.title("Time Series Interpolation Experiment")
    plt.legend()
    plt.show()
    
    # Plot mask interpolation: show each mask as an image in a row.
    n = len(interpolated_masks)
    plt.figure(figsize=(15, 3))
    for i, mask in enumerate(interpolated_masks):
        plt.subplot(1, n, i+1)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.title(f"alpha={alphas[i]:.2f}")
    plt.suptitle("Mask Interpolation Experiment")
    plt.show()


##########################
# 5. Diversity and Coverage Metrics
##########################
def diversity_coverage_metrics(real_data, synthetic_data):
    """
    Compare variances along features as a simple metric of diversity and coverage.
    """
    real_var = np.var(real_data, axis=0)
    synthetic_var = np.var(synthetic_data, axis=0)
    diversity_score = np.mean(np.abs(real_var - synthetic_var))
    return diversity_score

##########################
# 6. Main Routine
##########################
def main(args):
    # Load data and unpack the tuple into real_signals and real_masks.
    real_signals, real_masks = load_real_data()
    print(f"Loaded real signals with shape: {real_signals.shape}")
    print(f"Loaded real masks with shape: {real_masks.shape}")
    
    synthetic_data, synthetic_masks, metadata = load_synthetic_data(args.synthetic_folder)
    print(f"Loaded synthetic data with shape: {synthetic_data.shape}")
    
    # Use real_signals for evaluation tests:
    mse = calculate_reconstruction_error(real_signals, synthetic_data)
    print(f"Mean Squared Reconstruction Error: {mse:.4f}")
    
    ks_stat, ks_p = run_ks_test(real_signals, synthetic_data)
    print(f"KS Test Statistic: {ks_stat:.4f}, p-value: {ks_p:.4f}")
    
    if real_signals.shape[0] > 0 and synthetic_data.shape[0] > 0:
        plot_spectral_comparison(real_signals[0], synthetic_data[0], sample_idx=0)
    
    if args.use_latent:
        latent_real_file = os.path.join(args.synthetic_folder, "latent_real.npy")
        latent_synthetic_file = os.path.join(args.synthetic_folder, "latent_synthetic.npy")
        if os.path.exists(latent_real_file) and os.path.exists(latent_synthetic_file):
            latent_real = np.load(latent_real_file)
            latent_synthetic = np.load(latent_synthetic_file)
            latent_space_visualization(latent_real, latent_synthetic, method=args.latent_method)
        else:
            print("Latent representations not found in synthetic folder. Skipping latent space visualization.")
    
    diversity_score = diversity_coverage_metrics(real_signals, synthetic_data)
    print(f"Diversity Coverage Score (mean absolute variance difference): {diversity_score:.4f}")
    
    if args.use_interpolation:
        if args.model in ["vae", "gan"]:
            if args.encoder_module and args.decoder_module:
                import importlib
                encoder_module = importlib.import_module(args.encoder_module)
                decoder_module = importlib.import_module(args.decoder_module)
                # Load the trained model weights so that vae_model is set in the module.
                encoder_module.load_trained_model("results/vae_weights.h5")
                
                encoder = encoder_module.encoder
                decoder = decoder_module.decoder
                # Pass real_signals to the interpolation experiment.
                interpolation_experiment(encoder, decoder, real_signals)
            else:
                print("Encoder/Decoder modules not specified. Skipping interpolation experiment.")
        else:
            print(f"Interpolation experiment skipped because the chosen model ({args.model}) does not support latent space interpolation.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script for Synthetic Data")
    parser.add_argument("--synthetic_folder", type=str, required=True,
                        help="Folder containing synthetic data (e.g., vae_results) and optionally latent representations")
    parser.add_argument("--use_latent", action="store_true",
                        help="Perform latent space visualization if latent representations are available")
    parser.add_argument("--latent_method", type=str, default="pca", choices=["pca", "tsne"],
                        help="Method for latent space visualization (pca or tsne)")
    parser.add_argument("--use_interpolation", action="store_true",
                        help="Perform interpolation experiments for models with a latent space (e.g., VAE, GAN)")
    parser.add_argument("--encoder_module", type=str,
                        help="Python module path for the encoder (e.g., vae_generator)")
    parser.add_argument("--decoder_module", type=str,
                        help="Python module path for the decoder (e.g., vae_generator)")
    parser.add_argument("--model", type=str, default="vae", choices=["vae", "gan", "diffusion"],
                        help="Specify the type of generative model used. This helps determine if interpolation is applicable.")
    args = parser.parse_args()
    main(args)
