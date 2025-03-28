import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

# Import our modules
from custom_distributions import compute_js_divergence, reparameterize, compute_mixture_prior
from vae_generator import SpectralMMVAE

def test_cross_modal_generation(model, test_data_path="test_data"):
    """
    Test cross-modal generation with the trained model.
    
    Args:
        model: Trained SpectralMMVAE model
        test_data_path: Path to test data
    """
    # Load test data
    # This should be a small representative set of your data
    spec_file = os.path.join(test_data_path, "test_spectrograms.npy")
    desc_file = os.path.join(test_data_path, "test_descriptors.npy")
    
    # Check if files exist
    if not os.path.exists(spec_file) or not os.path.exists(desc_file):
        print(f"❌ Test data not found in {test_data_path}")
        return
    
    # Load data
    test_specs = np.load(spec_file)
    test_descs = np.load(desc_file)
    
    print(f"Loaded test data: {test_specs.shape[0]} samples")
    
    # Create output directory
    output_dir = os.path.join(test_data_path, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process a few samples
    for i in range(min(5, test_specs.shape[0])):
        # Extract single sample
        spec_sample = tf.expand_dims(test_specs[i], 0)
        desc_sample = tf.expand_dims(test_descs[i], 0)
        
        # 1. Normal reconstruction (both modalities)
        recon_spec_both, recon_desc_both, _ = model(
            spec_sample, desc_sample, 
            training=False, 
            missing_modality=None
        )
        
        # 2. Generate spectrogram from descriptor only
        recon_spec_from_desc, _, _ = model(
            spec_sample, desc_sample, 
            training=False, 
            missing_modality='spec'
        )
        
        # 3. Generate descriptor from spectrogram only
        _, recon_desc_from_spec, _ = model(
            spec_sample, desc_sample, 
            training=False, 
            missing_modality='desc'
        )
        
        # Visualize results
        # Original vs reconstructed spectrogram
        fig = plt.figure(figsize=(15, 5))
        
        # Original spectrogram
        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(spec_sample[0, :, :, 0], cmap='viridis', aspect='auto')
        ax1.set_title("Original Spectrogram")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Frequency")
        
        # Reconstructed with both modalities
        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(recon_spec_both[0, :, :, 0].numpy(), cmap='viridis', aspect='auto')
        ax2.set_title("Reconstructed (Both Modalities)")
        ax2.set_xlabel("Time")
        
        # Reconstructed from descriptor only
        ax3 = fig.add_subplot(133)
        im3 = ax3.imshow(recon_spec_from_desc[0, :, :, 0].numpy(), cmap='viridis', aspect='auto')
        ax3.set_title("Generated from Descriptor Only")
        ax3.set_xlabel("Time")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{i}_spectrograms.png"), dpi=300)
        plt.close()
        
        # Report reconstruction quality
        orig_vs_both_mse = np.mean(np.square(spec_sample[0, :, :, 0] - recon_spec_both[0, :, :, 0].numpy()))
        orig_vs_desc_mse = np.mean(np.square(spec_sample[0, :, :, 0] - recon_spec_from_desc[0, :, :, 0].numpy()))
        
        print(f"Sample {i}:")
        print(f"  MSE (original vs both): {orig_vs_both_mse:.6f}")
        print(f"  MSE (original vs desc-only): {orig_vs_desc_mse:.6f}")
        print(f"  Relative quality: {orig_vs_both_mse / orig_vs_desc_mse:.2%}")
        print()

def test_latent_interpolation(model, test_data_path="test_data"):
    """
    Test latent space interpolation between samples.
    
    Args:
        model: Trained SpectralMMVAE model
        test_data_path: Path to test data
    """
    # Load test data
    spec_file = os.path.join(test_data_path, "test_spectrograms.npy")
    desc_file = os.path.join(test_data_path, "test_descriptors.npy")
    
    # Check if files exist
    if not os.path.exists(spec_file) or not os.path.exists(desc_file):
        print(f"❌ Test data not found in {test_data_path}")
        return
    
    # Load data
    test_specs = np.load(spec_file)
    test_descs = np.load(desc_file)
    
    # Create output directory
    output_dir = os.path.join(test_data_path, "results", "interpolation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Pick two samples to interpolate between
    if test_specs.shape[0] < 2:
        print("❌ Need at least 2 samples for interpolation")
        return
    
    # Source and target samples
    source_spec = tf.expand_dims(test_specs[0], 0)
    source_desc = tf.expand_dims(test_descs[0], 0)
    
    target_spec = tf.expand_dims(test_specs[1], 0)
    target_desc = tf.expand_dims(test_descs[1], 0)
    
    # Encode samples to get latent vectors
    source_mus, source_logvars, source_mixture_mu, source_mixture_logvar = model.encode_all_modalities(
        source_spec, source_desc, training=False
    )
    
    target_mus, target_logvars, target_mixture_mu, target_mixture_logvar = model.encode_all_modalities(
        target_spec, target_desc, training=False
    )
    
    # Use mixture means for interpolation
    source_z = source_mixture_mu
    target_z = target_mixture_mu
    
    # Create interpolations in latent space
    num_steps = 10
    alphas = np.linspace(0, 1, num_steps)
    
    # Generate samples from the interpolated latent vectors
    interp_specs = []
    interp_descs = []
    
    for alpha in alphas:
        # Linear interpolation in latent space
        interp_z = (1 - alpha) * source_z + alpha * target_z
        
        # Decode the interpolated latent vector
        interp_spec = model.spec_decoder(interp_z, training=False)
        interp_desc = model.desc_decoder(interp_z, training=False)
        
        interp_specs.append(interp_spec.numpy())
        interp_descs.append(interp_desc.numpy())
    
    # Visualize the interpolation for spectrograms
    # Create a grid of subplots
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, (alpha, interp_spec) in enumerate(zip(alphas, interp_specs)):
        ax = axes[i]
        im = ax.imshow(interp_spec[0, :, :, 0], cmap='viridis', aspect='auto')
        ax.set_title(f"α = {alpha:.1f}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spectrogram_interpolation.png"), dpi=300)
    plt.close()
    
    print("✅ Latent interpolation visualizations saved")

def test_mixture_sampling(model, n_samples=5):
    """
    Test sampling from the mixture prior.
    
    Args:
        model: Trained SpectralMMVAE model
        n_samples: Number of samples to generate
    """
    # Create output directory
    output_dir = "results/mixture_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample from standard normal prior
    latent_dim = model.latent_dim
    z_samples = tf.random.normal(shape=(n_samples, latent_dim))
    
    # Generate samples for each modality
    spec_samples = model.spec_decoder(z_samples, training=False)
    desc_samples = model.desc_decoder(z_samples, training=False)
    
    # Visualize spectrogram samples
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
    
    for i in range(n_samples):
        ax = axes[i] if n_samples > 1 else axes
        im = ax.imshow(spec_samples[i, :, :, 0].numpy(), cmap='viridis', aspect='auto')
        ax.set_title(f"Sample {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "random_spectrograms.png"), dpi=300)
    plt.close()
    
    print(f"✅ Generated {n_samples} random samples from the prior")

def main():
    """
    Main function for testing the MoE MMVAE model.
    """
    # Path to saved model
    model_path = "results/best_spectral_mmvae.weights.h5"
    
    # Model parameters (must match the trained model)
    latent_dim = 64
    spec_shape = (129, 24, 24)  # (freq_bins, time_bins, channels*2)
    max_num_cracks = 770
    desc_length = 42
    
    # Create model with the same architecture
    model = SpectralMMVAE(latent_dim, spec_shape, max_num_cracks, desc_length)
    
    # Build model with dummy inputs to initialize weights
    dummy_spec = tf.zeros((1, *spec_shape))
    dummy_desc = tf.zeros((1, max_num_cracks, desc_length))
    model(dummy_spec, dummy_desc, training=False)
    
    # Load weights
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print(f"✅ Loaded model weights from {model_path}")
    else:
        print(f"❌ Model weights not found at {model_path}")
        return
    
    # Run tests
    test_cross_modal_generation(model)
    test_latent_interpolation(model)
    test_mixture_sampling(model)

if __name__ == "__main__":
    main()