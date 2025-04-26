import os
import sys
import gc
import psutil
import numpy as np
import random
import cv2
import keras
from keras import layers, Model, optimizers
from keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts  #type: ignore
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim #type: ignore
from scipy import signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
import pandas as pd
import umap
import GPUtil
import tensorflow as tf
from tensorflow.keras import mixed_precision #type: ignore

# Multithreading
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

#enable mixed precision
mixed_precision.set_global_policy('mixed_float16')
print(f"Mixed precision policy: {mixed_precision.global_policy()}")

# Set working directory
import platform
if platform.system() == "Windows":
    # Local Windows path
    os.chdir("c:/SP-Master-Local/SP_DamageLocalization-MasonryArchBridge_SimonScandella/ProbabilisticApproach/Euler_MMVAE")
else:
    # Original Euler cluster path
    os.chdir("/cluster/scratch/scansimo/Euler_MMVAE")
print("✅ Script has started executing")

# GPU Configuration
def configure_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"✅ Memory growth enabled on {len(physical_devices)} GPU(s)")
        except RuntimeError as e:
            print(f"❌ GPU Memory Growth Error: {e}")
    else:
        print("⚠️ No GPU devices found — this script will run on CPU.")

    print(f"🔍 TensorFlow will run on: {tf.config.list_logical_devices('GPU')}")

configure_gpu()


# Import custom modules
from custom_distributions import (
    compute_js_divergence,
    reparameterize,
    compute_mixture_prior,
    compute_kl_divergence,
)
from data_loader import inverse_spectrogram as dl_inverse_spectrogram



# Ensure access to sibling files
sys.path.append(os.path.dirname(__file__))
import data_loader


# ----- Recource Monitoring and Usage -----
def print_memory_stats():
    process = psutil.Process()
    print(f"RAM Memory Use: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id} Memory Use: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

def clear_gpu_memory():
    tf.keras.backend.clear_session()
    import gc
    gc.collect()

def print_detailed_memory():
    process = psutil.Process()
    
    # RAM details
    ram_info = process.memory_info()
    print(f"RAM Usage:")
    print(f"  RSS (Resident Set Size): {ram_info.rss / 1024 / 1024:.2f} MB")
    print(f"  VMS (Virtual Memory Size): {ram_info.vms / 1024 / 1024:.2f} MB")
    
    # GPU details
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"\nGPU {gpu.id} ({gpu.name}):")
        print(f"  Memory Use: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        print(f"  GPU Load: {gpu.load*100:.1f}%")
        print(f"  Memory Free: {gpu.memoryFree}MB")

def monitor_resources():
    """Prints CPU, RAM, and GPU usage stats."""
    # CPU & RAM Usage
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    ram_used = psutil.virtual_memory().used / (1024**3)  # Convert bytes to GB
    ram_total = psutil.virtual_memory().total / (1024**3)  # Convert bytes to GB

    print(f"🖥️ CPU Usage: {cpu_usage:.1f}% | 🏗️ RAM: {ram_used:.2f}/{ram_total:.2f} GB ({ram_usage:.1f}%)")

    # GPU Usage
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"🚀 GPU {gpu.id} ({gpu.name}): {gpu.memoryUsed:.1f}MB / {gpu.memoryTotal:.1f}MB "
              f"| Load: {gpu.load * 100:.1f}%")

def vram_cleanup_if_needed(threshold_mb=31000):
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        # Check if usage exceeds threshold
        if gpu.memoryUsed > threshold_mb:
            print(f"\n⚠️ VRAM usage is {gpu.memoryUsed}MB — clearing session to prevent OOM.")
            tf.keras.backend.clear_session()
            gc.collect()
            break

def log_vram_usage():
    """Simple utility to print VRAM usage for all GPUs."""
    for gpu in GPUtil.getGPUs():
        print(f"🔍 GPU {gpu.id} => {gpu.memoryUsed}/{gpu.memoryTotal} MB used")

# ----- Encoders -----
class SpectrogramEncoder(tf.keras.Model):
    """
    Adaptive encoder for spectrogram features that works with any input dimensions.
    """
    def __init__(self, latent_dim):
        super().__init__()
        
        # Target downsampled dimensions - will work for any input size
        self.target_freq = 32  # Can be adjusted
        self.target_time = 8   # Can be adjusted
        
        # Initial adaptive dimension adjustment
        self.adjust_conv = layers.Conv2D(32, 1, padding='same', activation='relu')
        
        # Convolutional layers
        self.conv1 = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        
        self.conv2 = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.3)
        
        self.conv3 = layers.Conv2D(128, 3, strides=1, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(0.3)
        
        # Global pooling and dense layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense_reduce = layers.Dense(512, activation='relu')
        
        # Latent parameter layers
        self.mu_layer = layers.Dense(latent_dim)
        self.logvar_layer = layers.Dense(latent_dim)
    
    def call(self, x, training=False):
        # Initial dimension adjustment
        x = self.adjust_conv(x)
        
        # Standard convolution path
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)
        
        # Pool and encode
        x = self.global_pool(x)
        x = self.dense_reduce(x)
        
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        
        return mu, logvar
    
class MaskEncoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        # Use 2D convolution layers to progressively downsample the mask
        self.conv1 = layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(128, activation='relu')
        self.mu_layer = layers.Dense(latent_dim)
        self.logvar_layer = layers.Dense(latent_dim)
    
    def call(self, x, training=False):
        # x shape: (batch, height, width, channels), e.g., (batch, 256, 768, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar
    
# ----- Decoders -----
class SpectrogramDecoder(tf.keras.Model):
    """
    Decoder for spectrogram features ensuring exact output dimensions.
    """
    def __init__(self, freq_bins=129, time_bins=24, channels=24):
        super().__init__()
        
        self.freq_bins = freq_bins
        self.time_bins = time_bins
        self.channels = channels
        
        # Determine minimum dimensions (ensure they're at least 1)
        self.min_freq = max(1, freq_bins // 4)
        self.min_time = max(1, time_bins // 4)
        
        # Dense and reshape layers
        self.fc = layers.Dense(self.min_freq * self.min_time * 128, activation='relu')
        self.reshape_layer = layers.Reshape((self.min_freq, self.min_time, 128))
        
        # Upsampling blocks - with explicit output_padding as needed
        # We need to carefully calculate output shapes to ensure we get exactly the dimensions we want
        self.conv_t1 = layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu'
        )
        self.bn1 = layers.BatchNormalization()
        self.drop1 = layers.Dropout(0.3)
        
        self.conv_t2 = layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu'
        )
        self.bn2 = layers.BatchNormalization()
        self.drop2 = layers.Dropout(0.3)
        
        # Final refinement layers
        self.conv_t3 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.drop3 = layers.Dropout(0.3)
        
        # Output projection 
        self.conv_out = layers.Conv2D(channels * 2, 3, padding='same', dtype='float32')

    def call(self, z, training=False):
        # Initial projection and reshape
        x = self.fc(z)
        x = self.reshape_layer(x)
        
        # First upsampling
        x = self.conv_t1(x)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)
        
        # Second upsampling
        x = self.conv_t2(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)
        
        # At this point x should be approximately the target size, but might not be exact
        
        # Final refinement (no dimension change)
        x = self.conv_t3(x)
        x = self.bn3(x, training=training)
        x = self.drop3(x, training=training)
        
        # Final projection
        x = self.conv_out(x)
        
        # Check and adjust dimensions if necessary
        # This is the key part that ensures exact output dimensions
        current_shape = tf.shape(x)
        
        # Use explicit reshape to get exactly the desired dimensions
        x = tf.reshape(x, [-1, current_shape[1], current_shape[2], self.channels * 2])
        
        # Now use resize operations to get exactly the dimensions we want
        if current_shape[1] != self.freq_bins or current_shape[2] != self.time_bins:
            # Create a new tensor with the exact desired shape
            resized = tf.image.resize(
                x, 
                [self.freq_bins, self.time_bins],
                method='bilinear'
            )
            # Make sure the resize operation preserves the batch and channel dimensions
            return tf.reshape(resized, [-1, self.freq_bins, self.time_bins, self.channels * 2])
        
        return x
       
class MaskDecoder(tf.keras.Model):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_shape = output_shape  # e.g., (256, 768, 1)
        # Calculate spatial dimensions after 3 downsampling steps (assuming factor of 2 each)
        self.down_height = output_shape[0] // 8
        self.down_width  = output_shape[1] // 8
        self.fc = layers.Dense(self.down_height * self.down_width * 128, activation='relu')
        self.reshape_layer = layers.Reshape((self.down_height, self.down_width, 128))
        self.conv_t1 = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')
        self.conv_t2 = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')
        self.conv_t3 = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')
        self.output_layer = layers.Conv2D(1, 3, padding='same', activation='sigmoid', dtype='float32')
    
    def call(self, z, training=False):
        x = self.fc(z)
        x = self.reshape_layer(x)
        x = self.conv_t1(x)
        x = self.conv_t2(x)
        x = self.conv_t3(x)
        x = self.output_layer(x)
        # x shape will be (batch, 256, 768, 1)
        return x
    
# ----- Loss Functions -----
@tf.function
def complex_spectrogram_loss(y_true, y_pred):
    """
    Custom loss for spectrogram reconstruction that separately handles magnitude and phase.
    Includes additional shape checking and debugging.
    
    Args:
        y_true: Ground truth spectrogram features [batch, freq, time, channels*2]
        y_pred: Predicted spectrogram features [batch, freq, time, channels*2]
        
    Returns:
        Combined loss with higher weight on magnitude
    """
    # Get total number of channels
    total_channels = tf.shape(y_true)[-1]
    
    # Separate magnitude and phase components
    # Create indices for even and odd positions
    mag_indices = tf.range(0, total_channels, delta=2)
    phase_indices = tf.range(1, total_channels, delta=2)
    
    # Gather the magnitude and phase components
    mag_true = tf.gather(y_true, mag_indices, axis=-1)
    mag_pred = tf.gather(y_pred, mag_indices, axis=-1)
    
    phase_true = tf.gather(y_true, phase_indices, axis=-1)
    phase_pred = tf.gather(y_pred, phase_indices, axis=-1)
    
    # MSE for magnitude (more important)
    mag_loss = tf.reduce_mean(tf.square(mag_true - mag_pred))
    
    # For phase, we need a circular distance metric
    # Convert to complex numbers on the unit circle
    phase_true_complex = tf.complex(
        tf.cos(phase_true), 
        tf.sin(phase_true)
    )
    phase_pred_complex = tf.complex(
        tf.cos(phase_pred), 
        tf.sin(phase_pred)
    )
    
    # Compute cosine of angle difference
    # z1·conj(z2) gives |z1|·|z2|·cos(θ1-θ2) + i·|z1|·|z2|·sin(θ1-θ2)
    # We just want the real part (cosine of angle difference)
    phase_diff_cos = tf.math.real(phase_true_complex * tf.math.conj(phase_pred_complex))
    
    # 1 - cos(diff) is a good metric for angular distance (0 when identical, 2 when opposite)
    phase_loss = tf.reduce_mean(1.0 - phase_diff_cos)
    
    # Combine losses with higher weight on magnitude
    return tf.cast(0.6 * mag_loss + 0.4 * phase_loss, tf.float32)

@tf.function
def custom_mask_loss(y_true, y_pred, 
                       weight_bce=0.2, 
                       weight_dice=0.3, 
                       weight_focal=0.5, 
                       gamma=2.0, # Focal loss focusing parameter
                       alpha=0.3 # Focal loss balancing parameter
                       ):
    """
    Computes a weighted combination of Binary Cross-Entropy (BCE), Dice, and Focal losses.
    
    Args:
        y_true (tf.Tensor): Ground truth mask with shape (batch, height, width, 1) and values in [0,1].
        y_pred (tf.Tensor): Predicted mask with shape (batch, height, width, 1) and values in [0,1].
        weight_bce (float): Weight for the BCE loss component.
        weight_dice (float): Weight for the Dice loss component.
        weight_focal (float): Weight for the Focal loss component.
        gamma (float): Focusing parameter for focal loss.
        alpha (float): Balancing parameter for focal loss.
        
    Returns:
        tf.Tensor: The combined loss scalar.
    """
    # Get a small constant for numerical stability.
    epsilon = tf.keras.backend.epsilon()
    
    # --- BCE Loss ---
    y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    bce = -(y_true * tf.math.log(y_pred_clipped) + (1.0 - y_true) * tf.math.log(1.0 - y_pred_clipped))
    bce_loss = tf.reduce_mean(bce)
    # Ensure the BCE loss is positive.
    bce_loss = tf.abs(bce_loss)
    
    # --- Dice Loss ---
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    y_pred_flat = tf.clip_by_value(y_pred_flat, epsilon, 1.0 - epsilon)
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    smooth = 1.0
    dice_coef = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_coef
    
    # --- Focal Loss ---
    y_pred_focal = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(tf.equal(y_true, 1), y_pred_focal, 1 - y_pred_focal)
    focal_weight = tf.pow(1.0 - pt, gamma)
    alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    focal_bce = -tf.math.log(pt)
    focal_loss = tf.reduce_mean(alpha_weight * focal_weight * focal_bce)
    
    # --- Combined Loss ---
    combined = weight_bce * bce_loss + weight_dice * dice_loss + weight_focal * focal_loss
    return tf.cast(combined, tf.float32)

def dynamic_weighting(epoch, max_epochs, min_weight=0.3, max_weight=0.5):
    """
    Gradually adjusts the weight between spectrogram and mask losses.
    
    Args:
        epoch: Current epoch
        max_epochs: Total epochs
        min_weight: Starting weight
        max_weight: Maximum weight
        
    Returns:
        Current weight
    """
    # Linear increase over time
    progress = min(1.0, epoch / (max_epochs * 0.2))  # Reach max_weight halfway through training
    return min_weight + progress * (max_weight - min_weight)

def get_beta_schedule(epoch, max_epochs, schedule_type='cyclical'):
    """
    Compute the beta coefficient for the JS divergence loss term.
    
    Args:
        epoch: Current epoch number (0-indexed)
        max_epochs: Total number of epochs
        schedule_type: Type of schedule ('linear', 'exponential', 'cyclical')
    
    Returns:
        beta: Beta coefficient for current epoch
    """
    # Define beta limits
    BETA_MIN = 1e-8   # Start with very small value
    BETA_MAX = 0.15   # Maximum value
    
    # Define warmup phase length (in epochs)
    WARMUP_EPOCHS = 60
    
    # Define schedule based on type
    if schedule_type == 'linear':
        # Linear ramp-up during warmup, then constant
        if epoch < WARMUP_EPOCHS:
            return BETA_MIN + (BETA_MAX - BETA_MIN) * (epoch / WARMUP_EPOCHS)
        else:
            return BETA_MAX
            
    elif schedule_type == 'exponential':
        # Exponential ramp-up (slower at start, faster later)
        if epoch < WARMUP_EPOCHS:
            # Normalized epoch in [0, 1]
            t = epoch / WARMUP_EPOCHS
            # Exponential curve that starts at BETA_MIN and ends at BETA_MAX
            return BETA_MIN + (BETA_MAX - BETA_MIN) * (np.exp(3 * t) - 1) / (np.exp(3) - 1)
        else:
            return BETA_MAX
            
    elif schedule_type == 'cyclical':
        # Cyclical schedule that peaks at BETA_MAX and falls to BETA_MAX/5
        if epoch < WARMUP_EPOCHS:
            # Linear warmup
            return BETA_MIN + (BETA_MAX - BETA_MIN) * (epoch / WARMUP_EPOCHS)
        else:
            # Cycle with period of 50 epochs
            cycle_period = 50
            cycle_position = ((epoch - WARMUP_EPOCHS) % cycle_period) / cycle_period
            # Cosine cycle between BETA_MAX and BETA_MAX/5
            beta_min_cycle = BETA_MAX / 5
            return beta_min_cycle + (BETA_MAX - beta_min_cycle) * 0.5 * (1 + np.cos(2 * np.pi * cycle_position))
    
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

# ----- Spectral MMVAE Model -----
class SpectralMMVAE(tf.keras.Model):
    """
    Spectral Multimodal VAE with Mixture-of-Experts prior.
    Implementation follows the "Unity by Diversity" paper approach.
    
    Modalities:
    1. Complex spectrograms (from time series)
    2. binary Mask
    """
    def __init__(self, latent_dim, spec_shape, mask_dim):
        super().__init__()
        
        # Store shapes and latent dimension
        self.latent_dim = latent_dim
        self.spec_shape = spec_shape
        self.mask_dim = mask_dim
        
        # Encoders
        self.spec_encoder = SpectrogramEncoder(latent_dim)
        self.mask_encoder = MaskEncoder(latent_dim)
        
        # Decoders
        self.spec_decoder = SpectrogramDecoder(
            freq_bins=spec_shape[0],
            time_bins=spec_shape[1],
            channels=spec_shape[2] // 2
        )
        self.mask_decoder = MaskDecoder(latent_dim, mask_dim)

    def call(self, spec_in, mask_in, test_id=None, training=False, missing_modality=None):
        """
        Forward pass for the Mixture-of-Experts MMVAE approach:
        1) Encode each modality to obtain μ (mu) and log-variance (logvar)
        2) Compute the Mixture-of-Experts (MoE) prior as a mixture of the unimodal posteriors
        3) Compute the JS divergence between the unimodal posteriors and the mixture prior
        4) Sample from each unimodal posterior (or from the mixture if a modality is missing)
        5) Decode each modality from its corresponding latent sample
        6) Return the reconstructed spectrogram and mask, along with the distribution parameters and JS divergence

        Args:
            spec_in: Input spectrogram.
            mask_in: Input binary mask (crack mask) with shape (height, width, channels).
            test_id: Optional test identifier.
            training: Boolean indicating whether the model is in training mode.
            missing_modality: Optional string indicating which modality is missing ('spec' or 'mask').

        Returns:
            A tuple of:
                - recon_spec: Reconstructed spectrogram.
                - recon_mask: Reconstructed binary mask.
                - (all_mus, all_logvars, mixture_prior, js_div): A tuple containing:
                    * all_mus: List of latent means.
                    * all_logvars: List of latent log-variances.
                    * mixture_prior: The computed MoE prior as a tuple (mixture_mu, mixture_logvar).
                    * js_div: The computed JS divergence loss term.
        """
        # Track available modalities
        available_modalities = []
        if missing_modality != 'spec':
            available_modalities.append('spec')
        if missing_modality != 'mask':
            available_modalities.append('mask')
        
        # 1) Encode available modalities
        mus = []
        logvars = []
        
        if 'spec' in available_modalities:
            mu_spec, logvar_spec = self.spec_encoder(spec_in, training=training)
            mus.append(mu_spec)
            logvars.append(logvar_spec)
        
        if 'mask' in available_modalities:
            mu_mask, logvar_mask = self.mask_encoder(mask_in, training=training)
            mus.append(mu_mask)
            logvars.append(logvar_mask)
        
        # 2) Compute MoE prior parameters
        mixture_mu, mixture_logvar = compute_mixture_prior(mus, logvars)
        
        # 3) Compute JS divergence
        js_div = compute_js_divergence(mus, logvars)
        
        # Store all distribution parameters
        all_mus = mus.copy()
        all_logvars = logvars.copy()
        
        # Handle missing modalities by imputing from the mixture
        if missing_modality == 'spec':
            # Impute spectrogram modality from the mixture
            z_spec = reparameterize(mixture_mu, mixture_logvar)
            # Add placeholders to keep indices consistent
            all_mus.insert(0, mixture_mu)
            all_logvars.insert(0, mixture_logvar)
        else:
            # Sample spectrogram latent from its posterior
            z_spec = reparameterize(mu_spec, logvar_spec)
        
        if missing_modality == 'mask':
            # Impute mask modality from the mixture
            z_mask = reparameterize(mixture_mu, mixture_logvar)
            # Add placeholders to keep indices consistent
            all_mus.append(mixture_mu)
            all_logvars.append(mixture_logvar)
        else:
            # Sample mask latent from its posterior
            z_mask = reparameterize(mu_mask, logvar_mask)
        
        # 4) Decode
        recon_spec = self.spec_decoder(z_spec, training=training)
        recon_mask = self.mask_decoder(z_mask, training=training)
        
        # 5) Return outputs
        mixture_prior = (mixture_mu, mixture_logvar)
        return recon_spec, recon_mask, (all_mus, all_logvars, mixture_prior, js_div)

    def generate(
        self, 
        modality='both', 
        conditioning_modality=None, 
        conditioning_input=None,
        conditioning_latent=None
        ):
        """
        Generate samples using the Mixture-of-Experts approach.
        
        Args:
            modality: Which modality to generate ('spec', 'mask', or 'both')
            conditioning_modality: Optional modality to condition on ('spec' or 'mask')
            conditioning_input: Input for the conditioning modality
            conditioning_latent: Optional latent vector to use directly
            
        Returns:
            If modality == 'both': returns a tuple (recon_spec, recon_mask)
            If modality == 'spec': returns recon_spec
            If modality == 'mask': returns recon_mask
        """
        # 1. Use the provided latent if available; otherwise, sample or encode.
        if conditioning_latent is not None:
            z = conditioning_latent
        else:
            if conditioning_modality is None:
                z = tf.random.normal(shape=(1, self.latent_dim))
            else:
                if conditioning_modality == 'spec':
                    mu, logvar = self.spec_encoder(conditioning_input)
                elif conditioning_modality == 'mask':
                    mu, logvar = self.mask_encoder(conditioning_input)
                else:
                    raise ValueError(f"Unknown conditioning modality: {conditioning_modality}")
                z = reparameterize(mu, logvar)

        # 2. Generate the requested modality.
        if modality == 'spec' or modality == 'both':
            recon_spec = self.spec_decoder(z)
        else:
            recon_spec = None

        if modality == 'mask' or modality == 'both':
            recon_mask = self.mask_decoder(z)
        else:
            recon_mask = None

        # 3. Return the generated output.
        if modality == 'both':
            return recon_spec, recon_mask
        elif modality == 'spec':
            return recon_spec
        elif modality == 'mask':
            return recon_mask

    def encode_all_modalities(self, spec_in, mask_in, training=False):
        """
        Encode all modalities and compute the mixture prior.
        
        Args:
            spec_in: Input spectrogram.
            mask_in: Input binary mask.
            training: Whether in training mode.
            
        Returns:
            Tuple of (mus, logvars, mixture_mu, mixture_logvar)
        """
        # Encode each modality
        mu_spec, logvar_spec = self.spec_encoder(spec_in, training=training)
        mu_mask, logvar_mask = self.mask_encoder(mask_in, training=training)
        
        # Compute mixture prior from the two modalities
        mus = [mu_spec, mu_mask]
        logvars = [logvar_spec, logvar_mask]
        mixture_mu, mixture_logvar = compute_mixture_prior(mus, logvars)
        
        return mus, logvars, mixture_mu, mixture_logvar

    def reconstruct_time_series(
        self,
        spec_features,
        fs=200,
        nperseg=256,
        noverlap=224,
        time_length=800,
        batch_processing_size=100,
        ):
        """
        Wrapper around data_loader.inverse_spectrogram.
        The `spec_features` tensor coming out of the MM‑VAE already has the
        expected shape (batch, F, T, 2*C) with log‑magnitude & phase stacked
        along the last axis, so we can call the helper directly.
        """
        # data_loader version is implemented in PyTorch but accepts NumPy just fine.
        return dl_inverse_spectrogram(
            spec_features,           # log‑mag & phase
            time_length=time_length,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            batch_processing_size=batch_processing_size,
        )
   
# ----- Data Processing and Dataset Creation -----
def create_tf_dataset(
    spectrograms, mask_array, test_id_array,
    batch_size=32, shuffle=True, debug_mode=False, debug_samples=500
    ):
    """
    Create a TensorFlow Dataset that loads data from memory.
    
    Args:
        spectrograms: Array or path to file containing spectrograms.
        mask_array: Array or path to file containing binary masks.
        test_id_array: Array of test IDs.
        batch_size: Batch size for the dataset.
        shuffle: Whether to shuffle the dataset.
        debug_mode: If True, only a limited number of samples are used.
        debug_samples: Number of samples to use in debug mode.
        
    Returns:
        A tf.data.Dataset yielding tuples (spectrogram, mask, test_id).
    """
    # Convert file path to memory-mapped array if a string is provided
    if isinstance(spectrograms, str):
        spectrograms = np.load(spectrograms, mmap_mode='r')
    if isinstance(mask_array, str):
        mask_array = np.load(mask_array, mmap_mode='r')
    
    # Apply debug mode limit
    if debug_mode:
        print(f"⚠️ Debug Mode ON: Using only {debug_samples} samples for quick testing!")
        spectrograms = spectrograms[:debug_samples]
        mask_array = mask_array[:debug_samples]
        test_id_array = test_id_array[:debug_samples]
    else:
        print(f"✅ Full dataset loaded: {len(mask_array)} samples.")

    # Ensure inputs have at least rank 1
    test_id_array = np.atleast_1d(test_id_array)
    mask_array = np.atleast_1d(mask_array)
    
    dataset = tf.data.Dataset.from_tensor_slices((spectrograms, mask_array, test_id_array))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(mask_array))

    dataset = (
        dataset
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset

# ---------- Training functions ----------
def train_step(model, optimizer,
               spec_mb, mask_mb, test_id_mb,
               beta, spec_weight, mask_weight,
               modality_dropout_prob):
    """
    One replica‑local training step.

    ‑  The dropout decision is a TensorFlow op, executed identically
       on every replica, so collective ops never disagree on tensor shape.
    ‑  We return spec_coeff / mask_coeff so the outer loop can still
       keep separate running averages.
    """
    with tf.GradientTape() as tape:
        # forward pass (nothing is really “missing”)
        recon_spec, recon_mask, (_, _, _, js_div) = model(
            spec_mb, mask_mb, test_id_mb,
            training=True, missing_modality=None)

        spec_loss = complex_spectrogram_loss(spec_mb, recon_spec)
        mask_loss = custom_mask_loss(mask_mb, recon_mask)

        # ---------- in‑graph modality dropout ----------
        rv         = tf.random.uniform([], 0., 1.)
        drop_spec  = rv <  modality_dropout_prob
        drop_mask  = (rv >= modality_dropout_prob) & \
                     (rv <  2. * modality_dropout_prob)

        spec_coeff = tf.cast(tf.logical_not(drop_spec), tf.float32)   # 1 or 0
        mask_coeff = tf.cast(tf.logical_not(drop_mask), tf.float32)   # 1 or 0
        # ----------------------------------------------

        recon_loss = (spec_weight * spec_loss * spec_coeff +
                      mask_weight * mask_loss * mask_coeff)

        total_loss = recon_loss + beta * js_div
        scaled_loss = optimizer.get_scaled_loss(total_loss)

    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads        = optimizer.get_unscaled_gradients(scaled_grads)

    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # also return the coeffs so the outer loop knows what was active
    return (total_loss, spec_loss, mask_loss, js_div,
            recon_loss, spec_coeff, mask_coeff)

def train_spectral_mmvae(
    model,
    train_dataset,
    val_dataset,
    optimizer,
    num_epochs        = 100,
    patience          = 10,
    beta_schedule     = "cyclical",
    modality_dropout_prob = 0.10,
    strategy          = None,
    ):
    """
    Multi–GPU / MirroredStrategy training loop with in‑graph modality dropout.
    Everything is replica‑safe: we only do reductions via `strategy.reduce`.
    """

    # -------------------------------------------------------------
    # Localized training call wrappers (redefine per chunk)
    # -------------------------------------------------------------
    def step_fn(spec_mb, mask_mb, test_id_mb, beta, spec_weight, mask_weight):
        return train_step(
            model, optimizer,
            spec_mb, mask_mb, test_id_mb,
            beta, spec_weight, mask_weight,
            modality_dropout_prob
        )

    @tf.function(reduce_retracing=True)
    def distributed_train_step(spec_mb, mask_mb, test_id_mb, beta, spec_weight, mask_weight):
        return strategy.run(
            step_fn,
            args=(spec_mb, mask_mb, test_id_mb, beta, spec_weight, mask_weight)
        )
    # ------------------------------------------------------------------ #
    metrics = {k: [] for k in
               ("train_total", "train_spec", "train_mask", "train_js",
                "train_mode",  "val_total", "val_spec",  "val_mask", "val_js")}

    best_val_loss       = float("inf")
    no_improvement_cnt  = 0

    train_batches = sum(1 for _ in train_dataset)
    val_batches   = sum(1 for _ in val_dataset)
    print(f"🔄 Starting Training: {train_batches} train batches, {val_batches} val batches")


    # ------------------------------------------------------------------ #
    for epoch in range(num_epochs):
        # ------------  logging / schedules  ---------------------------- #
        print(f"\n🔍 VRAM usage at start of epoch {epoch + 1}:")
        log_vram_usage()

        beta        = get_beta_schedule(epoch, num_epochs, beta_schedule)
        mask_weight = dynamic_weighting(epoch, num_epochs)
        spec_weight = 1.0 - mask_weight
        print(f"📌 Epoch {epoch + 1}/{num_epochs} | Beta={beta:.5f} | MaskW={mask_weight:.02f}")

        # ---- per‑epoch accumulators ----------------------------------- #
        acc = dict(train_total=0.0, train_spec=0.0, train_mask=0.0, train_js=0.0,
                   train_full=0.0, train_spec_only=0.0, train_mask_only=0.0,
                   n_full=0, n_spec_only=0, n_mask_only=0, train_steps=0)

        # =======================  TRAIN LOOP  ========================== #
        for step, (spec_in, mask_in, test_id_in) in enumerate(train_dataset):

            (tot, spec_l, mask_l, js_d,
                recon_l, spec_c, mask_c) = distributed_train_step(
                        spec_in, mask_in, test_id_in,
                        tf.constant(beta, tf.float32),
                        spec_weight, mask_weight)

            red = lambda x: strategy.reduce(tf.distribute.ReduceOp.MEAN, x, axis=None)
            tot, spec_l, mask_l, js_d, recon_l = map(red, (tot, spec_l, mask_l, js_d, recon_l))
            spec_c, mask_c = red(spec_c), red(mask_c)

            # ---- update epoch counters -------------------------------- #
            acc["train_total"] += float(tot.numpy())
            acc["train_spec"]  += float((spec_l * spec_c).numpy())
            acc["train_mask"]  += float((mask_l * mask_c).numpy())
            acc["train_js"]    += float(js_d.numpy())
            acc["train_steps"] += 1

            spec_c_val = float(spec_c.numpy())
            mask_c_val = float(mask_c.numpy())

            if spec_c_val == 1.0 and mask_c_val == 1.0:             # both kept
                acc["n_full"]     += 1
                acc["train_full"] += float(recon_l.numpy())
            elif spec_c_val == 0.0:                                 # spec dropped
                acc["n_mask_only"]     += 1
                acc["train_mask_only"] += float(recon_l.numpy())
            else:                                                   # mask dropped
                acc["n_spec_only"]     += 1
                acc["train_spec_only"] += float(recon_l.numpy())

        # --------------  epoch‑level averages & logs  ------------------ #
        if acc["train_steps"] > 0:
            metrics["train_total"].append(acc["train_total"] / acc["train_steps"])
            metrics["train_spec"].append(
                acc["train_spec"] / max(acc["train_steps"] - acc["n_mask_only"], 1))
            metrics["train_mask"].append(
                acc["train_mask"] / max(acc["train_steps"] - acc["n_spec_only"], 1))
            metrics["train_js"].append(acc["train_js"] / acc["train_steps"])
            metrics["train_mode"].append(dict(
                full      = acc["train_full"]      / max(acc["n_full"],       1),
                spec_only = acc["train_spec_only"] / max(acc["n_spec_only"],  1),
                mask_only = acc["train_mask_only"] / max(acc["n_mask_only"],  1),
            ))

            print(f"✅ [Train] Loss={metrics['train_total'][-1]:.4f} | "
                  f"Spec={metrics['train_spec'][-1]:.4f} | "
                  f"Mask={metrics['train_mask'][-1]:.4f} | "
                  f"JS={metrics['train_js'][-1]:.4f}")

        # =======================  VAL LOOP  ============================ #
        val_stats = dict(total=0., spec=0., mask=0., js=0., steps=0)
        for spec_in, mask_in, test_id_in in val_dataset:
            recon_spec, recon_mask, (_, _, _, js_div) = model(
                spec_in, mask_in, test_id_in, training=False, missing_modality=None)

            spec_l  = complex_spectrogram_loss(spec_in, recon_spec)
            mask_l  = custom_mask_loss(mask_in, recon_mask)
            recon_l = spec_weight * spec_l + mask_weight * mask_l
            tot_l   = recon_l + tf.constant(beta, tf.float32) * js_div

            for k, v in zip(("total", "spec", "mask", "js"),
                             (tot_l, spec_l, mask_l, js_div)):
                val_stats[k] += float(v.numpy())
            val_stats["steps"] += 1

        if val_stats["steps"] > 0:
            metrics["val_total"].append(val_stats["total"] / val_stats["steps"])
            metrics["val_spec"].append(val_stats["spec"]  / val_stats["steps"])
            metrics["val_mask"].append(val_stats["mask"]  / val_stats["steps"])
            metrics["val_js"].append(  val_stats["js"]    / val_stats["steps"])

            print(f"  🔵 [Val] => Total={metrics['val_total'][-1]:.4f} | "
                  f"Spec={metrics['val_spec'][-1]:.4f} | "
                  f"Mask={metrics['val_mask'][-1]:.4f} | "
                  f"JS={metrics['val_js'][-1]:.4f}")

        # ====================  EARLY‑STOPPING  ========================= #
        current_val = metrics["val_total"][-1] if val_stats["steps"] else float("inf")
        if current_val < best_val_loss:
            best_val_loss      = current_val
            no_improvement_cnt = 0
            model.save_weights("../results_mmvae/best_spectral_mmvae.weights.h5")
            print("✅ Saved best weights")
        else:
            no_improvement_cnt += 1
            print(f"🚨 No improvement for {no_improvement_cnt}/{patience}")

        if no_improvement_cnt >= patience:
            print(f"🛑 Early stopping at epoch {epoch + 1}.")
            model.save_weights("../results_mmvae/final_spectral_mmvae.weights.h5")
            break

    return metrics

# ----- Visualization Functions and tests -----
def save_visualizations_and_metrics(model, train_dataset, val_dataset, training_metrics, output_dir="results_mmvae"):
    """
    Aggregates and saves key graphs and model statistics:
      1. Training curves (train/val loss)
      2. 3D latent space visualization using UMAP (from latent representations on the train set)
      3. Latent analysis (cosine similarity and Euclidean distance histograms using validation data)
      4. Latent space interpolation between two validation samples
      5. Model weight statistics

    Args:
        model (tf.keras.Model): Your trained SpectralMMVAE model.
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
        training_metrics (dict): Dictionary containing training and validation loss curves.
        output_dir (str): Directory where the visualizations and stats will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Plot Training Curves
    def plot_training_curves(metrics):
        epochs = list(range(1, len(metrics['train_total']) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=metrics['train_total'],
                                 mode='lines+markers', name="Train Total", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=epochs, y=metrics['val_total'],
                                 mode='lines+markers', name="Val Total", line=dict(color='red')))
        fig.update_layout(title="Total Loss vs Epochs",
                          xaxis_title="Epoch",
                          yaxis_title="Loss",
                          template="plotly_white")
        file_path = os.path.join(plots_dir, "train_val_total_loss.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved training curves to {file_path}")
    
    plot_training_curves(training_metrics)

    # 2. 3D Latent Space Visualization
    def extract_and_reduce_latents(dataset):
        # Extract latent vectors from the spectrogram encoder only
        latent_vectors = []
        test_ids = []
        for spec_in, _, test_id_in in dataset:
            # Use spectrogram encoder to get latent means.
            mu, _ = model.spec_encoder(spec_in, training=False)
            latent_vectors.append(mu.numpy())
            # Ensure test IDs are flattened to a list
            if isinstance(test_id_in, tf.Tensor):
                test_ids.append(test_id_in.numpy().flatten())
            else:
                test_ids.append(np.array(test_id_in).flatten())
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        test_ids = np.concatenate(test_ids, axis=0)
        # Reduce dimensionality using UMAP.
        reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=100)
        latent_3d = reducer.fit_transform(latent_vectors.reshape(latent_vectors.shape[0], -1))
        return latent_3d, test_ids

    latent_3d, train_test_ids = extract_and_reduce_latents(train_dataset)
    def plot_latent_space_3d(latent_3d, test_ids):
        df = pd.DataFrame(latent_3d, columns=["UMAP_1", "UMAP_2", "UMAP_3"])
        df["Test ID"] = pd.to_numeric(test_ids, errors="coerce")
        fig = px.scatter_3d(df, x="UMAP_1", y="UMAP_2", z="UMAP_3",
                             color="Test ID", color_continuous_scale="Viridis",
                             title="Latent Space Visualization (3D UMAP)", opacity=0.8)
        file_path = os.path.join(plots_dir, "latent_space_3d.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved 3D latent space plot to {file_path}")
    plot_latent_space_3d(latent_3d, train_test_ids)

    # 3. Latent Analysis: Compute and save cosine similarity and Euclidean distance histograms using validation data.
    def latent_analysis(dataset):
        latent_vectors = []
        for spec_in, _, _ in dataset:
            mu, _ = model.spec_encoder(spec_in, training=False)
            latent_vectors.append(mu.numpy())
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        # For simplicity, compare each latent vector with itself (this is a proxy – in practice, you might compare across modalities)
        norms = np.linalg.norm(latent_vectors, axis=1, keepdims=True)
        normalized = latent_vectors / (norms + 1e-8)
        cosine_similarities = np.dot(normalized, normalized.T).diagonal()
        euclidean_distances = np.linalg.norm(latent_vectors - latent_vectors, axis=1)  # Will be zeros, so for demo we use a dummy.
        # For demonstration, we create histograms for cosine similarities.
        fig = go.Figure(data=go.Histogram(x=cosine_similarities, histnorm='probability density', marker_color='blue', opacity=0.7))
        fig.update_layout(title="Cosine Similarity Distribution (Validation Latents)",
                          xaxis_title="Cosine Similarity", yaxis_title="Probability Density", template="plotly_white")
        file_path = os.path.join(plots_dir, "cosine_similarity_hist.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved cosine similarity histogram to {file_path}")
        # (Similarly, you can plot other metrics if needed.)
        return {"avg_cosine_similarity": float(np.mean(cosine_similarities))}
    
    latent_metrics = latent_analysis(val_dataset)
    print("Latent analysis metrics:", latent_metrics)

    # 4. Latent Interpolation
    def latent_interpolation(dataset):
        for spec_batch, mask_batch, _ in dataset.take(1):
            if spec_batch.shape[0] < 2:
                print("Need at least 2 samples for interpolation")
                return
            source_spec = tf.expand_dims(spec_batch[0], 0)
            target_spec = tf.expand_dims(spec_batch[1], 0)
            mu_source, _ = model.spec_encoder(source_spec, training=False)
            mu_target, _ = model.spec_encoder(target_spec, training=False)
            source_z = mu_source
            target_z = mu_target
            num_steps = 8
            alphas = np.linspace(0, 1, num_steps)
            plt.figure(figsize=(num_steps * 2, 4))
            for i, alpha in enumerate(alphas):
                interp_z = (1 - alpha) * source_z + alpha * target_z
                interp_spec = model.spec_decoder(interp_z, training=False)
                plt.subplot(1, num_steps, i+1)
                plt.imshow(interp_spec[0, :, :, 0].numpy(), aspect='auto', cmap='viridis')
                plt.title(f"α={alpha:.1f}")
                plt.axis('off')
            plt.tight_layout()
            interp_path = os.path.join(output_dir, "latent_interpolation.png")
            plt.savefig(interp_path, dpi=300)
            plt.close()
            print(f"Saved latent interpolation plot to {interp_path}")
            break
    latent_interpolation(val_dataset)

    # 5. Save model weight statistics.
    def save_model_weights_stats():
        stats = []
        for i, layer in enumerate(model.layers):
            for w in layer.weights:
                w_np = w.numpy()
                stats.append(f"Layer {i} ({layer.name}): min={np.min(w_np):.4f}, max={np.max(w_np):.4f}, mean={np.mean(w_np):.4f}, std={np.std(w_np):.4f}")
        stats_str = "\n".join(stats)
        stats_path = os.path.join(output_dir, "model_weight_stats.txt")
        with open(stats_path, "w") as f:
            f.write(stats_str)
        print(f"Saved model weight statistics to {stats_path}")
    save_model_weights_stats()

    # Optionally, you can return all gathered metrics:
    return {
        "latent_metrics": latent_metrics,
        "latent_space_3d": latent_3d,
    }

# ----- Main Function -----
def main():
    # ------------------------------------------------------------------
    # 0)  Misc flags & folders
    # ------------------------------------------------------------------
    debug_mode     = False
    latent_dim     = 128
    batch_size     = 128
    total_epochs   = 600
    patience       = 100
    resume_training = False

    segment_duration = 4.0
    nperseg        = 256
    noverlap       = 224
    sample_rate    = 200


    # output folders ---------------------------------------------------
    os.makedirs("../results_mmvae",               exist_ok=True)
    os.makedirs("../results_mmvae/model_checkpoints", exist_ok=True)
    os.makedirs("../results_mmvae/latent_analysis",   exist_ok=True)
    os.makedirs("../results_mmvae/cross_modal",       exist_ok=True)

    # Cache paths 
    tag = f"{segment_duration:.2f}s_{nperseg}_{noverlap}"
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    final_path = os.path.join(cache_dir, f"specs_{tag}.npy")
    heatmaps_path = os.path.join(cache_dir, f"masks_{tag}.npy")  # you'd need to save this manually in the loader
    ids_path = os.path.join(cache_dir, f"segIDs_{tag}.npy")

    # ------------------------------------------------------------------
    # 1)  LOAD PROCESSED FEATURES
    # ------------------------------------------------------------------
    all_cached = (os.path.exists(final_path) and
                os.path.exists(heatmaps_path) and
                os.path.exists(ids_path))

    if all_cached:
        print("✅  Loading cached NumPy arrays …")
        spectral_features = np.load(final_path,    mmap_mode="r")
        mask_segments     = np.load(heatmaps_path, mmap_mode="r")
        test_ids          = np.load(ids_path,      mmap_mode="r")

    else:
        print("⚠️  Cache missing – computing everything from raw data …")
        # ------------- call the unified loader -------------
        (_,       # accel_dict (not needed here)
        _,       # binary_masks
        heatmaps,
        _,
        spectrograms,      # (N, F, T, 2C)         – log‑mag | phase
        test_ids
        ) = data_loader.load_data(
                segment_duration = segment_duration,
                nperseg          = nperseg,
                noverlap         = noverlap,
                sample_rate      = sample_rate,
                recompute        = False,
                cache_dir        = cache_dir 
            )

        # the loader already returns spectrograms in (N, F, T, 2C)
        spectral_features = spectrograms
        mask_segments     = np.stack([heatmaps[tid] for tid in test_ids], axis=0)

        # ---- write them to the “global” cache for next run -------------
        np.save(final_path,    spectral_features)
        np.save(heatmaps_path, mask_segments)
        np.save(ids_path,      test_ids)
        print("✅  Written new cache files.")

    print(f"Spectrograms : {spectral_features.shape}")
    print(f"Masks        : {mask_segments.shape}")
    print(f"Test‑IDs     : {test_ids.shape}")


    # ------------------------------------------------------------------
    # 2)  TRAIN / VAL split
    # ------------------------------------------------------------------
    N           = spectral_features.shape[0]
    perm        = np.random.permutation(N)
    train_size  = int(0.8 * N)
    train_idx   = perm[:train_size]
    val_idx     = perm[train_size:]

    train_spec  = spectral_features[train_idx]
    val_spec    = spectral_features[val_idx]
    train_mask  = mask_segments[train_idx]
    val_mask    = mask_segments[val_idx]
    train_ids   = test_ids[train_idx]
    val_ids     = test_ids[val_idx]

    print(f"Train set : {train_spec.shape[0]}  |  Val set : {val_spec.shape[0]}")

    # ------------------------------------------------------------------
    # 3)  Build tf.data Datasets
    # ------------------------------------------------------------------
    train_dataset = create_tf_dataset(train_spec, train_mask, train_ids,
                                      batch_size, debug_mode=debug_mode)
    val_dataset   = create_tf_dataset(val_spec,   val_mask,  val_ids,
                                      batch_size, debug_mode=debug_mode)

    print(f"✅  Train batches: {sum(1 for _ in train_dataset)}")
    print(f"✅  Val   batches: {sum(1 for _ in val_dataset)}")

    # ------------------------------------------------------------------
    # 4)  Training‑loop in chunks (unchanged)
    # ------------------------------------------------------------------
    best_weights_path  = "results_mmvae/best_spectral_mmvae.weights.h5"
    final_weights_path = "results_mmvae/final_spectral_mmvae.weights.h5"

    metrics_path = "results_mmvae/training_metrics.npy"
    if resume_training and os.path.exists(metrics_path):
        all_metrics = np.load(metrics_path, allow_pickle=True).item()
        print("✅  Loaded previous metrics for resuming.")
    else:
        all_metrics = {
            'train_total': [], 'train_spec': [], 'train_mask': [], 'train_js': [],
            'train_mode':  [], 'val_total':  [], 'val_spec':  [], 'val_mask':  [], 'val_js': []
        }

    CHUNK_SIZE    = 100
    start_epoch = len(all_metrics['train_total'])
    early_stopped = False

    # while start_epoch < total_epochs and not early_stopped:
    #     end_epoch        = min(start_epoch + CHUNK_SIZE, total_epochs)
    #     epochs_this_run  = end_epoch - start_epoch
    #     print(f"\n===  Chunk {start_epoch+1} → {end_epoch}  ===")


    # Indent here for chunked training
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        spec_shape = spectral_features.shape[1:]          # (F, T, 2C)
        mask_shape = (32, 96, 1)

        model = SpectralMMVAE(latent_dim, spec_shape, mask_shape)

        # build graph
        dummy_spec = tf.zeros((1, *spec_shape))
        dummy_mask = tf.zeros((1, *mask_shape))
        _ = model(dummy_spec, dummy_mask, training=True)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=5e-5,
            decay_steps=10_000,
            decay_rate=0.9,
            staircase=True)

        base_optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6)
        optimizer      = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)

        if start_epoch > 0:
            if os.path.exists(best_weights_path):
                model.load_weights(best_weights_path)
                print(f"✅ Loaded best weights at chunk start (epoch {start_epoch}).")
            elif os.path.exists(final_weights_path):
                model.load_weights(final_weights_path)
                print(f"✅ Loaded final weights at chunk start (epoch {start_epoch}).")
            else:
                print(f"🚨 WARNING: No weights file found to resume from at epoch {start_epoch}!")
        else:
            print("✨ Starting training from scratch (initial chunk).")

        chunk_metrics = train_spectral_mmvae(
            model,
            train_dataset,
            val_dataset,
            optimizer,
            num_epochs           = total_epochs, # change to epochs_this_run for chunked training
            patience             = patience,
            beta_schedule        = 'cyclical',
            modality_dropout_prob= 0.10,
            strategy             = strategy
        )

        for k in all_metrics:
            all_metrics[k].extend(chunk_metrics[k])

        # Save weights and metrics
        model.save_weights(final_weights_path)
        np.save("results_mmvae/training_metrics.npy", all_metrics)

    # --- CLEAN-UP: free VRAM & graph ---------------------------------
    # del model                                    # big tensors
    tf.keras.backend.clear_session()
    gc.collect()

    # # optional: close Python log/file handles on Windows
    # import logging
    # for h in logging.root.handlers:
    #     h.flush(); h.close()
    # logging.root.handlers.clear()
    # # -----------------------------------------------------------------

        # start_epoch = end_epoch

    print("\n✅  Training finished.")

    # ------------------------------------------------------------------
    # 5)  Save metrics & visualisations
    # ------------------------------------------------------------------
    np.save("results_mmvae/training_metrics.npy", all_metrics)

    vis_metrics = save_visualizations_and_metrics(
        model,
        train_dataset,
        val_dataset,
        all_metrics,
        output_dir="results_mmvae"
    )

    print("🎉  Everything done – results in  'results_mmvae/'")



if __name__ == "__main__":
    main()
