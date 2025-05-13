import os
import sys
import gc
import psutil
import GPUtil
import numpy as np
import random
import cv2

import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.optimizers import AdamW # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts # type: ignore
from tensorflow.keras import mixed_precision  # type: ignore
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras import regularizers # type: ignore

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

# Multithreading
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

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
from losses import (
    complex_spectrogram_loss,
    custom_mask_loss,
    multi_channel_mrstft_loss,
    dynamic_weighting,
    get_beta_schedule,
    get_time_weight,
    gradient_loss_phase_only,
    laplacian_loss_phase_only,
    waveform_l1_loss,
    waveform_si_l1_loss,
    magnitude_l1_loss,
)

from data_loader import (
    TFInverseISTFT,
    augment_fn
)

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

# ------ Net Utils ------
def GN(groups=8):
    """Factory that returns a GroupNorm layer with L2 on gamma."""
    return GroupNormalization(
        groups=groups,
        axis=-1,                          # channels_last
        gamma_regularizer=regularizers.l2(1e-4)
    )

@tf.keras.saving.register_keras_serializable('mmvae')
class ResidualBlock(tf.keras.layers.Layer):
    def get_config(self):
        return {"filters": self.conv1.filters}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def __init__(self, filters):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, 3, padding="same", activation='relu')
        self.gn1 = GN()
        self.conv2 = layers.Conv2D(filters, 3, padding="same")
        self.gn2 = GN()

    def call(self, x, training=False):
        shortcut = x
        x = self.conv1(x)
        x = self.gn1(x, training=training)
        x = self.conv2(x)
        x = self.gn2(x, training=training)
        x += shortcut
        return tf.nn.relu(x)  

# ----- Spectrogram AE -----
@tf.keras.saving.register_keras_serializable('mmvae')
class SpectrogramEncoder(tf.keras.Model):
    def get_config(self):
        return {"latent_dim": self.mu.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')
        self.gn1 = GN()
        self.res1 = ResidualBlock(32)

        self.conv2 = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')
        self.gn2 = GN()
        self.res2 = ResidualBlock(64)

        self.conv3 = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')
        self.gn3 = GN()
        self.res3 = ResidualBlock(128)

        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(256, activation='relu')

        self.mu = layers.Dense(latent_dim)
        self.logvar = layers.Dense(latent_dim)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.gn1(x, training=training)
        x = self.res1(x, training=training)

        x = self.conv2(x)
        x = self.gn2(x, training=training)
        x = self.res2(x, training=training)

        x = self.conv3(x)
        x = self.gn3(x, training=training)
        x = self.res3(x, training=training)

        x = self.global_pool(x)
        x = self.dense(x)

        return self.mu(x), self.logvar(x)

@tf.keras.saving.register_keras_serializable('mmvae')
class SpectrogramDecoder(tf.keras.Model):
    def get_config(self):
        return {
            "freq_bins": self.freq_bins,
            "time_bins": self.time_bins,
            "channels": self.channels
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def __init__(self, freq_bins=129, time_bins=24, channels=24):
        super().__init__()
        self.freq_bins = freq_bins
        self.time_bins = time_bins
        self.channels = channels

        self.fc = layers.Dense((freq_bins // 8) * (time_bins // 8) * 128, activation='relu')
        self.reshape = layers.Reshape((freq_bins // 8, time_bins // 8, 128))

        self.deconv1 = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')
        self.gn1 = GN()
        self.res1 = ResidualBlock(64)

        self.deconv2 = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')
        self.gn2 = GN()
        self.res2 = ResidualBlock(32)

        self.deconv3 = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')
        self.gn3 = GN()
        self.res3 = ResidualBlock(32)

        self.out_conv = layers.Conv2D(channels * 2, 3, padding='same', dtype='float32')

    def call(self, z, training=False):
        x = self.fc(z)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.gn1(x, training=training)
        x = self.res1(x, training=training)

        x = self.deconv2(x)
        x = self.gn2(x, training=training)
        x = self.res2(x, training=training)

        x = self.deconv3(x)
        x = self.gn3(x, training=training)
        x = self.res3(x, training=training)

        x = self.out_conv(x)

        current_shape = tf.shape(x)
        if current_shape[1] != self.freq_bins or current_shape[2] != self.time_bins:
            x = tf.image.resize(x, [self.freq_bins, self.time_bins], method='bilinear')
            x = tf.reshape(x, [-1, self.freq_bins, self.time_bins, self.channels * 2])

        return x

# ----- Mask AE -----
@tf.keras.saving.register_keras_serializable('mmvae')
class MaskEncoder(tf.keras.Model):
    def get_config(self):
        return {"latent_dim": self.mu_layer.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(128, activation='relu')
        self.mu_layer = layers.Dense(latent_dim)
        self.logvar_layer = layers.Dense(latent_dim)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

@tf.keras.saving.register_keras_serializable('mmvae')
class MaskDecoder(tf.keras.Model):
    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "output_shape": self.out_shape
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_shape = output_shape
        self.down_height = output_shape[0] // 8
        self.down_width = output_shape[1] // 8
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
        return x     

# ----- Spectral MMVAE Model -----
@tf.keras.saving.register_keras_serializable('mmvae')
class SpectralMMVAE(tf.keras.Model):
    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "spec_shape": self.spec_shape,
            "mask_dim": self.mask_dim,
            "nperseg": self.istft_layer.frame_length,
            "noverlap": self.istft_layer.frame_length - self.istft_layer.frame_step
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    """
    Spectral Multimodal VAE with Mixture-of-Experts prior.
    Implementation follows the "Unity by Diversity" paper approach.
    
    Modalities:
    1. Complex spectrograms (from time series)
    2. binary Mask
    """
    def __init__(self, latent_dim, spec_shape, mask_dim, nperseg, noverlap):
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

        # Inverse STFT layer for time series reconstruction from data_loader.py
        self.istft_layer = TFInverseISTFT(
            frame_length=nperseg,
            frame_step=nperseg - noverlap,
            name = "tf_inverse_istft" 
        )
        self.istft_layer.trainable = True

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
    
    def build(self, input_shape=None):
        """Avoid calling base build method, which triggers shape-based auto-build."""
        self.built = True
  
# ----- Data Processing and Dataset Creation -----
def create_tf_dataset(
    spectrograms, mask_array, test_id_array, segments,
    batch_size=32, shuffle=True, debug_mode=False, debug_samples=500, augment=True
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
    if isinstance(segments, str):
        segments = np.load(segments, mmap_mode='r')
    
    # Apply debug mode limit
    if debug_mode:
        print(f"⚠️ Debug Mode ON: Using only {debug_samples} samples for quick testing!")
        spectrograms = spectrograms[:debug_samples]
        mask_array = mask_array[:debug_samples]
        test_id_array = test_id_array[:debug_samples]
        segments = segments[:debug_samples]
    else:
        print(f"✅ Full dataset loaded: {len(mask_array)} samples.")
    
    dataset = tf.data.Dataset.from_tensor_slices((spectrograms, mask_array, test_id_array, segments))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(mask_array))

    if augment:
        dataset = dataset.map(augment_fn,
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = (dataset
               .batch(batch_size, drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE))
    return dataset

# ---------- Training functions ----------
ACC_STEPS = 4  # accumulate gradients over 4 mini-batches
grad_accum = None
step_in_acc = None

def train_step(model, optimizer,
               spec_mb, mask_mb, test_id_mb, wave_mb,
               beta, spec_weight, mask_weight,
               modality_dropout_prob, time_weight, epoch,
               unfreeze_epoch=20):
    """One training step **per replica**.

    * Computes all losses.
    * Performs **gradient accumulation** over ``ACC_STEPS`` mini‑batches.
    * Applies the optimiser _inside_ the replica context so that
      `LossScaleOptimizer.aggregate_gradients()` can call `merge_call()`
      safely.
    """
    global grad_accum, step_in_acc

    # ---------------------------------------------------------------------
    # Forward + losses
    # ---------------------------------------------------------------------
    with tf.GradientTape() as tape:
        recon_spec, recon_mask, (_, _, _, js_div) = model(
            spec_mb, mask_mb, test_id_mb,
            training=True, missing_modality=None)

        # -------------- time‑domain reconstruction -----------------------
        time_len   = tf.shape(wave_mb)[1]
        recon_wave = model.istft_layer(recon_spec, time_len)
        wave_mb_f  = tf.cast(wave_mb, tf.float32)

        alpha    = tf.minimum(1.0, tf.cast(epoch, tf.float32) / 30.0)
        L_time   = (1 - alpha) * waveform_l1_loss(wave_mb_f, recon_wave) + \
                    alpha       * waveform_si_l1_loss(wave_mb_f, recon_wave)
        L_mrstft = multi_channel_mrstft_loss(wave_mb_f, recon_wave)

        # -------------- spectral‑only losses ----------------------------
        grad_loss = gradient_loss_phase_only(spec_mb, recon_spec)
        lap_loss  = laplacian_loss_phase_only(spec_mb, recon_spec)
        L_mag     = magnitude_l1_loss(spec_mb, recon_spec)

        if epoch < unfreeze_epoch:  # freeze until ISTFT is reliable
            L_time   = tf.stop_gradient(L_time)
            L_mrstft = tf.stop_gradient(L_mrstft)
            grad_loss= tf.stop_gradient(grad_loss)
            lap_loss = tf.stop_gradient(lap_loss)
            L_mag    = tf.stop_gradient(L_mag)

        # -------------- mask‑branch losses ------------------------------
        mask_loss = custom_mask_loss(mask_mb, recon_mask)

        rv         = tf.random.uniform([])
        drop_spec  = rv <  modality_dropout_prob
        drop_mask  = (rv >= modality_dropout_prob) & (rv < 2.*modality_dropout_prob)
        spec_coeff = tf.cast(~drop_spec, tf.float32)
        mask_coeff = tf.cast(~drop_mask, tf.float32)

        damage_pred = tf.reduce_mean(recon_mask, axis=[1,2,3])
        damage_true = tf.reduce_mean(mask_mb,    axis=[1,2,3])
        loss_damage = tf.reduce_mean(tf.square(damage_pred - damage_true))

        recon_loss = mask_weight * mask_loss * mask_coeff

        total_loss = (
            recon_loss +
            beta * js_div +
            time_weight * L_time +
            1.5 * L_mrstft +
            0.3 * grad_loss +
            0.3 * lap_loss +
            0.3 * L_mag +
            100.0 * loss_damage
        )
        loss = total_loss / ACC_STEPS

    # ---------------------------------------------------------------------
    # Back‑prop ----------
    # ---------------------------------------------------------------------
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)

    # -------- accumulate --------------------------------------------------
    for g_acc, g in zip(grad_accum, grads):
        g_acc.assign_add(g)
    step_in_acc.assign_add(1)

    # ---------- return scalars (no grads) ---------------------------------
    return (
        total_loss, mask_loss, js_div, recon_loss, mask_coeff,
        L_time, time_weight, L_mrstft, grad_loss, lap_loss, loss_damage, L_mag
    )

def val_step(model, spec_in, mask_in, test_id_in, wave_in, beta, mask_weight, time_weight):
    """
    Performs a validation step for one batch.
    Returns individual losses for logging and early stopping.
    Includes the same losses as train_step but without dropout.
    """
    recon_spec, recon_mask, (_, _, _, js_div) = model(
        spec_in, mask_in, test_id_in, training=False, missing_modality=None
    )

    recon_wave = model.istft_layer(recon_spec, length=tf.shape(wave_in)[1])
    wave_in_f32 = tf.cast(wave_in, tf.float32)
    L_time_val = waveform_l1_loss(wave_in_f32, recon_wave)
    L_mrstft_val = multi_channel_mrstft_loss(wave_in_f32, recon_wave)

    grad_val = gradient_loss_phase_only(spec_in, recon_spec)
    lap_val = laplacian_loss_phase_only(spec_in, recon_spec)
    mag_val = magnitude_l1_loss(spec_in, recon_spec)


    damage_pred = tf.reduce_mean(recon_mask, axis=[1,2,3])
    damage_true = tf.reduce_mean(mask_in,    axis=[1,2,3])
    damage_val = tf.reduce_mean(tf.square(damage_pred - damage_true))

    mask_l  = custom_mask_loss(mask_in, recon_mask)
    recon_l = mask_weight * mask_l

    tot_l = (
        recon_l +
        beta * js_div +
        time_weight * L_time_val +
        1.5 * L_mrstft_val +
        0.3 * grad_val +
        0.3 * lap_val +
        0.3 * mag_val +
        100 * damage_val
    )

    return {
        "total": tot_l,
        "mask": mask_l,
        "js": js_div,
        "time": L_time_val,
        "mrstft": L_mrstft_val,
        "grad": grad_val,
        "lap": lap_val,
        "damage": damage_val,
        "mag": mag_val
    }

def train_spectral_mmvae(
    model,
    train_dataset,
    val_dataset,
    optimizer,
    num_epochs        = 100,
    patience          = 10,
    beta_schedule     = "linear",
    modality_dropout_prob = 0.10,
    strategy          = None,
    unfreeze_epoch   = 20,
    beta_warmup_epochs      = 60,
    max_beta         = 0.15,
    ):
    """
    Main training loop for the Spectral MMVAE model.
    Handles multi-GPU execution, beta and time-weight schedules, early stopping, and metric tracking.
    """
    metrics = {k: [] for k in (
        "train_total", "train_mask", "train_js", "train_time", "train_mrstft",
        "train_grad", "train_lap", "train_damage", "train_mag",
        "val_total", "val_mask", "val_js", "val_time", "val_mrstft",
        "val_grad", "val_lap", "val_damage", "val_mag")}

    best_val_loss = float("inf")
    no_improvement_cnt = 0
    train_batches = sum(1 for _ in train_dataset)
    val_batches = sum(1 for _ in val_dataset)
    print(f"🔄 Starting Training: {train_batches} train batches, {val_batches} val batches")

    beta_log = []

    global grad_accum, step_in_acc
    grad_accum = [tf.Variable(tf.zeros_like(v), trainable=False)
                  for v in model.trainable_variables]
    step_in_acc = tf.Variable(0, trainable=False, dtype=tf.int32)



    for epoch in range(num_epochs):
        print(f"\n🔍 VRAM usage at start of epoch {epoch + 1}:")
        log_vram_usage()

        beta = get_beta_schedule(epoch, num_epochs, beta_schedule, beta_warmup_epochs, max_beta)
        # weight for the L1 loss in the time domain
        time_weight = get_time_weight(epoch,warmup = 200, max_w = 0.1)
        mask_weight = dynamic_weighting(epoch, num_epochs)
        print(f"📌 Epoch {epoch + 1}/{num_epochs} | Beta={beta:.5f} | MaskW={mask_weight:.02f}")

        if epoch == unfreeze_epoch:
            model.istft_layer.trainable = True
            print("🔓 Unfroze TFInverseISTFT layer")

        acc = {k: 0.0 for k in (
            "train_total", "train_mask", "train_js", "train_time",
            "train_mrstft", "train_grad", "train_lap", "train_damage", "train_mag", "train_steps")}


        for step, (spec_in, mask_in, test_id_in, wave_in) in enumerate(train_dataset):
            results = distributed_train_step(
                strategy, model, optimizer,
                (spec_in, mask_in, test_id_in, wave_in),
                tf.constant(beta,  tf.float32),
                tf.constant(mask_weight, tf.float32),
                tf.constant(time_weight, tf.float32),
                tf.constant(epoch, tf.int32)
            )

            (tot, mask_l, js_d, recon_l, mask_c,
            time_l, _, mrstft_l, grad_l, lap_l, dmg_l, mag_l) = results

            # apply weights every ACC_STEPS *host* iterations
            if (step + 1) % ACC_STEPS == 0:
                # run the update on all replicas
                strategy.run(apply_accum_grads, args=(optimizer, model))

            red = lambda x: strategy.reduce(tf.distribute.ReduceOp.MEAN, x, axis=None)
            tot, mask_l, js_d, _, mask_c, time_l, _, mrstft_l, grad_l, lap_l, dmg_l, mag_l = map(red, results)

            # 💬 debug every 50 mini-batches
            if step % 50 == 0:
                tf.print("📉 [Debug] MRSTFT Loss:", mrstft_l)
                tf.print("🧮 [Debug] ISTFT beta:", model.istft_layer.beta)
            
            beta_log.append((epoch, step, float(mrstft_l.numpy()), float(model.istft_layer.beta.numpy())))


            acc["train_total"] += float(tot.numpy())
            acc["train_mask"] += float((mask_l * mask_c).numpy())
            acc["train_js"] += float(js_d.numpy())
            acc["train_time"] += float(time_l.numpy())
            acc["train_mrstft"] += float(mrstft_l.numpy())
            acc["train_grad"] += float(grad_l.numpy())
            acc["train_lap"] += float(lap_l.numpy())
            acc["train_damage"] += float(dmg_l.numpy())
            acc["train_mag"] += float(mag_l.numpy())
            acc["train_steps"] += 1

            print(f"... | TimeLoss={acc['train_time']/acc['train_steps']:.4f} | TimeW={time_weight:.4f} | MRSTFT={acc['train_mrstft']/acc['train_steps']:.4f}")

        for key in ["total", "mask", "js", "time", "mrstft", "grad", "lap", "damage", "mag"]:
            metrics[f"train_{key}"].append(acc[f"train_{key}"] / max(acc["train_steps"], 1))

        print(f"✅ [Train] Loss={metrics['train_total'][-1]:.4f} | "
              f"Mask={metrics['train_mask'][-1]:.4f} | JS={metrics['train_js'][-1]:.4f}")

        val_stats = {k: 0.0 for k in ("total", "mask", "js", "time", "mrstft", "grad", "lap", "damage", "mag", "steps")}

        for spec_in, mask_in, test_id_in, wave_in in val_dataset:
            res = val_step(model, spec_in, mask_in, test_id_in, wave_in,
                           tf.constant(beta, tf.float32),
                           tf.constant(mask_weight, tf.float32),
                           tf.constant(time_weight, tf.float32))

            for k in res:
                val_stats[k] += float(res[k].numpy())
            val_stats["steps"] += 1

        for k in ["total", "mask", "js", "time", "mrstft", "grad", "lap", "damage", "mag"]:
            metrics[f"val_{k}"].append(val_stats[k] / max(val_stats["steps"], 1))

        print(f"  🔵 [Val] => Total={metrics['val_total'][-1]:.4f} | "
              f"Mask={metrics['val_mask'][-1]:.4f} | JS={metrics['val_js'][-1]:.4f} | "
              f"🟣 TimeLoss={metrics['val_time'][-1]:.4f} | 🎯 MRSTFT={metrics['val_mrstft'][-1]:.4f}")

        current_val = metrics["val_total"][-1] if val_stats["steps"] else float("inf")
        if current_val < best_val_loss:
            best_val_loss = current_val
            no_improvement_cnt = 0
            model.save_weights("../results_mmvae/best_spectral_mmvae.weights.h5")
            model.save("../results_mmvae/best_model_spectral_mmvae.keras")
            pd.DataFrame(beta_log, columns=["epoch","step","mrstft","beta"]).to_csv("logs/beta_tracking.csv", index=False)
            print("✅ Saved best weights and model.")
        else:
            no_improvement_cnt += 1
            print(f"🚨 No improvement for {no_improvement_cnt}/{patience}")

        if no_improvement_cnt >= patience:
            print(f"🛑 Early stopping at epoch {epoch + 1}.")
            model.save_weights("results_mmvae/final_spectral_mmvae.weights.h5")
            model.save("results_mmvae/final_model_spectral_mmvae.keras")
            pd.DataFrame(beta_log, columns=["epoch","step","mrstft","beta"]).to_csv("logs/beta_tracking.csv", index=False)
            print("📦 Saved model to:", os.path.abspath("../results_mmvae/final_model_spectral_mmvae.keras"))
            break

    return metrics

@tf.function
def apply_accum_grads(optimizer, model):
    """Must be run via  strategy.run(apply_accum_grads, …)  so that we
    are in replica-context when apply_gradients() is executed."""
    optimizer.apply_gradients(zip(grad_accum, model.trainable_variables))
    for g in grad_accum:
        g.assign(tf.zeros_like(g))
    step_in_acc.assign(0)


@tf.function
def distributed_train_step(strategy, model, optimizer, batch,
                           beta, mask_w, time_w, epoch):

    spec_mb, mask_mb, test_id_mb, wave_mb = batch

    # ---- run on every replica ---------------------------------
    per_replica = strategy.run(
        train_step,
        args=(
            model, optimizer,
            spec_mb, mask_mb, test_id_mb, wave_mb,
            beta,          0.0,          # ← spec_weight (unused)
            mask_w,        0.1,          # ← modality_dropout_prob
            time_w,        epoch         # ←   ↙ these two were missing
        )
    )

    # reduce *once* here – no extra reduction in the caller
    return [strategy.reduce(tf.distribute.ReduceOp.MEAN, t, axis=None)
            for t in per_replica]


def init_accumulators(model):
    """Create the gradient‑accumulation buffers _once_."""
    global grad_accum, step_in_acc
    grad_accum = [tf.Variable(tf.zeros_like(v), trainable=False)
                  for v in model.trainable_variables]
    step_in_acc = tf.Variable(0, trainable=False, dtype=tf.int32)

# ----- Visualization Functions and tests -----
def save_visualizations_and_metrics(model, train_dataset, val_dataset, training_metrics, output_dir="results_mmvae"):
    """
    Aggregates and saves:
      1. Training/validation curves (total and per-loss)
      2. UMAP of latent space
      3. Cosine similarity histogram
      4. Interpolation through latent space
      5. Weight statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    # 1. Training/Validation Loss Curves
    def plot_total_loss_curves(metrics):
        epochs = list(range(1, len(metrics['train_total']) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=metrics['train_total'], mode='lines+markers', name="Train Total", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=epochs, y=metrics['val_total'], mode='lines+markers', name="Val Total", line=dict(color='red')))
        fig.update_layout(title="Total Loss vs Epochs", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_white")
        file_path = os.path.join(plots_dir, "total_loss_curves.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved total loss plot to {file_path}")

    def plot_individual_losses(metrics):
        epochs = list(range(1, len(metrics['train_total']) + 1))
        fig = go.Figure()
        loss_keys = ['time', 'mrstft', 'grad', 'lap', 'mask', 'damage', 'mag']
        colors = {
            'time': 'orange', 'mrstft': 'green', 'grad': 'blue',
            'lap': 'purple', 'mask': 'black', 'damage': 'gray', 'mag': 'pink'
        }
        for key in loss_keys:
            if f"train_{key}" in metrics:
                fig.add_trace(go.Scatter(
                    x=epochs, y=metrics[f"train_{key}"],
                    mode='lines', name=f"Train {key.title()}",
                    line=dict(color=colors.get(key, 'gray'), dash='dash')))
                fig.add_trace(go.Scatter(
                    x=epochs, y=metrics[f"val_{key}"],
                    mode='lines', name=f"Val {key.title()}",
                    line=dict(color=colors.get(key, 'gray'), dash='dot')))

        fig.update_layout(
            title="Individual Losses vs Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white")
        file_path = os.path.join(plots_dir, "individual_loss_curves.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved individual loss plots to {file_path}")

    plot_total_loss_curves(training_metrics)
    
    plot_individual_losses(training_metrics)
    
    #-----------------------------------------------------------------------
    # 2. 3D Latent Space Visualization
    def extract_and_reduce_latents(dataset):
        # Extract latent vectors from the spectrogram encoder only
        latent_vectors = []
        test_ids = []
        for spec_in, _, test_id_in, _wave_in in dataset:
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
        reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.05)
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
        for spec_in, _, _, _ in dataset:
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
        for spec_batch, mask_batch, _, _ in dataset.take(1):
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
    def save_model_weights_stats(model, out_path):
        with open(out_path, "w") as f:
            for layer in model.layers:
                for w in layer.weights:
                    w_np = w.numpy()
                    f.write(f"{w.name:<60} "
                            f"trainable={w.trainable:<5} "
                            f"min={w_np.min():.4f} "
                            f"max={w_np.max():.4f} "
                            f"mean={w_np.mean():.4f} "
                            f"std={w_np.std():.4f}\n")
        print("🔍 Weight stats saved →", out_path)

    
    save_model_weights_stats()

    # Optionally, you can return all gathered metrics:
    return {
        "latent_metrics": latent_metrics,
        "latent_space_3d": latent_3d,
    }

# ----- Main Function -----
def main():
    # ------------------------------------------------------------------
    # 0)  Params
    # ------------------------------------------------------------------
    debug_mode     = False
    latent_dim     = 256
    batch_size     = 128
    total_epochs   = 1000
    patience       = 300
    resume_training = False

    unfreeze_istft_epoch = 30
    max_beta       = 0.2
    beta_warmup_epochs = 100
    beta_schedule   = "linear" # "linear", "cyclical", "constant"
    modality_dropout_prob = 0.0

    #spec_params
    segment_duration = 4.0
    nperseg        = 256
    noverlap       = 224
    sample_rate    = 200


    # output folders ---------------------------------------------------
    os.makedirs("logs", exist_ok=True)
    os.makedirs("../results_mmvae",               exist_ok=True)
    os.makedirs("../results_mmvae/model_checkpoints", exist_ok=True)
    os.makedirs("../results_mmvae/latent_analysis",   exist_ok=True)
    os.makedirs("../results_mmvae/cross_modal",       exist_ok=True)

    # Cache paths 
    tag = f"{segment_duration:.2f}s_{nperseg}_{noverlap}"
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    final_path = os.path.join(cache_dir, f"specs_{tag}.npy")
    heatmaps_path = os.path.join(cache_dir, f"masks_{tag}.npy")
    ids_path = os.path.join(cache_dir, f"segIDs_{tag}.npy")
    segments_path = os.path.join(cache_dir, f"segments_{tag}.npy")

    # ------------------------------------------------------------------
    # 1)  LOAD PROCESSED FEATURES
    # ------------------------------------------------------------------
    all_cached = (os.path.exists(final_path) and
                os.path.exists(heatmaps_path) and
                os.path.exists(ids_path) and
                os.path.exists(segments_path))

    if all_cached:
        print("✅  Loading cached NumPy arrays …")
        spectral_features = np.load(final_path,    mmap_mode="r")
        mask_segments     = np.load(heatmaps_path, mmap_mode="r")
        test_ids          = np.load(ids_path,      mmap_mode="r")
        segments          = np.load(segments_path, mmap_mode="r")

    else:
        print("⚠️  Cache missing – computing everything from raw data …")
        # ------------- call the unified loader -------------
        (_,       # accel_dict (not needed here)
        _,       # binary_masks
        heatmaps,
        segments,
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
        np.save(segments_path, segments)
        print("✅  Written new cache files.")

    print(f"Spectrograms : {spectral_features.shape}")
    print(f"Masks        : {mask_segments.shape}")
    print(f"Test‑IDs     : {test_ids.shape}")
    print(f"Segments     : {segments.shape}")


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
    train_seg   = segments[train_idx]
    val_seg     = segments[val_idx]

    print(f"Train set : {train_spec.shape[0]}  |  Val set : {val_spec.shape[0]}")

    # ------------------------------------------------------------------
    # 3)  Build tf.data Datasets
    # ------------------------------------------------------------------
    train_dataset = create_tf_dataset(train_spec, train_mask, train_ids, train_seg,
                                      batch_size, debug_mode=debug_mode)
    val_dataset   = create_tf_dataset(val_spec,   val_mask,  val_ids, val_seg,
                                      batch_size, debug_mode=debug_mode)

    print(f"✅  Train batches: {sum(1 for _ in train_dataset)}")
    print(f"✅  Val   batches: {sum(1 for _ in val_dataset)}")

    # ------------------------------------------------------------------
    # 4)  Training‑loop
    # ------------------------------------------------------------------
    best_weights_path  = "results_mmvae/best_spectral_mmvae.weights.h5"
    final_weights_path = "results_mmvae/final_spectral_mmvae.weights.h5"

    metrics_path = "results_mmvae/training_metrics.npy"
    if resume_training and os.path.exists(metrics_path):
        all_metrics = np.load(metrics_path, allow_pickle=True).item()
        print("✅  Loaded previous metrics for resuming.")
    else:
        all_metrics = {
            'train_total': [], 'train_mask': [], 'train_js': [], 'train_time': [], 'train_mrstft': [],
            'train_grad': [], 'train_lap': [], 'train_damage': [],
            'val_total':   [], 'val_mask':   [], 'val_js':   [], 'val_time': [], 'val_mrstft': [],
            'val_grad':   [], 'val_lap':   [], 'val_damage': []
        }


    start_epoch = len(all_metrics['train_total'])
    early_stopped = False

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        spec_shape = spectral_features.shape[1:]          # (F, T, 2C)
        mask_shape = (32, 96, 1)

        model = SpectralMMVAE(latent_dim, spec_shape, mask_shape, nperseg, noverlap)

        # build graph
        dummy_spec = tf.zeros((1, *spec_shape))
        dummy_mask = tf.zeros((1, *mask_shape))
        _ = model(dummy_spec, dummy_mask, training=True)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=5e-5,
            decay_steps=10_000,
            decay_rate=0.9,
            staircase=True)

        # AdamW
        base_optimizer = AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6)

        optimizer      = base_optimizer

        init_accumulators(model)     

        if start_epoch > 0:
            if os.path.exists(best_weights_path):
                model.load_weights(best_weights_path)
                print(f"✅ Loaded best weights (epoch {start_epoch}).")
            elif os.path.exists(final_weights_path):
                model.load_weights(final_weights_path)
                print(f"✅ Loaded final weights (epoch {start_epoch}).")
            else:
                print(f"🚨 WARNING: No weights file found to resume from at epoch {start_epoch}!")
        else:
            print("✨ Starting training from scratch.")

        metrics = train_spectral_mmvae(
            model,
            train_dataset,
            val_dataset,
            optimizer,
            num_epochs           = total_epochs, 
            patience             = patience,
            beta_schedule        = beta_schedule,
            modality_dropout_prob= modality_dropout_prob,
            strategy             = strategy,
            unfreeze_epoch       = unfreeze_istft_epoch,
            beta_warmup_epochs   = beta_warmup_epochs,
            max_beta            = max_beta,
        )

        for k in all_metrics:
            all_metrics[k].extend(metrics[k])

        if not early_stopped:
            model.save_weights("results_mmvae/final_spectral_mmvae.weights.h5")
            model.save("results_mmvae/final_model_spectral_mmvae.keras")
            assert os.path.exists("results_mmvae/final_model_spectral_mmvae.keras"), "❌ Model save failed — file not found"
            print("📦 Saved model to:", os.path.abspath("../results_mmvae/final_model_spectral_mmvae.keras"))
            print("✅ Saved final model (full run, no early stopping)")

        np.save("results_mmvae/training_metrics.npy", all_metrics)

    # --- CLEAN-UP: free VRAM & graph ---------------------------------
    tf.keras.backend.clear_session()
    gc.collect()

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
