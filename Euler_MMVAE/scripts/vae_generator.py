import os
import sys
import gc
import time
import warnings
import logging
import psutil
import numpy as np
import cv2
import keras
from keras import layers, Model, optimizers
from keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts  
from scipy import signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
import pandas as pd
import umap
from sklearn.mixture import GaussianMixture
import GPUtil
import tensorflow as tf

# Multithreading
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Set working directory
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
        
# ----- Utility -----
def segment_and_transform(
    accel_dict, 
    heatmap_dict,
    chunk_size=1, 
    sample_rate=200, 
    segment_duration=4.0, 
    percentile=99
    ):
    """
    Extracts segments from time series data centered around peak RMS values.
    Returns raw segments and their corresponding binary masks.
    
    Args:
        accel_dict: Dictionary mapping test IDs to time series data.
        heatmap_dict: Dictionary mapping test IDs to binary mask data (downsampled).
        chunk_size: Number of test IDs to process in one batch.
        sample_rate: Sampling rate of the time series data (Hz).
        segment_duration: Duration of each segment (seconds).
        percentile: Percentile threshold for peak detection.
        
    Returns:
        Tuple of arrays: (raw_segments, mask_segments, test_ids)
    """
    window_size = int(sample_rate * segment_duration)
    half_window = window_size // 2
    
    test_ids = list(accel_dict.keys())
    
    # Instead of yield, we'll collect all data and return it at once
    all_raw_segments = []
    all_mask_segments = []
    all_test_ids = []
    
    # Dictionary to count segments per test ID for debugging
    seg_counts = {}
    
    for i in range(0, len(test_ids), chunk_size):
        chunk_ids = test_ids[i:i + chunk_size]
        
        for test_id in chunk_ids:
            if test_id not in heatmap_dict:
                print(f"Warning: Test ID {test_id} not found in heatmap_dict; skipping.")
                continue

            # Debug: print out the shape of the mask for this test
            mask_val = heatmap_dict[test_id]
            try:
                mask_shape = np.array(mask_val).shape
            except Exception:
                mask_shape = "unknown"
            print(f"Processing Test ID {test_id}: mask shape = {mask_shape}")
            
            all_ts = accel_dict[test_id]
            
            for ts_raw in all_ts:
                rms_each_sample = np.sqrt(np.mean(ts_raw**2, axis=1))
                threshold = np.percentile(rms_each_sample, percentile)
                peak_indices = np.where(rms_each_sample >= threshold)[0]
                
                for pk in peak_indices:
                    start = pk - half_window
                    end = pk + half_window
                    if start < 0:
                        start = 0
                        end = window_size
                    if end > ts_raw.shape[0]:
                        end = ts_raw.shape[0]
                        start = end - window_size
                    if (end - start) < window_size:
                        continue
                    
                    segment_raw = ts_raw[start:end, :]
                    
                    all_raw_segments.append(segment_raw)
                    all_mask_segments.append(heatmap_dict[test_id])
                    all_test_ids.append(int(test_id))
                    
                    seg_counts[test_id] = seg_counts.get(test_id, 0) + 1
                    
    # Debug: print summary of segments per test ID
    unique_test_ids = np.unique(all_test_ids)
    print("Segment counts by test ID:", seg_counts)
    print("Unique test IDs from segmentation:", unique_test_ids)
    
    # Convert lists to numpy arrays
    raw_segments = np.array(all_raw_segments, dtype=np.float32)
    mask_segments = np.array(all_mask_segments, dtype=np.float32)
    test_ids = np.array(all_test_ids, dtype=np.int32)
    
    print(f"Extracted {len(raw_segments)} segments, each with a corresponding test ID.")
    print(f"Number of unique test IDs in segmentation: {len(unique_test_ids)}")
    
    return raw_segments, mask_segments, test_ids

def compute_and_spectrograms(raw_segments, fs=200, nperseg=256, noverlap=192):
    """
    Compute or load cached spectrograms.
    
    Args:
        raw_segments: Raw time series (N, time_length, channels)
        fs, nperseg, noverlap: STFT parameters
        cache_path: File path to save/load spectrograms
        
    Returns:
        Spectrogram features (N, freq_bins, time_bins, channels*2)
    """
    print("⏳ Computing STFT for all segments...")
    complex_spectrograms = compute_complex_spectrogram(raw_segments, fs, nperseg, noverlap)
    return complex_spectrograms

def compute_complex_spectrogram(
    time_series,
    fs=200,
    nperseg=128,   # Adjusted window size (was 256)
    noverlap=64     # Adjusted overlap (was 192)
    ):
    """
    Compute STFT-based spectrograms with debugging.
    
    Args:
        time_series: shape (batch_size, time_steps, channels)
        fs: Sampling frequency in Hz
        nperseg: Window length for STFT
        noverlap: Overlap between windows
        
    Returns:
        Complex spectrograms (batch, freq_bins, time_bins, channels*2)
    """
    batch_size, time_steps, channels = time_series.shape

    # Compute frame step
    frame_step = nperseg - noverlap
    print(f"🔍 STFT Config: nperseg={nperseg}, noverlap={noverlap}, frame_step={frame_step}")

    # Test STFT on a single sample
    test_stft = tf.signal.stft(
        time_series[0, :, 0],
        frame_length=nperseg,
        frame_step=frame_step,
        fft_length=nperseg
    ).numpy()

    print(f"📏 Expected STFT shape: (time_bins={test_stft.shape[0]}, freq_bins={test_stft.shape[1]})")

    if test_stft.shape[0] == 0:
        raise ValueError("⚠️ STFT produced 0 time bins! Adjust `nperseg` or `noverlap`.")

    # Pre-allocate spectrograms
    all_spectrograms = np.zeros((batch_size, test_stft.shape[1], test_stft.shape[0], channels*2), dtype=np.float32)

    for i in range(batch_size):
        for c in range(channels):
            stft = tf.signal.stft(
                time_series[i, :, c],
                frame_length=nperseg,
                frame_step=frame_step,
                fft_length=nperseg
            ).numpy()

            if stft.shape[0] == 0:
                raise ValueError(f"⚠️ STFT returned 0 time bins for sample {i}, channel {c}!")

            # Extract magnitude & phase
            mag = np.log1p(np.abs(stft))  # log-magnitude
            phase = np.angle(stft)        # phase in [-pi, pi]

            # Store in output array
            all_spectrograms[i, :, :, 2*c] = mag.T
            all_spectrograms[i, :, :, 2*c+1] = phase.T

    print(f"✅ Final spectrogram shape: {all_spectrograms.shape}")
    return all_spectrograms

def inverse_spectrogram(complex_spectrograms, time_length, fs=200, nperseg=256, noverlap=128, batch_processing_size=100):
    """
    Convert complex spectrograms back to time series with batched processing.
    
    Args:
        complex_spectrograms: Complex spectrograms (batch, freq, time, channels)
        time_length: Original time series length
        fs: Sampling frequency
        nperseg: Length of each segment
        noverlap: Number of points to overlap between segments
        batch_processing_size: Number of samples to process at once
    
    Returns:
        Reconstructed time series (batch, time_length, channels)
    """
    batch_size, freq_bins, time_bins, channels = complex_spectrograms.shape
    num_orig_channels = channels // 2  # Since we have magnitude and phase for each channel
    
    # Pre-allocate the result array
    time_series = np.zeros((batch_size, time_length, num_orig_channels), dtype=np.float32)
    
    # Process in batches
    total_batches = (batch_size + batch_processing_size - 1) // batch_processing_size
    
    for batch_idx in range(total_batches):
        print(f"Reconstructing batch {batch_idx+1}/{total_batches}")
        start_idx = batch_idx * batch_processing_size
        end_idx = min((batch_idx + 1) * batch_processing_size, batch_size)
        
        # Extract the current batch
        current_batch = complex_spectrograms[start_idx:end_idx]
        
        # Convert the spectrogram features to complex spectrograms
        complex_specs = np.zeros((end_idx-start_idx, freq_bins, time_bins, num_orig_channels), dtype=np.complex64)
        
        for c in range(num_orig_channels):
            # Extract magnitude and phase
            log_magnitude = current_batch[:, :, :, c*2]
            phase = current_batch[:, :, :, c*2+1]
            
            # Convert back to linear magnitude
            magnitude = np.exp(log_magnitude) - 1.0
            
            # Combine magnitude and phase to get complex spectrogram
            complex_specs[:, :, :, c] = magnitude * np.exp(1j * phase)
        
        # Apply inverse STFT
        for b_rel, b_abs in enumerate(range(start_idx, end_idx)):
            for c in range(num_orig_channels):
                # Transpose back to TensorFlow's expected format
                stft_tensor = tf.constant(complex_specs[b_rel, :, :, c].T, dtype=tf.complex64)
                
                # Use TensorFlow's inverse STFT
                inverse_stft = tf.signal.inverse_stft(
                    stft_tensor,
                    frame_length=nperseg,
                    frame_step=nperseg-noverlap,
                    fft_length=nperseg
                )
                
                actual_length = min(inverse_stft.shape[0], time_length)
                time_series[b_abs, :actual_length, c] = inverse_stft[:actual_length].numpy()
        
        # Clear memory between batches
        if batch_idx < total_batches - 1:
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
    
    return time_series

def cache_final_features(complex_specs, cache_path="cached_spectral_features.npy"):
    """
    If 'cache_path' exists, load it via mmap. Otherwise,
    convert 'complex_specs' to magnitude+phase features,
    save to disk, then memory-map.
    """
    if os.path.exists(cache_path):
        print(f"📂 Loading final spectral features from {cache_path}")
        return np.load(cache_path)
    
    # Convert to final shape (mag+phase)
    print("⏳ Converting complex STFT -> final magnitude+phase features...")
    spectral_features = spectrogram_to_features(complex_specs)
    
    # Save the final shape
    np.save(cache_path, spectral_features)
    print(f"✅ Final spectral features saved to {cache_path}")

    return np.load(cache_path)

def spectrogram_to_features(complex_spectrograms):
    """
    Convert complex spectrograms to feature representation suitable for CNN processing.
    Now uses batched processing to save memory.
    
    Args:
        complex_spectrograms: Complex spectrograms with shape (batch, freq, time, channels)
    
    Returns:
        Features with shape (batch, freq, time, channels*2) where for each original channel
        we have magnitude and phase
    """
    batch_size, freq_bins, time_bins, channels = complex_spectrograms.shape
    
    # Initialize feature array (magnitude and phase for each channel)
    features = np.zeros((batch_size, freq_bins, time_bins, channels * 2), dtype=np.float32)
    
    # Process in batches of 1000 samples to save memory
    batch_size_proc = 1000
    
    for start_idx in range(0, batch_size, batch_size_proc):
        end_idx = min(start_idx + batch_size_proc, batch_size)
        print(f"Processing spectrograms {start_idx}-{end_idx-1}/{batch_size}")
        
        # Extract a batch of spectrograms
        batch_specs = complex_spectrograms[start_idx:end_idx]
        
        # Extract magnitude and phase
        for c in range(channels):
            # Magnitude (log scale for better dynamic range)
            magnitude = np.abs(batch_specs[:, :, :, c])
            # Add small constant to avoid log(0)
            log_magnitude = np.log1p(magnitude)
            
            # Phase
            phase = np.angle(batch_specs[:, :, :, c])
            
            # Store in feature array
            features[start_idx:end_idx, :, :, c*2] = log_magnitude
            features[start_idx:end_idx, :, :, c*2+1] = phase
        
        # Free memory
        del batch_specs
        import gc
        gc.collect()
    
    return features

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
        self.conv_out = layers.Conv2D(channels * 2, 3, padding='same')

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
        self.output_layer = layers.Conv2D(1, 3, padding='same', activation='sigmoid')
    
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
def weighted_descriptor_mse_loss(y_true, y_pred):
    """
    MSE loss for descriptor reconstruction that only considers non-zero (valid) elements.
    
    Args:
        y_true: Ground truth descriptor array [batch, max_cracks, mask_length]
        y_pred: Predicted descriptor array [batch, max_cracks, mask_length]
        
    Returns:
        Weighted MSE loss
    """
    # Create mask for valid (non-zero) descriptors
    # A descriptor is considered valid if any of its elements are non-zero
    valid_mask = tf.reduce_any(tf.not_equal(y_true, 0), axis=-1)
    valid_mask = tf.cast(valid_mask, dtype=tf.float32)
    
    # Expand mask to match descriptor dimensions
    valid_mask = tf.expand_dims(valid_mask, axis=-1)
    
    # Calculate squared error
    squared_error = tf.square(y_true - y_pred)
    
    # Apply mask to error
    masked_error = squared_error * valid_mask
    
    # Get number of valid elements for averaging
    num_valid = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    
    # Compute mean over valid elements only
    loss = tf.reduce_sum(masked_error) / num_valid
    
    return loss

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
    # # Print shapes for debugging
    # print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    
    # Ensure shapes match exactly
    if y_true.shape != y_pred.shape:
        print(f"WARNING: Shape mismatch - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
        # Make the shapes consistent by cropping to the smaller size
        min_freq = min(y_true.shape[1], y_pred.shape[1])
        min_time = min(y_true.shape[2], y_pred.shape[2])
        min_ch = min(y_true.shape[3], y_pred.shape[3])
        
        y_true = y_true[:, :min_freq, :min_time, :min_ch]
        y_pred = y_pred[:, :min_freq, :min_time, :min_ch]
        print(f"After adjustment - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    
    # Ensure tensor has even number of channels (for magnitude/phase pairs)
    if y_true.shape[-1] % 2 != 0:
        print(f"WARNING: Channel dimension {y_true.shape[-1]} is not even")
        # Truncate to even number of channels if needed
        y_true = y_true[..., :-1]
        y_pred = y_pred[..., :-1]
    
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
    return 0.8 * mag_loss + 0.2 * phase_loss

def dynamic_weighting(epoch, max_epochs, min_weight=0.3, max_weight=1.0):
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
    progress = min(1.0, epoch / (max_epochs * 0.5))  # Reach max_weight halfway through training
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
    BETA_MAX = 0.1   # Maximum value
    
    # Define warmup phase length (in epochs)
    WARMUP_EPOCHS = 150
    
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

def custom_mask_loss(y_true, y_pred, 
                       weight_bce=0.33, 
                       weight_dice=0.33, 
                       weight_focal=0.34, 
                       gamma=2.0, 
                       alpha=0.25):
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
    return combined

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

    def reconstruct_time_series(self, spec_features, fs=200, nperseg=256, noverlap=128, time_length=1000):
        """
        Reconstruct time series from spectrogram features by:
        1. Separating magnitude and phase
        2. Converting back to complex spectrogram
        3. Applying inverse STFT
        
        Args:
            spec_features: Spectrogram features with shape (batch, freq, time, channels*2)
            fs, nperseg, noverlap: Parameters for inverse STFT
            time_length: Original time series length
            
        Returns:
            Reconstructed time series with shape (batch, time_length, channels)
        """
        batch_size, freq_bins, time_bins, total_channels = spec_features.shape
        channels = total_channels // 2  # Each original channel has magnitude and phase
        
        # Convert features back to complex spectrograms
        complex_specs = np.zeros((batch_size, freq_bins, time_bins, channels), dtype=np.complex64)
        
        for c in range(channels):
            # Extract magnitude and phase
            log_magnitude = spec_features[:, :, :, c*2]
            phase = spec_features[:, :, :, c*2+1]
            
            # Convert back to linear magnitude
            magnitude = np.exp(log_magnitude) - 1.0
            
            # Combine magnitude and phase to get complex spectrogram
            complex_specs[:, :, :, c] = magnitude * np.exp(1j * phase)
        
        # Apply inverse STFT to get time series
        time_series = inverse_spectrogram(complex_specs, time_length, fs, nperseg, noverlap, batch_processing_size=100)
        
        return time_series
    
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
        spectrograms = np.load(spectrograms)
    if isinstance(mask_array, str):
        mask_array = np.load(mask_array)
    
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
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ----- Training Function -----
def train_step(
    model, 
    optimizer, 
    spec_in, 
    mask_in, 
    test_id_in,
    missing_modality_str,  # <-- now a normal Python string
    beta, 
    spec_weight, 
    mask_weight
    ):
    """
    Perform one training step (forward pass + backprop) on a single replica.
    missing_modality_str is a Python string in {'spec', 'mask', ''}.
    """
    with tf.GradientTape() as tape:
        # Forward pass
        recon_spec, recon_mask, (all_mus, all_logvars, mixture_prior, js_div) = model(
            spec_in, mask_in, test_id_in,
            training=True,
            missing_modality=missing_modality_str
        )

        # Always produce Tensors (not Python floats)
        spec_loss = complex_spectrogram_loss(spec_in, recon_spec)
        mask_loss = custom_mask_loss(mask_in, recon_mask)

        # Decide recon_loss with simple Python if-else
        if missing_modality_str == 'spec':
            # 'spec' is missing => only do mask_loss
            recon_loss = mask_loss
        elif missing_modality_str == 'mask':
            # 'mask' is missing => only do spec_loss
            recon_loss = spec_loss
        else:
            # no modality missing => combine with weights
            recon_loss = (tf.constant(spec_weight, tf.float32) * spec_loss
                         + tf.constant(mask_weight,  tf.float32) * mask_loss)

        total_loss = recon_loss + beta * js_div

    # Backprop
    grads = tape.gradient(total_loss, model.trainable_variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))

    # Return Tensors, not Python floats
    return total_loss, spec_loss, mask_loss, js_div, recon_loss

def train_spectral_mmvae(
    model, 
    train_dataset, 
    val_dataset, 
    optimizer, 
    num_epochs=100, 
    patience=10,
    beta_schedule='cyclical',
    modality_dropout_prob=0.1,
    strategy=None
    ):
    # Storage for metrics
    metrics = {
        'train_total': [], 'train_spec': [], 'train_mask': [], 
        'train_js': [], 'train_mode': [],
        'val_total': [], 'val_spec': [], 'val_mask': [], 'val_js': []
    }

    best_val_loss = float('inf')
    no_improvement_count = 0

    train_batches_count = sum(1 for _ in train_dataset)
    val_batches_count = sum(1 for _ in val_dataset)
    print(f"🔄 Starting Training: {train_batches_count} train batches, {val_batches_count} val batches")

    for epoch in range(num_epochs):
        epoch_metrics = {
            'train_total': 0.0, 
            'train_spec': 0.0, 
            'train_mask': 0.0, 
            'train_js': 0.0,
            'train_full': 0.0, 
            'train_spec_only': 0.0, 
            'train_mask_only': 0.0,
            'n_full': 0, 
            'n_spec_only': 0, 
            'n_mask_only': 0,
            'train_steps': 0
        }

        beta = get_beta_schedule(epoch, num_epochs, beta_schedule)
        mask_weight = dynamic_weighting(epoch, num_epochs)
        spec_weight = 1.0 - mask_weight
        print(f"📌 Epoch {epoch+1}/{num_epochs} | Beta={beta:.5f} | MaskW={mask_weight:.2f}")

        # ---------------- TRAIN LOOP ----------------
        for step, (spec_in, mask_in, test_id_in) in enumerate(train_dataset):
            # Use a Python random decision per batch
            # => same decision on both replicas (they see the same "random_value")
            rv = random.random()
            if rv < modality_dropout_prob:
                missing_modality_str = 'spec'
            elif rv < 2.0 * modality_dropout_prob:
                missing_modality_str = 'mask'
            else:
                missing_modality_str = ''

            def step_fn(spec_mb, mask_mb, test_id_mb):
                return train_step(
                    model, optimizer, spec_mb, mask_mb, test_id_mb,
                    missing_modality_str,  # a Python string
                    tf.constant(beta, tf.float32),
                    spec_weight,
                    mask_weight
                )

            # Distributed step across replicas
            total_loss, spec_loss_val, mask_loss_val, js_div_val, recon_loss_val = strategy.run(
                step_fn, 
                args=(spec_in, mask_in, test_id_in)
            )

            # Combine across replicas
            total_loss     = strategy.reduce(tf.distribute.ReduceOp.MEAN, total_loss,     axis=None)
            spec_loss_val  = strategy.reduce(tf.distribute.ReduceOp.MEAN, spec_loss_val,  axis=None)
            mask_loss_val  = strategy.reduce(tf.distribute.ReduceOp.MEAN, mask_loss_val,  axis=None)
            js_div_val     = strategy.reduce(tf.distribute.ReduceOp.MEAN, js_div_val,     axis=None)
            recon_loss_val = strategy.reduce(tf.distribute.ReduceOp.MEAN, recon_loss_val, axis=None)

            # Convert them to Python floats for logging outside the step function
            epoch_metrics['train_total'] += float(total_loss.numpy())
            if missing_modality_str != 'spec':
                epoch_metrics['train_spec'] += float(spec_loss_val.numpy())
            if missing_modality_str != 'mask':
                epoch_metrics['train_mask'] += float(mask_loss_val.numpy())
            epoch_metrics['train_js'] += float(js_div_val.numpy())
            epoch_metrics['train_steps'] += 1

            # Count how many times each scenario happened
            if missing_modality_str == '':
                epoch_metrics['n_full'] += 1
                epoch_metrics['train_full'] += float(recon_loss_val.numpy())
            elif missing_modality_str == 'spec':
                epoch_metrics['n_mask_only'] += 1
                epoch_metrics['train_mask_only'] += float(recon_loss_val.numpy())
            else:  # missing_modality_str == 'mask'
                epoch_metrics['n_spec_only'] += 1
                epoch_metrics['train_spec_only'] += float(recon_loss_val.numpy())

        # Averages
        if epoch_metrics['train_steps'] > 0:
            metrics['train_total'].append(epoch_metrics['train_total'] / epoch_metrics['train_steps'])
            full_steps_for_spec = epoch_metrics['train_steps'] - epoch_metrics['n_mask_only']
            full_steps_for_mask = epoch_metrics['train_steps'] - epoch_metrics['n_spec_only']
            metrics['train_spec'].append(epoch_metrics['train_spec'] / max(full_steps_for_spec, 1))
            metrics['train_mask'].append(epoch_metrics['train_mask'] / max(full_steps_for_mask, 1))
            metrics['train_js'].append(epoch_metrics['train_js'] / epoch_metrics['train_steps'])
            metrics['train_mode'].append({
                'full': epoch_metrics['train_full'] / max(epoch_metrics['n_full'], 1),
                'spec_only': epoch_metrics['train_spec_only'] / max(epoch_metrics['n_spec_only'], 1),
                'mask_only': epoch_metrics['train_mask_only'] / max(epoch_metrics['n_mask_only'], 1)
            })

            print(f"✅ [Train] Loss={metrics['train_total'][-1]:.4f} | "
                  f"Spec={metrics['train_spec'][-1]:.4f} | "
                  f"Mask={metrics['train_mask'][-1]:.4f} | "
                  f"JS={metrics['train_js'][-1]:.4f}")

        # ---------------- VAL LOOP ----------------
        val_dict = {'total': 0.0, 'spec': 0.0, 'mask': 0.0, 'js': 0.0, 'steps': 0}
        for step, (spec_in, mask_in, test_id_in) in enumerate(val_dataset):
            # No missing modality for validation
            recon_spec, recon_mask, (all_mus, all_logvars, mixture_prior, js_div_val) = model(
                spec_in, mask_in, test_id_in, training=False, missing_modality=None
            )

            spec_loss_val = complex_spectrogram_loss(spec_in, recon_spec)
            mask_loss_val = custom_mask_loss(mask_in, recon_mask)
            recon_loss_val = spec_weight * spec_loss_val + mask_weight * mask_loss_val
            total_loss_val = recon_loss_val + tf.constant(beta, tf.float32) * js_div_val

            val_dict['total'] += float(total_loss_val.numpy())
            val_dict['spec']  += float(spec_loss_val.numpy())
            val_dict['mask']  += float(mask_loss_val.numpy())
            val_dict['js']    += float(js_div_val.numpy())
            val_dict['steps'] += 1

        if val_dict['steps'] > 0:
            metrics['val_total'].append(val_dict['total']/val_dict['steps'])
            metrics['val_spec'].append(val_dict['spec']/val_dict['steps'])
            metrics['val_mask'].append(val_dict['mask']/val_dict['steps'])
            metrics['val_js'].append(val_dict['js']/val_dict['steps'])

            print(f"  🔵 [Val] => Total={metrics['val_total'][-1]:.4f} | "
                  f"Spec={metrics['val_spec'][-1]:.4f} | "
                  f"Mask={metrics['val_mask'][-1]:.4f} | "
                  f"JS={metrics['val_js'][-1]:.4f}")

        # Early stopping
        current_val_loss = metrics['val_total'][-1] if val_dict['steps'] > 0 else float('inf')
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            no_improvement_count = 0
            model.save_weights("results/best_spectral_mmvae.weights.h5")
            print("✅ Saved best weights")
        else:
            no_improvement_count += 1
            print(f"🚨 No improvement for {no_improvement_count}/{patience}")

        if no_improvement_count >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}.")
            model.save_weights("results/final_spectral_mmvae.weights.h5")
            break

    return metrics

# ----- Visualization Functions and tests -----
def save_visualizations_and_metrics(model, train_dataset, val_dataset, training_metrics, output_dir="results"):
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

def synthesize_examples(model, val_dataset, num_examples=5, fs=200, nperseg=256, noverlap=128):
    """
    For a few validation examples:
      1. Reconstruct time series signals from decoded spectrograms and compare them with the original.
      2. Upsample the decoded mask and overlay it with the original mask.
      3. Also perform latent space interpolation between two chosen examples.
      
    Args:
        model (tf.keras.Model): Your trained SpectralMMVAE model.
        val_dataset (tf.data.Dataset): Validation dataset yielding (spectrogram, mask, test_id).
        num_examples (int): Number of examples to display.
        fs (int): Sampling frequency.
        nperseg (int): STFT window length.
        noverlap (int): STFT overlap.
    """
    # We'll store a few examples
    examples = []
    for spec_batch, mask_batch, test_ids in val_dataset.take(1):
        for i in range(min(num_examples, spec_batch.shape[0])):
            examples.append((spec_batch[i], mask_batch[i], test_ids[i]))
    print(f"Using {len(examples)} examples for reconstruction comparison.")
    
    # For each example, reconstruct the time series from the decoded spectrogram,
    # and upsample the decoded mask (if needed) to compare with the original.
    for idx, (spec_sample, mask_sample, test_id) in enumerate(examples):
        spec_sample = tf.expand_dims(spec_sample, 0)  # shape: (1, freq, time, channels*2)
        mask_sample = tf.expand_dims(mask_sample, 0)  # shape: (1, H, W, 1)
        
        # Get reconstructions from the model.
        # Note: In your model, "recon_spec" comes from decoding a latent sampled from the spectrogram encoder.
        recon_spec, recon_mask, _ = model(spec_sample, mask_sample, tf.constant([[0]]), training=False)
        
        # Reconstruct time series signals using your inverse STFT routine.
        orig_ts = model.reconstruct_time_series(spec_sample.numpy(), fs=fs, nperseg=nperseg, noverlap=noverlap, time_length=800)
        recon_ts = model.reconstruct_time_series(recon_spec.numpy(), fs=fs, nperseg=nperseg, noverlap=noverlap, time_length=800)
        
        # Plot time series: Overlay all 12 channels in one figure.
        # You can either plot each channel separately or use transparency to see the overlap.
        plt.figure(figsize=(10, 6))
        time_axis = np.linspace(0, 4, orig_ts.shape[1])
        for ch in range(orig_ts.shape[2]):
            plt.plot(time_axis, orig_ts[0, :, ch], color='blue', alpha=0.4, label='Original' if ch == 0 else "")
            plt.plot(time_axis, recon_ts[0, :, ch], color='red', alpha=0.4, label='Reconstructed' if ch == 0 else "")
        plt.title(f"Time Series Comparison (Test ID: {test_id})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # For the mask, assume the original mask is at high resolution.
        # If needed, you can upsample the reconstructed mask to match the original.
        # Here we assume your reconstructed mask is already the desired resolution.
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(mask_sample.numpy()[0, :, :, 0], cmap='gray')
        plt.title("Original Mask")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(recon_mask.numpy()[0, :, :, 0], cmap='gray')
        plt.title("Reconstructed Mask")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    # Latent interpolation between two chosen examples (using the spectrogram encoder latents).
    # Here, we simply take the first two examples in the batch.
    if len(examples) >= 2:
        source_spec = tf.expand_dims(examples[0][0], 0)
        target_spec = tf.expand_dims(examples[1][0], 0)
        mu_source, _ = model.spec_encoder(source_spec, training=False)
        mu_target, _ = model.spec_encoder(target_spec, training=False)
        num_steps = 8
        alphas = np.linspace(0, 1, num_steps)
        plt.figure(figsize=(num_steps * 2, 4))
        for i, alpha in enumerate(alphas):
            interp_z = (1 - alpha) * mu_source + alpha * mu_target
            interp_spec = model.spec_decoder(interp_z, training=False)
            plt.subplot(1, num_steps, i+1)
            plt.imshow(interp_spec.numpy()[0, :, :, 0], aspect='auto', cmap='viridis')
            plt.title(f"α={alpha:.1f}")
            plt.axis('off')
        plt.suptitle("Latent Space Interpolation (Spectrograms)")
        plt.tight_layout()
        plt.show()


# ----- Main Function -----
def main():
    # Set debug mode
    debug_mode = False
    
    # ------------------------ 1) Configure Environment ------------------------
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../results/model_checkpoints", exist_ok=True)
    os.makedirs("../results/latent_analysis", exist_ok=True)
    os.makedirs("../results/cross_modal", exist_ok=True)

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # ------------------------ 2) Define Parameters ------------------------
    fs = 200   # Sampling frequency in Hz
    nperseg = 256  # STFT segment length
    noverlap = 224 # STFT overlap
    
    latent_dim = 128 
    batch_size = 32  
    num_epochs = 600 
    patience = 150  

    # Cached file paths
    final_path = "scripts/cached_spectral_features.npy"
    heatmaps_path  = "scripts/cached_masks.npy"
    ids_path   = "scripts/cached_test_ids.npy"

    # ------------------------ 3) Single Check: Are ALL caches present? ------------------------
    all_cached = (
        os.path.exists(final_path)
        and os.path.exists(heatmaps_path)
        and os.path.exists(ids_path)
    )

    if all_cached:
        # ------------------------ A) ALL Cached: Load everything ------------------------
        print("✅ All cached files found! Loading from disk...")

        spectral_features   = np.load(final_path)
        mask_segments       = np.load(heatmaps_path)
        test_ids            = np.load(ids_path)

        print(f"✅ spectral_features: {spectral_features.shape}")
        print(f"✅ mask_segments: {mask_segments.shape}")
        print(f"✅ test_ids: {test_ids.shape}")

    else:
        # ------------------------ B) Missing Any Cache: Re-run entire pipeline ------------------------
        print("⚠️ At least one cache file is missing. Recomputing EVERYTHING from scratch...")
        print("📥 Loading raw data...")
        accel_dict, binary_masks, heatmaps = data_loader.load_data()

        mask_segments = np.array([heatmaps[k] for k in sorted(heatmaps.keys())])
        test_ids = np.array(sorted(heatmaps.keys()))

        print(f"Loaded data for {len(accel_dict)} tests")

        # (2) Segment time series -> get raw_segments, masks, test_ids
        print("✂️ Segmenting data...")
        raw_segments, mask_segments, test_ids = segment_and_transform(accel_dict, heatmaps)
        print(f"✅ Extracted {len(raw_segments)} segments")

        # Save masks & test IDs for future use
        np.save(heatmaps_path, mask_segments)
        np.save(ids_path, test_ids)
        print(f"✅ Saved masks to {heatmaps_path}")
        print(f"✅ Saved test IDs to {ids_path}")

        # (3) Compute STFT -> complex spectrograms
        print("🔄 Computing spectrograms...")
        complex_specs = compute_and_spectrograms(raw_segments, fs, nperseg, noverlap)

        # (4) Convert spectrograms to magnitude/phase features
        print("📝 Converting spectrograms to features...")
        spectral_features = cache_final_features(complex_specs, cache_path=final_path)

    # ------------------------ 7) Split Data ------------------------
    N = spectral_features.shape[0]
    indices = np.random.permutation(N)
    train_size = int(0.8 * N)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    # Slice the cached arrays (they are NumPy arrays now)
    train_spec = spectral_features[train_idx]
    val_spec = spectral_features[val_idx]
    train_mask = mask_segments[train_idx]
    train_ids  = test_ids[train_idx]
    val_mask   = mask_segments[val_idx]
    val_ids    = test_ids[val_idx]

    print(f"Training set: {train_spec.shape[0]} samples")
    print(f"Validation set: {val_spec.shape[0]} samples")

    # ------------------------ 8) Create TensorFlow Datasets ------------------------
    train_dataset = create_tf_dataset(train_spec, train_mask, train_ids, batch_size, debug_mode=debug_mode)
    val_dataset   = create_tf_dataset(val_spec, val_mask, val_ids, batch_size, debug_mode=debug_mode)

    print(f"✅ Train batches: {sum(1 for _ in train_dataset)}")
    print(f"✅ Val batches: {sum(1 for _ in val_dataset)}")


    # ------------------------ 9) Build Model ------------------------
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        spec_shape = spectral_features.shape[1:]
        mask_shape = (32, 96, 1)
        print(f"📐 Building model with:")
        print(f"  - Latent dimension: {latent_dim}")
        print(f"  - Spectrogram shape: {spec_shape}")
        print(f"  - Mask shape: {mask_shape}")

        model = SpectralMMVAE(latent_dim, spec_shape, mask_shape)
        # Dummy forward pass
        dummy_spec = tf.zeros((1, *spec_shape))
        dummy_mask = tf.zeros((1, *mask_shape))
        _ = model(dummy_spec, dummy_mask, training=True)
        model.summary()

        # Create optimizer
        lr_schedule = ExponentialDecay(
            initial_learning_rate=5e-5,
            decay_steps=10000,
            decay_rate=0.9,
            staircase=True
        )

        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6
        )

        # 10) Train Model (Multi-GPU)
        print("🚀 Starting training...")
        training_metrics = train_spectral_mmvae(
            model, 
            train_dataset, 
            val_dataset, 
            optimizer, 
            num_epochs=num_epochs, 
            patience=patience,
            beta_schedule='cyclical',
            modality_dropout_prob=0.05,
            strategy=strategy
        )


    # ------------------------- 11.5) Synthesize Examples ------------------------
    synthesize_examples(model, val_dataset, num_examples=5, fs=200, nperseg=256, noverlap=128)

    # ------------------------ 12) Save Training Metrics ------------------------
    np.save("results/training_metrics.npy", training_metrics)

    # ------------------------ 13) Load Best Model ------------------------
    best_weights_path  = "results/best_spectral_mmvae.weights.h5"
    final_weights_path = "results/final_spectral_mmvae.weights.h5"

    if os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)
        print("✅ Loaded best model weights")
    elif os.path.exists(final_weights_path):
        model.load_weights(final_weights_path)
        print("✅ Loaded final model weights")
    else:
        print("⚠️ No saved weights found, using model from last epoch")

    # ------------------------ 14) Visualize & Save Metrics ------------------------
    # Save plots and metricse:
    #   - Training curves (train/val loss)
    #   - 3D latent space visualization (UMAP)
    #   - Latent analysis (e.g. cosine similarity histogram)
    #   - Latent space interpolation between two samples
    #   - Model weight statistics
    vis_metrics = save_visualizations_and_metrics(model, train_dataset, val_dataset, training_metrics, output_dir="results")

    print("✅ Training & evaluation complete. Results saved in 'results/' directory.")


if __name__ == "__main__":
    main()
