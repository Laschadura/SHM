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
tf.config.threading.set_intra_op_parallelism_threads(3)
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
def load_and_prepare_data():
    """
    Load data using the data_loader module with proper parameters.
    Extract the required dictionaries and prepare data for the VAE.
    Now using padded descriptor data instead of mask data.
    """
    # Define parameters for the data_loader
    params = {
        'keypoint_count': 15,
        'max_gap': 3,
        'curved_threshold': 10,
        'curved_angle_threshold': 75,
        'straight_angle_threshold': 15,
        'min_segment_length': 2,
        'line_thickness': 1,
    }
    
    # Call load_data with the required params argument
    # The function should now return padded_dict as well
    accel_dict, crack_dict, binary_masks, skeletons, padded_dict = data_loader.load_data(params)
    
    print(f"Loaded data for {len(accel_dict)} tests")
    
    # Check descriptor data shape
    sample_descriptor = next(iter(padded_dict.values()))
    print(f"Descriptor shape sample: {sample_descriptor.shape}")
    
    return accel_dict, padded_dict

def segment_and_transform(
    accel_dict, 
    mask_dict,
    chunk_size=1, 
    sample_rate=200, 
    segment_duration=5.0, 
    percentile=99
    ):
    """
    Extracts segments from time series data centered around peak RMS values.
    Returns raw segments and their corresponding binary masks.
    
    Args:
        accel_dict: Dictionary mapping test IDs to time series data.
        mask_dict: Dictionary mapping test IDs to binary mask data.
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
            if test_id not in mask_dict:
                print(f"Warning: Test ID {test_id} not found in mask_dict; skipping.")
                continue

            # Debug: print out the shape of the mask for this test
            mask_val = mask_dict[test_id]
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
                    all_mask_segments.append(mask_dict[test_id])
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

def compute_and_cache_spectrograms(raw_segments, fs=200, nperseg=256, noverlap=192, cache_path="cached_spectrograms.npy"):
    """
    Compute or load cached spectrograms.
    
    Args:
        raw_segments: Raw time series (N, time_length, channels)
        fs, nperseg, noverlap: STFT parameters
        cache_path: File path to save/load spectrograms
        
    Returns:
        Spectrogram features (N, freq_bins, time_bins, channels*2)
    """
    if os.path.exists(cache_path):
        print(f"📂 Loading raw STFT from {cache_path}")
        return np.load(cache_path)
    
    print("⏳ Computing STFT for all segments...")
    complex_spectrograms = compute_complex_spectrogram(raw_segments, fs, nperseg, noverlap)
    np.save(cache_path, complex_spectrograms)
    print(f"✅ Raw STFT saved to {cache_path}")
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

# ----- Enhanced SpectrogramEncoder -----
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
    
# ----- Keep DescriptorEncoder from original implementation -----
class MaskEncoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        # Use 2D convolution layers to progressively downsample the mask
        self.conv1 = layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256, activation='relu')
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

def dynamic_weighting(epoch, max_epochs, min_weight=0.3, max_weight=0.7):
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

def binary_mask_loss_fixed(y_true, y_pred):
    """
    Robust binary crossentropy that guarantees positive values.
    Provides detailed debugging information when issues occur.
    
    Args:
        y_true: Ground truth mask [batch, height, width, 1]
        y_pred: Predicted mask [batch, height, width, 1]
        
    Returns:
        Positive BCE loss scalar
    """
    # Ensure predictions are in valid range
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Implement BCE manually to have more control
    bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    
    # Calculate mean and check for validity
    mean_bce = tf.reduce_mean(bce)
    
    # Add assertion to catch problems
    with tf.control_dependencies([
        tf.debugging.assert_non_negative(
            mean_bce,
            message="BCE loss became negative, check your inputs and model"
        )
    ]):
        # Force positive - just to be absolutely sure
        return tf.abs(mean_bce)

def binary_dice_loss(y_true, y_pred):
    """
    Dice loss for binary segmentation, great for sparse masks.
    Always returns a positive value between 0 and 1.
    
    Args:
        y_true: Ground truth mask [batch, height, width, 1]
        y_pred: Predicted mask [batch, height, width, 1]
        
    Returns:
        Dice loss (1 - Dice coefficient)
    """
    # Flatten the tensors
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    # Ensure predictions are in valid range
    epsilon = tf.keras.backend.epsilon()
    y_pred_f = tf.clip_by_value(y_pred_f, epsilon, 1.0 - epsilon)
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    # Add smoothing term to avoid division by zero
    smooth = 1.0
    dice_coefficient = (2.0 * intersection + smooth) / (union + smooth)
    
    # Dice loss (ranges from 0 to 1)
    dice_loss = 1.0 - dice_coefficient
    
    return dice_loss

def robust_mask_loss(y_true, y_pred):
    """
    Ultra-robust mask loss function that handles extreme values and invalid outputs.
    This is designed to be extremely defensive against all possible issues.
    
    Args:
        y_true: Ground truth mask [batch, height, width, 1]
        y_pred: Predicted mask probabilities [batch, height, width, 1]
        
    Returns:
        Guaranteed positive loss value
    """
    try:
        # 1. First, aggressively clip predictions to valid probability range
        epsilon = 1e-7
        y_pred_safe = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # 2. Check for NaNs or Infs in predictions
        if tf.reduce_any(tf.math.is_nan(y_pred)) or tf.reduce_any(tf.math.is_inf(y_pred)):
            tf.print("⚠️ WARNING: NaN or Inf detected in predictions, using safe defaults")
            # Create a safe default prediction (all 0.5)
            y_pred_safe = tf.ones_like(y_true) * 0.5
        
        # 3. Implement BCE manually with extreme care
        bce = -(y_true * tf.math.log(y_pred_safe) + (1.0 - y_true) * tf.math.log(1.0 - y_pred_safe))
        
        # 4. Check each value and replace any problematic ones
        bce_clean = tf.where(
            tf.math.is_finite(bce),  # Keep only finite values
            bce,                      # Use original where valid
            tf.ones_like(bce) * 0.5   # Replace bad values with moderate constant
        )
        
        # 5. Calculate Dice loss as alternative
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred_safe, [-1])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        smooth = 1.0  # Larger smoothing factor for stability
        dice_coef = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        dice_loss = 1.0 - dice_coef
        
        # 6. Ensure dice loss is valid
        dice_loss = tf.where(
            tf.math.is_finite(dice_loss),
            dice_loss,
            tf.constant(0.5, dtype=tf.float32)
        )
        
        # 7. Take mean of BCE (carefully)
        bce_mean = tf.reduce_mean(bce_clean)
        
        # 8. Calculate final combined loss
        combined_loss = 0.5 * bce_mean + 0.5 * dice_loss
        
        # 9. Final safety check - ensure the result is positive
        combined_loss = tf.maximum(combined_loss, epsilon)
        
        # 10. Print diagnostics
        tf.print("BCE:", bce_mean, "Dice:", dice_loss, "Combined:", combined_loss,
                 "Min pred:", tf.reduce_min(y_pred), "Max pred:", tf.reduce_max(y_pred),
                 output_stream=sys.stdout)
        
        return combined_loss
        
    except Exception as e:
        # Absolute last resort - return a constant loss to keep training going
        tf.print("❌ CRITICAL ERROR in loss calculation:", e, output_stream=sys.stderr)
        return tf.constant(1.0, dtype=tf.float32)

def get_model_stats(model):
    """
    Function to check model weights for potential issues.
    
    Args:
        model: Keras model
        
    Returns:
        Text summary of weight statistics
    """
    stats = []
    for i, layer in enumerate(model.layers):
        if len(layer.weights) > 0:
            for j, w in enumerate(layer.weights):
                w_np = w.numpy()
                w_min = np.min(w_np)
                w_max = np.max(w_np)
                w_mean = np.mean(w_np)
                w_std = np.std(w_np)
                
                # Check for extreme values
                if w_max > 100 or w_min < -100 or w_std > 10:
                    status = "⚠️ EXTREME"
                else:
                    status = "✓ OK"
                
                stats.append(f"Layer {i} ({layer.name}) Weight {j}: min={w_min:.4f}, max={w_max:.4f}, mean={w_mean:.4f}, std={w_std:.4f} {status}")
    
    return "\n".join(stats)

def sparse_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss for highly imbalanced masks (very few positive pixels).
    
    Args:
        y_true: Ground truth mask [batch, height, width, 1]
        y_pred: Predicted mask [batch, height, width, 1]
        gamma: Focusing parameter (higher means more focus on hard examples)
        alpha: Balancing parameter (higher means more weight on positive class)
        
    Returns:
        Focal loss (always positive)
    """
    # Ensure predictions are in valid range
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Convert to logits if not already (TF's built-in BCE with logits is more stable)
    if not tf.is_tensor(y_pred) or y_pred.dtype != tf.float32:
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    
    # Calculate pt (probability of true class)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    
    # Calculate focal weight
    focal_weight = tf.pow(1.0 - pt, gamma)
    
    # Calculate alpha weight (class balancing)
    alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    
    # Calculate crossentropy
    bce = -tf.math.log(pt)
    
    # Apply weights
    loss = alpha_weight * focal_weight * bce
    
    # Return mean
    return tf.reduce_mean(loss)

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
def train_spectral_mmvae(
    model, 
    train_dataset, 
    val_dataset, 
    optimizer, 
    num_epochs=100, 
    patience=10,
    beta_schedule='cyclical',
    modality_dropout_prob=0.1
    ):
    metrics = {
        'train_total': [], 'train_spec': [], 'train_mask': [], 
        'train_js': [], 'train_mode': [],
        'val_total': [], 'val_spec': [], 'val_mask': [], 'val_js': []
    }

    best_val_loss = float('inf')
    no_improvement_count = 0

    print("🔄 Starting Training for Spectral MMVAE with Mixture-of-Experts...")

    train_batches_count = sum(1 for _ in train_dataset)
    val_batches_count = sum(1 for _ in val_dataset)
    print(f"Training on {train_batches_count} batches, validating on {val_batches_count} batches")

    for epoch in range(num_epochs):
        epoch_metrics = {
            'train_total': 0.0, 'train_spec': 0.0, 'train_mask': 0.0, 
            'train_js': 0.0, 'train_full': 0.0, 
            'train_spec_only': 0.0, 'train_mask_only': 0.0,
            'n_full': 0, 'n_spec_only': 0, 'n_mask_only': 0,
            'train_steps': 0
        }

        beta = get_beta_schedule(epoch, num_epochs, beta_schedule)
        # Use dynamic_weighting to get weight for mask loss
        mask_weight = dynamic_weighting(epoch, num_epochs)
        spec_weight = 1.0 - mask_weight

        print(f"📌 Epoch {epoch+1}/{num_epochs} | Beta: {beta:.6f} | Mask Weight: {mask_weight:.2f}")

        for step, (spec_in, mask_in, test_id_in) in enumerate(train_dataset):
            missing_modality = None
            random_value = tf.random.uniform(shape=[], minval=0, maxval=1)
            if random_value < modality_dropout_prob * 2:
                missing_modality = 'spec' if random_value < modality_dropout_prob else 'mask'

            with tf.GradientTape(persistent=True) as tape:
                try:
                    recon_spec, recon_mask, (all_mus, all_logvars, mixture_prior, js_div) = model(
                        spec_in, mask_in, test_id_in,
                        training=True,
                        missing_modality=missing_modality
                    )
                except Exception as e:
                    print(f"❌ Forward pass error: {e}")
                    continue

                try:
                    spec_loss = complex_spectrogram_loss(spec_in, recon_spec) if missing_modality != 'spec' else 0.0
                    mask_loss = robust_mask_loss(mask_in, recon_mask) if missing_modality != 'mask' else 0.0
                except Exception as e:
                    print(f"❌ Loss computation error: {e}")
                    continue

                if missing_modality is None:
                    recon_loss = spec_weight * spec_loss + mask_weight * mask_loss
                    epoch_metrics['n_full'] += 1
                    epoch_metrics['train_full'] += float(recon_loss)
                elif missing_modality == 'spec':
                    recon_loss = mask_loss
                    epoch_metrics['n_mask_only'] += 1
                    epoch_metrics['train_mask_only'] += float(recon_loss)
                else:
                    recon_loss = spec_loss
                    epoch_metrics['n_spec_only'] += 1
                    epoch_metrics['train_spec_only'] += float(recon_loss)

                total_loss = recon_loss + beta * js_div

                if tf.math.is_nan(total_loss) or tf.math.is_inf(total_loss):
                    print("❌ WARNING: Invalid loss detected! Skipping gradient update.")
                    continue

            try:
                grads = tape.gradient(total_loss, model.trainable_variables)
                clipped_grads, grad_norm = tf.clip_by_global_norm(grads, 5.0)
                optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
            except Exception as e:
                print(f"❌ Gradient computation error: {e}")
                continue

            epoch_metrics['train_total'] += float(total_loss)
            epoch_metrics['train_spec'] += float(spec_loss) if missing_modality != 'spec' else 0.0
            epoch_metrics['train_mask'] += float(mask_loss) if missing_modality != 'mask' else 0.0
            epoch_metrics['train_js'] += float(js_div)
            epoch_metrics['train_steps'] += 1

        if epoch_metrics['train_steps'] > 0:
            metrics['train_total'].append(epoch_metrics['train_total'] / epoch_metrics['train_steps'])
            metrics['train_spec'].append(epoch_metrics['train_spec'] / max(epoch_metrics['train_steps'] - epoch_metrics['n_mask_only'], 1))
            metrics['train_mask'].append(epoch_metrics['train_mask'] / max(epoch_metrics['train_steps'] - epoch_metrics['n_spec_only'], 1))
            metrics['train_js'].append(epoch_metrics['train_js'] / epoch_metrics['train_steps'])
            metrics['train_mode'].append({
                'full': epoch_metrics['train_full'] / max(epoch_metrics['n_full'], 1),
                'spec_only': epoch_metrics['train_spec_only'] / max(epoch_metrics['n_spec_only'], 1),
                'mask_only': epoch_metrics['train_mask_only'] / max(epoch_metrics['n_mask_only'], 1)
            })

            print(f"✅ Train Loss: {metrics['train_total'][-1]:.4f} | "
                  f"Spec: {metrics['train_spec'][-1]:.4f} | "
                  f"Mask: {metrics['train_mask'][-1]:.4f} | "
                  f"JS: {metrics['train_js'][-1]:.4f}")

        # Validation loop
        epoch_val_metrics = {'total': 0.0, 'spec': 0.0, 'mask': 0.0, 'js': 0.0, 'steps': 0}
        for step, (spec_in, mask_in, test_id_in) in enumerate(val_dataset):
            recon_spec, recon_mask, (all_mus, all_logvars, mixture_prior, js_div) = model(
                spec_in, mask_in, test_id_in, training=False, missing_modality=None)

            spec_loss = complex_spectrogram_loss(spec_in, recon_spec)
            mask_loss = robust_mask_loss(mask_in, recon_mask)
            recon_loss = spec_weight * spec_loss + mask_weight * mask_loss
            total_loss = recon_loss + beta * js_div

            epoch_val_metrics['total'] += float(total_loss)
            epoch_val_metrics['spec'] += float(spec_loss)
            epoch_val_metrics['mask'] += float(mask_loss)
            epoch_val_metrics['js'] += float(js_div)
            epoch_val_metrics['steps'] += 1

        if epoch_val_metrics['steps'] > 0:
            metrics['val_total'].append(epoch_val_metrics['total'] / epoch_val_metrics['steps'])
            metrics['val_spec'].append(epoch_val_metrics['spec'] / epoch_val_metrics['steps'])
            metrics['val_mask'].append(epoch_val_metrics['mask'] / epoch_val_metrics['steps'])
            metrics['val_js'].append(epoch_val_metrics['js'] / epoch_val_metrics['steps'])
            print(f"  🔵 Val => Total: {metrics['val_total'][-1]:.4f} | "
                  f"Spec: {metrics['val_spec'][-1]:.4f} | "
                  f"Mask: {metrics['val_mask'][-1]:.4f} | "
                  f"JS: {metrics['val_js'][-1]:.4f}")
        
        tf.keras.backend.clear_session()
        gc.collect()

        current_val_loss = metrics['val_total'][-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            no_improvement_count = 0
            model.save_weights("results/best_spectral_mmvae.weights.h5")
            print("✅ Saved best model weights")
        else:
            no_improvement_count += 1
            print(f"🚨 No improvement for {no_improvement_count}/{patience} epochs.")

        if no_improvement_count >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.")
            model.save_weights("results/final_spectral_mmvae.weights.h5")
            print("✅ Saved final model weights")
            break

    tf.keras.backend.clear_session()
    gc.collect()

    return metrics

# ----- Visualization Functions and tests -----
def plot_training_curves(metrics):
    """
    Plot training curves with improved visualization for MoE model.
    
    Args:
        metrics: Dictionary of training and validation metrics
    """
    epochs = list(range(1, len(metrics['train_total']) + 1))
    
    # Create output directory if it doesn't exist
    os.makedirs("results/plots", exist_ok=True)
    
    # 1. Total Loss
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_total'], mode='lines+markers', name="Train Total", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['val_total'], mode='lines+markers', name="Val Total", line=dict(color='red')))
    fig.update_layout(
        title="Total Loss vs Epochs", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        template="plotly_white"
    )
    pio.write_html(fig, file="results/plots/train_val_total_loss.html", auto_open=False)
    
    # 2. Spectrogram & Descriptor Losses
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_spec'], mode='lines+markers', name="Train Spec", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['val_spec'], mode='lines+markers', name="Val Spec", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_desc'], mode='lines+markers', name="Train Desc", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['val_desc'], mode='lines+markers', name="Val Desc", line=dict(color='orange')))
    fig.update_layout(
        title="Modality Losses vs Epochs", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        template="plotly_white"
    )
    pio.write_html(fig, file="results/plots/train_val_modality_loss.html", auto_open=False)
    
    # 3. JS Divergence
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_js'], mode='lines+markers', name="Train JS", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['val_js'], mode='lines+markers', name="Val JS", line=dict(color='red')))
    fig.update_layout(
        title="JS Divergence vs Epochs", 
        xaxis_title="Epoch", 
        yaxis_title="JS Divergence",
        template="plotly_white"
    )
    pio.write_html(fig, file="results/plots/train_val_js_div.html", auto_open=False)
    
    # 4. Missing Modality Performance (if available)
    if 'train_mode' in metrics:
        fig = go.Figure()
        full_losses = [mode_loss['full'] for mode_loss in metrics['train_mode']]
        spec_only_losses = [mode_loss['spec_only'] for mode_loss in metrics['train_mode']]
        mask_only_losses = [mode_loss['mask_only'] for mode_loss in metrics['train_mode']]
        
        fig.add_trace(go.Scatter(x=epochs, y=full_losses, mode='lines+markers', name="Both Modalities", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=epochs, y=spec_only_losses, mode='lines+markers', name="Spec Only", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=epochs, y=mask_only_losses, mode='lines+markers', name="Desc Only", line=dict(color='red')))
        
        fig.update_layout(
            title="Reconstruction Loss by Modality Combination", 
            xaxis_title="Epoch", 
            yaxis_title="Recon Loss",
            template="plotly_white"
        )
        pio.write_html(fig, file="results/plots/modality_combo_loss.html", auto_open=False)

def visualize_reconstructions(model, val_dataset, data_loader, num_samples=5, fs=200, nperseg=256, noverlap=128):
    """
    Visualize original vs. reconstructed spectrograms and descriptors/masks.
    
    Args:
        model: Trained SpectralMMVAE model
        val_dataset: Validation dataset
        data_loader: Data loader module with reconstruct_mask_from_descriptors function
        num_samples: Number of samples to visualize
        fs, nperseg, noverlap: Parameters for STFT
    """
    os.makedirs("results/visualizations", exist_ok=True)
    
    # Select samples from validation dataset
    for i, (spec_batch, mask_batch, _) in enumerate(val_dataset.take(1)):
        for j in range(min(num_samples, spec_batch.shape[0])):
            # Get sample
            spec_sample = tf.expand_dims(spec_batch[j], 0)
            mask_sample = tf.expand_dims(mask_batch[j], 0)
            
            # Get reconstructions
            recon_spec, recon_desc, _ = model(
                spec_sample, mask_sample, 
                tf.constant([[0]]), training=False
            )
            
            # Visualize spectrograms (channel 0 only for clarity)
            visualize_spectrogram(
                spec_sample.numpy(), 
                title=f"Original Spectrogram (Sample {j+1}, Channel 0)", 
                channel=0,
                output_path=f"results/visualizations/orig_spec_sample_{j+1}.html"
            )
            
            visualize_spectrogram(
                recon_spec.numpy(), 
                title=f"Reconstructed Spectrogram (Sample {j+1}, Channel 0)", 
                channel=0,
                output_path=f"results/visualizations/recon_spec_sample_{j+1}.html"
            )
            
            # Reconstruct time series from spectrograms
            orig_ts = model.reconstruct_time_series(
                spec_sample.numpy(), fs=fs, nperseg=nperseg, noverlap=noverlap
            )
            
            recon_ts = model.reconstruct_time_series(
                recon_spec.numpy(), fs=fs, nperseg=nperseg, noverlap=noverlap
            )
            
            # Plot time series (mean over channels)
            orig_ts_mean = np.mean(orig_ts[0], axis=1)
            recon_ts_mean = np.mean(recon_ts[0], axis=1)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=orig_ts_mean, mode='lines', name='Original', 
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                y=recon_ts_mean, mode='lines', name='Reconstructed', 
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f"Time Series Reconstruction Sample {j+1}",
                xaxis_title="Time",
                yaxis_title="Amplitude"
            )
            
            fig.write_html(f"results/visualizations/ts_recon_sample_{j+1}.html")
            
            # Convert descriptors to masks for visualization using data_loader function
            try:
                # Find valid descriptors (non-zero rows)
                mask_np = mask_sample.numpy()[0]
                valid_mask = np.any(mask_np != 0, axis=1)
                valid_descriptors = mask_np[valid_mask]
                
                recon_mask_np = recon_desc.numpy()[0]
                valid_recon_mask = np.any(recon_mask_np != 0, axis=1)
                valid_recon_descriptors = recon_mask_np[valid_recon_mask]
                
                # Use the data_loader function to reconstruct masks
                orig_mask = data_loader.reconstruct_mask_from_descriptors(
                    valid_descriptors, 
                    image_shape=(256, 768), 
                    line_thickness=1
                )
                
                recon_mask = data_loader.reconstruct_mask_from_descriptors(
                    valid_recon_descriptors, 
                    image_shape=(256, 768), 
                    line_thickness=1
                )
                
                # Visualize masks
                fig = go.Figure()
                fig.add_trace(go.Heatmap(
                    z=recon_mask, 
                    colorscale='Reds',
                    showscale=False
                ))
                fig.update_layout(title=f"Reconstructed Mask from Descriptors - Sample {j+1}")
                fig.write_html(f"results/visualizations/recon_mask_sample_{j+1}.html")
                
            except Exception as e:
                print(f"Error visualizing masks for sample {j+1}: {e}")

def extract_latent_representations(model, dataset):
    """
    Extract latent representations from the model.
    
    Args:
        model: Trained SpectralMMVAE model
        dataset: Dataset to encode
        
    Returns:
        Tuple of (latent_vectors, test_ids)
    """
    latent_vectors = []
    test_ids = []

    for spec_in, mask_in, test_id_in in dataset:
        # Convert test_id_in to proper format
        if isinstance(test_id_in, tf.Tensor):
            test_id_in = test_id_in.numpy()
        if isinstance(test_id_in, np.ndarray):
            test_id_in = test_id_in.flatten().tolist()

        # Get latent representations from spectrogram encoder (could also use descriptor encoder)
        mu_q, _ = model.spec_encoder(spec_in, training=False)
        
        latent_vectors.append(mu_q.numpy())
        test_ids.append(test_id_in)

    latent_vectors = np.concatenate(latent_vectors, axis=0)

    # Check for NaN values
    if np.isnan(latent_vectors).any():
        print("⚠️ Warning: NaN values detected in latent vectors.")
        # Replace NaN with zeros to allow visualization
        latent_vectors = np.nan_to_num(latent_vectors)

    return latent_vectors, np.concatenate(test_ids, axis=0)

def reduce_latent_dim_umap(latent_vectors):
    """
    Reduces latent space dimensionality to 3D using UMAP.
    
    Args:
        latent_vectors: Latent vectors to reduce
        
    Returns:
        3D UMAP projection
    """
    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=100)
    latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)  # Flatten to 2D
    latent_3d = reducer.fit_transform(latent_vectors)
    return latent_3d

def plot_latent_space_3d(latent_3d, test_ids, output_file="latent_space.html"):
    """
    Plots and saves a 3D UMAP visualization of the latent space with a continuous color gradient over Test ID.
    
    Args:
        latent_3d: 3D UMAP projection
        test_ids: Test IDs for coloring
        output_file: Output file path
    """
    # Create a Pandas DataFrame for Plotly
    df = pd.DataFrame(latent_3d, columns=["UMAP_1", "UMAP_2", "UMAP_3"])
    df["Test ID"] = test_ids  # Store original Test IDs

    # Ensure Test ID is treated as a continuous variable
    df["Test ID"] = pd.to_numeric(df["Test ID"], errors="coerce")  # Convert to numeric in case of issues

    # Print some debug information
    print(f"Test ID min: {df['Test ID'].min()}, max: {df['Test ID'].max()}")
    print(f"Unique Test ID count: {df['Test ID'].nunique()}")

    # Ensure color is treated as continuous
    fig = px.scatter_3d(
        df, x="UMAP_1", y="UMAP_2", z="UMAP_3",
        color=df["Test ID"],  # Use actual Test ID values to ensure a gradient
        color_continuous_scale="Viridis",  # Continuous color scale
        title="Latent Space Visualization (3D UMAP)",
        opacity=0.8
    )

    # Save as interactive HTML
    pio.write_html(fig, file=output_file, auto_open=True)
    print(f"✅ 3D UMAP plot saved as {output_file}")

def generate_samples(model, data_loader, num_samples=5, fs=200, nperseg=256, noverlap=128):
    """
    Generate random samples from the model.
    
    Args:
        model: Trained SpectralMMVAE model
        data_loader: Data loader module with reconstruct_mask_from_descriptors function
        num_samples: Number of samples to generate
        fs, nperseg, noverlap: Parameters for STFT
    """
    os.makedirs("results/generated", exist_ok=True)
    
    for i in range(num_samples):
        # Sample from the prior
        z = tf.random.normal(shape=(1, model.latent_dim))
        
        # Generate spectrograms and descriptors
        gen_spec = model.generate(modality='spec', conditioning_latent=z)
        gen_desc = model.generate(modality='desc', conditioning_latent=z)
        
        # Visualize generated spectrogram
        visualize_spectrogram(
            gen_spec.numpy(), 
            title=f"Generated Spectrogram (Sample {i+1}, Channel 0)", 
            channel=0,
            output_path=f"results/generated/gen_spec_sample_{i+1}.html"
        )
        
        # Reconstruct time series from spectrogram
        gen_ts = model.reconstruct_time_series(
            gen_spec.numpy(), fs=fs, nperseg=nperseg, noverlap=noverlap
        )
        
        # Plot generated time series (mean over channels)
        gen_ts_mean = np.mean(gen_ts[0], axis=1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=gen_ts_mean, mode='lines',
            line=dict(color='purple')
        ))
        
        fig.update_layout(
            title=f"Generated Time Series Sample {i+1}",
            xaxis_title="Time",
            yaxis_title="Amplitude"
        )
        
        fig.write_html(f"results/generated/gen_ts_sample_{i+1}.html")
        
        # Convert descriptors to mask for visualization
        try:
            # Find valid descriptors (non-zero rows)
            gen_mask_np = gen_desc.numpy()[0]
            valid_mask = np.any(gen_mask_np != 0, axis=1)
            valid_descriptors = gen_mask_np[valid_mask]
            
            # Use the data_loader function to reconstruct mask
            if len(valid_descriptors) > 0:
                gen_mask = data_loader.reconstruct_mask_from_descriptors(
                    valid_descriptors, 
                    image_shape=(256, 768), 
                    line_thickness=1
                )
                
                # Visualize mask
                fig = go.Figure()
                fig.add_trace(go.Heatmap(
                    z=gen_mask, 
                    colorscale='Viridis',
                    showscale=False
                ))
                fig.update_layout(title=f"Generated Mask from Descriptors - Sample {i+1}")
                fig.write_html(f"results/generated/gen_mask_sample_{i+1}.html")
            else:
                print(f"No valid descriptors in generated sample {i+1}")
                
        except Exception as e:
            print(f"Error visualizing generated mask for sample {i+1}: {e}")

def visualize_latent_structure(model, dataset, n_samples=500):
    """
    Visualize the latent space structure of the MoE model.
    Compares modality-specific encodings and the mixture prior.
    
    Args:
        model: The trained MoE MMVAE model
        dataset: Dataset to visualize
        n_samples: Number of samples to use
    """
    # Create output directory
    os.makedirs("results/latent_analysis", exist_ok=True)
    
    # Collect latent encodings
    spec_mus = []
    mask_mus = []
    mixture_mus = []
    test_ids = []
    
    # Limit to n_samples
    sample_count = 0
    for spec_in, mask_in, test_id_in in dataset:
        if sample_count >= n_samples:
            break
            
        # Encode all modalities
        mus, logvars, mixture_mu, mixture_logvar = model.encode_all_modalities(spec_in, mask_in, training=False)
        
        # Store means
        spec_mus.append(mus[0].numpy())
        mask_mus.append(mus[1].numpy())
        mixture_mus.append(mixture_mu.numpy())
        
        # Store test IDs
        if isinstance(test_id_in, tf.Tensor):
            test_id_in = test_id_in.numpy()
        test_ids.append(test_id_in)
        
        sample_count += spec_in.shape[0]
    
    # Concatenate results
    spec_mus = np.concatenate(spec_mus, axis=0)
    mask_mus = np.concatenate(mask_mus, axis=0)
    mixture_mus = np.concatenate(mixture_mus, axis=0)
    test_ids = np.concatenate(test_ids, axis=0).flatten()
    
    # Reduce dimensionality for visualization
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    
    # Combine all encodings for a single UMAP embedding
    all_mus = np.concatenate([spec_mus, mask_mus, mixture_mus], axis=0)
    all_embeddings = reducer.fit_transform(all_mus)
    
    # Split back into modality-specific embeddings
    n_points = spec_mus.shape[0]
    spec_embeddings = all_embeddings[:n_points]
    mask_embeddings = all_embeddings[n_points:2*n_points]
    mixture_embeddings = all_embeddings[2*n_points:]
    
    # Visualize the latent space
    fig = go.Figure()
    
    # Create a color scale based on test IDs
    # Add spectrogram embeddings
    fig.add_trace(go.Scatter(
        x=spec_embeddings[:, 0], 
        y=spec_embeddings[:, 1],
        mode='markers',
        marker=dict(
            color=test_ids,
            colorscale='Viridis',
            size=8,
            symbol='circle',
            line=dict(width=1, color='DarkSlateGrey')
        ),
        name='Spectrogram Encoder'
    ))
    
    # Add descriptor embeddings
    fig.add_trace(go.Scatter(
        x=mask_embeddings[:, 0], 
        y=mask_embeddings[:, 1],
        mode='markers',
        marker=dict(
            color=test_ids,
            colorscale='Viridis',
            size=8,
            symbol='x',
            line=dict(width=1, color='DarkSlateGrey')
        ),
        name='Descriptor Encoder'
    ))
    
    # Add mixture embeddings
    fig.add_trace(go.Scatter(
        x=mixture_embeddings[:, 0], 
        y=mixture_embeddings[:, 1],
        mode='markers',
        marker=dict(
            color=test_ids,
            colorscale='Viridis',
            size=8,
            symbol='diamond',
            line=dict(width=1, color='DarkSlateGrey')
        ),
        name='Mixture Prior'
    ))
    
    # Update layout
    fig.update_layout(
        title="Latent Space Structure of Mixture-of-Experts MMVAE",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend_title="Encoder Type",
        template="plotly_white"
    )
    
    # Save figure
    pio.write_html(fig, file="results/latent_analysis/latent_structure.html", auto_open=False)
    
    return {
        'spec_mus': spec_mus,
        'mask_mus': mask_mus,
        'mixture_mus': mixture_mus,
        'test_ids': test_ids
    }

def evaluate_cross_modal_generation(model, dataset, data_loader, n_samples=10):
    """
    Evaluate cross-modal generation capabilities of the MoE model.
    
    Args:
        model: The trained MoE MMVAE model
        dataset: Dataset to use for evaluation
        data_loader: Data loader module with utility functions
        n_samples: Number of samples to evaluate
    """
    # Create output directory
    os.makedirs("results/cross_modal", exist_ok=True)
    
    # Get a batch of samples
    sample_count = 0
    for spec_in, mask_in, test_id_in in dataset:
        if sample_count >= n_samples:
            break
            
        # Select a few samples from the batch
        for i in range(min(n_samples - sample_count, spec_in.shape[0])):
            # Extract single sample
            spec_sample = tf.expand_dims(spec_in[i], 0)
            mask_sample = tf.expand_dims(mask_in[i], 0)
            test_id = test_id_in[i].numpy() if isinstance(test_id_in, tf.Tensor) else test_id_in[i]
            
            # 1. Encode using spectrogram only
            mu_spec, logvar_spec = model.spec_encoder(spec_sample, training=False)
            z_spec = reparameterize(mu_spec, logvar_spec)
            
            # Generate descriptor from spectrogram encoding
            gen_mask_from_spec = model.mask_decoder(z_spec, training=False)
            
            # 2. Encode using descriptor only
            mu_desc, logvar_desc = model.mask_encoder(mask_sample, training=False)
            z_desc = reparameterize(mu_desc, logvar_desc)
            
            # Generate spectrogram from descriptor encoding
            gen_spec_from_desc = model.spec_decoder(z_desc, training=False)
            
            # 3. Encode using both to get mixture prior
            mus, logvars, mixture_mu, mixture_logvar = model.encode_all_modalities(spec_sample, mask_sample, training=False)
            z_mixture = reparameterize(mixture_mu, mixture_logvar)
            
            # Generate from mixture encoding
            gen_spec_from_mixture = model.spec_decoder(z_mixture, training=False)
            gen_mask_from_mixture = model.mask_decoder(z_mixture, training=False)
            
            # Visualize original and cross-modal reconstructions
            # A. Original spectrogram vs generated from descriptor/mixture
            fig = make_subplots(rows=1, cols=3, subplot_titles=[
                "Original Spectrogram", 
                "Generated from Descriptor", 
                "Generated from Mixture"
            ])
            
            # Plot original spectrogram
            visualize_spectrogram(
                spec_sample.numpy(), 
                title=f"Original Spectrogram (ID: {test_id})", 
                channel=0,
                output_path=f"results/cross_modal/sample_{sample_count+i}_orig_spec.html"
            )
            
            # Plot generated spectrograms
            visualize_spectrogram(
                gen_spec_from_desc.numpy(), 
                title=f"Spec from Desc (ID: {test_id})", 
                channel=0,
                output_path=f"results/cross_modal/sample_{sample_count+i}_spec_from_desc.html"
            )
            
            visualize_spectrogram(
                gen_spec_from_mixture.numpy(), 
                title=f"Spec from Mixture (ID: {test_id})", 
                channel=0,
                output_path=f"results/cross_modal/sample_{sample_count+i}_spec_from_mixture.html"
            )
            
            # B. Compare crack descriptors
            try:
                # Original descriptor
                mask_np = mask_sample.numpy()[0]
                valid_mask = np.any(mask_np != 0, axis=1)
                valid_descriptors = mask_np[valid_mask]
                
                # Generated from spectrogram
                gen_mask_from_spec_np = gen_mask_from_spec.numpy()[0]
                valid_gen_mask = np.any(gen_mask_from_spec_np != 0, axis=1)
                valid_gen_descriptors = gen_mask_from_spec_np[valid_gen_mask]
                
                # Generated from mixture
                gen_mask_from_mixture_np = gen_mask_from_mixture.numpy()[0]
                valid_mix_mask = np.any(gen_mask_from_mixture_np != 0, axis=1)
                valid_mix_descriptors = gen_mask_from_mixture_np[valid_mix_mask]
                
                # Use data_loader to visualize masks
                orig_mask = data_loader.reconstruct_mask_from_descriptors(
                    valid_descriptors, 
                    image_shape=(256, 768), 
                    line_thickness=1
                )
                
                gen_mask = data_loader.reconstruct_mask_from_descriptors(
                    valid_gen_descriptors, 
                    image_shape=(256, 768), 
                    line_thickness=1
                )
                
                mix_mask = data_loader.reconstruct_mask_from_descriptors(
                    valid_mix_descriptors, 
                    image_shape=(256, 768), 
                    line_thickness=1
                )
                
                # Visualize masks
                fig = go.Figure()
                fig.add_trace(go.Heatmap(
                    z=orig_mask, 
                    colorscale='Reds',
                    showscale=False
                ))
                fig.update_layout(title=f"Original Mask (ID: {test_id})")
                fig.write_html(f"results/cross_modal/sample_{sample_count+i}_orig_mask.html", auto_open=False)
                
                fig = go.Figure()
                fig.add_trace(go.Heatmap(
                    z=gen_mask, 
                    colorscale='Blues',
                    showscale=False
                ))
                fig.update_layout(title=f"Mask from Spectrogram (ID: {test_id})")
                fig.write_html(f"results/cross_modal/sample_{sample_count+i}_mask_from_spec.html", auto_open=False)
                
                fig = go.Figure()
                fig.add_trace(go.Heatmap(
                    z=mix_mask, 
                    colorscale='Greens',
                    showscale=False
                ))
                fig.update_layout(title=f"Mask from Mixture (ID: {test_id})")
                fig.write_html(f"results/cross_modal/sample_{sample_count+i}_mask_from_mixture.html", auto_open=False)
                
            except Exception as e:
                print(f"Error visualizing masks for sample {sample_count+i}: {e}")
            
            sample_count += 1
            if sample_count >= n_samples:
                break

def evaluate_missing_modality(model, dataset, n_samples=10):
    """
    Evaluate model performance when modalities are missing.
    
    Args:
        model: The trained MoE MMVAE model
        dataset: Dataset to use for evaluation
        n_samples: Number of samples to evaluate
    """
    # Create output directory
    os.makedirs("results/missing_modality", exist_ok=True)
    
    # Metrics to track
    spec_full_losses = []
    spec_miss_losses = []
    mask_full_losses = []
    mask_miss_losses = []
    
    # Get samples
    sample_count = 0
    for spec_in, mask_in, _ in dataset:
        if sample_count >= n_samples:
            break
            
        # Get full reconstructions
        recon_spec_full, recon_mask_full, _ = model(
            spec_in, mask_in,
            training=False,
            missing_modality=None  # No missing modality
        )
        
        # Get reconstructions with missing spectrogram
        recon_spec_miss_spec, recon_mask_miss_spec, _ = model(
            spec_in, mask_in,
            training=False,
            missing_modality='spec'  # Missing spectrogram
        )
        
        # Get reconstructions with missing descriptor
        recon_spec_miss_desc, recon_mask_miss_desc, _ = model(
            spec_in, mask_in,
            training=False,
            missing_modality='desc'  # Missing descriptor
        )
        
        # Compute losses
        spec_full_loss = complex_spectrogram_loss(spec_in, recon_spec_full).numpy()
        spec_miss_loss = complex_spectrogram_loss(spec_in, recon_spec_miss_spec).numpy()
        
        mask_full_loss = weighted_descriptor_mse_loss(mask_in, recon_mask_full).numpy()
        mask_miss_loss = weighted_descriptor_mse_loss(mask_in, recon_mask_miss_desc).numpy()
        
        # Track metrics
        spec_full_losses.append(spec_full_loss)
        spec_miss_losses.append(spec_miss_loss)
        mask_full_losses.append(mask_full_loss)
        mask_miss_losses.append(mask_miss_loss)
        
        sample_count += spec_in.shape[0]
    
    # Compute averages
    avg_spec_full = np.mean(spec_full_losses)
    avg_spec_miss = np.mean(spec_miss_losses)
    avg_mask_full = np.mean(mask_full_losses)
    avg_mask_miss = np.mean(mask_miss_losses)
    
    # Create summary table
    data = {
        'Modality': ['Spectrogram', 'Descriptor'],
        'Full Model': [avg_spec_full, avg_mask_full],
        'Missing Other Modality': [avg_spec_miss, avg_mask_miss],
        'Performance Drop (%)': [
            (avg_spec_miss - avg_spec_full) / avg_spec_full * 100,
            (avg_mask_miss - avg_mask_full) / avg_mask_full * 100
        ]
    }
    df = pd.DataFrame(data)
    
    # Save results
    df.to_csv("results/missing_modality/performance_summary.csv", index=False)
    
    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Spectrogram', 'Descriptor'],
        y=[avg_spec_full, avg_mask_full],
        name='Full Model'
    ))
    fig.add_trace(go.Bar(
        x=['Spectrogram', 'Descriptor'],
        y=[avg_spec_miss, avg_mask_miss],
        name='Missing Other Modality'
    ))
    
    fig.update_layout(
        title="Reconstruction Performance with Missing Modalities",
        xaxis_title="Reconstructed Modality",
        yaxis_title="Reconstruction Loss",
        barmode='group',
        template="plotly_white"
    )
    
    pio.write_html(fig, file="results/missing_modality/performance_comparison.html", auto_open=False)
    
    return {
        'spec_full': avg_spec_full,
        'spec_miss': avg_spec_miss,
        'mask_full': avg_mask_full,
        'mask_miss': avg_mask_miss
    }

def evaluate_modality_alignment(latent_data):
    """
    Evaluate how well modalities are aligned in the latent space.
    
    Args:
        latent_data: Dictionary with latent encodings from visualize_latent_structure
        
    Returns:
        Dictionary of alignment metrics
    """
    # Create output directory
    os.makedirs("results/latent_analysis", exist_ok=True)
    
    # Extract data
    spec_mus = latent_data['spec_mus']
    mask_mus = latent_data['mask_mus']
    mixture_mus = latent_data['mixture_mus']
    
    # 1. Compute average cosine distance between modality encodings
    # This measures how aligned the encodings are in direction
    
    # Normalize vectors for cosine similarity
    spec_norm = spec_mus / np.linalg.norm(spec_mus, axis=1, keepdims=True)
    mask_norm = mask_mus / np.linalg.norm(mask_mus, axis=1, keepdims=True)
    
    # Compute batch-wise cosine similarity
    cosine_similarities = np.sum(spec_norm * mask_norm, axis=1)
    avg_cosine_sim = np.mean(cosine_similarities)
    
    # 2. Compute Euclidean distances between modality encodings
    euclidean_distances = np.sqrt(np.sum(np.square(spec_mus - mask_mus), axis=1))
    avg_euclidean_dist = np.mean(euclidean_distances)
    
    # 3. Compute distances between each modality and the mixture
    spec_to_mixture_dist = np.sqrt(np.sum(np.square(spec_mus - mixture_mus), axis=1))
    mask_to_mixture_dist = np.sqrt(np.sum(np.square(mask_mus - mixture_mus), axis=1))
    
    avg_spec_to_mixture = np.mean(spec_to_mixture_dist)
    avg_mask_to_mixture = np.mean(mask_to_mixture_dist)
    
    # Create summary of results
    alignment_metrics = {
        'avg_cosine_similarity': avg_cosine_sim,
        'avg_euclidean_distance': avg_euclidean_dist,
        'avg_spec_to_mixture_distance': avg_spec_to_mixture,
        'avg_mask_to_mixture_distance': avg_mask_to_mixture
    }
    
    # Save metrics
    pd.DataFrame([alignment_metrics]).to_csv("results/latent_analysis/alignment_metrics.csv", index=False)
    
    # Create visualization of distances
    fig = go.Figure()
    
    # Add cosine similarity histogram
    fig.add_trace(go.Histogram(
        x=cosine_similarities,
        name='Cosine Similarity',
        histnorm='probability density',
        marker_color='blue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Distribution of Cosine Similarity Between Modality Encodings",
        xaxis_title="Cosine Similarity",
        yaxis_title="Probability Density",
        template="plotly_white"
    )
    
    pio.write_html(fig, file="results/latent_analysis/cosine_similarity_dist.html", auto_open=False)
    
    # Create visualization of modality to mixture distances
    fig = go.Figure()
    
    # Add distance histograms
    fig.add_trace(go.Histogram(
        x=spec_to_mixture_dist,
        name='Spectrogram to Mixture',
        histnorm='probability density',
        marker_color='blue',
        opacity=0.7
    ))
    
    fig.add_trace(go.Histogram(
        x=mask_to_mixture_dist,
        name='Descriptor to Mixture',
        histnorm='probability density',
        marker_color='green',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Distribution of Distances Between Modality and Mixture Encodings",
        xaxis_title="Euclidean Distance",
        yaxis_title="Probability Density",
        template="plotly_white",
        barmode='overlay'
    )
    
    pio.write_html(fig, file="results/latent_analysis/modality_mixture_dist.html", auto_open=False)
    
    return alignment_metrics

def visualize_spectrogram(spectrogram_features, title="Spectrogram", channel=0, output_path=None):
    """
    Visualize a single channel from spectrogram features.
    
    Args:
        spectrogram_features: Spectrogram features with shape (batch, freq, time, channels*2)
        title: Plot title
        channel: Which original channel to visualize (magnitude only)
        output_path: Path to save the visualization
    """
    # Extract magnitude for selected channel
    magnitude = spectrogram_features[0, :, :, channel*2]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=magnitude,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Frames",
        yaxis_title="Frequency Bins",
        yaxis=dict(autorange="reversed"),  # Flip y-axis to have low frequencies at bottom
        template="plotly_white"
    )
    
    if output_path:
        pio.write_html(fig, file=output_path, auto_open=False)
    else:
        fig.show()

def test_latent_interpolation(model, val_dataset, output_dir="results/interpolation"):
    """
    Perform latent space interpolation between samples in the validation set.
    
    Args:
        model: Trained SpectralMMVAE model
        output_dir: Directory to save interpolation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get two samples from validation dataset
    for spec_batch, mask_batch, _ in val_dataset.take(1):
        # Use first two samples in batch
        if spec_batch.shape[0] < 2:
            print("❌ Need at least 2 samples for interpolation")
            return
            
        source_spec = tf.expand_dims(spec_batch[0], 0)
        source_mask = tf.expand_dims(mask_batch[0], 0)
        
        target_spec = tf.expand_dims(spec_batch[1], 0)
        target_mask = tf.expand_dims(mask_batch[1], 0)
        
        # Encode samples to get latent vectors
        _, _, source_mixture = model(source_spec, source_mask, training=False)
        _, _, target_mixture = model(target_spec, target_mask, training=False)
        
        # Get mixture means
        source_mus, source_logvars, _, _ = source_mixture
        target_mus, target_logvars, _, _ = target_mixture
        
        # Use spectrogram encoder means for interpolation
        source_z = source_mus[0]  # Assuming index 0 is spectrogram encoder
        target_z = target_mus[0]  # Assuming index 0 is spectrogram encoder
        
        # Create interpolations
        num_steps = 8
        alphas = np.linspace(0, 1, num_steps)
        
        # Generate and visualize interpolated spectrograms
        plt.figure(figsize=(num_steps*2, 4))
        
        for i, alpha in enumerate(alphas):
            # Linear interpolation in latent space
            interp_z = (1 - alpha) * source_z + alpha * target_z
            
            # Generate spectrogram from interpolated latent
            interp_spec = model.spec_decoder(interp_z, training=False)
            
            # Plot
            plt.subplot(1, num_steps, i+1)
            plt.imshow(interp_spec[0, :, :, 0].numpy(), aspect='auto', cmap='viridis')
            plt.title(f"α={alpha:.1f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "latent_interpolation.png"), dpi=300)
        plt.close()
        
        print("✅ Generated latent space interpolation visualization")
        break

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
    
    latent_dim = 64 
    batch_size = 32  
    num_epochs = 500 
    patience = 150  

    # Cached file paths
    stft_path  = "scripts/cached_stft.npy"
    final_path = "scripts/cached_spectral_features.npy"
    mask_path  = "scripts/cached_masks.npy"
    ids_path   = "scripts/cached_test_ids.npy"

    # ------------------------ 3) Single Check: Are ALL caches present? ------------------------
    all_cached = (
        os.path.exists(final_path)
        and os.path.exists(mask_path)
        and os.path.exists(ids_path)
    )

    if all_cached:
        # ------------------------ A) ALL Cached: Load everything ------------------------
        print("✅ All cached files found! Loading from disk...")

        spectral_features   = np.load(final_path)
        mask_segments       = np.load(mask_path)
        test_ids            = np.load(ids_path)

        print(f"✅ spectral_features: {spectral_features.shape}")
        print(f"✅ mask_segments: {mask_segments.shape}")
        print(f"✅ test_ids: {test_ids.shape}")

    else:
        # ------------------------ B) Missing Any Cache: Re-run entire pipeline ------------------------
        print("⚠️ At least one cache file is missing. Recomputing EVERYTHING from scratch...")

        # (1) Load raw data
        params = {
            'keypoint_count': 15,
            'max_gap': 3,
            'curved_threshold': 10,
            'curved_angle_threshold': 75,
            'straight_angle_threshold': 15,
            'min_segment_length': 2,
            'line_thickness': 1,
        }
        print("📥 Loading raw data...")
        accel_dict, crack_dict, binary_masks, skeletons, padded_dict = data_loader.load_data(params)

        mask_segments = np.array([np.expand_dims(binary_masks[k], -1) for k in sorted(binary_masks.keys())])
        test_ids = np.array(sorted(binary_masks.keys()))


        print(f"Loaded data for {len(accel_dict)} tests")

        # (2) Segment time series -> get raw_segments, masks, test_ids
        print("✂️ Segmenting data...")
        raw_segments, mask_segments, test_ids = segment_and_transform(accel_dict, binary_masks)
        mask_segments = np.expand_dims(mask_segments, axis=-1)
        print(f"✅ Extracted {len(raw_segments)} segments")

        # Save masks & test IDs for future use
        np.save(mask_path, mask_segments)
        np.save(ids_path, test_ids)
        print(f"✅ Saved masks to {mask_path}")
        print(f"✅ Saved test IDs to {ids_path}")

        # (3) Compute STFT -> complex spectrograms
        print("🔄 Computing spectrograms...")
        complex_specs = compute_and_cache_spectrograms(raw_segments, fs, nperseg, noverlap, cache_path=stft_path)

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
    spec_shape = spectral_features.shape[1:]
    mask_shape = (256, 768, 1)
    print(f"📐 Building model with:")
    print(f"  - Latent dimension: {latent_dim}")
    print(f"  - Spectrogram shape: {spec_shape}")
    print(f"  - Mask shape: ({mask_shape}")

    model = SpectralMMVAE(latent_dim, spec_shape, mask_shape)

    # Build with a dummy forward pass
    dummy_spec = tf.zeros((1, *spec_shape))
    dummy_mask = tf.zeros((1, *mask_shape))
    _ = model(dummy_spec, dummy_mask, training=True)
    print("✅ Model built successfully with dummy inputs")
    #get architecture overview
    model.summary()
    model.spec_encoder.summary()
    model.mask_encoder.summary()
    model.spec_decoder.summary()
    model.mask_decoder.summary()


    # ------------------------ 10) Set Up Optimizer ------------------------
    lr_schedule = ExponentialDecay(
        initial_learning_rate=5e-5,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
    )

    # lr_schedule = CosineDecayRestarts(
    #     initial_learning_rate=5e-5,
    #     first_decay_steps=10000,
    #     t_mul=2.0,         # increase period each time
    #     m_mul=1.0,         # keep same max LR on restart
    #     alpha=0.1          # min LR = 10% of max
    # )

    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6
    )

    # ------------------------ 11) Train Model ------------------------
    print("🚀 Starting training...")
    training_metrics = train_spectral_mmvae(
        model, 
        train_dataset, 
        val_dataset, 
        optimizer, 
        num_epochs=num_epochs, 
        patience=patience,
        beta_schedule='cyclical',
        modality_dropout_prob=0.05
    )

    # ------------------------ 12) Save & Visualize Training Results ------------------------
    np.save("results/training_metrics.npy", training_metrics)
    plot_training_curves(training_metrics)

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

    # ------------------------ 14) Evaluate Latent Space ------------------------
    latent_vectors, test_ids_arr = extract_latent_representations(model, train_dataset)
    latent_3d = reduce_latent_dim_umap(latent_vectors)
    plot_latent_space_3d(latent_3d, test_ids_arr, output_file="results/latent_space_3d.html")
    
    latent_data = visualize_latent_structure(model, val_dataset, n_samples=500)
    alignment_metrics = evaluate_modality_alignment(latent_data)
    print("Modality Alignment Metrics:")
    for metric, value in alignment_metrics.items():
        print(f"  - {metric}: {value:.4f}")

    # ------------------------ 15) Evaluate Reconstructions and Cross-Modal Generation ------------------------
    visualize_reconstructions(model, val_dataset, data_loader, num_samples=5, fs=fs, nperseg=nperseg, noverlap=noverlap)
    evaluate_cross_modal_generation(model, val_dataset, data_loader, n_samples=5)
    
    missing_modality_metrics = evaluate_missing_modality(model, val_dataset, n_samples=10)
    print("Missing Modality Performance:")
    print(f"  - Spectrogram (full): {missing_modality_metrics['spec_full']:.4f}")
    print(f"  - Spectrogram (missing mask): {missing_modality_metrics['spec_miss']:.4f}")
    print(f"  - Mask (full): {missing_modality_metrics['mask_full']:.4f}")
    print(f"  - Mask (missing mask): {missing_modality_metrics['mask_miss']:.4f}")

    # ------------------------ 16) Generate New Samples ------------------------
    generate_samples(model, data_loader, num_samples=5, fs=fs, nperseg=nperseg, noverlap=noverlap)
    test_latent_interpolation(model, val_dataset,"results/samples")

    print("✅ Training & evaluation complete. Results saved in 'results/' directory.")

if __name__ == "__main__":
    main()
