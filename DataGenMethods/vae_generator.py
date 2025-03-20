import os
import warnings
import sys
import tensorflow as tf

def configure_gpu():
    """Enables memory growth for GPUs and prints the available devices."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"✅ Memory growth enabled on {len(physical_devices)} GPU(s)")
        except RuntimeError as e:
            print(f"❌ GPU Memory Growth Error: {e}")

    # Check which device TensorFlow is using
    print(f"🔍 TensorFlow will run on: {tf.config.list_logical_devices('GPU')}")

# Call GPU configuration before importing any other TensorFlow-related code
configure_gpu()

# ✅ Now import all other libraries
import cv2
import sys
import gc
import time
import psutil
import GPUtil
import numpy as np
import keras 
from keras import layers, Model, optimizers
from keras.optimizers.schedules import ExponentialDecay #type: ignore
import logging
from scipy import signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import umap
import plotly.express as px
import pandas as pd
from sklearn.mixture import GaussianMixture

# Import our custom module instead of tensorflow_probability
from custom_distributions import compute_js_divergence, reparameterize

# Append parent directory to find data_loader.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    import psutil
    import GPUtil
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
    descriptor_dict,
    chunk_size=1, 
    sample_rate=200, 
    segment_duration=5.0, 
    percentile=99
):
    """
    Extracts segments from time series data centered around peak RMS values.
    Returns raw segments and their corresponding descriptors.
    
    Args:
        accel_dict: Dictionary mapping test IDs to time series data
        descriptor_dict: Dictionary mapping test IDs to descriptor data
        chunk_size: Number of test IDs to process in one batch
        sample_rate: Sampling rate of the time series data (Hz)
        segment_duration: Duration of each segment (seconds)
        percentile: Percentile threshold for peak detection
        
    Returns:
        Tuple of arrays: (raw_segments, descriptor_segments, test_ids)
    """
    window_size = int(sample_rate * segment_duration)
    half_window = window_size // 2
    
    test_ids = list(accel_dict.keys())
    
    # Instead of yield, we'll collect all data and return it at once
    all_raw_segments = []
    all_descriptor_segments = []
    all_test_ids = []
    
    for i in range(0, len(test_ids), chunk_size):
        chunk_ids = test_ids[i:i + chunk_size]
        
        for test_id in chunk_ids:
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
                    all_descriptor_segments.append(descriptor_dict[test_id])
                    all_test_ids.append(int(test_id))
    
    # Convert lists to numpy arrays
    raw_segments = np.array(all_raw_segments, dtype=np.float32)
    descriptor_segments = np.array(all_descriptor_segments, dtype=np.float32)
    test_ids = np.array(all_test_ids, dtype=np.int32)
    
    print(f"Extracted {len(raw_segments)} segments from {len(test_ids)} unique test IDs")
    
    return raw_segments, descriptor_segments, test_ids

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
        print(f"📂 Loading raw STFT from {cache_path} (mmap_mode='r')...")
        return np.load(cache_path, mmap_mode='r')
    
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
                
                # Trim to original length
                time_series[b_abs, :time_length, c] = inverse_stft[:time_length].numpy()
        
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
        print(f"📂 Loading final spectral features from {cache_path} (mmap_mode='r')...")
        return np.load(cache_path, mmap_mode='r')
    
    # Convert to final shape (mag+phase)
    print("⏳ Converting complex STFT -> final magnitude+phase features...")
    spectral_features = spectrogram_to_features(complex_specs)
    
    # Save the final shape
    np.save(cache_path, spectral_features)
    print(f"✅ Final spectral features saved to {cache_path}")

    # Load in mmap mode to reduce memory usage
    return np.load(cache_path, mmap_mode='r')

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
class DescriptorEncoder(tf.keras.Model):
    """
    Encoder for crack descriptor data.
    Input shape: (batch_size, max_num_cracks=770, desc_length=42)
    Output: Mean and logvariance for latent distribution.
    """
    def __init__(self, latent_dim):
        super().__init__()
        
        # First 1D convolutional layer
        self.conv1 = layers.Conv1D(64, 3, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.drop1 = layers.Dropout(0.3)
        
        # Second 1D convolutional layer
        self.conv2 = layers.Conv1D(128, 3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.drop2 = layers.Dropout(0.3)
        
        # Global pooling for variable input lengths
        self.global_pool = layers.GlobalMaxPooling1D()
        
        # Dense layers for encoding
        self.dense1 = layers.Dense(256, activation='relu')
        self.drop3 = layers.Dropout(0.3)
        
        # Latent projections
        self.mu_layer = layers.Dense(latent_dim)
        self.logvar_layer = layers.Dense(latent_dim)

    def call(self, x, training=False):
        """
        x shape: (batch, max_num_cracks, desc_length)
        """
        # Create a mask for valid descriptors (non-zero entries)
        # This assumes zeros indicate padding
        input_mask = tf.reduce_any(tf.not_equal(x, 0), axis=-1)
        input_mask = tf.cast(input_mask, dtype=tf.float32)
        input_mask = tf.expand_dims(input_mask, axis=-1)
        
        # Apply convolutions with masking to ignore padded entries
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = x * input_mask  # Apply mask
        x = self.drop1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = x * input_mask  # Apply mask
        x = self.drop2(x, training=training)
        
        # Global pooling across crack descriptors
        x = self.global_pool(x)
        
        # Dense encoding
        x = self.dense1(x)
        x = self.drop3(x, training=training)
        
        # Output distribution parameters
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
       
class DescriptorDecoder(tf.keras.Model):
    """
    Decoder for crack descriptor data.
    Input: latent vector z
    Output shape: (batch_size, max_num_cracks=770, desc_length=42)
    """
    def __init__(self, max_num_cracks=770, desc_length=42):
        super().__init__()
        self.max_num_cracks = max_num_cracks
        self.desc_length = desc_length
        
        # Project latent to initial dense representation
        self.fc1 = layers.Dense(256, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.drop1 = layers.Dropout(0.3)
        
        # Project to shape that can be reshaped to (batch, max_cracks/10, 10*hidden_dim)
        # We'll reshape and then use Conv1DTranspose to expand
        hidden_dim = 64
        self.reshaped_cracks = max_num_cracks // 10  # Compress by factor of 10
        self.fc2 = layers.Dense(self.reshaped_cracks * hidden_dim * 10, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.drop2 = layers.Dropout(0.3)
        
        # Reshape layer
        self.reshape = layers.Reshape((self.reshaped_cracks, hidden_dim * 10))
        
        # Upsampling with Conv1DTranspose
        self.conv_t1 = layers.Conv1DTranspose(128, 3, strides=2, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.drop3 = layers.Dropout(0.3)
        
        self.conv_t2 = layers.Conv1DTranspose(64, 3, strides=2, padding='same', activation='relu')
        self.bn4 = layers.BatchNormalization()
        self.drop4 = layers.Dropout(0.3)
        
        # Final projection to descriptor space
        self.conv_t3 = layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation='relu')
        self.bn5 = layers.BatchNormalization()
        self.drop5 = layers.Dropout(0.3)
        
        # Output layer - no activation as descriptors can have any range of values
        self.output_layer = layers.Conv1D(desc_length, 1, padding='same')

    def call(self, z, training=False):
        # Initial dense projection
        x = self.fc1(z)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)
        
        # Project to reshapable dimension
        x = self.fc2(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)
        
        # Reshape to begin 1D convolution sequence
        x = self.reshape(x)
        
        # Upsample with transposed convolutions
        x = self.conv_t1(x)
        x = self.bn3(x, training=training)
        x = self.drop3(x, training=training)
        
        x = self.conv_t2(x)
        x = self.bn4(x, training=training)
        x = self.drop4(x, training=training)
        
        x = self.conv_t3(x)
        x = self.bn5(x, training=training)
        x = self.drop5(x, training=training)
        
        # Final output projection to descriptor space
        x = self.output_layer(x)
        
        # Apply padding and truncation to match expected output shape
        current_shape = tf.shape(x)[1]
        if current_shape < self.max_num_cracks:
            # Pad with zeros if we have fewer than max_num_cracks
            paddings = [[0, 0], [0, self.max_num_cracks - current_shape], [0, 0]]
            x = tf.pad(x, paddings)
        elif current_shape > self.max_num_cracks:
            # Truncate if we have more than max_num_cracks
            x = x[:, :self.max_num_cracks, :]
        
        return x

# ----- Loss Functions -----
def weighted_descriptor_mse_loss(y_true, y_pred):
    """
    MSE loss for descriptor reconstruction that only considers non-zero (valid) elements.
    
    Args:
        y_true: Ground truth descriptor array [batch, max_cracks, desc_length]
        y_pred: Predicted descriptor array [batch, max_cracks, desc_length]
        
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
    Gradually adjusts the weight between spectrogram and descriptor losses.
    
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
    BETA_MAX = 0.02   # Maximum value
    
    # Define warmup phase length (in epochs)
    WARMUP_EPOCHS = 50
    
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
    Spectral Multimodal VAE with two modalities:
    1. Complex spectrograms (from time series)
    2. Crack descriptors
    """
    def __init__(self, latent_dim, spec_shape, max_num_cracks=770, desc_length=42):
        super().__init__()
        
        # Store shapes and latent dimension
        self.latent_dim = latent_dim
        self.spec_shape = spec_shape
        self.max_num_cracks = max_num_cracks
        self.desc_length = desc_length
        
        # Encoders
        self.spec_encoder = SpectrogramEncoder(latent_dim)
        self.desc_encoder = DescriptorEncoder(latent_dim)
        
        # Decoders
        # spec_shape = (freq_bins, time_bins, channels*2)
        # So each "original channel" is actually spec_shape[2]//2
        self.spec_decoder = SpectrogramDecoder(
            freq_bins=spec_shape[0],
            time_bins=spec_shape[1],
            channels=spec_shape[2] // 2
        )
        self.desc_decoder = DescriptorDecoder(
            max_num_cracks=max_num_cracks,
            desc_length=desc_length
        )

    def call(self, spec_in, desc_in, test_id=None, training=False):
        """
        Forward pass for the MMVM approach:
         1) Encode each modality -> mu, logvar
         2) Compute JS divergence among all unimodal encoders
         3) Sample from each unimodal posterior
         4) Decode each modality from its own sample
         5) Return reconstructions + JS divergence
        """
        # 1) Encode
        mu_spec, logvar_spec = self.spec_encoder(spec_in, training=training)
        mu_desc, logvar_desc = self.desc_encoder(desc_in, training=training)

        # 2) Compute JS among these two unimodal distributions
        mus = [mu_spec, mu_desc]
        logvars = [logvar_spec, logvar_desc]
        js_div = compute_js_divergence(mus, logvars)  # Use imported function
        
        # 3) Sample from each unimodal
        z_spec = reparameterize(mu_spec, logvar_spec)  # Use imported function
        z_desc = reparameterize(mu_desc, logvar_desc)  # Use imported function
        
        # 4) Decode
        recon_spec = self.spec_decoder(z_spec, training=training)
        recon_desc = self.desc_decoder(z_desc, training=training)
        
        # 5) Return outputs
        return recon_spec, recon_desc, (mus, logvars, js_div)

    def generate(self, modality='spec', conditioning_latent=None):
        """
        Generate samples from the prior or conditioned on another modality.
        
        Args:
            modality: Which modality to generate ('spec' or 'desc')
            conditioning_latent: Optional latent vector to condition generation on
            
        Returns:
            Generated sample
        """
        # Sample from prior or use conditioning latent
        if conditioning_latent is None:
            # Sample from a standard Gaussian prior
            z = tf.random.normal(shape=(1, self.latent_dim))
        else:
            z = conditioning_latent
        
        # Generate requested modality
        if modality == 'spec':
            return self.spec_decoder(z)
        elif modality == 'desc':
            return self.desc_decoder(z)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
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
    spectrograms, descriptor_array, test_id_array,
    batch_size=32, shuffle=True, debug_mode=False, debug_samples=500
):
    """
    Create a TensorFlow Dataset that loads data from memory.
    """
    # Convert file path to memory-mapped array if string is provided
    if isinstance(spectrograms, str):
        spectrograms = np.load(spectrograms, mmap_mode='r')
    
    # Apply debug mode limit
    if debug_mode:
        print(f"⚠️ Debug Mode ON: Using only {debug_samples} samples for quick testing!")
        spectrograms = spectrograms[:debug_samples]
        descriptor_array = descriptor_array[:debug_samples]
        test_id_array = test_id_array[:debug_samples]
    else:
        print(f"✅ Full dataset loaded: {len(descriptor_array)} samples.")

    # Ensure all inputs have appropriate ranks (at least 1)
    test_id_array = np.atleast_1d(test_id_array)
    descriptor_array = np.atleast_1d(descriptor_array)
    
    # Create a dataset from the arrays
    dataset = tf.data.Dataset.from_tensor_slices((spectrograms, descriptor_array, test_id_array))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(descriptor_array))

    # Batch the dataset
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
    beta_schedule='cyclical'
):
    """
    Training function for Spectral MMVAE with system resource monitoring.

    Args:
        model: Spectral MMVAE model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        beta_schedule: Type of beta schedule ('linear', 'exponential', 'cyclical')

    Returns:
        Training and validation metrics
    """
    # Initialize metrics tracking
    train_total_losses = []
    train_spec_losses = []
    train_desc_losses = []
    train_js_losses = []

    val_total_losses = []
    val_spec_losses = []
    val_desc_losses = []
    val_js_losses = []

    # For early stopping
    best_val_loss = float('inf')
    no_improvement_count = 0

    print("🔄 Starting Training for Spectral MMVAE...")

    train_batches_count = sum(1 for _ in train_dataset)
    val_batches_count = sum(1 for _ in val_dataset)
    
    print(f"Training on {train_batches_count} batches, validating on {val_batches_count} batches")

    with tf.device('/GPU:0'):
        for epoch in range(num_epochs):
            start_time = time.time()  # Track epoch duration

            # Get beta value for this epoch
            beta = get_beta_schedule(epoch, num_epochs, beta_schedule)
            
            # Get dynamic loss weights
            desc_weight = dynamic_weighting(epoch, num_epochs)
            spec_weight = 1.0 - desc_weight
            
            print(f"📌 Epoch {epoch+1}/{num_epochs} | Beta: {beta:.6f} | Descriptor Weight: {desc_weight:.2f}")
            
            # Training loop
            epoch_train_total = 0.0
            epoch_train_spec = 0.0
            epoch_train_desc = 0.0
            epoch_train_js = 0.0
            train_steps = 0

            for step, (spec_in, desc_in, test_id_in) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    # Forward pass through the model
                    recon_spec, recon_desc, (all_mus, all_logvars, js_div) = model(
                        spec_in, desc_in, test_id_in,
                        training=True
                    )
                    
                    # Compute losses for each modality
                    spec_loss = complex_spectrogram_loss(spec_in, recon_spec)
                    desc_loss = weighted_descriptor_mse_loss(desc_in, recon_desc)
                    
                    # Combined reconstruction loss with dynamic weighting
                    recon_loss = spec_weight * spec_loss + desc_weight * desc_loss
                    
                    # Total loss = reconstruction loss + beta * JS divergence
                    total_loss = recon_loss + beta * js_div

                # Gradient clipping to prevent extreme updates
                grads = tape.gradient(total_loss, model.trainable_variables)
                # Clip gradients to prevent exploding gradients
                clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
                optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))

                # Track losses
                epoch_train_total += total_loss.numpy()
                epoch_train_spec += spec_loss.numpy()
                epoch_train_desc += desc_loss.numpy()
                epoch_train_js += js_div.numpy()
                train_steps += 1

            # Compute average training losses for the epoch
            if train_steps > 0:
                train_total_losses.append(epoch_train_total / train_steps)
                train_spec_losses.append(epoch_train_spec / train_steps)
                train_desc_losses.append(epoch_train_desc / train_steps)
                train_js_losses.append(epoch_train_js / train_steps)
            
            # Debug information
            print(f"✅ Train Loss: {train_total_losses[-1]:.4f} | "
                f"Spec: {train_spec_losses[-1]:.4f} | "
                f"Desc: {train_desc_losses[-1]:.4f} | " 
                f"JS: {train_js_losses[-1]:.4f}")

            # Validation loop
            epoch_val_total = 0.0
            epoch_val_spec = 0.0
            epoch_val_desc = 0.0
            epoch_val_js = 0.0
            val_steps = 0

            for step, (spec_in, desc_in, test_id_in) in enumerate(val_dataset):
                # Forward pass
                recon_spec, recon_desc, (all_mus, all_logvars, js_div) = model(
                    spec_in, desc_in, test_id_in, 
                    training=False
                )

                # Compute losses for each modality
                spec_loss = complex_spectrogram_loss(spec_in, recon_spec)
                desc_loss = weighted_descriptor_mse_loss(desc_in, recon_desc)
                
                # Combined reconstruction loss with the same weighting
                recon_loss = spec_weight * spec_loss + desc_weight * desc_loss
                
                # Total loss with the same Beta value
                total_loss = recon_loss + beta * js_div

                # Track validation losses
                epoch_val_total += total_loss.numpy()
                epoch_val_spec += spec_loss.numpy()
                epoch_val_desc += desc_loss.numpy()
                epoch_val_js += js_div.numpy()
                val_steps += 1

            # Compute average validation losses
            if val_steps > 0:
                val_total_losses.append(epoch_val_total / val_steps)
                val_spec_losses.append(epoch_val_spec / val_steps)
                val_desc_losses.append(epoch_val_desc / val_steps)
                val_js_losses.append(epoch_val_js / val_steps)

            print(f"  🔵 Val => Total: {val_total_losses[-1]:.4f} | "
                f"Spec: {val_spec_losses[-1]:.4f} | "
                f"Desc: {val_desc_losses[-1]:.4f} | "
                f"JS: {val_js_losses[-1]:.4f}")

            # Early Stopping
            current_val_loss = val_total_losses[-1]
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                no_improvement_count = 0
                # Save best model weights
                model.save_weights("results/best_spectral_mmvae.weights.h5")
                print("✅ Saved best model weights")
            else:
                no_improvement_count += 1
                print(f"🚨 No improvement for {no_improvement_count}/{patience} epochs.")

            # ✅ CLEAR MEMORY AFTER EACH EPOCH
            print(f"✅ Finished epoch {epoch+1}, clearing memory...")
            tf.keras.backend.clear_session()
            gc.collect()

            if no_improvement_count >= patience:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.")
                break

    return {
        'train_total': train_total_losses,
        'train_spec': train_spec_losses,
        'train_desc': train_desc_losses,
        'train_js': train_js_losses,
        'val_total': val_total_losses,
        'val_spec': val_spec_losses,
        'val_desc': val_desc_losses,
        'val_js': val_js_losses
    }

# ----- Visualization Functions -----
def plot_training_curves(metrics):
    """
    Plot training curves with improved visualization.
    
    Args:
        metrics: Dictionary of training and validation metrics
    """
    epochs = list(range(1, len(metrics['train_total']) + 1))
    
    # 1. Total Loss
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_total'], mode='lines+markers', name="Train Total", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['val_total'], mode='lines+markers', name="Val Total", line=dict(color='red')))
    fig.update_layout(title="Total Loss vs Epochs", xaxis_title="Epoch", yaxis_title="Loss")
    pio.write_html(fig, file="results/train_val_total_loss.html", auto_open=True)
    
    # 2. Spectrogram & Descriptor Losses
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_spec'], mode='lines+markers', name="Train Spec", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['val_spec'], mode='lines+markers', name="Val Spec", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_desc'], mode='lines+markers', name="Train Desc", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['val_desc'], mode='lines+markers', name="Val Desc", line=dict(color='orange')))
    fig.update_layout(title="Modality Losses vs Epochs", xaxis_title="Epoch", yaxis_title="Loss")
    pio.write_html(fig, file="results/train_val_modality_loss.html", auto_open=True)
    
    # 3. JS Divergence
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_js'], mode='lines+markers', name="Train JS", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['val_js'], mode='lines+markers', name="Val JS", line=dict(color='red')))
    fig.update_layout(title="JS Divergence vs Epochs", xaxis_title="Epoch", yaxis_title="JS Divergence")
    pio.write_html(fig, file="results/train_val_js_div.html", auto_open=True)

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
        yaxis=dict(autorange="reversed")  # Flip y-axis to have low frequencies at bottom
    )
    
    if output_path:
        pio.write_html(fig, file=output_path, auto_open=True)
    else:
        fig.show()

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
    for i, (spec_batch, desc_batch, _) in enumerate(val_dataset.take(1)):
        for j in range(min(num_samples, spec_batch.shape[0])):
            # Get sample
            spec_sample = tf.expand_dims(spec_batch[j], 0)
            desc_sample = tf.expand_dims(desc_batch[j], 0)
            
            # Get reconstructions
            recon_spec, recon_desc, _ = model(
                spec_sample, desc_sample, 
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
                desc_np = desc_sample.numpy()[0]
                valid_mask = np.any(desc_np != 0, axis=1)
                valid_descriptors = desc_np[valid_mask]
                
                recon_desc_np = recon_desc.numpy()[0]
                valid_recon_mask = np.any(recon_desc_np != 0, axis=1)
                valid_recon_descriptors = recon_desc_np[valid_recon_mask]
                
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

    for spec_in, desc_in, test_id_in in dataset:
        # Convert test_id_in to proper format
        if isinstance(test_id_in, tf.Tensor):
            test_id_in = test_id_in.numpy()
        if isinstance(test_id_in, np.ndarray):
            test_id_in = test_id_in.flatten().tolist()

        # Get latent representations from spectrogram encoder (could also use descriptor encoder)
        mu_q, _ = model.spec_encoder(spec_in, training=False)
        
        latent_vectors.append(mu_q.numpy())
        test_ids.append(test_id_in)

    return np.concatenate(latent_vectors, axis=0), np.concatenate(test_ids, axis=0)

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
            gen_desc_np = gen_desc.numpy()[0]
            valid_mask = np.any(gen_desc_np != 0, axis=1)
            valid_descriptors = gen_desc_np[valid_mask]
            
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

# ----- Main Function -----
def main():
    # Set debug mode
    debug_mode = True
    # ------------------------ 1) Configure Environment ------------------------
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/model_checkpoints", exist_ok=True)

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # ------------------------ 2) Define Parameters ------------------------
    fs = 200  # Sampling frequency in Hz
    nperseg = 256  # STFT segment length
    noverlap = 224  # STFT overlap
    
    latent_dim = 64 
    batch_size = 32  
    num_epochs = 200  
    patience = 5  

    # Cached file paths
    stft_path = "cached_stft.npy"
    final_path = "cached_spectral_features.npy"
    desc_path = "cached_descriptors.npy"
    ids_path = "cached_test_ids.npy"

    # Variables for data
    spectral_features = None
    descriptor_segments = None
    test_ids = None

    # ------------------------ 3) Load Cached Spectrograms ------------------------
    if os.path.exists(final_path):
        print(f"📂 Found cached spectrogram features: {final_path}")
        spectral_features = np.load(final_path, mmap_mode='r')
        print(f"✅ Loaded spectral features: {spectral_features.shape}")

    # ------------------------ 4) Load Cached Descriptors ------------------------
    if os.path.exists(desc_path):
        print(f"📂 Found cached descriptors: {desc_path}")
        descriptor_segments = np.load(desc_path, mmap_mode='r')
        print(f"✅ Loaded descriptors: {descriptor_segments.shape}")

    # ------------------------ 5) Load Cached Test IDs ------------------------
    if os.path.exists(ids_path):
        print(f"📂 Found cached test IDs: {ids_path}")
        test_ids = np.load(ids_path, mmap_mode='r')
        print(f"✅ Loaded test IDs: {test_ids.shape}")

    # ------------------------ 6) Process Missing Data ------------------------
    if spectral_features is None or descriptor_segments is None or test_ids is None:
        print("⚠️ Some cached data is missing. Processing raw data...")
        
        # Load raw data
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

        sample_desc = next(iter(padded_dict.values()))
        max_num_cracks, desc_length = sample_desc.shape
        print(f"Loaded data for {len(accel_dict)} tests")
        print(f"Descriptor shape: {max_num_cracks} cracks x {desc_length} features")

    else:
        # Thia is a hardcoded standart -> needs to be changed to dynamic later
        max_num_cracks = 770  
        desc_length = 42      

        # Only segment if descriptors or test IDs are missing
        if descriptor_segments is None or test_ids is None:
            print("✂️ Segmenting data...")
            raw_segments, descriptor_segments, test_ids = segment_and_transform(accel_dict, padded_dict)
            print(f"✅ Extracted {len(raw_segments)} segments")
        
            # Save segmented descriptors & test IDs
            np.save(desc_path, descriptor_segments)
            np.save(ids_path, test_ids)
            print(f"✅ Saved descriptors to {desc_path}")
            print(f"✅ Saved test IDs to {ids_path}")

        # Compute spectrograms only if missing
        if spectral_features is None:
            print("🔄 Computing spectrograms...")
            complex_specs = compute_and_cache_spectrograms(raw_segments, fs, nperseg, noverlap, cache_path=stft_path)
            
            print("📝 Converting spectrograms to features...")
            spectral_features = cache_final_features(complex_specs, cache_path=final_path)

    # ------------------------ 7) Split Data ------------------------
    N = spectral_features.shape[0]
    indices = np.random.permutation(N)
    train_size = int(0.8 * N)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_desc, train_ids = descriptor_segments[train_idx], test_ids[train_idx]
    val_desc, val_ids = descriptor_segments[val_idx], test_ids[val_idx]

    # ------------------------ 8) Create TensorFlow Datasets ------------------------
    train_dataset = create_tf_dataset(final_path, train_desc, train_ids, batch_size, debug_mode=debug_mode)
    val_dataset = create_tf_dataset(final_path, val_desc, val_ids, batch_size, debug_mode=debug_mode)

    print(f"✅ Train batches: {sum(1 for _ in train_dataset)}")
    print(f"✅ Val batches: {sum(1 for _ in val_dataset)}")

    # ------------------------ 9) Build Model ------------------------
    spec_shape = spectral_features.shape[1:]  
    
    print(f"📐 Building model with:")
    print(f"  - Latent dimension: {latent_dim}")
    print(f"  - Spectrogram shape: {spec_shape}")
    print(f"  - Descriptor shape: ({max_num_cracks}, {desc_length})")
    
    model = SpectralMMVAE(latent_dim, spec_shape, max_num_cracks, desc_length)

    # ------------------------ 10) Set Up Optimizer ------------------------
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
    )
    
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6
    )

    # ------------------------ 11) Train Model ------------------------
    print("🚀 Starting training...")
    training_metrics = train_spectral_mmvae(model, train_dataset, val_dataset, optimizer, num_epochs, patience)

    # ------------------------ 12) Save & Visualize Training Results ------------------------
    np.save("results/training_metrics.npy", training_metrics)
    plot_training_curves(training_metrics)

    # ------------------------ 13) Load Best Model ------------------------
    model.load_weights("results/best_spectral_mmvae.weights.h5")

    # ------------------------ 14) Extract & Visualize Latent Space ------------------------
    latent_vectors, test_ids_arr = extract_latent_representations(model, train_dataset)
    latent_3d = reduce_latent_dim_umap(latent_vectors)
    plot_latent_space_3d(latent_3d, test_ids_arr, output_file="results/latent_space_3d.html")

    # ------------------------ 15) Evaluate Reconstructions ------------------------
    visualize_reconstructions(model, val_dataset, data_loader, num_samples=5, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # ------------------------ 16) Generate New Samples ------------------------
    generate_samples(model, data_loader, num_samples=5, fs=fs, nperseg=nperseg, noverlap=noverlap)

    print("✅ Training & evaluation complete. Results saved in 'results/' directory.")


if __name__ == "__main__":
    main()
