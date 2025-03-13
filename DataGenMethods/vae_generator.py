import tensorflow as tf

# This must be the very first TensorFlow-related code
def configure_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled on {len(physical_devices)} GPU(s)")
        except RuntimeError as e:
            print(e)

# Call GPU configuration before importing any other TF-related code
configure_gpu()

import os
import cv2
import sys
import psutil
import GPUtil
import numpy as np
import keras 
from keras import layers, Model, optimizers
from keras.optimizers.schedules import ExponentialDecay
import logging
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import umap
import plotly.express as px
import pandas as pd
from sklearn.mixture import GaussianMixture




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
    """Generator that yields data in chunks, with descriptors from descriptor_dict."""
    
    window_size = int(sample_rate * segment_duration)
    half_window = window_size // 2
    
    test_ids = list(accel_dict.keys())
    for i in range(0, len(test_ids), chunk_size):
        chunk_ids = test_ids[i:i + chunk_size]
        
        raw_segments = []
        rms_segments = []
        psd_segments = []
        descriptor_segments = []
        test_ids_out = []
        
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

                    # Compute features
                    seg_rms = compute_rms(segment_raw)
                    seg_psd = compute_psd(segment_raw, sample_rate)

                    raw_segments.append(segment_raw)
                    rms_segments.append(seg_rms)
                    psd_segments.append(seg_psd)
                    
                    # Get descriptor data for this test_id
                    descriptor_segments.append(descriptor_dict[test_id])
                    test_ids_out.append(int(test_id))
        
        if raw_segments:  # Only yield if we have data
            yield (
                np.array(raw_segments, dtype=np.float32),
                np.array(rms_segments, dtype=np.float32),
                np.array(psd_segments, dtype=np.float32),
                np.array(descriptor_segments, dtype=np.float32),
                np.array(test_ids_out, dtype=np.int32)
            )

def create_tf_dataset(
    raw_segments, rms_segments, psd_segments, descriptor_segments, test_ids,
    batch_size=8
):
    print(f"Creating dataset with shapes:")
    print(f"  Raw segments: {raw_segments.shape}")
    print(f"  RMS segments: {rms_segments.shape}")
    print(f"  PSD segments: {psd_segments.shape}")
    print(f"  Descriptor segments: {descriptor_segments.shape}")
    print(f"  Test IDs: {test_ids.shape}\n")

    test_ids = np.array(test_ids, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            raw_segments.astype(np.float32),
            rms_segments.astype(np.float32),
            psd_segments.astype(np.float32),
            descriptor_segments.astype(np.float32),
            test_ids
        )
    )
    
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    for batch in dataset.take(1):
        print("\nFirst batch shapes:")
        print(f"  Raw: {batch[0].shape}")
        print(f"  RMS: {batch[1].shape}")
        print(f"  PSD: {batch[2].shape}")
        print(f"  Descriptor: {batch[3].shape}")
        print(f"  Test ID: {batch[4].shape}")

    return dataset

def compute_rms(segment):
    """
    If you want a short-time RMS or something else. For now, let's do a single RMS value per channel.
    """
    rms_val = np.sqrt(np.mean(segment**2, axis=0))
    return rms_val  # shape: (num_channels,)

def compute_psd(segment, sample_rate):
    """
    Example: Welch's method for PSD, per channel
    Return shape might be (freq_bins, num_channels)
    """
    # We'll do a quick placeholder: signal.welch along each channel
    seg_T = segment.T  # shape: (num_channels, window_size)
    psds = []
    for ch_data in seg_T:
        freqs, pxx = signal.welch(ch_data, fs=sample_rate)
        psds.append(pxx)
    psds = np.array(psds).T  # shape: (freq_bins, num_channels)
    return psds

# ----- Loss Functions -----
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, eps=1e-7):
    """
    Focal loss with clipping to avoid log(0).
    """
    # Validate inputs
    tf.debugging.assert_all_finite(y_true, "focal_loss: y_true has NaN/Inf")
    tf.debugging.assert_all_finite(y_pred, "focal_loss: y_pred has NaN/Inf")

    # Clip y_pred to avoid log(0) => -inf
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    # Cross-entropy terms
    ce_pos = -y_true * tf.math.log(y_pred)
    ce_neg = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)

    # Focal weights
    wt_pos = alpha * tf.pow((1.0 - y_pred), gamma)
    wt_neg = (1.0 - alpha) * tf.pow(y_pred, gamma)

    # Combine
    fl_pos = wt_pos * ce_pos
    fl_neg = wt_neg * ce_neg

    focal = tf.reduce_mean(fl_pos + fl_neg)

    # Assert focal is still finite
    tf.debugging.assert_all_finite(focal, "focal_loss result is NaN/Inf")

    return focal

def fft_loss(y_true, y_pred):
    """
    Frequency-aware loss: Mean Squared Error of FFT magnitudes.
    """
    y_true_fft = tf.abs(tf.signal.rfft(y_true))
    y_pred_fft = tf.abs(tf.signal.rfft(y_pred))
    return tf.reduce_mean(tf.square(y_true_fft - y_pred_fft))

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

def dynamic_mask_loss_weight(epoch, max_epochs, min_weight=0.3, max_weight=0.7):
    """
    Gradually increases the weight of mask loss relative to time-series loss.
    This helps focus on mask reconstruction once time-series learns good patterns.
    
    Args:
        epoch: Current epoch
        max_epochs: Total epochs
        min_weight: Starting weight for mask loss
        max_weight: Maximum weight for mask loss
        
    Returns:
        Current mask loss weight
    """
    # Linear increase over time
    progress = min(1.0, epoch / (max_epochs * 0.5))  # Reach max_weight halfway through training
    return min_weight + progress * (max_weight - min_weight)

def feature_weighted_descriptor_loss(y_true, y_pred, position_weight=2.0, shape_weight=1.5, other_weight=1.0):
    """
    Custom loss function that weights different descriptor features differently.
    
    Args:
        y_true: Ground truth descriptor array [batch, max_cracks, desc_length]
        y_pred: Predicted descriptor array [batch, max_cracks, desc_length]
        position_weight: Weight for position-related features
        shape_weight: Weight for shape-related features
        other_weight: Weight for other features
        
    Returns:
        Weighted MSE loss with different weights per feature
    """
    # Create mask for valid (non-zero) descriptors
    valid_mask = tf.reduce_any(tf.not_equal(y_true, 0), axis=-1)
    valid_mask = tf.cast(valid_mask, dtype=tf.float32)
    valid_mask = tf.expand_dims(valid_mask, axis=-1)
    
    # Assuming descriptor structure: first features are position, then shape, then others
    # Adjust these indices based on your actual descriptor layout
    position_indices = tf.range(0, 4)  # Example: first 4 features are position (x, y, etc.)
    shape_indices = tf.range(4, 10)    # Example: next 6 features are shape (width, length, etc.)
    
    # Create feature weight mask of shape [desc_length]
    desc_length = y_true.shape[-1]
    feature_weights = tf.ones([desc_length], dtype=tf.float32) * other_weight
    
    # Update weights for specific features
    for idx in position_indices:
        feature_weights = tf.tensor_scatter_nd_update(
            feature_weights, 
            tf.expand_dims([idx], axis=1), 
            [position_weight]
        )
    
    for idx in shape_indices:
        feature_weights = tf.tensor_scatter_nd_update(
            feature_weights, 
            tf.expand_dims([idx], axis=1), 
            [shape_weight]
        )
    
    # Expand feature weights to match input shape for broadcasting
    feature_weights = tf.reshape(feature_weights, [1, 1, desc_length])
    
    # Calculate weighted squared error
    squared_error = tf.square(y_true - y_pred) * feature_weights
    
    # Apply valid mask
    masked_error = squared_error * valid_mask
    
    # Get number of valid elements for averaging
    num_valid = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    
    # Compute mean over valid elements only
    loss = tf.reduce_sum(masked_error) / num_valid
    
    return loss

def combined_descriptor_loss(y_true, y_pred, mse_weight=0.7, feature_weight=0.3):
    """
    Combines regular MSE loss with feature-weighted loss for robustness.
    
    Args:
        y_true: Ground truth descriptor array
        y_pred: Predicted descriptor array
        mse_weight: Weight for basic MSE loss
        feature_weight: Weight for feature-weighted loss
        
    Returns:
        Combined loss
    """
    mse_loss = weighted_descriptor_mse_loss(y_true, y_pred)
    feat_loss = feature_weighted_descriptor_loss(y_true, y_pred)
    
    return mse_weight * mse_loss + feature_weight * feat_loss

def get_beta_schedule(epoch, max_epochs, schedule_type='linear'):
    """
    Compute the beta coefficient for the JS divergence loss term.
    Implements various scheduling options.
    
    Args:
        epoch: Current epoch number (0-indexed)
        max_epochs: Total number of epochs
        schedule_type: Type of schedule ('linear', 'exponential', 'cyclical')
    
    Returns:
        beta: Beta coefficient for current epoch
    """
    # Define beta limits
    BETA_MIN = 1e-8   # Start with very small value
    BETA_MAX = 0.02   # Significantly higher than the original 0.001
    
    # Define warmup phase length (in epochs)
    WARMUP_EPOCHS = 50  # Adjust based on your dataset and training dynamics
    
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
    
# ----- Encoder Branches -----
class RawEncoder(Model):
    """
    Convolution + LSTM encoder for raw time-series data.
    Input shape: (batch_size, 1000, 12)
    Output: Mean and logvariance for latent distribution.
    """

    def __init__(self, latent_dim):
        super().__init__()

        # 1) First CNN block
        self.conv1 = layers.Conv1D(
            filters=16,
            kernel_size=5,
            strides=2,
            padding='same',
            activation='relu'
        )
        self.dropout1 = layers.Dropout(0.3)

        # 2) Second CNN block
        self.conv2 = layers.Conv1D(
            filters=32,
            kernel_size=5,
            strides=2,
            padding='same',
            activation='relu'
        )
        self.dropout2 = layers.Dropout(0.3)

        # 3) Third CNN block
        self.conv3 = layers.Conv1D(
            filters=64,
            kernel_size=5,
            strides=2,
            padding='same',
            activation='relu'
        )
        self.dropout3 = layers.Dropout(0.3)

        # 4) Bidirectional LSTM layer
        self.bilstm = layers.Bidirectional(
            layers.LSTM(128, return_sequences=False)
        )

        # 5) Dense to reduce to 512
        self.dense_reduce = layers.Dense(512, activation='relu')

        # 6) Mean and logvariance projections
        self.mu_layer = layers.Dense(latent_dim)
        self.logvar_layer = layers.Dense(latent_dim)

    def call(self, x, training=False):
        """
        x shape: (batch, 1000, 12)
        """
        # CNN block 1
        x = self.conv1(x)                 # -> (batch, 500, 16)
        x = self.dropout1(x, training=training)

        # CNN block 2
        x = self.conv2(x)                 # -> (batch, 250, 32)
        x = self.dropout2(x, training=training)

        # CNN block 3
        x = self.conv3(x)                 # -> (batch, 125, 64)
        x = self.dropout3(x, training=training)

        # LSTM layer
        x = self.bilstm(x, training=training)  # -> (batch, 256)

        # Dense
        x = self.dense_reduce(x)          # -> (batch, 512)
        
        # Output distribution parameters
        mu = self.mu_layer(x)             # -> (batch, latent_dim)
        logvar = self.logvar_layer(x)     # -> (batch, latent_dim)
        
        return mu, logvar
    
class PSDEncoder(Model):
    """
    Encoder for PSD features.
    Input shape: (batch, freq_bins, 12), e.g., (batch, 129, 12)
    Output: Mean and logvariance for latent distribution.
    """

    def __init__(self, latent_dim):
        super().__init__()

        # First convolutional block
        self.conv1 = layers.Conv1D(filters=16, kernel_size=5, strides=2, padding='same', activation='relu')
        self.dropout1 = layers.Dropout(0.3)

        # Second convolutional block
        self.conv2 = layers.Conv1D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu')
        self.dropout2 = layers.Dropout(0.3)

        # Third convolutional block
        self.conv3 = layers.Conv1D(filters=64, kernel_size=5, strides=2, padding='same', activation='relu')
        self.dropout3 = layers.Dropout(0.3)

        # Global pooling
        self.global_pool = layers.GlobalAveragePooling1D()

        # Dense layers
        self.dense_reduce = layers.Dense(128, activation='relu')
        
        # Mean and logvariance projections
        self.mu_layer = layers.Dense(latent_dim)
        self.logvar_layer = layers.Dense(latent_dim)

    def call(self, x, training=False):
        """
        x shape: (batch, freq_bins, 12)
        """
        x = self.conv1(x)  # (batch, ~65, 16)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)  # (batch, ~33, 32)
        x = self.dropout2(x, training=training)

        x = self.conv3(x)  # (batch, ~17, 64)
        x = self.dropout3(x, training=training)

        # Global pooling
        x = self.global_pool(x)  # (batch, 64)

        # Fully connected layer
        x = self.dense_reduce(x)  # (batch, 128)
        
        # Output distribution parameters
        mu = self.mu_layer(x)         # -> (batch, latent_dim)
        logvar = self.logvar_layer(x) # -> (batch, latent_dim)
        
        return mu, logvar
    
class RMSEncoder(keras.Model):
    """
    Encoder for RMS features.
    Input shape: (batch, 12) (one value per channel)
    Output: Mean and logvariance for latent distribution.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.dense1 = layers.Dense(16, activation='relu')
        self.drop1 = layers.Dropout(0.3)
        
        # Mean and logvariance projections
        self.mu_layer = layers.Dense(latent_dim)
        self.logvar_layer = layers.Dense(latent_dim)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.drop1(x, training=training)
        
        # Output distribution parameters
        mu = self.mu_layer(x)         # -> (batch, latent_dim)
        logvar = self.logvar_layer(x) # -> (batch, latent_dim)
        
        return mu, logvar

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
        self.drop1 = layers.Dropout(0.3)
        
        # Second 1D convolutional layer
        self.conv2 = layers.Conv1D(128, 3, padding='same', activation='relu')
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
        x = x * input_mask  # Apply mask
        x = self.drop1(x, training=training)
        
        x = self.conv2(x)
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

# ----- Decoders with Cross-Attention -----
class TimeSeriesDecoder(keras.Model):
    """
    Decoder for a 5s segment of raw time-series data.
    Input: latent vector z.
    Uses cross-attention on mask features.
    Output shape: (1000, 12)
    """
    def __init__(self):
        super(TimeSeriesDecoder, self).__init__()
        self.fc = layers.Dense(250 * 128, activation='relu')
        self.reshape_layer = layers.Reshape((250, 128))
        self.attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=128)
        self.mask_proj_layer = layers.Dense(128)
        self.upsample1 = layers.UpSampling1D(size=2)   # 250 -> 500
        self.conv1 = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')
        self.drop1 = layers.Dropout(0.3)
        self.upsample2 = layers.UpSampling1D(size=2)   # 500 -> 1000
        self.conv2 = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu')
        self.drop2 = layers.Dropout(0.3)
        self.conv_out = layers.Conv1D(12, kernel_size=3, padding='same')

    def call(self, z, mask_features=None, training=False):
        x = self.fc(z)
        x = self.reshape_layer(x)  # [B, 250, 128]
        if mask_features is not None:
            mask_proj = self.mask_proj_layer(mask_features)  # [B, 128]
            mask_proj = tf.expand_dims(mask_proj, axis=1)     # [B, 1, 128]
            mask_proj = tf.tile(mask_proj, [1, tf.shape(x)[1], 1])
            attn_output = self.attention_layer(query=x, value=mask_proj, key=mask_proj)
            x = x + attn_output
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.drop1(x, training=training)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.drop2(x, training=training)
        x = self.conv_out(x)
        return x  # [B, 1000, 12]
 
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
        self.drop1 = layers.Dropout(0.3)
        
        # Project to shape that can be reshaped to (batch, max_cracks/10, 10*hidden_dim)
        # We'll reshape and then use Conv1DTranspose to expand
        hidden_dim = 64
        self.reshaped_cracks = max_num_cracks // 10  # Compress by factor of 10
        self.fc2 = layers.Dense(self.reshaped_cracks * hidden_dim * 10, activation='relu')
        self.drop2 = layers.Dropout(0.3)
        
        # Reshape layer
        self.reshape = layers.Reshape((self.reshaped_cracks, hidden_dim * 10))
        
        # Upsampling with Conv1DTranspose
        self.conv_t1 = layers.Conv1DTranspose(128, 3, strides=2, padding='same', activation='relu')
        self.drop3 = layers.Dropout(0.3)
        
        self.conv_t2 = layers.Conv1DTranspose(64, 3, strides=2, padding='same', activation='relu')
        self.drop4 = layers.Dropout(0.3)
        
        # Final projection to descriptor space
        self.conv_t3 = layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation='relu')
        self.drop5 = layers.Dropout(0.3)
        
        # Output layer - no activation as descriptors can have any range of values
        self.output_layer = layers.Conv1D(desc_length, 1, padding='same')

    def call(self, z, training=False):
        # Initial dense projection
        x = self.fc1(z)
        x = self.drop1(x, training=training)
        
        # Project to reshapable dimension
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        
        # Reshape to begin 1D convolution sequence
        x = self.reshape(x)
        
        # Upsample with transposed convolutions
        x = self.conv_t1(x)
        x = self.drop3(x, training=training)
        
        x = self.conv_t2(x)
        x = self.drop4(x, training=training)
        
        x = self.conv_t3(x)
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
          
# ----- VAE -----
class MMVM_VAE(tf.keras.Model):
    """
    Improved Multimodal VAE with true Variational Mixture-of-Experts prior (MMVM VAE).
    Updated to use descriptor data instead of mask data.
    """
    def __init__(self, latent_dim, max_num_cracks=770, desc_length=42):
        super(MMVM_VAE, self).__init__()
        
        # Independent encoders for each modality
        self.raw_encoder = RawEncoder(latent_dim)
        self.rms_encoder = RMSEncoder(latent_dim)
        self.psd_encoder = PSDEncoder(latent_dim)
        self.descriptor_encoder = DescriptorEncoder(latent_dim)
        
        # Decoders
        self.raw_decoder = TimeSeriesDecoder()
        # Replace mask decoder with descriptor decoder
        self.descriptor_decoder = DescriptorDecoder(max_num_cracks=max_num_cracks, desc_length=desc_length)
        
        # For self-attention refinement
        self.token_count = 8
        self.token_dim = latent_dim // self.token_count
        self.self_attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=self.token_dim)
        
        # Store latent dimension
        self.latent_dim = latent_dim
        
        # Store descriptor dimensions
        self.max_num_cracks = max_num_cracks
        self.desc_length = desc_length
        
        # Monte Carlo samples for KL estimation
        self.mc_samples = 100
    
    def call(self, raw_in, rms_in, psd_in, descriptor_in, test_id, training=False):
            """
            Forward pass of the improved MMVM VAE model.
            Each modality is encoded independently and decoded from its own latent representation.
            Now using descriptor data instead of mask data.
            """
            # Encode each modality to its own latent distribution
            mu_raw, logvar_raw = self.raw_encoder(raw_in, training=training)
            mu_rms, logvar_rms = self.rms_encoder(rms_in, training=training)
            mu_psd, logvar_psd = self.psd_encoder(psd_in, training=training)
            mu_desc, logvar_desc = self.descriptor_encoder(descriptor_in, training=training)
            
            # Collect all distribution parameters
            all_mus = [mu_raw, mu_rms, mu_psd, mu_desc]
            all_logvars = [logvar_raw, logvar_rms, logvar_psd, logvar_desc]
            
            # Compute JS divergence with true mixture-of-experts prior
            js_div = self.compute_mixture_js_divergence(all_mus, all_logvars)
            
            # Sample from each unimodal posterior
            z_raw = self.reparameterize(mu_raw, logvar_raw)
            z_desc = self.reparameterize(mu_desc, logvar_desc)
            
            # Apply self-attention to refine latent vectors
            z_raw_refined = self.apply_self_attention(z_raw)
            z_desc_refined = self.apply_self_attention(z_desc)
            
            # Decode each modality from its own latent vector
            recon_ts = self.raw_decoder(z_raw_refined, training=training)
            
            # Decode descriptors instead of mask
            recon_desc = self.descriptor_decoder(z_desc_refined, training=training)
            
            # Return reconstructions, latent parameters, and JS divergence
            return recon_ts, recon_desc, (all_mus, all_logvars, js_div)

    def compute_mixture_js_divergence(self, mus, logvars):
        """
        Compute Jensen–Shannon divergence using the 'all-modalities-included' mixture-of-experts prior.
        Each modality's prior is the average of ALL modality encoders, including its own.
        
        In the paper notation (Equation (3)):
            h(z_m | X) = 1/M * sum_{m'=1..M} q_{m'}(z_m | x_{m'})
        which is used in KL(q_m || h), producing M * JS(...) total. (Paper: Unity by Diversity - Thomas. M. Sutter et.al)

        Args:
            mus: list of Tensors (shape [batch_size, latent_dim]) for each modality’s mu
            logvars: list of Tensors (same shape) for each modality’s log-variance
        Returns:
            js_div: A scalar Tensor for the mean KL(q_m || mixture_of_all_modalities), i.e. the JS divergence
        """
        M = len(mus)
        if M <= 1:
            # No JS-div penalty if only one modality
            return tf.constant(0.0, dtype=tf.float32)

        kl_divs = []
        for m in range(M):
            # Current modality's distribution parameters
            mu_m = mus[m]
            logvar_m = tf.clip_by_value(logvars[m], -8.0, 8.0)

            # === Mixture includes *all* M modalities (including this one) ===
            mixture_mus = mus           
            mixture_logvars = logvars    

            # Compute KL( q_m || mixture_of_all_mods )
            current_kl = self._kl_to_mixture(
                mu_m,
                logvar_m,
                mixture_mus,
                mixture_logvars
            )
            kl_divs.append(current_kl)

        # JS divergence is the average of those M KL terms
        js_div = tf.reduce_mean(kl_divs)

        # Safety check
        tf.debugging.assert_all_finite(js_div, "JS divergence is not finite.")
        return js_div


    def _kl_to_mixture(self, mu, logvar, mixture_mus, mixture_logvars):
        """
        Compute KL(q || mixture) where q is a single Gaussian and mixture is a mixture of Gaussians.
        Uses Monte Carlo approximation for the expectation.
        
        Args:
            mu: Mean of q distribution
            logvar: Log variance of q distribution
            mixture_mus: List of means for mixture components
            mixture_logvars: List of log variances for mixture components
        
        Returns:
            KL(q || mixture) averaged over batch
        """
        # Number of mixture components
        K = len(mixture_mus)
        
        # Number of Monte Carlo samples
        n_samples = self.mc_samples
        
        # Sample from q
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=[n_samples, *tf.shape(std)])
        z_samples = mu + eps * std  # Shape: [n_samples, batch_size, latent_dim]
        
        # Compute log p(z|q) for each sample
        log_q = self._log_normal_pdf(z_samples, mu, logvar)  # Shape: [n_samples, batch_size]
        
        # Compute log p(z|mixture) for each sample and mixture component
        log_mixture = []
        for k in range(K):
            log_p_k = self._log_normal_pdf(z_samples, mixture_mus[k], mixture_logvars[k])
            log_mixture.append(log_p_k)
        
        # Log of average: log(1/K * sum_k exp(log_p_k))
        log_mixture = tf.reduce_logsumexp(tf.stack(log_mixture, axis=0), axis=0) - tf.math.log(tf.cast(K, tf.float32))
        
        # KL = E_q[log q(z) - log mixture(z)]
        kl = log_q - log_mixture  # Shape: [n_samples, batch_size]
        
        # Average over samples and batch
        return tf.reduce_mean(kl)

    def _log_normal_pdf(self, sample, mu, logvar):
        """
        Compute log probability of sample under Normal(mu, exp(logvar)).
        
        Args:
            sample: Tensor of shape [n_samples, batch_size, latent_dim]
            mu: Tensor of shape [batch_size, latent_dim]
            logvar: Tensor of shape [batch_size, latent_dim]
        
        Returns:
            Log probability of shape [n_samples, batch_size]
        """
        log_2pi = tf.math.log(2. * np.pi)
        
        # Expand dimensions for broadcasting
        mu = tf.expand_dims(mu, axis=0)  # [1, batch_size, latent_dim]
        logvar = tf.expand_dims(logvar, axis=0)  # [1, batch_size, latent_dim]
        
        # Compute log PDF
        log_prob = -0.5 * (
            log_2pi + 
            logvar + 
            tf.exp(-logvar) * tf.square(sample - mu)
        )
        
        # Sum over latent dimensions
        return tf.reduce_sum(log_prob, axis=-1)
    
    def reparameterize(self, mu, logvar):
        """Sample from a Gaussian distribution with given parameters"""
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(std))
        return mu + eps * std
    
    def apply_self_attention(self, z):
        """Apply self-attention to refine latent vector"""
        z_tokens = tf.reshape(z, (-1, self.token_count, self.token_dim))
        attn_output = self.self_attention_layer(query=z_tokens, key=z_tokens, value=z_tokens)
        z_refined = tf.reshape(attn_output, tf.shape(z))
        return z_refined
    
    def generate(self, modality='ts', conditioning_latent=None):
        """
        Generate samples from the prior or conditioned on another modality.
        
        Args:
            modality: Which modality to generate ('ts' or 'desc')
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
        
        # Refine with self-attention
        z_refined = self.apply_self_attention(z)
        
        # Generate requested modality
        if modality == 'ts':
            return self.raw_decoder(z_refined)
        elif modality == 'desc':
            return self.descriptor_decoder(z_refined)
        else:
            raise ValueError(f"Unknown modality: {modality}")
                     
# ----- Prior estimation -----
def compute_vae_loss(recon_loss, kl_loss, current_epoch, total_epochs):
    """
    Computes the VAE loss with KL annealing.
    
    Args:
        recon_loss: Reconstruction loss (MSE or BCE).
        kl_loss: KL divergence loss.
        current_epoch: Current training epoch.
        total_epochs: Total training epochs.
    
    Returns:
        Total loss with annealed KL term.
    """
    kl_weight = min(1.0, current_epoch / total_epochs)  # Gradually scale KL weight from 0 → 1
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, kl_weight  # Return KL weight for monitoring

def gaussian_log_prob(z, mu, logvar):
    """
    log N(z | mu, exp(logvar)) i.i.f. over the last dimension
    """
    const_term = 0.5 * z.shape[-1] * np.log(2 * np.pi)
    inv_var = tf.exp(-logvar)
    tmp = tf.reduce_sum(inv_var * tf.square(z - mu), axis=-1)
    log_det = 0.5 * tf.reduce_sum(logvar, axis=-1)
    return tf.cast(const_term, tf.float32) + 0.5 * tmp + log_det

def extract_latent_representations(model, dataset):
    """Pass data through the encoder to get latent embeddings.
    Updated to handle dataset with 5 elements instead of 6."""
    latent_vectors = []
    test_ids = []

    for raw_in, rms_in, psd_in, mask_in, test_id_in in dataset:
        # 🔹 Convert test_id_in to proper format
        if isinstance(test_id_in, tf.Tensor):
            test_id_in = test_id_in.numpy()
        if isinstance(test_id_in, np.ndarray):
            test_id_in = test_id_in.flatten().tolist()

        # Get latent representations from model, adapting to your specific model
        # This assumes your model's encoder can be called directly to get mu and logvar
        mu_q, logvar_q = model.raw_encoder(raw_in, training=False)
        
        latent_vectors.append(mu_q.numpy())
        test_ids.append(test_id_in)

    return np.concatenate(latent_vectors, axis=0), np.concatenate(test_ids, axis=0)

def reduce_latent_dim_umap(latent_vectors):
    """Reduces latent space dimensionality to 3D using UMAP."""
    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=100)
    latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)  # Flatten to 2D
    latent_3d = reducer.fit_transform(latent_vectors)
    return latent_3d

def plot_latent_space_3d(latent_3d, test_ids, output_file="latent_space.html"):
    """Plots and saves a 3D UMAP visualization of the latent space with a continuous color gradient over Test ID."""

    # Create a Pandas DataFrame for Plotly
    df = pd.DataFrame(latent_3d, columns=["UMAP_1", "UMAP_2", "UMAP_3"])
    df["Test ID"] = test_ids  # Store original Test IDs

    # Ensure Test ID is treated as a continuous variable
    df["Test ID"] = pd.to_numeric(df["Test ID"], errors="coerce")  # Convert to numeric in case of issues

    # Normalize Test IDs for a smooth color gradient
    df["Test ID Normalized"] = (df["Test ID"] - df["Test ID"].min()) / (df["Test ID"].max() - df["Test ID"].min())

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

def plot_latent_histograms(latent_vectors):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):  # Plot 3 random latent dimensions
        axes[i].hist(latent_vectors[:, i], bins=50, alpha=0.75)
        axes[i].set_title(f"Latent Dimension {i}")
    plt.show()

def estimate_mog_components(latent_vectors, max_components=10):
    bics = []
    for n in range(1, max_components):
        gmm = GaussianMixture(n_components=n, covariance_type="full")
        gmm.fit(latent_vectors)
        bics.append(gmm.bic(latent_vectors))  # Lower BIC is better
    plt.plot(range(1, max_components), bics, marker='o')
    plt.xlabel("Number of Mixture Components")
    plt.ylabel("BIC Score")
    plt.title("Estimating Best MoG Components for Z")
    plt.show()

#-------------------- Training -------------------------
def train_improved_mmvm_vae(
    model, 
    train_dataset, 
    val_dataset, 
    optimizer, 
    num_epochs=100, 
    patience=10,
    beta_schedule='cyclical',
    descriptor_loss_fn=None
):
    """
    Improved train loop for MMVM VAE with true mixture-of-experts prior.
    Updated to use descriptor data instead of mask data.
    
    Args:
        model: MMVM VAE model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        beta_schedule: Type of beta schedule ('linear', 'exponential', 'cyclical')
        descriptor_loss_fn: Custom descriptor loss function (if None, use weighted_descriptor_mse_loss)
    
    Returns:
        Training and validation metrics
    """
    # Define loss functions
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    if descriptor_loss_fn is None:
        descriptor_loss_fn = weighted_descriptor_mse_loss
    
    all_trainables = model.trainable_variables

    # Initialize metrics tracking
    train_total_losses = []
    train_recon_losses = []
    train_ts_losses = []
    train_desc_losses = []
    train_js_losses = []

    val_total_losses = []
    val_recon_losses = []
    val_ts_losses = []
    val_desc_losses = []
    val_js_losses = []

    # For early stopping
    best_val_loss = float('inf')
    no_improvement_count = 0

    print("🔄 Starting Training with Improved Mixture-of-Experts Prior...")

    for epoch in range(num_epochs):
        # Get beta value for this epoch
        beta = get_beta_schedule(epoch, num_epochs, beta_schedule)
        
        # Get dynamic loss weights
        desc_weight = dynamic_mask_loss_weight(epoch, num_epochs)  # Reusing the function but for descriptor weight
        ts_weight = 1.0 - desc_weight
        
        print(f"Epoch {epoch+1}/{num_epochs} | Beta: {beta:.6f} | Descriptor Weight: {desc_weight:.2f}")
        
        # Training loop
        epoch_train_total = 0.0
        epoch_train_recon = 0.0
        epoch_train_ts = 0.0
        epoch_train_desc = 0.0
        epoch_train_js = 0.0
        train_steps = 0

        # Modified to handle descriptor data
        for step, (raw_in, rms_in, psd_in, descriptor_in, test_id_in) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # Forward pass through the model
                recon_ts, recon_desc, (all_mus, all_logvars, js_div) = model(
                    raw_in, rms_in, psd_in, descriptor_in, test_id_in,
                    training=True
                )
                
                # Time-series reconstruction loss
                loss_ts = mse_loss_fn(raw_in, recon_ts)
                
                # Descriptor reconstruction loss
                desc_recon_loss = descriptor_loss_fn(descriptor_in, recon_desc)
                
                # Combined reconstruction loss with dynamic weighting
                recon_loss = ts_weight * loss_ts + desc_weight * desc_recon_loss
                
                # Total loss = reconstruction loss + beta * JS divergence
                total_loss = recon_loss + beta * js_div

            # Gradient clipping to prevent extreme updates
            grads = tape.gradient(total_loss, all_trainables)
            # Clip gradients to prevent exploding gradients
            clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(clipped_grads, all_trainables))

            # Track losses
            epoch_train_total += total_loss.numpy()
            epoch_train_recon += recon_loss.numpy()
            epoch_train_ts += loss_ts.numpy()
            epoch_train_desc += desc_recon_loss.numpy()
            epoch_train_js += js_div.numpy()
            train_steps += 1
            
            # Print progress every 20 steps
            if step % 20 == 0:
                print(f"  Step {step}/{len(train_dataset)}", end="\r")

        # Compute average training losses for the epoch
        if train_steps > 0:
            train_total_losses.append(epoch_train_total / train_steps)
            train_recon_losses.append(epoch_train_recon / train_steps)
            train_ts_losses.append(epoch_train_ts / train_steps)
            train_desc_losses.append(epoch_train_desc / train_steps)
            train_js_losses.append(epoch_train_js / train_steps)
        
        # Debug information
        print(f"Train Loss: {train_total_losses[-1]:.4f} | "
              f"Recon: {train_recon_losses[-1]:.4f} | "
              f"TS: {train_ts_losses[-1]:.4f} | "
              f"Desc: {train_desc_losses[-1]:.4f} | " 
              f"JS: {train_js_losses[-1]:.4f}")

        # Validation loop
        epoch_val_total, epoch_val_recon = 0.0, 0.0
        epoch_val_ts, epoch_val_desc = 0.0, 0.0
        epoch_val_js = 0.0
        val_steps = 0

        # Modified to handle descriptor data
        for step, (raw_in, rms_in, psd_in, descriptor_in, test_id_in) in enumerate(val_dataset):
            # Forward pass
            recon_ts, recon_desc, (all_mus, all_logvars, js_div) = model(
                raw_in, rms_in, psd_in, descriptor_in, test_id_in, 
                training=False
            )

            # Time-series reconstruction loss
            loss_ts = mse_loss_fn(raw_in, recon_ts)
            
            # Descriptor reconstruction loss
            desc_recon_loss = descriptor_loss_fn(descriptor_in, recon_desc)
            
            # Combined reconstruction loss with the same weighting
            recon_loss = ts_weight * loss_ts + desc_weight * desc_recon_loss
            
            # Total loss with the same Beta value
            total_loss = recon_loss + beta * js_div

            # Track validation losses
            epoch_val_total += total_loss.numpy()
            epoch_val_recon += recon_loss.numpy()
            epoch_val_ts += loss_ts.numpy()
            epoch_val_desc += desc_recon_loss.numpy()
            epoch_val_js += js_div.numpy()
            val_steps += 1

        # Compute average validation losses
        if val_steps > 0:
            val_total_losses.append(epoch_val_total / val_steps)
            val_recon_losses.append(epoch_val_recon / val_steps)
            val_ts_losses.append(epoch_val_ts / val_steps)
            val_desc_losses.append(epoch_val_desc / val_steps)
            val_js_losses.append(epoch_val_js / val_steps)

        print(f"  🔵 Val => Total: {val_total_losses[-1]:.4f} | "
              f"Recon: {val_recon_losses[-1]:.4f} | "
              f"TS: {val_ts_losses[-1]:.4f} | "
              f"Desc: {val_desc_losses[-1]:.4f} | "
              f"JS: {val_js_losses[-1]:.4f}")
              
        # Improved monitoring and debugging: check if descriptor loss is stuck
        if epoch > 10 and abs(val_desc_losses[-1] - val_desc_losses[-2]) < 1e-4:
            print(f"⚠️ Warning: Descriptor loss appears to be plateauing: {val_desc_losses[-1]:.4f}")

        # -------------- Early Stopping --------------
        current_val_loss = val_total_losses[-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            no_improvement_count = 0
            # Save best model weights
            model.save_weights("results/best_mmvm_model_weights.h5")
            print("✅ Saved best model weights")
        else:
            no_improvement_count += 1
            print(f"🚨 No improvement for {no_improvement_count}/{patience} epochs.")

        if no_improvement_count >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.")
            break

    return {
        'train_total': train_total_losses,
        'train_recon': train_recon_losses,
        'train_ts': train_ts_losses,
        'train_desc': train_desc_losses,
        'train_js': train_js_losses,
        'val_total': val_total_losses,
        'val_recon': val_recon_losses,
        'val_ts': val_ts_losses,
        'val_desc': val_desc_losses,
        'val_js': val_js_losses
    }

#--------------------- Plots ---------------------------
def plot_detailed_training_curves(metrics):
    """
    Plot detailed training curves with improved visualization.
    Updated to show descriptor loss instead of mask loss.
    
    Args:
        metrics: Dictionary of training and validation metrics
    """
    epochs = list(range(1, len(metrics['train_total']) + 1))
    
    # Create a 2x2 grid of plots
    fig = go.Figure()
    
    # 1. Total Loss
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_total'], mode='lines+markers', name="Train Total", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['val_total'], mode='lines+markers', name="Val Total", line=dict(color='red')))
    fig.update_layout(title="Total Loss vs Epochs", xaxis_title="Epoch", yaxis_title="Loss")
    pio.write_html(fig, file="train_val_total_loss.html", auto_open=True)
    
    # 2. Reconstruction Losses
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_recon'], mode='lines+markers', name="Train Recon", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['val_recon'], mode='lines+markers', name="Val Recon", line=dict(color='red')))
    fig.update_layout(title="Reconstruction Loss vs Epochs", xaxis_title="Epoch", yaxis_title="Loss")
    pio.write_html(fig, file="train_val_recon_loss.html", auto_open=True)
    
    # 3. Time-Series & Descriptor Losses
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_ts'], mode='lines+markers', name="Train TS", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['val_ts'], mode='lines+markers', name="Val TS", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_desc'], mode='lines+markers', name="Train Desc", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['val_desc'], mode='lines+markers', name="Val Desc", line=dict(color='orange')))
    fig.update_layout(title="TS & Descriptor Losses vs Epochs", xaxis_title="Epoch", yaxis_title="Loss")
    pio.write_html(fig, file="train_val_ts_desc_loss.html", auto_open=True)
    
    # 4. JS Divergence
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_js'], mode='lines+markers', name="Train JS", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs, y=metrics['val_js'], mode='lines+markers', name="Val JS", line=dict(color='red')))
    fig.update_layout(title="JS Divergence vs Epochs", xaxis_title="Epoch", yaxis_title="JS Divergence")
    pio.write_html(fig, file="train_val_js_div.html", auto_open=True)

def batch_reconstruct_masks(descriptors, image_shape=(256, 768), line_thickness=1):
    """
    Convert batched descriptor data back to mask image format for visualization
    using the data_loader.reconstruct_mask_from_descriptors function.
    Filters out zero-padded rows before passing to the reconstruction function.
    
    Args:
        descriptors: Tensor or array of shape [batch, max_num_cracks, desc_length]
        image_shape: Shape of the output mask (height, width)
        line_thickness: Thickness of the lines in the reconstructed mask
        
    Returns:
        Reconstructed binary mask of shape [batch, height, width, 1]
    """
    # Process one batch at a time
    batch_size = descriptors.shape[0]
    masks = []
    
    for b in range(batch_size):
        # Process each crack descriptor batch
        desc_batch = descriptors[b]
        
        # Find valid descriptors (non-zero rows)
        # A row is considered valid if it has any non-zero elements
        valid_mask = np.any(desc_batch != 0, axis=1)
        valid_descriptors = desc_batch[valid_mask]
        
        # Check if we have any valid descriptors before reconstruction
        if len(valid_descriptors) == 0:
            # No valid descriptors, return an empty mask
            mask = np.zeros(image_shape, dtype=np.uint8)
        else:
            try:
                # Use the data_loader function to reconstruct the mask
                mask = data_loader.reconstruct_mask_from_descriptors(
                    valid_descriptors, 
                    image_shape, 
                    line_thickness=line_thickness
                )
            except Exception as e:
                print(f"Error in reconstruct_mask_from_descriptors: {e}")
                # Fallback to empty mask in case of error
                mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Convert to float32 and add channel dimension if needed
        mask = mask.astype(np.float32)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
            
        masks.append(mask)
    
    # Stack all masks into a batch
    return np.stack(masks, axis=0)

def visualize_reconstructions(model, val_dataset, num_samples=5, line_thickness=1):
    """
    Visualize original vs. reconstructed time series and masks (from descriptors).
    Uses data_loader's reconstruct_mask_from_descriptors function.
    
    Args:
        model: Trained MMVM_VAE model
        val_dataset: Validation dataset
        num_samples: Number of samples to visualize
        line_thickness: Thickness of the lines in reconstructed masks
    """
    # Select samples from validation dataset
    for i, (raw_batch, rms_batch, psd_batch, desc_batch, _) in enumerate(val_dataset.take(1)):
        for j in range(min(num_samples, raw_batch.shape[0])):
            # Get sample
            raw_sample = tf.expand_dims(raw_batch[j], 0)
            rms_sample = tf.expand_dims(rms_batch[j], 0)
            psd_sample = tf.expand_dims(psd_batch[j], 0)
            desc_sample = tf.expand_dims(desc_batch[j], 0)
            
            # Get reconstructions
            recon_ts, recon_desc, _ = model(
                raw_sample, rms_sample, psd_sample, desc_sample, 
                tf.constant([[0]]), training=False
            )
            
            # Visualize time series
            raw_mean = tf.reduce_mean(raw_sample[0], axis=1).numpy()
            recon_mean = tf.reduce_mean(recon_ts[0], axis=1).numpy()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=raw_mean, mode='lines', name='Original', 
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                y=recon_mean, mode='lines', name='Reconstructed', 
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f"Time Series Reconstruction Sample {j+1}",
                xaxis_title="Time",
                yaxis_title="Amplitude"
            )
            
            fig.write_html(f"results/visualizations/ts_recon_sample_{j+1}.html")
            
            # Convert descriptors to masks for visualization using data_loader function
            original_mask = batch_reconstruct_masks(
                desc_sample.numpy(), 
                image_shape=(256, 768), 
                line_thickness=line_thickness
            )
            
            recon_mask = batch_reconstruct_masks(
                recon_desc.numpy(), 
                image_shape=(256, 768), 
                line_thickness=line_thickness
            )
            
            # Visualize masks
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=original_mask[0, :, :, 0], 
                colorscale='Blues',
                showscale=False
            ))
            fig.update_layout(title=f"Original Mask from Descriptors - Sample {j+1}")
            fig.write_html(f"results/visualizations/orig_mask_sample_{j+1}.html")
            
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=recon_mask[0, :, :, 0], 
                colorscale='Reds',
                showscale=False
            ))
            fig.update_layout(title=f"Reconstructed Mask from Descriptors - Sample {j+1}")
            fig.write_html(f"results/visualizations/recon_mask_sample_{j+1}.html")

def plot_training_curves(train_total, train_recon, train_kl, val_total, val_recon, val_kl):
    epochs = list(range(1, len(train_total) + 1))

    # Total Loss Plot
    fig_total = go.Figure()
    fig_total.add_trace(go.Scatter(x=epochs, y=train_total, mode='lines+markers', name="Train Total"))
    fig_total.add_trace(go.Scatter(x=epochs, y=val_total, mode='lines+markers', name="Val Total"))
    fig_total.update_layout(title="Total Loss vs Epochs", xaxis_title="Epoch", yaxis_title="Loss")
    pio.write_html(fig_total, file="train_val_total_loss.html", auto_open=True)

    # Reconstruction Loss Plot
    fig_recon = go.Figure()
    fig_recon.add_trace(go.Scatter(x=epochs, y=train_recon, mode='lines+markers', name="Train Recon"))
    fig_recon.add_trace(go.Scatter(x=epochs, y=val_recon, mode='lines+markers', name="Val Recon"))
    fig_recon.update_layout(title="Reconstruction Loss vs Epochs", xaxis_title="Epoch", yaxis_title="Loss")
    pio.write_html(fig_recon, file="train_val_recon_loss.html", auto_open=True)

    # KL Loss Plot
    fig_kl = go.Figure()
    fig_kl.add_trace(go.Scatter(x=epochs, y=train_kl, mode='lines+markers', name="Train KL"))
    fig_kl.add_trace(go.Scatter(x=epochs, y=val_kl, mode='lines+markers', name="Val KL"))
    fig_kl.update_layout(title="KL Loss vs Epochs", xaxis_title="Epoch", yaxis_title="Loss")
    pio.write_html(fig_kl, file="train_val_kl_loss.html", auto_open=True)

def plot_generated_samples(model, latent_dim, num_samples=5):
    for i in range(num_samples):
        z_rand = tf.random.normal(shape=(1, latent_dim))
        recon_ts, recon_mask = model.generate(z_rand)
        
        # Plot raw time-series (mean over channels)
        ts = recon_ts.numpy().squeeze(0)  # shape: (1000, 12)
        ts_mean = ts.mean(axis=1)
        fig_ts = go.Figure(data=go.Scatter(x=list(range(ts_mean.shape[0])), y=ts_mean))
        fig_ts.update_layout(title=f"Generated Time-Series Sample {i}", xaxis_title="Time", yaxis_title="Amplitude")
        pio.write_html(fig_ts, file=f"generated_ts_sample_{i}.html", auto_open=True)
        
        # Plot mask as a heatmap
        mask = recon_mask.numpy().squeeze(0)  # shape: (256, 768)
        fig_mask = go.Figure(data=go.Heatmap(z=mask))
        fig_mask.update_layout(title=f"Generated Mask Sample {i}")
        pio.write_html(fig_mask, file=f"generated_mask_sample_{i}.html", auto_open=True)

def visualize_reconstruction(true_mask, pred_ellipses, pred_polygons):
    """
    Visualizes:
    - Original damage mask
    - Predicted ellipses & polygons
    - Reconstructed mask
    """
    reconstructed_mask = np.zeros_like(true_mask)

    for ellipse in pred_ellipses:
        cv2.ellipse(reconstructed_mask, ellipse, 1, -1)

    for poly in pred_polygons:
        cv2.fillPoly(reconstructed_mask, [np.array(poly, np.int32)], 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(true_mask, cmap='gray')
    axes[0].set_title("Original Mask")

    axes[1].imshow(reconstructed_mask, cmap='gray')
    axes[1].set_title("Reconstructed Mask (Ellipses & Polygons)")

    axes[2].imshow(true_mask, cmap='gray')
    axes[2].imshow(reconstructed_mask, cmap='jet', alpha=0.5)
    axes[2].set_title("Overlay: True vs. Reconstructed")

    for ax in axes:
        ax.axis("off")

    plt.show()

# Global variable to store the trained VAE model for external access.
vae_model = None

def encoder(ts_sample, rms_sample=None, psd_sample=None, mask_sample=None):
    """
    Encodes samples from various modalities into their latent representations.
    
    Args:
        ts_sample: Time-series sample of shape [batch, 1000, 12]
        rms_sample: RMS features of shape [batch, 12] (optional)
        psd_sample: PSD features of shape [batch, freq_bins, 12] (optional)
        mask_sample: Either ellipses and polygons or image-like input (optional)
    
    Returns:
        Dictionary of modality-specific latent distributions (mu, logvar)
    """
    global vae_model
    if vae_model is None:
        raise ValueError("MMVM VAE model has not been trained or loaded.")
    
    latent_dists = {}
    
    # Encode each available modality
    if ts_sample is not None:
        latent_dists['ts'] = vae_model.raw_encoder(ts_sample)
    
    if rms_sample is not None:
        latent_dists['rms'] = vae_model.rms_encoder(rms_sample)
        
    if psd_sample is not None:
        latent_dists['psd'] = vae_model.psd_encoder(psd_sample)
    
    if mask_sample is not None:
        latent_dists['mask'] = vae_model.mask_encoder(mask_sample)
    
    return latent_dists

def decode(z_dict):
    """
    Decodes latent vectors from multiple modalities.
    
    Args:
        z_dict: Dictionary of modality-specific latent vectors
        
    Returns:
        Dictionary of reconstructed modalities
    """
    global vae_model
    if vae_model is None:
        raise ValueError("MMVM VAE model has not been trained or loaded.")
    
    reconstructions = {}
    
    # Decode each modality if its latent vector is provided
    if 'ts' in z_dict:
        reconstructions['ts'] = vae_model.raw_decoder(z_dict['ts'])
    
    if 'mask' in z_dict:
        reconstructions['mask'] = vae_model.mask_decoder(z_dict['mask'])
    
    return reconstructions

def load_trained_model(weights_path):
    """
    Load a trained VAE model from the given weights file and assign it to the global variable.
    This function builds the model (by calling it with dummy inputs) before loading the weights.
    
    Args:
        weights_path: Path to the saved weights file.
    """
    global vae_model
    latent_dim = 128  # Ensure this matches the latent_dim used in training.
    model = MMVM_VAE(latent_dim, feature_dim=128)
    
    # Build the model's variables by calling it with dummy inputs.
    dummy_ts = tf.zeros((1, 12000, 12), dtype=tf.float32)
    dummy_mask = tf.zeros((1, 256, 768), dtype=tf.float32)
    _ = model(dummy_ts, dummy_mask)
    
    # Now load the weights.
    model.load_weights(weights_path)
    vae_model = model
    print("Trained VAE model loaded successfully.")

def save_trained_model(weights_path):
    """
    Save the current trained VAE model's weights to the given path.
    
    Args:
        weights_path: Path to save the weights.
    """
    global vae_model
    if vae_model is None:
        raise ValueError("No trained model to save.")
    vae_model.save_weights(weights_path)
    print("Trained VAE model saved successfully.")


# ----- Main Script -----
def main():
    # ------------------------ GPU & Memory Setup ------------------------
    configure_gpu()
    clear_gpu_memory()
    print_detailed_memory()
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # ------------------------ 1) Load Data ------------------------
    # Define params for data_loader.load_data
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
    print("Loading data with parameters:")
    for key, value in params.items():
        print(f"  - {key}: {value}")
    accel_dict, crack_dict, binary_masks, skeletons, padded_dict = data_loader.load_data(params)
    
    # Get dimensions from the padded descriptors
    sample_desc = next(iter(padded_dict.values()))
    max_num_cracks = sample_desc.shape[0]
    desc_length = sample_desc.shape[1]
    print(f"Loaded data for {len(accel_dict)} tests")
    print(f"Descriptor dimensions: {max_num_cracks} cracks x {desc_length} features")
    
    # Check padded_dict values for descriptor data
    sample_desc = next(iter(padded_dict.values()))
    print(f"Loaded data for {len(accel_dict)} tests")
    print(f"Descriptor shape sample: {sample_desc.shape}")  # Should be (max_num_cracks, desc_length)
    
    # Store dimensions for model initialization
    max_num_cracks = sample_desc.shape[0]
    desc_length = sample_desc.shape[1]
    print(f"Max number of cracks: {max_num_cracks}, Descriptor length: {desc_length}")
    
    # Use the updated segment_and_transform function with descriptor data
    print("Segmenting data...")
    gen = segment_and_transform(accel_dict, padded_dict)
    try:
        raw_segments, rms_segments, psd_segments, descriptor_segments, test_ids = next(gen)
    except StopIteration:
        raise ValueError("segment_and_transform() did not yield any data.")

    print("Finished segmenting data.")
    print(f"Data shapes after segmentation:")
    print(f"  Raw: {raw_segments.shape}")     
    print(f"  RMS: {rms_segments.shape}")      
    print(f"  PSD: {psd_segments.shape}")      
    print(f"  Descriptor: {descriptor_segments.shape}")    
    print(f"  IDs: {test_ids.shape}")

    # ------------------------ 2) Convert to Float32 ------------------------
    raw_segments = raw_segments.astype(np.float32)
    rms_segments = rms_segments.astype(np.float32)
    psd_segments = psd_segments.astype(np.float32)
    descriptor_segments = descriptor_segments.astype(np.float32)
    test_ids = test_ids.astype(np.float32)

    # ------------------------ 3) Shuffle and Split into Train/Val ------------------------
    N = raw_segments.shape[0]
    indices = np.random.permutation(N)
    train_size = int(0.8 * N)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_raw = raw_segments[train_idx]
    train_rms = rms_segments[train_idx]
    train_psd = psd_segments[train_idx]
    train_desc = descriptor_segments[train_idx]
    train_ids = test_ids[train_idx]

    val_raw = raw_segments[val_idx]
    val_rms = rms_segments[val_idx]
    val_psd = psd_segments[val_idx]
    val_desc = descriptor_segments[val_idx]
    val_ids = test_ids[val_idx]

    # ------------------------ 4) Build tf.data Datasets ------------------------
    BATCH_SIZE = 32
    
    # Use updated create_tf_dataset function
    train_dataset = create_tf_dataset(
        train_raw, train_rms, train_psd, train_desc, train_ids, batch_size=BATCH_SIZE
    )
    val_dataset = create_tf_dataset(
        val_raw, val_rms, val_psd, val_desc, val_ids, batch_size=BATCH_SIZE
    )

    print(f"Train batches: {len(train_dataset)}")
    print(f"Val batches:   {len(val_dataset)}")

    # ------------------------ 5) Build Improved Model ------------------------
    latent_dim = 256
    
    # Use our new MMVM VAE with descriptor support
    print(f"Building model with descriptor dimensions: {max_num_cracks} cracks x {desc_length} features")
    model = MMVM_VAE(latent_dim, max_num_cracks=max_num_cracks, desc_length=desc_length)

    # Learning Rate Scheduler with Cosine Decay
    initial_learning_rate = 2e-4
    decay_steps = 10000
    
    # Cosine decay schedule with warm restart
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        decay_steps,
        t_mul=2.0,    # Double period after each restart
        m_mul=0.9,    # Reduce max LR by 10% after each restart
        alpha=1e-5    # Minimum LR factor
    )

    # Adam optimizer with improved parameters
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6   # Increased epsilon for better numerical stability
    )

    print_detailed_memory()

    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/model_checkpoints", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)

    # Use combined descriptor loss
    custom_descriptor_loss = lambda y_true, y_pred: combined_descriptor_loss(
        y_true, y_pred, 
        mse_weight=0.7, 
        feature_weight=0.3
    )

    # Train using our improved training function
    print("Starting training...")
    training_metrics = train_improved_mmvm_vae(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        num_epochs=200,
        patience=100,
        beta_schedule='cyclical',
        descriptor_loss_fn=custom_descriptor_loss
    )

    # ------------------------ 6) Visualize Training Results ------------------------
    # Save detailed training metrics to a file
    np.save("results/training_metrics.npy", training_metrics)
    
    # Plot detailed training curves
    plot_detailed_training_curves(training_metrics)

    # ------------------------ 7) Extract and Visualize Latent Space ------------------------
    # Extract latent vectors from trained model
    latent_vectors, test_ids_arr = extract_latent_representations(model, train_dataset)
    
    # Dimensionality reduction with UMAP
    latent_3d = reduce_latent_dim_umap(latent_vectors)
    
    # Plot 3D visualization
    plot_latent_space_3d(latent_3d, test_ids_arr, output_file="results/visualizations/latent_space_3d.html")
    
    # Line thickness parameter for mask reconstruction
    line_thickness = params.get('line_thickness', 1)
    
    # ------------------------ 8) Evaluate Reconstructions ------------------------
    visualize_reconstructions(model, val_dataset, num_samples=5, line_thickness=line_thickness)
    
    # ------------------------ 9) Generate New Samples ------------------------
    # Generate random samples from the latent space
    num_gen_samples = 5
    for i in range(num_gen_samples):
        # Random latent vector
        z = tf.random.normal(shape=(1, latent_dim))
        
        # Generate both time series and descriptors
        gen_ts = model.generate(modality='ts', conditioning_latent=z)
        gen_desc = model.generate(modality='desc', conditioning_latent=z)
        
        # Visualize time series
        fig = go.Figure()
        gen_ts_mean = tf.reduce_mean(gen_ts[0], axis=1).numpy()
        fig.add_trace(go.Scatter(y=gen_ts_mean, mode='lines'))
        fig.update_layout(title=f"Generated Time Series - Sample {i+1}")
        fig.write_html(f"results/visualizations/gen_ts_sample_{i+1}.html")
        
        # Convert generated descriptors to mask for visualization using data_loader function
        gen_mask = batch_reconstruct_masks(
            gen_desc.numpy(), 
            image_shape=(256, 768), 
            line_thickness=line_thickness
        )
        
        # Visualize generated mask
        fig = go.Figure(data=go.Heatmap(z=gen_mask[0, :, :, 0], colorscale='Viridis'))
        fig.update_layout(title=f"Generated Mask from Descriptors - Sample {i+1}")
        fig.write_html(f"results/visualizations/gen_mask_sample_{i+1}.html")

    print("✅ Finished training and evaluation. Results saved in 'results/' directory.")
    print(f"   Mask visualizations created using data_loader.reconstruct_mask_from_descriptors with line_thickness={line_thickness}")

if __name__ == "__main__":
    main()
