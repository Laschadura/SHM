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

# Import your data loader
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
def segment_and_transform(accel_dict, mask_dict, chunk_size=1, sample_rate=200, segment_duration=5.0, percentile=99):
    """Generator version of segment_and_transform that yields data in chunks"""
    
    window_size = int(sample_rate * segment_duration)
    half_window = window_size // 2
    
    # Process test_ids in chunks
    test_ids = list(accel_dict.keys())
    for i in range(0, len(test_ids), chunk_size):
        chunk_ids = test_ids[i:i + chunk_size]
        
        raw_segments = []
        fft_segments = []
        rms_segments = []
        psd_segments = []
        mask_segments = []
        test_ids_out = []
        
        for test_id in chunk_ids:
            all_ts = accel_dict[test_id]
            mask_data = mask_dict[test_id]
            
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
                    seg_fft = compute_fft(segment_raw, sample_rate)
                    seg_rms = compute_rms(segment_raw)
                    seg_psd = compute_psd(segment_raw, sample_rate)

                    raw_segments.append(segment_raw)
                    fft_segments.append(seg_fft)
                    rms_segments.append(seg_rms)
                    psd_segments.append(seg_psd)
                    mask_segments.append(mask_data)

                    # **Ensure `test_id` is an integer**
                    test_ids_out.append(int(test_id))  
        
        # Convert chunk to arrays and yield
        if raw_segments:  # Only yield if we have data
            yield (np.array(raw_segments, dtype=np.float32),
                  np.array(fft_segments, dtype=np.float32),
                  np.array(rms_segments, dtype=np.float32),
                  np.array(psd_segments, dtype=np.float32),
                  np.array(mask_segments, dtype=np.float32),
                  np.array(test_ids_out, dtype=np.int32))  # Ensure test_id is int32
           
def create_tf_dataset(raw_segments, fft_segments, rms_segments, psd_segments, mask_segments, test_ids, batch_size=8):
    print(f"Creating dataset with shapes:")
    print(f"  Raw segments: {raw_segments.shape}")
    print(f"  FFT segments: {fft_segments.shape}")
    print(f"  RMS segments: {rms_segments.shape}")
    print(f"  PSD segments: {psd_segments.shape}")
    print(f"  Mask segments: {mask_segments.shape}")
    print(f"  Test IDs: {test_ids.shape}\n")

    # Ensure test_ids are integers
    test_ids = np.array(test_ids, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((
        raw_segments.astype(np.float32),
        fft_segments.astype(np.float32),
        rms_segments.astype(np.float32),
        psd_segments.astype(np.float32),
        mask_segments.astype(np.float32),
        test_ids
    ))
    
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Verify dataset
    for batch in dataset.take(1):
        print("\nFirst batch shapes:")
        print(f"  Raw: {batch[0].shape}")
        print(f"  FFT: {batch[1].shape}")
        print(f"  RMS: {batch[2].shape}")
        print(f"  PSD: {batch[3].shape}")
        print(f"  Mask: {batch[4].shape}")
        print(f"  Test ID: {batch[5].shape}")

    return dataset

def compute_fft(segment, sample_rate):
    """
    Example: compute a simple FFT over time for each channel
    `segment` shape: (window_size, num_channels)
    Return shape could be (window_size//2, num_channels) if you keep only the positive frequencies
    Adjust as needed.
    """
    # Example using np.fft. This is just a placeholder.
    # Real code might do windowing, etc.
    fft_out = np.fft.rfft(segment, axis=0)
    fft_out = np.abs(fft_out)  # or complex if you prefer
    return fft_out

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

def mask_to_ellipses_and_polygons(mask):
    """
    Convert a binary mask into an adaptive set of ellipses and polygons.
    Handles TensorFlow tensors by converting them to NumPy arrays.

    Returns:
        - A list of fitted ellipses: [(x, y, major_axis, minor_axis, angle), ...]
        - A list of polygon points (if needed)
    """
    # Ensure mask is in NumPy format
    if isinstance(mask, tf.Tensor):
        mask = mask.numpy()  # Convert from TensorFlow tensor to NumPy

    mask = (mask > 0.5).astype(np.uint8)  # Convert to binary (0, 1) format for OpenCV

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return [], []  # No damage detected

    ellipses, polygons = [], []
    reconstructed_mask = np.zeros_like(mask)

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]  # Focus on top 3 regions

    for contour in contours:
        num_ellipses = 1  # Start with a single ellipse
        step = max(len(contour) // num_ellipses, 5)
        sub_contour = contour[::step].astype(np.float32)

        while num_ellipses <= 30:  # Adaptive number of ellipses
            if len(sub_contour) >= 5:
                ellipse = cv2.fitEllipse(sub_contour)
                ellipses.append(ellipse)

                # Draw ellipses to reconstruct mask
                cv2.ellipse(reconstructed_mask, ellipse, 1, -1)

                # Compute IoU
                intersection = np.logical_and(mask, reconstructed_mask)
                union = np.logical_or(mask, reconstructed_mask)
                iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

                # Stop adding ellipses if IoU is high enough
                if iou >= 0.9:
                    break

            num_ellipses += 1
            step = max(len(contour) // num_ellipses, 5)
            sub_contour = contour[::step].astype(np.float32)

        # If ellipses fail to reach target accuracy, use polygon approximation
        if iou < 0.9:
            epsilon = 0.02 * cv2.arcLength(contour, True)  # Adjust accuracy
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            polygons.append(approx_polygon)

    return ellipses, polygons

def preprocess_all_masks(mask_dict):
    """
    Precomputes ellipse and polygon encodings for unique test IDs.
    Ensures all masks are properly formatted as NumPy arrays.

    Returns:
        preprocessed_masks (dict): {test_id: {"ellipses": ellipses, "polygons": polygons}}
    """
    preprocessed_masks = {}

    for test_id, mask_data in mask_dict.items():
        # ---- Extract Only the Mask (Ignore Extra Tuple Elements) ----
        if isinstance(mask_data, tuple):  
            mask = mask_data[0]  
        else:
            mask = mask_data  

        # ---- Ensure the Mask is a NumPy Array ----
        if isinstance(mask, list):  
            mask = np.array(mask)  

        if isinstance(mask, tf.Tensor):  
            mask = mask.numpy()

        if not isinstance(mask, np.ndarray):
            print(f"[ERROR] Mask for Test ID {test_id} is not a NumPy array! Skipping...")
            preprocessed_masks[test_id] = {"ellipses": [], "polygons": []}
            continue

        # ---- Debugging Print ----
        print(f"Processing Test ID {test_id}, Mask Type: {type(mask)}, Shape: {mask.shape}")

        if len(mask.shape) == 3 and mask.shape[-1] == 1:  
            mask = mask.squeeze(-1)  

        mask = (mask > 0.5).astype(np.uint8) * 255  

        # ---- Initialize `ellipses` and `polygons` to Prevent Unpacking Error ----
        ellipses, polygons = [], []  

        if test_id in [1, 5, 10]:  # Only for a few test IDs
            plt.imshow(mask, cmap="gray")
            plt.title(f"Test ID {test_id} - Preprocessed Mask")
            plt.show()


        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print(f"[WARNING] No contours found for Test ID {test_id}. Assigning empty ellipses/polygons.")
            preprocessed_masks[test_id] = {"ellipses": [], "polygons": []}
            continue  

        reconstructed_mask = np.zeros_like(mask)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        print(f"[DEBUG] Test ID {test_id}: Found {len(contours)} contours.")

        for contour in contours:
            num_ellipses = 1  
            step = max(len(contour) // num_ellipses, 5)
            sub_contour = contour[::step].astype(np.float32)

            while num_ellipses <= 30:  
                if len(sub_contour) >= 5:
                    ellipse = cv2.fitEllipse(sub_contour)
                    ellipses.append(ellipse)

                    cv2.ellipse(reconstructed_mask, ellipse, 1, -1)

                    intersection = np.logical_and(mask, reconstructed_mask)
                    union = np.logical_or(mask, reconstructed_mask)
                    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

                    if iou >= 0.9:
                        break

                num_ellipses += 1
                step = max(len(contour) // num_ellipses, 5)
                sub_contour = contour[::step].astype(np.float32)

            if iou < 0.9:
                epsilon = 0.02 * cv2.arcLength(contour, True)  
                approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
                polygons.append(approx_polygon)

        # ---- Store `ellipses` and `polygons` as a Dictionary ----
        preprocessed_masks[test_id] = {"ellipses": ellipses, "polygons": polygons}

    return preprocessed_masks

# ----- Loss Functions -----
def dice_loss(pred, target, smooth=1e-6):
    """
    Dice loss with epsilon to avoid divide-by-zero.
    """
    # Assert that incoming pred/target don't contain NaNs/infs
    tf.debugging.assert_all_finite(pred, "dice_loss: pred contains NaN or Inf")
    tf.debugging.assert_all_finite(target, "dice_loss: target contains NaN or Inf")

    # Clip predictions to [0, 1] just in case
    pred = tf.clip_by_value(pred, 0.0, 1.0)

    # Flatten
    pred_flat = tf.reshape(pred, [tf.shape(pred)[0], -1])
    target_flat = tf.reshape(target, [tf.shape(target)[0], -1])

    intersection = tf.reduce_sum(pred_flat * target_flat, axis=1)
    # Add smooth to denominator and numerator to avoid zero
    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(target_flat, axis=1) + smooth
    )

    # Check that dice ratio is still finite
    tf.debugging.assert_all_finite(dice, "dice_loss: dice ratio is NaN or Inf")

    return 1.0 - tf.reduce_mean(dice)

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

def ellipse_loss(true_ellipses, pred_ellipses, max_ellipses=30):
    """
    Computes L2 loss between true and predicted ellipses.
    
    Args:
      - true_ellipses: (B, variable_num_ellipses, 5)
      - pred_ellipses: (B, max_ellipses, 5)
    
    Returns:
      - MSE loss between true and predicted ellipses
    """
    # 🔹 Check if true_ellipses is completely empty
    if tf.shape(true_ellipses)[0] == 0:
        return tf.constant(0.0, dtype=tf.float32)

    # Get current shape
    true_shape = tf.shape(true_ellipses)
    pred_shape = tf.shape(pred_ellipses)

    # Extract batch size
    batch_size = true_shape[0]

    # Handle shape mismatches
    num_true_ellipses = tf.cond(
        tf.shape(true_ellipses)[0] > 0, 
        lambda: tf.shape(true_ellipses)[1], 
        lambda: tf.constant(0, dtype=tf.int32)
    )

    num_pred_ellipses = pred_shape[1]

    if num_true_ellipses < max_ellipses:
        pad_size = max_ellipses - num_true_ellipses
        pad_tensor = tf.zeros((batch_size, pad_size, 5), dtype=tf.float32)
        true_ellipses = tf.concat([true_ellipses, pad_tensor], axis=1)
    elif num_true_ellipses > max_ellipses:
        true_ellipses = true_ellipses[:, :max_ellipses, :]

    return tf.reduce_mean(tf.square(true_ellipses - pred_ellipses))

def polygon_loss(true_polygons, pred_polygons, max_points=20):
    """
    Computes L2 loss between true and predicted polygons.
    
    Args:
      - true_polygons: (B, variable_num_points, 2)
      - pred_polygons: (B, max_points, 2)
    
    Returns:
      - MSE loss between true and predicted polygons
    """
    if tf.shape(true_polygons)[0] == 0:
        return tf.constant(0.0, dtype=tf.float32)

    true_shape = tf.shape(true_polygons)
    pred_shape = tf.shape(pred_polygons)

    batch_size = true_shape[0]

    num_true_points = tf.cond(
        tf.shape(true_polygons)[0] > 0, 
        lambda: tf.shape(true_polygons)[1], 
        lambda: tf.constant(0, dtype=tf.int32)
    )

    num_pred_points = pred_shape[1]

    if num_true_points < max_points:
        pad_size = max_points - num_true_points
        pad_tensor = tf.zeros((batch_size, pad_size, 2), dtype=tf.float32)
        true_polygons = tf.concat([true_polygons, pad_tensor], axis=1)
    elif num_true_points > max_points:
        true_polygons = true_polygons[:, :max_points, :]

    return tf.reduce_mean(tf.square(true_polygons - pred_polygons))

def mask_reconstruction_loss(true_mask, reconstructed_mask):
    """
    Computes Intersection over Union (IoU) loss.
    Ensures the final reconstructed mask aligns with the original.
    """
    intersection = tf.reduce_sum(true_mask * reconstructed_mask)
    union = tf.reduce_sum(true_mask) + tf.reduce_sum(reconstructed_mask) - intersection
    iou = intersection / (union + 1e-8)
    return 1 - iou  # IoU Loss (lower is better)

def reparameterize(mu, logvar):
    std = tf.exp(0.5 * logvar)
    eps = tf.random.normal(shape=tf.shape(std))
    return mu + eps * std

# ----- Encoder Branches -----
class RawEncoder(Model):
    """
    Convolution + LSTM encoder for raw time-series data.
    Input shape: (batch_size, 1000, 12)
    Output: A feature vector of dimension feature_dim.
    """

    def __init__(self, feature_dim):
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

        # 3) Third CNN block (optional deeper layer)
        self.conv3 = layers.Conv1D(
            filters=64,
            kernel_size=5,
            strides=2,
            padding='same',
            activation='relu'
        )
        self.dropout3 = layers.Dropout(0.3)

        # 4) Bidirectional LSTM layer
        #
        # return_sequences=False ensures we only get the final hidden state,
        # which is often enough for classification/encoding.
        self.bilstm = layers.Bidirectional(
            layers.LSTM(128, return_sequences=False)
        )

        # 5) Dense to reduce to 512 (you can change this size if desired)
        self.dense_reduce = layers.Dense(512, activation='relu')

        # 6) Final feature projection
        self.feature_layer = layers.Dense(feature_dim)

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

        # CNN block 3 (optional deeper layer)
        x = self.conv3(x)                 # -> (batch, 125, 64)
        x = self.dropout3(x, training=training)

        # LSTM layer: returns final hidden state of shape (batch, 256)
        x = self.bilstm(x, training=training)

        # Dense + final
        x = self.dense_reduce(x)          # -> (batch, 512)
        feature = self.feature_layer(x)   # -> (batch, feature_dim)
        return feature
 
class PSDEncoder(Model):
    """Encoder for PSD features.
       Input shape: (batch, freq_bins, 12), e.g., (batch, 129, 12)
       Outputs a feature vector of dimension feature_dim.
    """

    def __init__(self, feature_dim):
        super().__init__()

        # First convolutional block
        self.conv1 = layers.Conv1D(filters=16, kernel_size=5, strides=2, padding='same', activation='relu')
        self.dropout1 = layers.Dropout(0.3)

        # Second convolutional block
        self.conv2 = layers.Conv1D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu')
        self.dropout2 = layers.Dropout(0.3)

        # Third convolutional block (optional deeper layer)
        self.conv3 = layers.Conv1D(filters=64, kernel_size=5, strides=2, padding='same', activation='relu')
        self.dropout3 = layers.Dropout(0.3)

        # Global pooling to reduce to a single feature vector
        self.global_pool = layers.GlobalAveragePooling1D()

        # Dense layers for final encoding
        self.dense_reduce = layers.Dense(128, activation='relu')
        self.feature_layer = layers.Dense(feature_dim)  # Final feature vector

    def call(self, x, training=False):
        """
        x shape: (batch, freq_bins, 12), e.g., (batch, 129, 12)
        """

        x = self.conv1(x)  # (batch, ~65, 16)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)  # (batch, ~33, 32)
        x = self.dropout2(x, training=training)

        x = self.conv3(x)  # (batch, ~17, 64)
        x = self.dropout3(x, training=training)

        # Global pooling collapses the sequence dimension
        x = self.global_pool(x)  # (batch, 64)

        # Fully connected layers
        x = self.dense_reduce(x)  # (batch, 128)
        feature = self.feature_layer(x)  # (batch, feature_dim)

        return feature

class RMSEncoder(keras.Model):
    """Encoder for RMS features.
       Input shape: (batch, 12) (one value per channel)
       Outputs a feature vector of dimension feature_dim.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.dense1 = layers.Dense(16, activation='relu')
        self.drop1 = layers.Dropout(0.3)
        self.feature_layer = layers.Dense(feature_dim)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.drop1(x, training=training)
        feature = self.feature_layer(x)
        return feature  # shape: [batch, feature_dim]

class MaskEncoder(tf.keras.Model):
    def __init__(self, num_ellipses=30, num_polygon_points=10):
        super().__init__()
        self.num_ellipses = num_ellipses
        self.num_polygon_points = num_polygon_points

        self.conv1 = layers.Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(256, kernel_size=4, strides=2, padding='same', activation='relu')

        self.spp1 = layers.GlobalMaxPooling2D()
        self.spp2 = layers.AveragePooling2D(pool_size=(4, 4))
        self.spp3 = layers.AveragePooling2D(pool_size=(8, 8))

        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)

        self.dense_reduce = layers.Dense(256, activation='relu')

        # Output layers for ellipse and polygon parameters
        self.ellipse_layer = layers.Dense(self.num_ellipses * 5)  # Each ellipse has 5 params (x, y, major, minor, angle)
        self.polygon_layer = layers.Dense(self.num_polygon_points * 2)  # Each point has (x, y)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x1 = self.spp1(x)
        x2 = tf.reshape(self.spp2(x), [-1, self.spp2(x).shape[-1]])
        x3 = tf.reshape(self.spp3(x), [-1, self.spp3(x).shape[-1]])

        x_spp = tf.concat([x1, x2, x3], axis=-1)

        batch_size, h, w, c = x.shape
        x = tf.reshape(x, [batch_size, h * w, c])
        x = self.attention(x, x, x)
        x = tf.reshape(x, [batch_size, h, w, c])

        x = self.dense_reduce(x_spp)

        ellipses = self.ellipse_layer(x)  # Output ellipse parameters
        polygons = self.polygon_layer(x)  # Output polygon points

        return ellipses, polygons

# ----- Combined Encoder with Multi Branches -----
class MultiModalEncoder(keras.Model):
    """
    Aggregates feature vectors from raw, RMS, PSD, and mask encoders.
    Each sub-encoder outputs a deterministic feature vector.
    These are concatenated (along with test_id) and passed through an MLP to produce
    the final MoG parameters for the shared latent space.
    """
    def __init__(self, latent_dim, feature_dim, num_ellipses=30, num_polygon_points=10, num_mixtures=5):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.raw_enc = RawEncoder(feature_dim)
        self.rms_enc = RMSEncoder(feature_dim)
        self.psd_enc = PSDEncoder(feature_dim)
        self.mask_enc = MaskEncoder(num_ellipses=num_ellipses, num_polygon_points=num_polygon_points)

        # Aggregator: concatenate features from all 4 modalities plus test_id.
        self.fc_agg = layers.Dense(128, activation='relu')
        self.mu_layer = layers.Dense(latent_dim * num_mixtures)
        self.logvar_layer = layers.Dense(latent_dim * num_mixtures)
        self.pi_layer = layers.Dense(num_mixtures, activation="softmax")

    def call(self, raw_in, rms_in, psd_in, ellipses, polygons, test_id, training=False):
        feat_raw = self.raw_enc(raw_in, training=training)
        feat_rms = self.rms_enc(rms_in, training=training)
        feat_psd = self.psd_enc(psd_in, training=training)

        # Concatenate ellipses and polygons as a single mask encoding
        mask_encoding = tf.concat([ellipses, polygons], axis=-1)

        # Process test_id: convert to float and expand dimension => [B, 1]
        test_id = tf.cast(test_id, tf.float32)
        test_id = tf.expand_dims(test_id, axis=-1)

        # Concatenate all features: [B, (4*feature_dim + 1)]
        combined_features = tf.concat([feat_raw, feat_rms, feat_psd, mask_encoding, test_id], axis=-1)

        # Pass through aggregator MLP.
        agg = self.fc_agg(combined_features)
        mu_unshaped = self.mu_layer(agg)
        logvar_unshaped = self.logvar_layer(agg)
        pi_final = self.pi_layer(agg)

        # Reshape mu and logvar to [B, num_mixtures, latent_dim]
        latent_dim = tf.shape(mu_unshaped)[-1] // self.num_mixtures
        mu_final = tf.reshape(mu_unshaped, [-1, self.num_mixtures, latent_dim])
        logvar_final = tf.reshape(logvar_unshaped, [-1, self.num_mixtures, latent_dim])

        return mu_final, logvar_final, pi_final, combined_features

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
 
class MaskDecoder(tf.keras.Model):
    def __init__(self, num_ellipses=30, num_polygon_points=10):
        super().__init__()
        self.num_ellipses = num_ellipses
        self.num_polygon_points = num_polygon_points

        self.fc = layers.Dense(256, activation='relu')

        # **Cross-Attention on Time-Series Features**
        self.attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.ts_proj_layer = layers.Dense(64)  # Projects time-series features

        # **Output layers for ellipses and polygons**
        self.ellipse_layer = layers.Dense(self.num_ellipses * 5)  # (x, y, major, minor, angle)
        self.polygon_layer = layers.Dense(self.num_polygon_points * 2)  # (x, y)

    def call(self, z, ts_features=None):
        """
        Decodes z into ellipses and polygons, optionally conditioning on ts_features.

        Args:
            z: Latent vector [B, latent_dim]
            ts_features: Time-series encoded features for conditioning (optional)

        Returns:
            ellipses: Predicted ellipse parameters [B, num_ellipses * 5]
            polygons: Predicted polygon parameters [B, num_polygon_points * 2]
        """
        x = self.fc(z)  # Pass latent vector through FC layer

        # **Apply Cross-Attention if Time-Series Features are Available**
        if ts_features is not None:
            ts_proj = self.ts_proj_layer(ts_features)  # Project time-series features
            ts_proj = tf.expand_dims(ts_proj, axis=1)  # Expand dims for attention
            attn_output = self.attention_layer(query=tf.expand_dims(x, axis=1), value=ts_proj, key=ts_proj)
            x = x + tf.squeeze(attn_output, axis=1)  # Add residual connection

        # Predict ellipses and polygons
        ellipses = self.ellipse_layer(x)
        polygons = self.polygon_layer(x)

        return ellipses, polygons

# ----- VAE with Self-Attention in the Bottleneck -----
class VAE(keras.Model):
    """
    VAE with a Mixture of Gaussians (MoG) latent space:
      - Uses MultiModalEncoder to encode time-series and mask features.
      - Represents the latent space using a MoG prior.
      - Reparameterizes using a weighted sum of MoG components.
      - Decodes the latent vector into raw time-series and damage masks.

    Returns: recon_ts, recon_mask, (mu_final, logvar_final, pi_final)
    """
    def __init__(self, latent_dim, feature_dim, num_mixtures=5):
        super(VAE, self).__init__()
        self.num_mixtures = num_mixtures
        self.encoder = MultiModalEncoder(latent_dim, feature_dim, num_mixtures)
        self.decoder_ts = TimeSeriesDecoder()
        self.decoder_mask = MaskDecoder()
        self.token_count = 8
        self.token_dim = latent_dim // self.token_count
        self.self_attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=self.token_dim)

    def reparameterize(self, mu, logvar, pi):
        """
        Performs the reparameterization trick using a Mixture of Gaussians.
        Instead of sampling, we take a weighted sum of the mixture means.

        Args:
            mu: [B, num_mixtures, latent_dim] -> Mean of latent Gaussian components
            logvar: [B, num_mixtures, latent_dim] -> Log-variance of latent Gaussian components
            pi: [B, num_mixtures] -> Mixture weights (should sum to 1 per batch sample)

        Returns:
            z: [B, latent_dim] -> Sampled latent vector
        """
        z = tf.reduce_sum(pi[..., tf.newaxis] * mu, axis=1)
        return z

    def call(self, raw_in, fft_in, rms_in, psd_in, ellipses, polygons, test_id, training=False):
        """
        Forward pass of the VAE.

        Args:
            raw_in, fft_in, rms_in, psd_in: Time-series features.
            ellipses, polygons: Damage representation encodings.
            test_id: Test identifier.
            training: Whether the model is in training mode.

        Returns:
            recon_ts: Reconstructed time-series data.
            recon_mask: Reconstructed damage mask.
            (mu_final, logvar_final, pi_final): MoG latent space parameters.
        """
        # 1) Encode inputs into MoG latent space
        mu_final, logvar_final, pi_final, agg_features = self.encoder(
        raw_in, rms_in, psd_in, ellipses, polygons, test_id
        )

        # 2) Sample latent vector z using the MoG reparameterization trick
        z = self.reparameterize(mu_final, logvar_final, pi_final)  

        # 3) Apply self-attention to refine z (optional)
        z_tokens = tf.reshape(z, (-1, self.token_count, self.token_dim))
        attn_output = self.self_attention_layer(query=z_tokens, key=z_tokens, value=z_tokens)
        z_refined = tf.reshape(attn_output, (-1, tf.shape(z)[-1]))

        # 4) Decode time-series and damage mask using the refined latent vector
        recon_ts = self.decoder_ts(z_refined, mask_features=agg_features)
        recon_mask = self.decoder_mask(z_refined, ts_features=agg_features)

        return recon_ts, recon_mask, (mu_final, logvar_final, pi_final)

    def generate(self, z):
        """
        Generate synthetic time-series and masks from a latent vector.

        Args:
            z: Latent vector of shape [B, latent_dim]

        Returns:
            recon_ts: Generated time-series data
            recon_mask: Generated damage mask
        """
        z_tokens = tf.reshape(z, (-1, self.token_count, self.token_dim))
        attn_output = self.self_attention_layer(query=z_tokens, key=z_tokens, value=z_tokens)
        z_refined = tf.reshape(attn_output, (-1, tf.shape(z)[-1]))

        recon_ts = self.decoder_ts(z_refined, mask_features=None)
        recon_mask = self.decoder_mask(z_refined, ts_features=None)

        return recon_ts, recon_mask

# ----- Prior estimation -----
def mixture_of_gaussians_log_prob(z, mus, logvars, pis, eps=1e-8):
    """
    Evaluate log p(z) for a mixture of Gaussians:
      p(z) = sum_k pis[k] * N(z | mus[k], exp(logvars[k]))

    Args:
      z: shape (batch, latent_dim)
      mus: shape (K, latent_dim)
      logvars: shape (K, latent_dim)
      pis: shape (K,) => mixture weights (should sum to 1)
      eps: small constant to avoid log(0)

    Returns:
      log_p(z), shape (batch,)
    """
    # 1) Safe clamp mixture weights => never zero
    pis_safe = pis + eps
    pis_safe = pis_safe / tf.reduce_sum(pis_safe)  # re-normalize

    # 2) Clip logvars to safe range
    logvars = tf.clip_by_value(logvars, -10.0, 10.0)

    # Expand dims to broadcast
    K = tf.shape(mus)[0]  # K
    z_expanded = tf.expand_dims(z, axis=1)      # (batch, 1, latent_dim)
    mus = tf.expand_dims(mus, axis=0)           # (1, K, latent_dim)
    logvars = tf.expand_dims(logvars, axis=0)   # (1, K, latent_dim)

    # log p(z | comp k)
    # = -0.5 * [(z - mu)^2 / var + log var] - const
    const_term = 0.5 * tf.cast(z.shape[-1], tf.float32) * np.log(2.0 * np.pi)
    inv_var = tf.exp(-logvars)
    log_p_k = -0.5 * tf.reduce_sum(inv_var * tf.square(z_expanded - mus), axis=-1) \
              - 0.5 * tf.reduce_sum(logvars, axis=-1) \
              - const_term  # => shape (batch, K)

    # weighting by mixture pi_k => log pi_k
    weighted_log_p_k = log_p_k + tf.math.log(pis_safe)  # => shape (batch, K)

    # stable log-sum-exp across k
    log_p = tf.reduce_logsumexp(weighted_log_p_k, axis=-1)  # => (batch,)
    tf.debugging.assert_all_finite(log_p, "NaN in mixture_of_gaussians_log_prob")
    return log_p

def mog_log_prob_per_example(z, mu_q, logvar_q, pi_q, eps=1e-8):
    """
    Evaluate log q_i(z_i), where q_i is a mixture of Gaussians specific
    to each batch example i.

    Args:
      z: shape (batch, latent_dim)
      mu_q: shape (batch, K, latent_dim)
      logvar_q: shape (batch, K, latent_dim)
      pi_q: shape (batch, K)
      eps: small constant to avoid log(0)

    Returns:
      shape (batch,) => log of the mixture pdf at each z_i
    """
    # 1) clamp mixture weights => no zero
    pi_q_safe = pi_q + eps
    pi_q_safe = pi_q_safe / tf.reduce_sum(pi_q_safe, axis=1, keepdims=True)  # per example

    # 2) clamp logvar => [-10, 10]
    logvar_q = tf.clip_by_value(logvar_q, -10.0, 10.0)

    batch_size = tf.shape(z)[0]
    K = tf.shape(mu_q)[1]

    # expand z => [batch, 1, latent_dim]
    z_expanded = tf.expand_dims(z, axis=1)   # => [batch, 1, latent_dim]
    inv_var = tf.exp(-logvar_q)             # => [batch, K, latent_dim]
    diff_sq = tf.square(z_expanded - mu_q)  # => [batch, K, latent_dim]

    # log N(z|mu, var)
    const_term = 0.5 * tf.cast(z.shape[-1], tf.float32) * np.log(2.0 * np.pi)
    log_probs_per_comp = -0.5 * tf.reduce_sum(inv_var * diff_sq, axis=-1) \
                         - 0.5 * tf.reduce_sum(logvar_q, axis=-1) \
                         - const_term
    # => shape [batch, K]

    # add log pi_q
    weighted = log_probs_per_comp + tf.math.log(pi_q_safe)  # => [batch, K]
    log_mix = tf.reduce_logsumexp(weighted, axis=-1)        # => [batch]

    tf.debugging.assert_all_finite(log_mix, "NaN in mog_log_prob_per_example")
    return log_mix

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

def kl_q_p_mixture(
    mu_q, logvar_q, pi_q,       # posterior mixture: shape [batch, K, latent_dim], [batch, K], etc.
    mu_p, logvar_p, pi_p,       # prior mixture: shape [K, latent_dim], [K], etc.
    num_samples=1,
    eps=1e-8
):
    """
    KL( q(z) || p(z) ) where both q, p are mixture-of-Gaussians.

    We do:
    1) sample z ~ q
    2) compute log_q(z) - log_p(z)
    3) average over multiple samples if needed
    """
    # Quick checks
    tf.debugging.assert_all_finite(mu_q,     "mu_q is NaN/Inf in kl_q_p_mixture")
    tf.debugging.assert_all_finite(logvar_q, "logvar_q is NaN/Inf in kl_q_p_mixture")
    tf.debugging.assert_all_finite(pi_q,     "pi_q is NaN/Inf in kl_q_p_mixture")
    tf.debugging.assert_all_finite(mu_p,     "mu_p is NaN/Inf in kl_q_p_mixture")
    tf.debugging.assert_all_finite(logvar_p, "logvar_p is NaN/Inf in kl_q_p_mixture")
    tf.debugging.assert_all_finite(pi_p,     "pi_p is NaN/Inf in kl_q_p_mixture")

    # clamp posterior mixture weights
    pi_q_safe = pi_q + eps  # shape [batch, K]
    pi_q_safe = pi_q_safe / tf.reduce_sum(pi_q_safe, axis=1, keepdims=True)

    # clamp prior mixture weights
    pi_p_safe = pi_p + eps  # shape [K,]
    pi_p_safe = pi_p_safe / tf.reduce_sum(pi_p_safe)

    # clamp logvar
    logvar_q = tf.clip_by_value(logvar_q, -10.0, 10.0)
    logvar_p = tf.clip_by_value(logvar_p, -10.0, 10.0)

    batch_size = tf.shape(mu_q)[0]
    K = tf.shape(mu_q)[1]
    latent_dim = tf.shape(mu_q)[2]

    kl_accum = 0.0
    for _ in range(num_samples):
        # pick a mixture component j ~ pi_q
        chosen_js = tf.random.categorical(tf.math.log(pi_q_safe), num_samples=1, dtype=tf.int32)
        chosen_js = tf.squeeze(chosen_js, axis=1)  # => shape (batch,)

        batch_idxs = tf.range(batch_size, dtype=tf.int32)
        gather_idxs = tf.stack([batch_idxs, chosen_js], axis=1)  # => shape (batch,2)

        # gather mu_q[i,j], logvar_q[i,j]
        chosen_mu = tf.gather_nd(mu_q, gather_idxs)      # => [batch, latent_dim]
        chosen_lv = tf.gather_nd(logvar_q, gather_idxs)  # => [batch, latent_dim]

        # sample z
        eps_ = tf.random.normal(tf.shape(chosen_mu))
        chosen_lv = tf.clip_by_value(chosen_lv, -10.0, 10.0)  # again clamp
        z_sample = chosen_mu + tf.exp(0.5 * chosen_lv) * eps_  # => [batch, latent_dim]

        # Evaluate log q(z), log p(z)
        log_qz = mog_log_prob_per_example(z_sample, mu_q, logvar_q, pi_q_safe, eps=eps)
        log_pz = mixture_of_gaussians_log_prob(z_sample, mu_p, logvar_p, pi_p_safe, eps=eps)

        diff = (log_qz - log_pz)
        tf.debugging.assert_all_finite(diff, "NaN in (log_qz - log_pz)")

        kl_accum += diff

    kl_values = kl_accum / tf.cast(num_samples, tf.float32)
    kl_mean = tf.reduce_mean(kl_values)
    tf.debugging.assert_all_finite(kl_mean, "NaN in kl_mean (kl_q_p_mixture)")
    return kl_mean

def gaussian_log_prob(z, mu, logvar):
    """
    log N(z | mu, exp(logvar)) i.i.f. over the last dimension
    """
    const_term = 0.5 * z.shape[-1] * np.log(2 * np.pi)
    inv_var = tf.exp(-logvar)
    tmp = tf.reduce_sum(inv_var * tf.square(z - mu), axis=-1)
    log_det = 0.5 * tf.reduce_sum(logvar, axis=-1)
    return tf.cast(const_term, tf.float32) + 0.5 * tmp + log_det

def sample_from_mog(mu, logvar, pi):
    """
    Samples from a Mixture of Gaussians:
      mu, logvar: shape [K, latent_dim]
      pi: shape [K,] for mixture weights
    Returns:
      z: shape [1, latent_dim] if we do 1 sample
    """
    K = tf.shape(pi)[0]  # Number of components
    # Suppose we sample 1 latent vector at a time
    k = tf.random.categorical(tf.math.log(tf.expand_dims(pi, axis=0)), 1)  # shape [1,1]
    k = tf.squeeze(k, axis=-1)  # shape [1]
    # Extract the chosen component's (mu, logvar)
    chosen_mu = tf.gather(mu, k, axis=0)
    chosen_logvar = tf.gather(logvar, k, axis=0)
    eps = tf.random.normal(tf.shape(chosen_mu))
    z = chosen_mu + tf.exp(0.5 * chosen_logvar) * eps
    # shape [latent_dim], expand to [1, latent_dim] if needed
    return tf.expand_dims(z, axis=0)

def train_autoencoder(model, train_dataset, val_dataset, optimizer, num_epochs=50):
    """Train only the encoder (ignoring KL and MoG)."""
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    all_trainables = model.encoder.trainable_variables + model.decoder_ts.trainable_variables

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        steps = 0

        for raw_in, fft_in, rms_in, psd_in, mask_in, test_id_in in train_dataset:
            with tf.GradientTape() as tape:
                recon_ts, _, _ = model(raw_in, fft_in, rms_in, psd_in, mask_in, test_id_in, training=True)
                loss = mse_loss_fn(raw_in, recon_ts)

            grads = tape.gradient(loss, all_trainables)
            optimizer.apply_gradients(zip(grads, all_trainables))

            epoch_loss += loss.numpy()
            steps += 1

        print(f"Epoch {epoch+1}/{num_epochs} - Recon Loss: {epoch_loss / steps:.4f}")

def extract_latent_representations(model, dataset, preprocessed_masks):
    """Pass data through the encoder to get latent embeddings."""
    latent_vectors = []
    test_ids = []

    for raw_in, fft_in, rms_in, psd_in, mask_in, test_id_in in dataset:
        # 🔹 Convert test_id_in to proper format
        if isinstance(test_id_in, tf.Tensor):
            test_id_in = test_id_in.numpy()
        if isinstance(test_id_in, np.ndarray):
            test_id_in = test_id_in.flatten().tolist()

        # 🔹 Retrieve precomputed mask encodings
        batch_ellipses = []
        batch_polygons = []
        for tid in test_id_in:
            mask_data = preprocessed_masks.get(int(tid), {"ellipses": [], "polygons": []})
            batch_ellipses.append(mask_data["ellipses"])
            batch_polygons.append(mask_data["polygons"])

        batch_ellipses = np.array(batch_ellipses, dtype=np.float32)
        batch_polygons = np.array(batch_polygons, dtype=np.float32)
        test_id_in = np.array(test_id_in, dtype=np.int32).flatten()

        # ✅ Fix the Unpacking Issue - Use 4 Values
        mu_q, logvar_q, pi_q, _ = model.encoder(
            raw_in, fft_in, rms_in, psd_in, batch_ellipses, batch_polygons, test_id_in, training=False
        )

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

# ~~~~~~~~~~~~~~~~ DEFINE MOG PARAMETERS ~~~~~~~~~~~~~~~~~
K = 3  # Number of mixture components
latent_dim_raw = 128 # Must match the latend_dim in main
latent_dim_fft = 128
latent_dim_rms = 128
latent_dim_psd = 128
latent_dim_mask = 128

#-------------------- Training -------------------------
def train_vae(
    model, 
    train_dataset, 
    val_dataset, 
    mask_dict, 
    preprocessed_masks, 
    optimizer, 
    num_epochs=100, 
    use_mog=True,
    patience=10
):
    """
    Train loop for VAE using adaptive ellipses/polygons for mask representation.
    Incorporates KL annealing, LR scheduling, and early stopping.
    """

    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    all_trainables = model.trainable_variables

    train_total_losses = []
    train_recon_losses = []
    train_kl_losses = []

    val_total_losses = []
    val_recon_losses = []
    val_kl_losses = []

    # For early stopping
    best_val_loss = float('inf')
    no_improvement_count = 0

    print("🔄 Starting Training...")

    for epoch in range(num_epochs):
        epoch_train_total = 0.0
        epoch_train_recon = 0.0
        epoch_train_kl = 0.0
        train_steps = 0

        for step, (raw_in, fft_in, rms_in, psd_in, mask_in, test_id_in) in enumerate(train_dataset):
            # Convert test_id_in to list of integers
            if isinstance(test_id_in, tf.Tensor):
                test_id_in = test_id_in.numpy()
            if isinstance(test_id_in, np.ndarray):
                test_id_in = test_id_in.flatten().tolist()

            # Retrieve the correct mask for each test_id in the batch
            batch_ellipses = []
            batch_polygons = []
            for tid in test_id_in:
                mask_data = preprocessed_masks.get(int(tid), {"ellipses": [], "polygons": []})
                batch_ellipses.append(mask_data["ellipses"])
                batch_polygons.append(mask_data["polygons"])

            batch_ellipses = np.array(batch_ellipses, dtype=np.float32)
            batch_polygons = np.array(batch_polygons, dtype=np.float32)
            test_id_in = np.array([int(float(tid)) for tid in test_id_in], dtype=np.int32)

            with tf.GradientTape() as tape:
                recon_ts, recon_mask, (mu_final, logvar_final, pi_final) = model(
                    raw_in, fft_in, rms_in, psd_in,
                    batch_ellipses, batch_polygons, test_id_in,
                    training=True
                )
                # 🔹 Fix: Clip mu to prevent large values
                mu_final = tf.clip_by_value(mu_final, -10.0, 10.0)

                pred_ellipses, pred_polygons = recon_mask

                # Reconstruction loss (time-series)
                LAMBDA_FFT, LAMBDA_MSE = 0.7, 0.3
                loss_ts = mse_loss_fn(raw_in, recon_ts)

                # Reconstruction loss (mask)
                LAMBDA_ELLIPSE, LAMBDA_POLYGON = 0.7, 0.3
                mask_recon_loss = 0.0
                for true_e, true_p, pe, pp in zip(batch_ellipses, batch_polygons, pred_ellipses, pred_polygons):
                    if len(true_e) == 0 and len(true_p) == 0:
                        # If the model predicts something when there's no true damage -> penalty
                        if len(pe) != 0 or len(pp) != 0:
                            mask_recon_loss += 1.0
                    else:
                        mask_recon_loss += (
                            LAMBDA_ELLIPSE * ellipse_loss(true_e, pe)
                            + LAMBDA_POLYGON * polygon_loss(true_p, pp)
                        )

                recon_loss = 0.5 * (loss_ts + mask_recon_loss)

                # KL loss
                if use_mog:
                    kl_loss = kl_q_p_mixture(
                        mu_q=mu_final, logvar_q=logvar_final, pi_q=pi_final,
                        mu_p=tf.zeros_like(mu_final),
                        logvar_p=tf.zeros_like(logvar_final),
                        pi_p=tf.ones_like(pi_final) / tf.cast(tf.shape(pi_final)[-1], tf.float32),
                        num_samples=1
                    )
                else:
                    # Standard Gaussian prior
                    # 🔹 Fix: Clip logvar to prevent variance explosion
                    logvar_final = tf.clip_by_value(logvar_final, -5.0, 5.0)

                    # Compute KL loss
                    kl_loss = -0.5 * tf.reduce_mean(
                        1.0 + logvar_final - tf.square(mu_final) - tf.exp(logvar_final)
                    )


                # KL Annealing: linearly ramp beta from BETA_START -> BETA_MAX over KL_ANNEALING_EPOCHS
                KL_ANNEALING_EPOCHS = 400
                BETA_START = 1e-10
                BETA_MAX   = 1e-4
                BETA = min(
                    BETA_START + (BETA_MAX - BETA_START) * (epoch / KL_ANNEALING_EPOCHS),
                    BETA_MAX
                )

                total_loss = recon_loss + BETA * kl_loss

            grads = tape.gradient(total_loss, all_trainables)
            optimizer.apply_gradients(zip(grads, all_trainables))

            epoch_train_total += total_loss.numpy()
            epoch_train_recon += recon_loss.numpy()
            epoch_train_kl += kl_loss.numpy()
            train_steps += 1

        # Compute average training losses for the epoch
        if train_steps > 0:
            train_total_losses.append(epoch_train_total / train_steps)
            train_recon_losses.append(epoch_train_recon / train_steps)
            train_kl_losses.append(epoch_train_kl / train_steps)

        # 🔹 Debug: Print latent space mean and variance to check instability
        print(f"Epoch {epoch+1}/{num_epochs} | Mean(μ): {tf.reduce_mean(mu_final).numpy():.4f}, "
        f"Var(σ²): {tf.reduce_mean(tf.exp(logvar_final)).numpy():.4f}")

        print(f"  🟢 Train => Total: {train_total_losses[-1]:.4f}, "
              f"Recon: {train_recon_losses[-1]:.4f}, "
              f"KL: {train_kl_losses[-1]:.4f}, "
              f"Beta: {BETA:.7f}")

        # ------------------ Validation Loop ------------------
        epoch_val_total, epoch_val_recon, epoch_val_kl = 0.0, 0.0, 0.0
        val_steps = 0

        for step, (raw_in, fft_in, rms_in, psd_in, mask_in, test_id_in) in enumerate(val_dataset):
            if isinstance(test_id_in, tf.Tensor):
                test_id_in = test_id_in.numpy()
            if isinstance(test_id_in, np.ndarray):
                test_id_in = test_id_in.flatten().tolist()

            batch_ellipses = []
            batch_polygons = []
            for tid in test_id_in:
                mask_data = preprocessed_masks.get(int(tid), {"ellipses": [], "polygons": []})
                batch_ellipses.append(mask_data["ellipses"])
                batch_polygons.append(mask_data["polygons"])

            batch_ellipses = np.array(batch_ellipses, dtype=np.float32)
            batch_polygons = np.array(batch_polygons, dtype=np.float32)
            test_id_in = np.array([int(float(tid)) for tid in test_id_in], dtype=np.int32)

            # Forward pass
            recon_ts, recon_mask, (mu_final, logvar_final, pi_final) = model(
                raw_in, fft_in, rms_in, psd_in, batch_ellipses, batch_polygons, test_id_in, training=False
            )
            # 🔹 Fix: Clip `mu_final` to prevent extreme mean values
            mu_final = tf.clip_by_value(mu_final, -10.0, 10.0)

            # 🔹 Fix: Clip `logvar_final` to prevent variance explosion
            logvar_final = tf.clip_by_value(logvar_final, -5.0, 5.0)

            pred_ellipses, pred_polygons = recon_mask

            loss_ts = LAMBDA_MSE * mse_loss_fn(raw_in, recon_ts) + LAMBDA_FFT * fft_loss(raw_in, recon_ts)

            mask_recon_loss = 0.0
            for true_e, true_p, pe, pp in zip(batch_ellipses, batch_polygons, pred_ellipses, pred_polygons):
                if len(true_e) == 0 and len(true_p) == 0:
                    if len(pe) != 0 or len(pp) != 0:
                        mask_recon_loss += 1.0
                else:
                    mask_recon_loss += (
                        LAMBDA_ELLIPSE * ellipse_loss(true_e, pe)
                        + LAMBDA_POLYGON * polygon_loss(true_p, pp)
                    )

            recon_loss = 0.5 * (loss_ts + mask_recon_loss)

            if use_mog:
                kl_loss = kl_q_p_mixture(
                    mu_q=mu_final, logvar_q=logvar_final, pi_q=pi_final,
                    mu_p=tf.zeros_like(mu_final),
                    logvar_p=tf.zeros_like(logvar_final),
                    pi_p=tf.ones_like(pi_final) / tf.cast(tf.shape(pi_final)[-1], tf.float32),
                    num_samples=1
                )
            else:
                kl_loss = -0.5 * tf.reduce_mean(
                    1.0 + logvar_final - tf.square(mu_final) - tf.exp(logvar_final)
                )

            # Use the same Beta so validation is consistent with training
            total_loss = recon_loss + BETA * kl_loss

            epoch_val_total += total_loss.numpy()
            epoch_val_recon += recon_loss.numpy()
            epoch_val_kl += kl_loss.numpy()
            val_steps += 1

        if val_steps > 0:
            val_total_losses.append(epoch_val_total / val_steps)
            val_recon_losses.append(epoch_val_recon / val_steps)
            val_kl_losses.append(epoch_val_kl / val_steps)

        print(f"  🔵 Val   => Total: {val_total_losses[-1]:.4f}, "
              f"Recon: {val_recon_losses[-1]:.4f}, "
              f"KL: {val_kl_losses[-1]:.4f}")

        # -------------- Early Stopping --------------
        current_val_loss = val_total_losses[-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            no_improvement_count = 0
            # (Optional) Save best model weights
            # model.save_weights("results/best_model_weights.h5")
        else:
            no_improvement_count += 1
            print(f"🚨 No improvement for {no_improvement_count}/{patience} epochs.")

        if no_improvement_count >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.")
            break

    return (
        train_total_losses,
        train_recon_losses,
        train_kl_losses,
        val_total_losses,
        val_recon_losses,
        val_kl_losses
    )

#--------------------- Plots ---------------------------
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

def encoder(ts_sample, mask_sample=None):
    """
    Encodes a time-series sample into its latent representation.
    
    Args:
        ts_sample: Numpy array of shape [batch, 12000, 12]
        mask_sample: (Optional) Numpy array of shape [batch, 256, 768]. 
                     If not provided, a dummy mask of zeros is used.
    
    Returns:
        Latent vector (mu) from the encoder.
    """
    global vae_model
    if vae_model is None:
        raise ValueError("VAE model has not been trained or loaded.")
    if mask_sample is None:
        mask_sample = np.zeros((ts_sample.shape[0], 256, 768), dtype=np.float32)
    mu, logvar, _, _ = vae_model.encoder(ts_sample, mask_sample)
    return mu

def decoder(z):
    """
    Wrapper function to decode a latent vector z into a pair of outputs:
    the reconstructed time-series and its corresponding binary mask.
    
    Args:
        z: Latent vector of shape [batch, latent_dim]
    
    Returns:
        Tuple (recon_ts, recon_mask)
    """
    global vae_model
    if vae_model is None:
        raise ValueError("VAE model has not been trained or loaded.")
    recon_ts, recon_mask = vae_model.generate(z)
    return recon_ts, recon_mask

def load_trained_model(weights_path):
    """
    Load a trained VAE model from the given weights file and assign it to the global variable.
    This function builds the model (by calling it with dummy inputs) before loading the weights.
    
    Args:
        weights_path: Path to the saved weights file.
    """
    global vae_model
    latent_dim = 128  # Ensure this matches the latent_dim used in training.
    model = VAE(latent_dim, feature_dim=128)
    
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
    accel_dict, mask_dict = data_loader.load_data()
    gen = segment_and_transform(accel_dict, mask_dict)
    try:
        raw_segments, fft_segments, rms_segments, psd_segments, mask_segments, test_ids = next(gen)
    except StopIteration:
        raise ValueError("segment_and_transform() did not yield any data.")

    print("Finished segmenting data.")
    print(f"Data shapes after segmentation:")
    print(f"  Raw: {raw_segments.shape}")     
    print(f"  FFT: {fft_segments.shape}")     
    print(f"  RMS: {rms_segments.shape}")      
    print(f"  PSD: {psd_segments.shape}")      
    print(f"  Mask: {mask_segments.shape}")    
    print(f"  IDs: {test_ids.shape}")

    # ---- Debugging: Print Structure of mask_dict ----
    print("\nDEBUGGING MASK DICT STRUCTURE")
    print(f"Number of unique test IDs: {len(mask_dict)}")
    sample_test_id = next(iter(mask_dict.keys()))  # Get any test ID
    print(f"Sample test ID: {sample_test_id}")

    sample_mask_data = mask_dict[sample_test_id]
    print(f"Type of mask_dict[{sample_test_id}]: {type(sample_mask_data)}")

    # Ensure all mask data are NumPy arrays
    for key, mask in mask_dict.items():
        if not isinstance(mask, np.ndarray):
            print(f"[ERROR] Mask for Test ID {key} is not a NumPy array! Found: {type(mask)}")
    
    print("--------------------------------------------------")

    # ------------------------ 2) Convert to Float32 ------------------------
    raw_segments  = raw_segments.astype(np.float32)
    fft_segments  = fft_segments.astype(np.float32)
    rms_segments  = rms_segments.astype(np.float32)
    psd_segments  = psd_segments.astype(np.float32)
    mask_segments = mask_segments.astype(np.float32)
    test_ids      = test_ids.astype(np.float32)

    # ------------------------ 3) Shuffle and Split into Train/Val ------------------------
    N = raw_segments.shape[0]
    indices = np.random.permutation(N)
    train_size = int(0.8 * N)
    train_idx = indices[:train_size]
    val_idx   = indices[train_size:]

    train_raw   = raw_segments[train_idx]
    train_fft   = fft_segments[train_idx]
    train_rms   = rms_segments[train_idx]
    train_psd   = psd_segments[train_idx]
    train_mask  = mask_segments[train_idx]
    train_ids   = test_ids[train_idx]

    val_raw   = raw_segments[val_idx]
    val_fft   = fft_segments[val_idx]
    val_rms   = rms_segments[val_idx]
    val_psd   = psd_segments[val_idx]
    val_mask  = mask_segments[val_idx]
    val_ids   = test_ids[val_idx]

    # ------------------------ 4) Precompute Ellipse/Polygon Representations ------------------------
    print("\nPrecomputing ellipse/polygon encodings for unique test IDs...")
    preprocessed_masks = preprocess_all_masks(mask_dict)  # {test_id: {"ellipses": [...], "polygons": [...]} }
    print("Preprocessing complete.")

    print("\n🔎 Debug: Checking Preprocessed Masks")
    for key in list(preprocessed_masks.keys())[:5]:  # Print first 5 entries
        print(f"Test ID {key}: {preprocessed_masks[key]}")

    # ------------------------ 5) Build tf.data Datasets ------------------------
    BATCH_SIZE = 128
    train_dataset = create_tf_dataset(
        train_raw, train_fft, train_rms, train_psd, train_mask, train_ids, batch_size=BATCH_SIZE
    )
    val_dataset = create_tf_dataset(
        val_raw, val_fft, val_rms, val_psd, val_mask, val_ids, batch_size=BATCH_SIZE
    )

    print(f"Train batches: {len(train_dataset)}")
    print(f"Val batches:   {len(val_dataset)}")

    # ------------------------ 6) Build & Train Model ------------------------
    latent_dim  = 128
    feature_dim = 128
    model = VAE(latent_dim, feature_dim)  # Your custom VAE model

    # (a) Learning Rate Scheduler
    decay_steps = 20000
    decay_rate  = 0.95
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-6,  # start a bit higher
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False  # smooth decay
    )

    # (b) Build Optimizer with LR Schedule
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4
    )

    print_detailed_memory()

    # (c) Train using your custom train_vae function
    (
        train_total,
        train_recon,
        train_kl,
        val_total,
        val_recon,
        val_kl
    ) = train_vae(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        mask_dict=mask_dict,
        preprocessed_masks=preprocessed_masks,
        optimizer=optimizer,
        num_epochs=800,   # or whichever number of epochs you want
        use_mog=False,     # mixture-of-gaussians prior
        patience=800       # early stopping patience
    )

    # ------------------------ 7) Save Weights ------------------------
    model.save_weights("results/vae_mmodal_weights.h5")
    print("Saved model weights to results/vae_mmodal_weights.h5")

    # ------------------------ 8) Visualize Latent Space with UMAP ------------------------
    latent_vectors, test_ids_arr = extract_latent_representations(model, train_dataset, preprocessed_masks)
    latent_3d = reduce_latent_dim_umap(latent_vectors)
    plot_latent_space_3d(latent_3d, test_ids_arr)

    # Extract latent representations from the trained model
    latent_vectors, _ = extract_latent_representations(model, train_dataset, preprocessed_masks)

    # Plot histograms for 3 latent dimensions
    plot_latent_histograms(latent_vectors)

    # ------------------------ 9) Plot Training Curves ------------------------
    plot_training_curves(train_total, train_recon, train_kl, val_total, val_recon, val_kl)

    # # 10) Generate and plot 5 random samples.
    # plot_generated_samples(model, latent_dim, num_samples=5)

    # # 11) Generate synthetic data
    # results_dir = "results/vae_results"
    # os.makedirs(results_dir, exist_ok=True)

    # num_synthetic = 50
    # for i in range(num_synthetic):
    #     z_rand = tf.random.normal(shape=(1, latent_dim))
    #     gen_ts, gen_mask = model.generate(z_rand)

    #     np.save(os.path.join(results_dir, f"gen_ts_{i}.npy"), gen_ts.numpy())
    #     np.save(os.path.join(results_dir, f"gen_mask_{i}.npy"), gen_mask.numpy())
    #     logging.info(f"Saved synthetic sample {i}")


if __name__ == "__main__":
    main()
