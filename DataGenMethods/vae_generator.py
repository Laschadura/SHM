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
import sys
import psutil
import GPUtil
import numpy as np
import keras 
from keras import layers, Model
import logging
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio



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
                    test_ids_out.append(test_id)
                    
        # Convert chunk to arrays and yield
        if raw_segments:  # Only yield if we have data
            yield (np.array(raw_segments, dtype=np.float32),
                  np.array(fft_segments, dtype=np.float32),
                  np.array(rms_segments, dtype=np.float32),
                  np.array(psd_segments, dtype=np.float32),
                  np.array(mask_segments, dtype=np.float32),
                  np.array(test_ids_out))
            
def create_tf_dataset(raw_segments, fft_segments, rms_segments, psd_segments, mask_segments, test_ids, batch_size=8):
    print(f"Creating dataset with shapes:")
    print(f"  Raw segments: {raw_segments.shape}")
    print(f"  FFT segments: {fft_segments.shape}")
    print(f"  RMS segments: {rms_segments.shape}")
    print(f"  PSD segments: {psd_segments.shape}")
    print(f"  Mask segments: {mask_segments.shape}")
    print(f"  Test IDs: {test_ids.shape}\n")
    
    # Now we include test_ids as the 6th item
    dataset = tf.data.Dataset.from_tensor_slices((
        raw_segments,
        fft_segments,
        rms_segments,
        psd_segments,
        mask_segments,
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

def reparameterize(mu, logvar):
    std = tf.exp(0.5 * logvar)
    eps = tf.random.normal(shape=tf.shape(std))
    return mu + eps * std

# ----- Encoder Branches -----
class RawEncoder(keras.Model):
    """Convolutional encoder for raw time-series data.
       Input shape: (1000, 12)
       Outputs a feature vector of dimension feature_dim.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.conv1 = layers.Conv1D(16, kernel_size=4, strides=2, padding='same', activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.conv2 = layers.Conv1D(32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()
        self.dense_reduce = layers.Dense(512, activation='relu')
        self.feature_layer = layers.Dense(feature_dim)  # final feature vector

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        x = self.flatten(x)
        x = self.dense_reduce(x)
        feature = self.feature_layer(x)
        return feature  # shape: [batch, feature_dim]
    
class FFTEncoder(keras.Model):
    """Encoder for FFT features.
       Input shape: (501, 12)
       Outputs a feature vector of dimension feature_dim.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.conv1 = layers.Conv1D(8, 3, strides=1, padding='same', activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()
        self.dense_reduce = layers.Dense(128, activation='relu')
        self.feature_layer = layers.Dense(feature_dim)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense_reduce(x)
        feature = self.feature_layer(x)
        return feature  # shape: [batch, feature_dim]
    
class PSDEncoder(keras.Model):
    """Encoder for PSD features.
       Input shape: (freq_bins, 12) e.g. (129, 12)
       Outputs a feature vector of dimension feature_dim.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.conv1 = layers.Conv1D(8, 3, strides=1, padding='same', activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()
        self.dense_reduce = layers.Dense(128, activation='relu')
        self.feature_layer = layers.Dense(feature_dim)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense_reduce(x)
        feature = self.feature_layer(x)
        return feature  # shape: [batch, feature_dim]

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

class MaskEncoder(keras.Model):
    """Encoder for damage mask images.
       Input shape: (256, 768)
       Uses convolutional layers to extract spatial features and then a Dense layer to reduce to feature_dim.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.expand = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))
        self.conv1 = layers.Conv2D(16, kernel_size=4, strides=2, padding='same', activation='relu')
        self.drop1 = layers.Dropout(0.3)
        self.conv2 = layers.Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.drop2 = layers.Dropout(0.3)
        self.conv3 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')
        self.drop3 = layers.Dropout(0.3)
        self.conv4 = layers.Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu')
        self.drop4 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()
        self.dense_reduce = layers.Dense(256, activation='relu')
        self.feature_layer = layers.Dense(feature_dim)

    def call(self, x, training=False):
        x = self.expand(x)
        x = self.conv1(x)
        x = self.drop1(x, training=training)
        x = self.conv2(x)
        x = self.drop2(x, training=training)
        x = self.conv3(x)
        x = self.drop3(x, training=training)
        x = self.conv4(x)
        x = self.drop4(x, training=training)
        x = self.flatten(x)
        x = self.dense_reduce(x)
        feature = self.feature_layer(x)
        return feature  # shape: [batch, feature_dim]

# ----- Combined Encoder with Multi Branches -----
class MultiModalEncoder(keras.Model):
    """
    Aggregates feature vectors from raw, FFT, RMS, PSD, and mask encoders.
    Each sub-encoder outputs a deterministic feature vector.
    These are concatenated (along with test_id) and passed through an MLP to produce
    the final MoG parameters for the shared latent space.
    
    Outputs:
      mu_final: [B, num_mixtures, latent_dim]
      logvar_final: [B, num_mixtures, latent_dim]
      pi_final: [B, num_mixtures]
      combined_features: [B, combined_feature_dim] (for conditioning)
    """
    def __init__(self, latent_dim, feature_dim, num_mixtures=5):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.raw_enc = RawEncoder(feature_dim)
        self.fft_enc = FFTEncoder(feature_dim)
        self.rms_enc = RMSEncoder(feature_dim)
        self.psd_enc = PSDEncoder(feature_dim)
        self.mask_enc = MaskEncoder(feature_dim)
        
        # Aggregator: concatenate features from all 5 modalities plus test_id.
        self.fc_agg = layers.Dense(128, activation='relu')
        self.mu_layer = layers.Dense(latent_dim * num_mixtures)
        self.logvar_layer = layers.Dense(latent_dim * num_mixtures)
        self.pi_layer = layers.Dense(num_mixtures, activation="softmax")

    def call(self, raw_in, fft_in, rms_in, psd_in, mask_in, test_id, training=False):
        feat_raw  = self.raw_enc(raw_in, training=training)    # [B, feature_dim]
        feat_fft  = self.fft_enc(fft_in, training=training)
        feat_rms  = self.rms_enc(rms_in, training=training)
        feat_psd  = self.psd_enc(psd_in, training=training)
        feat_mask = self.mask_enc(mask_in, training=training)
        
        # Process test_id: convert to float and expand dimension => [B, 1]
        test_id = tf.cast(test_id, tf.float32)
        test_id = tf.expand_dims(test_id, axis=-1)
        
        # Concatenate all features: [B, (5*feature_dim + 1)]
        combined_features = tf.concat([feat_raw, feat_fft, feat_rms, feat_psd, feat_mask, test_id], axis=-1)
        
        # Pass through aggregator MLP.
        agg = self.fc_agg(combined_features)  # [B, 128]
        mu_unshaped = self.mu_layer(agg)      # [B, latent_dim*num_mixtures]
        logvar_unshaped = self.logvar_layer(agg)  # [B, latent_dim*num_mixtures]
        pi_final = self.pi_layer(agg)          # [B, num_mixtures]
        
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
 
class MaskDecoder(keras.Model):
    """
    Decoder for damage mask reconstruction.
    For sparse masks, we use an output activation (sigmoid) and can later combine Dice or focal losses.
    Output shape: (256, 768)
    """
    def __init__(self):
        super(MaskDecoder, self).__init__()
        self.fc = layers.Dense(48 * 16 * 64, activation='relu')
        self.reshape_layer = layers.Reshape((48, 16, 64))
        self.attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.ts_proj_layer = layers.Dense(64)
        self.deconv1 = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.deconv2 = layers.Conv2DTranspose(16, kernel_size=4, strides=2, padding='same', activation='relu')
        self.deconv3 = layers.Conv2DTranspose(8, kernel_size=4, strides=2, padding='same', activation='relu')
        self.deconv4 = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='sigmoid')

    def call(self, z, ts_features=None, training=False):
        x = self.fc(z)
        x = self.reshape_layer(x)  # [B, 48, 16, 64]
        b = tf.shape(x)[0]
        seq_len = 48 * 16
        x_seq = tf.reshape(x, (b, seq_len, 64))
        if ts_features is not None:
            ts_proj = self.ts_proj_layer(ts_features)  # [B, 64]
            ts_proj = tf.expand_dims(ts_proj, axis=1)   # [B, 1, 64]
            ts_proj = tf.tile(ts_proj, [1, seq_len, 1])
            attn_output = self.attention_layer(query=x_seq, value=ts_proj, key=ts_proj)
            x_seq = x_seq + attn_output
        x = tf.reshape(x_seq, (b, 48, 16, 64))
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = tf.squeeze(x, axis=-1)
        x = tf.transpose(x, perm=[0, 2, 1])
        return x  # [B, 256, 768]

# ----- VAE with Self-Attention in the Bottleneck -----
class VAE(keras.Model):
    """
    VAE that:
      - Uses sub-encoders to extract deterministic feature vectors.
      - Aggregates these features (plus test_id) into a shared latent distribution modeled as a MoG.
      - Reparameterizes using a weighted average of the mixture components.
      - Decodes the latent to produce raw time-series and mask outputs.
      
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
        For simplicity, we compute the weighted average of the mixture means.
        mu: [B, num_mixtures, latent_dim], pi: [B, num_mixtures]
        Returns: z of shape [B, latent_dim]
        """
        z = tf.reduce_sum(pi[..., tf.newaxis] * mu, axis=1)
        return z

    def call(self, raw_in, fft_in, rms_in, psd_in, mask_in, test_id, training=False):
        # 1) Obtain aggregated latent MoG parameters from the MultiModalEncoder.
        mu_final, logvar_final, pi_final, agg_features = self.encoder(raw_in, fft_in, rms_in, psd_in, mask_in, test_id, training=training)
        # 2) Sample latent vector z using weighted average.
        z = self.reparameterize(mu_final, logvar_final, pi_final)  # [B, latent_dim]
        # 3) Optionally refine z with self-attention.
        z_tokens = tf.reshape(z, (-1, self.token_count, self.token_dim))
        attn_output = self.self_attention_layer(query=z_tokens, key=z_tokens, value=z_tokens)
        z_refined = tf.reshape(attn_output, (-1, tf.shape(z)[-1]))
        # 4) Decode z to produce reconstructions.
        recon_ts = self.decoder_ts(z_refined, mask_features=agg_features, training=training)
        recon_mask = self.decoder_mask(z_refined, ts_features=agg_features, training=training)
        return recon_ts, recon_mask, (mu_final, logvar_final, pi_final)

    def generate(self, z):
        z_tokens = tf.reshape(z, (-1, self.token_count, self.token_dim))
        attn_output = self.self_attention_layer(query=z_tokens, key=z_tokens, value=z_tokens)
        z_refined = tf.reshape(attn_output, (-1, tf.shape(z)[-1]))
        recon_ts = self.decoder_ts(z_refined, mask_features=None, training=False)
        recon_mask = self.decoder_mask(z_refined, ts_features=None, training=False)
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

# ~~~~~~~~~~~~~~~~ DEFINE PER-BRANCH MOG PARAMETERS ~~~~~~~~~~~~~~~~~
K = 5  # Number of mixture components
latent_dim_raw = 128 # Must match the latend_dim in main
latent_dim_fft = 128
latent_dim_rms = 128
latent_dim_psd = 128
latent_dim_mask = 128

def train_vae(model, train_dataset, val_dataset, optimizer, num_epochs=100, use_mog=True):
    """
    Train loop with tf.debugging checks.
    Returns 6 lists: train_total, train_recon, train_kl, val_total, val_recon, val_kl.
    """
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    # Collect model variables.
    all_trainables = model.trainable_variables

    train_total_losses = []
    train_recon_losses = []
    train_kl_losses = []

    val_total_losses = []
    val_recon_losses = []
    val_kl_losses = []

    for epoch in range(num_epochs):
        epoch_train_total = 0.0
        epoch_train_recon = 0.0
        epoch_train_kl = 0.0
        train_steps = 0

        for step, (raw_in, fft_in, rms_in, psd_in, mask_in, test_id_in) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                (recon_ts, recon_mask, (mu_final, logvar_final, pi_final)) = model(
                    raw_in, fft_in, rms_in, psd_in, mask_in, test_id_in, training=True
                )
                tf.debugging.assert_all_finite(recon_ts, "NaN in recon_ts")
                tf.debugging.assert_all_finite(recon_mask, "NaN in recon_mask")
                tf.debugging.assert_all_finite(mu_final, "NaN in mu_final")

                loss_ts = mse_loss_fn(raw_in, recon_ts)
                tf.debugging.assert_all_finite(loss_ts, "NaN in loss_ts")

                loss_mask_focal = focal_loss(mask_in, recon_mask, alpha=0.25, gamma=2.0)
                tf.debugging.assert_all_finite(loss_mask_focal, "NaN in focal_loss")
                loss_mask_dice = dice_loss(recon_mask, mask_in)
                tf.debugging.assert_all_finite(loss_mask_dice, "NaN in dice_loss")
                mask_recon_loss = 0.5 * (loss_mask_focal + loss_mask_dice)
                tf.debugging.assert_all_finite(mask_recon_loss, "NaN in mask_recon_loss")

                recon_loss = 0.5 * (loss_ts + mask_recon_loss)
                tf.debugging.assert_all_finite(recon_loss, "NaN in recon_loss")

                if use_mog:
                    latent_dim = tf.shape(mu_final)[-1]
                    num_mixtures = tf.shape(pi_final)[-1]
                    prior_mu = tf.zeros([num_mixtures, latent_dim])
                    prior_logvar = tf.zeros([num_mixtures, latent_dim])
                    prior_pi = tf.ones([num_mixtures]) / tf.cast(num_mixtures, tf.float32)
                    kl_loss = kl_q_p_mixture(
                        mu_q=mu_final, logvar_q=logvar_final, pi_q=pi_final,
                        mu_p=prior_mu, logvar_p=prior_logvar, pi_p=prior_pi,
                        num_samples=1
                    )
                else:
                    kl_loss = -0.5 * tf.reduce_mean(
                        1.0 + logvar_final - tf.square(mu_final) - tf.exp(logvar_final)
                    )
                tf.debugging.assert_all_finite(kl_loss, "NaN in kl_loss")

                total_loss = recon_loss + kl_loss
                tf.debugging.assert_all_finite(total_loss, "NaN in total_loss")

            grads = tape.gradient(total_loss, all_trainables)
            for i, g in enumerate(grads):
                if g is not None:
                    tf.debugging.assert_all_finite(g, f"NaN in grad {i}")
            optimizer.apply_gradients(zip(grads, all_trainables))

            epoch_train_total += total_loss.numpy()
            epoch_train_recon += recon_loss.numpy()
            epoch_train_kl += kl_loss.numpy()
            train_steps += 1

        if train_steps > 0:
            train_total_losses.append(epoch_train_total / train_steps)
            train_recon_losses.append(epoch_train_recon / train_steps)
            train_kl_losses.append(epoch_train_kl / train_steps)
        else:
            train_total_losses.append(None)
            train_recon_losses.append(None)
            train_kl_losses.append(None)

        epoch_val_total = 0.0
        epoch_val_recon = 0.0
        epoch_val_kl = 0.0
        val_steps = 0

        for step, (raw_in, fft_in, rms_in, psd_in, mask_in, test_id_in) in enumerate(val_dataset):
            (recon_ts, recon_mask, (mu_final, logvar_final, pi_final)) = model(
                raw_in, fft_in, rms_in, psd_in, mask_in, test_id_in, training=False
            )
            loss_ts = mse_loss_fn(raw_in, recon_ts)
            tf.debugging.assert_all_finite(loss_ts, "NaN in val loss_ts")
            loss_mask_focal = focal_loss(mask_in, recon_mask, alpha=0.25, gamma=2.0)
            tf.debugging.assert_all_finite(loss_mask_focal, "NaN in val focal_loss")
            loss_mask_dice = dice_loss(recon_mask, mask_in)
            tf.debugging.assert_all_finite(loss_mask_dice, "NaN in val dice_loss")
            mask_recon_loss = 0.5 * (loss_mask_focal + loss_mask_dice)
            tf.debugging.assert_all_finite(mask_recon_loss, "NaN in val mask_recon_loss")
            recon_loss = 0.5 * (loss_ts + mask_recon_loss)
            tf.debugging.assert_all_finite(recon_loss, "NaN in val recon_loss")

            if use_mog:
                latent_dim = tf.shape(mu_final)[-1]
                num_mixtures = tf.shape(pi_final)[-1]
                prior_mu = tf.zeros([num_mixtures, latent_dim])
                prior_logvar = tf.zeros([num_mixtures, latent_dim])
                prior_pi = tf.ones([num_mixtures]) / tf.cast(num_mixtures, tf.float32)
                kl_loss = kl_q_p_mixture(
                    mu_q=mu_final, logvar_q=logvar_final, pi_q=pi_final,
                    mu_p=prior_mu, logvar_p=prior_logvar, pi_p=prior_pi,
                    num_samples=1
                )
            else:
                kl_loss = -0.5 * tf.reduce_mean(
                    1.0 + logvar_final - tf.square(mu_final) - tf.exp(logvar_final)
                )
            tf.debugging.assert_all_finite(kl_loss, "NaN in val kl_loss")

            total_loss = recon_loss + kl_loss
            tf.debugging.assert_all_finite(total_loss, "NaN in val total_loss")

            epoch_val_total += total_loss.numpy()
            epoch_val_recon += recon_loss.numpy()
            epoch_val_kl += kl_loss.numpy()
            val_steps += 1

        if val_steps > 0:
            val_total_losses.append(epoch_val_total / val_steps)
            val_recon_losses.append(epoch_val_recon / val_steps)
            val_kl_losses.append(epoch_val_kl / val_steps)
        else:
            val_total_losses.append(None)
            val_recon_losses.append(None)
            val_kl_losses.append(None)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        if train_steps > 0:
            print(f"  Train => Total: {train_total_losses[-1]:.4f}, Recon: {train_recon_losses[-1]:.4f}, KL: {train_kl_losses[-1]:.4f}")
        if val_steps > 0:
            print(f"  Val   => Total: {val_total_losses[-1]:.4f}, Recon: {val_recon_losses[-1]:.4f}, KL: {val_kl_losses[-1]:.4f}")

    return (train_total_losses, train_recon_losses, train_kl_losses,
            val_total_losses, val_recon_losses, val_kl_losses)

#-------plots-------------------------
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
    configure_gpu()
    clear_gpu_memory()
    print_detailed_memory()
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # 1) Load data
    accel_dict, mask_dict = data_loader.load_data()
    gen = segment_and_transform(accel_dict, mask_dict)
    try:
        raw_segments, fft_segments, rms_segments, psd_segments, mask_segments, test_ids = next(gen)
    except StopIteration:
        raise ValueError("segment_and_transform() did not yield any data.")

    print("Finished segmenting data.")
    print(f"Data shapes after segmentation:")
    print(f"  Raw: {raw_segments.shape}")     # e.g., (600, 1000, 12)
    print(f"  FFT: {fft_segments.shape}")     # e.g., (600, 501, 12)
    print(f"  RMS: {rms_segments.shape}")      # e.g., (600, 12)
    print(f"  PSD: {psd_segments.shape}")      # e.g., (600, 129, 12)
    print(f"  Mask: {mask_segments.shape}")    # e.g., (600, 256, 768)
    print(f"  IDs: {test_ids.shape}")          # e.g., (600,)

    # 2) Convert to float32
    raw_segments  = raw_segments.astype(np.float32)
    fft_segments  = fft_segments.astype(np.float32)
    rms_segments  = rms_segments.astype(np.float32)
    psd_segments  = psd_segments.astype(np.float32)
    mask_segments = mask_segments.astype(np.float32)
    test_ids      = test_ids.astype(np.float32)

    # 3) Shuffle and split into train/val (80/20)
    N = raw_segments.shape[0]
    indices = np.random.permutation(N)
    train_size = int(0.8 * N)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

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

    # 4) Build tf.data Datasets.
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_raw, train_fft, train_rms, train_psd, train_mask, train_ids)
    ).batch(16)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_raw, val_fft, val_rms, val_psd, val_mask, val_ids)
    ).batch(16)

    print(f"Train batches: {len(train_dataset)}")
    print(f"Val batches:   {len(val_dataset)}")

    # 5) Build & train model.
    latent_dim = 128
    feature_dim = 128
    model = VAE(latent_dim, feature_dim)
    optimizer = tf.keras.optimizers.Adam(1e-5)
    print_detailed_memory()

    (train_total, train_recon, train_kl,
     val_total, val_recon, val_kl) = train_vae(
         model=model,
         train_dataset=train_dataset,
         val_dataset=val_dataset,
         optimizer=optimizer,
         num_epochs=300,
         use_mog=True
     )

    # 6) Save weights.
    model.save_weights("results/vae_mmodal_weights.h5")
    print("Saved model weights to results/vae_mmodal_weights.h5")

    # 7) Plot training curves using Plotly.
    plot_training_curves(train_total, train_recon, train_kl, val_total, val_recon, val_kl)

    # 8) Generate and plot 5 random samples.
    plot_generated_samples(model, latent_dim, num_samples=5)

    # 8) Generate synthetic data
    results_dir = "results/vae_results"
    os.makedirs(results_dir, exist_ok=True)

    num_synthetic = 50
    for i in range(num_synthetic):
        z_rand = tf.random.normal(shape=(1, latent_dim))
        gen_ts, gen_mask = model.generate(z_rand)

        np.save(os.path.join(results_dir, f"gen_ts_{i}.npy"), gen_ts.numpy())
        np.save(os.path.join(results_dir, f"gen_mask_{i}.npy"), gen_mask.numpy())
        logging.info(f"Saved synthetic sample {i}")


if __name__ == "__main__":
    main()
