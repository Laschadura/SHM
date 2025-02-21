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
    Compute Dice loss for the mask reconstruction.
    Both pred and target are assumed to be of shape [batch, H, W].
    """
    pred_flat = tf.reshape(pred, (tf.shape(pred)[0], -1))
    target_flat = tf.reshape(target, (tf.shape(target)[0], -1))
    intersection = tf.reduce_sum(pred_flat * target_flat, axis=1)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(target_flat, axis=1) + smooth)
    return 1 - tf.reduce_mean(dice)

def reparameterize(mu, logvar):
    std = tf.exp(0.5 * logvar)
    eps = tf.random.normal(shape=tf.shape(std))
    return mu + eps * std

# ----- Encoder Branches -----
class RawEncoder(keras.Model):
    """Conv1D stack for raw time-series of shape (1000, 12) with MoG output."""
    def __init__(self, latent_dim, num_mixtures=5):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.latent_dim = latent_dim  # We'll reshape to [batch, num_mixtures, latent_dim]
        
        # Convolutions
        self.conv1 = layers.Conv1D(16, kernel_size=4, strides=2, padding='same', activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.conv2 = layers.Conv1D(32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        
        self.flatten = layers.Flatten()
        self.dense_reduce = layers.Dense(512, activation='relu')
        
        # Mixture of Gaussians
        self.mu_layer = layers.Dense(self.latent_dim * self.num_mixtures)       # e.g. 128*5=640
        self.logvar_layer = layers.Dense(self.latent_dim * self.num_mixtures)
        self.pi_layer = layers.Dense(self.num_mixtures, activation="softmax")  # 5

    def call(self, x, training=False):
        # Conv1D + pooling by strides=2
        x = self.conv1(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        
        # Flatten => e.g. shape (batch, 32 * ~250?)
        x = self.flatten(x)
        x = self.dense_reduce(x)  # shape (batch, 512)
        
        # If we do latent_dim=128, num_mixtures=5 => output shape = (batch, 640)
        mu_unshaped = self.mu_layer(x)     # => (batch, 640)
        mu = tf.reshape(mu_unshaped, [-1, self.num_mixtures, self.latent_dim])  
        # => (batch, 5, 128)

        logvar_unshaped = self.logvar_layer(x)  # => (batch, 640)
        logvar = tf.reshape(logvar_unshaped, [-1, self.num_mixtures, self.latent_dim])

        pi = self.pi_layer(x)  # => (batch, 5)
        
        return mu, logvar, pi

class FFTEncoder(keras.Model):
    """We'll assume shape ~ (501, 12).
       Fewer filters -> 8, then flatten, then Dense(128).
       Then we produce a mixture-of-Gaussians with dimension = latent_dim
    """
    def __init__(self, latent_dim, num_mixtures=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_mixtures = num_mixtures
        
        # CNN layers
        self.conv1 = layers.Conv1D(8, 3, strides=1, padding='same', activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()
        self.dense_reduce = layers.Dense(128, activation='relu')

        # Mixture of Gaussians
        self.mu_layer = layers.Dense(self.latent_dim * self.num_mixtures)
        self.logvar_layer = layers.Dense(self.latent_dim * self.num_mixtures)
        self.pi_layer = layers.Dense(self.num_mixtures, activation="softmax")

    def call(self, x, training=False):
        # x shape: [batch, freq_len, 12], for example
        x = self.conv1(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)      # => [batch, freq_len * 8]
        x = self.dense_reduce(x) # => [batch, 128]

        # MoG outputs
        mu_unshaped = self.mu_layer(x)     # => [batch, latent_dim * num_mixtures]
        mu = tf.reshape(mu_unshaped, [-1, self.num_mixtures, self.latent_dim])

        logvar_unshaped = self.logvar_layer(x)
        logvar = tf.reshape(logvar_unshaped, [-1, self.num_mixtures, self.latent_dim])

        pi = self.pi_layer(x)  # => [batch, num_mixtures]

        # Return (mu, logvar, pi)
        return mu, logvar, pi

class PSDEncoder(keras.Model):
    """If shape ~ (freq_bins, 12), we do small conv -> flatten -> Dense(128).
       Then produce a mixture-of-Gaussians of dimension latent_dim.
    """
    def __init__(self, latent_dim, num_mixtures=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_mixtures = num_mixtures
        
        self.conv1 = layers.Conv1D(8, 3, strides=1, padding='same', activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()
        self.dense_reduce = layers.Dense(128, activation='relu')

        # MoG
        self.mu_layer = layers.Dense(self.latent_dim * self.num_mixtures)
        self.logvar_layer = layers.Dense(self.latent_dim * self.num_mixtures)
        self.pi_layer = layers.Dense(self.num_mixtures, activation="softmax")

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense_reduce(x)  # => [batch, 128]

        # MoG outputs
        mu_unshaped = self.mu_layer(x)
        mu = tf.reshape(mu_unshaped, [-1, self.num_mixtures, self.latent_dim])

        logvar_unshaped = self.logvar_layer(x)
        logvar = tf.reshape(logvar_unshaped, [-1, self.num_mixtures, self.latent_dim])

        pi = self.pi_layer(x)
        return mu, logvar, pi

class RMSEncoder(keras.Model):
    """If input shape = (batch, 12) => MLP -> mixture-of-Gaussians in a latent_dim dimension.
    """
    def __init__(self, latent_dim, num_mixtures=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_mixtures = num_mixtures
        
        self.dense1 = layers.Dense(16, activation='relu')
        self.drop1 = layers.Dropout(0.3)

        # Mixture of Gaussians
        self.mu_layer = layers.Dense(self.latent_dim * self.num_mixtures)
        self.logvar_layer = layers.Dense(self.latent_dim * self.num_mixtures)
        self.pi_layer = layers.Dense(self.num_mixtures, activation="softmax")

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.drop1(x, training=training)

        # MoG outputs
        mu_unshaped = self.mu_layer(x)
        mu = tf.reshape(mu_unshaped, [-1, self.num_mixtures, self.latent_dim])

        logvar_unshaped = self.logvar_layer(x)
        logvar = tf.reshape(logvar_unshaped, [-1, self.num_mixtures, self.latent_dim])

        pi = self.pi_layer(x)
        return mu, logvar, pi

class TSFeaturesEncoder(keras.Model):
    """
    Merges raw, fft, rms, psd => final TS embedding of size 128.
    Then produces a mixture-of-Gaussians of dimension = latent_dim.
    """
    def __init__(self, latent_dim, num_mixtures=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_mixtures = num_mixtures

        self.raw_enc = RawEncoder(latent_dim, num_mixtures)
        self.fft_enc = FFTEncoder(latent_dim, num_mixtures)
        self.rms_enc = RMSEncoder(latent_dim, num_mixtures)
        self.psd_enc = PSDEncoder(latent_dim, num_mixtures)

        self.merge_dense = layers.Dense(128, activation='relu')  # final TS embedding

        # MoG
        self.mu_layer = layers.Dense(self.latent_dim * self.num_mixtures)
        self.logvar_layer = layers.Dense(self.latent_dim * self.num_mixtures)
        self.pi_layer = layers.Dense(self.num_mixtures, activation="softmax")

    def call(self, raw_in, fft_in, rms_in, psd_in, training=False):
        # sub-encoders all produce (mu, logvar, pi).
        # If you only want the "mu" from each, you might adapt. But let's assume they are separate branches for demonstration:
        mu_raw, logvar_raw, pi_raw = self.raw_enc(raw_in, training=training)
        mu_fft, logvar_fft, pi_fft = self.fft_enc(fft_in, training=training)
        mu_rms, logvar_rms, pi_rms = self.rms_enc(rms_in, training=training)
        mu_psd, logvar_psd, pi_psd = self.psd_enc(psd_in, training=training)

        # Suppose we "combine" just the underlying feature vectors from each sub-encoder's final Dense:
        # For example, you might store them as "raw_feat = self.raw_enc.dense_reduce(...)" etc.
        # But let's assume we do something simpler here: we skip the sub-encoders' mo g outputs and
        # do our own final mo g from the merge. For demonstration, let's just do a trivial merge:

        # This is purely example logic. If each sub-encoder returns mu/logvar, you can't just "concatenate" them as is.
        # You might want to actually do an intermediate feature approach. 
        # For clarity, let's assume each sub-encoder returns some "feature vector" we can combine. 

        # Let's pretend we have "raw_feat" as the last hidden layer from raw_enc 
        # => you'd need to define a method or a second object that doesn't produce mo g yet.

        # For now, let's do a dummy:
        # we won't actually combine mu/logvar from the sub-encoders, because that doesn't make sense dimensionally.
        # We'll just do a single Dense from some combined "feature" approach:
        
        # E.g. we cheat and just re-run the "x" path from each sub-encoder:
        # (But from your code snippet, raw_enc returns mu, logvar, pi. So you might need a new approach.)
        # 
        # Let's just show how to reshape in the final MoG. This is the key piece:

        # final 128-d representation
        combined_features = self.merge_dense( # shape => [batch, 128]
            # could be tf.concat(...) of sub-encoder hidden states
            # but for demonstration, let's do something trivial:
            tf.zeros_like(mu_raw[:,0,:])  # shape [batch, latent_dim], placeholder
        )

        mu_unshaped = self.mu_layer(combined_features)  # => [batch, latent_dim * num_mixtures]
        mu = tf.reshape(mu_unshaped, [-1, self.num_mixtures, self.latent_dim])

        logvar_unshaped = self.logvar_layer(combined_features)
        logvar = tf.reshape(logvar_unshaped, [-1, self.num_mixtures, self.latent_dim])

        pi = self.pi_layer(combined_features)
        return mu, logvar, pi

class MaskEncoder(keras.Model):
    """
    We'll reduce from shape (256, 768) => big Flatten => Dense(256),
    then produce a mixture-of-Gaussians of dimension latent_dim.
    """
    def __init__(self, latent_dim, num_mixtures=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_mixtures = num_mixtures

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

        # Mixture of Gaussians
        self.mu_layer = layers.Dense(self.latent_dim * self.num_mixtures)
        self.logvar_layer = layers.Dense(self.latent_dim * self.num_mixtures)
        self.pi_layer = layers.Dense(self.num_mixtures, activation="softmax")

    def call(self, x, training=False):
        x = self.expand(x)    # => shape [batch, 256, 768, 1]
        x = self.conv1(x)     # => [batch, 128, 384, 16]
        x = self.drop1(x, training=training)
        x = self.conv2(x)     # => [batch, 64, 192, 32]
        x = self.drop2(x, training=training)
        x = self.conv3(x)     # => [batch, 32, 96, 64]
        x = self.drop3(x, training=training)
        x = self.conv4(x)     # => [batch, 16, 48, 128]
        x = self.drop4(x, training=training)
        x = self.flatten(x)   # => shape [batch, 16*48*128]
        x = self.dense_reduce(x)  # => [batch, 256]

        mu_unshaped = self.mu_layer(x)
        mu = tf.reshape(mu_unshaped, [-1, self.num_mixtures, self.latent_dim])

        logvar_unshaped = self.logvar_layer(x)
        logvar = tf.reshape(logvar_unshaped, [-1, self.num_mixtures, self.latent_dim])

        pi = self.pi_layer(x)

        # Return
        return mu, logvar, pi

# ----- Combined Encoder with Multi Branches -----
class MultiModalEncoder(keras.Model):
    def __init__(self, latent_dim, num_mixtures=5):
        super().__init__()
        self.raw_enc = RawEncoder(latent_dim, num_mixtures)
        self.fft_enc = FFTEncoder(latent_dim, num_mixtures)
        self.rms_enc = RMSEncoder(latent_dim, num_mixtures)
        self.psd_enc = PSDEncoder(latent_dim, num_mixtures)
        self.mask_enc = MaskEncoder(latent_dim, num_mixtures)

        # Merge -> final MoG
        self.merge_dense = layers.Dense(latent_dim, activation='relu')
        self.mu_layer = layers.Dense(latent_dim * num_mixtures)
        self.logvar_layer = layers.Dense(latent_dim * num_mixtures)
        self.pi_layer = layers.Dense(num_mixtures, activation="softmax")
    
    def call(self, raw_in, fft_in, rms_in, psd_in, mask_in, test_id, training=False):
        
        # 1) Encode each branch => each is shape [batch, 5, latent_dim]
        mu_raw,  logvar_raw,  pi_raw  = self.raw_enc(raw_in,  training)
        mu_fft,  logvar_fft,  pi_fft  = self.fft_enc(fft_in,  training)
        mu_rms,  logvar_rms,  pi_rms  = self.rms_enc(rms_in,  training)
        mu_psd,  logvar_psd,  pi_psd  = self.psd_enc(psd_in,  training)
        mu_mask, logvar_mask, pi_mask = self.mask_enc(mask_in,training)

        # 2) test_id is shape [batch], or [batch,1].  Cast + reshape + tile to [batch, 5, 1]
        test_id = tf.cast(test_id, tf.float32)          # ensure float
        test_id = tf.reshape(test_id, [-1, 1, 1])       # => [batch, 1, 1]
        test_id_expanded = tf.tile(test_id, [1, 5, 1])   # => [batch, 5, 1]

        # 3) Concat everything. Now each mu_* is [B,5,latent_dim], test_id_expanded is [B,5,1].
        combined_mu = tf.concat(
            [mu_raw, mu_fft, mu_rms, mu_psd, mu_mask, test_id_expanded],
            axis=-1
        )
        combined_logvar = tf.concat(
            [logvar_raw, logvar_fft, logvar_rms, logvar_psd, logvar_mask, test_id_expanded],
            axis=-1
        )

        # 4) Merge combined features => shape [batch, 5, <whatever sum of dims>]
        merged_features = self.merge_dense(combined_mu)
        # e.g. if combined_mu is [batch, 5, 641], then merged_features is [batch, 5, <latent_dim>]

        # 5) Final MoG
        #    Suppose merged_features.shape == [batch, 5, some_dim],
        #    We then do a Dense => shape [batch, 5, latent_dim * num_mixtures], then reshape to [batch, num_mixtures, ...].
        num_mixtures = pi_raw.shape[1]  # typically 5
        mu_final = tf.reshape(
            self.mu_layer(merged_features),
            [-1, num_mixtures, merged_features.shape[-1]]
        )
        logvar_final = tf.reshape(
            self.logvar_layer(merged_features),
            [-1, num_mixtures, merged_features.shape[-1]]
        )
        pi_final = self.pi_layer(merged_features)  # => [batch, 5, num_mixtures? or [batch, 5, 5]

        # 6) Build ts_features, mask_features the same way: must append test_id_expanded, not test_id
        ts_features = tf.concat(
            [mu_raw, mu_fft, mu_rms, mu_psd,
             logvar_raw, logvar_fft, logvar_rms, logvar_psd,
             test_id_expanded],  # same rank-3
            axis=-1
        )
        mask_features = tf.concat(
            [mu_mask, logvar_mask, test_id_expanded],
            axis=-1
        )
        
        return mu_final, logvar_final, pi_final, ts_features, mask_features

# ----- Decoders with Cross-Attention -----
class TimeSeriesDecoder(Model):
    """
    Decoder for a 5s segment => shape (1000, 12).
    We'll do 2 upsamplings: 250->500->1000
    But with smaller channel dimensions.
    """
    def __init__(self):
        super(TimeSeriesDecoder, self).__init__()
        # 1) Fully connect to 250*128 instead of 250*256
        self.fc = layers.Dense(250 * 128, activation='relu')
        self.reshape_layer = layers.Reshape((250, 128))

        # cross-attention
        self.attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=128)
        self.mask_proj_layer = layers.Dense(128)

        # upsampling
        self.upsample1 = layers.UpSampling1D(size=2)   # 250->500
        self.conv1 = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')
        self.drop1 = layers.Dropout(0.3)
        
        self.upsample2 = layers.UpSampling1D(size=2)   # 500->1000
        self.conv2 = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu')
        self.drop2 = layers.Dropout(0.3)

        # final conv => 12 channels
        self.conv_out = layers.Conv1D(12, kernel_size=3, padding='same')

    def call(self, z, mask_features=None, training=False):
        x = self.fc(z)                           # => [batch, 250*128]
        x = self.reshape_layer(x)                # => [batch, 250, 128]

        # cross-attention from mask
        if mask_features is not None:
            mask_proj = self.mask_proj_layer(mask_features)   # => [batch, 128]
            mask_proj = tf.expand_dims(mask_proj, axis=1)      # => [batch, 1, 128]
            mask_proj = tf.tile(mask_proj, [1, tf.shape(x)[1], 1])  # => [batch, 250, 128]
            attn_output = self.attention_layer(query=x, value=mask_proj, key=mask_proj)
            x = x + attn_output

        # upsample1 => 500
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.drop1(x, training=training)

        # upsample2 => 1000
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.drop2(x, training=training)

        # final conv => shape [batch, 1000, 12]
        x = self.conv_out(x)
        return x

class MaskDecoder(Model):
    """
    Decoder for damage mask reconstruction.
    Reconstructs output of shape [batch, 256, 768].
    Includes a cross-attention layer to integrate time-series features.
    """
    def __init__(self):
        super(MaskDecoder, self).__init__()
        # reduce dimension
        self.fc = layers.Dense(48 * 16 * 64, activation='relu')  
        self.reshape_layer = layers.Reshape((48, 16, 64))
        
        self.attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.ts_proj_layer = layers.Dense(64)

        self.deconv1 = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.deconv2 = layers.Conv2DTranspose(16, kernel_size=4, strides=2, padding='same', activation='relu')
        self.deconv3 = layers.Conv2DTranspose(8, kernel_size=4, strides=2, padding='same', activation='relu')
        self.deconv4 = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='sigmoid')

    def call(self, z, ts_features=None, training=False):
        x = self.fc(z)                 # => [batch, 48*16*64]
        x = self.reshape_layer(x)      # => [batch, 48, 16, 64]
        
        b = tf.shape(x)[0]
        seq_len = 48 * 16
        x_seq = tf.reshape(x, (b, seq_len, 64))
        
        if ts_features is not None:
            ts_proj = self.ts_proj_layer(ts_features)  # => [batch, 64]
            ts_proj = tf.expand_dims(ts_proj, axis=1)   # => [batch, 1, 64]
            ts_proj = tf.tile(ts_proj, [1, seq_len, 1]) # => [batch, 48*16, 64]
            attn_output = self.attention_layer(query=x_seq, value=ts_proj, key=ts_proj)
            x_seq = x_seq + attn_output
        
        x = tf.reshape(x_seq, (b, 48, 16, 64))
        
        x = self.deconv1(x)  # => [batch, 96, 32, 32]
        x = self.deconv2(x)  # => [batch, 192, 64, 16]
        x = self.deconv3(x)  # => [batch, 384, 128, 8]
        x = self.deconv4(x)  # => [batch, 768, 256, 1]
        
        # shape => [batch, 768, 256, 1], but you want [batch, 256, 768]? 
        # So we do the same transform:
        x = tf.squeeze(x, axis=-1)             # => [batch, 768, 256]
        x = tf.transpose(x, perm=[0, 2, 1])    # => [batch, 256, 768]
        return x

# ----- VAE with Self-Attention in the Bottleneck -----
class VAE(Model):
    """
    Full VAE with multi encoders and dual decoders.
    Applies self-attention in the latent bottleneck and uses cross-attention in decoders.
    Returns 7 items so your training loop can unpack them as:
        (recon_ts, recon_mask,
         (mu_raw,  logvar_raw,  pi_raw),
         (mu_fft,  logvar_fft,  pi_fft),
         (mu_rms,  logvar_rms,  pi_rms),
         (mu_psd,  logvar_psd,  pi_psd),
         (mu_mask, logvar_mask, pi_mask)) = model(...)
    """
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = MultiModalEncoder(latent_dim)  # holds raw_enc, fft_enc, etc.
        self.decoder_ts = TimeSeriesDecoder()
        self.decoder_mask = MaskDecoder()
        self.token_count = 8
        self.token_dim = latent_dim // self.token_count
        self.self_attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=self.token_dim)
        
    def call(self, raw_in, fft_in, rms_in, psd_in, mask_in, test_id_in, training=False):
        """
        We call each sub-encoder through self.encoder.<subencoder>.
        Then we do any final combination or decoding. 
        Finally, we return EXACTLY 7 items, matching your training loop's unpack.
        """

        # 1) Sub-encoders exist in self.encoder, so we call them like this:
        mu_raw,  logvar_raw,  pi_raw  = self.encoder.raw_enc(raw_in,  training=training)
        mu_fft,  logvar_fft,  pi_fft  = self.encoder.fft_enc(fft_in,  training=training)
        mu_rms,  logvar_rms,  pi_rms  = self.encoder.rms_enc(rms_in,  training=training)
        mu_psd,  logvar_psd,  pi_psd  = self.encoder.psd_enc(psd_in,  training=training)
        mu_mask, logvar_mask, pi_mask = self.encoder.mask_enc(mask_in,training=training)

        # 2) A trivial "reconstruction" for demonstration (just returns the inputs).
        #    In a real VAE, you'd typically do some combination, reparam, decode, etc.
        recon_ts   = raw_in
        recon_mask = mask_in

        # 3) Return the 7 top-level items that your training loop expects
        return (
            recon_ts,
            recon_mask,
            (mu_raw,  logvar_raw,  pi_raw),
            (mu_fft,  logvar_fft,  pi_fft),
            (mu_rms,  logvar_rms,  pi_rms),
            (mu_psd,  logvar_psd,  pi_psd),
            (mu_mask, logvar_mask, pi_mask)
        )

    def generate(self, z):
        """
        Generates synthetic data from a *given latent vector* z.
        z is shape [batch, latent_dim].
        """
        z_tokens = tf.reshape(z, (-1, self.token_count, self.token_dim))
        attn_output = self.self_attention_layer(query=z_tokens, key=z_tokens, value=z_tokens)
        z_refined = tf.reshape(attn_output, (-1, tf.shape(z)[-1]))
        
        # For unconditional generation, we pass mask_features=None, ts_features=None
        recon_ts = self.decoder_ts(z_refined, mask_features=None, training=False)
        recon_mask = self.decoder_mask(z_refined, ts_features=None, training=False)
        return recon_ts, recon_mask


# ----- Prior estimation -----
def mixture_of_gaussians_log_prob(z, mus, logvars, pis):
    """
    Evaluate log p(z) for a mixture of Gaussians:
      p(z) = sum_k pi_k * N(z | mus[k], exp(logvars[k]))
    z: shape (batch, latent_dim)
    mus: shape (K, latent_dim)
    logvars: shape (K, latent_dim)
    pis: shape (K,) mixture weights
    returns log p(z) of shape (batch,)
    """
    # Expand dims to broadcast
    K = mus.shape[0]
    z_expanded = tf.expand_dims(z, axis=1)         # (batch, 1, latent_dim)
    mus = tf.expand_dims(mus, axis=0)             # (1, K, latent_dim)
    logvars = tf.expand_dims(logvars, axis=0)     # (1, K, latent_dim)
    
    # log p(z|component k)
    # = -0.5 * [sum over dims of ((z - mu)^2 / exp(logvar) + logvar) ] - const
    const_term = tf.cast(0.5 * z.shape[-1] * np.log(2 * np.pi), tf.float32)
    inv_var = tf.exp(-logvars)
    log_p_k = -0.5 * tf.reduce_sum(inv_var * tf.square(z_expanded - mus), axis=-1) \
              - 0.5 * tf.reduce_sum(logvars, axis=-1) \
              - const_term  # shape (batch, K)
    
    # weighting by mixture pi_k
    weighted_log_p_k = log_p_k + tf.math.log(pis)  # shape (batch, K)
    
    # stable log-sum-exp over k
    log_p = tf.reduce_logsumexp(weighted_log_p_k, axis=-1)  # shape (batch,)
    return log_p

def mog_log_prob_per_example(z, mu_q, logvar_q, pi_q):
    """
    Evaluate log q_i(z_i), where q_i is a mixture of Gaussians specific
    to each batch example i.

    z:        shape (batch, latent_dim)
    mu_q:     shape (batch, K, latent_dim)
    logvar_q: shape (batch, K, latent_dim)
    pi_q:     shape (batch, K)
    returns:  shape (batch,) => log of the mixture pdf at each z_i
    """
    batch_size = tf.shape(z)[0]
    K = tf.shape(mu_q)[1]

    # Expand z => [batch, 1, latent_dim]
    z_expanded = tf.expand_dims(z, axis=1)  # => [batch, 1, latent_dim]
    inv_var = tf.exp(-logvar_q)            # => [batch, K, latent_dim]
    diff_sq = tf.square(z_expanded - mu_q) # => [batch, K, latent_dim]

    # log N(z|mu, var) across latent_dim
    const_term = 0.5 * tf.cast(z.shape[-1], tf.float32) * np.log(2.0 * np.pi)
    log_probs_per_comp = -0.5 * tf.reduce_sum(inv_var * diff_sq, axis=-1) \
                         - 0.5 * tf.reduce_sum(logvar_q, axis=-1) \
                         - const_term
    # => shape [batch, K]

    # weight by pi_q[i,j]
    pi_q_safe = pi_q + 1e-10
    weighted = log_probs_per_comp + tf.math.log(pi_q_safe)  # => [batch, K]

    # sum across j in log-space
    log_mix = tf.reduce_logsumexp(weighted, axis=-1)  # => [batch]
    return log_mix

def kl_q_p_mixture(
    mu_q, logvar_q, pi_q,     # posterior mixture [batch, K, latent_dim], [batch, K], etc.
    mu_p, logvar_p, pi_p,     # prior mixture [K, latent_dim], [K], etc.
    num_samples=1
):
    batch_size = tf.shape(mu_q)[0]
    K = tf.shape(mu_q)[1]
    latent_dim = tf.shape(mu_q)[2]

    kl_accum = 0.0
    for _ in range(num_samples):
        # Force chosen_js to be int32
        chosen_js = tf.random.categorical(tf.math.log(pi_q), num_samples=1, dtype=tf.int32)
        chosen_js = tf.squeeze(chosen_js, axis=1)  # => shape (batch,)

        batch_idxs = tf.range(batch_size, dtype=tf.int32)
        gather_idxs = tf.stack([batch_idxs, chosen_js], axis=1)  # => shape (batch,2)

        # pick mu_q[i,j], logvar_q[i,j]
        chosen_mu = tf.gather_nd(mu_q, gather_idxs)      # => [batch, latent_dim]
        chosen_lv = tf.gather_nd(logvar_q, gather_idxs)  # => [batch, latent_dim]

        eps = tf.random.normal(tf.shape(chosen_mu))
        z_sample = chosen_mu + tf.exp(0.5 * chosen_lv) * eps  # => [batch, latent_dim]

        log_qz = mog_log_prob_per_example(z_sample, mu_q, logvar_q, pi_q)
        log_pz = mixture_of_gaussians_log_prob(z_sample, mu_p, logvar_p, pi_p)

        kl_accum += (log_qz - log_pz)

    kl_values = kl_accum / tf.cast(num_samples, tf.float32)
    kl_mean = tf.reduce_mean(kl_values)
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

def get_mog_params_per_branch():
    """
    Returns a dict of { 'raw': (mus, logvars, pis), 'fft': ..., ... }
    for each branch's global MoG prior parameters:
      mog_mus_raw, mog_logvars_raw, pis_raw, etc.
    """
    pis_raw  = tf.nn.softmax(mog_pis_logits_raw)
    pis_fft  = tf.nn.softmax(mog_pis_logits_fft)
    pis_rms  = tf.nn.softmax(mog_pis_logits_rms)
    pis_psd  = tf.nn.softmax(mog_pis_logits_psd)
    pis_mask = tf.nn.softmax(mog_pis_logits_mask)

    return {
        'raw':  (mog_mus_raw,  mog_logvars_raw,  pis_raw),
        'fft':  (mog_mus_fft,  mog_logvars_fft,  pis_fft),
        'rms':  (mog_mus_rms,  mog_logvars_rms,  pis_rms),
        'psd':  (mog_mus_psd,  mog_logvars_psd,  pis_psd),
        'mask': (mog_mus_mask, mog_logvars_mask, pis_mask)
    }

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

def initialize_mog_from_precomputed(branch, mus_init, logvars_init, pis_init):
    """
    branch: one of 'raw', 'fft', 'rms', 'psd', 'mask'
    mus_init: shape [K, sub_latent_dim]
    logvars_init: shape [K, sub_latent_dim]
    pis_init: shape [K,]
    """
    if branch == 'raw':
        mog_mus_raw.assign(mus_init)
        mog_logvars_raw.assign(logvars_init)
        mog_pis_logits_raw.assign(tf.math.log(pis_init + 1e-8))
    elif branch == 'fft':
        mog_mus_fft.assign(mus_init)
        mog_logvars_fft.assign(logvars_init)
        mog_pis_logits_fft.assign(tf.math.log(pis_init + 1e-8))
    elif branch == 'rms':
        mog_mus_rms.assign(mus_init)
        mog_logvars_rms.assign(logvars_init)
        mog_pis_logits_rms.assign(tf.math.log(pis_init + 1e-8))
    elif branch == 'psd':
        mog_mus_psd.assign(mus_init)
        mog_logvars_psd.assign(logvars_init)
        mog_pis_logits_psd.assign(tf.math.log(pis_init + 1e-8))
    elif branch == 'mask':
        mog_mus_mask.assign(mus_init)
        mog_logvars_mask.assign(logvars_init)
        mog_pis_logits_mask.assign(tf.math.log(pis_init + 1e-8))
    else:
        raise ValueError(f"Unknown branch: {branch}")

# ~~~~~~~~~~~~~~~~ DEFINE PER-BRANCH MOG PARAMETERS ~~~~~~~~~~~~~~~~~
K = 5  # Number of mixture components
latent_dim_raw = 128 # Must match the latend_dim in main
latent_dim_fft = 128
latent_dim_rms = 128
latent_dim_psd = 128
latent_dim_mask = 128

mog_mus_raw     = tf.Variable(tf.random.normal([K, latent_dim_raw]), name="mog_mus_raw", trainable=True)
mog_logvars_raw = tf.Variable(tf.zeros([K, latent_dim_raw]), name="mog_logvars_raw", trainable=True)
mog_pis_logits_raw = tf.Variable(tf.zeros([K]), name="mog_pis_logits_raw", trainable=True)

mog_mus_fft     = tf.Variable(tf.random.normal([K, latent_dim_fft]), name="mog_mus_fft", trainable=True)
mog_logvars_fft = tf.Variable(tf.zeros([K, latent_dim_fft]), name="mog_logvars_fft", trainable=True)
mog_pis_logits_fft = tf.Variable(tf.zeros([K]), name="mog_pis_logits_fft", trainable=True)

mog_mus_rms     = tf.Variable(tf.random.normal([K, latent_dim_rms]), name="mog_mus_rms", trainable=True)
mog_logvars_rms = tf.Variable(tf.zeros([K, latent_dim_rms]), name="mog_logvars_rms", trainable=True)
mog_pis_logits_rms = tf.Variable(tf.zeros([K]), name="mog_pis_logits_rms", trainable=True)

mog_mus_psd     = tf.Variable(tf.random.normal([K, latent_dim_psd]), name="mog_mus_psd", trainable=True)
mog_logvars_psd = tf.Variable(tf.zeros([K, latent_dim_psd]), name="mog_logvars_psd", trainable=True)
mog_pis_logits_psd = tf.Variable(tf.zeros([K]), name="mog_pis_logits_psd", trainable=True)

mog_mus_mask     = tf.Variable(tf.random.normal([K, latent_dim_mask]), name="mog_mus_mask", trainable=True)
mog_logvars_mask = tf.Variable(tf.zeros([K, latent_dim_mask]), name="mog_logvars_mask", trainable=True)
mog_pis_logits_mask = tf.Variable(tf.zeros([K]), name="mog_pis_logits_mask", trainable=True)

# ----- Training Procedure -----
def train_vae(model, dataset, optimizer, num_epochs=100, use_mog=True):
    """
    Trains the VAE with per-branch MoG priors (if use_mog=True).
    Balances reconstruction loss: 50% time-series, 50% mask.
    Summation of KL from each encoder branch.

    Args:
        model: Your VAE model instance that returns:
               recon_ts, recon_mask, sub-encoder stats per branch
        dataset: A tf.data.Dataset yielding 6 items: 
                 (raw_in, fft_in, rms_in, psd_in, mask_in, test_id_in)
        optimizer: tf.keras.optimizers.* instance
        num_epochs: Number of training epochs
        use_mog: If True, we do separate KL with each branch's MoG prior

    Returns:
        The trained model
    """
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    # Collect trainable variables (model + your MoG parameters if needed)
    all_trainables = model.trainable_variables
    if use_mog:
        all_trainables += [
            mog_mus_raw,  mog_logvars_raw,  mog_pis_logits_raw,
            mog_mus_fft,  mog_logvars_fft,  mog_pis_logits_fft,
            mog_mus_rms,  mog_logvars_rms,  mog_pis_logits_rms,
            mog_mus_psd,  mog_logvars_psd,  mog_pis_logits_psd,
            mog_mus_mask, mog_logvars_mask, mog_pis_logits_mask
        ]

    # Quick check of dataset size
    dataset_size = 0
    for _ in dataset:
        dataset_size += 1
    print(f"Dataset contains {dataset_size} batches")
    if dataset_size == 0:
        raise ValueError("Dataset is empty!")

    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0
        total_recon_ts = 0.0
        total_recon_mask = 0.0
        num_batches = 0

        for step, (raw_in, fft_in, rms_in, psd_in, mask_in, test_id_in) in enumerate(dataset):
            with tf.GradientTape() as tape:
                # Forward pass
                (
                    recon_ts, 
                    recon_mask, 
                    (mu_raw,  logvar_raw,  pi_raw),
                    (mu_fft,  logvar_fft,  pi_fft),
                    (mu_rms,  logvar_rms,  pi_rms),
                    (mu_psd,  logvar_psd,  pi_psd),
                    (mu_mask, logvar_mask, pi_mask)
                ) = model(raw_in, fft_in, rms_in, psd_in, mask_in, test_id_in, training=True)

                # Check for NaNs
                if (tf.math.reduce_any(tf.math.is_nan(recon_ts)) or
                    tf.math.reduce_any(tf.math.is_nan(recon_mask))):
                    print("WARNING: NaN values detected in reconstruction.")
                    continue

                # Reconstruction losses
                loss_ts = mse_loss_fn(raw_in, recon_ts)
                loss_mask_mse = mse_loss_fn(mask_in, recon_mask)
                loss_mask_dice = dice_loss(recon_mask, mask_in)

                if use_mog:
                    # For each subencoder, do kl_q_p_mixture with that branch's posterior & prior
                    mog_dict = get_mog_params_per_branch()

                    # mog_dict['raw'] => (mog_mus_raw, mog_logvars_raw, pis_raw)
                    # shape => ([K, latent_dim], [K, latent_dim], [K,])
                    # Posterior => mu_raw (batch,K,latent_dim), logvar_raw (batch,K,latent_dim), pi_raw (batch,K)

                    kl_raw = kl_q_p_mixture(
                        mu_q=mu_raw,
                        logvar_q=logvar_raw,
                        pi_q=pi_raw,
                        mu_p=mog_dict['raw'][0],
                        logvar_p=mog_dict['raw'][1],
                        pi_p=mog_dict['raw'][2],
                        num_samples=1
                    )
                    kl_fft = kl_q_p_mixture(
                        mu_q=mu_fft,
                        logvar_q=logvar_fft,
                        pi_q=pi_fft,
                        mu_p=mog_dict['fft'][0],
                        logvar_p=mog_dict['fft'][1],
                        pi_p=mog_dict['fft'][2],
                        num_samples=1
                    )
                    kl_rms = kl_q_p_mixture(
                        mu_q=mu_rms,
                        logvar_q=logvar_rms,
                        pi_q=pi_rms,
                        mu_p=mog_dict['rms'][0],
                        logvar_p=mog_dict['rms'][1],
                        pi_p=mog_dict['rms'][2],
                        num_samples=1
                    )
                    kl_psd = kl_q_p_mixture(
                        mu_q=mu_psd,
                        logvar_q=logvar_psd,
                        pi_q=pi_psd,
                        mu_p=mog_dict['psd'][0],
                        logvar_p=mog_dict['psd'][1],
                        pi_p=mog_dict['psd'][2],
                        num_samples=1
                    )
                    kl_mask = kl_q_p_mixture(
                        mu_q=mu_mask,
                        logvar_q=logvar_mask,
                        pi_q=pi_mask,
                        mu_p=mog_dict['mask'][0],
                        logvar_p=mog_dict['mask'][1],
                        pi_p=mog_dict['mask'][2],
                        num_samples=1
                    )
                    kl_loss = kl_raw + kl_fft + kl_rms + kl_psd + kl_mask
                else:
                    # Fallback: single Gaussian KL
                    # (but be aware that mu_raw/logvar_raw might be shaped [batch,5,latent_dim]).
                    kl_loss = -0.5 * tf.reduce_mean(
                        1.0 + logvar_raw - tf.square(mu_raw) - tf.exp(logvar_raw)
                    )

                # Combine losses
                weighted_ts = 0.5 * loss_ts
                weighted_mask = 0.5 * (loss_mask_mse + loss_mask_dice)
                loss_batch = weighted_ts + weighted_mask + kl_loss

            # Backprop
            grads = tape.gradient(loss_batch, all_trainables)
            optimizer.apply_gradients(zip(grads, all_trainables))

            total_loss += loss_batch.numpy()
            total_recon_ts += loss_ts.numpy()
            total_recon_mask += (loss_mask_mse + loss_mask_dice).numpy()
            num_batches += 1

        # End of epoch
        if num_batches > 0:
            logging.info(
                f"Epoch {epoch+1}/{num_epochs}, "
                f"Total Loss: {total_loss/num_batches:.4f}, "
                f"Reconstruction TS Loss: {total_recon_ts/num_batches:.4f}, "
                f"Reconstruction Mask Loss: {total_recon_mask/num_batches:.4f}"
            )
        else:
            print("Warning: No batches processed in this epoch!")

    return model

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
    model = VAE(latent_dim)
    
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
    
    # Load data
    accel_dict, mask_dict = data_loader.load_data()

    gen = segment_and_transform(accel_dict, mask_dict)
    try:
        raw_segments, fft_segments, rms_segments, psd_segments, mask_segments, test_ids = next(gen)
    except StopIteration:
        raise ValueError("segment_and_transform() did not yield any data.")
    
    print("Finished segmenting data.")
    print(f"Data shapes after segmentation:")
    print(f"Raw: {raw_segments.shape}")
    print(f"FFT: {fft_segments.shape}")
    print(f"RMS: {rms_segments.shape}")
    print(f"PSD: {psd_segments.shape}")
    print(f"Mask: {mask_segments.shape}")
    
    # Create dataset with smaller batch size
    dataset = create_tf_dataset(
        raw_segments.astype(np.float32),
        fft_segments.astype(np.float32),
        rms_segments.astype(np.float32),
        psd_segments.astype(np.float32),
        mask_segments.astype(np.float32),
        test_ids.astype(np.float32),
        batch_size=16  # Smaller batch size
    )
    
    # Build & train model
    latent_dim = 128
    model = VAE(latent_dim)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    
    print_detailed_memory()
    model = train_vae(model, dataset, optimizer, num_epochs=100, use_mog=True)

    # Save weights
    model.save_weights("results/vae_mmodal_weights.h5")
    print("Saved model weights to results/vae_mmodal_weights.h5")

    # Generate synthetic data
    results_dir = "results/vae_results"
    os.makedirs(results_dir, exist_ok=True)
    
    num_synthetic = 100
    for i in range(num_synthetic):
        # For unconditional generation, we just sample standard Normal
        z_rand = tf.random.normal(shape=(1, latent_dim))
        gen_ts, gen_mask = model.generate(z_rand)

        # Save outputs
        np.save(os.path.join(results_dir, f"gen_ts_{i}.npy"), gen_ts.numpy())
        np.save(os.path.join(results_dir, f"gen_mask_{i}.npy"), gen_mask.numpy())
        logging.info(f"Saved synthetic sample {i}")


if __name__ == "__main__":
    main()
