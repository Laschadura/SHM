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
            
def create_tf_dataset(raw_segments, fft_segments, rms_segments, psd_segments, mask_segments, batch_size=8):
    print(f"Creating dataset with shapes:")
    print(f"Raw segments: {raw_segments.shape}")
    print(f"FFT segments: {fft_segments.shape}")
    print(f"RMS segments: {rms_segments.shape}")
    print(f"PSD segments: {psd_segments.shape}")
    print(f"Mask segments: {mask_segments.shape}")
    
    dataset = tf.data.Dataset.from_tensor_slices((
        raw_segments,
        fft_segments,
        rms_segments,
        psd_segments,
        mask_segments
    ))
    
    # Shuffle with a reasonable buffer size
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Verify dataset
    for batch in dataset.take(1):
        print("\nFirst batch shapes:")
        print(f"Raw: {batch[0].shape}")
        print(f"FFT: {batch[1].shape}")
        print(f"RMS: {batch[2].shape}")
        print(f"PSD: {batch[3].shape}")
        print(f"Mask: {batch[4].shape}")
    
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
    """Conv1D stack for raw time-series of shape (1000, 12)."""
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv1D(16, kernel_size=4, strides=2, padding='same', activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.conv2 = layers.Conv1D(32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()

        # Additional Dense to reduce dimension after flatten
        self.dense_reduce = layers.Dense(512, activation='relu')

    def call(self, x, training=False):
        # x shape: [batch, 1000, 12]
        x = self.conv1(x)        # => [batch, 500, 16]
        x = self.dropout1(x, training=training)
        x = self.conv2(x)        # => [batch, 250, 32]
        x = self.dropout2(x, training=training)
        x = self.flatten(x)      # => [batch, 250*32] = 8000
        x = self.dense_reduce(x) # => [batch, 512]
        return x

class FFTEncoder(keras.Model):
    """We'll assume shape ~ (501, 12). 
       Fewer filters -> 8, then flatten, then Dense(128).
    """
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv1D(8, 3, strides=1, padding='same', activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()
        self.dense_reduce = layers.Dense(128, activation='relu')
    def call(self, x, training=False):
        x = self.conv1(x)        # => [batch, freq_len, 8]
        x = self.dropout1(x, training=training)
        x = self.flatten(x)      # => [batch, freq_len*8]
        x = self.dense_reduce(x) # => [batch, 128]
        return x

class PSDEncoder(keras.Model):
    """If shape ~ (freq_bins, 12), we do small conv -> flatten -> Dense(128)."""
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv1D(8, 3, strides=1, padding='same', activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()
        self.dense_reduce = layers.Dense(128, activation='relu')
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense_reduce(x)
        return x

class RMSEncoder(keras.Model):
    """If shape = (12,) => MLP to reduce to 16 dims."""
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(16, activation='relu')
        self.drop1 = layers.Dropout(0.3)
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.drop1(x, training=training)
        return x  # => shape [batch, 16]

class TSFeaturesEncoder(keras.Model):
    """
    Merges raw, fft, rms, psd => final TS embedding of size 128
    """
    def __init__(self):
        super().__init__()
        self.raw_enc = RawEncoder()
        self.fft_enc = FFTEncoder()
        self.rms_enc = RMSEncoder()
        self.psd_enc = PSDEncoder()
        self.merge_dense = layers.Dense(128, activation='relu')  # final TS embedding

    def call(self, raw_in, fft_in, rms_in, psd_in, training=False):
        raw_feat = self.raw_enc(raw_in, training=training)   
        fft_feat = self.fft_enc(fft_in, training=training)   
        rms_feat = self.rms_enc(rms_in, training=training)   
        psd_feat = self.psd_enc(psd_in, training=training)
        
        combined = layers.concatenate([raw_feat, fft_feat, rms_feat, psd_feat], axis=-1)
        # shape: [batch, 512 + 128 + 128 + 16 = 784 approx]
        
        x = self.merge_dense(combined)  # => [batch, 128]
        return x

class MaskEncoder(Model):
    """
    We'll reduce from 32,64,128,256 -> 16,32,64,128 
    and add a final dense to reduce dimension further.
    """
    def __init__(self):
        super(MaskEncoder, self).__init__()
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
        # reduce dimension with a Dense
        self.dense_reduce = layers.Dense(256, activation='relu')
        
    def call(self, x, training=False):
        x = self.expand(x)    # [batch, 256, 768, 1]
        x = self.conv1(x)     # => [batch, 128, 384, 16]
        x = self.drop1(x, training=training)
        x = self.conv2(x)     # => [batch, 64, 192, 32]
        x = self.drop2(x, training=training)
        x = self.conv3(x)     # => [batch, 32, 96, 64]
        x = self.drop3(x, training=training)
        x = self.conv4(x)     # => [batch, 16, 48, 128]
        x = self.drop4(x, training=training)
        x = self.flatten(x)   # => shape [batch, 16*48*128] = 98304 (still big, but less than original)
        x = self.dense_reduce(x)  # => [batch, 256]
        return x

# ----- Combined Encoder with Multi Branches -----
class MultiModalEncoder(keras.Model):
    """
    Combines the multi-branch TSFeaturesEncoder + MaskEncoder.
    Then outputs mu, logvar, plus the separate TS/mask features for cross-attention.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.ts_encoder = TSFeaturesEncoder()
        self.mask_encoder = MaskEncoder()
        self.fc_mu = layers.Dense(latent_dim)
        self.fc_logvar = layers.Dense(latent_dim)
    
    def call(self, raw_in, fft_in, rms_in, psd_in, mask_in, training=False):
        ts_feat = self.ts_encoder(raw_in, fft_in, rms_in, psd_in, training=training)
        mask_feat = self.mask_encoder(mask_in, training=training)
        combined = layers.concatenate([ts_feat, mask_feat], axis=-1)
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar, ts_feat, mask_feat

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
    """
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = MultiModalEncoder(latent_dim)
        self.decoder_ts = TimeSeriesDecoder()
        self.decoder_mask = MaskDecoder()
        self.token_count = 8
        self.token_dim = latent_dim // self.token_count
        self.self_attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=self.token_dim)
        
    def call(self, raw_in, fft_in, rms_in, psd_in, mask_in, training=False):
        # Pass all five to the encoder:
        mu, logvar, ts_features, mask_features = self.encoder(raw_in, fft_in, rms_in, psd_in, mask_in, training=training)
        
        z = reparameterize(mu, logvar)
        z_tokens = tf.reshape(z, (-1, self.token_count, self.token_dim))
        attn_output = self.self_attention_layer(query=z_tokens, key=z_tokens, value=z_tokens)
        z_refined = tf.reshape(attn_output, (-1, tf.shape(z)[-1]))
        
        recon_ts = self.decoder_ts(z_refined, mask_features=mask_features, training=training)
        recon_mask = self.decoder_mask(z_refined, ts_features=ts_features, training=training)
        return recon_ts, recon_mask, mu, logvar

    def generate(self, z):
        """
        Generate synthetic data from a latent vector z.
        Applies self-attention on z and decodes using both decoders.
        """
        z_tokens = tf.reshape(z, (-1, self.token_count, self.token_dim))
        attn_output = self.self_attention_layer(query=z_tokens, key=z_tokens, value=z_tokens)
        z_refined = tf.reshape(attn_output, (-1, tf.shape(z)[-1]))
        recon_ts = self.decoder_ts(z_refined, mask_features=None)
        recon_mask = self.decoder_mask(z_refined, ts_features=None)
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

def kl_q_p_mog(mu, logvar, mus_mixture, logvars_mixture, pis_mixture):
    """
    Approximate KL(q(z|x) || p(z)) using a single sample from q.
    """
    # Reparameterize from q
    std = tf.exp(0.5 * logvar)
    eps = tf.random.normal(shape=tf.shape(std))
    z_sample = mu + eps * std
    
    # log q(z|x)
    log_qz_x = gaussian_log_prob(z_sample, mu, logvar)
    # log p(z)
    log_pz = mixture_of_gaussians_log_prob(z_sample, mus_mixture, logvars_mixture, pis_mixture)
    
    return tf.reduce_mean(log_qz_x - log_pz)

def gaussian_log_prob(z, mu, logvar):
    """
    log N(z | mu, exp(logvar)) i.i.f. over the last dimension
    """
    const_term = 0.5 * z.shape[-1] * np.log(2 * np.pi)
    inv_var = tf.exp(-logvar)
    tmp = tf.reduce_sum(inv_var * tf.square(z - mu), axis=-1)
    log_det = 0.5 * tf.reduce_sum(logvar, axis=-1)
    return tf.cast(const_term, tf.float32) + 0.5 * tmp + log_det

def get_mog_params():
    """
    Return the mixture means, log-variances, and mixture weights 
    from the trainable variables.
    """
    pis = tf.nn.softmax(mog_pis_logits)  # shape [K], sums to 1
    return mog_mus, mog_logvars, pis


# ~~~~~~~~~~~~~~~~ DEFINE MOG PARAMETERS ~~~~~~~~~~~~~~~~~
K = 5  # number of mixture components
latent_dim = 128  # must match the latent dimension of your VAE

# We'll create trainable variables for the mixture means, log-variances, and mixture logits:
mog_mus = tf.Variable(tf.random.normal([K, latent_dim]), name="mog_mus", trainable=True)
mog_logvars = tf.Variable(tf.zeros([K, latent_dim]), name="mog_logvars", trainable=True)
mog_pis_logits = tf.Variable(tf.zeros([K]), name="mog_pis_logits", trainable=True)


# ----- Training Procedure -----
def train_vae(model, dataset, optimizer, num_epochs=10, use_mog=False):
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    
    all_trainables = model.trainable_variables
    if use_mog:
        all_trainables += [mog_mus, mog_logvars, mog_pis_logits]

    # Add dataset size check
    dataset_size = 0
    for _ in dataset:
        dataset_size += 1
    print(f"Dataset contains {dataset_size} batches")
    
    if dataset_size == 0:
        raise ValueError("Dataset is empty!")

    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch+1}/{num_epochs}")
        total_loss = 0.
        total_recon_ts = 0.
        total_recon_mask = 0.
        num_batches = 0
        
        for batch_data in dataset:
            print(f"Processing batch {num_batches+1}", end='\r')
            
            try:
                raw_in, fft_in, rms_in, psd_in, mask_in = batch_data
                
                with tf.GradientTape() as tape:
                    # Forward pass
                    recon_ts, recon_mask, mu, logvar = model(
                        raw_in, fft_in, rms_in, psd_in, mask_in, 
                        training=True
                    )

                    # Check for NaN values
                    if tf.math.reduce_any(tf.math.is_nan(recon_ts)) or tf.math.reduce_any(tf.math.is_nan(recon_mask)):
                        print("WARNING: NaN values detected in reconstruction")
                        continue

                    # Reconstruction losses
                    loss_ts = mse_loss_fn(raw_in, recon_ts)
                    loss_mask_mse = mse_loss_fn(mask_in, recon_mask)
                    loss_mask_dice = dice_loss(recon_mask, mask_in)
                    
                    if use_mog:
                        mus_mixture, logvars_mixture, pis_mixture = get_mog_params()
                        mog_kl = kl_q_p_mog(mu, logvar, mus_mixture, logvars_mixture, pis_mixture)
                        kl_loss = mog_kl
                    else:
                        kl_loss = -0.5 * tf.reduce_mean(
                            1.0 + logvar - tf.square(mu) - tf.exp(logvar)
                        )
                    
                    loss_batch = loss_ts + (loss_mask_mse + loss_mask_dice) + kl_loss

                # Gradient step
                grads = tape.gradient(loss_batch, all_trainables)
                optimizer.apply_gradients(zip(grads, all_trainables))
                
                # Update metrics
                total_loss += loss_batch.numpy()
                total_recon_ts += loss_ts.numpy()
                total_recon_mask += (loss_mask_mse + loss_mask_dice).numpy()
                num_batches += 1
                
            except Exception as e:
                print(f"\nError processing batch: {e}")
                continue

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
        
    # # Process data in smaller chunks
    # raw_segments, fft_segments, rms_segments, psd_segments, mask_segments, test_ids = \
    #     segment_and_transform(accel_dict, mask_dict)
    
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
        batch_size=8  # Smaller batch size
    )
    
    # Build & train model
    latent_dim = 128
    model = VAE(latent_dim)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    
    print_detailed_memory()
    model = train_vae(model, dataset, optimizer, num_epochs=10, use_mog=True)

    # 5) Save
    model.save_weights("results/vae_mmodal_weights.h5")

    # 6) Generate synthetic data
    results_dir = "results/vae_results"
    os.makedirs(results_dir, exist_ok=True)
    
    num_synthetic = 10
    for i in range(num_synthetic):
        z_rand = tf.random.normal(shape=(1, latent_dim))
        gen_ts, gen_mask = model.generate(z_rand)
        np.save(os.path.join(results_dir, f"gen_ts_{i}.npy"), gen_ts.numpy())
        np.save(os.path.join(results_dir, f"gen_mask_{i}.npy"), gen_mask.numpy())
        logging.info(f"Saved synthetic sample {i}")

if __name__ == "__main__":
    main()
