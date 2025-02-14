import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import logging

# Append parent directory to find data_loader.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your data loader
import data_loader

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
class TimeSeriesEncoder(Model):
    """
    Encoder branch for accelerometer data.
    Expects input of shape [batch, 12000, 12].
    """
    def __init__(self):
        super(TimeSeriesEncoder, self).__init__()
        self.conv1 = layers.Conv1D(32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(64, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv1D(128, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv4 = layers.Conv1D(256, kernel_size=4, strides=2, padding='same', activation='relu')
        self.flatten = layers.Flatten()
        
    def call(self, x):
        x = self.conv1(x)   # -> [batch, 6000, 32]
        x = self.conv2(x)   # -> [batch, 3000, 64]
        x = self.conv3(x)   # -> [batch, 1500, 128]
        x = self.conv4(x)   # -> [batch, 750, 256]
        x = self.flatten(x) # -> [batch, 750*256]
        return x

class MaskEncoder(Model):
    """
    Encoder branch for damage masks.
    Expects input of shape [batch, 256, 768].
    """
    def __init__(self):
        super(MaskEncoder, self).__init__()
        self.expand = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))
        self.conv1 = layers.Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(256, kernel_size=4, strides=2, padding='same', activation='relu')
        self.flatten = layers.Flatten()
        
    def call(self, x):
        x = self.expand(x)   # -> [batch, 256, 768, 1]
        x = self.conv1(x)    # -> [batch, 128, 384, 32]
        x = self.conv2(x)    # -> [batch, 64, 192, 64]
        x = self.conv3(x)    # -> [batch, 32, 96, 128]
        x = self.conv4(x)    # -> [batch, 16, 48, 256]
        x = self.flatten(x)
        return x

# ----- Combined Encoder with Dual Branches -----
class DualEncoder(Model):
    """
    Combines the time-series and mask encoders.
    Computes latent parameters from the concatenated features.
    """
    def __init__(self, latent_dim):
        super(DualEncoder, self).__init__()
        self.ts_encoder = TimeSeriesEncoder()
        self.mask_encoder = MaskEncoder()
        self.fc_mu = layers.Dense(latent_dim)
        self.fc_logvar = layers.Dense(latent_dim)
        
    def call(self, ts_input, mask_input):
        ts_features = self.ts_encoder(ts_input)       # shape: (batch, D1)
        mask_features = self.mask_encoder(mask_input)   # shape: (batch, D2)
        combined = layers.concatenate([ts_features, mask_features])
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar, ts_features, mask_features

# ----- Decoders with Cross-Attention -----
class TimeSeriesDecoder(Model):
    """
    Decoder for accelerometer time-series reconstruction.
    Reconstructs output of shape [batch, 12000, 12].
    Includes a cross-attention layer to integrate mask features.
    """
    def __init__(self):
        super(TimeSeriesDecoder, self).__init__()
        self.fc = layers.Dense(192000, activation='relu')
        self.reshape_layer = layers.Reshape((750, 256))
        self.attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=256)
        self.mask_proj_layer = layers.Dense(256)
        self.upsample1 = layers.UpSampling1D(size=2)
        self.conv1 = layers.Conv1D(128, kernel_size=4, padding='same', activation='relu')
        self.upsample2 = layers.UpSampling1D(size=2)
        self.conv2 = layers.Conv1D(64, kernel_size=4, padding='same', activation='relu')
        self.upsample3 = layers.UpSampling1D(size=2)
        self.conv3 = layers.Conv1D(32, kernel_size=4, padding='same', activation='relu')
        self.upsample4 = layers.UpSampling1D(size=2)
        self.conv4 = layers.Conv1D(12, kernel_size=4, padding='same')
        
    def call(self, z, mask_features=None):
        x = self.fc(z)
        x = self.reshape_layer(x)  # (batch, 750, 256)
        if mask_features is not None:
            mask_proj = self.mask_proj_layer(mask_features)  # (batch, 256)
            mask_proj = tf.expand_dims(mask_proj, axis=1)      # (batch, 1, 256)
            mask_proj = tf.tile(mask_proj, [1, tf.shape(x)[1], 1])
            attn_output = self.attention_layer(query=x, value=mask_proj, key=mask_proj)
            x = x + attn_output
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.upsample3(x)
        x = self.conv3(x)
        x = self.upsample4(x)
        x = self.conv4(x)
        return x

class MaskDecoder(Model):
    """
    Decoder for damage mask reconstruction.
    Reconstructs output of shape [batch, 256, 768].
    Includes a cross-attention layer to integrate time-series features.
    """
    def __init__(self):
        super(MaskDecoder, self).__init__()
        self.fc = layers.Dense(256 * 48 * 16, activation='relu')
        self.reshape_layer = layers.Reshape((48, 16, 256))
        self.attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=256)
        self.ts_proj_layer = layers.Dense(256)
        self.deconv1 = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu')
        self.deconv2 = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')
        self.deconv3 = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.deconv4 = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='sigmoid')
        
    def call(self, z, ts_features=None):
        x = self.fc(z)
        x = self.reshape_layer(x)  # (batch, 48, 16, 256)
        b = tf.shape(x)[0]
        seq_len = 48 * 16
        x_seq = tf.reshape(x, (b, seq_len, 256))
        if ts_features is not None:
            ts_proj = self.ts_proj_layer(ts_features)  # (batch, 256)
            ts_proj = tf.expand_dims(ts_proj, axis=1)
            ts_proj = tf.tile(ts_proj, [1, seq_len, 1])
            attn_output = self.attention_layer(query=x_seq, value=ts_proj, key=ts_proj)
            x_seq = x_seq + attn_output
        x = tf.reshape(x_seq, (b, 48, 16, 256))
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = tf.squeeze(x, axis=-1)
        x = tf.transpose(x, perm=[0, 2, 1])
        return x

# ----- VAE with Self-Attention in the Bottleneck -----
class VAE(Model):
    """
    Full VAE with dual encoders and dual decoders.
    Applies self-attention in the latent bottleneck and uses cross-attention in decoders.
    """
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = DualEncoder(latent_dim)
        self.decoder_ts = TimeSeriesDecoder()
        self.decoder_mask = MaskDecoder()
        self.token_count = 8
        self.token_dim = latent_dim // self.token_count  # e.g., 16 if latent_dim==128
        self.self_attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=self.token_dim)
        
    def call(self, ts_input, mask_input, training=False):
        mu, logvar, ts_features, mask_features = self.encoder(ts_input, mask_input)
        z = reparameterize(mu, logvar)
        z_tokens = tf.reshape(z, (-1, self.token_count, self.token_dim))
        attn_output = self.self_attention_layer(query=z_tokens, key=z_tokens, value=z_tokens)
        z_refined = tf.reshape(attn_output, (-1, tf.shape(z)[-1]))
        recon_ts = self.decoder_ts(z_refined, mask_features=mask_features)
        recon_mask = self.decoder_mask(z_refined, ts_features=ts_features)
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

# ----- Training Procedure -----
def train_vae(model, dataset, optimizer, num_epochs=10):
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_recon_ts = 0.0
        total_recon_mask = 0.0
        num_batches = 0
        for ts_data, mask_data in dataset:
            with tf.GradientTape() as tape:
                recon_ts, recon_mask, mu, logvar = model(ts_data, mask_data, training=True)
                loss_ts = mse_loss_fn(ts_data, recon_ts)
                loss_mask_mse = mse_loss_fn(mask_data, recon_mask)
                loss_mask_dice = dice_loss(recon_mask, mask_data)
                kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
                loss = loss_ts + (loss_mask_mse + loss_mask_dice) + kl_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()
            total_recon_ts += loss_ts.numpy()
            total_recon_mask += (loss_mask_mse + loss_mask_dice).numpy()
            num_batches += 1
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss/num_batches:.4f}, "
                     f"Reconstruction TS Loss: {total_recon_ts/num_batches:.4f}, "
                     f"Reconstruction Mask Loss: {total_recon_mask/num_batches:.4f}")
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
    
    Args:
        weights_path: Path to the saved weights file.
    """
    global vae_model
    latent_dim = 128  # Ensure this matches the latent_dim used in training.
    model = VAE(latent_dim)
    # Load the weights from the file
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
    logging.basicConfig(level=logging.INFO)
    
    # Load real data using data_loader.load_data() (ensure this function exists in data_loader.py)
    accel_dict, mask_dict = data_loader.load_data()
    
    # Consolidate data: pair each accelerometer sample with its corresponding mask.
    acc_samples = []
    mask_samples = []
    for test_id in accel_dict:
        samples = accel_dict[test_id]
        mask = mask_dict[test_id]
        for s in samples:
            acc_samples.append(s)      # Each s is (12000, 12)
            mask_samples.append(mask)  # Same mask for all samples from a test
    acc_samples = np.stack(acc_samples)    # Shape: [num_samples, 12000, 12]
    mask_samples = np.stack(mask_samples)  # Shape: [num_samples, 256, 768]
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (acc_samples.astype(np.float32), mask_samples.astype(np.float32))
    ).shuffle(buffer_size=1024).batch(8)
    
    latent_dim = 128  # Hyperparameter
    model = VAE(latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    num_epochs = 10  # Adjust as needed
    model = train_vae(model, dataset, optimizer, num_epochs=num_epochs)
    
    # Set the global variable so that the encoder() and decoder() functions can use the trained model.
    global vae_model
    vae_model = model

    save_trained_model("results/vae_weights.h5")

    
    # ----- Generating New Synthetic Data -----
    results_dir = os.path.join("results", "vae_results")
    os.makedirs(results_dir, exist_ok=True)
    
    num_synthetic_samples = 100  # Generate 100 new samples from the latent space
    for i in range(num_synthetic_samples):
        # Sample a new latent vector from a standard Gaussian
        random_latent = tf.random.normal(shape=(1, latent_dim))
        recon_ts, recon_mask = model.generate(random_latent)
        
        ts_filename = os.path.join(results_dir, f"synthetic_sample_{i}_acc.npy")
        mask_filename = os.path.join(results_dir, f"synthetic_sample_{i}_mask.npy")
        
        np.save(ts_filename, recon_ts.numpy().squeeze(0))
        np.save(mask_filename, recon_mask.numpy().squeeze(0))
        logging.info(f"Saved synthetic sample {i}: {ts_filename} and {mask_filename}")

if __name__ == "__main__":
    main()
