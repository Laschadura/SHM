import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras import regularizers # type: ignore

from bridge_data.custom_distributions import compute_mixture_prior, compute_js_divergence, reparameterize




# ------ Custom ISTFT Layer ------
# trainable TensorFlow inverse stft for MMVAE
@tf.keras.saving.register_keras_serializable('mmvae')
class TFInverseISTFT(tf.keras.layers.Layer):
    def __init__(self, frame_length=256, frame_step=32, **kwargs):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step   = frame_step
        self.window_fn = tf.signal.inverse_stft_window_fn(
            frame_step,
            forward_window_fn=tf.signal.hann_window
        )
       
        self.beta = None                            # placeholder

    def build(self, input_shape):
        if self.beta is None:                       # create only once
            self.beta = self.add_weight(
                name="beta",
                shape=(),
                initializer=tf.keras.initializers.Ones(),
                trainable=True,
            )
    # ----------------------------------------------------------------

    def call(self, spec_logits, length):
        B = tf.shape(spec_logits)[0]
        D = tf.shape(spec_logits)[3]
        C = D // 2

        # Clip to prevent extreme values before softplus
        log_mag = tf.clip_by_value(spec_logits[..., 0::2], -8.0, 5.0)
        phase   = spec_logits[..., 1::2]

        # Learnable softplus for magnitude
        mag = tf.nn.softplus(self.beta * log_mag)

        mag   = tf.reshape(mag,   [B*C, tf.shape(mag)[1], tf.shape(mag)[2]])
        phase = tf.reshape(phase, [B*C, tf.shape(phase)[1], tf.shape(phase)[2]])

        real = mag * tf.cos(phase)
        imag = mag * tf.sin(phase)
        real = tf.cast(real, tf.float32)
        imag = tf.cast(imag, tf.float32)

        complex_spec = tf.complex(real, imag)
        complex_spec = tf.transpose(complex_spec, [0, 2, 1])  # [B*C, T, F]

        wav = tf.signal.inverse_stft(
            complex_spec,
            self.frame_length,
            self.frame_step,
            window_fn=self.window_fn
        )

        # pad or crop
        wav = wav[:, :length]
        wav = tf.pad(wav, [[0, 0], [0, tf.maximum(0, length - tf.shape(wav)[1])]])

        # [B*C, L] → [B, L, C]
        wav = tf.reshape(wav, [B, C, length])
        wav = tf.transpose(wav, [0, 2, 1])
        return wav

    def get_config(self):
        config = super().get_config()
        config.update({
            "frame_length": self.frame_length,
            "frame_step": self.frame_step
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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
  