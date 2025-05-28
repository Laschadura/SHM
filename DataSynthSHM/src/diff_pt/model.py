import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from tqdm import tqdm

from .losses import (
    complex_spectrogram_loss,
    magnitude_l1_loss,
    laplacian_loss_phase_only,
    gradient_loss_phase_only,
    waveform_si_l1_loss,
    focal_tversky_loss,
    damage_amount_loss,
    custom_mask_loss,
)

from .utils import (
    energy_sigma,
    sinusoidal_time_embedding,
    betas_for_alpha_bar,
    dynamic_weighting,
)

# ----- Helpers -----
def _upsample_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.LeakyReLU(0.1)
    )

# ----- Custom Layers -----
class DifferentiableISTFT(nn.Module):
    def __init__(self, nperseg=256, noverlap=224, epsilon=1e-5):
        super().__init__()
        self.nperseg = nperseg
        self.hop_len = nperseg - noverlap  # → 32
        self.epsilon = epsilon
        self.register_buffer(
            'window',
            torch.hann_window(nperseg),
            persistent=False
        )

    def forward(self, spec, length):
        # Accepts (B, 2C, F, T)
        spec = spec.permute(0, 2, 3, 1).contiguous()  # → (B, F, T, 2C)

        B, F, T, D = spec.shape
        C = D // 2

        # Separate log-magnitude and phase
        spec = spec.view(B, F, T, C, 2)
        log_mag = spec[..., 0]        # (B, F, T, C)
        phase   = spec[..., 1]

        # Flatten for batch ISTFT: (B*C, F, T)
        log_mag = log_mag.permute(0, 3, 1, 2).reshape(-1, F, T)
        phase   = phase.permute(0, 3, 1, 2).reshape(-1, F, T)

        # Reconstruct magnitude
        mag = torch.exp(log_mag) - self.epsilon
        mag = torch.clamp(mag, min=0.)

        # Form complex spectrogram
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        complex_spec = torch.complex(real, imag)  # shape: (B*C, F, T)

        # Apply ISTFT
        wav = torch.istft(
            complex_spec,
            n_fft=self.nperseg,
            hop_length=self.hop_len,
            win_length=self.nperseg,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
            length=length
        )

        # Reshape back to (B, length, C)
        wav = wav.view(B, C, -1).permute(0, 2, 1)
        return wav

# ----- Encoders -----
class SpectrogramEncoder(nn.Module):
    """
    Adaptive encoder for spectrogram features that works with any input dimensions.
    """
    def __init__(self, latent_dim, channels):
        super().__init__()
        
        # Initial adaptive dimension adjustment: e.g. channels*2 if you're storing mag+phase
        self.adjust_conv = nn.Conv2d(
            in_channels=channels*2,
            out_channels=32,
            kernel_size=1,
            padding='same'
        )
        self.adjust_relu = nn.ReLU()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.gn1 = nn.GroupNorm(8, 32)
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.gn2 = nn.GroupNorm(16, 64)
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.gn3 = nn.GroupNorm(32, 128)
        self.relu3 = nn.LeakyReLU(0.1)
        self.dropout3 = nn.Dropout(0.3)
        
        # Global pooling and dense layers
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.dense_reduce = nn.Linear(128 * 2 * 2, 512)
        self.relu4 = nn.LeakyReLU(0.1)

        # Latent layer (for standard autoencoder)
        self.latent_layer = nn.Linear(512, latent_dim)
    
    def forward(self, x):
        x = self.adjust_relu(self.adjust_conv(x))

        x1 = self.relu1(self.gn1(self.conv1(x)))
        x1 = self.dropout1(x1)

        x2 = self.relu2(self.gn2(self.conv2(x1)))
        x2 = self.dropout2(x2)

        x3 = self.relu3(self.gn3(self.conv3(x2)))
        x3 = self.dropout3(x3)

        x_pool = self.global_pool(x3)        # (B,128,2,2)
        x_flat = x_pool.view(x_pool.size(0), -1)
        x_fc   = self.relu4(self.dense_reduce(x_flat))
        latent = self.latent_layer(x_fc)
        return latent

class MaskEncoder(nn.Module):
    """
    Adaptive encoder for mask features that works with any input dimensions.
    """
    def __init__(self, latent_dim):
        super().__init__()
        
        # Initial adaptive dimension adjustment
        self.adjust_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, padding='same')
        self.adjust_relu = nn.LeakyReLU(0.1)
        
        # Convolutional layers for downsampling
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.LeakyReLU(0.1)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU(0.1)
        
        # Global pooling to make it adaptive to any input size
        self.global_pool = nn.AdaptiveMaxPool2d(1) 

        
        # Dense layers
        self.dense1 = nn.Linear(64, 128)
        self.relu4 = nn.LeakyReLU(0.1)
        
        # Direct latent representation
        self.latent_layer = nn.Linear(128, latent_dim)

        nn.init.kaiming_normal_(self.adjust_conv.weight, nonlinearity='leaky_relu')

    
    def forward(self, x):
        # x shape: (batch, channels, height, width), e.g., (batch, 1, height, width)
        # Initial dimension adjustment
        x = self.adjust_relu(self.adjust_conv(x))
        
        # Downsampling path
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        
        # Global pooling (makes the network adaptive to any input size)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Dense layers to get latent representation
        x = self.relu4(self.dense1(x))
        latent = self.latent_layer(x)
        
        return latent

# ----- Decoders -----
class SpectrogramDecoder(nn.Module):
    """
    Decoder for spectrogram features ensuring exact output dimensions.
    """
    def __init__(self, latent_dim, freq_bins=129, time_bins=24, channels=24):
        super().__init__()
        
        self.freq_bins = freq_bins
        self.time_bins = time_bins
        self.channels = channels
        
        # Determine minimum dimensions (ensure they're at least 1)
        self.min_freq = max(1, freq_bins // 4)
        self.min_time = max(1, time_bins // 4)
        
        # Dense and reshape layers
        self.fc = nn.Linear(latent_dim, self.min_freq * self.min_time * 128)
        self.relu1 = nn.LeakyReLU(0.1)
        
        # Upsampling blocks
        self.conv_t1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1  # Required for odd dimensions in PyTorch
        )
        self.gn1 = nn.GroupNorm(16, 64)
        self.relu2 = nn.LeakyReLU(0.1)
        self.drop1 = nn.Dropout(0.3)
        
        self.conv_t2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1  # Required for odd dimensions in PyTorch
        )
        self.gn2 = nn.GroupNorm(8, 32)
        self.relu3 = nn.LeakyReLU(0.1)
        self.drop2 = nn.Dropout(0.3)

        # Final refinement layers
        self.conv_t3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(8, 32)
        self.relu4 = nn.LeakyReLU(0.1)
        self.drop3 = nn.Dropout(0.3)
        
        # Output projection 
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=channels, kernel_size=3, padding=1)

        self.output_gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, z):
        # Initial projection and reshape
        x = self.relu1(self.fc(z))
        x = x.view(-1, 128, self.min_freq, self.min_time)
        
        # First upsampling
        x  = self.conv_t1(x)
        x = self.relu2(self.gn1(x))
        x = self.drop1(x)
        
        # Second upsampling
        x = self.conv_t2(x)
        x = self.relu3(self.gn2(x))
        x = self.drop2(x)
        
        # Final refinement
        x = self.conv_t3(x)
        x  = self.relu4(self.gn3(x))
        x  = self.drop3(x)
        
        # Final projection
        x = self.output_gain * self.conv_out(x)
        
        # Ensure the output has the exact desired dimensions
        if (x.size(2) != self.freq_bins) or (x.size(3) != self.time_bins):
            x = F.interpolate(
                x, 
                size=(self.freq_bins, self.time_bins),
                mode='bilinear',
                align_corners=False
            )
        
        return x

class MaskDecoder(nn.Module):
    """
    Decoder for mask features ensuring exact output dimensions.
    """
    def __init__(self, latent_dim, output_shape=(32, 96)):
        super().__init__()
        H_out, W_out = output_shape
        H_min = max(1, H_out // 8)
        W_min = max(1, W_out // 8)

        # dense → (128, H_min, W_min)
        self.fc = nn.Linear(latent_dim, H_min * W_min * 128)
        self.act = nn.LeakyReLU(0.1)

        # three 2× upsample stages
        self.up1 = _upsample_block(128, 64)
        self.up2 = _upsample_block(64,  32)
        self.up3 = _upsample_block(32,  16)

        # final 3×3 conv + sigmoid
        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.sigmoid  = nn.Sigmoid()

        nn.init.xavier_uniform_(self.out_conv.weight, gain=0.01)
        nn.init.constant_(self.out_conv.bias, 0.0)

        self.H_out, self.W_out = H_out, W_out
    
    def forward(self, z):
        x = self.act(self.fc(z))
        x = x.view(-1, 128, self.H_out // 8, self.W_out // 8)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        # last, precise resize (handles odd sizes)
        x = F.interpolate(x, size=(self.H_out, self.W_out),
                          mode="bilinear", align_corners=False)

        x = self.sigmoid(self.out_conv(x))
        return x

# ----- Autoencoders -----
class SpectrogramAutoencoder(nn.Module):
    def __init__(self, latent_dim, channels, freq_bins, time_bins):
        super().__init__()
        self.encoder = SpectrogramEncoder(latent_dim, channels)
        self.decoder = SpectrogramDecoder(latent_dim, freq_bins, time_bins, channels*2)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
class MaskAutoencoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.encoder = MaskEncoder(latent_dim)
        self.decoder = MaskDecoder(latent_dim, output_shape)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# ===== Diffusion Model Core Components =====
class DiffusionModel(nn.Module):
    """
    Score-based diffusion model for the joint latent space of multiple modalities.
    Implements both unconditional and conditional generation capabilities.
    """
    def __init__(self, latent_dim, hidden_dims=[512, 1024, 512], modality_dims=None):
        """
        Initialize the diffusion model.
        
        Args:
            latent_dim: Dimension of the complete latent space (sum of all modality latent dimensions)
            hidden_dims: Hidden dimensions for the score network
            modality_dims: List of individual modality latent dimensions (if None, assumes single joint latent space)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.modality_dims = modality_dims
        
        # Calculate cumulative dimensions for modality indexing
        if modality_dims is not None:
            self.cumulative_dims = [0]
            for dim in modality_dims:
                self.cumulative_dims.append(self.cumulative_dims[-1] + dim)
            assert self.cumulative_dims[-1] == latent_dim, "Sum of modality dims must equal latent_dim"
        
        self.embed_dim = 256
        
        # Score prediction network (U-Net like structure)
        self.input_layer = nn.Linear(latent_dim, hidden_dims[0])

        print("Input layer weight shape:", self.input_layer.weight.shape)
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i] + 256, hidden_dims[i]),  # Add time embedding
                    nn.SiLU(),
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.SiLU()
                )
            )
        
        # Middle block
        self.middle_block = nn.Sequential(
            nn.Linear(hidden_dims[-1] + 256, hidden_dims[-1]),  # Add time embedding
            nn.SiLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.SiLU()
        )
        
        # Up blocks with skip connections
        self.up_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            dim = hidden_dims[i]
            self.up_blocks.append(
                nn.Sequential(
                    nn.Linear(dim * 2 + 256, hidden_dims[i - 1]),
                    nn.SiLU(),
                    nn.Linear(hidden_dims[i - 1], hidden_dims[i - 1]),
                    nn.SiLU()
                )
            )

        # Output layer for predicting noise
        self.output_layer = nn.Linear(hidden_dims[0], latent_dim)
        
    def forward(self, x, t, mask=None):
        # cast to float
        x = x.float()
        t = t.float()
        if mask is not None:
            mask = mask.float()

        # Embed diffusion time
        t_emb = sinusoidal_time_embedding(t, self.embed_dim)

        # Initial layer
        h = self.input_layer(x)

        # Store intermediate activations for skip connections
        skips = [h]

        # Down path
        for block in self.down_blocks:
            h = torch.cat([h, t_emb], dim=1)
            h = block(h)
            skips.append(h)

        # Middle
        h = torch.cat([h, t_emb], dim=1)
        h = self.middle_block(h)

        # Up path with skip connections

        for i, block in enumerate(self.up_blocks):
            skip = skips.pop()
            h_concat = torch.cat([h, skip, t_emb], dim=1)
            h = block(h_concat)

        # Output projection
        score = self.output_layer(h)

        # For conditional generation, zero out the score for known elements
        if mask is not None:
            score = score * (1.0 - mask)

        return score

    def get_modality_slice(self, modality_idx):
        """Get the slice indices for a specific modality"""
        if self.modality_dims is None:
            raise ValueError("Modality dimensions not defined")
        
        start_idx = self.cumulative_dims[modality_idx]
        end_idx = self.cumulative_dims[modality_idx + 1]
        return slice(start_idx, end_idx)

# ===== Noise Schedulers =====
class NoiseScheduler:
    """
    Handles diffusion forward and reverse processes.
    Implements the variance preserving stochastic differential equation (VP-SDE).
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        """
        Initialize the noise scheduler.
        
        Args:
            num_timesteps: Total number of diffusion steps
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Cosine beta schedule
        self.betas = betas_for_alpha_bar(num_timesteps)

        # Pre-compute diffusion parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
   
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Sample from q(x_t | x_0) - the forward diffusion process
        
        Args:
            x_0: Initial clean samples
            t: Time steps to diffuse to
            noise: Optional pre-generated noise to use
        
        Returns:
            Diffused samples at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Extract the corresponding alpha and sigma values for timestep t
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        # Diffuse the data
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and the predicted noise.
        
        Args:
            x_t: Diffused samples at timestep t
            t: Current timesteps
            noise: Predicted noise
        
        Returns:
            Predicted clean samples x_0
        """
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Extract for the specific timesteps
        sqrt_recip_alphas_cumprod_t = sqrt_recip_alphas_cumprod[t].reshape(-1, 1)
        sqrt_recipm1_alphas_cumprod_t = sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1)
        
        # Predict x_0
        pred_x0 = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
        
        return pred_x0
    
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        
        Args:
            x_0: Predicted clean samples
            x_t: Current diffused samples
            t: Current timesteps
        
        Returns:
            Posterior mean and variance
        """
        posterior_mean = (
            self.posterior_mean_coef1[t].reshape(-1, 1) * x_0 +
            self.posterior_mean_coef2[t].reshape(-1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].reshape(-1, 1)
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(-1, 1)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(self, model, x_t, t, mask=None):
        """
        Compute the mean and variance of the diffusion posterior p(x_{t-1} | x_t)
        using the model's predicted noise.
        
        Args:
            model: Diffusion model that predicts noise
            x_t: Current diffused samples
            t: Current timesteps
            mask: Optional mask for conditional generation
        
        Returns:
            Model posterior mean and variance
        """
        # Predict noise using the model
        pred_noise = model(x_t, t.reshape(-1, 1).float(), mask)
        
        # Predict the clean sample x_0
        pred_x0 = self.predict_start_from_noise(x_t, t, pred_noise)
        
        # Clip x_0 for stability (optional)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Get the parameters for q(x_{t-1} | x_t, x_0)
        model_mean, model_variance, model_log_variance = self.q_posterior_mean_variance(pred_x0, x_t, t)
        
        return model_mean, model_variance, model_log_variance
    
    def p_sample(self, model, x_t, t, mask=None):
        """
        Sample from p(x_{t-1} | x_t) - the reverse diffusion process
        
        Args:
            model: Diffusion model that predicts noise
            x_t: Current diffused samples
            t: Current timesteps
            mask: Optional mask for conditional generation
        
        Returns:
            Previous diffusion state x_{t-1}
        """
        # Get model's predicted mean and variance
        model_mean, model_variance, model_log_variance = self.p_mean_variance(model, x_t, t, mask)
        
        # No noise when t == 0
        noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        
        # Sample from the posterior
        x_t_prev = model_mean + torch.exp(0.5 * model_log_variance) * noise
        
        # For conditional generation, replace known parts with original values
        if mask is not None:
            x_t_prev = x_t_prev * (1.0 - mask) + x_t * mask
        
        return x_t_prev
    
    def p_sample_loop(self, model, shape, device, mask=None, x_T=None):
        """
        Generate samples from the model by iteratively sampling from p(x_{t-1} | x_t).
        
        Args:
            model: Diffusion model that predicts noise
            shape: Shape of the samples to generate
            device: Device to generate on
            mask: Optional mask for conditional generation
            x_T: Optional starting point for the reverse process
            
        Returns:
            Generated samples
        """
        # Start from pure noise (unless x_T is given)
        if x_T is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = x_T
        
        # Iterative sampling from timestep T to 0
        for time_step in tqdm(range(self.num_timesteps - 1, -1, -1), desc="Sampling"):
            t = torch.full((shape[0],), time_step, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, mask)
        
        return x_t
    
    def get_loss(self, model, x_0, t, mask=None, noise=None):
        """
        Calculate the diffusion loss (noise prediction MSE).
        
        Args:
            model: Diffusion model that predicts noise
            x_0: Clean samples
            t: Timesteps to calculate loss at
            mask: Optional mask for conditional generation
            noise: Optional pre-generated noise
        
        Returns:
            Noise prediction loss
        """
        # Generate diffused samples and the added noise
        x_t, noise = self.q_sample(x_0, t, noise)
        
        # Predict noise using the model
        pred_noise = model(x_t, t.reshape(-1, 1).float(), mask)
        
        # Calculate loss
        loss = F.mse_loss(noise, pred_noise)
        
        return loss, x_t, pred_noise

# ===== Multi-Modal Latent Diffusion Implementation =====
class MultiModalLatentDiffusion:
    """
    Implementation of the Multi-Modal Latent Diffusion (MLD) approach.
    
    Consists of:
    1. Deterministic autoencoders for each modality
    2. A diffusion model in the joint latent space
    """
    def __init__(
        self, 
        spec_autoencoder,
        mask_autoencoder,
        latent_dim=128,
        modality_names=["spec", "mask"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        mu_spec=None,
        sig_spec=None
        ):
        """
        Initialize the MLD system.
        
        Args:
            spec_autoencoder: Pretrained spectrogram autoencoder
            mask_autoencoder: Pretrained mask autoencoder
            latent_dim: Latent dimension for each modality
            modality_names: Names of the modalities
            device: Device to use for computation
        """
        self.device = device
        self.modality_names = modality_names
        self.num_modalities = len(modality_names)

        self.mu_spec  = mu_spec.to(device) if mu_spec is not None else None
        self.sig_spec = sig_spec.to(device) if sig_spec is not None else None
        
        # Store the pretrained autoencoders
        self.autoencoders = {
            "spec": spec_autoencoder.to(device),
            "mask": mask_autoencoder.to(device)
        }
        
        # Set autoencoders to evaluation mode (we won't train them further)
        for ae in self.autoencoders.values():
            ae.eval()
        
        # Initialize modality dimensions
        self.modality_dims = [latent_dim] * self.num_modalities
        self.total_latent_dim = sum(self.modality_dims)
        
        # Initialize the diffusion model and noise scheduler
        self.diffusion_model = DiffusionModel(
            latent_dim=self.total_latent_dim,
            hidden_dims=[512, 1024, 512],
            modality_dims=self.modality_dims
        ).to(device)
        
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=1000,
            beta_start=1e-4,
            beta_end=0.02
        )
    
    def eval(self):
        self.autoencoders["spec"].eval()
        self.autoencoders["mask"].eval()
        self.diffusion_model.eval()
        return self

    def train(self, mode=True):
        self.autoencoders["spec"].train(mode)
        self.autoencoders["mask"].train(mode)
        self.diffusion_model.train(mode)
        return self

    def train_diffusion_model(
        self,
        train_dataloader,
        val_dataloader=None,
        num_epochs=100,
        learning_rate=1e-4,
        save_dir="./results/diffusion"
        ):
        """
        Train the diffusion model on the joint latent space.
        This is usefiul if autoencoders are pretrained and we want to train the diffusion model only.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            save_dir: Directory to save checkpoints and logs
        
        Returns:
            Training history
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize optimizer
        optimizer = optim.AdamW(self.diffusion_model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [] if val_dataloader else None
        }
        
        best_val_loss = float("inf")
        
        # Training loop
        for epoch in range(num_epochs):
            # Dynamic loss weights
            damage_weight = 0.5 if epoch >= 200 else 2.0
            focal_gamma = 1.5 if epoch >= 250 else 1.0
            contrast_weight = min(0.3, epoch / 100 * 0.3)

            # Training
            self.diffusion_model.train()
            train_losses = []
            
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                # Get data for each modality
                modality_data = {
                    name: batch[i] if i < len(batch) else None 
                    for i, name in enumerate(self.modality_names)
                }

                # 🔍 Debug: Print shape before autoencoder is called
                if epoch == 0 and len(train_losses) == 0:  # Only for first batch of first epoch
                    print("🔍 Debug Info (first batch only):")
                    for modality_name, data in modality_data.items():
                        if data is not None:
                            print(f"📦 Input modality '{modality_name}' shape: {data.shape}")
                            ae = self.autoencoders[modality_name]
                            print(f"🔧 Encoder conv weight shape: {ae.encoder.adjust_conv.weight.shape}")
                            print(f"🔧 Expected in_channels: {ae.encoder.adjust_conv.in_channels}")
                            try:
                                with torch.no_grad():
                                    _, z = ae(data.to(self.device))
                                    print(f"✅ Forward pass succeeded, latent shape: {z.shape}")
                            except Exception as e:
                                print(f"❌ Forward pass FAILED for modality '{modality_name}': {e}")
                                raise e

                
                # Encode each modality to get latent representations
                latents, joint_latent = self.encode_modalities(modality_data)
                
                if joint_latent is None:
                    continue

                # Compute σ from spec magnitude
                spec = modality_data["spec"].to(self.device)
                # invert global normalization
                spec_dn = spec * self.sig_spec + self.mu_spec
                real, imag = spec_dn[:, 0::2], spec_dn[:, 1::2]
                mag = torch.sqrt(real**2 + imag**2)

                sigma_t = energy_sigma(mag)                          # (B, T)
                sigma_mean = sigma_t.mean(dim=1, keepdim=True)       # (B, 1)

                # Sample z0 from adaptive prior
                z0 = torch.randn_like(joint_latent) * sigma_mean     # (B, D)


                # ---------------- flow-matching targets -------------------------
                batch_size = joint_latent.size(0)
                t_int  = torch.randint(0, self.noise_scheduler.num_timesteps,
                                    (batch_size,), device=self.device)
                tau    = t_int.view(-1,1).float() / (self.noise_scheduler.num_timesteps-1)

                x_t    = tau * joint_latent + (1. - tau) * z0
                v_tgt  = joint_latent - z0
                v_pred = self.diffusion_model(x_t, tau)

                # --- FLOW-MATCH loss ---
                loss_fm = F.mse_loss(v_pred, v_tgt)


                # ---------------- spectrogram auxiliaries ----------------------
                # Decode
                z_recon = v_pred + z0
                outputs = self.decode_modalities(z_recon)

                spec_pred = outputs["spec"]
                spec_true = modality_data["spec"].to(self.device)

                loss_complex   = complex_spectrogram_loss(spec_true, spec_pred)
                loss_phase_g   = gradient_loss_phase_only(spec_true, spec_pred)
                loss_phase_l   = laplacian_loss_phase_only(spec_true, spec_pred)
                loss_mag_l1    = magnitude_l1_loss(spec_true, spec_pred)

                # robust .istft access
                core = getattr(self, "module", self)
                istft_layer = getattr(core, "istft", None)

                if istft_layer is not None:
                    spec_pred_dn = spec_pred.clone()
                    C = spec_pred_dn.shape[1] // 2
                    spec_pred_dn[:, :C] = spec_pred_dn[:, :C] * self.sig_spec + self.mu_spec
                    # spec_pred_dn[:, C:] stays untouched (raw phase)
                    wav_pred = istft_layer(spec_pred_dn, length=spec_true.size(2) * 4)
                    wav_true = istft_layer(spec_true, length=spec_true.size(2) * 4)
                    loss_wave_si = waveform_si_l1_loss(wav_true, wav_pred)
                else:
                    loss_wave_si = torch.tensor(0.0, device=spec_true.device)


                mask_pred  = outputs["mask"]
                mask_true  = modality_data["mask"].to(self.device)
                loss_mask_px = focal_tversky_loss(mask_true, mask_pred, alpha=0.3, beta=0.8, gamma=focal_gamma)
                loss_damage  = damage_amount_loss(
                        mask_true,
                        mask_pred,
                        contrast_weight=contrast_weight,
                        margin=0.005
                        )

                # Combined loss
                loss = (
                    1               * loss_fm + 
                    0.5             * loss_complex +
                    0               * loss_phase_g +
                    0               * loss_phase_l +
                    0               * loss_mag_l1 +
                    0.5               * loss_wave_si +
                    1               * loss_mask_px + 
                    damage_weight   * loss_damage
                )

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track progress
                train_losses.append(loss.item())
                pbar.set_postfix({"loss": loss.item()})
            
            # Calculate average training loss
            avg_train_loss = sum(train_losses) / len(train_losses)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            if val_dataloader:
                self.diffusion_model.eval()
                val_losses = []

                with torch.no_grad():
                    for batch in tqdm(val_dataloader, desc="Validation"):
                        # Unpack modal inputs
                        modality_data = {
                            name: batch[i] if i < len(batch) else None 
                            for i, name in enumerate(self.modality_names)
                        }

                        latents, joint_latent = self.encode_modalities(modality_data)
                        if joint_latent is None:
                            continue

                        batch_size = joint_latent.size(0)

                        # --- Adaptive prior noise ---
                        spec = modality_data["spec"].to(self.device)
                        real, imag = spec[:, 0::2], spec[:, 1::2]
                        mag = torch.sqrt(real**2 + imag**2)
                        sigma_t = energy_sigma(mag)
                        sigma_mean = sigma_t.mean(dim=1, keepdim=True)
                        z0 = torch.randn_like(joint_latent) * sigma_mean

                        # --- Flow-matching targets ---
                        t_int = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device)
                        tau   = t_int.view(-1, 1).float() / (self.noise_scheduler.num_timesteps - 1)
                        x_t   = tau * joint_latent + (1 - tau) * z0
                        v_tgt = joint_latent - z0
                        v_pred = self.diffusion_model(x_t, tau)

                        loss_fm = F.mse_loss(v_pred, v_tgt)

                        # --- Decode and add auxiliary losses ---
                        z_recon = v_pred + z0
                        outputs = self.decode_modalities(z_recon)

                        spec_pred = outputs["spec"]
                        spec_true = modality_data["spec"].to(self.device)

                        loss_spec   =  complex_spectrogram_loss(spec_true, spec_pred)
                        loss_p_g    =  gradient_loss_phase_only(spec_true, spec_pred)
                        loss_p_l    =  laplacian_loss_phase_only(spec_true, spec_pred)
                        loss_mag    =  magnitude_l1_loss(spec_true, spec_pred)

                        # robust .istft access (wrapped or not)
                        core = getattr(self, "module", self)
                        istft_layer = getattr(core, "istft", None)

                        if istft_layer is not None:
                            spec_pred_dn = spec_pred.clone()
                            C = spec_pred_dn.shape[1] // 2
                            spec_pred_dn[:, :C] = spec_pred_dn[:, :C] * self.sig_spec + self.mu_spec
                            # spec_pred_dn[:, C:] stays untouched (raw phase)
                            wav_pred = istft_layer(spec_pred_dn, length=spec_true.size(2) * 4)
                            wav_true = istft_layer(spec_true, length=spec_true.size(2) * 4)
                            loss_wave_si = waveform_si_l1_loss(wav_true, wav_pred)
                        else:
                            loss_wave_si = torch.tensor(0.0, device=spec_true.device)


                        mask_pred = outputs["mask"]
                        mask_true = modality_data["mask"].to(self.device)

                        loss_mask = focal_tversky_loss(mask_true, mask_pred, alpha=0.3, beta=0.8, gamma=focal_gamma)
                        loss_damage = damage_amount_loss(
                            mask_true,
                            mask_pred,
                            contrast_weight=contrast_weight,
                            margin=0.005
                        )

                        Loss = (
                            1               * loss_fm + 
                            0.5             * loss_spec +
                            0               * loss_p_g +
                            0               * loss_p_l +
                            0               * loss_mag +
                            0.5               * loss_wave_si +
                            1               * loss_mask + 
                            damage_weight   * loss_damage
                        )

                        val_losses.append(Loss.item())

                avg_val_loss = sum(val_losses) / len(val_losses)
                history["val_loss"].append(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.diffusion_model.state_dict(), f"{save_dir}/best_diffusion_model.pt")

            # Print epoch results
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}", end="")
            if val_dataloader:
                print(f", Val Loss: {avg_val_loss:.4f}")
            else:
                print("")
            
            # Save checkpoint
            if (epoch + 1) % 500 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.diffusion_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_train_loss,
                }, f"{save_dir}/diffusion_checkpoint_epoch{epoch+1}.pt")
        
        # Save final model
        torch.save(self.diffusion_model.state_dict(), f"{save_dir}/final_diffusion_model.pt")
        
        return history

    def encode_modalities(self, modality_data):
        latents = {}

        for modality_name, data in modality_data.items():
            if data is not None:

                # Defensive fix for spec shape
                if modality_name == "spec" and data.shape[1] != 24:
                    print(f"⚠️ WARNING: Expected spec channel=48 but got {data.shape[1]}, transposing...")
                    data = data.permute(0, 2, 3, 1)

                data = data.to(self.device)
                autoencoder = self.autoencoders[modality_name]
                with torch.no_grad():
                    _, latent = autoencoder(data)
                latents[modality_name] = latent
            else:
                latents[modality_name] = None

        joint_latent = torch.cat(
            [latents[name] for name in self.modality_names if latents[name] is not None], dim=1
        ) if any(latents[name] is not None for name in self.modality_names) else None

        return latents, joint_latent

    def decode_modalities(self, joint_latent):
        """
        Decode a joint latent representation back to each modality.
        
        Args:
            joint_latent: Joint latent representation
        
        Returns:
            Dict of decoded outputs for each modality
        """
        outputs = {}
        
        # Split the joint latent into individual modality latents
        start_idx = 0
        for i, modality_name in enumerate(self.modality_names):
            end_idx = start_idx + self.modality_dims[i]
            modality_latent = joint_latent[:, start_idx:end_idx]

            autoencoder = self.autoencoders[modality_name]
            with torch.no_grad():
                if modality_name == "spec":
                    output = autoencoder.decoder(modality_latent)
                else:
                    output = autoencoder.decoder(modality_latent)

            outputs[modality_name] = output
            start_idx = end_idx

        return outputs
    
    def create_modality_mask(self, batch_size, available_modalities):
        """
        Create a mask tensor for conditional generation.
        
        Args:
            batch_size: Batch size
            available_modalities: List of available modality names
        
        Returns:
            Binary mask tensor (1 for known elements, 0 for elements to predict)
        """
        mask = torch.zeros((batch_size, self.total_latent_dim), device=self.device)
        
        start_idx = 0
        for i, modality_name in enumerate(self.modality_names):
            if modality_name in available_modalities:
                end_idx = start_idx + self.modality_dims[i]
                mask[:, start_idx:end_idx] = 1.0
            start_idx += self.modality_dims[i]
        
        return mask
    
    @staticmethod
    def from_checkpoint(
        ae_dir: str,
        diff_ckpt_path: str,
        spec_norm_path: str,
        latent_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ):
        """
        Loads a full MultiModalLatentDiffusion system from disk:
        - Autoencoders
        - Diffusion model
        - Normalisation constants
        """
        # 1. Load normalization constants
        norm = np.load(spec_norm_path)
        mu_spec = torch.tensor(norm["mu"], dtype=torch.float32).to(device)
        sig_spec = torch.tensor(norm["sigma"], dtype=torch.float32).to(device)

        # 2. Rebuild autoencoders (must match dimensions from training)
        spec_autoencoder = SpectrogramAutoencoder(
            latent_dim, channels=12, freq_bins=129, time_bins=18
        ).to(device)
        mask_autoencoder = MaskAutoencoder(
            latent_dim, output_shape=(32, 96)
        ).to(device)

        # spec_autoencoder.load_state_dict(torch.load(os.path.join(ae_dir, "spec_autoencoder_best.pt"), map_location=device))


        spec_ckpt_path = os.path.join(ae_dir, "spec_autoencoder_best.pt")
        state_dict = torch.load(spec_ckpt_path, map_location=device)

        missing, unexpected = spec_autoencoder.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded AE from {spec_ckpt_path}")
        print("‼️  Missing keys:", missing)
        print("‼️  Unexpected keys:", unexpected)
        print("✅ adjust_conv.weight norm:", spec_autoencoder.encoder.adjust_conv.weight.norm().item())

        print("✅ Loaded spec autoencoder state dict")
        print("‣ adjust_conv.weight norm:", spec_autoencoder.encoder.adjust_conv.weight.norm().item())

        mask_autoencoder.load_state_dict(torch.load(os.path.join(ae_dir, "mask_autoencoder_best.pt"), map_location=device))
        print("‣ mask encoder norm:", mask_autoencoder.encoder.adjust_conv.weight.norm().item())


        # 3. Construct MLD wrapper
        mld = MultiModalLatentDiffusion(
            spec_autoencoder,
            mask_autoencoder,
            latent_dim=latent_dim,
            modality_names=["spec", "mask"],
            device=device,
            mu_spec=mu_spec,
            sig_spec=sig_spec
        )

        # 4. Load diffusion weights
        mld.diffusion_model.load_state_dict(torch.load(diff_ckpt_path, map_location=device))
        mld.diffusion_model.to(device)
        mld.eval()

        print("Final weight norm:", mld.autoencoders["spec"].encoder.adjust_conv.weight.norm().item())

        return mld

    def sample(self, batch_size=1, condition_on=None, condition_data=None):
        """
        Generate samples using the diffusion model.
        
        Args:
            batch_size: Number of samples to generate
            condition_on: Optional list of modality names to condition on
            condition_data: Optional dict of condition data for each modality
        
        Returns:
            Dict of generated outputs for each modality
        """
        self.diffusion_model.eval()
        
        # Determine what we're conditioning on (if anything)
        if condition_on and condition_data:
            # Encode condition data
            condition_latents = {}
            for modality_name in condition_on:
                if modality_name in condition_data and condition_data[modality_name] is not None:
                    data = condition_data[modality_name].to(self.device)
                    autoencoder = self.autoencoders[modality_name]
                    with torch.no_grad():
                        _, latent = autoencoder(data)
                    condition_latents[modality_name] = latent
                    batch_size = latent.shape[0]  # Update batch_size based on condition data
            
            # Create starting noise for unconditioned modalities
            x_T = torch.randn((batch_size, self.total_latent_dim), device=self.device)
            
            # Create modality mask
            mask = self.create_modality_mask(batch_size, condition_on)
            
            # Copy condition latents into the appropriate parts of x_T
            start_idx = 0
            for i, modality_name in enumerate(self.modality_names):
                if modality_name in condition_latents:
                    end_idx = start_idx + self.modality_dims[i]
                    x_T[:, start_idx:end_idx] = condition_latents[modality_name]
                start_idx += self.modality_dims[i]
            
            # Generate samples with conditioning
            with torch.no_grad():
                joint_latent = self.noise_scheduler.p_sample_loop(
                    self.diffusion_model,
                    shape=(batch_size, self.total_latent_dim),
                    device=self.device,
                    mask=mask,
                    x_T=x_T
                )
        else:
            # Unconditional generation using fixed σ (WaveFM prior fallback)
            default_sigma = 0.4
            with torch.no_grad():
                z0 = torch.randn((batch_size, self.total_latent_dim), device=self.device) * default_sigma
                joint_latent = self.noise_scheduler.p_sample_loop(
                    self.diffusion_model,
                    shape=(batch_size, self.total_latent_dim),
                    device=self.device,
                    x_T=z0
                )
        
        # Decode the joint latent back to each modality
        outputs = self.decode_modalities(joint_latent.float())

        return outputs
    

