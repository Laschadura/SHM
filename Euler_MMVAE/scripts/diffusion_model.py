import os
import platform
import sys
import gc
import psutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import math
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
from tqdm import tqdm
import pandas as pd
import umap
import GPUtil

# Set working directory based on platform
if platform.system() == "Windows":
    os.chdir("c:/SP-Master-Local/SP_DamageLocalization-MasonryArchBridge_SimonScandella/ProbabilisticApproach/Euler_MMVAE")
else:
    os.chdir("/cluster/scratch/scansimo/Euler_MMVAE")
print("✅ Script has started executing")

# GPU Configuration
def configure_gpu():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✅ CUDA is available. {num_gpus} GPU(s) detected.")
        for i in range(num_gpus):
            print(f"   🔹 GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️ No GPU devices found — this script will run on CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 PyTorch will run on: {device}")

configure_gpu()

# Ensure access to sibling files
sys.path.append(os.path.dirname(__file__))
import data_loader

from utils import (
    dynamic_weighting,
    betas_for_alpha_bar,
    sinusoidal_time_embedding,
    energy_sigma,
    complex_spectrogram_loss,
    multi_channel_mrstft_loss,
    gradient_loss_phase_only,
    laplacian_loss_phase_only,
    damage_amount_loss,
    energy_sigma,
    magnitude_l1_loss,
    waveform_si_l1_loss,
    custom_mask_loss,
    focal_tversky_loss,
    DifferentiableISTFT
)


# ----- Recource Monitoring and Usage -----
def print_memory_stats():
    process = psutil.Process()
    print(f"RAM Memory Use: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id} Memory Use: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Cleared GPU memory and ran garbage collector.")

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
        
# ----- Loss Functions and schedules -----
def complex_spectrogram_loss(y_true, y_pred):
    """
    Spectrogram loss handling magnitude and phase separately.
    
    Args:
        y_true: Tensor of shape [batch, freq, time, channels*2]
        y_pred: Same shape as y_true

    Returns:
        Scalar tensor loss
    """
    # Separate magnitude and phase
    mag_true   = y_true[:, 0::2, :, :]
    mag_pred   = y_pred[:, 0::2, :, :]

    phase_true = y_true[:, 1::2, :, :]
    phase_pred = y_pred[:, 1::2, :, :]

    # Magnitude loss (log-MSE)
    eps = 1e-6
    mag_pred_safe = torch.clamp(mag_pred, min=0.0)
    log_mag_true  = torch.log(mag_true + eps)
    log_mag_pred  = torch.log(mag_pred_safe + eps)
    mag_loss = F.mse_loss(log_mag_true, log_mag_pred)


    # Relative magnitude error (optional, replace or add)
    rel_mag_err = ((mag_true - mag_pred) ** 2 / (mag_true**2 + 1e-6)).mean()

    # Phase loss (cosine similarity)
    phase_true_complex = torch.complex(torch.cos(phase_true), torch.sin(phase_true))
    phase_pred_complex = torch.complex(torch.cos(phase_pred), torch.sin(phase_pred))
    phase_diff_cos = torch.real(phase_true_complex * torch.conj(phase_pred_complex))
    phase_loss = torch.mean(1.0 - phase_diff_cos)

    # Total weighted
    return 0.6 * mag_loss + 0.2 * rel_mag_err + 0.2 * phase_loss

def spectro_time_consistency_loss(orig_wave, recon_spec,
                                  fs=200, nperseg=256, noverlap=224,
                                  weight=1.0):
    """
    orig_wave : (B, T, C)    pre-processed window
    recon_spec: (B, 2C, F, T) decoder output (log-mag | phase)

    Computes ISTFT of recon_spec, aligns RMS to orig_wave, and
    returns L1 distance; multiply by `weight` before adding.
    """
    with torch.no_grad():
        spec_np = recon_spec.detach().cpu().numpy().transpose(0, 2, 3, 1)
        wav_rec = data_loader.inverse_spectrogram(
            spec_np, time_length=orig_wave.shape[1],
            fs=fs, nperseg=nperseg, noverlap=noverlap)
        wav_rec = torch.tensor(wav_rec, device=orig_wave.device)

    # RMS-match before L1/L2
    scale = orig_wave.std(dim=1, keepdim=True) / (wav_rec.std(dim=1, keepdim=True) + 1e-8)
    wav_rec *= scale
    return weight * F.l1_loss(wav_rec, orig_wave)

def _quick_mask_stats(mask_true, mask_logits, mask_prob):
    """
    Prints one-line stats for masks.
    mask_true   : (B,1,H,W) float 0–1
    mask_logits : (B,1,H,W) pre-sigmoid
    mask_prob   : (B,1,H,W) after sigmoid
    """
    with torch.no_grad():
        dice = (2 * (mask_true * (mask_prob > 0.5)).sum(dim=(-2,-1))
                / (mask_true.sum(dim=(-2,-1)) + (mask_prob > 0.5).sum(dim=(-2,-1)) + 1e-6))
        print(f"   GT mean={mask_true.mean():.4f} ┃ "
              f"logit μ={mask_logits.mean():+.2f} σ={mask_logits.std():.2f} ┃ "
              f"prob μ={mask_prob.mean():.4f} ┃ "
              f"Dice[0]={dice[0].item():.3f}")

# ----- Helpers -----
def _upsample_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.LeakyReLU(0.1)
    )

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
        x = self.conv_out(x)
        
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
        device="cuda" if torch.cuda.is_available() else "cpu"
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
                real, imag = spec[:, 0::2], spec[:, 1::2]
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

                if hasattr(self, 'istft'):
                    wav_pred = self.istft(spec_pred, length=spec_true.size(2) * 4)
                    wav_true = self.istft(spec_true, length=spec_true.size(2) * 4)
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
                    0.1             * loss_complex +
                    0               * loss_phase_g +
                    0               * loss_phase_l +
                    0               * loss_mag_l1 +
                    0               * loss_wave_si +
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

                        if hasattr(self, 'istft'):
                            wav_pred = self.istft(spec_pred, length=spec_true.size(2) * 4)
                            wav_true = self.istft(spec_true, length=spec_true.size(2) * 4)
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
                            0.1             * loss_spec +
                            0               * loss_p_g +
                            0               * loss_p_l +
                            0               * loss_mag +
                            0               * loss_wave_si +
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
    
    def save_autoencoders(self, save_dir="./results/autoencoders"):
        """Save the autoencoders"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, ae in self.autoencoders.items():
            torch.save(ae.state_dict(), f"{save_dir}/{name}_autoencoder.pt")
    
    def load_autoencoders(self, load_dir="./results/autoencoders"):
        """Load the autoencoders"""
        for name, ae in self.autoencoders.items():
            path = f"{load_dir}/{name}_autoencoder.pt"
            if os.path.exists(path):
                ae.load_state_dict(torch.load(path, map_location=self.device))
                print(f"Loaded {name} autoencoder from {path}")
    
    def save_diffusion_model(self, path="./results/diffusion/diffusion_model.pt"):
        """Save the diffusion model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.diffusion_model.state_dict(), path)
    
    def load_diffusion_model(self, path="./results/diffusion/diffusion_model.pt"):
        """Load the diffusion model"""
        if os.path.exists(path):
            self.diffusion_model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded diffusion model from {path}")


# ===== Helper Function for Creating Datasets =====
def create_torch_dataset(spec_data, mask_data, seg_data, test_ids, batch_size, shuffle=True):
    """
    Create a PyTorch DataLoader from the spectrogram and mask data.
    
    Args:
        spec_data: Spectrogram data as a numpy array
        mask_data: Mask data as a numpy array
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
    
    Returns:
        PyTorch DataLoader
    """
    # Convert numpy arrays to PyTorch tensors
    spec    = torch.tensor(spec_data, dtype=torch.float32)
    mask    = torch.tensor(mask_data, dtype=torch.float32)
    seg     = torch.tensor(seg_data,  dtype=torch.float32)
    ids     = torch.tensor(test_ids, dtype=torch.int32)
    
    # Create TensorDataset
    dataset = TensorDataset(spec, mask, seg, ids)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader

# ===== Training Functions =====
def train_autoencoders(
        spec_autoencoder,
        mask_autoencoder,
        train_loader,
        val_loader,
        device,
        epochs: int = 100,
        lr: float = 5e-4,
        patience: int = 20
    ):
    """
    Trains the spectrogram and mask autoencoders separately.

    Args
    ----
    spec_autoencoder : SpectrogramAutoencoder
    mask_autoencoder : MaskAutoencoder
    train_loader     : DataLoader yielding (spec, mask, seg, id)
    device           : torch.device
    epochs           : int  – number of epochs
    lr               : float – learning‑rate
    """
    spec_autoencoder.to(device)
    mask_autoencoder.to(device)

    optimizer_spec = optim.AdamW(spec_autoencoder.parameters(), lr=lr)
    optimizer_mask = optim.AdamW(mask_autoencoder.parameters(), lr=lr)

    scheduler_spec = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_spec, T_max=epochs
    )
    scheduler_mask = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_mask, T_max=epochs
    )

    best_spec_loss = float("inf")
    best_mask_loss = float("inf")
    patience_counter = 0

    #track losses for plotting
    spec_history = {"train_loss": [], "val_loss": []}
    mask_history = {"train_loss": [], "val_loss": []}


    for epoch in range(epochs):
        # --- Dynamic hyperparameters ---
        damage_weight = 0.3 if epoch >= 200 else 2          # (1) ↓ loss_damage weight
        focal_gamma   = 1.5 if epoch >= 250 else 1.0        # (2) ↑ gamma focality
        contrast_weight = min(0.3, epoch / 100 * 0.3)       # (3) adaptive ramp-up
        
        # Custom mask loss weights: schedule
        if epoch < 200:
            weight_bce   = 0.4
            weight_dice  = 0.5
            weight_focal = 0.1
        else:
            weight_bce   = 0.1
            weight_dice  = 0.8
            weight_focal = 0.1
        # ------------------------------------------------
        spec_autoencoder.train()
        mask_autoencoder.train()

        spec_loss_sum  = 0.0     # running *sum* for this epoch
        mask_loss_sum  = 0.0
        batch_counter  = 0

        for spec_batch, mask_batch, seg_batch, _ in train_loader:
            batch_counter += 1

            spec_batch = spec_batch.to(device)
            mask_batch = mask_batch.to(device)
            seg_batch  = seg_batch.to(device)

            # ────────────────────── SPECTROGRAM AUTOENCODER ──────────────────────
            optimizer_spec.zero_grad()
            recon_spec, _ = spec_autoencoder(spec_batch)

            loss_spec_mag  = complex_spectrogram_loss(spec_batch, recon_spec)         # main spec loss
            loss_time      = spectro_time_consistency_loss(seg_batch, recon_spec)     # waveform ↔ STFT alignment
            loss_mag_l1    = magnitude_l1_loss(spec_batch, recon_spec)                # adds sharpness to mag
            loss_laplacian = laplacian_loss_phase_only(spec_batch, recon_spec)        # improves phase structure

            # Reconstruct waveform and apply waveform-domain loss (optional)
            if hasattr(spec_autoencoder, 'istft'):
                wav_recon = spec_autoencoder.istft(recon_spec, length=seg_batch.size(1))  # (B, T, C)
                loss_wave_si = waveform_si_l1_loss(seg_batch, wav_recon)
            else:
                loss_wave_si = torch.tensor(0.0, device=spec_batch.device) # zero if not using istft

            loss_spec_total = (
                loss_spec_mag +
                0.0 * loss_time +
                0.1 * loss_mag_l1 +
                0.1 * loss_laplacian + 
                5 * loss_wave_si
            )

            loss_spec_total.backward()
            optimizer_spec.step()
            spec_loss_sum += loss_spec_total.item()

            # ─────────────────────── MASK AUTOENCODER ─────────────────────────────
            optimizer_mask.zero_grad()
            recon_mask, _ = mask_autoencoder(mask_batch)

            # ── diagnostics (first batch each epoch) ──────────────────────────────
            if batch_counter == 1:
                mask_logits = torch.logit(recon_mask.clamp(1e-4, 1-1e-4))
                _quick_mask_stats(mask_batch, mask_logits, recon_mask)
            # ---------------------------------------------------------------------


            loss_mask_px = focal_tversky_loss(mask_batch, recon_mask, alpha=0.3, beta=0.8, gamma=focal_gamma)

            loss_damage = damage_amount_loss(
                mask_batch,
                recon_mask,
                contrast_weight=contrast_weight,
                margin=0.005
            )

            loss_dice = custom_mask_loss(
                    mask_batch, recon_mask,
                    weight_bce=weight_bce,
                    weight_dice=weight_dice,
                    weight_focal=weight_focal,
                    gamma= 2.0,
                    alpha= 0.3
            )

            loss_mask_total = (
                0               * loss_mask_px +
                damage_weight   * loss_damage +
                1               * loss_dice
            )

            loss_mask_total.backward()
            optimizer_mask.step()
            mask_loss_sum += loss_mask_total.item()

        # -------- end of epoch: compute means & log --------
        avg_spec_loss = spec_loss_sum / batch_counter
        avg_mask_loss = mask_loss_sum / batch_counter
        print(f"[Epoch {epoch+1:3d}/{epochs}]  "
            f"spec={avg_spec_loss:.4f}  mask={avg_mask_loss:.4f}")
        
        # AE Validation (mirrors training structure)
        spec_autoencoder.eval()
        mask_autoencoder.eval()
        val_loss_spec, val_loss_mask = [], []

        with torch.no_grad():
            for spec_batch_val, mask_batch_val, seg_batch_val, _ in val_loader:
                spec_batch_val = spec_batch_val.to(device)
                mask_batch_val = mask_batch_val.to(device)
                seg_batch_val  = seg_batch_val.to(device)

                recon_spec_val, _ = spec_autoencoder(spec_batch_val)
                loss_spec_mag  = complex_spectrogram_loss(spec_batch_val, recon_spec_val)
                loss_time      = spectro_time_consistency_loss(seg_batch_val, recon_spec_val)
                loss_mag_l1    = magnitude_l1_loss(spec_batch_val, recon_spec_val)
                loss_laplacian = laplacian_loss_phase_only(spec_batch_val, recon_spec_val)

                if hasattr(spec_autoencoder, 'istft'):
                    wav_recon_val = spec_autoencoder.istft(recon_spec_val, length=seg_batch_val.size(1))
                    loss_wave_si = waveform_si_l1_loss(seg_batch_val, wav_recon_val)
                else:
                    loss_wave_si = torch.tensor(0.0, device=spec_batch_val.device)

                loss_spec_val_total = (
                    loss_spec_mag +
                    0.0 * loss_time +
                    0.1 * loss_mag_l1 +
                    0.1 * loss_laplacian +
                    5 * loss_wave_si
                )
                val_loss_spec.append(loss_spec_val_total.item())

                recon_mask_val, _ = mask_autoencoder(mask_batch_val)
                loss_mask_px = focal_tversky_loss(mask_batch_val, recon_mask_val, alpha=0.3, beta=0.8, gamma=focal_gamma)

                loss_damage  = damage_amount_loss(
                        mask_batch_val,
                        recon_mask_val,
                        contrast_weight=contrast_weight,
                        margin=0.005
                )
                
                loss_dice = custom_mask_loss(
                    mask_batch_val, recon_mask_val,
                    weight_bce=weight_bce,
                    weight_dice=weight_dice,
                    weight_focal=weight_focal,
                    gamma= 2.0,
                    alpha= 0.3
                )

                loss_mask_val_total = (
                    0               * loss_mask_px +
                    damage_weight   * loss_damage +
                    1               * loss_dice
                )
                val_loss_mask.append(loss_mask_val_total.item())

        print(f"AE Val Spec Loss: {np.mean(val_loss_spec):.4f}, AE Val Mask Loss: {np.mean(val_loss_mask):.4f}")

        #track losses for plotting
        spec_history["train_loss"].append(avg_spec_loss)
        spec_history["val_loss"].append(np.mean(val_loss_spec))
        mask_history["train_loss"].append(avg_mask_loss)
        mask_history["val_loss"].append(np.mean(val_loss_mask))

        # step the LR schedulers
        scheduler_spec.step()
        scheduler_mask.step()

        # Early stopping logic based on spec loss (or mask loss, or both)
        if avg_spec_loss < best_spec_loss:
            best_spec_loss = avg_spec_loss
            best_mask_loss = avg_mask_loss
            patience_counter = 0
            # Save best model weights
            torch.save(spec_autoencoder.state_dict(), "results_diff/autoencoders/spec_autoencoder_best.pt")
            torch.save(mask_autoencoder.state_dict(), "results_diff/autoencoders/mask_autoencoder_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1} after no improvement for {patience} epochs.")
                break
        
        return spec_history, mask_history




# ----- Visualization Functions and tests -----
def visualize_training_history(history, save_path=None):
    """
    Visualize the training history.

    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')

    if history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.close()

def save_visualizations_and_metrics(model, train_loader, val_loader, training_metrics, output_dir="results_diff"):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    def plot_training_curves(metrics):
        epochs = list(range(1, len(metrics['train_loss']) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=metrics['train_loss'],
                                 mode='lines+markers', name="Train Loss", line=dict(color='blue')))
        if metrics['val_loss']:
            fig.add_trace(go.Scatter(x=epochs, y=metrics['val_loss'],
                                     mode='lines+markers', name="Val Loss", line=dict(color='red')))
        fig.update_layout(title="Loss vs Epochs",
                          xaxis_title="Epoch", yaxis_title="Loss",
                          template="plotly_white")
        file_path = os.path.join(plots_dir, "train_val_loss.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved training curves to {file_path}")

    plot_training_curves(training_metrics)

    def extract_and_reduce_latents(loader):
        # Explicitly set evaluation mode for internal modules
        model.diffusion_model.eval()
        for ae in model.autoencoders.values():
            ae.eval()

        latents, ids = [], []
        device = model.device  # Correct way to get the device

        for spec, _, _, test_id in loader:
            spec = spec.to(device)
            with torch.no_grad():
                _, z = model.autoencoders["spec"](spec)
            latents.append(z.cpu().numpy())
            ids.append(test_id.numpy().flatten())

        latents = np.concatenate(latents, axis=0)
        ids = np.concatenate(ids, axis=0)

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.05, n_components=3, random_state=42)

        latent_3d = reducer.fit_transform(latents)

        return latent_3d, ids

    latent_3d, train_ids = extract_and_reduce_latents(train_loader)

    def plot_latent_space_3d(latent_3d, ids):
        df = pd.DataFrame(latent_3d, columns=["UMAP_1", "UMAP_2", "UMAP_3"])
        df["Test ID"] = pd.to_numeric(ids, errors="coerce")
        fig = px.scatter_3d(df, x="UMAP_1", y="UMAP_2", z="UMAP_3", color="Test ID",
                             color_continuous_scale="Viridis", title="Latent Space Visualization (3D UMAP)", opacity=0.8)
        file_path = os.path.join(plots_dir, "latent_space_3d.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved 3D latent space plot to {file_path}")

    plot_latent_space_3d(latent_3d, train_ids)

    def latent_analysis(loader):
        model.eval()
        latents = []
        for spec, _, _, _ in loader:
            spec = spec.to(next(model.parameters()).device)
            with torch.no_grad():
                _, z = model.autoencoders["spec"](spec)
            latents.append(z.cpu().numpy())
        latents = np.concatenate(latents, axis=0)
        norms = np.linalg.norm(latents, axis=1, keepdims=True)
        normalized = latents / (norms + 1e-8)
        cosine_sim = np.dot(normalized, normalized.T).diagonal()
        fig = go.Figure(data=go.Histogram(x=cosine_sim, histnorm='probability density', marker_color='blue', opacity=0.7))
        fig.update_layout(title="Cosine Similarity Distribution (Validation Latents)",
                          xaxis_title="Cosine Similarity", yaxis_title="Probability Density", template="plotly_white")
        file_path = os.path.join(plots_dir, "cosine_similarity_hist.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved cosine similarity histogram to {file_path}")
        return {"avg_cosine_similarity": float(np.mean(cosine_sim))}

    latent_metrics = latent_analysis(val_loader)
    print("Latent analysis metrics:", latent_metrics)

    return {
        "latent_metrics": latent_metrics,
        "latent_space_3d": latent_3d
    }


# ----- Main Function -----
def main():
    # ─── Mode Control ───────────────────────────────────────────────
    train_AE        = True                   # whether to train AE
    ae_epochs      = 500                   # epochs for autoencoder training
    patience_ae     = 50                     # patience for autoencoder training
    recompute_data  = False                   # whether to recompute cache
    dm_mode         = "scratch"              # "scratch" or "continue"
    dm_epochs       = 1                    # extra epochs for diffusion
    learning_rate_dm = 5e-4
    learning_rate_ae = 5e-4
    diff_ckpt       = "results_diff/diffusion/final_diffusion_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results_diff/autoencoders", exist_ok=True)
    os.makedirs("results_diff/diffusion", exist_ok=True)

    # ─── Data Parameters ────────────────────────────────────────────
    fs = 200
    segment_duration = 4.0
    nperseg = 256
    noverlap = 224
    latent_dim = 256
    batch_size = 200

    tag = f"{segment_duration:.2f}s_{nperseg}_{noverlap}"
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    final_path    = os.path.join(cache_dir, f"specs_{tag}.npy")
    heatmaps_path = os.path.join(cache_dir, f"masks_{tag}.npy")
    ids_path      = os.path.join(cache_dir, f"segIDs_{tag}.npy")
    seg_path      = os.path.join(cache_dir, f"segments_{tag}.npy")

    # ─── Load or compute features ───────────────────────────────────
    all_cached = all(os.path.exists(p) for p in [final_path, heatmaps_path, ids_path, seg_path])
    if all_cached and not recompute_data:
        print("✅ Loading cached files...")
        spectral_features = np.load(final_path)
        mask_segments     = np.load(heatmaps_path)
        test_ids          = np.load(ids_path)
        segments          = np.load(seg_path)
    else:
        print("⚠️  Cache missing or recompute requested … building fresh.")
        _, _, heatmaps, segments, spectrograms, test_ids = data_loader.load_data(
            segment_duration = segment_duration,
            nperseg          = nperseg,
            noverlap         = noverlap,
            sample_rate      = fs,
            recompute        = recompute_data,
            cache_dir        = cache_dir
        )
        spectral_features = spectrograms
        mask_segments     = np.stack([heatmaps[tid] for tid in test_ids], axis=0)

        np.save(final_path,    spectral_features)
        np.save(heatmaps_path, mask_segments)
        np.save(ids_path,      test_ids)
        np.save(seg_path,      segments)
        print("✅ New cache written.")

    # ─── Dataset Splitting ──────────────────────────────────────────
    spectral_features = spectral_features.transpose(0, 3, 1, 2)  # (N,2C,F,T)
    mask_segments     = mask_segments.transpose(0, 3, 1, 2)

    N = spectral_features.shape[0]
    indices = np.random.permutation(N)
    split = int(0.8 * N)
    train_idx, val_idx = indices[:split], indices[split:]

    train_spec = spectral_features[train_idx]
    val_spec   = spectral_features[val_idx]
    train_mask = mask_segments[train_idx]
    val_mask   = mask_segments[val_idx]
    train_ids  = test_ids[train_idx]
    val_ids    = test_ids[val_idx]
    train_seg  = segments[train_idx]
    val_seg    = segments[val_idx]

    print(f"✅ Train: {train_spec.shape[0]} samples | Val: {val_spec.shape[0]} samples")

    # ─── Create Torch Loaders ───────────────────────────────────────
    train_loader = create_torch_dataset(train_spec, train_mask, train_seg, train_ids, batch_size=batch_size)
    val_loader   = create_torch_dataset(val_spec,   val_mask,   val_seg,   val_ids,   batch_size=batch_size)

    freq_bins, time_bins = train_spec.shape[2:]
    channels = train_spec.shape[1] // 2
    mask_height, mask_width = mask_segments.shape[-2:]

    # ─── Build Model ────────────────────────────────────────────────
    spec_autoencoder = SpectrogramAutoencoder(latent_dim, channels, freq_bins, time_bins).to(device)
    mask_autoencoder = MaskAutoencoder(latent_dim, (mask_height, mask_width)).to(device)

    spec_autoencoder.istft = DifferentiableISTFT(nperseg=256, noverlap=224).to(device)



    if train_AE:
        print("🚀 Training Autoencoders...")
        spec_history, mask_history = train_autoencoders(
            spec_autoencoder, mask_autoencoder,
            train_loader, val_loader,
            device, epochs=ae_epochs,
            lr=learning_rate_ae, patience=patience_ae
        )
        visualize_training_history(spec_history, save_path="results_diff/autoencoders/spec_autoencoder_loss.png")
        visualize_training_history(mask_history, save_path="results_diff/autoencoders/mask_autoencoder_loss.png")

        torch.save(spec_autoencoder.state_dict(), "results_diff/autoencoders/spec_autoencoder.pt")
        torch.save(mask_autoencoder.state_dict(), "results_diff/autoencoders/mask_autoencoder.pt")
    else:
        print("🔄 Loading pretrained autoencoders...")
        spec_autoencoder.load_state_dict(torch.load("results_diff/autoencoders/spec_autoencoder.pt", map_location=device))
        mask_autoencoder.load_state_dict(torch.load("results_diff/autoencoders/mask_autoencoder.pt", map_location=device))

    mld = MultiModalLatentDiffusion(spec_autoencoder, mask_autoencoder, latent_dim, ["spec", "mask"], device)

    mld.istft = DifferentiableISTFT(nperseg=256, noverlap=224).to(device)


    # ─── Diffusion training control ─────────────────────────────────
    if dm_mode == "continue":
        if os.path.exists(diff_ckpt):
            print("🔄 Continuing diffusion training from checkpoint.")
            mld.load_diffusion_model(diff_ckpt)
        else:
            print("❌ No diffusion checkpoint found – aborting continue mode.")
            return
    elif dm_mode == "scratch":
        print("🆕 Training diffusion model from scratch.")
    else:
        raise ValueError(f"Invalid dm_mode: {dm_mode}. Choose 'scratch' or 'continue'.")

    print(f"🚀 Training diffusion model for {dm_epochs} epoch(s).")
    history = mld.train_diffusion_model(
        train_dataloader = train_loader,
        val_dataloader   = val_loader,
        num_epochs       = dm_epochs,
        learning_rate    = learning_rate_dm,
        save_dir         = "results_diff/diffusion"
    )

    mld.save_diffusion_model(diff_ckpt)

    visualize_training_history(history, save_path="results_diff/diffusion/training_curve.png")
    save_visualizations_and_metrics(mld, train_loader, val_loader, training_metrics=history, output_dir="results_diff")

    print("✅ All training, evaluation, and synthesis complete.")



if __name__ == "__main__":
    main()
