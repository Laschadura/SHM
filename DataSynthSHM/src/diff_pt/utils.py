import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
import psutil
import GPUtil


# ---------- helper -----------------------------------------------------------
def _split_mag_sincos_from_phase(spec):
    """
    spec : (B, 2*C, F, T)   = [mag, φ]
    Returns:
        mag   : (B, C, F, T)
        sinφ  : (B, C, F, T)
        cosφ  : (B, C, F, T)
    """
    C = spec.shape[1] // 2
    mag   = spec[:, 0::2]
    phase = spec[:, 1::2]
    return mag, torch.sin(phase), torch.cos(phase)


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

# ---------- Schedules -----------------------------------------------------------
def energy_sigma(mag: torch.Tensor, eps=1e-6, clip=(0.1, 1.0)) -> torch.Tensor:
    """
    Compute per-frame RMS energy → scale for prior noise.
    Input: mag (B, C, F, T)
    Output: σ (B, T)
    """
    energy = mag.square().sum(1)         # (B, F, T)  ← sum over channels
    rms    = energy.sqrt()               # RMS energy (B, F, T)
    sigma  = rms.mean(1)                 # mean over F → (B, T)

    # Normalize to 0–1
    sigma = (sigma - sigma.mean()) / (sigma.std() + eps)
    sigma = sigma.sigmoid()             # squash to (0,1)
    return sigma.clamp(*clip)           # final σ ∈ [0.1, 1.0]

def dynamic_weighting(epoch, max_epochs, min_weight=0.3, max_weight=1.0):
    """
    Gradually adjusts the weight between spectrogram and mask losses.
    
    Args:
        epoch: Current epoch
        max_epochs: Total epochs
        min_weight: Starting weight
        max_weight: Maximum weight
        
    Returns:
        Current weight
    """
    # Linear increase over time
    progress = min(1.0, epoch / (max_epochs * 0.1))  # Reach max_weight halfway through training
    return min_weight + progress * (max_weight - min_weight)

def betas_for_alpha_bar(num_timesteps, max_beta=0.999):
    ts = torch.linspace(0, num_timesteps, num_timesteps, dtype=torch.float64)
    
    # Convert the constant angle to a torch.Tensor:
    angle_const = torch.tensor(
        (0.008 / 1.008) * math.pi * 0.5,
        dtype=ts.dtype,
        device=ts.device
    )

    # Now both cos() calls receive torch.Tensor arguments
    numerator = torch.cos(((ts / num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
    denominator = torch.cos(angle_const) ** 2

    alphas_cumprod = numerator / denominator
    alphas = alphas_cumprod / alphas_cumprod.roll(1, 0)
    betas = 1.0 - alphas
    betas[0] = 1e-8
    return torch.clip(betas, 0, max_beta)

def sinusoidal_time_embedding(timesteps, dim):
    # timesteps: (batch_size,) or (batch_size,1)
    # This returns an embedding of shape (batch_size, dim)
    import math
    half_dim = dim // 2
    embeddings = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        * (-math.log(10000) / half_dim)
    )
    # If timesteps is (B,1), flatten it:
    timesteps = timesteps.view(-1)
    embeddings = timesteps[:, None] * embeddings[None, :]
    embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1)
    return embeddings

def scheduler(start_epoch, end_epoch, curent_epoch, min_weight=0.3, max_weight=1.0):
    """
    A simple linear scheduler for a weight.
    """
    if curent_epoch < start_epoch:
        return min_weight
    elif start_epoch <= curent_epoch < end_epoch:
        progress = (curent_epoch - start_epoch) / (end_epoch - start_epoch)
        return min_weight + progress * (max_weight - min_weight)
    else:
        return max_weight

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