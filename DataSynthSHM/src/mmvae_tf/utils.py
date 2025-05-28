import tensorflow as tf
import psutil
import GPUtil
import gc
import numpy as np

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

def vram_cleanup_if_needed(threshold_mb=31000):
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        # Check if usage exceeds threshold
        if gpu.memoryUsed > threshold_mb:
            print(f"\n⚠️ VRAM usage is {gpu.memoryUsed}MB — clearing session to prevent OOM.")
            tf.keras.backend.clear_session()
            gc.collect()
            break

def log_vram_usage():
    """Simple utility to print VRAM usage for all GPUs."""
    for gpu in GPUtil.getGPUs():
        print(f"🔍 GPU {gpu.id} => {gpu.memoryUsed}/{gpu.memoryTotal} MB used")

################################################################################
#---------Schedules-------------------------------------------------------------
################################################################################
def dynamic_weighting(epoch, max_epochs, min_weight=0.3, max_weight=0.5):
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
    progress = min(1.0, epoch / (max_epochs * 0.2))  # Reach max_weight halfway through training
    return min_weight + progress * (max_weight - min_weight)

def get_beta_schedule(epoch, max_epochs, schedule_type='cyclical', warmup=60, beta_cap=0.1):
    """
    Compute the beta coefficient for the JS divergence loss term.
    
    Args:
        epoch: Current epoch number (0-indexed)
        max_epochs: Total number of epochs
        schedule_type: Type of schedule ('linear', 'exponential', 'cyclical')
        warmup: Number of epochs for warmup phase
        beta_cap: Maximum allowed beta value to avoid over-regularization
    
    Returns:
        beta: Clipped beta coefficient for current epoch
    """
    BETA_MIN = 1e-8
    BETA_MAX = beta_cap

    WARMUP_EPOCHS = warmup

    if schedule_type == 'linear':
        if epoch < WARMUP_EPOCHS:
            beta = BETA_MIN + (BETA_MAX - BETA_MIN) * (epoch / WARMUP_EPOCHS)
        else:
            beta = BETA_MAX

    elif schedule_type == 'exponential':
        if epoch < WARMUP_EPOCHS:
            t = epoch / WARMUP_EPOCHS
            beta = BETA_MIN + (BETA_MAX - BETA_MIN) * (np.exp(3 * t) - 1) / (np.exp(3) - 1)
        else:
            beta = BETA_MAX

    elif schedule_type == 'cyclical':
        if epoch < WARMUP_EPOCHS:
            beta = BETA_MIN + (BETA_MAX - BETA_MIN) * (epoch / WARMUP_EPOCHS)
        else:
            cycle_period = 50
            cycle_position = ((epoch - WARMUP_EPOCHS) % cycle_period) / cycle_period
            beta_min_cycle = BETA_MAX / 5
            beta = beta_min_cycle + (BETA_MAX - beta_min_cycle) * 0.5 * (1 + np.cos(2 * np.pi * cycle_position))

    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")


    return beta

def get_time_weight(epoch, warmup=200, max_w=0.02):
    return max_w * tf.minimum(1.0, epoch / warmup)
