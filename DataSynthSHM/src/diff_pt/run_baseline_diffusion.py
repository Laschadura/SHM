# run_baseline_diffusion.py
import torch
import numpy as np
import diffusion_model as dm
import data_loader
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Script has started executing")

# ─── load cached preprocessed arrays ─────────────────────────────
accel_dict, binary_masks, heats, segments, specs, ids = data_loader.load_data(
    segment_duration=4.0,
    nperseg=256,
    noverlap=224,
    sample_rate=200,
    recompute=False,
    cache_dir="cache"
)

# Prepare tensors
specs = specs.transpose(0, 3, 1, 2)          # (N, 2C, F, T)
masks = np.stack([heats[i] for i in ids], 0).transpose(0, 3, 1, 2)  # (N,1,H,W)
ids   = ids.astype("int32")

# Wrap in dataset
dataset = TensorDataset(
    torch.from_numpy(specs).float(),
    torch.from_numpy(masks).float(),
    torch.from_numpy(ids)
)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ─── build model ─────────────────────────────────────────────────
latent_dim = 256
C, F, T = specs.shape[1]//2, specs.shape[2], specs.shape[3]
H, W    = masks.shape[2:]

specAE = dm.SpectrogramAutoencoder(latent_dim, C, F, T).to(device)
maskAE = dm.MaskAutoencoder(latent_dim, (H, W)).to(device)
mld    = dm.MultiModalLatentDiffusion(specAE, maskAE, latent_dim, ["spec", "mask"], device)

# ─── Train 1 epoch of diffusion ─────────────────────────────────
history = mld.train_diffusion_model(
    train_dataloader = train_loader,
    val_dataloader   = None,
    num_epochs       = 1,
    learning_rate    = 1e-4,
    save_dir         = "baseline_diff"
)

print("📉 Avg. training loss after 1 epoch:", history["train_loss"][-1])
