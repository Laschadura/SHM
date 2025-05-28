import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import warnings


def save_autoencoders(autoencoders: dict, save_dir="./results/autoencoders"):
    os.makedirs(save_dir, exist_ok=True)
    for name, ae in autoencoders.items():
        torch.save(ae.state_dict(), f"{save_dir}/{name}_autoencoder.pt")

def load_autoencoders(autoencoders: dict, device, load_dir="./results/autoencoders"):
    for name, ae in autoencoders.items():
        cand = [
            f"{load_dir}/{name}_autoencoder_best.pt",
            f"{load_dir}/{name}_autoencoder_final.pt",
            f"{load_dir}/{name}_autoencoder.pt",
        ]
        for path in cand:
            if os.path.exists(path):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
                    state_dict = torch.load(path, map_location=device)
                ae.load_state_dict(state_dict)
                print(f"Loaded {name} autoencoder from {path}")
                break
        else:
            print(f"⚠️  No weights found for {name} in {load_dir}")

def save_diffusion_model(model, path="./results/diffusion/diffusion_model.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_diffusion_model(model, device, path="./results/diffusion/diffusion_model.pt"):
    if os.path.exists(path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
            state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded diffusion model from {path}")

# ===== Helper Function for Creating Datasets =====
def create_torch_dataset(spec_data, mask_data, seg_data, test_ids, seg_stats, mu, std, batch_size, shuffle=True):
    """
    Create a PyTorch DataLoader from the spectrogram and mask data.
    
    Args:
        spec_data: Spectrogram data as a numpy array
        mask_data: Mask data as a numpy array
        seg_data: ts segments as a numpy array
        test_ids: Test IDs as a numpy array
        seg_stats: Segment statistics (mean, std) as a list of dicts
        mu: Global (train) mean for normalization
        std: Global (train) std for normalization
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
    
    Returns:
        PyTorch DataLoader
    """
    # Create TensorDataset
    dataset = SpectrogramDataset(spec_data, mask_data, seg_data, test_ids, seg_stats, mu, std)
    
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

# ===== Helper for Nomalizing Spectograms =====
class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, specs, masks, segs, ids, seg_stats, mu=None, std=None):
        self.specs  = torch.as_tensor(specs,  dtype=torch.float32)
        self.masks  = torch.as_tensor(masks,  dtype=torch.float32)
        self.segs   = torch.as_tensor(segs,   dtype=torch.float32)
        self.ids    = torch.as_tensor(ids,    dtype=torch.int32)
        self.stats  = seg_stats  # list of dicts
        self.mu     = torch.as_tensor(mu, dtype=torch.float32) if mu is not None else None  # shape: (C, 1, 1)
        self.std    = torch.as_tensor(std, dtype=torch.float32) if std is not None else None

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        x = self.specs[idx].clone()
        if self.mu is not None and self.std is not None:
            C = x.shape[0] // 2
            x[:C] = (x[:C] - self.mu) / self.std  # Only normalize log-magnitude
            # x[C:] is phase → remains untouched
        return x, self.masks[idx], self.segs[idx], self.ids[idx], self.stats[idx]

