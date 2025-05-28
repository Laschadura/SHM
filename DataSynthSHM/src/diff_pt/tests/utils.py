"""
Small helper utilities that both test-modules need.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import correlate

from bridge_data.postprocess import inverse_spectrogram
from bridge_data.postprocess import align_by_xcorr


# --------------------------------------------------------------------------- #
#                       NORMALISATION HANDLING                                #
# --------------------------------------------------------------------------- #
def load_train_stats(stats_path: Path):
    """
    Returns global train μ and σ that were saved by the training script
    (see diff_pt.io.save_train_stats).  Falls back to (0,1) if not found.
    """
    if stats_path.exists():
        mu, sigma = np.load(stats_path)
        print(f"✅ Loaded train μ={mu:.5f}  σ={sigma:.5f}")
        return float(mu), float(sigma)
    print("⚠️  Train-statistics file not found – assuming μ=0, σ=1.")
    return 0.0, 1.0


# --------------------------------------------------------------------------- #
#                   SPECTROGRAM  →  TIME-SERIES                               #
# --------------------------------------------------------------------------- #
def spectrograms_to_timeseries(spec_batch: torch.Tensor,
                               *,
                               mu: float,
                               sigma: float,
                               fs: int = 200,
                               nperseg: int = 256,
                               noverlap: int = 224,
                               hop_align: bool = True):
    """
    1. Undo global μ/σ on **log-magnitude**
    2. ISTFT  → (B, T, C)
    3. Optional sample-accurate alignment via xcorr
    """

    # (B, 2C, F, T) → (B, F, T, 2C)
    spec_np = spec_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)

    # ------ 1) reverse global normalisation (only log-mag channels) --------
    spec_np[..., 0::2] = spec_np[..., 0::2] * sigma.cpu().numpy() + mu.cpu().numpy()

    # ------ 2) ISTFT --------------------------------------------------------
    time_len = int(fs * 4.0)          # 4-second windows everywhere
    ts = inverse_spectrogram(spec_np,
                             time_length=time_len,
                             fs=fs,
                             nperseg=nperseg,
                             noverlap=noverlap)              # (B,T,C)

    if not hop_align:
        return ts

    # ------ 3) cross-correlation alignment ---------------------------------
    hop = nperseg - noverlap
    max_lag = hop // 2

    # the first channel of the first window is the reference
    ref_window = ts[0]                # (T, C)
    ref_sig    = ref_window[:, 0:1]   # keep shape (T, 1)

    # align each window en masse — returns rolled copy
    ts_aligned, _ = zip(*[
        align_by_xcorr(ref_sig, win, max_lag=max_lag) for win in ts
    ])
    ts_aligned = np.stack(ts_aligned, axis=0)

    return ts_aligned
