import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import _split_mag_phase
from bridge_data.postprocess import inverse_spectrogram


# --------------Spectrogram Losses-------------------
def complex_spectrogram_loss(spec_true: torch.Tensor,
                              spec_pred: torch.Tensor) -> torch.Tensor:
    """
    • magnitude MSE  
    • phase cosine distance  
    • instantaneous-frequency L1
    """
    mag_t, phase_t = _split_mag_phase(spec_true)
    mag_p, phase_p = _split_mag_phase(spec_pred)

    # 1) magnitude (L2)
    mag_loss = F.mse_loss(mag_p, mag_t)

    # 2) phase cosine distance
    diff_cos = torch.cos(phase_p - phase_t)      # cos(Δφ)
    phase_loss = torch.mean(1.0 - diff_cos)

    # 3) instantaneous frequency (optional)
    if_t = phase_t[..., 1:] - phase_t[..., :-1]
    if_p = phase_p[..., 1:] - phase_p[..., :-1]
    if_loss = torch.mean(torch.abs(if_p - if_t))

    return 0.5 * mag_loss + 0.3 * phase_loss + 0.2 * if_loss

def gradient_loss_phase_only(spec_true: torch.Tensor,
                             spec_pred: torch.Tensor) -> torch.Tensor:
    _, phase_t = _split_mag_phase(spec_true)
    _, phase_p = _split_mag_phase(spec_pred)

    dx_t = phase_t[:, :, 1:, :] - phase_t[:, :, :-1, :]
    dx_p = phase_p[:, :, 1:, :] - phase_p[:, :, :-1, :]

    dy_t = phase_t[:, :, :, 1:] - phase_t[:, :, :, :-1]
    dy_p = phase_p[:, :, :, 1:] - phase_p[:, :, :, :-1]

    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)

def laplacian_loss_phase_only(spec_true: torch.Tensor,
                              spec_pred: torch.Tensor) -> torch.Tensor:
    _, phase_t = _split_mag_phase(spec_true)
    _, phase_p = _split_mag_phase(spec_pred)

    def lap(x):
        return (-4*x[:, :, 1:-1, 1:-1] +
                x[:, :, :-2, 1:-1] + x[:, :, 2:, 1:-1] +
                x[:, :, 1:-1, :-2] + x[:, :, 1:-1, 2:])

    return F.l1_loss(lap(phase_p), lap(phase_t))

def spectro_time_consistency_loss(orig_wave, recon_spec_norm, sigma, mu,
                                  istft_layer, weight=1.0):
    recon_spec = recon_spec_norm.clone()
    C = recon_spec.shape[1] // 2
    recon_spec[:, :C] = recon_spec[:, :C] * sigma + mu

    wav_rec    = istft_layer(recon_spec, length=orig_wave.shape[1])
    scale      = orig_wave.std(dim=1, keepdim=True) / (wav_rec.std(dim=1, keepdim=True)+1e-8)
    wav_rec = wav_rec * scale
    return weight * F.l1_loss(wav_rec, orig_wave)

def magnitude_l1_loss(spec_true: torch.Tensor,
                      spec_pred: torch.Tensor) -> torch.Tensor:
    """
    L1 distance between magnitudes only (ignores phase).
    spec_*  (B, 2C, F, T)  or (B, F, T, 2C) – handled transparently.
    """
    if spec_true.ndim != 4:
        raise ValueError("Expected 4-D spectrogram tensors")

    # Make sure we are channel-first for our internal helpers
    if spec_true.shape[1] % 2 != 0:         # channel-last → swap
        spec_true = spec_true.permute(0, 3, 1, 2).contiguous()
        spec_pred = spec_pred.permute(0, 3, 1, 2).contiguous()

    mag_t, _ = _split_mag_phase(spec_true)
    mag_p, _ = _split_mag_phase(spec_pred)
    return F.l1_loss(mag_p, mag_t)

def robust_phase_loss(spec_true: torch.Tensor,
                      spec_pred: torch.Tensor,
                      mag_weight: float = 0.6,
                      if_weight:  float = 0.4) -> torch.Tensor:
    """
    Robust phase alignment loss.

    Parameters
    ----------
    spec_true, spec_pred : (B, 2C, F, T)  or  (B, F, T, 2C)
        Spectrograms storing **log–magnitude** and **phase** (radians)
        in alternating channel order:  [log|S|, φ, log|S|, φ, …].

    mag_weight : float
        Weight for the absolute phase distance term.

    if_weight  : float
        Weight for the instantaneous-frequency (time-derivative) term.

    Returns
    -------
    torch.Tensor  (scalar)
        A convex combination of
        • **Amplitude-weighted absolute phase error**  
          \|Δφ\| is wrapped to [-π, π] and scaled by normalised |S| so that
          low-SNR bins contribute less.  
        • **Amplitude-weighted IF error**  
          Δφ̇ along the **time** axis only – empirically more stable than
          spatial gradients on noisy data.
    """
    # 1. split into magnitude / phase
    _, φ_t = _split_mag_phase(spec_true)
    _, φ_p = _split_mag_phase(spec_pred)

    # 2. shortest-path phase difference  Δφ ∈ [-π, π]
    Δφ = torch.atan2(torch.sin(φ_p - φ_t), torch.cos(φ_p - φ_t))

    # 3. magnitude weights  w ∈ [0,1]  (broadcast over phase channels)
    w = (spec_true[:, 0::2]).exp()                # |S|
    w = w / (w.amax(dim=(-2, -1), keepdim=True) + 1e-6)

    # 4.a amplitude-weighted absolute phase
    phase_abs = (w * Δφ.abs()).mean()

    # 4.b instantaneous frequency along **time**
    if_t = φ_t[..., 1:] - φ_t[..., :-1]
    if_p = φ_p[..., 1:] - φ_p[..., :-1]
    if_loss = (w[..., 1:] * (if_p - if_t).abs()).mean()

    return mag_weight * phase_abs + if_weight * if_loss

# --------------Time Domain Losses-------------------
def waveform_l1_loss(wav_true: torch.Tensor,
                     wav_pred: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(wav_pred, wav_true)

def waveform_si_l1_loss(wav_true: torch.Tensor,
                        wav_pred: torch.Tensor,
                        eps: float = 1e-7) -> torch.Tensor:
    """
    Scale-invariant L1 in time domain.
    """
    diff  = torch.mean(torch.abs(wav_true - wav_pred))
    denom = torch.mean(torch.abs(wav_true)) + eps
    return diff / denom

def multi_channel_mrstft_loss(wav_true: torch.Tensor,
                              wav_pred: torch.Tensor,
                              fft_sizes=(256, 512, 1024)) -> torch.Tensor:
    """
    wav_* : (B, T, C)   time-domain windows after ISTFT
    """
    B, T, C = wav_true.shape
    loss = 0.0
    window_fns = {n: torch.hann_window(n, device=wav_true.device) for n in fft_sizes}

    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = window_fns[n_fft]
        for ch in range(C):
            S_t = torch.stft(wav_true[:, :, ch], n_fft, hop_length=hop,
                             window=window, return_complex=True)
            S_p = torch.stft(wav_pred[:, :, ch], n_fft, hop_length=hop,
                             window=window, return_complex=True)
            loss += torch.mean(torch.abs(torch.log1p(S_t.abs()) -
                                         torch.log1p(S_p.abs())))
    return loss / (len(fft_sizes) * C)

# --------------Mask Losses-------------------
def damage_amount_loss(mask_true: torch.Tensor,
                       mask_pred: torch.Tensor,
                       l2_weight: float = 1.0,
                       contrast_weight: float = 0.3,
                       margin: float = 0.05,
                       eps: float = 1e-6) -> torch.Tensor:
    """
    Combines standard L2 loss on average damage amount with a contrastive margin penalty
    to prevent the model from collapsing slightly different damage scenarios into 
    the same prediction.

    Args:
        mask_true: Ground-truth binary damage masks (B, 1, H, W) or (B, H, W)
        mask_pred: Predicted damage masks (B, 1, H, W) or (B, H, W)
        l2_weight: Weight of the MSE loss term (default 1.0)
        contrast_weight: Weight of the margin repulsion term (default 0.3)
        margin: Minimum separation between predicted and true damage averages
        eps: Tolerance below which we consider prediction to be “perfect” (to avoid penalizing it)

    Returns:
        Scalar loss encouraging both accuracy and separation of close-but-different scenarios.
    """
    # Ensure masks are (B, H, W)
    if mask_true.dim() == 4:
        mask_true = mask_true.squeeze(1)
        mask_pred = mask_pred.squeeze(1)

    # Compute average damage per sample
    avg_t = mask_true.mean(dim=(-2, -1))  # (B,)
    avg_p = mask_pred.mean(dim=(-2, -1))  # (B,)

    # 1. Accuracy: L2 loss between predicted and true average damage
    l2 = F.mse_loss(avg_p, avg_t)

    # 2. Contrastive repulsion: penalize if predicted value is too close *but wrong*
    diff = avg_p - avg_t
    is_wrong = (torch.abs(diff) > eps).float()  # Don't penalize perfect matches
    margin_penalty = torch.clamp(margin - torch.abs(diff), min=0.0)
    margin_loss = torch.mean(margin_penalty * is_wrong)

    # Total loss: accuracy + separation
    return l2_weight * l2 + contrast_weight * margin_loss

def custom_mask_loss(y_true: torch.Tensor,
                     y_pred: torch.Tensor,
                     weight_bce: float = 0.3,
                     weight_dice: float = 0.5,
                     weight_focal: float = 0.2,
                     gamma: float = 2.0,
                     alpha: float = 0.3) -> torch.Tensor:
    """
    Weighted BCE + Dice + Focal.  Exactly the same signature you had in
    `diffusion_model.py`, but now lives in utils.
    y_* : (B,1,H,W)   or (B,H,W)
    """
    eps = 1e-7
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
        y_pred = y_pred.unsqueeze(1)

    y_pred = torch.clamp(y_pred, eps, 1.0 - eps)

    # ----- BCE -----
    bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    bce  = torch.mean(bce)

    # ----- Dice -----
    y_true_f = y_true.reshape(y_true.size(0), -1)
    y_pred_f = y_pred.reshape(y_pred.size(0), -1)
    inter    = (y_true_f * y_pred_f).sum(1)
    union    = y_true_f.sum(1) + y_pred_f.sum(1)
    dice     = 1.0 - ((2.*inter + eps) / (union + eps)).mean()

    # ----- Focal -----
    pt    = torch.where(y_true == 1, y_pred, 1 - y_pred)
    focal = -(alpha * (1-pt) ** gamma * torch.log(pt + eps)).mean()

    return weight_bce * bce + weight_dice * dice + weight_focal * focal

def tversky_loss(y_true: torch.Tensor,
                 y_pred: torch.Tensor,
                 alpha: float = 0.3,     # penalise FP
                 beta: float  = 0.7,     # penalise FN
                 eps: float   = 1e-6):
    """
    y_* shape (B,1,H,W) or (B,H,W); values in [0,1]
    """
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
        y_pred = y_pred.unsqueeze(1)

    tp = (y_true * y_pred).sum(dim=(-2, -1))
    fp = ((1 - y_true) * y_pred).sum(dim=(-2, -1))
    fn = (y_true * (1 - y_pred)).sum(dim=(-2, -1))

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - tversky.mean()

def focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.5):
    tv = tversky_loss(y_true, y_pred, alpha, beta, eps=1e-6)
    return tv.pow(gamma)
