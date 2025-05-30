import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import _split_mag_phase
from bridge_data.postprocess import inverse_spectrogram

from .utils import _split_mag_sincos_from_phase as _split_mag_sincos

# ---------- magnitude ------------------------------------------------------
def loss_mag_mse(spec_t, spec_p):
    M_t, _, _ = _split_mag_sincos(spec_t)
    M_p, _, _ = _split_mag_sincos(spec_p)
    return F.mse_loss(M_p, M_t)

def loss_mag_l1(spec_t, spec_p):
    M_t, _, _ = _split_mag_sincos(spec_t)
    M_p, _, _ = _split_mag_sincos(spec_p)
    return F.l1_loss(M_p, M_t)

# ---------- helpers for phase ---------------------------------------------
def _phase(spec):
    _, s, c = _split_mag_sincos(spec)
    return torch.atan2(s, c)

# ---------- phase: absolute dot-distance ----------------------------------
def loss_phase_dot(spec_t, spec_p):
    _, s_t, c_t = _split_mag_sincos(spec_t)
    _, s_p, c_p = _split_mag_sincos(spec_p)
    dot = (s_p * s_t + c_p * c_t).mean()      # ⟨u·v⟩   (u,v are unit vectors)
    return 1.0 - dot                          # = 0 when perfectly aligned

# ---------- phase: instantaneous frequency (time derivative) ---------------
def loss_phase_if(spec_t, spec_p):
    φ_t = _phase(spec_t)
    φ_p = _phase(spec_p)
    dt_t = φ_t[..., 1:] - φ_t[..., :-1]
    dt_p = φ_p[..., 1:] - φ_p[..., :-1]
    return F.l1_loss(dt_p, dt_t)

# ---------- phase: spatial X/Y gradients -----------------------------------
def loss_phase_grad(spec_t, spec_p):
    φ_t = _phase(spec_t)
    φ_p = _phase(spec_p)
    dx_t, dx_p = φ_t[:, :, 1:, :] - φ_t[:, :, :-1, :], φ_p[:, :, 1:, :] - φ_p[:, :, :-1, :]
    dy_t, dy_p = φ_t[:, :, :, 1:] - φ_t[:, :, :, :-1], φ_p[:, :, :, 1:] - φ_p[:, :, :, :-1]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)

# ---------- phase: Laplacian -----------------------------------------------
def loss_phase_lap(spec_t, spec_p):
    φ_t = _phase(spec_t)
    φ_p = _phase(spec_p)
    def lap(x):
        return (-4*x[:, :, 1:-1, 1:-1] +
                x[:, :, :-2, 1:-1] + x[:, :, 2:, 1:-1] +
                x[:, :, 1:-1, :-2] + x[:, :, 1:-1, 2:])
    return F.l1_loss(lap(φ_p), lap(φ_t))

# ---------- phase: amplitude-weighted absolute error -----------------------
def loss_phase_abs_aw(spec_t, spec_p):
    M_t, _, _ = _split_mag_sincos(spec_t)
    φ_t = _phase(spec_t);  φ_p = _phase(spec_p)
    dφ = torch.atan2(torch.sin(φ_p - φ_t), torch.cos(φ_p - φ_t)).abs()
    w  = M_t.exp()                                   # undo log
    w  = w / (w.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    return (w * dφ).mean()

# ===========================================================================
#  T I M E - D O M A I N   L O S S E S
# ===========================================================================

def loss_wave_l1(w_t, w_p):
    return F.l1_loss(w_p, w_t)

def loss_wave_mrstft(w_t, w_p, fft_sizes=(256, 512, 1024)):
    """
    Multi-resolution STFT log-magnitude distance (channel-wise).
    w_* : (B, T, C)
    """
    _, _, C = w_t.shape
    loss = 0.0
    windows = {n: torch.hann_window(n, device=w_t.device) for n in fft_sizes}
    for n_fft in fft_sizes:
        hop = n_fft // 4
        win = windows[n_fft]
        for ch in range(C):
            St = torch.stft(w_t[:, :, ch], n_fft, hop_length=hop,
                            window=win, return_complex=True)
            Sp = torch.stft(w_p[:, :, ch], n_fft, hop_length=hop,
                            window=win, return_complex=True)
            loss += torch.mean(torch.abs(torch.log1p(St.abs()) -
                                         torch.log1p(Sp.abs())))
    return loss / (len(fft_sizes) * C)

# ===========================================================================
#  S P E C –>  W A V   C O N S I S T E N C Y
# ===========================================================================

def loss_spectro_time_consistency(wav_ref,
                                  spec_pred_norm,
                                  sigma, mu,
                                  istft_layer):
    """
    • Denormalises **only** log-magnitude  
    • Runs differentiable ISTFT  
    • Returns scale-invariant L1 in the time domain
    """
    spec_dn = spec_pred_norm.clone()
    C = spec_dn.shape[1] // 3
    spec_dn[:, :C] = spec_dn[:, :C] * sigma + mu
    wav_rec = istft_layer(spec_dn, length=wav_ref.size(1))

    scale = wav_ref.std(dim=1, keepdim=True) / (wav_rec.std(dim=1, keepdim=True) + 1e-8)
    wav_rec = wav_rec * scale
    return F.l1_loss(wav_rec, wav_ref)
# ────────────────────────────────────────────────────────────────────────────

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
