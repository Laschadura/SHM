import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ────────────────────────────────────────────────────────────────────────────
#  Losses & helpers
# ────────────────────────────────────────────────────────────────────────────
from diff_pt.losses import (
    complex_spectrogram_loss,
    robust_phase_loss,
    magnitude_l1_loss,
    waveform_l1_loss,
    spectro_time_consistency_loss,
    damage_amount_loss,
    custom_mask_loss,
    focal_tversky_loss,
)

from diff_pt.utils import _quick_mask_stats, scheduler

# ╔══════════════════════════════════════╗
# ║  🔧  ONE‑PLACE GLOBAL HYPER‑PARAMS  ║
# ╚══════════════════════════════════════╝
# Tune ↕ or set to **0** to disable a term completely – train *and* val.

# ── Spectrogram auto‑encoder ────────────────────────────────────────────────
COMPLEX_W         = 1.0   # main |S|+phase composite loss
PHASE_ABS_W       = 1.0   # |Δφ|  inside robust_phase_loss
IF_W              = 0.5   # inst‑freq term  inside robust_phase_loss
PHASE_CURRIC_MIN  = 0.3   # ↓ weight after curriculum
PHASE_CURRIC_MAX  = 1.5   # ↑ early weight
MAG_L1_W          = 0.5
TIME_CONSIST_W    = 0.5
WAVE_L1_W         = 2.0

# ── Mask   auto‑encoder ─────────────────────────────────────────────────────
MASK_PX_W         = 0.0   # pixel‑wise focal‑Tversky (0 ⇒ skip)
DAMAGE_W_INITIAL  = 2.0   # < 200 epochs
DAMAGE_W_FINAL    = 0.3   # ≥ 200 epochs
FOCAL_GAMMA_INIT  = 1.0
FOCAL_GAMMA_LATE  = 1.5   # ≥ 250 epochs

# ────────────────────────────────────────────────────────────────────────────
#  Helper schedules
# ────────────────────────────────────────────────────────────────────────────

def phase_curriculum(epoch: int) -> float:
    """Linear decay **PHASE_CURRIC_MAX → PHASE_CURRIC_MIN** over first 80 epochs."""
    return scheduler(0, 80, epoch, PHASE_CURRIC_MIN, PHASE_CURRIC_MAX)

# ────────────────────────────────────────────────────────────────────────────
#  Main training routine
# ────────────────────────────────────────────────────────────────────────────

def train_autoencoders(
    spec_autoencoder,
    mask_autoencoder,
    train_loader,
    val_loader,
    device,
    epochs: int = 300,
    lr: float = 5e-4,
    patience: int = 50,
    cache_dir: str = "cache",
):
    """Joint loop that trains *two* deterministic auto‑encoders (spectrogram
    + mask) side‑by‑side.  We keep their optimisers, schedulers, early‑stop &
    best‑ckpt logic independent, yet share progress bars / epoch counters.
    Returns two history dicts for downstream Plotly visualisation.
    """

    # ------------------------------------------------------------------ setup
    spec_autoencoder.to(device)
    mask_autoencoder.to(device)

    opt_spec = optim.AdamW(spec_autoencoder.parameters(), lr=lr)
    opt_mask = optim.AdamW(mask_autoencoder.parameters(), lr=lr)

    sch_spec = ReduceLROnPlateau(           # ⇠ NEW
            opt_spec,
            mode="min",
            factor=0.5,        # halve LR when we plateau
            patience=8,        # “plateau” = no spec-val improvement for 8 epochs
            min_lr=5e-6)

    sch_mask = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mask, T_max=epochs)

    best_spec, best_mask = float("inf"), float("inf")
    patience_ctr = 0

    history = {
        "spec_train": [], "spec_val": [],
        "mask_train": [], "mask_val": [],
    }

    # μ,σ saved by data‑pipeline (used for ISTFT consistency)
    norm = np.load(os.path.join(cache_dir, "spec_norm_magonly.npz"))
    MU_SPEC = torch.as_tensor(norm["mu"], device=device)
    SIG_SPEC = torch.as_tensor(norm["sigma"], device=device)

    # ─────────────────────────────── training loop ──────────────────────────
    for epoch in range(epochs):
        spec_autoencoder.train(); mask_autoencoder.train()

        # Dynamic weight tweaks (mask branch)
        damage_w   = DAMAGE_W_FINAL if epoch >= 200 else DAMAGE_W_INITIAL
        focal_gamma = FOCAL_GAMMA_LATE if epoch >= 250 else FOCAL_GAMMA_INIT
        contrast_w  = min(0.3, epoch / 100 * 0.3)

        # classic BCE / Dice / Focal anneal
        if epoch < 200:
            w_bce, w_dice, w_focal = 0.4, 0.5, 0.1
        else:
            w_bce, w_dice, w_focal = 0.1, 0.8, 0.1

        sum_spec, sum_mask, n_batches = 0.0, 0.0, 0

        # ───────────────────────── mini‑batch loop ──────────────────────────
        for spec_b, mask_b, seg_b, *_ in train_loader:
            n_batches += 1
            spec_b = spec_b.to(device); mask_b = mask_b.to(device); seg_b = seg_b.to(device)

            # ===== S P E C  branch ======================================
            opt_spec.zero_grad()
            recon_spec, _ = spec_autoencoder(spec_b)

            loss_complex = COMPLEX_W * complex_spectrogram_loss(spec_b, recon_spec)
            loss_phase   = PHASE_ABS_W * robust_phase_loss(
                spec_b, recon_spec, mag_weight=1.0, if_weight=IF_W
            ) * phase_curriculum(epoch)
            loss_mag     = MAG_L1_W * magnitude_l1_loss(spec_b, recon_spec)

            istft_layer = getattr(spec_autoencoder, "istft", None)
            if istft_layer is not None:
                loss_time = TIME_CONSIST_W * spectro_time_consistency_loss(
                    seg_b, recon_spec, SIG_SPEC, MU_SPEC, istft_layer)
                recon_spec_dn = recon_spec.clone()
                C = recon_spec_dn.shape[1] // 2
                recon_spec_dn[:, :C] = recon_spec_dn[:, :C] * SIG_SPEC + MU_SPEC
                wav_rec = istft_layer(recon_spec_dn, length=seg_b.size(1))

                loss_wave = WAVE_L1_W * waveform_l1_loss(seg_b, wav_rec)
            else:
                loss_time = torch.tensor(0., device=device); loss_wave = loss_time

            spec_total = loss_complex + loss_phase + loss_mag + loss_time + loss_wave
            spec_total.backward(); opt_spec.step()

            # ===== M A S K  branch ======================================
            opt_mask.zero_grad()
            recon_mask, _ = mask_autoencoder(mask_b)

            if n_batches == 1:
                _quick_mask_stats(mask_b,
                                   torch.logit(recon_mask.clamp(1e-4, 1-1e-4)),
                                   recon_mask)

            loss_mask_px = focal_tversky_loss(mask_b, recon_mask,
                                              alpha=0.3, beta=0.8,
                                              gamma=focal_gamma)
            loss_damage  = damage_amount_loss(mask_b, recon_mask,
                                               contrast_weight=contrast_w,
                                               margin=0.005)
            loss_dice    = custom_mask_loss(mask_b, recon_mask,
                                             weight_bce=w_bce, weight_dice=w_dice,
                                             weight_focal=w_focal)
            mask_total = (
                MASK_PX_W * loss_mask_px +
                damage_w  * loss_damage  +
                loss_dice
            )
            mask_total.backward(); opt_mask.step()

            sum_spec += spec_total.item(); sum_mask += mask_total.item()

        # —— epoch summary ———————————————————————————————
        train_spec_loss = sum_spec / n_batches
        train_mask_loss = sum_mask / n_batches
        print(f"[Epoch {epoch+1:3d}/{epochs}]  spec={train_spec_loss:.4f}  mask={train_mask_loss:.4f}")

        # ================= V A L =========================
        spec_autoencoder.eval(); mask_autoencoder.eval()
        val_spec_losses, val_mask_losses = [], []

        with torch.no_grad():
            for spec_v, mask_v, seg_v, *_ in val_loader:
                spec_v = spec_v.to(device); mask_v = mask_v.to(device); seg_v = seg_v.to(device)

                recon_spec_v, _ = spec_autoencoder(spec_v)

                loss_complex = COMPLEX_W * complex_spectrogram_loss(spec_v, recon_spec_v)
                loss_phase   = PHASE_ABS_W * robust_phase_loss(
                    spec_v, recon_spec_v, mag_weight=1.0, if_weight=IF_W
                ) * phase_curriculum(epoch)
                loss_mag     = MAG_L1_W * magnitude_l1_loss(spec_v, recon_spec_v)

                istft_layer = getattr(spec_autoencoder, "istft", None)
                if istft_layer is not None:
                    loss_time = TIME_CONSIST_W * spectro_time_consistency_loss(
                        seg_v, recon_spec_v, SIG_SPEC, MU_SPEC, istft_layer)
                    recon_spec_dn = recon_spec_v.clone()
                    C = recon_spec_dn.shape[1] // 2
                    recon_spec_dn[:, :C] = recon_spec_dn[:, :C] * SIG_SPEC + MU_SPEC
                    wav_rec = istft_layer(recon_spec_dn, length=seg_b.size(1))
                    loss_wave = WAVE_L1_W * waveform_l1_loss(seg_v, wav_rec)
                else:
                    loss_time = torch.tensor(0., device=device); loss_wave = loss_time

                val_spec_total = loss_complex + loss_phase + loss_mag + loss_time + loss_wave
                val_spec_losses.append(val_spec_total.item())

                recon_mask_v, _ = mask_autoencoder(mask_v)
                loss_mask_px = focal_tversky_loss(mask_v, recon_mask_v,
                                                  alpha=0.3, beta=0.8,
                                                  gamma=focal_gamma)
                loss_damage  = damage_amount_loss(mask_v, recon_mask_v,
                                                   contrast_weight=contrast_w,
                                                   margin=0.005)
                loss_dice    = custom_mask_loss(mask_v, recon_mask_v,
                                                 weight_bce=w_bce, weight_dice=w_dice,
                                                 weight_focal=w_focal)
                val_mask_total = (
                    MASK_PX_W * loss_mask_px +
                    damage_w  * loss_damage  +
                    loss_dice
                )
                val_mask_losses.append(val_mask_total.item())

        val_spec_loss = float(np.mean(val_spec_losses))
        val_mask_loss = float(np.mean(val_mask_losses))
        print(f"     ↳ val  spec={val_spec_loss:.4f}  mask={val_mask_loss:.4f}")

        # —— record history ——
        history["spec_train"].append(train_spec_loss); history["spec_val"].append(val_spec_loss)
        history["mask_train"].append(train_mask_loss); history["mask_val"].append(val_mask_loss)

        # —— schedulers ——
        sch_spec.step(val_spec_loss)
        sch_mask.step()

        # —— best‑model checkpoints & early‑stop ——
        EPS = 1e-4
        improved_spec = val_spec_loss < best_spec - EPS
        improved_mask = val_mask_loss < best_mask - EPS

        if improved_spec:
            best_spec = val_spec_loss
            torch.save(spec_autoencoder.state_dict(),
                       "results_diff/autoencoders/spec_autoencoder_best.pt")
            print(f"     ✅ new best • spec‑AE {best_spec:.4f}")

        if improved_mask:
            best_mask = val_mask_loss
            torch.save(mask_autoencoder.state_dict(),
                       "results_diff/autoencoders/mask_autoencoder_best.pt")
            print(f"     ✅ new best • mask‑AE {best_mask:.4f}")

        if improved_spec or improved_mask:
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"⏹️  Early stop @ epoch {epoch+1} – no improv for {patience} ep.")
                break

    # Convert to the two‑dict structure the rest of the pipeline expects
    spec_hist = {"train_loss": history["spec_train"], "val_loss": history["spec_val"]}
    mask_hist = {"train_loss": history["mask_train"], "val_loss": history["mask_val"]}
    return spec_hist, mask_hist
