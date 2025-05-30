# ─────────────────────────── diff_pt/train.py ────────────────────────────
import os, numpy as np, torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ──────────────────────────────────────────────────────────────────────────
#  Atomic losses
# ──────────────────────────────────────────────────────────────────────────
from diff_pt.losses import (
    loss_mag_mse,              # |S|²  MSE
    loss_phase_dot,            # 1−cos  (unit-vec dot)
    loss_phase_if,             # IF  (Δφ_t)
    loss_phase_abs_aw,         # |Δφ|  amplitude-weighted
    loss_wave_l1,              # time-L1
    loss_spectro_time_consistency,
    damage_amount_loss,
    custom_mask_loss,
    focal_tversky_loss,
)

# (keep MR-STFT or grad / lap losses if you later need them)
# from diff_pt.losses import loss_phase_grad, loss_phase_lap, loss_wave_mrstft

from diff_pt.utils import _quick_mask_stats, scheduler

# ──────────────────────────────────────────────────────────────────────────
#  GLOBAL HYPER-PARAMETERS
# ──────────────────────────────────────────────────────────────────────────
# ––– Spectrogram AE –––
MAG_MSE_W          = 1.0
PHASE_DOT_W        = 1.0
PHASE_IF_W         = 0.5
PHASE_AW_ABS_W     = 0.0         # set >0 if you want the weighted |Δφ|
MAG_CURRIC_MIN_W   = 0.3
MAG_CURRIC_MAX_W   = 1.5
TIME_CONSIST_W     = 0.5
WAVE_L1_W          = 2.0

# ––– Mask AE –––
MASK_PX_W          = 0.0
DAMAGE_W_INITIAL   = 2.0
DAMAGE_W_FINAL     = 0.3
FOCAL_GAMMA_INIT   = 1.0
FOCAL_GAMMA_LATE   = 1.5

# ──────────────────────────────────────────────────────────────────────────
#  Helper schedules
# ──────────────────────────────────────────────────────────────────────────
def phase_curriculum(epoch: int) -> float:
    """Linear decay MAG_CURRIC_MAX_W → MAG_CURRIC_MIN_W in first 80 epochs."""
    return scheduler(0, 80, epoch, MAG_CURRIC_MIN_W, MAG_CURRIC_MAX_W)

# ──────────────────────────────────────────────────────────────────────────
#  Main training routine
# ──────────────────────────────────────────────────────────────────────────
def train_autoencoders(
    spec_autoencoder,
    mask_autoencoder,
    train_loader,
    val_loader,
    device,
    epochs:   int = 300,
    lr:       float = 5e-4,
    patience: int = 50,
    cache_dir: str = "cache",
):
    """
    Joint training of spectrogram- and mask-autoencoder.
    Returns *two* history dicts for visualisation.
    """
    # ---------------------------------------------------------------- setup
    spec_autoencoder.to(device)
    mask_autoencoder.to(device)

    opt_spec = optim.AdamW(spec_autoencoder.parameters(), lr=lr)
    opt_mask = optim.AdamW(mask_autoencoder.parameters(), lr=lr)

    sch_spec = ReduceLROnPlateau(opt_spec, mode="min",
                                 factor=0.5, patience=8, min_lr=5e-6)
    sch_mask = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mask, T_max=epochs)

    best_spec, best_mask, patience_ctr = float("inf"), float("inf"), 0
    hist = {"spec_train": [], "spec_val": [],
            "mask_train": [], "mask_val": []}

    # μ,σ saved by data-pipeline  (used for ISTFT consistency)
    norm = np.load(os.path.join(cache_dir, "spec_norm_magonly.npz"))
    MU_SPEC = torch.as_tensor(norm["mu"], device=device)
    SIG_SPEC = torch.as_tensor(norm["sigma"], device=device)

    # ───────────────────────────── training loop ──────────────────────────
    for epoch in range(epochs):
        spec_autoencoder.train();  mask_autoencoder.train()

        # —— dynamic mask weights ——
        damage_w    = DAMAGE_W_FINAL  if epoch >= 200 else DAMAGE_W_INITIAL
        focal_gamma = FOCAL_GAMMA_LATE if epoch >= 250 else FOCAL_GAMMA_INIT
        contrast_w  = min(0.3, epoch / 100 * 0.3)
        w_bce, w_dice, w_focal = (0.4, 0.5, 0.1) if epoch < 200 else (0.1, 0.8, 0.1)

        sum_spec, sum_mask, n_batches = 0.0, 0.0, 0

        # ─────────────── mini-batch loop ───────────────
        for spec_b, mask_b, seg_b, *_ in train_loader:
            n_batches += 1
            spec_b = spec_b.to(device); mask_b = mask_b.to(device)
            seg_b  = seg_b .to(device)

            # ——— S P E C ———
            opt_spec.zero_grad()
            recon_spec, _ = spec_autoencoder(spec_b)

            w_curr = phase_curriculum(epoch)  # curriculum factor

            loss_mag   = MAG_MSE_W   * loss_mag_mse(spec_b, recon_spec)
            loss_p_dot = PHASE_DOT_W * loss_phase_dot(spec_b, recon_spec) * w_curr
            loss_p_if  = PHASE_IF_W  * loss_phase_if (spec_b, recon_spec) * w_curr
            loss_p_aw  = PHASE_AW_ABS_W * loss_phase_abs_aw(spec_b, recon_spec)

            istft_layer = getattr(spec_autoencoder, "istft", None)
            if istft_layer is not None:
                loss_time = TIME_CONSIST_W * loss_spectro_time_consistency(
                    seg_b, recon_spec, SIG_SPEC, MU_SPEC, istft_layer)
                # Wave L1 (scale-invariant via the helper)
                recon_dn = recon_spec.clone()
                C = recon_dn.shape[1] // 3
                recon_dn[:, :C] = recon_dn[:, :C] * SIG_SPEC + MU_SPEC
                wav_rec = istft_layer(recon_dn, length=seg_b.size(1))
                loss_wave = WAVE_L1_W * loss_wave_l1(seg_b, wav_rec)
            else:
                loss_time = loss_wave = torch.tensor(0.0, device=device)

            spec_total = (loss_mag + loss_p_dot + loss_p_if +
                          loss_p_aw + loss_time + loss_wave)
            spec_total.backward();  opt_spec.step()

            # ——— M A S K ———
            opt_mask.zero_grad()
            recon_mask, _ = mask_autoencoder(mask_b)

            if n_batches == 1:
                _quick_mask_stats(mask_b,
                                  torch.logit(recon_mask.clamp(1e-4, 1-1e-4)),
                                  recon_mask)

            loss_mask_px = focal_tversky_loss(mask_b, recon_mask,
                                              alpha=0.3, beta=0.8, gamma=focal_gamma)
            loss_damage  = damage_amount_loss(mask_b, recon_mask,
                                              contrast_weight=contrast_w, margin=0.005)
            loss_dice    = custom_mask_loss(mask_b, recon_mask,
                                            weight_bce=w_bce, weight_dice=w_dice,
                                            weight_focal=w_focal)
            mask_total = MASK_PX_W * loss_mask_px + damage_w * loss_damage + loss_dice
            mask_total.backward();  opt_mask.step()

            sum_spec += spec_total.item();  sum_mask += mask_total.item()

        # —— epoch summary ——
        train_spec_loss = sum_spec  / n_batches
        train_mask_loss = sum_mask  / n_batches
        print(f"[Ep {epoch+1:3d}/{epochs}] spec={train_spec_loss:.4f}  "
              f"mask={train_mask_loss:.4f}")

        # ─────────────── validation ───────────────
        spec_autoencoder.eval(); mask_autoencoder.eval()
        val_spec_losses, val_mask_losses = [], []

        with torch.no_grad():
            for spec_v, mask_v, seg_v, *_ in val_loader:
                spec_v = spec_v.to(device); mask_v = mask_v.to(device); seg_v = seg_v.to(device)
                recon_v, _ = spec_autoencoder(spec_v)

                # —— spec losses ——
                loss_mag   = MAG_MSE_W   * loss_mag_mse(spec_v, recon_v)
                loss_p_dot = PHASE_DOT_W * loss_phase_dot(spec_v, recon_v) * w_curr
                loss_p_if  = PHASE_IF_W  * loss_phase_if (spec_v, recon_v) * w_curr
                loss_p_aw  = PHASE_AW_ABS_W * loss_phase_abs_aw(spec_v, recon_v)

                if istft_layer is not None:
                    loss_time = TIME_CONSIST_W * loss_spectro_time_consistency(
                        seg_v, recon_v, SIG_SPEC, MU_SPEC, istft_layer)
                    recon_dn = recon_v.clone()
                    C = recon_dn.shape[1] // 3
                    recon_dn[:, :C] = recon_dn[:, :C] * SIG_SPEC + MU_SPEC
                    wav_rec = istft_layer(recon_dn, length=seg_b.size(1))
                    loss_wave = WAVE_L1_W * loss_wave_l1(seg_v, wav_rec)
                else:
                    loss_time = loss_wave = torch.tensor(0.0, device=device)

                val_spec_total = (loss_mag + loss_p_dot + loss_p_if +
                                  loss_p_aw + loss_time + loss_wave)
                val_spec_losses.append(val_spec_total.item())

                # —— mask losses ——
                recon_mask_v, _ = mask_autoencoder(mask_v)
                loss_mask_px = focal_tversky_loss(mask_v, recon_mask_v,
                                                  alpha=0.3, beta=0.8, gamma=focal_gamma)
                loss_damage  = damage_amount_loss(mask_v, recon_mask_v,
                                                  contrast_weight=contrast_w, margin=0.005)
                loss_dice    = custom_mask_loss(mask_v, recon_mask_v,
                                                weight_bce=w_bce, weight_dice=w_dice,
                                                weight_focal=w_focal)

                val_mask_total = MASK_PX_W * loss_mask_px + damage_w * loss_damage + loss_dice
                val_mask_losses.append(val_mask_total.item())

        val_spec_loss = float(np.mean(val_spec_losses))
        val_mask_loss = float(np.mean(val_mask_losses))
        print(f"     ↳ val  spec={val_spec_loss:.4f}  mask={val_mask_loss:.4f}")

        # —— log history ——
        hist["spec_train"].append(train_spec_loss); hist["spec_val"].append(val_spec_loss)
        hist["mask_train"].append(train_mask_loss); hist["mask_val"].append(val_mask_loss)

        # —— schedulers ——
        sch_spec.step(val_spec_loss); sch_mask.step()

        # —— checkpoints / early stop ——
        EPS = 1e-4
        improved_spec = val_spec_loss < best_spec - EPS
        improved_mask = val_mask_loss < best_mask - EPS

        if improved_spec:
            best_spec = val_spec_loss
            torch.save(spec_autoencoder.state_dict(),
                       "results_diff/autoencoders/spec_autoencoder_best.pt")
            print(f"     ✅ new best • spec-AE {best_spec:.4f}")

        if improved_mask:
            best_mask = val_mask_loss
            torch.save(mask_autoencoder.state_dict(),
                       "results_diff/autoencoders/mask_autoencoder_best.pt")
            print(f"     ✅ new best • mask-AE {best_mask:.4f}")

        patience_ctr = 0 if (improved_spec or improved_mask) else patience_ctr + 1
        if patience_ctr >= patience:
            print(f"⏹ Early stop @ epoch {epoch+1} – no improv for {patience} ep.")
            break

        # ─────────── W&B hook (optional) ───────────
        # if wandb.run is not None:
        #     wandb.log({
        #         "train/spec": train_spec_loss,
        #         "train/mask": train_mask_loss,
        #         "val/spec":   val_spec_loss,
        #         "val/mask":   val_mask_loss,
        #         "lr/spec":    sch_spec.optimizer.param_groups[0]["lr"],
        #         "lr/mask":    sch_mask.get_last_lr()[0],
        #     }, step=epoch)

    # —— return history dicts ——
    spec_hist = {"train_loss": hist["spec_train"], "val_loss": hist["spec_val"]}
    mask_hist = {"train_loss": hist["mask_train"], "val_loss": hist["mask_val"]}
    return spec_hist, mask_hist
# ──────────────────────────────────────────────────────────────────────────
