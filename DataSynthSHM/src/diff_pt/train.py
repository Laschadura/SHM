import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

# ─────────────────────────────────
#  Atomic losses
# ────────────────────────────────
from diff_pt.losses import (
    loss_mag_mse,
    loss_phase_dot,
    loss_phase_if,
    loss_phase_abs_aw,
    loss_wave_l1,
    loss_spectro_time_consistency,
    damage_amount_loss,
    custom_mask_loss,
    focal_tversky_loss,
)

from diff_pt.utils import _quick_mask_stats, scheduler, split_mag_phase

from omegaconf import DictConfig


# -----------------------------------------------------------------------------
# Helper to gradually increase phase‑related weights
# -----------------------------------------------------------------------------

def phase_curriculum(epoch: int, cfg) -> float:
    return scheduler(0, 80, epoch, cfg.mag_curric_min, cfg.mag_curric_max)


# -----------------------------------------------------------------------------
#  Main training routine
# -----------------------------------------------------------------------------

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
    loss_cfg: DictConfig | dict = None,
    ):
    """Joint training of spectrogram & mask auto‑encoders with adaptive losses."""

    w = loss_cfg      # short alias for loss weights
    # --- setup ---------------------------------------------------------------
    spec_autoencoder.to(device)
    mask_autoencoder.to(device)

    opt_spec = optim.AdamW(spec_autoencoder.parameters(), lr=lr)
    opt_mask = optim.AdamW(mask_autoencoder.parameters(), lr=lr)

    sch_spec = ReduceLROnPlateau(opt_spec, mode="min", factor=0.5, patience=8, min_lr=5e-6)
    sch_mask = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mask, T_max=epochs, eta_min=1e-4)

    best_spec, best_mask, patience_ctr = float("inf"), float("inf"), 0
    hist = {"spec_train": [], "spec_val": [], "mask_train": [], "mask_val": []}

    # global μ,σ for ISTFT consistency loss
    norm = np.load(os.path.join(cache_dir, "spec_norm_magonly.npz"))
    MU_SPEC = torch.as_tensor(norm["mu"], device=device)
    SIG_SPEC = torch.as_tensor(norm["sigma"], device=device)

    # ======================================================================
    #  EPOCH LOOP
    # ======================================================================
    for epoch in range(epochs):
        spec_autoencoder.train()
        mask_autoencoder.train()

        # --- dynamic mask‑loss schedule ------------------------------------
        damage_w    = w.damage_final      if epoch >= 200 else w.damage_initial
        focal_gamma = w.focal_gamma_late  if epoch >= 250 else w.focal_gamma_init
        contrast_w  = min(0.3, epoch / 100 * 0.3)
        w_bce, w_dice, w_focal = (0.4, 0.5, 0.1) if epoch < 200 else (0.1, 0.8, 0.1)

        # running sums for logging
        n_batches = 0
        loss_acc = {
            "mag_mse": 0.0,
            "phase_dot": 0.0,
            "phase_if": 0.0,
            "phase_aw": 0.0,
            "time_cons": 0.0,
            "wave_l1": 0.0,
            "spec_total": 0.0,
            "mask_total": 0.0,
            "mask_px":     0.0,
            "mask_damage": 0.0,
            "mask_dice":   0.0,
        }

        # -----------------------------------------------------------------
        #  MINI‑BATCH TRAIN LOOP
        # -----------------------------------------------------------------
        for spec_b, mask_b, seg_b, *_ in train_loader:
            n_batches += 1
            spec_b = spec_b.to(device)
            mask_b = mask_b.to(device)
            seg_b  = seg_b.to(device)

            # ===== S P E C  branch ======================================
            opt_spec.zero_grad()
            recon_spec, _ = spec_autoencoder(spec_b)
            w_curr = phase_curriculum(epoch, w)

            # --------  split --------
            mag_t, phi_t = split_mag_phase(spec_b)          # targets
            mag_p, phi_p = split_mag_phase(recon_spec)      # predictions

            loss_mag   = loss_mag_mse(mag_p, mag_t)
            loss_p_dot = loss_phase_dot(phi_t, phi_p) * w_curr 
            loss_p_if  = loss_phase_if (phi_t, phi_p) * w_curr 
            loss_p_aw  = loss_phase_abs_aw(phi_t, phi_p)

            istft_layer = getattr(spec_autoencoder, "istft", None)
            if istft_layer is not None:
                loss_time = (
                    loss_spectro_time_consistency(seg_b, recon_spec, SIG_SPEC, MU_SPEC, istft_layer)
                    if w.time_consistency > 0 else torch.tensor(0.0, device=device)
                )
                # wave‑domain L1 (scale‑invariant)
                recon_dn = recon_spec.clone()
                C = recon_dn.shape[1] // 3
                recon_dn[:, :C] = recon_dn[:, :C] * SIG_SPEC + MU_SPEC
                wav_rec = istft_layer(recon_dn, length=seg_b.size(1))
                loss_wave = loss_wave_l1(seg_b, wav_rec) if w.wave_l1 > 0 else torch.tensor(0.0, device=device)
            else:
                loss_time = loss_wave = torch.tensor(0.0, device=device)

            spec_total = (
                w.mag_mse      * loss_mag   +
                w.phase_dot    * loss_p_dot +
                w.phase_if     * loss_p_if  +
                w.phase_aw_abs * loss_p_aw  +
                w.time_consistency * loss_time +
                w.wave_l1      * loss_wave
            )
            spec_total.backward()
            opt_spec.step()

            # ===== M A S K  branch ======================================
            opt_mask.zero_grad()
            recon_mask, _ = mask_autoencoder(mask_b)

            if n_batches == 1:  # quick diagnostic once per epoch
                _quick_mask_stats(mask_b, torch.logit(recon_mask.clamp(1e-4, 1-1e-4)), recon_mask)

            loss_mask_px = focal_tversky_loss(mask_b, recon_mask, alpha=0.3, beta=0.8, gamma=focal_gamma)
            loss_damage  = damage_amount_loss(mask_b, recon_mask, contrast_weight=contrast_w, margin=0.005)
            loss_dice    = custom_mask_loss(mask_b, recon_mask, weight_bce=w_bce, weight_dice=w_dice, weight_focal=w_focal)

            mask_total = w.mask_px * loss_mask_px + damage_w * loss_damage + w.dice_w * loss_dice
            mask_total.backward()
            opt_mask.step()

            # accumulate losses
            loss_acc["mag_mse"]     += loss_mag.item()
            loss_acc["phase_dot"]   += loss_p_dot.item()
            loss_acc["phase_if"]    += loss_p_if.item()
            loss_acc["phase_aw"]    += loss_p_aw.item()
            loss_acc["time_cons"]   += loss_time.item()
            loss_acc["wave_l1"]     += loss_wave.item()
            loss_acc["spec_total"]  += spec_total.item()
            loss_acc["mask_total"]  += mask_total.item()
            loss_acc["mask_px"]     += loss_mask_px.item()
            loss_acc["mask_damage"] += loss_damage.item()
            loss_acc["mask_dice"]   += loss_dice.item()


        # === epoch summary (train) =========================================
        # average losses
        for k in loss_acc:
            loss_acc[k] /= n_batches
        train_spec_loss = loss_acc["spec_total"]
        train_mask_loss = loss_acc["mask_total"]

        # -----------------------------------------------------------------
        #  VALIDATION LOOP
        # -----------------------------------------------------------------
        spec_autoencoder.eval()
        mask_autoencoder.eval()
        val_spec_losses, val_mask_losses = [], []

        with torch.no_grad():
            for spec_v, mask_v, seg_v, *_ in val_loader:
                spec_v = spec_v.to(device)
                mask_v = mask_v.to(device)
                seg_v  = seg_v.to(device)

                recon_v, _ = spec_autoencoder(spec_v)

                # --------  split --------
                mag_t, phi_t = split_mag_phase(spec_v)          # targets
                mag_p, phi_p = split_mag_phase(recon_v)      # predictions

                loss_mag   = loss_mag_mse(mag_p, mag_t)
                loss_p_dot = loss_phase_dot(phi_t, phi_p) * w_curr
                loss_p_if  = loss_phase_if (phi_t, phi_p) * w_curr
                loss_p_aw  = loss_phase_abs_aw(phi_t, phi_p)

                if istft_layer is not None:
                    loss_time = (
                        loss_spectro_time_consistency(seg_v, recon_v, SIG_SPEC, MU_SPEC, istft_layer)
                        if w.time_consistency > 0 else torch.tensor(0.0, device=device)
                    )
                    recon_dn = recon_v.clone()
                    C = recon_dn.shape[1] // 3
                    recon_dn[:, :C] = recon_dn[:, :C] * SIG_SPEC + MU_SPEC
                    wav_rec = istft_layer(recon_dn, length=seg_v.size(1))
                    loss_wave = loss_wave_l1(seg_v, wav_rec) if w.wave_l1 > 0 else torch.tensor(0.0, device=device)
                else:
                    loss_time = loss_wave = torch.tensor(0.0, device=device)

                val_spec_total = (
                    w.mag_mse      * loss_mag   +
                    w.phase_dot    * loss_p_dot +
                    w.phase_if     * loss_p_if  +
                    w.phase_aw_abs * loss_p_aw  +
                    w.time_consistency * loss_time +
                    w.wave_l1      * loss_wave
                )
                val_spec_losses.append({
                    "spec_total": val_spec_total.item(),
                    "mag_mse":   loss_mag.item(),
                    "phase_dot": loss_p_dot.item(),
                    "phase_if":  loss_p_if.item(),
                    "phase_aw":  loss_p_aw.item(),
                    "time_cons": loss_time.item(),
                    "wave_l1":   loss_wave.item(),
                })

                # ---- mask ----
                recon_mask_v, _ = mask_autoencoder(mask_v)
                loss_mask_px = focal_tversky_loss(mask_v, recon_mask_v, alpha=0.3, beta=0.8, gamma=focal_gamma)
                loss_damage  = damage_amount_loss(mask_v, recon_mask_v, contrast_weight=contrast_w, margin=0.005)
                loss_dice    = custom_mask_loss(mask_v, recon_mask_v, weight_bce=w_bce, weight_dice=w_dice, weight_focal=w_focal)

                val_mask_total = w.mask_px * loss_mask_px + damage_w * loss_damage + w.dice_w * loss_dice
                val_mask_losses.append({
                    "mask_total":  val_mask_total.item(),
                    "mask_px":     loss_mask_px.item(),
                    "mask_damage": loss_damage.item(),
                    "mask_dice":   loss_dice.item(),
                })

        # convert list-of-dicts → dict-of-means
        val_means = {k: float(np.mean([d[k] for d in val_spec_losses])) for k in val_spec_losses[0]}
        val_spec_loss = val_means["spec_total"]

        if val_mask_losses:     # avoid KeyError if loader is empty
            val_mask_means = {k: float(np.mean([d[k] for d in val_mask_losses]))
                              for k in val_mask_losses[0]}
        else:
            val_mask_means = {k: 0.0 for k in ["mask_total", "mask_px", "mask_damage", "mask_dice"]}
        val_mask_loss  = val_mask_means["mask_total"]


        # record history
        hist["spec_train"].append(train_spec_loss)
        hist["spec_val"].append(val_spec_loss)
        hist["mask_train"].append(train_mask_loss)
        hist["mask_val"].append(val_mask_loss)

        if epoch % 1 == 0:          # every epoch
            print(
                f"[AE] Epoch {epoch+1:03d} | "
                f"train_spec: {train_spec_loss:.4f} | "
                f"train_mask: {train_mask_loss:.4f} | "
                f"val_spec: {val_spec_loss:.4f} | "
                f"val_mask: {val_mask_loss:.4f}",
                flush=True,
            )

        LOG_EVERY = 100  # log every N epochs

        # ───────── WANDB logging (throttled) ────────
        if (epoch % LOG_EVERY == 0) or (epoch == epochs - 1):
            wandb.log(
                {
                    "train/spec": train_spec_loss,
                    "train/mask": train_mask_loss,
                    "val/spec":   val_spec_loss,
                    "val/mask":   val_mask_loss,
                    # atomic train losses
                    "train/mag_mse":   loss_acc["mag_mse"],
                    "train/phase_dot": loss_acc["phase_dot"],
                    "train/phase_if":  loss_acc["phase_if"],
                    "train/phase_aw":  loss_acc["phase_aw"],
                    "train/time_cons": loss_acc["time_cons"],
                    "train/wave_l1":   loss_acc["wave_l1"],
                    # atomic val losses
                    "val/mag_mse":     val_means["mag_mse"],
                    "val/phase_dot":   val_means["phase_dot"],
                    "val/phase_if":    val_means["phase_if"],
                    "val/phase_aw":    val_means["phase_aw"],
                    "val/time_cons":   val_means["time_cons"],
                    "val/wave_l1":     val_means["wave_l1"],
                    "epoch":      epoch,
                    "lr/spec":    opt_spec.param_groups[0]["lr"],
                    "lr/mask":    opt_mask.param_groups[0]["lr"],
                    "phase_weight/curriculum": w_curr,
                    "damage_w":   damage_w,
                    # --- mask atomics ------------------
                    "train/mask_px":     loss_acc["mask_px"],
                    "train/mask_damage": loss_acc["mask_damage"],
                    "train/mask_dice":   loss_acc["mask_dice"],

                    "val/mask_px":       val_mask_means["mask_px"],
                    "val/mask_damage":   val_mask_means["mask_damage"],
                    "val/mask_dice":     val_mask_means["mask_dice"],
                },
                step=epoch,
            )

        # ---- schedulers / early‑stop -------------------------------------
        sch_spec.step(val_spec_loss)
        sch_mask.step()

        EPS = 1e-4
        improved_spec = val_spec_loss < best_spec - EPS
        improved_mask = val_mask_loss < best_mask - EPS

        if improved_spec:
            best_spec = val_spec_loss
            torch.save(spec_autoencoder.state_dict(), "results_diff/autoencoders/spec_autoencoder_best.pt")
        if improved_mask:
            best_mask = val_mask_loss
            torch.save(mask_autoencoder.state_dict(), "results_diff/autoencoders/mask_autoencoder_best.pt")

        patience_ctr = 0 if (improved_spec or improved_mask) else patience_ctr + 1
        if patience_ctr >= patience:
            break

    # ---------------------------------------------------------------------
    #  Finish
    # ---------------------------------------------------------------------
    spec_hist = {"train_loss": hist["spec_train"], "val_loss": hist["spec_val"]}
    mask_hist = {"train_loss": hist["mask_train"], "val_loss": hist["mask_val"]}

    wandb.run.summary["val/spec"] = val_spec_loss
    wandb.run.summary["val/mask"] = val_mask_loss

    return spec_hist, mask_hist
