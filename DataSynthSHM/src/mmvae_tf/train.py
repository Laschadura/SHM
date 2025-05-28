import os
import pandas as pd
import tensorflow as tf

from .losses import (
    custom_mask_loss, gradient_loss_phase_only,
    laplacian_loss_phase_only, magnitude_l1_loss,
    multi_channel_mrstft_loss, waveform_l1_loss,
    waveform_si_l1_loss
)
from .utils import (    
    get_beta_schedule, get_time_weight, dynamic_weighting,
    log_vram_usage
)


# ---------- Training functions ----------
ACC_STEPS = 4  # accumulate gradients over 4 mini-batches
grad_accum = None
step_in_acc = None

def train_step(model, optimizer,
               spec_mb, mask_mb, test_id_mb, wave_mb,
               beta, spec_weight, mask_weight,
               modality_dropout_prob, time_weight, epoch, loss_weights,
               unfreeze_epoch=20):
    """One training step **per replica**.

    * Computes all losses.
    * Performs **gradient accumulation** over ``ACC_STEPS`` mini‑batches.
    * Applies the optimiser _inside_ the replica context so that
      `LossScaleOptimizer.aggregate_gradients()` can call `merge_call()`
      safely.
    """
    global grad_accum, step_in_acc

    # ---------------------------------------------------------------------
    # Forward + losses
    # ---------------------------------------------------------------------
    with tf.GradientTape() as tape:
        recon_spec, recon_mask, (_, _, _, js_div) = model(
            spec_mb, mask_mb, test_id_mb,
            training=True, missing_modality=None)

        # -------------- time‑domain reconstruction -----------------------
        time_len   = tf.shape(wave_mb)[1]
        recon_wave = model.istft_layer(recon_spec, time_len)
        wave_mb_f  = tf.cast(wave_mb, tf.float32)

        alpha    = tf.minimum(1.0, tf.cast(epoch, tf.float32) / 30.0)
        L_time   = (1 - alpha) * waveform_l1_loss(wave_mb_f, recon_wave) + \
                    alpha       * waveform_si_l1_loss(wave_mb_f, recon_wave)
        L_mrstft = multi_channel_mrstft_loss(wave_mb_f, recon_wave)

        # -------------- spectral‑only losses ----------------------------
        grad_loss = gradient_loss_phase_only(spec_mb, recon_spec)
        lap_loss  = laplacian_loss_phase_only(spec_mb, recon_spec)
        L_mag     = magnitude_l1_loss(spec_mb, recon_spec)

        if epoch < unfreeze_epoch:  # freeze until ISTFT is reliable
            L_time   = tf.stop_gradient(L_time)
            L_mrstft = tf.stop_gradient(L_mrstft)
            grad_loss= tf.stop_gradient(grad_loss)
            lap_loss = tf.stop_gradient(lap_loss)
            L_mag    = tf.stop_gradient(L_mag)

        # -------------- mask‑branch losses ------------------------------
        mask_loss = custom_mask_loss(mask_mb, recon_mask)

        rv         = tf.random.uniform([])
        drop_spec  = rv <  modality_dropout_prob
        drop_mask  = (rv >= modality_dropout_prob) & (rv < 2.*modality_dropout_prob)
        spec_coeff = tf.cast(~drop_spec, tf.float32)
        mask_coeff = tf.cast(~drop_mask, tf.float32)

        damage_pred = tf.reduce_mean(recon_mask, axis=[1,2,3])
        damage_true = tf.reduce_mean(mask_mb,    axis=[1,2,3])
        loss_damage = tf.reduce_mean(tf.square(damage_pred - damage_true))

        recon_loss = mask_weight * mask_loss * mask_coeff

        w = loss_weights

        total_loss = (
            recon_loss +
            beta * js_div +
            time_weight * L_time +
            w["mrstft"] * L_mrstft +
            w["grad"]   * grad_loss +
            w["lap"]    * lap_loss +
            w["mag"]    * L_mag +
            w["damage"] * loss_damage
        )

        loss = total_loss / ACC_STEPS

    # ---------------------------------------------------------------------
    # Back‑prop ----------
    # ---------------------------------------------------------------------
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)

    # -------- accumulate --------------------------------------------------
    for g_acc, g in zip(grad_accum, grads):
        g_acc.assign_add(g)
    step_in_acc.assign_add(1)

    # ---------- return scalars (no grads) ---------------------------------
    return (
        total_loss, mask_loss, js_div, recon_loss, mask_coeff,
        L_time, time_weight, L_mrstft, grad_loss, lap_loss, loss_damage, L_mag
    )

def val_step(model, spec_in, mask_in, test_id_in, wave_in, beta, mask_weight, time_weight, loss_weights):
    """
    Performs a validation step for one batch.
    Returns individual losses for logging and early stopping.
    Includes the same losses as train_step but without dropout.
    """
    recon_spec, recon_mask, (_, _, _, js_div) = model(
        spec_in, mask_in, test_id_in, training=False, missing_modality=None
    )

    recon_wave = model.istft_layer(recon_spec, length=tf.shape(wave_in)[1])
    wave_in_f32 = tf.cast(wave_in, tf.float32)
    L_time_val = waveform_l1_loss(wave_in_f32, recon_wave)
    L_mrstft_val = multi_channel_mrstft_loss(wave_in_f32, recon_wave)

    grad_val = gradient_loss_phase_only(spec_in, recon_spec)
    lap_val = laplacian_loss_phase_only(spec_in, recon_spec)
    mag_val = magnitude_l1_loss(spec_in, recon_spec)


    damage_pred = tf.reduce_mean(recon_mask, axis=[1,2,3])
    damage_true = tf.reduce_mean(mask_in,    axis=[1,2,3])
    damage_val = tf.reduce_mean(tf.square(damage_pred - damage_true))

    mask_l  = custom_mask_loss(mask_in, recon_mask)
    recon_l = mask_weight * mask_l

    w = loss_weights

    tot_l = (
        recon_l +
        beta * js_div +
        time_weight * L_time_val +
        w["mrstft"] * L_mrstft_val +
        w["grad"]   * grad_val +
        w["lap"]    * lap_val +
        w["mag"]    * mag_val +
        w["damage"] * damage_val
    )


    return {
        "total": tot_l,
        "mask": mask_l,
        "js": js_div,
        "time": L_time_val,
        "mrstft": L_mrstft_val,
        "grad": grad_val,
        "lap": lap_val,
        "damage": damage_val,
        "mag": mag_val
    }

def train_spectral_mmvae(
    model,
    output_dir,
    train_dataset,
    val_dataset,
    optimizer,
    num_epochs: int = 100,
    patience: int = 10,
    beta_schedule: str = "linear",
    modality_dropout_prob: float = 0.10,
    strategy=None,
    unfreeze_epoch: int = 20,
    beta_warmup_epochs: int = 60,
    max_beta: float = 0.15,
    loss_weights: dict | None = None,
    ):
    """Spectral‑MMVAE training loop.

    Everything is scoped to *output_dir* so each sweep run keeps its own
    checkpoints and logs:

    ``output_dir/
        ├── best_spectral_mmvae.weights.h5
        ├── best_model_spectral_mmvae.keras
        ├── final_spectral_mmvae.weights.h5
        ├── final_model_spectral_mmvae.keras
        └── logs/beta_tracking.csv``
    """

    # ------------------------------------------------------------------
    # 0)  per‑run paths & default loss weights
    # ------------------------------------------------------------------
    best_weights_path  = os.path.join(output_dir, "best_spectral_mmvae.weights.h5")
    best_model_path    = os.path.join(output_dir, "best_model_spectral_mmvae.keras")
    final_weights_path = os.path.join(output_dir, "final_spectral_mmvae.weights.h5")
    final_model_path   = os.path.join(output_dir, "final_model_spectral_mmvae.keras")
    metrics_path       = os.path.join(output_dir, "training_metrics.npy")

    if loss_weights is None:
        loss_weights = {
            "mrstft": 1.0,
            "grad": 0.3,
            "lap": 0.3,
            "mag": 0.3,
            "damage": 150.0,
        }

    # ------------------------------------------------------------------
    # 1)  bookkeeping
    # ------------------------------------------------------------------
    metrics = {k: [] for k in (
        "train_total", "train_mask", "train_js", "train_time", "train_mrstft",
        "train_grad", "train_lap", "train_damage", "train_mag",
        "val_total", "val_mask", "val_js", "val_time", "val_mrstft",
        "val_grad", "val_lap", "val_damage", "val_mag",
    )}

    best_val_loss      = float("inf")
    no_improvement_cnt = 0
    train_batches      = sum(1 for _ in train_dataset)
    val_batches        = sum(1 for _ in val_dataset)
    print(f"🔄 Starting Training: {train_batches} train batches, {val_batches} val batches")

    beta_log = []

    # gradient‑accumulation buffers
    global grad_accum, step_in_acc
    grad_accum = [tf.Variable(tf.zeros_like(v), trainable=False) for v in model.trainable_variables]
    step_in_acc = tf.Variable(0, trainable=False, dtype=tf.int32)

    # ------------------------------------------------------------------
    # 2)  epoch loop
    # ------------------------------------------------------------------
    for epoch in range(num_epochs):
        print(f"\n🔍 VRAM usage at start of epoch {epoch + 1}:")
        log_vram_usage()

        beta = get_beta_schedule(epoch, num_epochs, beta_schedule, beta_warmup_epochs, max_beta)
        # weight for the L1 loss in the time domain
        time_weight = get_time_weight(epoch,warmup = 200, max_w = 0.1)
        mask_weight = dynamic_weighting(epoch, num_epochs)
        print(f"📌 Epoch {epoch + 1}/{num_epochs} | Beta={beta:.5f} | MaskW={mask_weight:.02f}")

        if epoch == unfreeze_epoch:
            model.istft_layer.trainable = True
            print("🔓 Unfroze TFInverseISTFT layer")

        acc = {k: 0.0 for k in (
            "train_total", "train_mask", "train_js", "train_time",
            "train_mrstft", "train_grad", "train_lap", "train_damage", "train_mag", "train_steps")}


        for step, (spec_in, mask_in, test_id_in, wave_in) in enumerate(train_dataset):
            # check if the latent collapsed
            if step == 0:
                mu_dbg, logvar_dbg = model.spec_encoder(spec_in, training=False)
                tf.print("🧠 Epoch", epoch, "| μ.std =", tf.math.reduce_std(mu_dbg),
                        " | log σ² mean =", tf.reduce_mean(logvar_dbg))

            results = distributed_train_step(
                strategy, model, optimizer,
                (spec_in, mask_in, test_id_in, wave_in),
                tf.constant(beta,  tf.float32),
                tf.constant(mask_weight, tf.float32),
                tf.constant(time_weight, tf.float32),
                tf.constant(epoch, tf.int32),
                tf.constant(modality_dropout_prob, tf.float32),
                loss_weights
            )

            (tot, mask_l, js_d, recon_l, mask_c,
            time_l, _, mrstft_l, grad_l, lap_l, dmg_l, mag_l) = results

            # apply weights every ACC_STEPS *host* iterations
            if (step + 1) % ACC_STEPS == 0:
                # run the update on all replicas
                strategy.run(apply_accum_grads, args=(optimizer, model))

            red = lambda x: strategy.reduce(tf.distribute.ReduceOp.MEAN, x, axis=None)
            tot, mask_l, js_d, _, mask_c, time_l, _, mrstft_l, grad_l, lap_l, dmg_l, mag_l = map(red, results)

            # 💬 debug every 50 mini-batches
            if step % 50 == 0:
                tf.print("📉 [Debug] MRSTFT Loss:", mrstft_l)
                tf.print("🧮 [Debug] ISTFT beta:", model.istft_layer.beta)
            
            beta_log.append((epoch, step, float(mrstft_l.numpy()), float(model.istft_layer.beta.numpy())))


            acc["train_total"] += float(tot.numpy())
            acc["train_mask"] += float((mask_l * mask_c).numpy())
            acc["train_js"] += float(js_d.numpy())
            acc["train_time"] += float(time_l.numpy())
            acc["train_mrstft"] += float(mrstft_l.numpy())
            acc["train_grad"] += float(grad_l.numpy())
            acc["train_lap"] += float(lap_l.numpy())
            acc["train_damage"] += float(dmg_l.numpy())
            acc["train_mag"] += float(mag_l.numpy())
            acc["train_steps"] += 1

            print(f"... | TimeLoss={acc['train_time']/acc['train_steps']:.4f} | TimeW={time_weight:.4f} | MRSTFT={acc['train_mrstft']/acc['train_steps']:.4f}")

        for key in ["total", "mask", "js", "time", "mrstft", "grad", "lap", "damage", "mag"]:
            metrics[f"train_{key}"].append(acc[f"train_{key}"] / max(acc["train_steps"], 1))

        print(f"✅ [Train] Loss={metrics['train_total'][-1]:.4f} | "
              f"Mask={metrics['train_mask'][-1]:.4f} | JS={metrics['train_js'][-1]:.4f}")

        val_stats = {k: 0.0 for k in ("total", "mask", "js", "time", "mrstft", "grad", "lap", "damage", "mag", "steps")}

        for spec_in, mask_in, test_id_in, wave_in in val_dataset:
            res = val_step(model, spec_in, mask_in, test_id_in, wave_in,
                           tf.constant(beta, tf.float32),
                           tf.constant(mask_weight, tf.float32),
                           tf.constant(time_weight, tf.float32),
                           loss_weights)

            for k in res:
                val_stats[k] += float(res[k].numpy())
            val_stats["steps"] += 1

        for k in ["total", "mask", "js", "time", "mrstft", "grad", "lap", "damage", "mag"]:
            metrics[f"val_{k}"].append(val_stats[k] / max(val_stats["steps"], 1))

        print(f"  🔵 [Val] => Total={metrics['val_total'][-1]:.4f} | "
              f"Mask={metrics['val_mask'][-1]:.4f} | JS={metrics['val_js'][-1]:.4f} | "
              f"🟣 TimeLoss={metrics['val_time'][-1]:.4f} | 🎯 MRSTFT={metrics['val_mrstft'][-1]:.4f}")

        current_val = metrics["val_total"][-1] if val_stats["steps"] else float("inf")
        if current_val < best_val_loss:
            best_val_loss     = current_val
            no_improvement_cnt = 0

            model.save_weights(best_weights_path)
            model.save(best_model_path)

            pd.DataFrame(beta_log, columns=["epoch", "step", "mrstft", "beta"]).to_csv(
                os.path.join(output_dir, "logs", "beta_tracking.csv"), index=False
            )
            print("✅ Saved best weights and model →", os.path.relpath(best_weights_path, output_dir))
        else:
            no_improvement_cnt += 1
            print(f"🚨 No improvement for {no_improvement_cnt}/{patience}")

        # ------------- EARLY‑STOPPING ----------------------------------
        if no_improvement_cnt >= patience:
            print(f"🛑 Early stopping at epoch {epoch + 1}.")
            model.save_weights(final_weights_path)
            model.save(final_model_path)

            pd.DataFrame(beta_log, columns=["epoch", "step", "mrstft", "beta"]).to_csv(
                os.path.join(output_dir, "logs", "beta_tracking.csv"), index=False
            )
            print("📦 Saved final model to:", os.path.relpath(final_model_path, output_dir))
            break

    return metrics

@tf.function
def apply_accum_grads(optimizer, model):
    """Must be run via  strategy.run(apply_accum_grads, …)  so that we
    are in replica-context when apply_gradients() is executed."""
    optimizer.apply_gradients(zip(grad_accum, model.trainable_variables))
    for g in grad_accum:
        g.assign(tf.zeros_like(g))
    step_in_acc.assign(0)

@tf.function
def distributed_train_step(strategy, model, optimizer, batch,
                           beta, mask_w, time_w, epoch, dropout_prob, loss_weights):


    spec_mb, mask_mb, test_id_mb, wave_mb = batch

    # ---- run on every replica ---------------------------------
    print("⏱ Tracing... (first step)")
    per_replica = strategy.run(
    train_step,
    args=(model, optimizer,
          spec_mb, mask_mb, test_id_mb, wave_mb,
          beta, 0.0,                                # spec_weight (unused)
          mask_w, dropout_prob,
          time_w, epoch,
          loss_weights)
        )
    print("✅ Traced — now starts real training")

    # reduce *once* here – no extra reduction in the caller
    return [strategy.reduce(tf.distribute.ReduceOp.MEAN, t, axis=None)
            for t in per_replica]

def init_accumulators(model):
    """Create the gradient‑accumulation buffers _once_."""
    global grad_accum, step_in_acc
    grad_accum = [tf.Variable(tf.zeros_like(v), trainable=False)
                  for v in model.trainable_variables]
    step_in_acc = tf.Variable(0, trainable=False, dtype=tf.int32)
