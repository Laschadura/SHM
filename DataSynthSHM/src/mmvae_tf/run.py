from mmvae_tf.env import configure_tf
configure_tf()

import os
import gc
import numpy as np
import csv

import tensorflow as tf
from tensorflow.keras.optimizers import AdamW # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts # type: ignore

from bridge_data.loader import load_data

from .io import create_tf_dataset
from .model import SpectralMMVAE
from .train import train_spectral_mmvae, init_accumulators
from .vis import save_visualizations_and_metrics

# ----- Main Function -----
def main():
    """Run a sweep over different β‑values and weight configurations.
    Each configuration gets its own sub‑folder (e.g. `results_mmvae/beta_0_03_cfg_1/`).
    After every run we append one line to `results_mmvae/beta_sweep_summary.csv`
    so you can compare them later.

    If `resume_training` is **True** and a sub‑folder already contains
    `training_metrics.npy` and model weights, the run will be resumed
    instead of re‑trained from scratch.
    """

    # ------------------------------------------------------------------
    # 0)  Sweep parameters & globals
    # ------------------------------------------------------------------
    debug_mode           = False
    latent_dim           = 256
    batch_size           = 128
    total_epochs         = 500
    patience             = 50
    resume_training      = False

    unfreeze_istft_epoch = 100
    beta_warmup_epochs   = 60
    beta_schedule        = "linear"
    modality_dropout_prob = 0.0

    #  Dataset params
    segment_duration = 4.0
    nperseg          = 256
    noverlap         = 224
    sample_rate      = 200

    #  Sweep values
    beta_sweep = [0.06, 0.10]

    weight_configs = [
        # A (second-best baseline)
        {"mrstft": 1.0, "grad": 0.3, "lap": 0.3, "mag": 0.3, "damage": 150.0},

        # B (best baseline)
        {"mrstft": 0.7, "grad": 0.1, "lap": 0.1, "mag": 0.1, "damage": 300.0},

        # B-variant 1: a bit more magnitude emphasis
        {"mrstft": 0.7, "grad": 0.1, "lap": 0.1, "mag": 0.2, "damage": 300.0},

        # B-variant 2: slightly stronger gradient & laplacian terms
        {"mrstft": 0.7, "grad": 0.2, "lap": 0.2, "mag": 0.1, "damage": 300.0},

        # B-variant 3: reduce the damage weight from 300 → 250
        {"mrstft": 0.7, "grad": 0.1, "lap": 0.1, "mag": 0.1, "damage": 250.0},

        # A-variant: keep A but lower damage loss to 100
        {"mrstft": 1.0, "grad": 0.3, "lap": 0.3, "mag": 0.3, "damage": 100.0},
    ]


    #  Base directory
    project_root = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.abspath(os.path.join(project_root, ".."))

    #  One summary file for the whole sweep
    sweep_log_path = os.path.join(base_dir, "results_mmvae", "beta_sweep_summary.csv")
    os.makedirs(os.path.join(base_dir, "results_mmvae"), exist_ok=True)

    with open(sweep_log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["beta", "cfg", "best_val_total", "final_val_total", "final_val_js", "final_val_mrstft"])

        for max_beta in beta_sweep:
            for cfg_idx, weights in enumerate(weight_configs):
                run_name = f"beta_{max_beta:.2f}_cfg_{cfg_idx}".replace(".", "_")
                output_dir = os.path.join(base_dir, "results_mmvae", run_name)
                os.makedirs(output_dir, exist_ok=True)

                for sub in ("logs", "model_checkpoints", "latent_analysis", "cross_modal", "plots"):
                    os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

                # 1)  Load / compute cached input tensors
                tag = f"{segment_duration:.2f}s_{nperseg}_{noverlap}"
                cache_dir       = os.path.join(base_dir, "cache")
                os.makedirs(cache_dir, exist_ok=True)
                final_path      = os.path.join(cache_dir, f"specs_{tag}.npy")
                heatmaps_path   = os.path.join(cache_dir, f"masks_{tag}.npy")
                ids_path        = os.path.join(cache_dir, f"segIDs_{tag}.npy")
                segments_path   = os.path.join(cache_dir, f"segments_{tag}.npy")

                if all(map(os.path.exists, [final_path, heatmaps_path, ids_path, segments_path])):
                    print("✅  Loading cached NumPy arrays …")
                    spectral_features = np.load(final_path, mmap_mode="r")
                    mask_segments = np.load(heatmaps_path, mmap_mode="r")
                    test_ids = np.load(ids_path, mmap_mode="r")
                    segments = np.load(segments_path, mmap_mode="r")
                else:
                    print("⚠️  Cache missing – computing everything from raw data …")
                    (_, _, heatmaps, segments, spectrograms, test_ids) = load_data(
                        segment_duration=segment_duration,
                        nperseg=nperseg,
                        noverlap=noverlap,
                        sample_rate=sample_rate,
                        recompute=False,
                        cache_dir=cache_dir)
                    spectral_features = spectrograms
                    mask_segments = np.stack([heatmaps[tid] for tid in test_ids], axis=0)
                    np.save(final_path, spectral_features)
                    np.save(heatmaps_path, mask_segments)
                    np.save(ids_path, test_ids)
                    np.save(segments_path, segments)
                    print("✅  Written new cache files.")

                # 2)  Train / Val split + tf.data
                N = spectral_features.shape[0]
                perm = np.random.permutation(N)
                train_size = int(0.8 * N)
                train_idx, val_idx = perm[:train_size], perm[train_size:]

                train_ds = create_tf_dataset(spectral_features[train_idx], mask_segments[train_idx],
                                            test_ids[train_idx], segments[train_idx],
                                            batch_size, debug_mode, augment=True)
                val_ds = create_tf_dataset(spectral_features[val_idx], mask_segments[val_idx],
                                        test_ids[val_idx], segments[val_idx],
                                        batch_size, debug_mode, augment=False)

                print(f"✅  Train batches: {sum(1 for _ in train_ds)}  |  Val batches: {sum(1 for _ in val_ds)}")

                strategy = tf.distribute.MirroredStrategy()

                with strategy.scope():
                    spec_shape = spectral_features.shape[1:]
                    mask_shape = (32, 96, 1)

                    model = SpectralMMVAE(latent_dim, spec_shape, mask_shape, nperseg, noverlap)
                    _ = model(tf.zeros((1, *spec_shape)), tf.zeros((1, *mask_shape)), training=True)

                    dummy_time_len = tf.constant(int(segment_duration * sample_rate), tf.int32)
                    _ = model.istft_layer(tf.zeros((1, *spec_shape)), dummy_time_len)

                    lr_schedule = ExponentialDecay(5e-5, decay_steps=10_000, decay_rate=0.9, staircase=True)
                    optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
                    init_accumulators(model)

                    metrics_path = os.path.join(output_dir, "training_metrics.npy")
                    final_weights_path = os.path.join(output_dir, "final_spectral_mmvae.weights.h5")

                    if resume_training and os.path.exists(metrics_path) and os.path.exists(final_weights_path):
                        print("🔄  Resuming previous run …")
                        model.load_weights(final_weights_path)
                        metrics = np.load(metrics_path, allow_pickle=True).item()
                        start_epoch_offset = len(metrics["train_total"])
                    else:
                        metrics = {k: [] for k in (
                            "train_total", "train_mask", "train_js", "train_time", "train_mrstft",
                            "train_grad", "train_lap", "train_damage", "train_mag",     # <-- NEW
                            "val_total",   "val_mask",   "val_js",   "val_time", "val_mrstft",
                            "val_grad",   "val_lap",   "val_damage", "val_mag"          # <-- NEW
                        )}

                        start_epoch_offset = 0

                    new_metrics = train_spectral_mmvae(
                        model, output_dir, train_ds, val_ds, optimizer,
                        num_epochs=total_epochs - start_epoch_offset,
                        patience=patience,
                        beta_schedule=beta_schedule,
                        modality_dropout_prob=modality_dropout_prob,
                        strategy=strategy,
                        unfreeze_epoch=unfreeze_istft_epoch,
                        beta_warmup_epochs=beta_warmup_epochs,
                        max_beta=max_beta,
                        loss_weights=weights  # <-- NEW ARGUMENT
                    )

                    for k in metrics:
                        metrics[k].extend(new_metrics[k])

                    model.save_weights(final_weights_path)
                    np.save(metrics_path, metrics)

                    try:
                        save_visualizations_and_metrics(model, train_ds, val_ds, metrics, output_dir=output_dir)
                    except Exception as e:
                        print("❌ Visualization failed:", e)

                    # ----------------------------------------------------------
                    # 4)  Log summary rows
                    # ----------------------------------------------------------
                    best_val_total  = float(np.min(metrics['val_total']))
                    final_val_total = float(metrics['val_total'][-1])
                    writer.writerow([
                        max_beta, cfg_idx,  # <-- add cfg_idx here
                        best_val_total,
                        final_val_total,
                        float(np.min(metrics['val_js'])),
                        float(np.min(metrics['val_mrstft'])),
                    ])
                    log_file.flush()

                    # ---------- NEW: per-loss summary ----------
                    loss_csv = os.path.join(base_dir, "results_mmvae", "loss_sweep_summary.csv")
                    header = [
                        "beta", "cfg",
                        "cfg_mrstft", "cfg_grad", "cfg_lap", "cfg_mag", "cfg_damage",  # 🆕
                        "best_total", "best_mask", "best_js",
                        "best_time", "best_mrstft", "best_grad", "best_lap", "best_mag", "best_damage"
                    ]
                    write_header = not os.path.exists(loss_csv)

                    def _best(lst):
                        return float(np.min(lst)) if lst else ''

                    with open(loss_csv, "a", newline="") as f_loss:
                        w = csv.writer(f_loss)
                        if write_header:
                            w.writerow(header)
                        w.writerow([
                            max_beta, cfg_idx,
                            weights["mrstft"], weights["grad"], weights["lap"], weights["mag"], weights["damage"],
                            _best(metrics['val_total']),
                            _best(metrics['val_mask']),
                            _best(metrics['val_js']),
                            _best(metrics['val_time']),
                            _best(metrics['val_mrstft']),
                            _best(metrics['val_grad']),
                            _best(metrics['val_lap']),
                            _best(metrics['val_mag']),
                            _best(metrics['val_damage']),
                        ])

                print(f"🎉  Finished run for β = {max_beta:.3f} — results in '{output_dir}'")

                tf.keras.backend.clear_session()
                gc.collect()


if __name__ == "__main__":
    main()
