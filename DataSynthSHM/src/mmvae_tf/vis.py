import os
import gc
import json
import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import umap

# ----- Visualization Functions and tests -----
def save_visualizations_and_metrics(model, train_dataset, val_dataset, training_metrics, output_dir="results_mmvae"):
    """
    Aggregates and saves:
      1. Training/validation curves (total and per-loss)
      2. UMAP of latent space
      3. Cosine similarity histogram
      4. Interpolation through latent space
      5. Weight statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    # 1. Training/Validation Loss Curves
    def plot_total_loss_curves(metrics):
        epochs = list(range(1, len(metrics['train_total']) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=metrics['train_total'], mode='lines+markers', name="Train Total", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=epochs, y=metrics['val_total'], mode='lines+markers', name="Val Total", line=dict(color='red')))
        fig.update_layout(title="Total Loss vs Epochs", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_white")
        file_path = os.path.join(plots_dir, "total_loss_curves.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved total loss plot to {file_path}")

    def plot_individual_losses(metrics):
        epochs = list(range(1, len(metrics['train_total']) + 1))
        fig = go.Figure()
        loss_keys = ['time', 'mrstft', 'grad', 'lap', 'mask', 'damage', 'mag']
        colors = {
            'time': 'orange', 'mrstft': 'green', 'grad': 'blue',
            'lap': 'purple', 'mask': 'black', 'damage': 'gray', 'mag': 'pink'
        }
        for key in loss_keys:
            if f"train_{key}" in metrics:
                fig.add_trace(go.Scatter(
                    x=epochs, y=metrics[f"train_{key}"],
                    mode='lines', name=f"Train {key.title()}",
                    line=dict(color=colors.get(key, 'gray'), dash='dash')))
                fig.add_trace(go.Scatter(
                    x=epochs, y=metrics[f"val_{key}"],
                    mode='lines', name=f"Val {key.title()}",
                    line=dict(color=colors.get(key, 'gray'), dash='dot')))

        fig.update_layout(
            title="Individual Losses vs Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white")
        file_path = os.path.join(plots_dir, "individual_loss_curves.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved individual loss plots to {file_path}")

    plot_total_loss_curves(training_metrics)
    
    plot_individual_losses(training_metrics)
    
    #-----------------------------------------------------------------------
    # 2. 3D Latent Space Visualization
    def extract_and_reduce_latents(dataset):
        latent_vectors = []
        test_ids = []
        for spec_in, mask_in, test_id_in, _wave_in in dataset:
            # Get latent means from both encoders
            mu_spec, _ = model.spec_encoder(spec_in, training=False)
            mu_mask, _ = model.mask_encoder(mask_in, training=False)

            # Strategy A: concatenate
            z = tf.concat([mu_spec, mu_mask], axis=-1)
            # Alternatively, Strategy B (average): z = (mu_spec + mu_mask) / 2.0

            latent_vectors.append(z.numpy())

            # Test IDs
            if isinstance(test_id_in, tf.Tensor):
                test_ids.append(test_id_in.numpy().flatten())
            else:
                test_ids.append(np.array(test_id_in).flatten())

        latent_vectors = np.concatenate(latent_vectors, axis=0)
        test_ids = np.concatenate(test_ids, axis=0)

        # Dimensionality reduction with UMAP
        reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.05)
        latent_3d = reducer.fit_transform(latent_vectors)
        return latent_3d, test_ids


    latent_3d, train_test_ids = extract_and_reduce_latents(train_dataset)

    def plot_latent_space_3d(latent_3d, test_ids):
        df = pd.DataFrame(latent_3d, columns=["UMAP_1", "UMAP_2", "UMAP_3"])
        df["Test ID"] = pd.to_numeric(test_ids, errors="coerce")
        fig = px.scatter_3d(df, x="UMAP_1", y="UMAP_2", z="UMAP_3",
                             color="Test ID", color_continuous_scale="Viridis",
                             title="Latent Space Visualization (3D UMAP)", opacity=0.8)
        file_path = os.path.join(plots_dir, "latent_space_3d.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved 3D latent space plot to {file_path}")

    plot_latent_space_3d(latent_3d, train_test_ids)

    # 3. Latent Analysis: Compute and save cosine similarity and Euclidean distance histograms using validation data.
    def latent_analysis(dataset):
        latent_vectors = []

        # Run over validation set
        for spec_in, mask_in, _, _ in dataset:
            mu, logvar = model.spec_encoder(spec_in, training=False)

            # 🔍 Collapse diagnostic
            print("μ std across batch:", tf.math.reduce_std(mu).numpy(),
                "   log σ² mean:", tf.reduce_mean(logvar).numpy())

            mu_mask, _ = model.mask_encoder(mask_in, training=False)
            z = tf.concat([mu, mu_mask], axis=-1)
            latent_vectors.append(z.numpy())

        latent_vectors = np.concatenate(latent_vectors, axis=0)

        # Cosine similarity (across pairs)
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim_matrix = cosine_similarity(latent_vectors)
        upper = cos_sim_matrix[np.triu_indices_from(cos_sim_matrix, k=1)]

        fig = go.Figure(data=go.Histogram(
            x=upper,
            histnorm='probability density',
            marker_color='blue',
            opacity=0.7))
        fig.update_layout(
            title="Cosine Similarity Distribution (Validation Latents)",
            xaxis_title="Cosine Similarity",
            yaxis_title="Probability Density",
            template="plotly_white")
        file_path = os.path.join(plots_dir, "cosine_similarity_hist.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved cosine similarity histogram to {file_path}")

        return {"avg_cosine_similarity": float(np.mean(upper))}

    latent_metrics = latent_analysis(val_dataset)
    print("Latent analysis metrics:", latent_metrics)


    # 4. Latent Interpolation
    def latent_interpolation(dataset, latent_dim=256):
        for spec_batch, mask_batch, _, _ in dataset.take(1):
            if spec_batch.shape[0] < 2:
                print("Need at least 2 samples for interpolation")
                return

            # pick two samples
            src_spec = tf.expand_dims(spec_batch[0], 0)
            tgt_spec = tf.expand_dims(spec_batch[1], 0)
            src_mask = tf.expand_dims(mask_batch[0], 0)
            tgt_mask = tf.expand_dims(mask_batch[1], 0)

            μ_spec_src, _ = model.spec_encoder(src_spec,  training=False)
            μ_spec_tgt, _ = model.spec_encoder(tgt_spec,  training=False)
            μ_mask_src, _ = model.mask_encoder(src_mask,  training=False)
            μ_mask_tgt, _ = model.mask_encoder(tgt_mask,  training=False)

            num_steps = 8
            alphas = np.linspace(0, 1, num_steps)

            plt.figure(figsize=(num_steps * 2, 6))
            for i, a in enumerate(alphas):
                z_spec = (1 - a) * μ_spec_src + a * μ_spec_tgt
                z_mask = (1 - a) * μ_mask_src + a * μ_mask_tgt

                recon_spec = model.spec_decoder(z_spec,  training=False)
                recon_mask = model.mask_decoder(z_mask,  training=False)

                # top row: spectrogram
                plt.subplot(2, num_steps, i + 1)
                plt.imshow(recon_spec[0, :, :, 0], aspect='auto', cmap='viridis')
                plt.title(f"α={a:.1f}")
                plt.axis('off')

                # bottom row: mask
                plt.subplot(2, num_steps, num_steps + i + 1)
                plt.imshow(recon_mask[0, :, :, 0], cmap='gray')
                plt.axis('off')

            plt.tight_layout()
            out = os.path.join(output_dir, "latent_interpolation.png")
            plt.savefig(out, dpi=300)
            plt.close()
            print(f"Saved latent interpolation plot to {out}")
            break

    
    latent_interpolation(val_dataset)

    # 5. Save model weight statistics.
    def save_model_weights_stats(model, out_path):
        with open(out_path, "w") as f:
            for layer in model.layers:
                for w in layer.weights:
                    w_np = w.numpy()
                    f.write(f"{w.name:<60} "
                            f"trainable={w.trainable:<5} "
                            f"min={w_np.min():.4f} "
                            f"max={w_np.max():.4f} "
                            f"mean={w_np.mean():.4f} "
                            f"std={w_np.std():.4f}\n")
        print("🔍 Weight stats saved →", out_path)

    
    save_model_weights_stats(model, os.path.join(output_dir, "weights_summary.txt"))


    # 6. Print training summary
    def summarize_training(metrics, out_path=None):
        print("\n📊 Final Training Summary:")

        def delta(final, first):
            return f"{final:.4f} (Δ {final - first:+.4f})"

        n_epochs = len(metrics['train_total'])
        best_epoch = np.argmin(metrics['val_total']) + 1
        best_val_total = np.min(metrics['val_total'])

        summary = {
            "epochs": n_epochs,
            "best_val_loss": float(best_val_total),
            "best_val_epoch": best_epoch,
            "final": {},
            "deltas": {}
        }

        def print_metric(key, name, train=True):
            train_key = f"train_{key}"
            val_key = f"val_{key}"

            if train_key in metrics and val_key in metrics:
                val_first, val_final = metrics[val_key][0], metrics[val_key][-1]
                train_first, train_final = metrics[train_key][0], metrics[train_key][-1]
                print(f"  {name:10} | "
                    f"Train: {delta(train_final, train_first)}   "
                    f"Val: {delta(val_final, val_first)}")

                summary['final'][train_key] = float(train_final)
                summary['final'][val_key] = float(val_final)
                summary['deltas'][train_key] = float(train_final - train_first)
                summary['deltas'][val_key] = float(val_final - val_first)

        print(f"🧪 Epochs: {n_epochs}")
        print(f"📌 Best Val Total Loss: {best_val_total:.4f} (Epoch {best_epoch})")
        print()

        print_metric("total", "Total Loss")
        print_metric("mask",  "Mask Loss")
        print_metric("js",    "JS Divergence")
        print_metric("time",  "Time Loss")
        print_metric("mrstft","MRSTFT")
        print_metric("grad",  "Grad Loss")
        print_metric("lap",   "Laplacian")
        print_metric("damage","Damage")

        if out_path:
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n📝 Summary saved to: {out_path}")
        
        summarize_training(training_metrics, out_path=os.path.join(output_dir, "training_summary.json"))

    # Optionally, you can return all gathered metrics:
    return {
        "latent_metrics": latent_metrics,
        "latent_space_3d": latent_3d,
    }
