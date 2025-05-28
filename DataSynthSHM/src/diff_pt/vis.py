import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from tqdm import tqdm
import umap

# ----- Visualization Functions and tests -----
def visualize_training_history(history, save_path=None):
    """
    Visualize the training history.

    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')

    if history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.close()

def save_visualizations_and_metrics(model, train_loader, val_loader, training_metrics, output_dir="results_diff"):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    def plot_training_curves(metrics):
        epochs = list(range(1, len(metrics['train_loss']) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=metrics['train_loss'],
                                 mode='lines+markers', name="Train Loss", line=dict(color='blue')))
        if metrics['val_loss']:
            fig.add_trace(go.Scatter(x=epochs, y=metrics['val_loss'],
                                     mode='lines+markers', name="Val Loss", line=dict(color='red')))
        fig.update_layout(title="Loss vs Epochs",
                          xaxis_title="Epoch", yaxis_title="Loss",
                          template="plotly_white")
        file_path = os.path.join(plots_dir, "train_val_loss.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved training curves to {file_path}")

    plot_training_curves(training_metrics)

    def extract_and_reduce_latents(loader):
        # Explicitly set evaluation mode for internal modules
        model.diffusion_model.eval()
        for ae in model.autoencoders.values():
            ae.eval()

        latents, ids = [], []
        device = model.device  # Correct way to get the device

        for spec, _, _, test_id, _ in loader:
            spec = spec.to(device)
            with torch.no_grad():
                _, z = model.autoencoders["spec"](spec)
            latents.append(z.cpu().numpy())
            ids.append(test_id.numpy().flatten())

        latents = np.concatenate(latents, axis=0)
        ids = np.concatenate(ids, axis=0)

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.05, n_components=3, random_state=42)

        latent_3d = reducer.fit_transform(latents)

        return latent_3d, ids

    latent_3d, train_ids = extract_and_reduce_latents(train_loader)

    def plot_latent_space_3d(latent_3d, ids):
        df = pd.DataFrame(latent_3d, columns=["UMAP_1", "UMAP_2", "UMAP_3"])
        df["Test ID"] = pd.to_numeric(ids, errors="coerce")
        fig = px.scatter_3d(df, x="UMAP_1", y="UMAP_2", z="UMAP_3", color="Test ID",
                             color_continuous_scale="Viridis", title="Latent Space Visualization (3D UMAP)", opacity=0.8)
        file_path = os.path.join(plots_dir, "latent_space_3d.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved 3D latent space plot to {file_path}")

    plot_latent_space_3d(latent_3d, train_ids)

    def latent_analysis(loader):
        model.eval()
        latents = []
        for spec, _, _, _, _ in loader:
            spec = spec.to(model.device)
            with torch.no_grad():
                _, z = model.autoencoders["spec"](spec)
            latents.append(z.cpu().numpy())
        latents = np.concatenate(latents, axis=0)
        norms = np.linalg.norm(latents, axis=1, keepdims=True)
        normalized = latents / (norms + 1e-8)
        cosine_sim = np.dot(normalized, normalized.T).diagonal()
        fig = go.Figure(data=go.Histogram(x=cosine_sim, histnorm='probability density', marker_color='blue', opacity=0.7))
        fig.update_layout(title="Cosine Similarity Distribution (Validation Latents)",
                          xaxis_title="Cosine Similarity", yaxis_title="Probability Density", template="plotly_white")
        file_path = os.path.join(plots_dir, "cosine_similarity_hist.html")
        pio.write_html(fig, file=file_path, auto_open=False)
        print(f"Saved cosine similarity histogram to {file_path}")
        return {"avg_cosine_similarity": float(np.mean(cosine_sim))}

    latent_metrics = latent_analysis(val_loader)
    print("Latent analysis metrics:", latent_metrics)

    return {
        "latent_metrics": latent_metrics,
        "latent_space_3d": latent_3d
    }

def save_plotly_loss_curve(metrics, save_path, title="Loss vs Epochs"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = list(range(1, len(metrics['train_loss']) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=metrics['train_loss'],
                             mode='lines+markers', name="Train Loss", line=dict(color='blue')))
    if metrics['val_loss']:
        fig.add_trace(go.Scatter(x=epochs, y=metrics['val_loss'],
                                 mode='lines+markers', name="Val Loss", line=dict(color='red')))

    fig.update_layout(title=title,
                      xaxis_title="Epoch", yaxis_title="Loss",
                      template="plotly_white")

    pio.write_html(fig, file=save_path, auto_open=False)
    print(f"✅ Saved Plotly loss curve to {save_path}")
