import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calculate_reconstruction_metrics(original, reconstructed):
    """
    Calculate RMSE and NCC between original and reconstructed waveforms.

    Args:
        original (np.ndarray): (batch, time, channels)
        reconstructed (np.ndarray): (batch, time, channels)

    Returns:
        dict: Contains per-segment RMSE & NCC, along with mean/std.
    """
    mse_per_segment = np.mean((original - reconstructed) ** 2, axis=(1, 2))
    rmse_per_segment = np.sqrt(mse_per_segment)

    ncc_values = []
    for i in range(original.shape[0]):
        seg_ncc = []
        for c in range(original.shape[2]):
            x = original[i, :, c]
            y = reconstructed[i, :, c]

            x = (x - x.mean()) / (x.std() + 1e-8)
            y = (y - y.mean()) / (y.std() + 1e-8)

            corr = np.correlate(x, y, mode='valid')[0] / len(x)
            seg_ncc.append(corr)

        ncc_values.append(np.mean(seg_ncc))

    return {
        "rmse_mean": float(np.mean(rmse_per_segment)),
        "rmse_std": float(np.std(rmse_per_segment)),
        "ncc_mean": float(np.mean(ncc_values)),
        "ncc_std": float(np.std(ncc_values)),
        "rmse_per_segment": rmse_per_segment,
        "ncc_per_segment": ncc_values
    }


def plot_overlay_all_channels(original, reconstructed, segment_idx=0, fs=200, normalized=True, title_extra=""):
    """
    Overlay original and reconstructed waveforms across all channels.

    Args:
        original (np.ndarray): (batch, time, channels)
        reconstructed (np.ndarray): (batch, time, channels)
        segment_idx (int): Which sample to visualize
        fs (int): Sampling rate
        normalized (bool): Annotate plot with normalization flag
        title_extra (str): Extra string to append to the title
    """
    T, C = original.shape[1], original.shape[2]
    t = np.arange(T) / fs
    rmse = np.sqrt(np.mean((original[segment_idx] - reconstructed[segment_idx]) ** 2))

    fig, ax = plt.subplots(figsize=(10, 5))
    for ch in range(C):
        ax.plot(t, original[segment_idx, :, ch], color='blue', alpha=0.5)
        ax.plot(t, reconstructed[segment_idx, :, ch], color='red', alpha=0.5)

    ax.plot([], [], color='blue', label='Original')
    ax.plot([], [], color='red', label='Reconstructed')
    ax.legend()

    label = f"Segment {segment_idx} | RMSE: {rmse:.4f}"
    if normalized:
        label += " (normalized)"
    if title_extra:
        label += f" • {title_extra}"
    ax.set_title(label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_metrics_to_csv(metrics_dict, test_ids, save_path):
    """
    Save per-segment metrics to a CSV.

    Args:
        metrics_dict (dict): Output from `calculate_reconstruction_metrics()`
        test_ids (np.ndarray): (N,)
        save_path (str): Output .csv path
    """
    df = pd.DataFrame({
        "Segment": np.arange(len(test_ids)),
        "Test ID": test_ids,
        "RMSE": metrics_dict["rmse_per_segment"],
        "NCC": metrics_dict["ncc_per_segment"]
    })
    df.to_csv(save_path, index=False)
    print(f"✅ Saved metrics to {save_path}")
