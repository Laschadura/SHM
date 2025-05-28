# bridge_data/tests/test_waveform_reconstruction.py

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from bridge_data.loader import load_data
from bridge_data.preprocess import compute_complex_spectrogram
from bridge_data.postprocess import inverse_spectrogram, align_by_xcorr  # ✅ include align
from bridge_data.tests.utils import (
    calculate_reconstruction_metrics,
    plot_overlay_all_channels
)

def test_waveform_reconstruction(n_segments=3, fs=200, nperseg=256, noverlap=224):
    print("📥 Loading normalized segments from cache...")
    _, _, _, segments, _, test_ids, metadata, *_ = load_data(recompute=False)

    # Slice subset for quick test
    segments = segments[:n_segments]
    test_ids = test_ids[:n_segments]
    metadata = metadata[:n_segments]

    print("🔄 Computing spectrograms (log-magnitude + phase)...")
    specs = compute_complex_spectrogram(segments, fs, nperseg, noverlap)

    print("🔄 Reconstructing time-domain signals via ISTFT...")
    recon_raw = inverse_spectrogram(specs, segments.shape[1], fs, nperseg, noverlap)

    print("🔧 Aligning reconstructed signals to original via xcorr...")
    aligned_recon = np.empty_like(recon_raw)
    lags = []

    for i in range(n_segments):
        aligned, lag = align_by_xcorr(segments[i], recon_raw[i])
        aligned_recon[i] = aligned
        lags.append(lag)

    print("📊 Computing reconstruction metrics...")
    metrics = calculate_reconstruction_metrics(segments, aligned_recon)
    print(f"✅ RMSE: {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}")
    print(f"✅ NCC:  {metrics['ncc_mean']:.4f} ± {metrics['ncc_std']:.4f}")

    # Save metrics
    os.makedirs("reconstruction_results", exist_ok=True)
    df = pd.DataFrame({
        "Segment": list(range(n_segments)),
        "Test ID": [m["test_id"] for m in metadata],
        "Trace Index": [m["trace_index"] for m in metadata],
        "Peak Position": [m["peak_position"] for m in metadata],
        "Lag": lags,
        "RMSE": metrics["rmse_per_segment"],
        "NCC": metrics["ncc_per_segment"]
    })
    df.to_csv("reconstruction_results/metrics.csv", index=False)
    print("📁 Saved detailed metrics to reconstruction_results/metrics.csv")

    # Plot overlays for visual inspection
    print("📈 Plotting original vs reconstructed overlays (aligned)...")
    for i in range(n_segments):
        plot_overlay_all_channels(
            original=segments,
            reconstructed=aligned_recon,
            segment_idx=i,
            fs=fs
        )

    print("✅ All done. Compared normalized inputs ↔ aligned ISTFT reconstructions.")

if __name__ == "__main__":
    test_waveform_reconstruction()
