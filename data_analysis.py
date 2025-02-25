"""
analysis.py

Script to analyze time-series (accelerometer) data and damage mask data:
1) Loads data from data_loader.
2) Computes and prints fundamental stats (mean, std, histograms) for both modalities.
3) FFT or spectrogram analysis for the time-series to see which frequencies matter.
4) PCA (optionally UMAP) on each modality separately, and (optionally) combined.
5) Basic correlation between high-level summary features of the time-series and the masks.

Prerequisites:
    pip install scikit-learn matplotlib numpy umap-learn (optional)
"""

import os
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.signal import spectrogram
from scipy.signal import welch
import umap.umap_ as umap
import plotly.graph_objs as go
import plotly.io as pio


import data_loader  # You have this from your original script

#============= Utility =================
def create_white_hot_colormap():
    """
    Creates a custom colormap that maps 0 to white and higher values to 'hot' colors.
    """
    colors = [(1, 1, 1),  # white at 0
              (1, 0, 0),  # red
              (1, 1, 0)]  # yellow near the top
    cmap = mcolors.LinearSegmentedColormap.from_list("white_hot", colors, N=256)
    return cmap

def plot_gaussian_ellipse(ax, mean, cov, n_std=1.0, facecolor='none', edgecolor='blue', **kwargs):
    """
    Plot an ellipse representing a Gaussian distribution.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                      facecolor=facecolor, edgecolor=edgecolor, **kwargs)
    ax.add_patch(ellipse)

def plot_3d_gaussian_ellipsoid(ax, mean, cov, n_std=1.0, n_points=50, edgecolor='blue', alpha=0.2):
    """
    Plot a 3D ellipsoid representing the Gaussian distribution with mean & covariance.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    rx, ry, rz = n_std * np.sqrt(vals)
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))
    xyz = np.array([x.flatten(), y.flatten(), z.flatten()])
    xyz_rot = vecs @ xyz
    x_rot = xyz_rot[0, :].reshape((n_points, n_points))
    y_rot = xyz_rot[1, :].reshape((n_points, n_points))
    z_rot = xyz_rot[2, :].reshape((n_points, n_points))
    x_plot = x_rot + mean[0]
    y_plot = y_rot + mean[1]
    z_plot = z_rot + mean[2]
    ax.plot_surface(x_plot, y_plot, z_plot, rstride=2, cstride=2,
                    color=edgecolor, alpha=alpha, linewidth=0.5, edgecolors=edgecolor)

def extrude_curve_to_surface(x, z, y_center, width=0.5, n_y=5):
    """
    Given a 1D curve defined by x (domain) and z (amplitude),
    create a narrow surface by extruding the curve along the Y-axis.
    
    Args:
      x: 1D array of x-values.
      z: 1D array of z-values (same length as x).
      y_center: The central y-value (e.g., the test ID offset for that event).
      width: The total width in y-direction to extrude.
      n_y: Number of points in y-direction.
      
    Returns:
      X, Y, Z: 2D arrays suitable for plotting with ax.plot_surface.
    """
    Y = np.linspace(y_center - width/2, y_center + width/2, n_y)
    X, Y_mesh = np.meshgrid(x, Y)
    # Replicate z along the y-direction to create a surface:
    Z = np.tile(z, (n_y, 1))
    return X, Y_mesh, Z

#============= 1) Load Data ================
def load_data_as_arrays():
    """
    Loads accelerometer data and masks from your data_loader and returns them
    as numpy arrays (acc_samples, mask_samples) for easy processing.
    """
    accel_dict, mask_dict = data_loader.load_data()
    acc_samples = []
    mask_samples = []
    test_ids = []
    for test_id in accel_dict:
        samples = accel_dict[test_id]  # list of (12000, 12) arrays
        mask = mask_dict[test_id]      # (256, 768)
        for s in samples:
            acc_samples.append(s)
            mask_samples.append(mask)
            test_ids.append(test_id)
    acc_samples = np.stack(acc_samples)    # [num_samples, 12000, 12]
    mask_samples = np.stack(mask_samples)    # [num_samples, 256, 768]
    return acc_samples, mask_samples, test_ids

#============= Time-Series Impact Extraction =============
def extract_impacts(acc_samples, sample_rate=200, impact_window=5):
    """
    Identify impact events in time-series data and extract fixed-length time windows.
    
    Args:
        acc_samples: np.array of shape [num_samples, time_steps, num_channels]
        sample_rate: Sampling rate in Hz (default 200Hz)
        impact_window: Window size in seconds to extract (default 5s)
    
    Returns:
        extracted_events: np.array of shape [num_events, window_size, num_channels]
    """
    num_samples, time_steps, num_channels = acc_samples.shape
    window_size = impact_window * sample_rate 
    extracted_events = []
    for sample in acc_samples:
        rms_energy = np.sqrt(np.mean(sample ** 2, axis=1))
        threshold = np.percentile(rms_energy, 99)
        impact_indices = np.where(rms_energy > threshold)[0]
        for idx in impact_indices:
            start_idx = idx - window_size // 2
            end_idx = start_idx + window_size
            if start_idx < 0:
                start_idx = 0
                end_idx = window_size
            if end_idx > time_steps:
                end_idx = time_steps
                start_idx = time_steps - window_size
            if end_idx - start_idx == window_size:
                extracted_events.append(sample[start_idx:end_idx, :])
    extracted_events = np.array(extracted_events)
    return extracted_events

def plot_rms_threshold(sample, sample_rate=200):
    rms_energy = np.sqrt(np.mean(sample ** 2, axis=1))
    threshold = np.percentile(rms_energy, 99)
    plt.figure(figsize=(10, 4))
    plt.plot(rms_energy, label="RMS Energy")
    plt.axhline(threshold, color='r', linestyle='--', label="Threshold (99th percentile)")
    plt.xlabel("Time Steps")
    plt.ylabel("RMS Energy")
    plt.legend()
    plt.title("RMS Energy with Impact Detection Threshold")
    plt.show()

def extract_high_damage_masks(mask_samples, damage_percentile=90):
    """
    Extract high-damage masks by computing fraction of damaged pixels.
    """
    damage_fractions = np.mean(mask_samples, axis=(1, 2))
    threshold = np.percentile(damage_fractions, damage_percentile)
    high_damage_indices = np.where(damage_fractions >= threshold)[0]
    high_damage_masks = mask_samples[high_damage_indices]
    return high_damage_masks

#============= 3D GMM Plotting (Using Duplication workaround if needed) =============
def duplicate_points_by_weight(X, scale=1.0):
    """
    Duplicate each row in X in proportion to its amplitude (here, using column 2).
    """
    X_dup = []
    for row in X:
        weight = row[2] + 1e-6
        count = int(np.round(weight * scale))
        count = max(count, 1)
        for _ in range(count):
            X_dup.append(row)
    return np.array(X_dup)

def plot_3d_gmm(points_3d, n_components=5, n_std=1.0, weight_scale=1.0):
    """
    Fit a 3D GMM to points_3d using a weighted (via duplication) approach and plot the result.
    """
    X_dup = duplicate_points_by_weight(points_3d, scale=weight_scale)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X_dup)
    means = gmm.means_
    covs = gmm.covariances_
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                         c=points_3d[:, 2], cmap='viridis', alpha=0.5, s=5)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Z Value")
    for i in range(n_components):
        plot_3d_gaussian_ellipsoid(ax, means[i], covs[i], n_std=n_std, edgecolor='blue', alpha=0.15)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Weighted GMM with n_components={n_components}")
    plt.tight_layout()
    plt.show()

#============= 3D Time-Series Analysis Workflow =============
def plot_fft_curves_by_test(accel_dict, sampling_rate=200):
    """
    For each test in accel_dict, compute the FFT for each recording and each channel,
    and plot the FFT curves in 3D with the test ID as the y-axis.
    
    X-axis: Frequency (Hz)
    Y-axis: Test ID (with a slight offset for each recording)
    Z-axis: FFT amplitude
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    sorted_test_ids = sorted(accel_dict.keys())
    cmap = plt.get_cmap('tab20', 12)  # one color per channel
    
    for i, test_id in enumerate(sorted_test_ids):
        recordings = accel_dict[test_id]  # list of recordings for this test
        for j, rec in enumerate(recordings):
            # Offset along Y: test ID + small offset for recording index
            y_offset = i + j * 0.1
            T = rec.shape[0]
            freqs = np.fft.rfftfreq(T, d=1.0/sampling_rate)
            for ch in range(rec.shape[1]):
                fft_vals = np.abs(np.fft.rfft(rec[:, ch]))
                ax.plot(freqs, [y_offset]*len(freqs), fft_vals,
                        color=cmap(ch),
                        label=f"Ch {ch}" if (i==0 and j==0) else "")
    
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Test ID")
    ax.set_zlabel("FFT Amplitude")
    ax.set_title("FFT Curves for Each Recording Stacked by Test ID")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_rms_curves_by_test(accel_dict, sampling_rate=200):
    """
    For each test in accel_dict, compute the RMS (L2 norm across channels)
    for each recording and plot the time series (RMS over time) in 3D.
    
    X-axis: Time index
    Y-axis: Test ID (with a slight offset for each recording)
    Z-axis: RMS value
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    sorted_test_ids = sorted(accel_dict.keys())
    for i, test_id in enumerate(sorted_test_ids):
        recordings = accel_dict[test_id]
        for j, rec in enumerate(recordings):
            y_offset = i + j * 0.1
            # Compute RMS across channels for each time step
            rms_vals = np.linalg.norm(rec, axis=1)
            t = np.arange(len(rms_vals))
            ax.plot(t, [y_offset]*len(t), rms_vals, label=f"Test {test_id} rec {j}" if (i==0 and j==0) else "")
    
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Test ID")
    ax.set_zlabel("RMS Value")
    ax.set_title("RMS Curves for Each Recording Stacked by Test ID")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

def compute_event_features(event, sampling_rate=200):
    """
    Compute various features for a given impact event segment.
    
    event: np.array of shape (window_size, num_channels)
    
    Returns a dictionary with:
      'fft': (freqs, fft_combined) where fft_combined is the L2 norm across channels,
      'rms': the RMS time series (1D array),
      'max_rms': maximum RMS value,
      'psd': (freqs_psd, psd_avg) average PSD across channels,
      'raw': the raw event (as is).
    """
    # --- FFT Feature ---
    # Compute FFT for each channel (along time), then combine using L2 norm:
    fft_vals = np.abs(np.fft.rfft(event, axis=0))  # shape: [n_freqs, num_channels]
    fft_combined = np.linalg.norm(fft_vals, axis=1)
    freqs_fft = np.fft.rfftfreq(event.shape[0], d=1.0/sampling_rate)
    
    # --- RMS Feature ---
    # Compute RMS across channels for each time step:
    rms_series = np.sqrt(np.mean(event**2, axis=1))
    max_rms = np.max(rms_series)
    
    # --- PSD Feature ---
    psd_list = []
    freqs_psd = None
    for ch in range(event.shape[1]):
        f, pxx = welch(event[:, ch], fs=sampling_rate)
        psd_list.append(pxx)
        if freqs_psd is None:
            freqs_psd = f
    psd_avg = np.mean(np.array(psd_list), axis=0)
    
    return {
        'fft': (freqs_fft, fft_combined),
        'rms': rms_series,
        'max_rms': max_rms,
        'psd': (freqs_psd, psd_avg),
        'raw': event
    }

def plot_event_features(features, event_index=0):
    """
    Plot the features of a single impact event in a 2x2 subplot grid.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # FFT Plot
    freqs_fft, fft_combined = features['fft']
    axs[0, 0].plot(freqs_fft, fft_combined, color='purple')
    axs[0, 0].set_title(f"Event {event_index} - FFT")
    axs[0, 0].set_xlabel("Frequency (Hz)")
    axs[0, 0].set_ylabel("Combined FFT Amplitude")
    
    # RMS Plot
    axs[0, 1].plot(features['rms'], color='green')
    axs[0, 1].set_title(f"Event {event_index} - RMS Time Series")
    axs[0, 1].set_xlabel("Time Steps")
    axs[0, 1].set_ylabel("RMS Value")
    
    # PSD Plot
    freqs_psd, psd_avg = features['psd']
    axs[1, 0].plot(freqs_psd, psd_avg, color='blue')
    axs[1, 0].set_title(f"Event {event_index} - PSD")
    axs[1, 0].set_xlabel("Frequency (Hz)")
    axs[1, 0].set_ylabel("Power Spectral Density")
    
    # Raw Signal Plot (average across channels)
    raw_mean = np.mean(features['raw'], axis=1)
    axs[1, 1].plot(raw_mean, color='red')
    axs[1, 1].set_title(f"Event {event_index} - Raw Signal (Mean)")
    axs[1, 1].set_xlabel("Time Steps")
    axs[1, 1].set_ylabel("Amplitude")
    
    plt.tight_layout()
    plt.show()

def plot_raw_event_surfaces(accel_dict, sampling_rate=200, impact_window=5):
    """
    For each test and each recording, extract impact events and plot the raw signal
    (using the L2 norm across channels) as a surface.
    
    X-axis: Time index within event
    Y-axis: Test ID + a small offset per event
    Z-axis: Raw signal amplitude
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sorted_test_ids = sorted(accel_dict.keys())
    base_offset = 10  # spacing between tests
    for i, test_id in enumerate(sorted_test_ids):
        recordings = accel_dict[test_id]
        for j, rec in enumerate(recordings):
            events = extract_impacts(rec[np.newaxis, ...], sample_rate=sampling_rate, impact_window=impact_window)
            for k, event in enumerate(events):
                # Compute raw signal as L2 norm across channels (per time step)
                raw_curve = np.linalg.norm(event, axis=1)
                t = np.arange(len(raw_curve))
                # Unique Y offset per event
                y_offset = i * base_offset + j * 0.05 + k * 0.005
                X_surf, Y_surf, Z_surf = extrude_curve_to_surface(t, raw_curve, y_offset, width=0.2, n_y=5)
                ax.plot_surface(X_surf, Y_surf, Z_surf, color='blue', alpha=0.7)
    ax.set_xlabel("Time Index (within event)")
    ax.set_ylabel("Test ID + offset")
    ax.set_zlabel("Raw Amplitude")
    ax.set_title("Raw Signal Surfaces for Each Impact Event")
    plt.tight_layout()
    plt.show()

def plot_fft_event_surfaces(accel_dict, sampling_rate=200, impact_window=5):
    """
    For each test and each recording, segment impact events,
    compute the FFT (L2 norm across channels) for each event,
    and plot the FFT curve as a surface.
    
    X-axis: Frequency (Hz)
    Y-axis: Test ID + offset
    Z-axis: FFT amplitude
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sorted_test_ids = sorted(accel_dict.keys())
    base_offset = 10
    for i, test_id in enumerate(sorted_test_ids):
        recordings = accel_dict[test_id]
        for j, rec in enumerate(recordings):
            events = extract_impacts(rec[np.newaxis, ...], sample_rate=sampling_rate, impact_window=impact_window)
            for k, event in enumerate(events):
                T = event.shape[0]
                freqs = np.fft.rfftfreq(T, d=1.0/sampling_rate)
                fft_vals = np.abs(np.fft.rfft(event, axis=0))  # shape: [n_freqs, num_channels]
                # Combine across channels using L2 norm:
                fft_curve = np.linalg.norm(fft_vals, axis=1)
                y_offset = i * base_offset + j * 0.05 + k * 0.005
                X_surf, Y_surf, Z_surf = extrude_curve_to_surface(freqs, fft_curve, y_offset, width=0.2, n_y=5)
                ax.plot_surface(X_surf, Y_surf, Z_surf, color='purple', alpha=0.7)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Test ID + offset")
    ax.set_zlabel("FFT Amplitude")
    ax.set_title("FFT Surfaces for Each Impact Event")
    plt.tight_layout()
    plt.show()

def plot_rms_event_surfaces(accel_dict, sampling_rate=200, impact_window=5):
    """
    For each test and each recording, segment impact events,
    compute the RMS curve (L2 norm across channels) for each event,
    and plot it as a surface.
    
    X-axis: Time index (within event)
    Y-axis: Test ID + offset
    Z-axis: RMS value
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sorted_test_ids = sorted(accel_dict.keys())
    base_offset = 10
    for i, test_id in enumerate(sorted_test_ids):
        recordings = accel_dict[test_id]
        for j, rec in enumerate(recordings):
            events = extract_impacts(rec[np.newaxis, ...], sample_rate=sampling_rate, impact_window=impact_window)
            for k, event in enumerate(events):
                rms_curve = np.sqrt(np.mean(event**2, axis=1))
                t = np.arange(len(rms_curve))
                y_offset = i * base_offset + j * 0.05 + k * 0.005
                X_surf, Y_surf, Z_surf = extrude_curve_to_surface(t, rms_curve, y_offset, width=0.2, n_y=5)
                ax.plot_surface(X_surf, Y_surf, Z_surf, color='green', alpha=0.7)
    ax.set_xlabel("Time Index (within event)")
    ax.set_ylabel("Test ID + offset")
    ax.set_zlabel("RMS Value")
    ax.set_title("RMS Surfaces for Each Impact Event")
    plt.tight_layout()
    plt.show()

def plot_psd_event_surfaces(accel_dict, sampling_rate=200, impact_window=5):
    """
    For each test and each recording, segment impact events,
    compute the PSD (using Welch's method) for each event (averaged across channels),
    and plot the PSD curve as a surface.
    
    X-axis: Frequency (Hz)
    Y-axis: Test ID + offset
    Z-axis: PSD value
    """
    from scipy.signal import welch
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sorted_test_ids = sorted(accel_dict.keys())
    base_offset = 10
    for i, test_id in enumerate(sorted_test_ids):
        recordings = accel_dict[test_id]
        for j, rec in enumerate(recordings):
            events = extract_impacts(rec[np.newaxis, ...], sample_rate=sampling_rate, impact_window=impact_window)
            for k, event in enumerate(events):
                psd_list = []
                freqs_psd = None
                for ch in range(event.shape[1]):
                    f, pxx = welch(event[:, ch], fs=sampling_rate)
                    psd_list.append(pxx)
                    if freqs_psd is None:
                        freqs_psd = f
                psd_avg = np.mean(np.array(psd_list), axis=0)
                y_offset = i * base_offset + j * 0.05 + k * 0.005
                X_surf, Y_surf, Z_surf = extrude_curve_to_surface(freqs_psd, psd_avg, y_offset, width=0.2, n_y=5)
                ax.plot_surface(X_surf, Y_surf, Z_surf, color='orange', alpha=0.7)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Test ID + offset")
    ax.set_zlabel("PSD Value")
    ax.set_title("PSD Surfaces for Each Impact Event")
    plt.tight_layout()
    plt.show()

#========================== UMAP ==========================
def compute_event_features(event, sampling_rate=200):
    """
    Given an impact event (shape: [window_size, num_channels]),
    compute feature vectors for:
      - raw: flatten the event.
      - fft: compute FFT (L2 norm across channels) and return the resulting spectrum.
      - rms: compute the RMS time series (L2 norm across channels).
      - psd: compute the average PSD (using Welch's method) across channels.
    Returns a dictionary of feature vectors.
    """
    # Raw feature: flatten event
    raw_feature = event.flatten()
    
    # FFT feature:
    fft_vals = np.abs(np.fft.rfft(event, axis=0))  # shape: [n_freqs, num_channels]
    fft_feature = np.linalg.norm(fft_vals, axis=1)   # combine across channels
    # (We drop the frequency axis labels for UMAP)
    
    # RMS feature:
    rms_feature = np.sqrt(np.mean(event**2, axis=1))  # 1D array
    
    # PSD feature:
    from scipy.signal import welch
    psd_list = []
    for ch in range(event.shape[1]):
        f, pxx = welch(event[:, ch], fs=sampling_rate)
        psd_list.append(pxx)
    psd_feature = np.mean(np.array(psd_list), axis=0)  # average PSD across channels
    
    return {
        'raw': raw_feature,
        'fft': fft_feature,
        'rms': rms_feature,
        'psd': psd_feature
    }

def extract_events_features(accel_dict, sampling_rate=200, impact_window=5):
    """
    For each test in accel_dict, segment each recording into impact events,
    and compute a feature vector for each event (for raw, FFT, RMS, and PSD).
    Returns a dictionary with keys 'raw', 'fft', 'rms', and 'psd'.
    Each entry is a tuple (features, labels), where features is an array of shape (N, D)
    and labels contains the corresponding test ID (as an integer).
    """
    raw_feats, fft_feats, rms_feats, psd_feats, labels = [], [], [], [], []
    sorted_test_ids = sorted(accel_dict.keys())
    for i, test_id in enumerate(sorted_test_ids):
        recordings = accel_dict[test_id]  # list of recordings for this test
        for rec in recordings:
            # Extract impact events (each event is a [window_size, num_channels] array)
            events = extract_impacts(rec[np.newaxis, ...], sample_rate=sampling_rate, impact_window=impact_window)
            for event in events:
                feats = compute_event_features(event, sampling_rate=sampling_rate)
                raw_feats.append(feats['raw'])
                fft_feats.append(feats['fft'])
                rms_feats.append(feats['rms'])
                psd_feats.append(feats['psd'])
                labels.append(i)  # Test ID index
    return {
        'raw': (np.array(raw_feats), np.array(labels)),
        'fft': (np.array(fft_feats), np.array(labels)),
        'rms': (np.array(rms_feats), np.array(labels)),
        'psd': (np.array(psd_feats), np.array(labels))
    }

def plot_umap_features_3d_interactive(features, labels, title="UMAP 3D Plot"):
    """
    Reduce feature vectors to 3D using UMAP and create an interactive 3D scatter plot with Plotly.
    """
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding = reducer.fit_transform(features)
    
    # Build a Plotly scatter_3d
    fig = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=labels,
            colorscale='Viridis',
            opacity=0.7,
            colorbar=dict(title='Test ID')
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3"
        ),
        title=title
    )
    
    # Save to an interactive HTML file
    pio.write_html(fig, file=f"{title.replace(' ', '_')}.html", auto_open=False)
    print(f"Saved interactive plot to {title.replace(' ', '_')}.html")

#============= Other Analysis Functions (PCA, Clustering, etc.) =============
def basic_stats_and_histograms_focused(acc_samples, mask_samples):
    impact_events = extract_impacts(acc_samples)  # [num_events, 600, 12]
    high_damage_masks = extract_high_damage_masks(mask_samples, 90)
    acc_flat = impact_events.reshape(-1)
    print("Focused Time-Series Data (Impact Events) Stats:")
    print(f"  Mean: {np.mean(acc_flat):.4f}")
    print(f"  Std:  {np.std(acc_flat):.4f}")
    print(f"  Min:  {np.min(acc_flat):.4f}, Max: {np.max(acc_flat):.4f}")
    plt.figure(figsize=(6, 4))
    plt.hist(acc_flat, bins=100, color='blue', alpha=0.7)
    plt.title("Time-Series Amplitude Distribution (Impacts)")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    mask_flat = high_damage_masks.reshape(-1)
    print("\nHigh-Damage Mask Data Stats:")
    print(f"  Mean: {np.mean(mask_flat):.4f}")
    print(f"  Std:  {np.std(mask_flat):.4f}")
    print(f"  Min:  {np.min(mask_flat):.4f}, Max: {np.max(mask_flat):.4f}")
    plt.figure(figsize=(6, 4))
    plt.hist(mask_flat, bins=50, color='red', alpha=0.7)
    plt.title("Mask Value Distribution (High-Damage)")
    plt.xlabel("Value (0 or 1 for binary mask)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def frequency_analysis(acc_samples, sample_rate=200):
    num_samples = acc_samples.shape[0]
    n = acc_samples.shape[1]
    num_channels = acc_samples.shape[2]
    freqs = np.fft.rfftfreq(n, d=1.0/sample_rate)
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab20', num_channels)
    for ch in range(num_channels):
        all_fft = []
        for i in range(num_samples):
            signal = acc_samples[i, :, ch]
            fft_vals = np.abs(np.fft.rfft(signal))
            all_fft.append(fft_vals)
        all_fft = np.array(all_fft)
        avg_fft = np.mean(all_fft, axis=0)
        plt.plot(freqs, avg_fft, color=cmap(ch), label=f"Channel {ch}")
    plt.title("Average FFT Spectrum over All Samples for All Channels")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Average Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

def dimensionality_reduction(acc_samples, mask_samples, method='PCA', n_components=2):
    max_len = min(2000, acc_samples.shape[1])
    acc_small = acc_samples[:, :max_len, :].reshape(acc_samples.shape[0], -1)
    mask_ds = mask_samples[:, ::4, ::4].reshape(mask_samples.shape[0], -1)
    if method.lower() == 'pca':
        reducer_acc = PCA(n_components=n_components)
        reducer_mask = PCA(n_components=n_components)
        acc_reduced = reducer_acc.fit_transform(acc_small)
        mask_reduced = reducer_mask.fit_transform(mask_ds)
    else:
        raise ValueError("Only PCA is implemented in this example. Install UMAP if needed.")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(acc_reduced[:, 0], acc_reduced[:, 1], s=5, alpha=0.5, c='blue')
    axs[0].set_title("Time-Series Reduced")
    axs[1].scatter(mask_reduced[:, 0], mask_reduced[:, 1], s=5, alpha=0.5, c='red')
    axs[1].set_title("Mask Reduced")
    plt.tight_layout()
    plt.show()
    return acc_reduced, mask_reduced

def combined_dimensionality_reduction(acc_samples, mask_samples, method='PCA', n_components=2):
    max_len = min(2000, acc_samples.shape[1])
    acc_small = acc_samples[:, :max_len, :].reshape(acc_samples.shape[0], -1)
    mask_ds = mask_samples[:, ::4, ::4].reshape(mask_samples.shape[0], -1)
    combined = np.concatenate([acc_small, mask_ds], axis=1)
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
        combined_reduced = reducer.fit_transform(combined)
    else:
        raise ValueError("Only PCA is implemented in this example.")
    plt.figure(figsize=(5, 4))
    plt.scatter(combined_reduced[:, 0], combined_reduced[:, 1], s=5, alpha=0.5, c='purple')
    plt.title("Combined PCA of Time-Series + Mask")
    plt.tight_layout()
    plt.show()
    return combined_reduced

def clustering_example(features, method='gmm', n_clusters=3):
    if method.lower() == 'gmm':
        model = GaussianMixture(n_components=n_clusters)
    else:
        model = KMeans(n_clusters=n_clusters)
    cluster_labels = model.fit_predict(features)
    return cluster_labels

def correlation_between_summaries(acc_samples, mask_samples):
    num_samples = acc_samples.shape[0]
    acc_rms = []
    acc_max = []
    for i in range(num_samples):
        x = acc_samples[i]
        rms_val = np.sqrt(np.mean(x**2))
        max_val = np.max(np.abs(x))
        acc_rms.append(rms_val)
        acc_max.append(max_val)
    acc_rms = np.array(acc_rms)
    acc_max = np.array(acc_max)
    mask_fraction = []
    for i in range(num_samples):
        m = mask_samples[i]
        frac = np.mean(m)
        mask_fraction.append(frac)
    mask_fraction = np.array(mask_fraction)
    summary_feats = np.column_stack([acc_rms, acc_max, mask_fraction])
    corr_matrix = np.corrcoef(summary_feats.T)
    print("\nCorrelation between [acc_rms, acc_max, mask_fraction]:")
    print(corr_matrix)
    plt.imshow(corr_matrix, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(range(3), ['acc_rms', 'acc_max', 'mask_frac'])
    plt.yticks(range(3), ['acc_rms', 'acc_max', 'mask_frac'])
    plt.title("Correlation Matrix")
    plt.show()

def sum_and_fit_masks(mask_samples, num_clusters=3, method='GMM'):
    summed_mask = np.sum(mask_samples, axis=0).astype(float)
    summed_mask /= np.max(summed_mask)
    y_indices, x_indices = np.where(summed_mask > 0)
    coords = np.column_stack((x_indices, y_indices))
    if method == 'GMM':
        model = GaussianMixture(n_components=num_clusters, covariance_type='full',
                                random_state=42, n_init=10, max_iter=400)
    elif method == 'KMeans':
        model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    else:
        raise ValueError("Method should be 'GMM' or 'KMeans'.")
    model.fit(coords)
    cluster_centers = model.means_
    fig, ax = plt.subplots(figsize=(15, 7), facecolor='w')
    white_hot = create_white_hot_colormap()
    im = ax.imshow(summed_mask, cmap=white_hot, origin='upper', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Damage Frequency")
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
               marker='x', color='blue', s=150, label="Cluster Centers")
    if method == 'GMM':
        for i in range(num_clusters):
            mean = model.means_[i]
            cov = model.covariances_[i]
            plot_gaussian_ellipse(ax, mean, cov, n_std=1.0, edgecolor='blue', lw=2)
    ax.set_title(f"Summed Mask Heatmap with {num_clusters} Clusters ({method})", fontsize=16)
    ax.set_xlabel("X (Width)", fontsize=14)
    ax.set_ylabel("Y (Height)", fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def sum_and_fit_masks_3d(mask_samples, n_components=3, n_std=1.0):
    summed_mask = np.sum(mask_samples, axis=0).astype(float)
    max_val = np.max(summed_mask)
    if max_val > 0:
        summed_mask /= max_val
    H, W = summed_mask.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    xs_flat = xs.ravel()
    ys_flat = ys.ravel()
    vals_flat = summed_mask.ravel()
    nonzero_idx = np.where(vals_flat > 0)
    X_3D = np.column_stack((xs_flat[nonzero_idx],
                             ys_flat[nonzero_idx],
                             vals_flat[nonzero_idx]))
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X_3D)
    means = gmm.means_
    covs = gmm.covariances_
    fig = plt.figure(figsize=(12, 8), facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_3D[:, 0], X_3D[:, 1], X_3D[:, 2],
                         c=X_3D[:, 2], cmap='hot', alpha=0.5, s=2)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Normalized Damage Frequency (Z)")
    for i in range(n_components):
        plot_3d_gaussian_ellipsoid(ax, means[i], covs[i],
                                   n_std=n_std, edgecolor='blue', alpha=0.15)
    ax.set_title(f"3D GMM Fit to Summed Mask (n_components={n_components})")
    ax.set_xlabel("X (Width)")
    ax.set_ylabel("Y (Height)")
    ax.set_zlabel("Damage Frequency (Z)")
    plt.tight_layout()
    plt.savefig("output_plot.png", dpi=300, bbox_inches="tight")
    plt.close()  # Closes the figure to free memory


#============= 3D Time-Series Analysis Workflow (Modified) =============
def main_time_series_3d_analysis(accel_dict, sampling_rate=200):
    """
    Instead of combining the 5 recordings per test, compute features per recording.
    For each recording, extract:
      - FFT features: dominant frequency and its amplitude.
      - RMS features: time index of maximum RMS and the corresponding RMS value.
    The 3D points will have:
      For FFT: (dominant frequency, test index, max FFT amplitude)
      For RMS: (time index of max RMS, test index, max RMS value)
    Then fit a 3D GMM and plot each.
    """
    fft_points = []
    rms_points = []
    sorted_test_ids = sorted(accel_dict.keys())
    for i, test_id in enumerate(sorted_test_ids):
        recordings = accel_dict[test_id]  # list of arrays for each test
        for rec in recordings:
            T = rec.shape[0]
            # FFT features:
            fft_vals = np.abs(np.fft.rfft(rec, axis=0))  # shape: [n_freqs, num_channels]
            # Compute the overall spectrum by taking the L2 norm across channels:
            fft_norm = np.linalg.norm(fft_vals, axis=1)
            freqs = np.fft.rfftfreq(T, d=1.0/sampling_rate)
            idx_max_fft = np.argmax(fft_norm)
            dom_freq = freqs[idx_max_fft]
            max_fft_amp = fft_norm[idx_max_fft]
            fft_points.append([dom_freq, i, max_fft_amp])
            
            # RMS features:
            # Compute L2 norm across channels for each time step:
            rec_norm = np.linalg.norm(rec, axis=1)
            idx_max_rms = np.argmax(rec_norm)
            max_rms = rec_norm[idx_max_rms]
            rms_points.append([idx_max_rms, i, max_rms])
    
    fft_points = np.array(fft_points)
    rms_points = np.array(rms_points)
    
    plt.figure()
    plt.title("3D GMM Fit - FFT-based (Individual Recordings)")
    plot_3d_gmm(fft_points, n_components=5, n_std=1.0)
    
    plt.figure()
    plt.title("3D GMM Fit - RMS-based (Individual Recordings)")
    plot_3d_gmm(rms_points, n_components=5, n_std=1.0)

#============= Main: Putting it all together =============
def main():
    acc_samples, mask_samples, test_ids = load_data_as_arrays()
    # print(f"Loaded {acc_samples.shape[0]} samples total.")
    # Example: Plot RMS threshold on first sample
    # plot_rms_threshold(acc_samples[0])
    # Focused stats on impact events and high-damage masks
    # basic_stats_and_histograms_focused(acc_samples, mask_samples)
    # frequency_analysis(acc_samples, sample_rate=200)
    sum_and_fit_masks(mask_samples, num_clusters=30, method='GMM')
    sum_and_fit_masks_3d(mask_samples, n_components=30, n_std=1)
    # correlation_between_summaries(acc_samples, mask_samples)
    # 3D Time-Series Analysis: Use data from data_loader
    # Instead of combining the 5 recordings per test, we extract features per recording.
    accel_dict, _ = data_loader.load_data()
    # Run our modified time-series 3D analysis (both FFT-based and RMS-based) 
    # For FFT-based curves:
    # plot_fft_curves_by_test(accel_dict, sampling_rate=200)

    # # For RMS-based curves:
    # plot_rms_curves_by_test(accel_dict, sampling_rate=200)

    # # After loading acc_samples...
    # impacts = extract_impacts(acc_samples, sample_rate=200, impact_window=5)
    # print(f"Extracted {impacts.shape[0]} impact events.")

    # # For example, plot features for the first 5 events:
    # for idx in range(min(1, impacts.shape[0])):
    #     features = compute_event_features(impacts[idx], sampling_rate=200)
    #     plot_event_features(features, event_index=idx)
    
    # # Plot the event surfaces for each feature type:
    # plot_raw_event_surfaces(accel_dict, sampling_rate=200, impact_window=5)
    # plot_fft_event_surfaces(accel_dict, sampling_rate=200, impact_window=5)
    # plot_rms_event_surfaces(accel_dict, sampling_rate=200, impact_window=5)
    # plot_psd_event_surfaces(accel_dict, sampling_rate=200, impact_window=5)

    # Extract event features (raw, fft, rms, psd) for each impact event
    events_features = extract_events_features(accel_dict, sampling_rate=200, impact_window=5)
    
    # Apply UMAP and plot each feature type in 3D:
    plot_umap_features_3d_interactive(events_features['raw'][0], events_features['raw'][1],
                          title="UMAP 3D of Raw Event Features")
    plot_umap_features_3d_interactive(events_features['fft'][0], events_features['fft'][1],
                          title="UMAP 3D of FFT Event Features")
    plot_umap_features_3d_interactive(events_features['rms'][0], events_features['rms'][1],
                          title="UMAP 3D of RMS Event Features")
    plot_umap_features_3d_interactive(events_features['psd'][0], events_features['psd'][1],
                          title="UMAP 3D of PSD Event Features")



if __name__ == "__main__":
    main()
