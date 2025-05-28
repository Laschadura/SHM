import numpy as np
import matplotlib.pyplot as plt
import random
from bridge_data.loader import load_data
from bridge_data.preprocess import highpass_filter, preprocess_segment, compute_complex_spectrogram

def test_preprocessing_pipeline(test_id=None, fs=200, win_sec=4.0, nperseg=256, noverlap=224, epsilon=1e-5):
    # 1. Load raw traces
    accel_dict, *_ = load_data(recompute=False)
    if test_id is None:
        test_id = random.choice(list(accel_dict.keys()))

    ts_raw = random.choice(accel_dict[test_id])  # shape (T, C)
    print(f"🧪 Using test ID {test_id} with shape {ts_raw.shape}")

    # 2. Compute RMS and find peak
    rms = np.sqrt(np.mean(highpass_filter(ts_raw, 2.0, fs, 4)**2, axis=1))
    peak_idx = int(np.argmax(rms))

    # 3. Extract window
    win_len = int(fs * win_sec)
    half_win = win_len // 2
    start = max(0, peak_idx - half_win)
    end = min(start + win_len, ts_raw.shape[0])
    if end - start < win_len:
        start = end - win_len
    segment = ts_raw[start:end, :]

    # 4. Preprocess steps (correct order)
    seg_hpf = highpass_filter(segment, cutoff=10.0, fs=fs, order=6)
    seg_centered = seg_hpf - seg_hpf.mean(axis=0, keepdims=True)
    seg_std = np.maximum(seg_centered.std(axis=0, keepdims=True), 1e-8)
    seg_final = seg_centered / seg_std

    stages = [segment, seg_hpf, seg_final]
    labels = ["Original (Raw)", "After HPF", "Final (HPF + Center + σ-norm)"]


    # 5. Plot RMS with chosen window
    t_full = np.arange(ts_raw.shape[0]) / fs
    thresh = np.percentile(rms, 97.5)
    plt.figure(figsize=(12, 3))
    plt.plot(t_full, rms, label="RMS")
    plt.axhline(thresh, color="orange", linestyle="--", label="Threshold (97.5%)")
    plt.axvline(peak_idx / fs, color="crimson", linestyle="--", label="Max RMS")
    plt.axvspan(start / fs, end / fs, color="gold", alpha=0.25, label="Chosen window")
    plt.title(f"Full-trace RMS (Test ID {test_id})")
    plt.xlabel("Time [s]")
    plt.ylabel("RMS")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6. Plot preprocessing stages
    t_seg = np.arange(win_len) / fs
    fig, axes = plt.subplots(len(stages), 1, figsize=(12, 2.5 * len(stages)))
    for ax, data, label in zip(axes, stages, labels):
        for c in range(data.shape[1]):
            ax.plot(t_seg, data[:, c], lw=0.8, alpha=0.7)
        ax.set_title(label, loc="left")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Preprocessing steps – {win_sec}s window (Test ID {test_id})", y=1.02)
    fig.tight_layout()
    plt.show()

    # 7. STFT of preprocessed segment
    spec = compute_complex_spectrogram(seg_final[None, ...], fs, nperseg, noverlap)[0]
    logmag = spec[:, :, 0]
    phase = spec[:, :, 1]
    mag = np.exp(logmag) - epsilon

    # 8. Plot STFT and log-mag
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    im0 = ax[0].imshow(mag, origin="lower", aspect="auto", cmap="magma")
    ax[0].set_title("Magnitude |STFT|")
    ax[0].set_xlabel("Time bins")
    ax[0].set_ylabel("Frequency bins")
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(logmag, origin="lower", aspect="auto", cmap="viridis")
    ax[1].set_title("Log-Magnitude log(|STFT| + ε)")
    ax[1].set_xlabel("Time bins")
    ax[1].set_ylabel("Frequency bins")
    fig.colorbar(im1, ax=ax[1])

    fig.suptitle("Spectrogram Visualization (Channel 0)", y=1.02)
    plt.tight_layout()
    plt.show()

    # 9. Histogram of log-magnitude values
    plt.figure(figsize=(6, 3))
    plt.hist(logmag.ravel(), bins=100, color='purple')
    plt.title("Distribution of log(|STFT| + ε) values")
    plt.xlabel("log-mag value")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 🔁 10. Histogram of globally normalized spec values
    # Use train-set μ and σ (simulate this step — replace with your actual values)
    mu_train = logmag.mean()
    std_train = logmag.std() + 1e-8
    logmag_norm = (logmag - mu_train) / std_train

    plt.figure(figsize=(6, 3))
    plt.hist(logmag_norm.ravel(), bins=100, color='teal')
    plt.title("Distribution of normalized log-magnitude values")
    plt.xlabel("Normalized log-mag")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_preprocessing_pipeline()
