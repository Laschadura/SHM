import numpy as np
import matplotlib.pyplot as plt
from bridge_data.loader import load_data
from bridge_data.preprocess import compute_complex_spectrogram

# Parameters (match your training config)
FS = 200
NPERSEG = 256
NOVERLAP = 224
EPS = 1e-5   # same as in compute_complex_spectrogram

def main():
    print("🔄 Loading segments…")
    _, _, _, segments, *_ = load_data(recompute=False)

    print("🔊 Computing spectrograms…")
    specs = compute_complex_spectrogram(
        segments,
        fs       = FS,
        nperseg  = NPERSEG,
        noverlap = NOVERLAP,
    )  # shape: (N, F, T, 2C)

    print("📐 Computing linear magnitudes |S|…")
    C = specs.shape[-1] // 2
    mag = np.sqrt(specs[..., 0::2]**2 + specs[..., 1::2]**2)  # (N, F, T, C)

    mag_flat = mag.reshape(-1)
    print(f"🔍 Magnitude stats:")
    print(f"‣ Min:       {mag_flat.min():.6f}")
    print(f"‣ Max:       {mag_flat.max():.6f}")
    print(f"‣ Mean:      {mag_flat.mean():.6f}")
    print(f"‣ Std:       {mag_flat.std():.6f}")
    print(f"‣ 1st perc:  {np.percentile(mag_flat, 1):.6f}")
    print(f"‣ 99th perc: {np.percentile(mag_flat, 99):.6f}")
    print(f"‣ 99.9th p.: {np.percentile(mag_flat, 99.9):.6f}")

    # ---------- Plot histogram ----------
    plt.figure(figsize=(8, 4))
    plt.hist(mag_flat, bins=500, log=True, color='slateblue', alpha=0.8)
    plt.xlabel("|S| magnitude")
    plt.ylabel("Count (log)")
    plt.title("Histogram of |S| (linear magnitude)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------- Optional: log-magnitude histogram ----------
    log_mag = np.log(mag_flat + EPS)
    plt.figure(figsize=(8, 4))
    plt.hist(log_mag, bins=500, color='orangered', alpha=0.8)
    plt.xlabel("log(|S| + ε)")
    plt.title("Histogram of log-magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    log_mag = np.log(mag_flat + EPS)
    print("\n🔍 Log-magnitude stats:")
    print(f"‣ Min:       {log_mag.min():.6f}")
    print(f"‣ Max:       {log_mag.max():.6f}")
    print(f"‣ Mean:      {log_mag.mean():.6f}")
    print(f"‣ Std:       {log_mag.std():.6f}")
    print(f"‣ 1st perc:  {np.percentile(log_mag, 1):.6f}")
    print(f"‣ 99th perc: {np.percentile(log_mag, 99):.6f}")
    print(f"‣ 99.9th p.: {np.percentile(log_mag, 99.9):.6f}")


if __name__ == "__main__":
    main()
