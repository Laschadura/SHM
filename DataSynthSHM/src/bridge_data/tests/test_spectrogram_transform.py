# bridge_data/tests/test_spectrogram_transform.py
import numpy as np
import matplotlib.pyplot as plt
from bridge_data.loader import load_data
from bridge_data.preprocess import compute_complex_spectrogram

def test_spectrogram_transform():
    _, _, _, segments, _, _ ,_ ,_= load_data(recompute=False)

    sample = segments[0]  # shape: (T, C)
    spec = compute_complex_spectrogram(sample[None, ...])[0]  # (F, T, 2C)

    F, T, _ = spec.shape
    print(f"Spectrogram shape: {spec.shape} → F={F}, T={T}, 2C")

    mag = spec[:, :, 0]  # channel 0 log-magnitude

    plt.figure(figsize=(8, 4))
    plt.imshow(mag, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="log(|STFT| + ε)")
    plt.title("Log-Magnitude Spectrogram – Channel 0")
    plt.xlabel("Time bins")
    plt.ylabel("Frequency bins")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_spectrogram_transform()
