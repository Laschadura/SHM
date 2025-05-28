import numpy as np
from scipy.signal import istft, get_window, welch
import cv2

# ISTFT
def inverse_spectrogram(
    complex_spectrograms: np.ndarray,
    time_length: int,
    fs: int = 200,
    nperseg: int = 256,
    noverlap: int = 224,
    batch_processing_size: int = 100
    ) -> np.ndarray:
    """
    Reconstruct time-domain signal from log-magnitude + phase.

    Args:
        complex_spectrograms: (N, F, T, 2C) = log(|S|+1), phase
        time_length: Desired output waveform length
        fs: Sample rate
        nperseg: STFT window size
        noverlap: Overlap between frames
        batch_processing_size: Batch chunk size (optional)

    Returns:
        np.ndarray: (N, time_length, C) — time-domain signals
    """
    frame_step = nperseg - noverlap
    N, F, T, double_C = complex_spectrograms.shape
    C = double_C // 2

    out = np.zeros((N, time_length, C), dtype=np.float32)
    win = get_window("hann", nperseg, fftbins=True)
    epsilon = 1e-5  # check if this is whats used in the compute_complex_spectrogram() function

    for batch_start in range(0, N, batch_processing_size):
        batch_end = min(batch_start + batch_processing_size, N)
        batch = complex_spectrograms[batch_start:batch_end]

        for i, i_abs in enumerate(range(batch_start, batch_end)):
            for c in range(C):
                log_mag = batch[i, :, :, 2*c]
                phase   = batch[i, :, :, 2*c + 1]
                mag = np.exp(log_mag) - epsilon
                real    = mag * np.cos(phase)
                imag    = mag * np.sin(phase)

                Z = real + 1j * imag
                _, x_rec = istft(Z, fs=fs, window=win,
                                 nperseg=nperseg, noverlap=noverlap,
                                 nfft=nperseg, input_onesided=True)

                # Truncate or zero-pad to match time_length
                x_out = np.zeros(time_length, dtype=np.float32)
                L     = min(time_length, len(x_rec))
                x_out[:L] = x_rec[:L]
                out[i_abs, :, c] = x_out

    return out

# Computes max corrolation between two waveforms and returns rolled sample and lag. Thats due to +/- hop/2 shifts after ISTFT
def align_by_xcorr(ref, test, max_lag=128):
    """
    Return test rolled so that it is maximally correlated with ref.
    Channels are aligned independently and then the majority lag
    (median over channels) is applied to all channels to keep them in sync.
    """
    lags = []
    for c in range(ref.shape[1]):
        corr = np.correlate(ref[:, c], test[:, c], mode="full")
        lag = np.argmax(corr) - (len(ref) - 1)
        lags.append(np.clip(lag, -max_lag, max_lag))

    lag = int(np.median(lags))
    rolled = np.roll(test, lag, axis=0)

    # zero the wrapped edge
    if lag > 0:
        rolled[-lag:, :] = 0.0
    elif lag < 0:
        rolled[:-lag, :] = 0.0

    return rolled, lag

# Mask Upsample
def mask_recon(downsampled_masks, target_size=(256, 768), interpolation=cv2.INTER_LINEAR):
    """
    Reconstruct full-resolution masks from downsampled ones (e.g., 32x96 -> 256x768).

    Args:
        downsampled_masks (np.ndarray): Array of shape (N, H_low, W_low, 1) or (N, H_low, W_low)
        target_size (tuple): Desired (height, width) in pixels, default (256, 768)
        interpolation (int): OpenCV interpolation method (e.g., cv2.INTER_LINEAR)

    Returns:
        np.ndarray: Reconstructed masks of shape (N, target_height, target_width)
    """
    # Ensure shape is (N, H, W)
    if downsampled_masks.ndim == 4 and downsampled_masks.shape[-1] == 1:
        downsampled_masks = np.squeeze(downsampled_masks, axis=-1)

    N, H, W = downsampled_masks.shape
    H_target, W_target = target_size
    recon_masks = np.zeros((N, H_target, W_target), dtype=np.float32)

    for i in range(N):
        recon_masks[i] = cv2.resize(
            downsampled_masks[i], (W_target, H_target), interpolation=interpolation
        )

    return recon_masks

# PSD analysis
def inspect_frequency_content(
        segments: np.ndarray,
        fs: float = 200.0,
        nfft: int = 1024,
        avg_over_segments: bool = False,
    ):
    """
    Compute the frequency content (Power Spectral Density, PSD) of segments.

    Args:
        segments (np.ndarray): Array of shape (N, T, C).
                               N = number of segments,
                               T = time steps,
                               C = number of channels.
        fs (float): Sampling frequency in Hz.
        nfft (int): Number of FFT points for Welch's method.
        avg_over_segments (bool): Whether to average PSDs across segments.

    Returns:
        f (np.ndarray): Frequency vector (Hz).
        psd (np.ndarray): 
            If avg_over_segments=True: shape (freq, channels).
            If avg_over_segments=False: shape (segments, freq, channels).
    """
    N, T, C = segments.shape
    frame_length = min(T, nfft)   # automatic handling if segment shorter than nfft
    all_psd = []

    for i in range(N):
        psd_per_seg = []
        for ch in range(C):
            f, Pxx = welch(segments[i, :, ch], fs=fs, nperseg=frame_length)
            psd_per_seg.append(Pxx)
        psd_per_seg = np.stack(psd_per_seg, axis=-1)  # (freq, channels)
        all_psd.append(psd_per_seg)

    all_psd = np.stack(all_psd, axis=0)  # (segments, freq, channels)

    if avg_over_segments:
        psd_avg = np.mean(all_psd, axis=0)  # (freq, channels)
        return f, psd_avg
    else:
        return f, all_psd
