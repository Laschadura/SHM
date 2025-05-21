import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import glob
import re
import random
import cv2
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.interpolate import splprep, splev

######################################
# Configuration
######################################
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "Data")
LABELS_DIR = os.path.join(BASE_DIR, "data", "Labels")
IMAGE_SHAPE = (256, 768)
SKIP_TESTS = [23, 24]
EXPECTED_LENGTH = 12000

perspective_map = {
    'A': 'Arch_Intrados',
    'B': 'North_Spandrel_Wall',
    'C': 'South_Spandrel_Wall'
}

# ────────────  A U G M E N T A T I O N  ──────────────────────────
_FRAME_STEP  = 32        #  nperseg 256 – noverlap 224  ⇒ 256-224 = 32
_MAX_SHIFT   = 160       #  ≤0.8 s at 200 Hz  (tweakable)
_PINK_ALPHA  = 0.5       #  0→white, 0.5→pink, 1→brown

def _make_pink(shape, alpha=0.5):
    """TensorFlow-compatible 1/f^alpha pink noise generator (approx)."""
    T = shape[0]
    C = shape[1]

    # Ensure FFT length is next power of 2 for efficiency
    fft_len = tf.cast(tf.math.pow(2.0, tf.math.ceil(tf.math.log(tf.cast(T, tf.float32)) / tf.math.log(2.0))), tf.int32)

    # Frequency shaping
    freqs = tf.cast(tf.range(1, fft_len // 2 + 1), tf.float32)
    scale = tf.pow(freqs, -alpha)[:, tf.newaxis]  # (F,1)

    # Random phase and magnitude
    real = tf.random.normal([fft_len // 2, C])
    imag = tf.random.normal([fft_len // 2, C])
    comp = tf.complex(real * scale, imag * scale)

    # Mirror to get full spectrum (DC + pos + neg freq)
    comp_full = tf.concat(
        [tf.complex(tf.zeros([1, C]), tf.zeros([1, C])), comp, tf.reverse(tf.math.conj(comp), axis=[0])],
        axis=0
    )

    # IFFT to time-domain
    pink = tf.signal.ifft(comp_full)
    pink = tf.math.real(pink[:T])  # (T,C)
    return tf.cast(pink, tf.float32)

def augment_fn(spec, mask, tid, wave):
    """Synchronised spec + wave augmentation."""
    # 1. random time-shift
    samp_shift  = tf.random.uniform([], -_MAX_SHIFT, _MAX_SHIFT, tf.int32)
    frame_shift = samp_shift // _FRAME_STEP
    wave = tf.roll(wave,  samp_shift,  axis=0)
    spec = tf.roll(spec, frame_shift, axis=1)

    # 2. random gain  ±3 dB
    gain = tf.pow(10.0, tf.random.uniform([], -3.0, 3.0) / 20.0)
    wave = wave * gain
    mag  = spec[..., 0::2] * gain               # scale magnitude only
    spec = tf.concat([mag, spec[..., 1::2]], -1)

    # 3. sensor-axis sign-flip  (25 %)
    flip = tf.random.uniform([]) > 0.25
    wave = tf.where(flip, -wave, wave)
    phase = spec[..., 1::2] + tf.where(flip, np.pi, 0.0)
    spec  = tf.concat([spec[..., 0::2], phase], -1)

    # 4. pink noise, σ≈0.005
    noise = _make_pink(tf.shape(wave)) * 0.005
    wave  = wave + noise

    # 5. SpecAug-style frequency drop (1–3 bins), applied with 30% probability
    if tf.random.uniform([]) < 0.3:
        f_drop = tf.random.uniform([], 1, 4, tf.int32)
        f0     = tf.random.uniform([], 0, tf.shape(spec)[0] - f_drop, tf.int32)
        zeros  = tf.zeros_like(spec[f0:f0+f_drop, :, 0::2])
        spec = tf.tensor_scatter_nd_update(
            spec,
            indices=tf.range(f0, f0 + f_drop)[:, None],
            updates=tf.concat([zeros,
                               spec[f0:f0 + f_drop, :, 1::2]], -1))


    return spec, mask, tid, wave
# ─────────────────────────────────────────────────────────────────


######################################
# Time-series data pre- and postprocessing
######################################
def highpass_filter(data, cutoff=10.0, fs=200.0, order=4):
    """
    Apply a Butterworth high-pass filter to 1D or 2D time-series data.

    Args:
        data: NumPy array of shape (N,) or (N, C). If 2D, filters each channel independently.
        cutoff: High-pass cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Order of the Butterworth filter.
    
    Returns:
        Filtered data with same shape as input.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    if data.ndim == 1:
        return filtfilt(b, a, data, axis=0)

    filtered = np.zeros_like(data)
    for c in range(data.shape[1]):
        filtered[:, c] = filtfilt(b, a, data[:, c], axis=0)
    return filtered

def load_accelerometer_data(data_dir=DATA_DIR, skip_tests=SKIP_TESTS):
    """
    Load raw accelerometer CSV files for all test directories.

    Args:
        data_dir: Path to the dataset directory.
        skip_tests: List of test IDs to ignore.
    
    Returns:
        Dictionary mapping test ID to a list of raw (time × channel) arrays.
    """
    test_dirs = [d for d in glob.glob(os.path.join(data_dir, "Test_*")) if os.path.isdir(d)]
    tests_data = {}

    for test_dir in test_dirs:
        match = re.search(r"Test[_]?(\d+)", os.path.basename(test_dir))
        if not match:
            continue
        test_id = int(match.group(1))
        if test_id in skip_tests:
            continue

        csv_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
        if not csv_files:
            continue

        samples = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                accel_cols = [col for col in df.columns if "Accel" in col]
                if not accel_cols:
                    continue

                data_matrix = df[accel_cols].values.astype(np.float32)
                if data_matrix.shape[0] > EXPECTED_LENGTH:
                    data_matrix = data_matrix[:EXPECTED_LENGTH, :]
                samples.append(data_matrix)

            except Exception as e:
                print(f"Skipping file {csv_file} due to error: {e}")
                continue

        if samples:
            tests_data[test_id] = samples

    return tests_data

def preprocess_segment(
        seg: np.ndarray,
        fs: int = 200,
        rms_norm: bool = True,
        peak_norm: bool = False,
    ):
    """
    Parameters
    ----------
    seg : (T, C)      raw window (float32)
    hp_cut : float    high‑pass cut‑off in Hz for drift removal
    rms_norm : bool   divide each channel by its σ  (recommended)
    peak_norm: bool   divide each channel by its max(|x|)
                     (use at most **one** of rms_norm / peak_norm)

    Returns
    -------
    seg_proc : (T, C)  float32
    """
    # -- 1) remove per‑channel mean ---------------------------------
    seg = seg - seg.mean(axis=0, keepdims=True)

    # -- 2) scale to comparable energy ------------------------------
    if rms_norm and peak_norm:
        raise ValueError("Choose either rms_norm OR peak_norm – not both.")

    if rms_norm:                    # preferred
        sigma = np.maximum(seg.std(axis=0, keepdims=True), 1e-8)
        seg   = seg / sigma
    elif peak_norm:
        peak  = np.maximum(np.abs(seg).max(axis=0, keepdims=True), 1e-8)
        seg   = seg / peak

    return seg.astype(np.float32)

def segment_and_transform(
    accel_dict: dict,
    heatmap_dict: dict,
    sample_rate: int   = 200,
    segment_duration: float = 4.0,
    percentile: float = 99.9,
    min_separation: float = 0.25,          # s – ignore peaks closer than this
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice every raw trace into fixed‑length windows centred on the largest‑RMS
    instants and apply per‑segment preprocessing (demean ➜ HP‑filter ➜ min‑max).

    Returns
    -------
    raw_segments : (N , win , C)  float32  [-1,1]
    mask_segments: (N , H  , W)   float32
    test_ids     : (N,)           int32
    """
    win          = int(round(sample_rate * segment_duration))          # samples
    half_win     = win // 2
    min_sep_samp = int(round(sample_rate * min_separation))

    all_segs, all_masks, all_ids = [], [], []
    seg_counts = {}

    for tid, traces in accel_dict.items():
        mask = heatmap_dict[tid]

        for ts in traces:                                             # (T,C)
            # ------- 1.  find candidate peak positions ---------------
            ts_hp  = highpass_filter(ts, cutoff=10.0, fs=sample_rate, order=6)
            rms    = np.sqrt(np.mean(ts_hp ** 2, axis=1))             # (T,)
            thresh = np.percentile(rms, percentile)
            peaks  = np.where(rms >= thresh)[0]

            # keep only well‑separated peaks (largest first)
            peaks = peaks[np.argsort(rms[peaks])[::-1]]
            selected = []
            for p in peaks:
                if all(abs(p - q) > min_sep_samp for q in selected):
                    selected.append(p)
            # ---------------------------------------------------------

            for pk in selected:
                start = pk - half_win
                end   = start + win                                       # non‑inclusive

                # clip to valid range and pad if needed
                pad_before = max(0, -start)
                pad_after  = max(0, end - ts.shape[0])
                start      = max(start, 0)
                end        = min(end, ts.shape[0])

                seg = ts[start:end, :]
                if pad_before or pad_after:
                    seg = np.pad(seg,
                                 ((pad_before, pad_after), (0, 0)),
                                 mode="constant")

                if seg.shape[0] != win:          # safety – should not happen
                    continue

                # ---- per‑segment preprocessing ----------------------
                seg = preprocess_segment(
                        seg,
                        fs=sample_rate,
                        rms_norm=True,        # σ‑normalisation
                        peak_norm=False
                )
                # ------------------------------------------------------

                all_segs.append(seg.astype(np.float32))
                all_masks.append(mask.astype(np.float32))
                all_ids.append(tid)
                seg_counts[tid] = seg_counts.get(tid, 0) + 1

    print("Segment counts:", seg_counts)
    print(f"✅ Extracted {len(all_segs)} segments of "
          f"{segment_duration:.1f} s ({win} samples) each.")

    return (
        np.stack(all_segs,  axis=0),
        np.stack(all_masks, axis=0),
        np.array(all_ids,   dtype=np.int32)
    )


#--------------- Spectogram computation ---------------
def compute_or_load_spectrograms(raw_segments, fs=200, nperseg=256, noverlap=192):
    """
    Compute or load cached spectrograms.
    
    Args:
        raw_segments: Raw time series (N, time_length, channels)
        fs, nperseg, noverlap: STFT parameters
        cache_path: File path to save/load spectrograms
        
    Returns:
        Spectrogram features (N, freq_bins, time_bins, channels*2)
    """
    print("⏳ Computing STFT for all segments...")
    complex_spectrograms = compute_complex_spectrogram(raw_segments, fs, nperseg, noverlap)
    return complex_spectrograms

def compute_complex_spectrogram(
    time_series,
    fs=200,
    nperseg=256,
    noverlap=192
    ):
    """
    Compute STFT-based spectrograms with PyTorch.
    
    Args:
        time_series: shape (batch_size, time_steps, channels) — numpy or torch tensor
        fs: Sampling frequency in Hz
        nperseg: Window length for STFT
        noverlap: Overlap between windows

    Returns:
        Spectrograms with shape (batch, freq_bins, time_bins, channels * 2)
    """
    if isinstance(time_series, np.ndarray):
        time_series = torch.tensor(time_series, dtype=torch.float32)

    batch_size, time_steps, channels = time_series.shape
    frame_step = nperseg - noverlap
    window = torch.hann_window(nperseg)

    # Compute STFT shape using a test sample
    test_stft = torch.stft(
        time_series[0, :, 0],
        n_fft=nperseg,
        hop_length=frame_step,
        win_length=nperseg,
        window=window,
        return_complex=True
    )

    freq_bins, time_bins = test_stft.shape
    print(f"🔍 STFT Config: nperseg={nperseg}, noverlap={noverlap}, frame_step={frame_step}")
    print(f"📏 Expected STFT shape: (freq_bins={freq_bins}, time_bins={time_bins})")

    if time_bins == 0:
        raise ValueError("⚠️ STFT produced 0 time bins! Adjust `nperseg` or `noverlap`.")

    # Pre-allocate spectrograms: (batch, freq_bins, time_bins, channels * 2)
    all_spectrograms = torch.zeros(batch_size, freq_bins, time_bins, channels * 2)

    for i in range(batch_size):
        for c in range(channels):
            stft = torch.stft(
                time_series[i, :, c],
                n_fft=nperseg,
                hop_length=frame_step,
                win_length=nperseg,
                window=window,
                return_complex=True
            )

            if stft.shape[1] == 0:
                raise ValueError(f"⚠️ STFT returned 0 time bins for sample {i}, channel {c}!")

            # Magnitude and phase
            mag = torch.log1p(torch.abs(stft))  # log(1 + |stft|)
            phase = torch.angle(stft)           # phase in radians

            # Store in final tensor (transpose to match shape: freq, time)
            all_spectrograms[i, :, :, 2*c]   = mag
            all_spectrograms[i, :, :, 2*c+1] = phase
  

    print(f"✅ Final spectrogram shape: {all_spectrograms.shape}")
    return all_spectrograms.numpy()

#---------------- caching and time-series reconstruction ----------------
def cache_final_features(complex_specs, cache_path="cached_spectral_features.npy"):
    """
    If 'cache_path' exists, load it via mmap. Otherwise,
    convert 'complex_specs' to magnitude+phase features,
    save to disk, then memory-map.
    """
    if os.path.exists(cache_path):
        print(f"📂 Loading final spectral features from {cache_path}")
        return np.load(cache_path)
    
    # Save the final shape
    np.save(cache_path, complex_specs)
    print(f"✅ Final spectral features saved to {cache_path}")

    return np.load(cache_path)

def inverse_spectrogram(
    complex_spectrograms,
    time_length,
    fs=200,
    nperseg=256,
    noverlap=224,
    batch_processing_size=100
    ):
    """
    Convert magnitude + phase spectrograms back to time-domain signals using PyTorch.
    
    Args:
        complex_spectrograms: np.ndarray of shape (batch, freq, time, channels*2)
        time_length: Desired output length in time steps
        fs: Sampling frequency (unused, for compatibility)
        nperseg: Frame size (window length)
        noverlap: Overlap between frames
        batch_processing_size: Number of samples processed at once

    Returns:
        np.ndarray: Time-domain signal, shape (batch, time_length, channels)
    """
    if isinstance(complex_spectrograms, np.ndarray):
        complex_spectrograms = torch.tensor(complex_spectrograms, dtype=torch.float32)

    frame_step = nperseg - noverlap
    batch_size, freq_bins, time_bins, double_channels = complex_spectrograms.shape
    num_channels = double_channels // 2
    window = torch.hann_window(nperseg)

    time_series = torch.zeros((batch_size, time_length, num_channels), dtype=torch.float32)

    total_batches = (batch_size + batch_processing_size - 1) // batch_processing_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_processing_size
        end_idx = min((batch_idx + 1) * batch_processing_size, batch_size)

        batch_spec = complex_spectrograms[start_idx:end_idx]

        for b_rel, b_abs in enumerate(range(start_idx, end_idx)):
            for c in range(num_channels):
                # Extract magnitude and phase
                log_mag = batch_spec[b_rel, :, :, 2*c]
                phase = batch_spec[b_rel, :, :, 2*c+1]
                magnitude = torch.expm1(log_mag)
                complex_spec = magnitude * torch.exp(1j * phase)

                # Transpose to shape [time, freq] for PyTorch
                stft_input = complex_spec  # (time_bins, freq_bins)

                # Perform inverse STFT
                waveform = torch.istft(
                    stft_input,
                    n_fft=nperseg,
                    hop_length=frame_step,
                    win_length=nperseg,
                    window=window,
                    length=time_length
                )

                # Save reconstructed waveform
                time_series[b_abs, :waveform.shape[0], c] = waveform

    return time_series.numpy()

# trainable TensorFlow inverse stft for MMVAE
@tf.keras.saving.register_keras_serializable('mmvae')
class TFInverseISTFT(tf.keras.layers.Layer):
    def __init__(self, frame_length=256, frame_step=32, **kwargs):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step   = frame_step
        self.window_fn = tf.signal.inverse_stft_window_fn(
            frame_step,
            forward_window_fn=tf.signal.hann_window
        )
       
        self.beta = None                            # placeholder

    def build(self, input_shape):
        if self.beta is None:                       # create only once
            self.beta = self.add_weight(
                name="beta",
                shape=(),
                initializer=tf.keras.initializers.Ones(),
                trainable=True,
            )
    # ----------------------------------------------------------------

    def call(self, spec_logits, length):
        B = tf.shape(spec_logits)[0]
        D = tf.shape(spec_logits)[3]
        C = D // 2

        # Clip to prevent extreme values before softplus
        log_mag = tf.clip_by_value(spec_logits[..., 0::2], -8.0, 5.0)
        phase   = spec_logits[..., 1::2]

        # Learnable softplus for magnitude
        mag = tf.nn.softplus(self.beta * log_mag)

        mag   = tf.reshape(mag,   [B*C, tf.shape(mag)[1], tf.shape(mag)[2]])
        phase = tf.reshape(phase, [B*C, tf.shape(phase)[1], tf.shape(phase)[2]])

        real = mag * tf.cos(phase)
        imag = mag * tf.sin(phase)
        real = tf.cast(real, tf.float32)
        imag = tf.cast(imag, tf.float32)

        complex_spec = tf.complex(real, imag)
        complex_spec = tf.transpose(complex_spec, [0, 2, 1])  # [B*C, T, F]

        wav = tf.signal.inverse_stft(
            complex_spec,
            self.frame_length,
            self.frame_step,
            window_fn=self.window_fn
        )

        # pad or crop
        wav = wav[:, :length]
        wav = tf.pad(wav, [[0, 0], [0, tf.maximum(0, length - tf.shape(wav)[1])]])

        # [B*C, L] → [B, L, C]
        wav = tf.reshape(wav, [B, C, length])
        wav = tf.transpose(wav, [0, 2, 1])
        return wav

    def get_config(self):
        config = super().get_config()
        config.update({
            "frame_length": self.frame_length,
            "frame_step": self.frame_step
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

######################################
# Mask pre- and postprocessing
######################################
def load_perspective_image(test_id, perspective, labels_dir=LABELS_DIR, target_size=(256,256)):
    # Your existing code - unchanged
    label_name = perspective_map.get(perspective)
    file_path = os.path.join(labels_dir, f"Test_{test_id}", f"{label_name}_T{test_id}.png")

    if not os.path.exists(file_path):
        return None

    img = cv2.imread(file_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img_resized

def load_combined_label(test_id, labels_dir=LABELS_DIR, image_shape=IMAGE_SHAPE):
    # Your existing code - unchanged
    images = [load_perspective_image(test_id, p, labels_dir, (image_shape[0], image_shape[1] // 3)) for p in ['A', 'B', 'C']]
    images = [img if img is not None else np.zeros((image_shape[0], image_shape[1] // 3, 3), dtype=np.uint8) for img in images]
    return np.concatenate(images, axis=1)

def compute_binary_mask(combined_image):
    hsv = cv2.cvtColor(combined_image, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    return cv2.bitwise_or(mask1, mask2).astype(np.uint8)

def mask_to_heatmap(orig_mask,
                    target_size=(32, 96),
                    interpolation=cv2.INTER_AREA,
                    apply_blur=False,
                    blur_kernel=(3,3),
                    binarize=False,
                    threshold=0.0001):
    """
    Converts a high-res binary mask (H×W) to a coarse or binary downsampled mask.
    """
    if orig_mask.max() > 1:
        orig_mask = (orig_mask > 0).astype(np.float32)

    newH, newW = target_size
    heatmap = cv2.resize(orig_mask, (newW, newH), interpolation=interpolation)

    if apply_blur:
        heatmap = cv2.GaussianBlur(heatmap, blur_kernel, sigmaX=0)

    heatmap = np.clip(heatmap, 0.0, 1.0)

    if binarize:
        heatmap = (heatmap > threshold).astype(np.float32)

    return heatmap

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

#######################################
# Postprocessing analysis
#######################################
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


#######################################
# Data Loading module
#######################################
def load_data(segment_duration: float = 4.0,
              nperseg: int        = 256,
              noverlap: int       = 224,
              sample_rate: int    = 200,
              recompute: bool     = False,
              cache_dir: str      = "cache"):
    """
    Unified loader.
    ───────────────
    * Always returns the three dictionaries you already use
      (accel_dict, binary_masks, heatmaps).
    * Additionally returns:
        segments      : (N, win, C)    pre‑processed windows
        spectrograms  : (N, F, T, 2C)  log‑mag | phase
        test_ids      : (N,)           int32  – ID per segment
    * `segment_duration`, `nperseg`, `noverlap`
      are **baked into the cache file‑names** so multiple
      configs can coexist.
    """
    # -----------------------------------------------------------------
    accel_dict, binary_masks, heatmaps = load_accelerometer_data(
        DATA_DIR, SKIP_TESTS), {}, {}

    for tid in sorted(accel_dict.keys()):
        if tid in SKIP_TESTS:
            continue
        comb = load_combined_label(tid, LABELS_DIR, IMAGE_SHAPE)
        bin_mask = compute_binary_mask(comb)
        binary_masks[tid] = bin_mask
        heatmap = mask_to_heatmap(
            bin_mask,
            target_size=(32, 96),
            interpolation=cv2.INTER_AREA,  # soft downsampling
            binarize=True,                 # binarize afterward
            threshold=0.03                 # or adjust if needed
        )
        heatmaps[tid] = heatmap[..., None]                 # (32,96,1)
    # -----------------------------------------------------------------
    # ---- caching ----
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"{segment_duration:.2f}s_{nperseg}_{noverlap}"
    seg_path  = os.path.join(cache_dir, f"segments_{tag}.npy")
    id_path   = os.path.join(cache_dir, f"segIDs_{tag}.npy")
    spec_path = os.path.join(cache_dir, f"specs_{tag}.npy")

    if not recompute and all(os.path.exists(p) for p in (seg_path, id_path, spec_path)):
        print("📂  Loading segments & spectrograms from cache …")
        segments     = np.load(seg_path,  mmap_mode="r")
        test_ids     = np.load(id_path,   mmap_mode="r")
        spectrograms = np.load(spec_path, mmap_mode="r")
        return (accel_dict, binary_masks, heatmaps,
                segments, spectrograms, test_ids)

    # -----------------------------------------------------------------
    print("✂️  Segmenting traces …")
    segments, masks, test_ids = segment_and_transform(
        accel_dict, heatmaps,
        sample_rate      = sample_rate,
        segment_duration = segment_duration)

    print("🔄  Computing/ caching spectrograms …")
    spectrograms = compute_or_load_spectrograms(
        segments,
        fs       = sample_rate,
        nperseg  = nperseg,
        noverlap = noverlap)

    np.save(seg_path,  segments)
    np.save(id_path,   test_ids)
    np.save(spec_path, spectrograms)
    print(f"✅  Cached → {cache_dir}")

    return (accel_dict, binary_masks, heatmaps,
            segments, spectrograms, test_ids)


######################################
# For testing and visualization
######################################
def main():
    _, binary_masks, _, *_ = load_data(recompute=False)

    # Pick a test ID
    test_id = 25
    highres_mask = binary_masks[test_id]  # shape (256, 768)

    # Downsample using 3 strategies
    heatmap_soft = mask_to_heatmap(highres_mask, (32, 96), interpolation=cv2.INTER_AREA)
    heatmap_bin  = mask_to_heatmap(highres_mask, (32, 96), interpolation=cv2.INTER_NEAREST)
    heatmap_thresh = mask_to_heatmap(highres_mask, (32, 96),
                                    interpolation=cv2.INTER_AREA,
                                    binarize=True,
                                    threshold=0.03)

    # Add channel dimension for upsampling
    heatmap_thresh_exp = heatmap_thresh[None, ..., None]
    up_thresh = mask_recon(heatmap_thresh_exp)[0]

    # Plot all
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    ax[0].imshow(highres_mask,    cmap="gray");  ax[0].set_title("High-res GT")
    ax[1].imshow(heatmap_soft,    cmap="gray");  ax[1].set_title("Soft Heatmap (↓ INTER_AREA)")
    ax[2].imshow(heatmap_thresh,  cmap="gray");  ax[2].set_title("Thresholded AREA (↓ & bin)")
    ax[3].imshow(up_thresh,       cmap="gray");  ax[3].set_title("Upsampled (↑ INTER_LINEAR)")

    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.show()

    # print("🚀 Starting main()...")

    # # ---------------------------------------------------------------
    # # 0.  Hyper‑parameters for windowing / STFT
    # # ---------------------------------------------------------------
    # SEG_LEN   = 4.0      # [s] – window length used in the global cache
    # NPERSEG   = 256
    # NOVERLAP  = 224
    # fs        = 200      # [Hz] – sampling rate

    # # ---------------------------------------------------------------
    # # 1.  Load everything (accel_dict + masks + *optionally* segments)
    # # ---------------------------------------------------------------
    # (accel_dict, binary_masks, heatmaps,
    # segments, specs, segIDs) = load_data(
    #         segment_duration = SEG_LEN,
    #         nperseg          = NPERSEG,
    #         noverlap         = NOVERLAP,
    #         recompute        = False)      # True → overwrite cache

    # print("✅ Data loaded.")
    # print(f"   • accel_dict tests .......... {len(accel_dict):>5}")
    # print(f"   • cached segments  .......... {segments.shape}")
    # print(f"   • cached spectrograms ....... {specs.shape}")

    # # ---------------------------------------------------------------
    # # 2.  Pick one random raw trace  (unchanged code below)
    # # ---------------------------------------------------------------
    # tid    = random.choice(list(accel_dict.keys()))
    # ts_raw = random.choice(accel_dict[tid])
    # n_chan = ts_raw.shape[1]
    # print(f"🧪 Using Test {tid} – trace with shape {ts_raw.shape}")

    # # ---------------------------------------------------------------
    # # 3.  Full‑trace RMS to find the loudest instant
    # # ---------------------------------------------------------------
    # rms      = np.sqrt(np.mean(highpass_filter(ts_raw, 2.0, fs, 4)**2,
    #                            axis=1))
    # thresh   = np.percentile(rms, 99.5)
    # peak_idx = int(np.argmax(rms))                            # global max

    # # ---------------------------------------------------------------
    # # 4.  Extract 5‑s window centred on that peak
    # # ---------------------------------------------------------------
    # win      = int(fs * 5.0)                                  # samples
    # half_win = win // 2
    # start    = max(0, peak_idx - half_win)
    # end      = start + win
    # if end > ts_raw.shape[0]:                                 # shift if needed
    #     end   = ts_raw.shape[0]
    #     start = end - win
    # segment  = ts_raw[start:end, :]                           # (win, C)

    # # ---------------------------------------------------------------
    # # 5.  Pre‑processing chain for visual inspection
    # # ---------------------------------------------------------------
    # seg_zero = segment - segment.mean(axis=0, keepdims=True)       # demean
    # seg_hp   = highpass_filter(seg_zero, cutoff=10.0, fs=fs,
    #                            order=6)                            # HP 10 Hz
    # seg_proc = preprocess_segment(segment, fs=fs,
    #                               hp_cut=10.0, rms_norm=True)      # final

    # stages  = [segment, seg_zero, seg_hp, seg_proc]
    # labels  = ["Original",
    #            "Zero‑mean",
    #            "HP > 10 Hz",
    #            "HP + σ‑norm (final)"]
    
    # # ------------------------- NEW  --------------------------------
    # # Compute STFT of the final‑processed window for demo purposes
    # spec_demo = compute_complex_spectrogram(
    #                 seg_proc[None, ...],     # add batch‑dim
    #                 fs       = fs,
    #                 nperseg  = NPERSEG,
    #                 noverlap = NOVERLAP)[0]   # (F,T,2C) → remove batch

    # F, T, _ = spec_demo.shape
    # mag_demo = spec_demo[:, :, 0]            # log‑mag of channel 0

    # # ---------------------------------------------------------------
    # # 6.  FIG 1 – RMS over full trace with chosen window highlighted
    # # ---------------------------------------------------------------
    # t_full = np.arange(len(rms)) / fs
    # fig1, ax1 = plt.subplots(figsize=(12, 3))
    # ax1.plot(t_full, rms, label="RMS")
    # ax1.axhline(thresh, color="crimson", ls="--",
    #             label="99.5‑percentile")
    # ax1.axvspan(start / fs, end / fs, color="gold", alpha=.25,
    #             label="chosen 5 s window")
    # ax1.set(title=f"RMS – Test {tid}", xlabel="Time [s]", ylabel="RMS")
    # ax1.grid(alpha=.3); ax1.legend()
    # plt.show()

    # # ---------------------------------------------------------------
    # # 7.  FIG 2 – Evolution of the 5‑s window through the pipeline
    # # ---------------------------------------------------------------
    # t_seg = np.arange(win) / fs
    # fig2, axes = plt.subplots(len(stages), 1,
    #                           figsize=(12, 2.4 * len(stages)))

    # for ax, data, lbl in zip(axes, stages, labels):
    #     for c in range(n_chan):
    #         ax.plot(t_seg, data[:, c], lw=.8, alpha=.7)
    #     ax.set_title(lbl, loc="left", fontsize=11)
    #     ax.set_ylabel("Amplitude")
    #     ax.set_xlabel("Time [s]")
    #     ax.grid(alpha=.3)

    # fig2.suptitle(f"All {n_chan} channels – 5‑s window centred on max‑RMS "
    #               f"(Test {tid})",
    #               y=1.02, fontsize=14)
    # fig2.tight_layout()
    # plt.show()

    # # ---------------------------------------------------------------
    # # 8.  FIG 3 –  log‑magnitude spectrogram of processed window
    # # ---------------------------------------------------------------
    # plt.figure(figsize=(8, 4))
    # plt.imshow(mag_demo,
    #         origin="lower",
    #         aspect="auto",
    #         cmap="viridis")
    # plt.colorbar(label="log(1 + |STFT|)")
    # plt.title("Channel 0  •  log‑magnitude spectrogram "
    #         f"(win={SEG_LEN:.1f} s, nperseg={NPERSEG})")
    # plt.xlabel("Time bins");  plt.ylabel("Frequency bins")
    # plt.tight_layout()
    # plt.show()



    # test_ids = sorted(accel_dict.keys())
    # first_test_id = test_ids[0]
    # samples_for_first_test = accel_dict[first_test_id]
    # # In your CSV data, each "sample" is a (time_steps×channels) np.array
    # # We'll just pick the first sample
    # raw_sample = samples_for_first_test[0]  # shape ~ (12000, 12) if 60 s at 200 Hz

    # # Plot amplitude vs. time for each channel
    # fs = 200.0  # or your known sampling rate
    # time_axis = np.arange(raw_sample.shape[0]) / fs  # in seconds

    # plt.figure(figsize=(12,6))
    # for ch in range(raw_sample.shape[1]):
    #     plt.plot(time_axis, raw_sample[:, ch], label=f"Ch {ch+1}")

    # plt.title(f"Test ID {first_test_id}: First 60s sample (12 channels)")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # Choose three test IDs you want to visualize (change these IDs to whatever exist in your dataset)
    # test_ids = [8, 18, 25]  # Example IDs - update as needed

    # for tid in test_ids:
    #     if tid not in binary_masks or tid not in heatmaps:
    #         print(f"Test ID {tid} not found in data. Skipping.")
    #         continue
        
    #     mask = binary_masks[tid]         # shape (256,768), values {0,255}
    #     heatmap_3d = heatmaps[tid]      # shape (32,96,1)
    #     heatmap_2d = np.squeeze(heatmap_3d, axis=-1)  # (32,96), float in [0,1]

    #     # For demonstration, also create a blurred version:
    #     heatmap_blur_2d = mask_to_heatmap(mask, target_size=(32,96), apply_blur=True, blur_kernel=(3,3))

    #     # "Reconstructed" (upsampled) heatmap back to 256×768
    #     upsampled = cv2.resize(heatmap_2d, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    #     upsampled_blur = cv2.resize(heatmap_blur_2d, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    #     # Plot: Original, coarse, blurred coarse, and upsampled
    #     fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    #     axes[0].imshow(mask, cmap='gray')
    #     axes[0].set_title(f"Original Binary Mask ")

    #     axes[1].imshow(heatmap_2d, cmap='hot')
    #     axes[1].set_title("Coarse Heatmap (32×96)")

    #     axes[2].imshow(heatmap_blur_2d, cmap='hot')
    #     axes[2].set_title("Blurred Heatmap (32×96)")

    #     axes[3].imshow(upsampled, cmap='hot')
    #     axes[3].set_title("Upsampled (From Unblurred)")

    #     axes[4].imshow(upsampled_blur, cmap='hot')
    #     axes[4].set_title("Upsampled (From Blurred)")

    #     plt.tight_layout()
    #     plt.show()


if __name__ == "__main__":
    main()

