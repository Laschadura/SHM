import numpy as np, cv2
from scipy.signal import butter, filtfilt, stft, get_window
from .config import IMAGE_SHAPE


# utils
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

# Mask generation
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


# accelerometer preprocessing
def preprocess_segment(
        seg: np.ndarray,
        fs: int = 200,
        rms_norm: bool = True,
        peak_norm: bool = False,
        return_stats: bool = False,
        hp_cut: float = 10.0,
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
    # 0) High-pass drift removal
    seg = highpass_filter(seg, cutoff=hp_cut, fs=fs, order=6)

    # 1) remove per-channel mean ---------------------------------
    mean = seg.mean(axis=0, keepdims=True)
    seg  = seg - mean

    # -- 2) scale to comparable energy ------------------------------
    if rms_norm and peak_norm:
        raise ValueError("Choose either rms_norm OR peak_norm – not both.")

    if rms_norm:
        std   = np.maximum(seg.std(axis=0, keepdims=True), 1e-8)
        seg   = seg / std
    elif peak_norm:
        peak  = np.maximum(np.abs(seg).max(axis=0, keepdims=True), 1e-8)
        seg   = seg / peak
    
    if return_stats:
        return seg.astype(np.float32), mean.squeeze(), std.squeeze()

    return seg.astype(np.float32)

def segment_and_transform(
    accel_dict: dict,
    heatmap_dict: dict,
    sample_rate: int   = 200,
    segment_duration: float = 4.0,
    percentile: float = 97.5,
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
    segment_metadata = []
    seg_stats = []

    for tid, traces in accel_dict.items():
        mask = heatmap_dict[tid]

        for trace_idx, ts in enumerate(traces):  # ✅ use enumerate here
            ts_hp  = highpass_filter(ts, cutoff=10.0, fs=sample_rate, order=6)
            rms    = np.sqrt(np.mean(ts_hp ** 2, axis=1))             
            thresh = np.percentile(rms, percentile)
            peaks  = np.where(rms >= thresh)[0]

            peaks = peaks[np.argsort(rms[peaks])[::-1]]
            selected = []

            for p in peaks:
                if all(abs(p - q) > min_sep_samp for q in selected):
                    selected.append(p)
            # ---------------------------------------------------------

            for pk in selected:
                start = pk - half_win
                end   = start + win

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
                seg, μ_seg, σ_seg = preprocess_segment(
                        seg,
                        fs=sample_rate,
                        rms_norm=True,
                        return_stats=True
                       )
                seg_stats.append({"mean": μ_seg, "std": σ_seg})
                segment_metadata.append({
                    "test_id": tid,
                    "trace_index": trace_idx,
                    "peak_position": int(pk)
                })

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
        np.array(all_ids,   dtype=np.int32),
        segment_metadata,
        seg_stats
    )

# Spectogram generation
def compute_or_load_spectrograms(raw_segments, fs=200, nperseg=256, noverlap=192):
    """Wrapper kept for backward-compatibility."""
    print("⏳ Computing STFT for all segments (NumPy)…")
    return compute_complex_spectrogram(raw_segments, fs, nperseg, noverlap)

def compute_complex_spectrogram(
    time_series: np.ndarray,
    fs:           int   = 200,
    nperseg:      int   = 256,
    noverlap:     int   = 224,
    window_name:  str   = "hann",
    ) -> np.ndarray:
    """
    NumPy/SciPy implementation (no Torch):

    Parameters
    ----------
    time_series : (N, T, C) float32
        Raw (or pre-processed) windows.
    fs          : int
        Sample rate [Hz].
    nperseg     : int
        STFT window length.
    noverlap    : int
        Overlap between windows (= nperseg – hop).
    window_name : str
        Window passed to scipy.signal.get_window.

    Returns
    -------
    specs : (N, F, B, 2*C) float32
        Stack of log-magnitude **and** phase [rad].
        `specs[..., 0::2]` → log(1 + |STFT|)
        `specs[..., 1::2]` → angle
    """
    if time_series.ndim != 3:
        raise ValueError("time_series must have shape (N, T, C)")

    N, T, C       = time_series.shape
    win           = get_window(window_name, nperseg, fftbins=True)

    # probe once to know F×B grid
    _, _, Zprobe  = stft(time_series[0, :, 0],
                         fs=fs, window=win,
                         nperseg=nperseg, noverlap=noverlap,
                         boundary=None, padded=False)
    F, B = Zprobe.shape
    out  = np.zeros((N, F, B, 2 * C), dtype=np.float32)

    for i in range(N):
        for c in range(C):
            _, _, Z = stft(time_series[i, :, c],
                           fs=fs, window=win,
                           nperseg=nperseg, noverlap=noverlap,
                           boundary=None, padded=False)


            # -- log-magnitude with small epsilon ------------------
            epsilon = 1e-5
            mag     = np.abs(Z)
            logmag  = np.log(mag + epsilon) # this is better than np.log1p(mag) because it gives us more control when inverting
            out[i, :, :, 2 * c]     = logmag
            out[i, :, :, 2 * c + 1] = np.angle(Z)

    return out
