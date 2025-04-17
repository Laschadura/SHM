import os
import glob
import re
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
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

######################################
# Accelerometer Data Loading
######################################
def highpass_filter(data, cutoff=10.0, fs=200.0, order=4):
    """
    Apply a Butterworth high-pass filter to data.

    Args:
        data: 1D NumPy array of shape (N,) or 2D (N, C). 
              If 2D, we apply filter to each column.
        cutoff: High-pass cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Order of the Butterworth filter.
    
    Returns:
        Filtered data, same shape as input.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    # Get filter coefficients
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    # If data is 1D, just apply it directly
    if data.ndim == 1:
        return filtfilt(b, a, data, axis=0)

    # If data is 2D (N,C), apply filter channel by channel
    filtered = np.zeros_like(data)
    for c in range(data.shape[1]):
        filtered[:, c] = filtfilt(b, a, data[:, c], axis=0)
    return filtered

def load_accelerometer_data(data_dir=DATA_DIR, skip_tests=SKIP_TESTS):
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

                # Truncate to EXPECTED_LENGTH if needed
                if data_matrix.shape[0] > EXPECTED_LENGTH:
                    data_matrix = data_matrix[:EXPECTED_LENGTH, :]

                # 1) High-pass filter to remove low-frequency drift
                data_matrix = highpass_filter(data_matrix, 
                                              cutoff=10.0,
                                              fs=200.0,
                                              order=6)

                # 2) Subtract file-wide mean and perform min–max normalization to [-1, 1]
                channel_means = np.mean(data_matrix, axis=0)  # shape (num_channels,)
                data_matrix = data_matrix - channel_means     # broadcasted subtraction

                data_min = np.min(data_matrix)
                data_max = np.max(data_matrix)
                eps = 1e-8
                if data_max - data_min < eps:
                    data_max = data_min + eps

                # Scale data into [0, 1] first
                data_matrix_scaled = (data_matrix - data_min) / (data_max - data_min)
                # Then transform into [-1, 1]
                data_matrix = 2 * data_matrix_scaled - 1

                samples.append(data_matrix)
            except Exception as e:
                print(f"Skipping file {csv_file} due to error: {e}")
                continue

        if samples:
            tests_data[test_id] = samples

    return tests_data

######################################
# Label Image Processing
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

######################################
# Mask and Heatmap computation
######################################
def compute_binary_mask(combined_image):
    hsv = cv2.cvtColor(combined_image, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    return cv2.bitwise_or(mask1, mask2).astype(np.uint8)

def mask_to_heatmap(orig_mask, target_size=(32, 96), apply_blur=False, blur_kernel=(3,3)):
    """
    Converts a high-res binary mask (H×W) to a coarse heatmap (target_size).
    Each pixel of the heatmap ~ fraction_of_masked_pixels_in_that_region.
    """
    # If mask is in {0,255}, convert to {0,1}
    if orig_mask.max() > 1:
        orig_mask = (orig_mask > 0).astype(np.float32)
    
    # Downsample using INTER_AREA (area averaging)
    newH, newW = target_size
    heatmap = cv2.resize(orig_mask.astype(np.float32),
                         (newW, newH),
                         interpolation=cv2.INTER_AREA)
    
    # Optional smoothing
    if apply_blur:
        heatmap = cv2.GaussianBlur(heatmap, blur_kernel, sigmaX=0)
    
    # Ensure in [0,1]
    heatmap = np.clip(heatmap, 0.0, 1.0)
    return heatmap

########################################
# Segmentation & Spectogram computation
########################################
def segment_and_transform(
    accel_dict, 
    heatmap_dict,
    chunk_size=1, 
    sample_rate=200, 
    segment_duration=4.0, 
    percentile=99,
    test_ids_to_use=None
    ):
    """
    Extracts segments from time series data centered around peak RMS values.
    Returns raw segments and their corresponding binary masks.
    
    Args:
        accel_dict: Dictionary mapping test IDs to time series data.
        heatmap_dict: Dictionary mapping test IDs to binary mask data (downsampled).
        chunk_size: Number of test IDs to process in one batch.
        sample_rate: Sampling rate of the time series data (Hz).
        segment_duration: Duration of each segment (seconds).
        percentile: Percentile threshold for peak detection.
        
    Returns:
        Tuple of arrays: (raw_segments, mask_segments, test_ids)
    """
    window_size = int(sample_rate * segment_duration)
    half_window = window_size // 2
    
    test_ids = list(accel_dict.keys())
    
    # Instead of yield, we'll collect all data and return it at once
    all_raw_segments = []
    all_mask_segments = []
    all_test_ids = []
    
    # Dictionary to count segments per test ID for debugging
    seg_counts = {}

    if test_ids_to_use is not None:
        test_ids_to_use = set(str(tid) for tid in test_ids_to_use)
    
    for i in range(0, len(test_ids), chunk_size):
        chunk_ids = test_ids[i:i + chunk_size]
        
        for test_id in chunk_ids:
            if test_ids_to_use is not None and str(test_id) not in test_ids_to_use:
                continue


            # Debug: print out the shape of the mask for this test
            mask_val = heatmap_dict[test_id]
            try:
                mask_shape = np.array(mask_val).shape
            except Exception:
                mask_shape = "unknown"
            print(f"Processing Test ID {test_id}: mask shape = {mask_shape}")
            
            all_ts = accel_dict[test_id]
            
            for ts_raw in all_ts:
                rms_each_sample = np.sqrt(np.mean(ts_raw**2, axis=1))
                threshold = np.percentile(rms_each_sample, percentile)
                peak_indices = np.where(rms_each_sample >= threshold)[0]
                
                for pk in peak_indices:
                    start = pk - half_window
                    end = pk + half_window
                    if start < 0:
                        start = 0
                        end = window_size
                    if end > ts_raw.shape[0]:
                        end = ts_raw.shape[0]
                        start = end - window_size
                    if (end - start) < window_size:
                        continue
                    
                    segment_raw = ts_raw[start:end, :]
                    
                    all_raw_segments.append(segment_raw)
                    all_mask_segments.append(heatmap_dict[test_id])
                    all_test_ids.append(int(test_id))
                    
                    seg_counts[test_id] = seg_counts.get(test_id, 0) + 1
                    
    # Debug: print summary of segments per test ID
    unique_test_ids = np.unique(all_test_ids)
    print("Segment counts by test ID:", seg_counts)
    print("Unique test IDs from segmentation:", unique_test_ids)
    
    # Convert lists to numpy arrays
    raw_segments = np.array(all_raw_segments, dtype=np.float32)
    mask_segments = np.array(all_mask_segments, dtype=np.float32)
    test_ids = np.array(all_test_ids, dtype=np.int32)
    
    print(f"Extracted {len(raw_segments)} segments, each with a corresponding test ID.")
    print(f"Number of unique test IDs in segmentation: {len(unique_test_ids)}")
    
    return raw_segments, mask_segments, test_ids

def compute_or_load_spectograms(raw_segments, fs=200, nperseg=256, noverlap=192):
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

########################################
# Data Reconstruction (Spectrograms -> Time Series) & Mask upsample
########################################
def inverse_spectrogram(
    complex_spectrograms,
    time_length,
    fs=200,
    nperseg=256,
    noverlap=128,
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
    num_channels = double_channels
    window = torch.hann_window(nperseg)

    time_series = torch.zeros((batch_size, time_length, num_channels), dtype=torch.float32)

    total_batches = (batch_size + batch_processing_size - 1) // batch_processing_size

    for batch_idx in range(total_batches):
        print(f"🔁 Reconstructing batch {batch_idx + 1}/{total_batches}")
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
                stft_input = complex_spec.T  # (time_bins, freq_bins)

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
# Data Loading module
#######################################

def load_data():
    accel_dict = load_accelerometer_data(DATA_DIR, SKIP_TESTS)
    binary_masks = {}
    heatmaps = {}

    test_ids = sorted(accel_dict.keys())
    
    for test_id in test_ids:
        if test_id in SKIP_TESTS:
            continue

        combined_image = load_combined_label(test_id, LABELS_DIR, IMAGE_SHAPE)
        binary_mask = compute_binary_mask(combined_image)

        # Store original binary mask
        binary_masks[test_id] = binary_mask

        # Create a coarse heatmap (32×96) without blur
        heatmap_coarse = mask_to_heatmap(binary_mask, target_size=(32, 96), apply_blur=True, blur_kernel=(3, 3))
        # Expand dims so shape is (32, 96, 1)
        heatmaps[test_id] = np.expand_dims(heatmap_coarse, axis=-1)

    return accel_dict, binary_masks, heatmaps

######################################
# For testing and visualization
######################################
def main():
    # Load data
    accel_dict, binary_masks, heatmaps = load_data()

    test_ids = sorted(accel_dict.keys())
    if not test_ids:
        print("No data loaded, check your data path or skip-tests list.")
        return
    
    first_test_id = test_ids[0]
    samples_for_first_test = accel_dict[first_test_id]

    if not samples_for_first_test:
        print(f"No samples found in Test ID {first_test_id}.")
        return

    # In your CSV data, each "sample" is a (time_steps×channels) np.array
    # We'll just pick the first sample
    raw_sample = samples_for_first_test[0]  # shape ~ (12000, 12) if 60 s at 200 Hz

    # Plot amplitude vs. time for each channel
    fs = 200.0  # or your known sampling rate
    time_axis = np.arange(raw_sample.shape[0]) / fs  # in seconds

    plt.figure(figsize=(12,6))
    for ch in range(raw_sample.shape[1]):
        plt.plot(time_axis, raw_sample[:, ch], label=f"Ch {ch+1}")

    plt.title(f"Test ID {first_test_id}: First 60s sample (12 channels)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Choose three test IDs you want to visualize (change these IDs to whatever exist in your dataset)
    test_ids = [8, 18, 25]  # Example IDs - update as needed

    for tid in test_ids:
        if tid not in binary_masks or tid not in heatmaps:
            print(f"Test ID {tid} not found in data. Skipping.")
            continue
        
        mask = binary_masks[tid]         # shape (256,768), values {0,255}
        heatmap_3d = heatmaps[tid]      # shape (32,96,1)
        heatmap_2d = np.squeeze(heatmap_3d, axis=-1)  # (32,96), float in [0,1]

        # For demonstration, also create a blurred version:
        heatmap_blur_2d = mask_to_heatmap(mask, target_size=(32,96), apply_blur=True, blur_kernel=(3,3))

        # "Reconstructed" (upsampled) heatmap back to 256×768
        upsampled = cv2.resize(heatmap_2d, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        upsampled_blur = cv2.resize(heatmap_blur_2d, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Plot: Original, coarse, blurred coarse, and upsampled
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))

        axes[0].imshow(mask, cmap='gray')
        axes[0].set_title(f"Original Binary Mask ")

        axes[1].imshow(heatmap_2d, cmap='hot')
        axes[1].set_title("Coarse Heatmap (32×96)")

        axes[2].imshow(heatmap_blur_2d, cmap='hot')
        axes[2].set_title("Blurred Heatmap (32×96)")

        axes[3].imshow(upsampled, cmap='hot')
        axes[3].set_title("Upsampled (From Unblurred)")

        axes[4].imshow(upsampled_blur, cmap='hot')
        axes[4].set_title("Upsampled (From Blurred)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()