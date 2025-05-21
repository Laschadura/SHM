import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import signal
import pandas as pd

# Add parent directory to path to import data_loader and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_loader  # Assuming this is available in the parent directory

# ----- Utility Functions from Original Script -----
def compute_complex_spectrogram(
    time_series,
    fs=200,
    nperseg=128,
    noverlap=64
    ):
    """
    Compute STFT-based spectrograms.
    
    Args:
        time_series: shape (batch_size, time_steps, channels)
        fs: Sampling frequency in Hz
        nperseg: Window length for STFT
        noverlap: Overlap between windows
        
    Returns:
        Spectral features (batch, freq_bins, time_bins, channels*2)
        where for each channel we have log-magnitude and phase components
    """
    batch_size, time_steps, channels = time_series.shape

    # Compute frame step
    frame_step = nperseg - noverlap
    print(f"🔍 STFT Config: nperseg={nperseg}, noverlap={noverlap}, frame_step={frame_step}")

    # Test STFT on a single sample
    test_stft = tf.signal.stft(
        time_series[0, :, 0],
        frame_length=nperseg,
        frame_step=frame_step,
        fft_length=nperseg,
        window_fn=tf.signal.hann_window
    ).numpy()

    print(f"📏 Expected STFT shape: (time_bins={test_stft.shape[0]}, freq_bins={test_stft.shape[1]})")

    if test_stft.shape[0] == 0:
        raise ValueError("⚠️ STFT produced 0 time bins! Adjust `nperseg` or `noverlap`.")

    # Pre-allocate spectrograms
    all_spectrograms = np.zeros((batch_size, test_stft.shape[1], test_stft.shape[0], channels*2), dtype=np.float32)

    for i in range(batch_size):
        for c in range(channels):
            stft = tf.signal.stft(
                time_series[i, :, c],
                frame_length=nperseg,
                frame_step=frame_step,
                fft_length=nperseg,
                window_fn=tf.signal.hann_window
            ).numpy()

            if stft.shape[0] == 0:
                raise ValueError(f"⚠️ STFT returned 0 time bins for sample {i}, channel {c}!")

            # Extract magnitude & phase
            mag = np.log1p(np.abs(stft))  # log-magnitude
            phase = np.angle(stft)        # phase in [-pi, pi]

            # Store in output array
            all_spectrograms[i, :, :, 2*c] = mag.T
            all_spectrograms[i, :, :, 2*c+1] = phase.T

    print(f"✅ Final spectrogram shape: {all_spectrograms.shape}")
    return all_spectrograms

def spectrogram_to_features(complex_spectrograms):
    """
    Convert complex spectrograms to feature representation suitable for CNN processing.
    
    Args:
        complex_spectrograms: Complex spectrograms with shape (batch, freq, time, channels)
    
    Returns:
        Features with shape (batch, freq, time, channels*2) where for each original channel
        we have magnitude and phase
    """
    batch_size, freq_bins, time_bins, channels = complex_spectrograms.shape
    
    # Initialize feature array (magnitude and phase for each channel)
    features = np.zeros((batch_size, freq_bins, time_bins, channels * 2), dtype=np.float32)
    
    # Process in smaller batches to save memory
    batch_size_proc = min(1000, batch_size)
    
    for start_idx in range(0, batch_size, batch_size_proc):
        end_idx = min(start_idx + batch_size_proc, batch_size)
        print(f"Processing spectrograms {start_idx}-{end_idx-1}/{batch_size}")
        
        # Extract a batch of spectrograms
        batch_specs = complex_spectrograms[start_idx:end_idx]
        
        # Extract magnitude and phase
        for c in range(channels):
            # Magnitude (log scale for better dynamic range)
            magnitude = np.abs(batch_specs[:, :, :, c])
            # Add small constant to avoid log(0)
            log_magnitude = np.log1p(magnitude)
            
            # Phase
            phase = np.angle(batch_specs[:, :, :, c])
            
            # Store in feature array
            features[start_idx:end_idx, :, :, c*2] = log_magnitude
            features[start_idx:end_idx, :, :, c*2+1] = phase
        
        # Free memory
        del batch_specs
        import gc
        gc.collect()
    
    return features

def inverse_spectrogram(complex_spectrograms, time_length, fs=200, nperseg=128, noverlap=64):
    """
    Convert complex spectrograms back to time series.
    
    Args:
        complex_spectrograms: Complex spectrograms (batch, freq, time, channels*2)
        time_length: Original time series length
        fs: Sampling frequency
        nperseg: Length of each segment
        noverlap: Number of points to overlap between segments
    
    Returns:
        Reconstructed time series (batch, time_length, channels)
    """
    batch_size, freq_bins, time_bins, total_channels = complex_spectrograms.shape
    num_orig_channels = total_channels // 2  # Since we have magnitude and phase for each channel
    frame_step = nperseg - noverlap

    # Pre-allocate the result array
    time_series = np.zeros((batch_size, time_length, num_orig_channels), dtype=np.float32)

    inv_window_fn = tf.signal.inverse_stft_window_fn(
        frame_step,
        forward_window_fn=lambda length, dtype: tf.signal.hann_window(length, dtype=dtype)
    )
    
    
    # Process each sample and channel
    for b in range(batch_size):
        for c in range(num_orig_channels):
            # Extract magnitude and phase
            log_magnitude = complex_spectrograms[b, :, :, c*2]
            phase = complex_spectrograms[b, :, :, c*2+1]
            
            # Convert back to linear magnitude
            magnitude = np.exp(log_magnitude) - 1.0
            
            # Combine magnitude and phase to get complex spectrogram
            complex_spec = magnitude * np.exp(1j * phase)
            
            # Transpose to match TensorFlow's expected format
            complex_spec_tf = tf.constant(complex_spec.T, dtype=tf.complex64)
            
            # Use TensorFlow's inverse STFT
            inverse_stft = tf.signal.inverse_stft(
                complex_spec_tf,
                frame_length=nperseg,
                frame_step=frame_step,
                fft_length=nperseg,
                window_fn=inv_window_fn
            )
            
            # Trim to the original length
            actual_length = min(inverse_stft.shape[0], time_length)
            time_series[b, :actual_length, c] = inverse_stft[:actual_length].numpy()
    
    return time_series

def segment_and_transform(
    accel_dict, 
    chunk_size=1, 
    sample_rate=200, 
    segment_duration=3.0, 
    percentile=99,
    max_segments=10
    ):
    """
    Extracts segments from time series data centered around peak RMS values.
    Simplified version for testing that returns only raw segments.
    
    Args:
        accel_dict: Dictionary mapping test IDs to time series data.
        chunk_size: Number of test IDs to process in one batch.
        sample_rate: Sampling rate of the time series data (Hz).
        segment_duration: Duration of each segment (seconds).
        percentile: Percentile threshold for peak detection.
        max_segments: Maximum number of segments to extract (for testing).
        
    Returns:
        Array of raw segments
    """
    window_size = int(sample_rate * segment_duration)
    half_window = window_size // 2
    
    test_ids = list(accel_dict.keys())
    
    all_raw_segments = []
    all_test_ids = []
    
    for i in range(0, len(test_ids), chunk_size):
        if len(all_raw_segments) >= max_segments:
            break
            
        chunk_ids = test_ids[i:i + chunk_size]
        
        for test_id in chunk_ids:
            all_ts = accel_dict[test_id]
            
            for ts_raw in all_ts:
                if len(all_raw_segments) >= max_segments:
                    break
                    
                rms_each_sample = np.sqrt(np.mean(ts_raw**2, axis=1))
                threshold = np.percentile(rms_each_sample, percentile)
                peak_indices = np.where(rms_each_sample >= threshold)[0]
                
                for pk in peak_indices:
                    if len(all_raw_segments) >= max_segments:
                        break
                        
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
                    all_test_ids.append(int(test_id))
    
    raw_segments = np.array(all_raw_segments, dtype=np.float32)
    test_ids = np.array(all_test_ids, dtype=np.int32)
    
    print(f"Extracted {len(raw_segments)} segments for testing.")
    
    return raw_segments, test_ids
 
def calculate_reconstruction_metrics(original, reconstructed):
    """
    Calculate various reconstruction metrics.
    
    Args:
        original: Original time series array (batch, time, channels)
        reconstructed: Reconstructed time series array (batch, time, channels)
        
    Returns:
        Dictionary of metrics
    """
    batch_size, time_length, channels = original.shape
    
    # Calculate metrics
    mse_per_segment = np.mean((original - reconstructed)**2, axis=(1, 2))
    rmse_per_segment = np.sqrt(mse_per_segment)
    
    # Calculate normalized cross-correlation
    ncc_values = []
    for i in range(batch_size):
        segment_ncc = []
        for c in range(channels):
            orig = original[i, :, c]
            recon = reconstructed[i, :, c]
            
            # Normalize signals
            orig_norm = (orig - np.mean(orig)) / (np.std(orig) + 1e-8)
            recon_norm = (recon - np.mean(recon)) / (np.std(recon) + 1e-8)
            
            # Calculate cross-correlation
            corr = np.correlate(orig_norm, recon_norm, mode='valid')[0] / time_length
            segment_ncc.append(corr)
        
        ncc_values.append(np.mean(segment_ncc))
    
    # Aggregate metrics
    metrics = {
        'rmse_mean': np.mean(rmse_per_segment),
        'rmse_std': np.std(rmse_per_segment),
        'ncc_mean': np.mean(ncc_values),
        'ncc_std': np.std(ncc_values),
        'rmse_per_segment': rmse_per_segment,
        'ncc_per_segment': ncc_values
    }
    
    return metrics

def plot_overlay_all_channels(original, reconstructed, segment_idx=0, fs=200, normalized=True):
    """
    Plot a single figure for the specified segment, overlaying all channel signals
    from 'original' and 'reconstructed' in one axes.

    Args:
        original (ndarray):       Shape (batch, time, channels)
        reconstructed (ndarray):  Shape (batch, time, channels)
        segment_idx (int):        Which segment (row in 'original' & 'reconstructed') to plot
        fs (int):                 Sampling frequency (used for time axis)
        normalized (bool):        Whether the data is normalized
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Dimensions
    num_channels = original.shape[2]
    time_length = original.shape[1]
    time_axis = np.linspace(0, time_length / fs, time_length)

    # Calculate overall RMSE across all channels for this segment
    rmse = np.sqrt(np.mean((original[segment_idx] - reconstructed[segment_idx])**2))

    # Create a single plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all channels from original (blue) and reconstructed (red)
    for ch in range(num_channels):
        ax.plot(
            time_axis,
            original[segment_idx, :, ch],
            color='blue',
            alpha=0.5
        )
        ax.plot(
            time_axis,
            reconstructed[segment_idx, :, ch],
            color='red',
            alpha=0.5
        )

    # Create two "dummy" lines for the legend (so it doesn't show a line for each channel)
    ax.plot([], [], color='blue', label='Original')
    ax.plot([], [], color='red', label='Reconstructed')
    ax.legend()

    normalized_text = "(Normalized)" if normalized else ""
    ax.set_title(f'Segment {segment_idx} {normalized_text} | Overlaid Channels (RMSE: {rmse:.4f})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'all_channels_segment_{segment_idx}.png', dpi=300)
    plt.show()

def test_chatgpt():
    """
    Test function to check if the script is running correctly.
    This is a placeholder and can be removed or modified as needed.
    """
    print("ChatGPT is smart!")


def main():
    """Main function to run the reconstruction test with normalization"""
    # Parameters
    fs = 200                    # Sampling frequency in Hz
    nperseg = 256               # STFT segment length
    noverlap = 224              # STFT overlap
    max_segments = 1            # Number of segments to test
    
    print("📥 Loading raw data...")
    try:
        # Load data from data_loader
        accel_dict, binary_masks, heatmaps = data_loader.load_data()
        print(f"Loaded data for {len(accel_dict)} tests")
    except Exception as e:
        print(f"Error loading data: {e}")
        # If data_loader doesn't work, create a synthetic dataset for testing
        print("Creating synthetic data for testing...")
        accel_dict = {}
        # Create a synthetic time series (3 seconds at 200Hz with 12 channels)
        t = np.linspace(0, 4, 800)
        for i in range(5):
            # Create signals with different frequencies
            signals = []
            for _ in range(3):  # Create 3 separate time series per test
                channels = []
                for j in range(12):  # 12 channels
                    freq = 5 + j*2  # Different frequency for each channel
                    channel = np.sin(2 * np.pi * freq * t + np.random.rand())
                    channel += 0.2 * np.random.randn(len(t))  # Add noise
                    channels.append(channel)
                signals.append(np.column_stack(channels))
            accel_dict[str(i+1)] = signals
    
    # Extract segments from normalized data
    print("✂️ Segmenting normalized data...")
    segments, test_ids = segment_and_transform(
        accel_dict,
        max_segments=max_segments,
        segment_duration=5.0,
    )
    print(f"✅ Extracted {len(segments)} segments with shape {segments.shape}")

    
    # Calculate spectrograms on normalized segments
    print("🔄 Computing spectrograms from normalized segments...")
    complex_specs = compute_complex_spectrogram(segments, fs, nperseg, noverlap)
    
    # The complex_specs output from compute_complex_spectrogram is already in feature form
    print("📝 Using computed spectral features...")
    spec_features = complex_specs  # These are already in the right format
    print(f"Feature shape: {spec_features.shape}")
    
    # Reconstruct time series
    print("🔄 Reconstructing time series from spectrograms...")
    reconstructed_ts = inverse_spectrogram(spec_features, segments.shape[1], fs, nperseg, noverlap)
    print(f"Reconstructed shape: {reconstructed_ts.shape}")
    
    # Debug: Check reconstruction ranges
    print("\n🔍 Checking reconstruction ranges:")
    for i in range(min(3, len(reconstructed_ts))):
        for c in range(min(3, reconstructed_ts.shape[2])):
            max_val = np.max(reconstructed_ts[i, :, c])
            min_val = np.min(reconstructed_ts[i, :, c])
            print(f"Reconstructed Segment {i}, Channel {c}: Min={min_val:.4f}, Max={max_val:.4f}, Range={max_val-min_val:.4f}")
    
    # Calculate metrics against normalized segments
    print("📊 Calculating reconstruction metrics...")
    metrics = calculate_reconstruction_metrics(segments, reconstructed_ts)
    print(f"Average RMSE: {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}")
    print(f"Average NCC: {metrics['ncc_mean']:.4f} ± {metrics['ncc_std']:.4f}")
    
    # Create results directory
    os.makedirs("reconstruction_results", exist_ok=True)
    os.chdir("reconstruction_results")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Segment': range(len(metrics['rmse_per_segment'])),
        'Test ID': test_ids,
        'RMSE': metrics['rmse_per_segment'],
        'NCC': metrics['ncc_per_segment']
    })
    metrics_df.to_csv('reconstruction_metrics.csv', index=False)
    print("✅ Saved metrics to reconstruction_metrics.csv")
    
    # Plot comparisons
    print("📊 Generating plots...")
    for i in range(len(segments)):
        plot_overlay_all_channels(
            original=segments, 
            reconstructed=reconstructed_ts, 
            segment_idx=i,
            fs=fs,
            normalized=True
        )

    print("✅ Done! Results saved in the 'reconstruction_results' directory.")

if __name__ == "__main__":
    main()