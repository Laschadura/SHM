import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.cross_decomposition import CCA
from skimage.metrics import structural_similarity as ssim
from scipy.signal import find_peaks

# Import your existing data loader directly
import data_loader
from data_loader import *


class FeatureExtractor:
    """Extracts different features from accelerometer signals."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        pass
    
    def compute_rms(self, accel_signal, window_size=256, overlap=0.5):
        """
        Compute Root Mean Square (RMS) values for a signal.
        
        Args:
            accel_signal: Input accelerometer signal, shape [time_steps, channels]
            window_size: Size of the sliding window
            overlap: Overlap between consecutive windows (0-1)
            
        Returns:
            numpy.ndarray: RMS values for each window and channel
        """
        # Calculate step size based on window_size and overlap
        step = int(window_size * (1 - overlap))
        
        # Determine number of windows
        n_windows = (accel_signal.shape[0] - window_size) // step + 1
        n_channels = accel_signal.shape[1]
        
        # Initialize RMS array
        rms_values = np.zeros((n_windows, n_channels))
        
        # Calculate RMS for each window and channel
        for i in range(n_windows):
            start = i * step
            end = start + window_size
            window = accel_signal[start:end, :]
            rms_values[i, :] = np.sqrt(np.mean(window**2, axis=0))
        
        return rms_values
    
    def compute_psd(self, accel_signal, fs=1000, nperseg=256, noverlap=None, scaling='density'):
        """
        Compute Power Spectral Density (PSD) using Welch's method.
        
        Args:
            accel_signal: Input accelerometer signal, shape [time_steps, channels]
            fs: Sampling frequency in Hz
            nperseg: Length of each segment
            noverlap: Number of points to overlap between segments
            scaling: Scaling mode ('density' or 'spectrum')
            
        Returns:
            tuple: (frequencies, PSD values for each channel)
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        n_channels = accel_signal.shape[1]
        
        # Compute PSD for each channel
        freq_list = []
        psd_list = []
        
        for ch in range(n_channels):
            freqs, psd = signal.welch(accel_signal[:, ch], fs=fs, nperseg=nperseg, 
                                     noverlap=noverlap, scaling=scaling)
            freq_list.append(freqs)
            psd_list.append(psd)
        
        # All channels should have the same frequency bins
        return freq_list[0], np.column_stack(psd_list)
    
    def compute_spectrogram(self, accel_signal, fs=1000, nperseg=256, noverlap=None, mode='complex'):
        """
        Compute spectrogram using Short-Time Fourier Transform (STFT).
        
        This is an enhanced implementation based on SciPy's STFT that allows for:
        1. Magnitude spectrograms (default, power preserved)
        2. Complex spectrograms (reversible with istft)
        3. Power spectrograms (traditional spectrogram)
        
        Args:
            accel_signal: Input accelerometer signal, shape [time_steps, channels]
            fs: Sampling frequency in Hz
            nperseg: Length of each segment
            noverlap: Number of points to overlap between segments
            mode: 'magnitude', 'complex', or 'power'
                - 'magnitude': sqrt(abs(STFT)²) - preserves power and suitable for visualization
                - 'complex': raw complex STFT output - fully reversible with istft
                - 'power': abs(STFT)² - traditional spectrogram, not directly reversible
            
        Returns:
            tuple: (frequencies, times, spectrogram values for each channel)
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        n_channels = accel_signal.shape[1]
        
        # Compute spectrogram for each channel
        f_list = []
        t_list = []
        spec_list = []
        
        for ch in range(n_channels):
            # Calculate STFT using SciPy
            f, t, Zxx = signal.stft(accel_signal[:, ch], fs=fs, nperseg=nperseg, 
                                   noverlap=noverlap, return_onesided=True)
            
            # Process according to requested mode
            if mode == 'complex':
                # Return complex STFT (fully reversible)
                processed_spec = Zxx
            elif mode == 'power':
                # Traditional power spectrogram (abs(STFT)²)
                processed_spec = np.abs(Zxx)**2
            else:  # Default is 'magnitude'
                # Magnitude spectrogram (sqrt(abs(STFT)²))
                processed_spec = np.abs(Zxx)
            
            f_list.append(f)
            t_list.append(t)
            spec_list.append(processed_spec)
        
        # All channels should have the same frequency and time bins
        return f_list[0], t_list[0], np.array(spec_list)
    
    def _segment_by_rms_peaks(self, accel_signal, fs=1000, segment_size=5, percentile_threshold=99):
        """
        Segment the accelerometer signal based on RMS peaks.
        
        Args:
            accel_signal: Input accelerometer signal, shape [time_steps, channels]
            fs: Sampling frequency in Hz
            segment_size: Size of segments in seconds
            percentile_threshold: Percentile for peak detection (e.g., 99 for 99th percentile)
            
        Returns:
            list: List of segments, each a numpy array of shape [time_steps, channels]
            dict: Information about the segmentation process
        """
        # Calculate RMS for each time point across all channels
        rms_values = np.sqrt(np.mean(accel_signal**2, axis=1))
        
        # Find threshold based on percentile
        threshold = np.percentile(rms_values, percentile_threshold)
        
        # Find peaks above threshold
        peaks, _ = find_peaks(rms_values, height=threshold, distance=fs)  # Minimum distance of 1 second
        
        segments = []
        segment_info = {
            'peak_indices': peaks,
            'peak_values': rms_values[peaks] if len(peaks) > 0 else [],
            'threshold': threshold,
            'num_segments': 0
        }
        
        if len(peaks) == 0:
            # If no peaks are found, use the entire signal as a single segment
            segments.append(accel_signal)
            segment_info['num_segments'] = 1
            segment_info['segment_type'] = 'full_signal'
            return segments, segment_info
        
        # Calculate segment boundaries based on peaks
        segment_samples = int(segment_size * fs)
        half_segment = segment_samples // 2
        
        for peak_idx in peaks:
            start = max(0, peak_idx - half_segment)
            end = min(len(accel_signal), peak_idx + half_segment)
            
            # Extract segment
            segment = accel_signal[start:end, :]
            
            # Only add segments of sufficient length
            if segment.shape[0] >= fs:  # At least 1 second of data
                segments.append(segment)
        
        segment_info['num_segments'] = len(segments)
        segment_info['segment_type'] = 'rms_peak_based'
        
        return segments, segment_info
    
    def extract_features(self, accel_data, fs=1000, window_size=256, overlap=0.5, segment_by_peaks=True):
        """
        Extract multiple feature representations from accelerometer data.
        
        Args:
            accel_data: List of accelerometer signals
            fs: Sampling frequency in Hz
            window_size: Size of the window for feature computation
            overlap: Overlap between consecutive windows
            segment_by_peaks: Whether to segment signals by RMS peaks
            
        Returns:
            dict: Dictionary containing raw, RMS, PSD, and spectrogram features
        """
        all_features = {
            'raw': [],
            'rms': [],
            'psd': [],
            'spectrogram': [],
            'complex_spectrogram': [],
            'psd_freqs': None,
            'spec_freqs': None,
            'spec_times': None,
            'segments_info': []
        }
        
        for signal_idx, accel_signal in enumerate(accel_data):
            print(f"Extracting features from signal {signal_idx+1}/{len(accel_data)}")
            
            # Process the entire signal first
            # Store raw signal
            all_features['raw'].append(accel_signal)
            
            # Compute RMS
            rms = self.compute_rms(accel_signal, window_size, overlap)
            all_features['rms'].append(rms)
            
            # Compute PSD
            freqs, psd = self.compute_psd(accel_signal, fs, nperseg=window_size)
            all_features['psd'].append(psd)
            
            # Set frequency bins (same for all signals)
            if all_features['psd_freqs'] is None:
                all_features['psd_freqs'] = freqs
            
            # Compute magnitude spectrogram for visualization and analysis
            f, t, mag_spec = self.compute_spectrogram(accel_signal, fs, nperseg=window_size, mode='magnitude')
            all_features['spectrogram'].append(mag_spec)
            
            # Also compute complex spectrogram for potential reconstruction
            _, _, complex_spec = self.compute_spectrogram(accel_signal, fs, nperseg=window_size, mode='complex')
            all_features['complex_spectrogram'].append(complex_spec)
            
            # Set frequency and time bins (same for all signals)
            if all_features['spec_freqs'] is None:
                all_features['spec_freqs'] = f
            
            if all_features['spec_times'] is None:
                all_features['spec_times'] = t
            
            # If segmentation is enabled, extract features from peak-based segments
            if segment_by_peaks:
                segments, segment_info = self._segment_by_rms_peaks(accel_signal, fs=fs)
                
                # Store segmentation info
                segment_info['signal_idx'] = signal_idx
                all_features['segments_info'].append(segment_info)
                
                # Process each segment
                segment_features = {
                    'raw': [],
                    'rms': [],
                    'psd': [],
                    'spectrogram': []
                }
                
                for i, segment in enumerate(segments):
                    # Process each segment similarly to the whole signal
                    segment_features['raw'].append(segment)
                    
                    # RMS
                    segment_rms = self.compute_rms(segment, window_size, overlap)
                    segment_features['rms'].append(segment_rms)
                    
                    # PSD
                    _, segment_psd = self.compute_psd(segment, fs, nperseg=window_size)
                    segment_features['psd'].append(segment_psd)
                    
                    # Spectrogram
                    _, _, segment_spec = self.compute_spectrogram(segment, fs, nperseg=window_size, mode='magnitude')
                    segment_features['spectrogram'].append(segment_spec)
                
                # Append segment features to main feature dictionary
                segment_info['features'] = segment_features
        
        return all_features

class CorrelationAnalyzer:
    """Analyzes correlations between time-series features and damage indicators."""
    
    def __init__(self, output_dir="correlation_results"):
        """
        Initialize the correlation analyzer.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Feature containers
        self.features = None
        self.masks = None
        self.descriptors = None
        
        # Results containers
        self.mask_correlations = {}
        self.descriptor_correlations = {}
        self.feature_importance = {}
        
        # Feature types
        self.feature_types = ['raw', 'rms', 'psd', 'spectrogram']
    
    def visualize_results(self, mask_correlations, descriptor_correlations, feature_rankings, segments_info=None):
        print("Creating visualizations...")
        os.makedirs(self.output_dir, exist_ok=True)

        # Existing calls
        self._plot_feature_rankings(feature_rankings)
        self._plot_mask_correlations(mask_correlations, feature_rankings['overall'])
        self._plot_descriptor_correlations(descriptor_correlations, feature_rankings['overall'])
        # Possibly your existing CCA, segmentation, etc.

        # NEW calls:
        self._plot_mask_correlation_distributions(mask_correlations, feature_rankings['overall'])
        self._plot_combined_mask_correlations(mask_correlations, feature_rankings['overall'])
        self._plot_descriptor_correlation_distributions(descriptor_correlations, feature_rankings['overall'])
        self._plot_feature_correlation_scatter(feature_rankings)
        
        # For a property-level bar chart on Pearson, for instance:
        if 'pearson' in descriptor_correlations:
            self._plot_descriptor_property_bar_chart(descriptor_correlations, 
                                                    feature_rankings['overall'], 
                                                    method='pearson')

        if segments_info:
            self._plot_segmentation_info(segments_info)

        print(f"Visualizations saved to {self.output_dir}")


    def _plot_feature_rankings(self, rankings):
        """
        Plot feature rankings.
        
        Args:
            rankings: Dictionary of feature rankings
        """
        scores = rankings['scores']
        feature_types = rankings['overall']
        
        # Bar plot of aggregate scores
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(feature_types))
        width = 0.25
        
        mask_scores = [scores[ft]['mask'] for ft in feature_types]
        desc_scores = [scores[ft]['descriptor'] for ft in feature_types]
        total_scores = [scores[ft]['total'] for ft in feature_types]
        
        plt.bar(x - width, mask_scores, width, label='Mask Correlation')
        plt.bar(x, desc_scores, width, label='Descriptor Correlation')
        plt.bar(x + width, total_scores, width, label='Overall Score')
        
        plt.xlabel('Feature Type')
        plt.ylabel('Normalized Score')
        plt.title('Feature Importance Scores by Correlation Type')
        plt.xticks(x, [ft.upper() for ft in feature_types])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance_scores.png'))
        plt.close()
        
        # Create horizontal bar chart for rankings
        plt.figure(figsize=(10, 6))
        
        # For each ranking type, create horizontal bars
        y_pos = np.arange(len(feature_types))
        
        # Overall ranking (most important)
        sorted_by_overall = sorted(feature_types, key=lambda x: scores[x]['total'], reverse=True)
        overall_scores = [scores[ft]['total'] for ft in sorted_by_overall]
        
        plt.barh(y_pos, overall_scores, align='center', color='skyblue', alpha=0.8)
        plt.yticks(y_pos, [ft.upper() for ft in sorted_by_overall])
        plt.xlabel('Overall Score')
        plt.title('Feature Types Ranked by Overall Correlation')
        
        # Add score values as text
        for i, score in enumerate(overall_scores):
            plt.text(score + 0.01, i, f'{score:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_rankings.png'))
        plt.close()
        
        # Create ranking table as CSV
        rank_data = {'Rank': list(range(1, len(feature_types) + 1))}
        
        # Add rankings by each criteria
        for criteria in ['By Mask Correlation', 'By Descriptor Correlation', 'Overall']:
            key = criteria.lower().replace('by ', '').replace(' ', '_')
            if key == 'overall':
                rank_list = rankings['overall']
            elif key == 'mask_correlation':
                rank_list = rankings['mask_correlation']
            elif key == 'descriptor_correlation':
                rank_list = rankings['descriptor_correlation']
            else:
                continue
            
            rank_data[criteria] = rank_list
        
        df = pd.DataFrame(rank_data)
        df.to_csv(os.path.join(self.output_dir, 'feature_rankings.csv'), index=False)
        
        # Create a detailed scores table
        score_data = {'Feature Type': feature_types}
        for score_type in ['Mask', 'Descriptor', 'Total']:
            key = score_type.lower()
            scores_list = [scores[ft][key] for ft in feature_types]
            score_data[f'{score_type} Score'] = scores_list
        
        df_scores = pd.DataFrame(score_data)
        df_scores.to_csv(os.path.join(self.output_dir, 'feature_scores.csv'), index=False)

    def _plot_mask_correlations(self, mask_correlations, feature_types):
        """
        Plot mask correlation results.
        
        Args:
            mask_correlations: Dictionary of mask correlation results
            feature_types: List of feature types to include
        """
        # Heatmap of correlation scores
        plt.figure(figsize=(10, 6))
        
        # Prepare data for heatmap
        methods = list(mask_correlations.keys())
        
        # Filter feature types to those that have data
        available_types = set()
        for method in methods:
            available_types.update(mask_correlations[method].keys())
        
        # Intersection of available types and requested types
        plot_types = [ft for ft in feature_types if ft in available_types]
        
        if not plot_types:
            print("Warning: No feature types available for mask correlation plot")
            return
        
        data = np.zeros((len(methods), len(plot_types)))
        
        for i, method in enumerate(methods):
            for j, feature_type in enumerate(plot_types):
                if feature_type in mask_correlations[method]:
                    values = mask_correlations[method][feature_type]
                    data[i, j] = np.mean(values) if values else 0
        
        # Create heatmap
        sns.heatmap(data, annot=True, fmt='.3f', 
                xticklabels=[ft.upper() for ft in plot_types], 
                yticklabels=methods,
                cmap='YlGnBu')
        
        plt.title('Feature-to-Mask Correlation Scores')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mask_correlations.png'))
        plt.close()
        
        # Individual bar plots for each method
        for method in methods:
            plt.figure(figsize=(10, 5))
            
            feature_values = []
            for ft in plot_types:
                if ft in mask_correlations[method]:
                    values = mask_correlations[method][ft]
                    feature_values.append(np.mean(values) if values else 0)
                else:
                    feature_values.append(0)
            
            # Create bar plot
            bars = plt.bar(range(len(plot_types)), feature_values, color='skyblue')
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.xticks(range(len(plot_types)), [ft.upper() for ft in plot_types])
            plt.ylabel(f'{method.capitalize()} Correlation')
            plt.title(f'{method.capitalize()} Correlation to Damage Masks')
            plt.ylim(0, max(feature_values) * 1.2)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'mask_{method}_correlations.png'))
            plt.close()

    def _plot_mask_correlation_distributions(self, mask_correlations, feature_types):
        """
        Create distribution (box or violin) plots of per-test mask correlation
        for each method and feature type.
        """
        output_dir = self.output_dir  # ensure we can save figures in the class's output directory
        
        # Go through each correlation method (e.g., 'ssim', 'pearson', etc.)
        for method, ft_dict in mask_correlations.items():
            # Build a long-form DataFrame for Seaborn
            rows = []
            for ft in feature_types:
                # Skip if no data
                if ft not in ft_dict:
                    continue
                corr_values = ft_dict[ft]  # list of correlation values, 1 per test
                for val in corr_values:
                    rows.append({
                        'Method': method.capitalize(),
                        'FeatureType': ft.upper(),
                        'Correlation': val
                    })
            
            if not rows:
                continue

            df = pd.DataFrame(rows)

            # Make a boxplot and swarmplot (or violinplot)
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='FeatureType', y='Correlation', data=df, palette='Set3')
            sns.swarmplot(x='FeatureType', y='Correlation', data=df, color='0.25', alpha=0.7)
            
            plt.title(f"Distribution of {method.capitalize()} Correlations to Mask by Feature Type")
            plt.ylabel("Correlation")
            # If you expect negative correlations or correlation up to 1.0:
            plt.ylim(-1, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{method}_mask_correlation_distribution.png"))
            plt.close()

    def _plot_combined_mask_correlations(self, mask_correlations, feature_types):
        """
        Create a grouped bar chart showing the average correlation for each 
        method (e.g. SSIM, Pearson, etc.) for each feature type.
        """
        output_dir = self.output_dir
        
        methods = list(mask_correlations.keys())  # e.g. ['ssim', 'pearson']
        
        # Calculate average correlation per feature type, per method
        avg_data = {}
        for method in methods:
            avg_data[method] = []
            for ft in feature_types:
                if ft in mask_correlations[method]:
                    vals = mask_correlations[method][ft]
                    avg_data[method].append(np.mean(vals) if vals else 0.0)
                else:
                    avg_data[method].append(0.0)
        
        # Now create a grouped bar chart
        x = np.arange(len(feature_types))
        width = 0.8 / len(methods)  # so all method bars fit side-by-side
        
        plt.figure(figsize=(12, 6))
        
        for i, method in enumerate(methods):
            offset = (i - (len(methods) - 1)/2) * width
            plt.bar(
                x + offset, 
                avg_data[method], 
                width, 
                label=method.capitalize()
            )
        
        plt.xticks(x, [ft.upper() for ft in feature_types])
        plt.ylabel("Average Correlation")
        plt.title("Mask Correlation by Method and Feature Type")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mask_correlation_grouped_bar.png"))
        plt.close()

    def _plot_descriptor_correlations(self, descriptor_correlations, feature_types):
        """
        Plot descriptor correlation results.
        
        Args:
            descriptor_correlations: Dictionary of descriptor correlation results
            feature_types: List of feature types to include
        """
        # For Pearson and Spearman correlations
        for method in ['pearson', 'spearman']:
            if method not in descriptor_correlations:
                continue
            
            # Create a summary plot for the method
            plt.figure(figsize=(12, 8))
            
            # For each feature type, get average correlation magnitude across all properties
            feature_avgs = []
            feature_labels = []
            
            for feature_type in feature_types:
                if feature_type in descriptor_correlations[method]:
                    scores = descriptor_correlations[method][feature_type]
                    
                    if not scores:
                        continue
                    
                    # Calculate average absolute correlation across all properties and features
                    avg_corr = np.mean([np.mean(np.abs(scores[prop])) for prop in scores])
                    feature_avgs.append(avg_corr)
                    feature_labels.append(feature_type.upper())
            
            # Create bar plot
            if feature_avgs:
                bars = plt.bar(range(len(feature_avgs)), feature_avgs, color='skyblue')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom')
                
                plt.xticks(range(len(feature_labels)), feature_labels)
                plt.ylabel(f'Average |{method.capitalize()} Correlation|')
                plt.title(f'Average {method.capitalize()} Correlation to Crack Descriptors')
                plt.ylim(0, max(feature_avgs) * 1.2)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{method}_average_correlations.png'))
                plt.close()
            
            # Detailed heatmaps for each feature type
            for feature_type in feature_types:
                if feature_type not in descriptor_correlations[method]:
                    continue
                    
                scores = descriptor_correlations[method][feature_type]
                
                if not scores:
                    continue
                
                # Create dataframe for heatmap
                prop_names = list(scores.keys())
                if not prop_names:
                    continue
                    
                n_features = len(scores[prop_names[0]])
                
                data = np.zeros((len(prop_names), n_features))
                
                for i, prop in enumerate(prop_names):
                    data[i, :] = np.abs(scores[prop])
                
                # Plot heatmap
                plt.figure(figsize=(12, max(6, len(prop_names) * 0.5)))
                sns.heatmap(data, annot=False, cmap='YlGnBu',
                        xticklabels=range(1, n_features+1),
                        yticklabels=prop_names)
                
                plt.title(f'{method.capitalize()} Correlation: {feature_type.upper()} vs Descriptors')
                plt.xlabel('Feature Dimension')
                plt.ylabel('Descriptor Property')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{method}_{feature_type}_correlations.png'))
                plt.close()
        
        # Random Forest importance
        if 'rf_importance' in descriptor_correlations:
            # Summary plot for RF importance
            plt.figure(figsize=(12, 8))
            
            # Calculate average importance for each feature type
            rf_avgs = []
            rf_labels = []
            
            for feature_type in feature_types:
                if feature_type in descriptor_correlations['rf_importance']:
                    scores = descriptor_correlations['rf_importance'][feature_type]
                    
                    if not scores:
                        continue
                    
                    # Average across all properties and features
                    avg_importance = np.mean([np.mean(scores[prop]) for prop in scores])
                    rf_avgs.append(avg_importance)
                    rf_labels.append(feature_type.upper())
            
            # Create summary bar plot
            if rf_avgs:
                bars = plt.bar(range(len(rf_avgs)), rf_avgs, color='lightgreen')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                            f'{height:.3f}', ha='center', va='bottom')
                
                plt.xticks(range(len(rf_labels)), rf_labels)
                plt.ylabel('Average Feature Importance')
                plt.title('Random Forest Feature Importance by Feature Type')
                plt.ylim(0, max(rf_avgs) * 1.2)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'rf_average_importance.png'))
                plt.close()
            
            # Detailed heatmaps for each feature type
            for feature_type in feature_types:
                if feature_type not in descriptor_correlations['rf_importance']:
                    continue
                    
                scores = descriptor_correlations['rf_importance'][feature_type]
                
                if not scores:
                    continue
                
                # Create dataframe for heatmap
                prop_names = list(scores.keys())
                if not prop_names:
                    continue
                    
                n_features = len(scores[prop_names[0]])
                
                data = np.zeros((len(prop_names), n_features))
                
                for i, prop in enumerate(prop_names):
                    data[i, :] = scores[prop]
                
                # Plot heatmap
                plt.figure(figsize=(12, max(6, len(prop_names) * 0.5)))
                sns.heatmap(data, annot=False, cmap='YlGnBu',
                        xticklabels=range(1, n_features+1),
                        yticklabels=prop_names)
                
                plt.title(f'RF Importance: {feature_type.upper()} vs Descriptors')
                plt.xlabel('Feature Dimension')
                plt.ylabel('Descriptor Property')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'rf_{feature_type}_importance.png'))
                plt.close()
        
        # CCA scores
        if 'cca' in descriptor_correlations:
            cca_scores = {k: v for k, v in descriptor_correlations['cca'].items() 
                        if k != 'details' and k in feature_types}
            
            if cca_scores:
                plt.figure(figsize=(10, 6))
                
                # Get feature types and scores
                cca_features = list(cca_scores.keys())
                cca_values = [np.abs(cca_scores[ft]) for ft in cca_features]
                
                # Create bar chart
                bars = plt.bar(range(len(cca_features)), cca_values, color='salmon')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom')
                
                plt.xticks(range(len(cca_features)), [ft.upper() for ft in cca_features])
                plt.ylabel('CCA Correlation')
                plt.title('Canonical Correlation Analysis Scores')
                plt.ylim(0, max(max(cca_values) * 1.2, 0.1))
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'cca_scores.png'))
                plt.close()

    def _plot_descriptor_correlation_distributions(self, descriptor_correlations, feature_types):
        """
        For each method (pearson, spearman, etc.), create distribution plots (box/violin)
        of the absolute correlation values across feature dimensions, for all descriptor properties.
        """
        output_dir = self.output_dir
        # Only do for known methods with dimension-based correlation
        methods_to_plot = ['pearson', 'spearman', 'mutual_info']
        
        for method in methods_to_plot:
            if method not in descriptor_correlations:
                continue
            
            # Build a DataFrame
            data_rows = []
            for ft in feature_types:
                if ft not in descriptor_correlations[method]:
                    continue
                
                # For each descriptor property
                prop_dict = descriptor_correlations[method][ft]
                for prop_name, corr_array in prop_dict.items():
                    # `corr_array` is an array of shape [n_features] typically
                    for val in corr_array:
                        data_rows.append({
                            'Method': method.capitalize(),
                            'FeatureType': ft.upper(),
                            'DescriptorProperty': prop_name,
                            'Correlation': abs(val)  # or just val if you don't want absolute
                        })
            
            if not data_rows:
                continue
            
            df = pd.DataFrame(data_rows)
            
            # Option 1: a single boxplot for each feature type (all properties & dimensions combined)
            # We can group by feature type:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='FeatureType', y='Correlation', data=df, palette='Spectral')
            sns.swarmplot(x='FeatureType', y='Correlation', data=df, color='0.3', alpha=0.5)
            plt.title(f"Distribution of {method.capitalize()} Correlations (Abs) Across Feature Dims & Properties")
            plt.ylim(0, 1)  # Typically correlation is in [0,1] if absolute value
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{method}_descriptor_correlation_distribution.png"))
            plt.close()
            
            # Option 2: If you want a separate plot per feature type:
            # (just comment out the above single plot, and do the below in a loop)
            # ...

    def _plot_feature_correlation_scatter(self, rankings):
        """
        Create a 2D scatter plot of Mask vs. Descriptor correlation scores
        for each feature type (from the final normalized scores).
        """
        output_dir = self.output_dir
        
        scores = rankings['scores']  # e.g. { 'raw': {'mask':..., 'descriptor':..., 'total':...}, ...}
        feature_types = list(scores.keys())
        
        x_vals = [scores[ft]['mask'] for ft in feature_types]
        y_vals = [scores[ft]['descriptor'] for ft in feature_types]
        
        plt.figure(figsize=(8, 6))
        for ft, x, y in zip(feature_types, x_vals, y_vals):
            plt.scatter(x, y, label=ft.upper(), s=80)
            # Optionally annotate:
            plt.text(x+0.01, y, ft.upper(), fontsize=9)
        
        plt.xlabel("Normalized Mask Score")
        plt.ylabel("Normalized Descriptor Score")
        plt.title("Feature Comparison: Mask vs. Descriptor Correlation Scores")
        plt.grid(True, linestyle='--', alpha=0.7)
        # If there's room, you can place the legend or rely on text labels
        # plt.legend()
        
        # Optionally force axis [0..1] if your scores are normalized
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_correlation_scatter.png"))
        plt.close()

    def _plot_cca_results(self, cca_data):
        """
        Plot detailed results from Canonical Correlation Analysis.
        
        Args:
            cca_data: Dictionary containing CCA results and details
        """
        if 'details' not in cca_data:
            return
        
        cca_details = cca_data['details']
        
        # For each feature type with CCA details, create visualization
        for feature_type, details in cca_details.items():
            # Extract data
            x_loadings = details.get('x_loadings')
            y_loadings = details.get('y_loadings')
            x_scores = details.get('x_scores')
            y_scores = details.get('y_scores')
            correlations = details.get('correlations')
            feature_names = details.get('feature_names')
            descriptor_names = details.get('descriptor_names')
            
            if x_loadings is None or y_loadings is None or len(correlations) == 0:
                print(f"Warning: Incomplete CCA details for {feature_type}")
                continue
            
            # 1. Canonical Correlations plot
            plt.figure(figsize=(10, 5))
            components = range(1, len(correlations) + 1)
            plt.bar(components, correlations, color='skyblue')
            plt.xlabel('Canonical Component')
            plt.ylabel('Correlation')
            plt.title(f'Canonical Correlations for {feature_type.upper()} Features')
            plt.xticks(components)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'cca_{feature_type}_correlations.png'))
            plt.close()
            
            # 2. First canonical variate scatter plot
            if x_scores is not None and y_scores is not None and x_scores.shape[1] > 0 and y_scores.shape[1] > 0:
                plt.figure(figsize=(8, 8))
                plt.scatter(x_scores[:, 0], y_scores[:, 0], alpha=0.7)
                plt.xlabel('Features Canonical Variate 1')
                plt.ylabel('Descriptors Canonical Variate 1')
                plt.title(f'First Canonical Variate Scatter Plot ({feature_type.upper()})')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add correlation in title
                if len(correlations) > 0:
                    plt.title(f'First Canonical Variate Scatter Plot ({feature_type.upper()})\nCorrelation: {correlations[0]:.3f}')
                
                # Add reference line
                min_val = min(np.min(x_scores[:, 0]), np.min(y_scores[:, 0]))
                max_val = max(np.max(x_scores[:, 0]), np.max(y_scores[:, 0]))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'cca_{feature_type}_scatter.png'))
                plt.close()
            
            # 3. Loadings heatmap
            if x_loadings is not None and y_loadings is not None:
                # Feature loadings
                plt.figure(figsize=(12, 6))
                n_components = min(3, x_loadings.shape[1])  # Show at most first 3 components
                feat_data = x_loadings[:, :n_components]
                
                if len(feature_names) != feat_data.shape[0]:
                    feature_names = [f"Feature {i+1}" for i in range(feat_data.shape[0])]
                
                if len(feature_names) > 20:
                    # For large number of features, show only top contributors
                    feat_data = feat_data[:20, :]
                    feature_names = feature_names[:20]
                
                sns.heatmap(feat_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                        xticklabels=[f"CV{i+1}" for i in range(n_components)],
                        yticklabels=feature_names)
                plt.title(f'Feature Loadings for {feature_type.upper()}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'cca_{feature_type}_feature_loadings.png'))
                plt.close()
                
                # Descriptor loadings
                plt.figure(figsize=(10, 6))
                n_components = min(3, y_loadings.shape[1])  # Show at most first 3 components
                desc_data = y_loadings[:, :n_components]
                
                sns.heatmap(desc_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                        xticklabels=[f"CV{i+1}" for i in range(n_components)],
                        yticklabels=descriptor_names)
                plt.title(f'Descriptor Loadings for {feature_type.upper()}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'cca_{feature_type}_descriptor_loadings.png'))
                plt.close()

    def _plot_segmentation_info(self, segments_info):
        """
        Plot information about signal segmentation.
        
        Args:
            segments_info: List of dictionaries containing segmentation information
        """
        # Count the number of segments per signal
        if not segments_info:
            return
        
        num_segments = [info.get('num_segments', 0) for info in segments_info]
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(1, len(num_segments) + 1), num_segments, color='skyblue')
        plt.xlabel('Signal Index')
        plt.ylabel('Number of Segments')
        plt.title('Number of RMS Peak-Based Segments per Signal')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'segmentation_counts.png'))
        plt.close()
        
        # Plot information about the thresholds and peak values
        thresholds = [info.get('threshold', 0) for info in segments_info]
        
        # Collect all peak values
        all_peaks = []
        for info in segments_info:
            peak_values = info.get('peak_values', [])
            if len(peak_values) > 0:
                all_peaks.extend(peak_values)
        
        if all_peaks:
            plt.figure(figsize=(10, 5))
            plt.hist(all_peaks, bins=20, alpha=0.7, color='skyblue')
            
            # Plot the thresholds as vertical lines
            for i, threshold in enumerate(thresholds):
                plt.axvline(threshold, color='r', linestyle='--', alpha=0.3, 
                        label='Threshold' if i == 0 else None)
            
            plt.xlabel('RMS Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of RMS Peak Values')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'rms_peak_distribution.png'))
            plt.close()

    def _calculate_rf_importance(self, feature_matrices, descriptors):
        """
        Calculate feature importance using Random Forest regression.
        
        Args:
            feature_matrices: Dictionary of feature matrices for each feature type
            descriptors: Dictionary of descriptor properties
            
        Returns:
            dict: Dictionary of feature importance scores for each feature type
        """
        importance_scores = {ft: {} for ft in self.feature_types}
        
        for prop_name, prop_values in descriptors.items():
            prop_array = np.array(prop_values)
            
            if np.std(prop_array) == 0:
                print(f"Warning: No variation in property '{prop_name}', skipping RF importance calculation")
                continue
            
            # Process each feature type
            for feature_type in self.feature_types:
                if feature_type not in feature_matrices:
                    continue
                
                features = feature_matrices[feature_type]
                
                if len(prop_array) != features.shape[0]:
                    # Skip if lengths don't match
                    continue
                
                # Train Random Forest
                if features.shape[0] > 3 and np.std(prop_array) > 0:
                    try:
                        rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf.fit(features, prop_array)
                        importance_scores[feature_type][prop_name] = rf.feature_importances_
                    except Exception as e:
                        print(f"Error training Random Forest for {feature_type} on {prop_name}: {e}")
                        importance_scores[feature_type][prop_name] = np.zeros(features.shape[1])
                else:
                    importance_scores[feature_type][prop_name] = np.zeros(features.shape[1])
        
        return importance_scores

    def _calculate_canonical_correlation(self, feature_matrices, descriptors):
        """
        Perform Canonical Correlation Analysis between features and descriptors.
        
        Args:
            feature_matrices: Dictionary of feature matrices for each feature type
            descriptors: Dictionary of descriptor properties
            
        Returns:
            dict: Dictionary of CCA scores and additional information
        """
        print("\nPerforming Canonical Correlation Analysis...")
        
        # Create a matrix of descriptor properties
        prop_names = list(descriptors.keys())
        
        # Check if there are enough descriptor properties
        if len(prop_names) < 2:
            print("Warning: Not enough descriptor properties for CCA, need at least 2")
            return {ft: 0 for ft in self.feature_types}
        
        # Create descriptor matrix, removing any columns with no variation
        descriptor_cols = []
        valid_prop_names = []
        
        for prop in prop_names:
            prop_array = np.array(descriptors[prop])
            if np.std(prop_array) > 0:
                descriptor_cols.append(prop_array)
                valid_prop_names.append(prop)
        
        if len(descriptor_cols) < 2:
            print("Warning: Not enough varying descriptor properties for CCA, need at least 2")
            return {ft: 0 for ft in self.feature_types}
        
        descriptor_matrix = np.column_stack(descriptor_cols)
        
        # Initialize results
        cca_results = {ft: 0 for ft in self.feature_types}
        cca_details = {}
        
        # Perform CCA for each feature type
        for feature_type in self.feature_types:
            if feature_type not in feature_matrices:
                continue
            
            features = feature_matrices[feature_type]
            
            # Check if we have enough samples and features
            if features.shape[0] <= max(features.shape[1], descriptor_matrix.shape[1]):
                print(f"Warning: Not enough samples for CCA on {feature_type} features")
                print(f"  Samples: {features.shape[0]}, Features: {features.shape[1]}, Descriptors: {descriptor_matrix.shape[1]}")
                continue
            
            # Standardize features and descriptors
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            
            try:
                X_std = scaler_x.fit_transform(features)
                Y_std = scaler_y.fit_transform(descriptor_matrix)
                
                # Determine number of components
                n_components = min(features.shape[1], descriptor_matrix.shape[1])
                
                if n_components > 0:
                    cca = CCA(n_components=n_components)
                    
                    # Fit CCA
                    cca.fit(X_std, Y_std)
                    
                    # Transform data to get canonical variates
                    X_c, Y_c = cca.transform(X_std, Y_std)
                    
                    # Calculate correlations between canonical variates
                    correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
                    
                    # Store the first canonical correlation
                    cca_results[feature_type] = correlations[0]
                    
                    # Store details for visualization
                    cca_details[feature_type] = {
                        'x_loadings': cca.x_loadings_,
                        'y_loadings': cca.y_loadings_,
                        'x_scores': X_c,
                        'y_scores': Y_c,
                        'correlations': correlations,
                        'feature_names': [f"Feature {i}" for i in range(features.shape[1])],
                        'descriptor_names': valid_prop_names
                    }
                    
                    print(f"CCA for {feature_type} features:")
                    print(f"  Number of components: {n_components}")
                    print(f"  Canonical correlations: {[round(c, 3) for c in correlations[:min(3, len(correlations))]]}" + 
                        ("..." if n_components > 3 else ""))
                    
            except Exception as e:
                print(f"Error performing CCA for {feature_type} features: {e}")
                cca_results[feature_type] = 0
        
        # Add the details to the results
        cca_results['details'] = cca_details
        
        return cca_results

    def rank_features(self, mask_correlations, descriptor_correlations):
        """
        Rank feature representations based on their correlation with masks and descriptors.
        
        Args:
            mask_correlations: Dictionary of mask correlation results
            descriptor_correlations: Dictionary of descriptor correlation results
            
        Returns:
            dict: Dictionary of feature rankings
        """
        print("Ranking features by correlation strength...")
        
        # Get available feature types
        feature_types = set()
        
        # Check mask correlations
        for method, feature_scores in mask_correlations.items():
            feature_types.update(feature_scores.keys())
        
        # Check descriptor correlations
        for method in ['pearson', 'spearman', 'mutual_info']:
            if method in descriptor_correlations:
                feature_types.update(descriptor_correlations[method].keys())
        
        # Convert to list and sort for consistent ordering
        feature_types = sorted(list(feature_types))
        print(f"Available feature types for ranking: {feature_types}")
        
        # Initialize scores
        scores = {ft: {'mask': 0, 'descriptor': 0, 'total': 0} for ft in feature_types}
        
        # Aggregate mask correlation scores
        for method, feature_scores in mask_correlations.items():
            for feature_type in feature_types:
                if feature_type in feature_scores:
                    values = feature_scores[feature_type]
                    scores[feature_type]['mask'] += np.mean(values) if values else 0
        
        # Normalize mask scores
        mask_values = [scores[ft]['mask'] for ft in feature_types]
        if mask_values:
            min_mask, max_mask = min(mask_values), max(mask_values)
            if max_mask > min_mask:
                for feature_type in scores:
                    scores[feature_type]['mask'] = (scores[feature_type]['mask'] - min_mask) / (max_mask - min_mask)
        
        # Aggregate descriptor correlation scores
        # Pearson and Spearman: take mean of absolute values
        for method in ['pearson', 'spearman']:
            if method in descriptor_correlations:
                for feature_type in feature_types:
                    if feature_type in descriptor_correlations[method]:
                        feature_scores = descriptor_correlations[method][feature_type]
                        avg_abs_corr = 0
                        count = 0
                        
                        for prop_name, values in feature_scores.items():
                            avg_abs_corr += np.mean(np.abs(values))
                            count += 1
                        
                        if count > 0:
                            scores[feature_type]['descriptor'] += avg_abs_corr / count
        
        # Mutual Information: take mean
        if 'mutual_info' in descriptor_correlations:
            for feature_type in feature_types:
                if feature_type in descriptor_correlations['mutual_info']:
                    feature_scores = descriptor_correlations['mutual_info'][feature_type]
                    avg_mi = 0
                    count = 0
                    
                    for prop_name, values in feature_scores.items():
                        avg_mi += np.mean(values)
                        count += 1
                    
                    if count > 0:
                        scores[feature_type]['descriptor'] += avg_mi / count
        
        # CCA scores
        if 'cca' in descriptor_correlations:
            for feature_type in feature_types:
                if feature_type in descriptor_correlations['cca'] and feature_type != 'details':
                    scores[feature_type]['descriptor'] += np.abs(descriptor_correlations['cca'][feature_type])
        
        # Random Forest importance: take mean across properties
        if 'rf_importance' in descriptor_correlations:
            for feature_type in feature_types:
                if feature_type in descriptor_correlations['rf_importance']:
                    feature_scores = descriptor_correlations['rf_importance'][feature_type]
                    avg_importance = 0
                    count = 0
                    
                    for prop_name, values in feature_scores.items():
                        avg_importance += np.mean(values)
                        count += 1
                    
                    if count > 0:
                        scores[feature_type]['descriptor'] += avg_importance / count
        
        # Normalize descriptor scores
        desc_values = [scores[ft]['descriptor'] for ft in feature_types]
        if desc_values:
            min_desc, max_desc = min(desc_values), max(desc_values)
            if max_desc > min_desc:
                for feature_type in scores:
                    scores[feature_type]['descriptor'] = (scores[feature_type]['descriptor'] - min_desc) / (max_desc - min_desc)
        
        # Calculate total scores (equal weight to mask and descriptor correlations)
        for feature_type in scores:
            scores[feature_type]['total'] = (scores[feature_type]['mask'] + scores[feature_type]['descriptor']) / 2
        
        # Create rankings
        rankings = {
            'mask_correlation': sorted(feature_types, key=lambda x: scores[x]['mask'], reverse=True),
            'descriptor_correlation': sorted(feature_types, key=lambda x: scores[x]['descriptor'], reverse=True),
            'overall': sorted(feature_types, key=lambda x: scores[x]['total'], reverse=True),
            'scores': scores
        }
        
        return rankings

    def load_data(self):
        """
        Load accelerometer data, damage masks, and crack descriptors.
        
        Returns:
            tuple: (accel_data, masks, descriptors)
        """
        print("Loading accelerometer data...")
        accel_dict = load_accelerometer_data()
        
        print("Loading and processing damage masks & descriptors...")
        test_ids = sorted(list(accel_dict.keys()))
        masks = {}
        binary_masks = {}
        descriptors = {}
        skeletons = {}
        
        # Process each test using functions from data_loader.py
        previous_skeleton = None
        params = {
            'keypoint_count': 20,
            'max_gap': 5,
            'curved_threshold': 10,
            'curved_angle_threshold': 85,
            'straight_angle_threshold': 20,
            'min_segment_length': 2
        }
        
        for test_id in test_ids:
            # Load image and create binary mask
            combined_image = load_combined_label(test_id)
            binary_mask = compute_binary_mask(combined_image)
            
            # Store the binary mask
            binary_masks[test_id] = binary_mask
            
            # Process cracks with identification
            desc, current_skeleton = process_test_cracks(
                binary_mask, 
                previous_skeleton,
                keypoint_count=params.get('keypoint_count', 20),
                max_gap=params.get('max_gap', 5),
                curved_threshold=params.get('curved_threshold', 10),
                curved_angle_threshold=params.get('curved_angle_threshold', 85),
                straight_angle_threshold=params.get('straight_angle_threshold', 20),
                min_segment_length=params.get('min_segment_length', 2)
            )
            
            # Store descriptors, mask, and skeleton
            descriptors[test_id] = desc
            masks[test_id] = combined_image
            skeletons[test_id] = current_skeleton
            
            # Update for next test
            previous_skeleton = current_skeleton
        
        # Convert accel_dict to list for easier processing
        accel_data = []
        for test_id in test_ids:
            # Each test may have multiple accelerometer signals, take the first one
            if test_id in accel_dict and len(accel_dict[test_id]) > 0:
                accel_data.append(accel_dict[test_id][0])
        
        print(f"Loaded {len(accel_data)} accelerometer signals")
        print(f"Loaded {len(masks)} masks")
        print(f"Loaded {len(descriptors)} sets of crack descriptors")
        
        # Analyze descriptor structure
        self._analyze_descriptor_structure(descriptors)
        
        return accel_data, masks, binary_masks, descriptors, test_ids
    
    def _calculate_time_domain_statistics(self, signal):
        """
        Calculate time-domain statistical properties from a signal.
        
        Args:
            signal: Input time-domain signal, shape [time_steps, channels]
            
        Returns:
            numpy.ndarray: Vector of time-domain statistical properties
        """
        # Calculate statistics for each channel, then average
        n_channels = signal.shape[1]
        channel_stats = []
        
        for ch in range(n_channels):
            channel_data = signal[:, ch]
            
            # Basic statistics
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            max_val = np.max(channel_data)
            min_val = np.min(channel_data)
            median_val = np.median(channel_data)
            rms_val = np.sqrt(np.mean(channel_data**2))
            
            # Peak-to-peak amplitude
            p2p = max_val - min_val
            
            # Zero-crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(channel_data))))
            zero_crossing_rate = zero_crossings / (len(channel_data) - 1)
            
            # Kurtosis (peakedness of the distribution)
            kurtosis = np.mean((channel_data - mean_val)**4) / (std_val**4) if std_val > 0 else 0
            
            # Skewness (asymmetry of the distribution)
            skewness = np.mean((channel_data - mean_val)**3) / (std_val**3) if std_val > 0 else 0
            
            # Energy
            energy = np.sum(channel_data**2)
            
            channel_stats.append([mean_val, std_val, max_val, min_val, median_val, 
                                rms_val, p2p, zero_crossing_rate, kurtosis, skewness, energy])
        
        # Average across channels
        avg_stats = np.mean(channel_stats, axis=0)
        return avg_stats

    def _calculate_feature_statistics(self, feature_array):
        """
        Calculate statistical properties from a feature array.
        
        Args:
            feature_array: Input feature array
            
        Returns:
            numpy.ndarray: Vector of statistical properties
        """
        # Basic statistics
        mean_val = np.mean(feature_array)
        std_val = np.std(feature_array)
        max_val = np.max(feature_array)
        min_val = np.min(feature_array)
        median_val = np.median(feature_array)
        
        # Energy
        energy = np.sum(feature_array**2)
        
        # Entropy (if applicable)
        if np.min(feature_array) >= 0:
            normalized = feature_array / (np.sum(feature_array) + 1e-10)
            entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
        else:
            entropy = 0
        
        # Return vector of statistics
        return np.array([mean_val, std_val, max_val, min_val, median_val, energy, entropy])
    
    def _prepare_features_for_correlation(self, features, test_ids):
        """
        Prepare features for correlation analysis by creating consistent representations.
        
        Args:
            features: Dictionary of extracted features
            test_ids: List of test IDs
            
        Returns:
            dict: Dictionary of feature matrices for each feature type
        """
        # Initialize feature arrays
        feature_lists = {ft: [] for ft in self.feature_types}
        
        for i, test_id in enumerate(test_ids):
            # Process each feature type
            for feature_type in self.feature_types:
                # Skip if feature type is not in features
                if feature_type not in features or i >= len(features[feature_type]):
                    continue
                
                # Get feature for this test
                feature = features[feature_type][i]
                
                # Calculate statistics based on feature type
                if feature_type == 'raw':
                    # For raw data, calculate various time-domain statistics
                    raw_stats = self._calculate_time_domain_statistics(feature)
                    feature_lists[feature_type].append(raw_stats)
                    
                elif feature_type == 'rms':
                    # For RMS, calculate statistics (mean, std, max, etc.)
                    rms_stats = self._calculate_feature_statistics(feature)
                    feature_lists[feature_type].append(rms_stats)
                    
                elif feature_type == 'psd':
                    # For PSD, calculate statistics across frequency bands
                    psd_stats = self._calculate_feature_statistics(feature)
                    feature_lists[feature_type].append(psd_stats)
                    
                elif feature_type == 'spectrogram':
                    # For spectrogram, calculate statistics across time-frequency representation
                    if len(feature.shape) == 3 and feature.shape[0] > 0:
                        # Average across channels
                        avg_spec = np.mean(feature, axis=0)
                        spec_stats = self._calculate_feature_statistics(avg_spec)
                    else:
                        # Handle empty spectrogram
                        spec_stats = np.zeros(7)  # Default stats vector
                    
                    feature_lists[feature_type].append(spec_stats)
        
        # Convert lists to arrays
        feature_matrices = {}
        for ft in self.feature_types:
            if feature_lists[ft]:
                feature_matrices[ft] = np.array(feature_lists[ft])
        
        return feature_matrices
    
    def run_full_analysis(self, fs=1000, window_size=256, overlap=0.5, segment_by_peaks=True):
        """
        Run the complete correlation analysis pipeline.
        
        Args:
            fs: Sampling frequency in Hz
            window_size: Size of the window for feature computation
            overlap: Overlap between consecutive windows
            segment_by_peaks: Whether to segment signals by RMS peaks
            
        Returns:
            dict: Summary of results
        """
        # 1. Load data
        accel_data, masks, binary_masks, descriptors, test_ids = self.load_data()
        
        # 2. Extract features
        features = self.extract_features(accel_data, fs, window_size, overlap, segment_by_peaks)
        
        # Update feature types based on the extracted features
        self.feature_types = sorted([ft for ft in features.keys() 
                                if ft not in ['psd_freqs', 'spec_freqs', 'spec_times', 'segments_info', 'complex_spectrogram']])
        
        print(f"Analyzing feature types: {self.feature_types}")
        
        # 3. Compute correlations with masks
        mask_correlations = self.compute_mask_correlations(features, binary_masks, test_ids)
        
        # 4. Compute correlations with descriptors
        descriptor_correlations = self.compute_descriptor_correlations(features, descriptors, test_ids)
        
        # 5. Rank features
        rankings = self.rank_features(mask_correlations, descriptor_correlations)
        
        # 6. Visualize results
        segments_info = features.get('segments_info')
        self.visualize_results(mask_correlations, descriptor_correlations, rankings, segments_info)
        
        # Create summary
        summary = {
            'best_for_masks': rankings['mask_correlation'][0] if rankings['mask_correlation'] else None,
            'best_for_descriptors': rankings['descriptor_correlation'][0] if rankings['descriptor_correlation'] else None,
            'best_overall': rankings['overall'][0] if rankings['overall'] else None,
            'rankings': rankings,
            'mask_correlations': mask_correlations,
            'descriptor_correlations': descriptor_correlations,
            'segments_info': segments_info
        }
        
        return summary
    
    def _analyze_descriptor_structure(self, descriptors):
        """
        Analyze the structure of crack descriptors to understand their format.
        
        Args:
            descriptors: Dictionary of crack descriptors
        """
        print("\nAnalyzing descriptor structure...")
        
        # Get a sample descriptor
        sample_desc = None
        for test_id, desc_list in descriptors.items():
            if desc_list and len(desc_list) > 0:
                sample_desc = desc_list[0]
                break
        
        if sample_desc is None:
            print("No descriptors found.")
            return
        
        # Print sample descriptor structure
        print(f"Sample descriptor format (length {len(sample_desc)}):")
        if len(sample_desc) >= 9:
            print(f"  [0, 1]: Start point coordinates ({sample_desc[0]}, {sample_desc[1]})")
            print(f"  [2, 3]: End point coordinates ({sample_desc[2]}, {sample_desc[3]})")
            print(f"  [4]: Length: {sample_desc[4]}")
            print(f"  [5]: Width: {sample_desc[5]}")
            print(f"  [6]: Curvature: {sample_desc[6]}")
            print(f"  [7]: Angle: {sample_desc[7]}")
            print(f"  [8]: New crack flag: {sample_desc[8]}")
        else:
            print(f"  Format: {sample_desc}")
        
        # Count total number of descriptors
        total_descriptors = sum(len(desc_list) for desc_list in descriptors.values())
        print(f"Total number of crack descriptors: {total_descriptors}")
        
        # Count descriptors per test
        counts = [len(desc_list) for desc_list in descriptors.values()]
        if counts:
            print(f"Descriptors per test: min={min(counts)}, max={max(counts)}, average={np.mean(counts):.2f}")
        
        # Check for new cracks
        new_cracks_count = 0
        for desc_list in descriptors.values():
            for desc in desc_list:
                if len(desc) > 8 and desc[8] > 0.5:
                    new_cracks_count += 1
        
        print(f"Number of new cracks detected: {new_cracks_count}")
    
    def extract_features(self, accel_data, fs=1000, window_size=256, overlap=0.5, segment_by_peaks=True):
        """
        Extract features from accelerometer data.
        
        Args:
            accel_data: List of accelerometer signals
            fs: Sampling frequency in Hz
            window_size: Size of the window for feature computation
            overlap: Overlap between consecutive windows
            segment_by_peaks: Whether to segment signals by RMS peaks
            
        Returns:
            dict: Dictionary of features
        """
        print("Extracting features from accelerometer data...")
        return self.feature_extractor.extract_features(accel_data, fs, window_size, overlap, segment_by_peaks)
    
    def compute_mask_correlations(self, features, binary_masks, test_ids):
        """
        Compute correlations between feature representations and damage masks.
        
        Args:
            features: Dictionary of extracted features
            binary_masks: Dictionary of binary masks
            test_ids: List of test IDs
            
        Returns:
            dict: Dictionary of correlation results
        """
        print("Computing correlations between features and damage masks...")
        
        # Initialize correlation containers
        ssim_scores = {ft: [] for ft in self.feature_types}
        pearson_scores = {ft: [] for ft in self.feature_types}
        
        # Process each test
        for i, test_id in enumerate(test_ids):
            if test_id not in binary_masks:
                continue
            
            # Get mask for this test
            mask = binary_masks[test_id]
            
            # Process each feature type
            for feature_type in self.feature_types:
                # Skip if feature type is not in features
                if feature_type not in features or i >= len(features[feature_type]):
                    continue
                
                # Get feature for this test
                feature = features[feature_type][i]
                
                # Convert to suitable format for correlation
                feature_image = self._reshape_feature_to_image(feature, mask.shape, feature_type)
                
                # Calculate SSIM (Structural Similarity Index)
                mask_float = mask.astype(float) / np.max(mask) if np.max(mask) > 0 else mask.astype(float)
                
                # Normalize feature image for comparison
                feature_norm = self._normalize_image(feature_image)
                
                # Calculate SSIM
                ssim_score = ssim(feature_norm, mask_float, data_range=1.0)
                ssim_scores[feature_type].append(ssim_score)
                
                # Calculate Pearson correlation
                pearson_score = self._calculate_spatial_correlation(feature_norm, mask_float)
                pearson_scores[feature_type].append(pearson_score)
        
        # Organize results
        correlation_results = {
            'ssim': ssim_scores,
            'pearson': pearson_scores
        }
        
        return correlation_results
    
    def _normalize_image(self, image):
        """Normalize image to range [0, 1]"""
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return np.zeros_like(image)
    
    def _calculate_spatial_correlation(self, image1, image2):
        """Calculate correlation between two spatial images"""
        # Flatten images
        flat1 = image1.flatten()
        flat2 = image2.flatten()
        
        # Calculate Pearson correlation
        if np.std(flat1) > 0 and np.std(flat2) > 0:
            corr, _ = pearsonr(flat1, flat2)
            return corr
        return 0.0
    
    def _reshape_feature_to_image(self, feature, target_shape, feature_type):
        """
        Reshape feature array to target image shape for comparison.
        
        Args:
            feature: Feature array
            target_shape: Target shape for the image
            feature_type: Type of feature ('raw', 'rms', 'psd', or 'spectrogram')
            
        Returns:
            numpy.ndarray: Reshaped feature as 2D image
        """
        # Handle different feature types differently
        if feature_type == 'raw':
            # For raw data, use RMS across time to create a 1D profile
            feature_2d = np.sqrt(np.mean(feature**2, axis=1))
            # Convert 1D to 2D by reshaping
            size = int(np.ceil(np.sqrt(feature_2d.shape[0])))
            padded = np.zeros((size*size,))
            padded[:feature_2d.shape[0]] = feature_2d
            feature_2d = padded.reshape(size, size)
            
        elif feature_type == 'rms':
            # For RMS data (n_windows, n_channels)
            if len(feature.shape) == 1:
                # 1D feature, reshape to square
                size = int(np.ceil(np.sqrt(feature.shape[0])))
                padded = np.zeros((size*size,))
                padded[:feature.shape[0]] = feature
                feature_2d = padded.reshape(size, size)
            else:
                # Use as is for 2D
                feature_2d = feature
                
        elif feature_type == 'psd':
            # For PSD data, reshape as needed
            if len(feature.shape) == 1:
                # 1D PSD, reshape to square
                size = int(np.ceil(np.sqrt(feature.shape[0])))
                padded = np.zeros((size*size,))
                padded[:feature.shape[0]] = feature
                feature_2d = padded.reshape(size, size)
            else:
                # 2D PSD, use as is
                feature_2d = feature
                
        elif feature_type == 'spectrogram':
            # For spectrogram (n_channels, n_freq, n_time)
            if len(feature.shape) == 3 and feature.shape[0] > 0:
                # Average across channels
                feature_2d = np.mean(feature, axis=0)
            elif len(feature.shape) == 2:
                # Already 2D
                feature_2d = feature
            else:
                # Handle unexpected shape
                feature_2d = np.zeros((10, 10))  # Placeholder
        else:
            # Unknown feature type
            feature_2d = np.zeros((10, 10))  # Placeholder
        
        # Resize to target shape using simple interpolation
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_factors = (target_shape[0] / feature_2d.shape[0], 
                        target_shape[1] / feature_2d.shape[1])
        
        # Resize
        resized = zoom(feature_2d, zoom_factors, order=1)
        
        return resized
    
    def compute_descriptor_correlations(self, features, descriptors, test_ids):
        """
        Compute correlations between feature representations and crack descriptors.
        
        Args:
            features: Dictionary of extracted features
            descriptors: Dictionary of crack descriptors
            test_ids: List of test IDs
            
        Returns:
            dict: Dictionary of correlation results
        """
        print("Computing correlations between features and crack descriptors...")
        
        # Debug: Check input parameters
        print(f"descriptors type: {type(descriptors)}")
        print(f"test_ids length: {len(test_ids) if test_ids else 'None'}")
        
        # Safely extract descriptor properties
        if descriptors is None:
            print("Warning: descriptors is None")
            return {}
        
        # Extract descriptor properties for correlation
        descriptor_properties = self._extract_descriptor_properties(descriptors, test_ids)
        
        # Debug: Check extracted properties
        print(f"descriptor_properties type: {type(descriptor_properties)}")
        print(f"descriptor_properties keys: {list(descriptor_properties.keys()) if descriptor_properties else 'None'}")
        
        # Prepare features for correlation
        feature_matrices = self._prepare_features_for_correlation(features, test_ids)
        
        # Debug: Check feature matrices
        print(f"feature_matrices keys: {list(feature_matrices.keys()) if feature_matrices else 'None'}")
        
        # Calculate correlations
        correlations = {}
        
        # Pearson correlation
        pearson_results = {}
        for ft in self.feature_types:
            if ft in feature_matrices:
                pearson_results[ft] = self._calculate_feature_descriptor_correlation(
                    feature_matrices[ft], descriptor_properties, method='pearson')
        
        correlations['pearson'] = pearson_results
        
        # Spearman correlation
        spearman_results = {}
        for ft in self.feature_types:
            if ft in feature_matrices:
                spearman_results[ft] = self._calculate_feature_descriptor_correlation(
                    feature_matrices[ft], descriptor_properties, method='spearman')
        
        correlations['spearman'] = spearman_results
        
        # Mutual Information
        mi_results = {}
        for ft in self.feature_types:
            if ft in feature_matrices:
                mi_results[ft] = self._calculate_feature_descriptor_correlation(
                    feature_matrices[ft], descriptor_properties, method='mutual_info')
        
        correlations['mutual_info'] = mi_results
        
        # Feature importance using Random Forest
        rf_importance = self._calculate_rf_importance(feature_matrices, descriptor_properties)
        correlations['rf_importance'] = rf_importance
        
        # Canonical Correlation Analysis
        cca_results = self._calculate_canonical_correlation(feature_matrices, descriptor_properties)
        correlations['cca'] = cca_results
        
        return correlations
    
    def _calculate_feature_descriptor_correlation(self, features, descriptors, method='pearson'):
        """
        Calculate correlation between features and descriptor properties.
        
        Args:
            features: Feature array [n_tests, n_features]
            descriptors: Dictionary of descriptor properties
            method: Correlation method ('pearson', 'spearman', or 'mutual_info')
            
        Returns:
            dict: Dictionary of correlation scores for each descriptor property
        """
        correlation_scores = {}

         # Safety check
        if descriptors is None:
            print("Warning: descriptors is None in _calculate_feature_descriptor_correlation")
            return correlation_scores
        
        for prop_name, prop_values in descriptors.items():
            if len(prop_values) != features.shape[0]:
                # Skip if lengths don't match
                continue
            
            # Convert to array
            prop_array = np.array(prop_values)
            
            # Check if there's enough variation in the property values
            if np.std(prop_array) == 0:
                print(f"Warning: No variation in property '{prop_name}', skipping correlation calculation")
                correlation_scores[prop_name] = np.zeros(features.shape[1])
                continue
            
            # Calculate correlation for each feature dimension
            if method == 'pearson':
                scores = []
                for i in range(features.shape[1]):
                    if np.std(features[:, i]) > 0 and np.std(prop_array) > 0:
                        try:
                            corr, _ = pearsonr(features[:, i], prop_array)
                            scores.append(corr)
                        except Exception as e:
                            print(f"Error calculating Pearson correlation for feature {i}: {e}")
                            scores.append(0)
                    else:
                        scores.append(0)
            
            elif method == 'spearman':
                scores = []
                for i in range(features.shape[1]):
                    if np.std(features[:, i]) > 0 and np.std(prop_array) > 0:
                        try:
                            corr, _ = spearmanr(features[:, i], prop_array)
                            scores.append(corr)
                        except Exception as e:
                            print(f"Error calculating Spearman correlation for feature {i}: {e}")
                            scores.append(0)
                    else:
                        scores.append(0)
            
            elif method == 'mutual_info':
                try:
                    scores = mutual_info_regression(features, prop_array)
                except Exception as e:
                    print(f"Error calculating mutual information: {e}")
                    scores = np.zeros(features.shape[1])
            
            else:
                raise ValueError(f"Unknown correlation method: {method}")
            
            correlation_scores[prop_name] = np.array(scores)
        
        return correlation_scores

    def _extract_descriptor_properties(self, descriptors, test_ids):
        """
        Extract relevant properties from crack descriptors for correlation analysis.
        
        Args:
            descriptors: Dictionary of crack descriptors
            test_ids: List of test IDs
            
        Returns:
            dict: Dictionary of descriptor properties by test
        """
        properties = {
            'count': [],            # Number of cracks
            'total_length': [],     # Total crack length
            'avg_length': [],       # Average crack length
            'new_cracks': [],       # Number of new cracks
            'avg_curvature': [],    # Average curvature
            'max_length': [],       # Maximum crack length
            'spatial_extent': [],   # Spatial extent of cracks
            'avg_width': [],        # Average crack width
            'total_area': []        # Approximate total area of cracks
        }
        
        for test_id in test_ids:
            if test_id not in descriptors or not descriptors[test_id]:
                # Fill with zeros if no descriptors
                for prop in properties:
                    properties[prop].append(0)
                continue
            
            # Extract properties from descriptors
            desc_list = descriptors[test_id]
            
            # Count
            count = len(desc_list)
            properties['count'].append(count)
            
            # Lengths
            lengths = [desc[4] for desc in desc_list if len(desc) > 4]
            
            if lengths:
                total_length = sum(lengths)
                avg_length = total_length / len(lengths)
                max_length = max(lengths)
            else:
                total_length = 0
                avg_length = 0
                max_length = 0
            
            properties['total_length'].append(total_length)
            properties['avg_length'].append(avg_length)
            properties['max_length'].append(max_length)
            
            # Widths
            widths = [desc[5] for desc in desc_list if len(desc) > 5]
            avg_width = sum(widths) / len(widths) if widths else 0
            properties['avg_width'].append(avg_width)
            
            # Approximate total area (length * width for each crack)
            total_area = 0
            for desc in desc_list:
                if len(desc) > 5:
                    total_area += desc[4] * desc[5]  # length * width
            
            properties['total_area'].append(total_area)
            
            # New cracks
            new_cracks = sum(1 for desc in desc_list if len(desc) > 8 and desc[8] > 0.5)
            properties['new_cracks'].append(new_cracks)
            
            # Curvature
            curvatures = [desc[6] for desc in desc_list if len(desc) > 6]
            avg_curvature = sum(curvatures) / len(curvatures) if curvatures else 0
            properties['avg_curvature'].append(avg_curvature)
            
            # Spatial extent (approximated by bounding box diagonal)
            min_row, min_col = float('inf'), float('inf')
            max_row, max_col = 0, 0
            
            for desc in desc_list:
                if len(desc) < 4:
                    continue
                
                # Check start point
                min_row = min(min_row, desc[0])
                min_col = min(min_col, desc[1])
                max_row = max(max_row, desc[0])
                max_col = max(max_col, desc[1])
                
                # Check end point
                min_row = min(min_row, desc[2])
                min_col = min(min_col, desc[3])
                max_row = max(max_row, desc[2])
                max_col = max(max_col, desc[3])
                
            # Calculate bounding box diagonal
            if min_row < float('inf'):
                diagonal = np.sqrt((max_row - min_row)**2 + (max_col - min_col)**2)
            else:
                diagonal = 0
            
            properties['spatial_extent'].append(diagonal)
        
        return properties

def reconstruct_signal_from_spectrogram(complex_spectrogram, fs=1000, nperseg=256, noverlap=None):
    """
    Reconstruct a time-domain signal from a complex spectrogram using inverse STFT.
    
    Args:
        complex_spectrogram: Complex spectrogram from compute_spectrogram(mode='complex')
        fs: Sampling frequency in Hz
        nperseg: Length of each segment used in the STFT
        noverlap: Number of points to overlap between segments
        
    Returns:
        numpy.ndarray: Reconstructed time-domain signal
    """
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Get the number of channels
    n_channels = complex_spectrogram.shape[0]
    reconstructed_signals = []
    
    # Process each channel
    for ch in range(n_channels):
        # Get the complex spectrogram for this channel
        Zxx = complex_spectrogram[ch]
        
        # Use SciPy's istft to reconstruct the time-domain signal
        _, reconstructed = signal.istft(Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap)
        reconstructed_signals.append(reconstructed)
    
    # Stack the reconstructed signals for all channels
    return np.column_stack(reconstructed_signals)


def main():
    """Main function to run the correlation analysis."""
    print("Starting time-series feature correlation analysis...")
    
    # Create output directory
    output_dir = "correlation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize correlation analyzer
    analyzer = CorrelationAnalyzer(output_dir=output_dir)
    
    # Run full analysis with segmentation enabled
    print("\nRunning analysis with RMS peak-based segmentation...")
    summary = analyzer.run_full_analysis(fs=1000, window_size=256, overlap=0.5, segment_by_peaks=True)
    
    # Print summary
    print("\n=== CORRELATION ANALYSIS SUMMARY ===")
    
    if summary['best_overall']:
        print(f"Best overall feature type: {summary['best_overall'].upper()}")
    if summary['best_for_masks']:
        print(f"Best feature for mask correlation: {summary['best_for_masks'].upper()}")
    if summary['best_for_descriptors']:
        print(f"Best feature for descriptor correlation: {summary['best_for_descriptors'].upper()}")
    
    print("\nFeature rankings:")
    for category, ranking in summary['rankings'].items():
        if category != 'scores' and ranking:
            print(f"  {category}: {[ft.upper() for ft in ranking]}")
    
    # Information about segmentation
    segments_info = summary.get('segments_info')
    if segments_info:
        total_segments = sum(info.get('num_segments', 0) for info in segments_info)
        print(f"\nSegmentation summary:")
        print(f"  Total number of signals: {len(segments_info)}")
        print(f"  Total number of segments: {total_segments}")
        print(f"  Average segments per signal: {total_segments / len(segments_info):.2f}")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()