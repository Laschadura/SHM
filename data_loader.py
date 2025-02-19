import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt  # For high-pass filtering

######################################
# Configuration
######################################
DATA_DIR = "Data"          # Directory containing Test_* folders with CSV files
LABELS_DIR = "Labels"      # Directory containing label images for each test
IMAGE_SHAPE = (256, 768)   # Output combined image shape (height, width)
SKIP_TESTS = [23, 24]      # List of test numbers to skip (if any)
GAUSSIAN_SIGMA = 2.0       # Sigma for Gaussian smoothing
EXPECTED_LENGTH = 12000    # Expected number of datapoints (60s at 200Hz)

# Mapping for perspective images (adjust names if necessary)
perspective_map = {
    'A': 'Arch_Intrados',
    'B': 'North_Spandrel_Wall',
    'C': 'South_Spandrel_Wall'
}

######################################
# Signal Processing Functions
######################################
def high_pass_filter_data(data, cutoff=10.0, fs=200.0, order=5):
    """
    Applies a high-pass Butterworth filter to the data.
    
    Args:
        data (np.ndarray): Input data of shape (time_steps, channels).
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the filter.
        
    Returns:
        np.ndarray: High-pass filtered data of the same shape.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        try:
            filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
        except Exception as e:
            print(f"ERROR: High pass filtering failed for channel {ch}: {e}")
            filtered_data[:, ch] = data[:, ch]
    return filtered_data

def robust_normalize_data(data):
    """
    Normalizes the data using the median and interquartile range (IQR) for each channel.
    
    This approach is robust to outliers (such as the high amplitude impact events).
    
    Args:
        data (np.ndarray): Input data of shape (time_steps, channels).
        
    Returns:
        np.ndarray: Normalized data with median 0 per channel.
    """
    medians = np.median(data, axis=0, keepdims=True)
    q75 = np.percentile(data, 75, axis=0, keepdims=True)
    q25 = np.percentile(data, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0  # Avoid division by zero
    normalized = (data - medians) / iqr
    return normalized

######################################
# Accelerometer Data Loading
######################################
def load_accelerometer_data(data_dir=DATA_DIR, skip_tests=SKIP_TESTS):
    """
    Loads accelerometer CSV files from test directories, applies a high-pass filter
    and robust normalization (median and IQR) to each sample.
    
    If a file contains more than 12,000 datapoints, it is truncated to the first 12,000 rows.
    
    Each test folder (e.g., Test_1) is expected to contain one or more CSV files.
    Accelerometer columns are selected based on the presence of "Accel" in their names.
    
    Returns:
        dict: Keys are test IDs and values are lists of NumPy arrays (each array is one processed CSV sample).
    """
    test_dirs = [d for d in glob.glob(os.path.join(data_dir, "Test_*")) if os.path.isdir(d)]
    tests_data = {}
    
    for test_dir in test_dirs:
        match = re.search(r"Test[_]?(\d+)", os.path.basename(test_dir))
        if not match:
            continue
        test_id = int(match.group(1))
        if test_id in skip_tests:
            print(f"INFO: Skipping Test {test_id}")
            continue

        csv_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
        if not csv_files:
            print(f"WARNING: No CSV files found in {test_dir}")
            continue

        samples = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                accel_cols = [col for col in df.columns if "Accel" in col]
                if not accel_cols:
                    print(f"WARNING: No accelerometer columns in {csv_file}")
                    continue
                data_matrix = df[accel_cols].values.astype(np.float32)
                # Truncate files longer than EXPECTED_LENGTH
                if data_matrix.shape[0] > EXPECTED_LENGTH:
                    data_matrix = data_matrix[:EXPECTED_LENGTH, :]
                # Apply high-pass filtering (10 Hz cutoff) first
                filtered_data = high_pass_filter_data(data_matrix, cutoff=10.0, fs=200.0, order=5)
                # Normalize the filtered data using robust statistics (median and IQR)
                normalized_data = robust_normalize_data(filtered_data)
                samples.append(normalized_data)
            except Exception as e:
                print(f"ERROR: Failed to load {csv_file}: {e}")
        if samples:
            tests_data[test_id] = samples
            print(f"INFO: Loaded Test {test_id} with {len(samples)} sample(s).")
    
    return tests_data

######################################
# Label (Image) Processing
######################################
def load_perspective_image(test_id, perspective, labels_dir=LABELS_DIR, target_size=(256,256)):
    """
    Loads and resizes a single perspective image while maintaining the aspect ratio.
    
    Args:
        test_id (int): Test identifier.
        perspective (str): Perspective key ('A', 'B', or 'C').
        target_size (tuple): Desired output size (height, width) for each perspective.
    
    Returns:
        np.ndarray: The resized and padded image in RGB format.
    """
    label_name = perspective_map.get(perspective)
    file_path = os.path.join(labels_dir, f"Test_{test_id}", f"{label_name}_T{test_id}.png")
    if not os.path.exists(file_path):
        print(f"WARNING: Image not found: {file_path}")
        return None

    img = cv2.imread(file_path)
    if img is None:
        print(f"WARNING: Failed to read image: {file_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    return img_padded

def load_combined_label(test_id, labels_dir=LABELS_DIR, image_shape=IMAGE_SHAPE):
    """
    Loads images from all three perspectives, resizes them, and combines them horizontally.
    
    Returns:
        np.ndarray: Combined RGB image of shape defined by image_shape.
    """
    target_w_each = image_shape[1] // 3
    target_size = (image_shape[0], target_w_each)
    images = []
    for perspective in ['A', 'B', 'C']:
        img = load_perspective_image(test_id, perspective, labels_dir, target_size)
        if img is None:
            img = np.zeros((target_size[0], target_w_each, 3), dtype=np.uint8)
        images.append(img)
    combined_image = np.concatenate(images, axis=1)
    return combined_image

def compute_binary_mask(combined_image):
    """
    Computes a binary mask by detecting red regions in the combined image using HSV thresholding.
    
    Returns:
        np.ndarray: Binary mask where 1 indicates damage (red regions) and 0 otherwise.
    """
    hsv = cv2.cvtColor(combined_image, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    binary_mask = (red_mask > 0).astype(np.uint8)
    return binary_mask

def smooth_mask(binary_mask, sigma=GAUSSIAN_SIGMA):
    """
    Applies Gaussian smoothing to the binary mask.
    
    Returns:
        np.ndarray: Smoothed mask normalized to [0, 1].
    """
    smoothed = gaussian_filter(binary_mask.astype(np.float32), sigma=sigma)
    if smoothed.max() > 0:
        smoothed = smoothed / smoothed.max()
    return smoothed

def load_labels_for_test(test_id, labels_dir=LABELS_DIR, image_shape=IMAGE_SHAPE, sigma=GAUSSIAN_SIGMA):
    """
    Loads and processes label images for a given test.
    
    Steps:
      1. Load and combine perspective images.
      2. Compute a binary damage mask via HSV thresholding.
      3. Apply Gaussian smoothing to the binary mask.
    
    Returns:
        np.ndarray: Preprocessed (smoothed) label mask.
    """
    combined_image = load_combined_label(test_id, labels_dir, image_shape)
    binary_mask = compute_binary_mask(combined_image)
    smoothed_mask = smooth_mask(binary_mask, sigma)
    return smoothed_mask

############################################################################
#Load data function -> lets us access the data simply with .load_data()
############################################################################
def load_data():
    accel_dict = load_accelerometer_data(DATA_DIR, SKIP_TESTS)
    mask_dict = {}
    for test_id in accel_dict.keys():
        mask = load_labels_for_test(test_id, LABELS_DIR, IMAGE_SHAPE, GAUSSIAN_SIGMA)
        mask_dict[test_id] = mask
    return accel_dict, mask_dict


######################################
# Main Function
######################################
def main():
    print("INFO: Loading accelerometer data...")
    accel_data = load_accelerometer_data(DATA_DIR, SKIP_TESTS)
    
    labels = {}
    for test_id in accel_data.keys():
        print(f"INFO: Processing labels for Test {test_id}...")
        label_mask = load_labels_for_test(test_id, LABELS_DIR, IMAGE_SHAPE, GAUSSIAN_SIGMA)
        labels[test_id] = label_mask
        print(f"INFO: Label for Test {test_id} has shape: {label_mask.shape}")
    
    print("INFO: Data loading complete.")

if __name__ == "__main__":
    main()
