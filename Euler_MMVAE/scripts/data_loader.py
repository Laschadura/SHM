import os
import glob
import re
import cv2
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

                if data_matrix.shape[0] > EXPECTED_LENGTH:
                    data_matrix = data_matrix[:EXPECTED_LENGTH, :]

                samples.append(data_matrix)
            except Exception:
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