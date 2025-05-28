from .config import DATA_DIR, LABELS_DIR, SKIP_TESTS, EXPECTED_LENGTH, PERSPECTIVE_MAP, IMAGE_SHAPE
import glob, os, re, cv2, numpy as np, pandas as pd   

# Load accelerometer data
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

# Load Images and combine perspectives into Labels
def load_perspective_image(test_id, perspective, labels_dir=LABELS_DIR, target_size=(256,256)):
    label_name = PERSPECTIVE_MAP.get(perspective)
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

#caching logic
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
