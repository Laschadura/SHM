import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import splprep, splev  # <-- For spline fitting

######################################
# Configuration
######################################
DATA_DIR = "Data"          # Directory containing Test_* folders with CSV files
LABELS_DIR = "Labels"      # Directory containing label images for each test
IMAGE_SHAPE = (256, 768)   # Output combined image shape (height, width)
SKIP_TESTS = [23, 24]      # List of test numbers to skip (if any)
EXPECTED_LENGTH = 12000    # Expected number of datapoints (60s at 200Hz)

# Mapping for perspective images
perspective_map = {
    'A': 'Arch_Intrados',
    'B': 'North_Spandrel_Wall',
    'C': 'South_Spandrel_Wall'
}

######################################
# Signal Processing Functions
######################################
def high_pass_filter_data(data, cutoff=10.0, fs=200.0, order=5):
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
    medians = np.median(data, axis=0, keepdims=True)
    q75 = np.percentile(data, 75, axis=0, keepdims=True)
    q25 = np.percentile(data, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    normalized = (data - medians) / iqr
    return normalized

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

                # Truncate if longer than EXPECTED_LENGTH
                if data_matrix.shape[0] > EXPECTED_LENGTH:
                    data_matrix = data_matrix[:EXPECTED_LENGTH, :]

                filtered_data = high_pass_filter_data(data_matrix, cutoff=10.0, fs=200.0, order=5)
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
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=[0,0,0]
    )
    return img_padded

def load_combined_label(test_id, labels_dir=LABELS_DIR, image_shape=IMAGE_SHAPE):
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

######################################
# 1) Skeleton
######################################
def skeletonize_mask(mask):
    mask_bin = (mask > 0).astype(np.uint8)
    skel = np.zeros_like(mask_bin)
    temp = np.zeros_like(mask_bin)
    eroded = np.zeros_like(mask_bin)

    done = False
    while not done:
        cv2.erode(mask_bin, None, eroded)
        cv2.dilate(eroded, None, temp)
        temp = cv2.subtract(mask_bin, temp)
        skel = cv2.bitwise_or(skel, temp)
        mask_bin, eroded = eroded, mask_bin
        done = (cv2.countNonZero(mask_bin) == 0)
    return skel

def skeleton_to_components(skeleton):
    n_labels, labels = cv2.connectedComponents(skeleton.astype(np.uint8))
    components = []
    for label_id in range(1, n_labels):
        coords = np.column_stack(np.where(labels == label_id))  # shape (N, 2) -> (row, col)
        components.append(coords)
    return components

######################################
# 2) Order Skeleton Points
######################################
def find_endpoints_and_order(points):
    """
    Attempts to order the points along a single path from one endpoint to the other.
    **Requires** that the component is effectively a single chain (no branching).
    Returns a list of (row, col) in path order.
    If branching or multiple endpoints exist, we only handle the largest chain or
    do a BFS from an endpoint to the other.
    """
    if len(points) <= 2:
        return points  # trivial case

    # Convert point set to a dict for fast lookup
    point_set = set(tuple(p) for p in points)
    
    # Build adjacency (using 8-neighborhood)
    adjacency = {}
    for (r, c) in point_set:
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in point_set:
                    neighbors.append((nr, nc))
        adjacency[(r, c)] = neighbors
    
    # Find endpoints: nodes with exactly 1 neighbor
    endpoints = [pt for pt, nbrs in adjacency.items() if len(nbrs) == 1]
    
    if len(endpoints) < 2:
        # Could be a loop or a small artifact. We'll just pick the first point as start
        endpoints = [points[0], points[-1]]
    
    start = endpoints[0]
    visited = set()
    ordered = []

    # BFS or DFS from start
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        ordered.append(cur)
        for nxt in adjacency[cur]:
            if nxt not in visited:
                stack.append(nxt)

    return np.array(ordered)

######################################
# 3) Fit Spline
######################################
def fit_spline_to_path(path_points, smoothing=0):
    """
    path_points: array of shape (N, 2) with [row, col] in order.
    We'll treat these as x=col, y=row for normal Cartesian usage.
    smoothing: a float controlling spline smoothing (s=0 is exact interpolation).
    Returns (tck, u), which are the B-spline representation and parameter array.
    """
    if len(path_points) < 3:
        return None, None  # can't really fit a spline to fewer than 3 points
    
    xs = path_points[:,1].astype(float)  # col
    ys = path_points[:,0].astype(float)  # row
    
    # Normalize or scale if you want. We'll just do them as is.
    # 'u' is the parameter along the curve (0..1).
    # 'tck' are the knots + coefficients describing the B-spline
    tck, u = splprep([xs, ys], s=smoothing, k=2) 
    return tck, u

def sample_spline(tck, num=50):
    """
    Sample 'num' points from a fitted B-spline 'tck'.
    Returns array of shape (num, 2): columns first, then rows.
    """
    # Evaluate for param values from 0..1 in 'num' steps
    if tck is None:
        return np.empty((0,2))
    u_eval = np.linspace(0, 1, num)
    x_s, y_s = splev(u_eval, tck)
    # x_s is col, y_s is row
    return np.column_stack([y_s, x_s])  # (row, col)

######################################
# 4) Master function: from mask -> piecewise-spline
######################################
def mask_to_piecewise_splines(mask):
    """
    1) Skeletonize
    2) Find connected components (each presumably a single crack or branch)
    3) For each component, order points
    4) Fit a B-spline
    5) Return a list of { 'tck':..., 'num_points':..., 'samples':... } 
       for each connected component that has enough points
    """
    skeleton = skeletonize_mask(mask)
    components = skeleton_to_components(skeleton)
    
    splines = []
    for comp in components:
        # comp is array of shape (N,2) [row, col]
        if len(comp) < 3:
            continue
        
        ordered_pts = find_endpoints_and_order(comp)
        tck, u = fit_spline_to_path(ordered_pts, smoothing=0)  # or a bit of smoothing
        if tck is not None:
            # we can store the tck or store a sampled version
            sampled = sample_spline(tck, num=50)
            splines.append({
                'tck': tck,
                'count': len(ordered_pts),
                'samples': sampled
            })
    return splines

######################################
# Use the piecewise-spline approach in load_labels_for_test
######################################
def load_labels_for_test(test_id, labels_dir=LABELS_DIR, image_shape=IMAGE_SHAPE):
    """
    Loads label images for a given test, obtains a binary mask, and
    returns a list of B-spline structures for each crack component.
    """
    combined_image = load_combined_label(test_id, labels_dir, image_shape)
    binary_mask = compute_binary_mask(combined_image)

    # We'll do the piecewise-spline approach
    splines = mask_to_piecewise_splines(binary_mask)
    return splines  # list of dictionaries (one per connected crack), each has 'tck', 'samples', etc.

######################################
# Main load_data function
######################################
def load_data():
    """
    Returns:
      accel_dict: {test_id -> [list of NxM accelerometer arrays]}
      mask_dict:  {test_id -> list of piecewise-spline dicts for each crack}
    """
    accel_dict = load_accelerometer_data(DATA_DIR, SKIP_TESTS)
    mask_dict = {}

    for test_id in accel_dict.keys():
        # Fit piecewise splines for the cracks
        splines = load_labels_for_test(test_id, LABELS_DIR, IMAGE_SHAPE)
        mask_dict[test_id] = splines

    return accel_dict, mask_dict

######################################
# For testing
######################################
def main():
    print("INFO: Loading accelerometer data...")
    accel_data = load_accelerometer_data(DATA_DIR, SKIP_TESTS)
    
    # Test label -> splines
    for test_id in accel_data.keys():
        print(f"INFO: Processing labels for Test {test_id}...")
        splines = load_labels_for_test(test_id, LABELS_DIR, IMAGE_SHAPE)
        print(f"  Found {len(splines)} crack components with splines.")
        if splines:
            first_comp = splines[0]
            print(f"  Example: fitted {first_comp['count']} points, sampling => shape {first_comp['samples'].shape}")
    
    print("INFO: Data loading complete.")

if __name__ == "__main__":
    main()
