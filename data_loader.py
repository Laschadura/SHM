import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import splprep, splev

######################################
# Configuration
######################################
DATA_DIR = "Data"
LABELS_DIR = "Labels"
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
    # Your existing accelerometer loading code - unchanged
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
# Mask and skeleton computations
######################################
def compute_binary_mask(combined_image):
    # Your existing code - unchanged
    hsv = cv2.cvtColor(combined_image, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    return cv2.bitwise_or(mask1, mask2).astype(np.uint8)

def skeletonize_mask(mask):
    # Your existing code - unchanged
    skel = np.zeros_like(mask, dtype=np.uint8)
    while np.count_nonzero(mask) > 0:
        eroded = cv2.erode(mask, None)
        temp = cv2.dilate(eroded, None)
        temp = cv2.subtract(mask, temp)
        skel = cv2.bitwise_or(skel, temp)
        mask = eroded
    return skel

def skeleton_to_components(skeleton):
    """
    Uses connected components analysis to identify individual crack segments
    Each component (a group of connected pixels) represents a separate crack or crack segment
    These components are then ordered, split into smaller segments, and analyzed
    """
    num_labels, labels = cv2.connectedComponents(skeleton)
    return [np.column_stack(np.where(labels == label)) for label in range(1, num_labels)]

def enhance_skeleton(skeleton, max_gap=8, curving_preference=True):
    """
    Enhanced skeleton processing with better handling of curved segments.
    It identifies endpoints (pixels with only one neighbor)
    For nearby endpoints, it creates connections (straight lines or Bezier curves)
    This helps recover connections that might have been broken during processing
    
    Args:
        skeleton: Binary skeleton image
        max_gap: Maximum gap to connect (larger for curved cracks)
        curving_preference: Whether to prefer connections that maintain curve orientation
    """
    # Find endpoints in the skeleton
    kernel = np.array([[1, 1, 1],
                      [1, 10, 1],
                      [1, 1, 1]], dtype=np.uint8)
    
    # Convolve to find points with only one neighbor
    conv = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    endpoints = np.logical_and(conv == 11, skeleton > 0)
    endpoint_coords = np.argwhere(endpoints)
    
    if len(endpoint_coords) < 2:
        return skeleton
    
    # For each endpoint, find direction of the segment it belongs to
    directions = []
    for pt in endpoint_coords:
        # Sample a few points along the skeleton from this endpoint
        r, c = pt
        patch_size = 5
        patch = skeleton[max(0, r-patch_size):min(skeleton.shape[0], r+patch_size+1),
                        max(0, c-patch_size):min(skeleton.shape[1], c+patch_size+1)]
        
        # Find all non-zero points in the patch
        local_pts = np.argwhere(patch > 0)
        if len(local_pts) < 2:
            directions.append(np.array([0, 0]))
            continue
            
        # Convert to original coordinates
        local_pts[:, 0] += max(0, r-patch_size)
        local_pts[:, 1] += max(0, c-patch_size)
        
        # Exclude the endpoint itself
        local_pts = local_pts[~np.all(local_pts == pt, axis=1)]
        
        # Calculate direction vector - from furthest point to endpoint
        if len(local_pts) > 0:
            # Get the point furthest from the endpoint
            dists = np.linalg.norm(local_pts - pt, axis=1)
            furthest_idx = np.argmax(dists)
            furthest_pt = local_pts[furthest_idx]
            
            # Direction vector points from furthest point to endpoint
            direction = (pt - furthest_pt) / (dists[furthest_idx] + 1e-8)
            directions.append(direction)
        else:
            directions.append(np.array([0, 0]))
    
    # Connect endpoints, preferring those that maintain curvature
    enhanced = skeleton.copy()
    connections_made = 0
    
    # Sort endpoint pairs by distance
    endpoint_pairs = []
    for i in range(len(endpoint_coords)):
        for j in range(i+1, len(endpoint_coords)):
            pt1 = endpoint_coords[i]
            pt2 = endpoint_coords[j]
            dist = np.linalg.norm(pt1 - pt2)
            
            if dist < max_gap:
                # Check if directions are compatible (for continuing curves)
                dir1 = directions[i]
                dir2 = directions[j]
                
                # Direction score - higher for compatible directions
                # We want endpoints with directions pointing toward each other
                direction_score = 0
                if np.linalg.norm(dir1) > 0 and np.linalg.norm(dir2) > 0:
                    # Connection vector from pt1 to pt2
                    connection_vec = (pt2 - pt1) / dist
                    
                    # Check alignment of dir1 with connection
                    # and dir2 with reverse connection
                    align1 = np.dot(dir1, connection_vec)
                    align2 = np.dot(dir2, -connection_vec)
                    
                    # Higher score for better alignment
                    direction_score = (align1 + align2) / 2
                
                # Final score combines distance and direction
                if curving_preference:
                    score = dist - (direction_score * max_gap * 0.5)
                else:
                    score = dist
                    
                endpoint_pairs.append((i, j, dist, score))
    
    # Sort by score (lower is better)
    endpoint_pairs.sort(key=lambda x: x[3])
    
    # Keep track of which endpoints have been connected
    connected = set()
    
    # Connect endpoints
    for i, j, dist, score in endpoint_pairs:
        # Skip if either endpoint already connected
        if i in connected or j in connected:
            continue
            
        pt1 = endpoint_coords[i]
        pt2 = endpoint_coords[j]
        
        # Draw curve between endpoints
        # For short distances, a straight line is fine
        if dist < 4:
            cv2.line(enhanced, (pt1[1], pt1[0]), (pt2[1], pt2[0]), 255, 1)
        else:
            # For longer distances, use a quadratic Bezier curve 
            # to better maintain the curvature
            dir1 = directions[i]
            dir2 = directions[j]
            
            # Use directions to create control point
            if np.linalg.norm(dir1) > 0 and np.linalg.norm(dir2) > 0:
                # Control point is at midpoint, offset by combined directions
                mid = (pt1 + pt2) / 2
                offset = (dir1 + dir2) * dist * 0.25
                control = mid + offset
                
                # Create points along Bezier curve
                t_values = np.linspace(0, 1, int(dist * 2))
                curve_pts = []
                
                for t in t_values:
                    # Quadratic Bezier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
                    b_pt = (1-t)**2 * pt1 + 2*(1-t)*t * control + t**2 * pt2
                    curve_pts.append(b_pt.astype(int))
                
                # Draw the curve
                for k in range(len(curve_pts)-1):
                    cv2.line(enhanced, 
                            (curve_pts[k][1], curve_pts[k][0]), 
                            (curve_pts[k+1][1], curve_pts[k+1][0]), 
                            255, 1)
            else:
                # Fall back to straight line if directions unavailable
                cv2.line(enhanced, (pt1[1], pt1[0]), (pt2[1], pt2[0]), 255, 1)
        
        # Mark these endpoints as connected
        connected.add(i)
        connected.add(j)
        connections_made += 1
    
    print(f"Enhanced skeleton with {connections_made} connections")
    return enhanced

######################################
# Improved Crack Representation
######################################
def order_points_along_path(points):
    """Order points along a curve/path using minimum spanning tree approach."""
    if len(points) <= 1:
        return points
    
    # Compute pairwise distances
    dist_matrix = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    # Initialize with first point
    ordered_indices = [0]
    unvisited = set(range(1, len(points)))
    
    # Find path using nearest neighbor approach
    while unvisited:
        last_idx = ordered_indices[-1]
        # Get distances from current point to all unvisited points
        distances = [(dist_matrix[last_idx, j], j) for j in unvisited]
        # Find closest unvisited point
        _, next_idx = min(distances)
        ordered_indices.append(next_idx)
        unvisited.remove(next_idx)
    
    return points[ordered_indices]

def split_crack_segments(ordered_pts, min_segment_length=2, 
                         curved_threshold=10, 
                         curved_angle_threshold=85,
                         straight_angle_threshold=20):
    """
    Improved split function with better handling of ladder-like structures.
    Uses a smaller minimum segment length to capture short segments.
    
    Args:
        ordered_pts: Points along the crack path
        min_segment_length: Minimum segment length to keep
        curved_threshold: Threshold to identify curved regions (degrees)
        curved_angle_threshold: Angle threshold for curved regions (degrees)
        straight_angle_threshold: Angle threshold for straight regions (degrees)
    """
    if len(ordered_pts) < 3:
        return [ordered_pts]

    # Calculate local curvature
    curvatures = []
    for i in range(1, len(ordered_pts)-1):
        v1 = ordered_pts[i] - ordered_pts[i-1]
        v2 = ordered_pts[i+1] - ordered_pts[i]
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-8 or v2_norm < 1e-8:
            curvatures.append(0)
            continue
            
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))
        curvatures.append(angle)
    
    # Pad with zeros for the endpoints
    curvatures = [0] + curvatures + [0]
    
    # Compute a rolling average to detect curved regions
    window_size = min(5, len(curvatures)-2)
    if window_size > 0:
        smoothed_curvatures = np.convolve(
            curvatures, np.ones(window_size)/window_size, mode='valid')
        pad_size = len(curvatures) - len(smoothed_curvatures)
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        smoothed_curvatures = np.pad(smoothed_curvatures, (pad_left, pad_right), 'edge')
    else:
        smoothed_curvatures = np.array(curvatures)
    
    # A point is in a curved region if curvature exceeds threshold
    is_curved_region = smoothed_curvatures > curved_threshold
    
    # Define split points
    split_indices = [0]  # Start with the first point
    
    # Look for sharp turns
    for i in range(2, len(ordered_pts)):
        v1 = ordered_pts[i-1] - ordered_pts[i-2]
        v2 = ordered_pts[i] - ordered_pts[i-1]
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-8 or v2_norm < 1e-8:
            continue
            
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))
        
        # Use different angle thresholds based on region type
        if is_curved_region[i]:
            angle_threshold = curved_angle_threshold  # For curved regions (arches)
        else:
            angle_threshold = straight_angle_threshold  # For straight regions (ladders)
            
        if angle > angle_threshold:
            # Check if this is part of a ladder structure - consecutive sharp turns
            # We define ladder as having multiple sharp turns in close proximity
            is_ladder = False
            
            # Look for another turn nearby
            for j in range(max(2, i-5), min(len(ordered_pts), i+5)):
                if j == i:
                    continue
                    
                if j-1 < 1 or j+1 >= len(ordered_pts):
                    continue
                    
                v1_j = ordered_pts[j-1] - ordered_pts[j-2]
                v2_j = ordered_pts[j] - ordered_pts[j-1]
                
                v1j_norm = np.linalg.norm(v1_j)
                v2j_norm = np.linalg.norm(v2_j)
                
                if v1j_norm < 1e-8 or v2j_norm < 1e-8:
                    continue
                    
                v1_j = v1_j / v1j_norm
                v2_j = v2_j / v2j_norm
                
                dot_j = np.clip(np.dot(v1_j, v2_j), -1.0, 1.0)
                angle_j = np.degrees(np.arccos(dot_j))
                
                if angle_j > angle_threshold:
                    is_ladder = True
                    break
            
            # If this is a ladder, we can use an even smaller minimum length
            effective_min_length = 2 if is_ladder else min_segment_length
                
            # Add split point if we meet minimum length requirement
            if len(split_indices) == 0 or (i - split_indices[-1]) >= effective_min_length:
                split_indices.append(i)
    
    split_indices.append(len(ordered_pts))  # End with the last point
    
    # Create segments based on split points
    segments = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        
        segment_length = end_idx - start_idx
        if segment_length >= min_segment_length:
            segments.append(ordered_pts[start_idx:end_idx])
    
    return segments

def compute_keypoints(ordered_pts, num_keypoints=18):
    """
    Improved keypoint sampling with better handling of curves and corners.
    Increased keypoint density for better representation.
    """
    if len(ordered_pts) < 2:
        return ordered_pts
    
    # If very few points, return them directly
    if len(ordered_pts) <= num_keypoints:
        return ordered_pts
    
    # Calculate local curvature at each point
    curvatures = [0]  # First point has no curvature
    for i in range(1, len(ordered_pts)-1):
        v1 = ordered_pts[i] - ordered_pts[i-1]
        v2 = ordered_pts[i+1] - ordered_pts[i]
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-8 or v2_norm < 1e-8:
            curvatures.append(0)
            continue
            
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(dot)
        curvatures.append(angle)
    
    curvatures.append(0)  # Last point has no curvature
    curvatures = np.array(curvatures)
    
    # Find corners (points with high curvature)
    corner_threshold = 0.5  # radians (~30 degrees)
    corners = curvatures > corner_threshold
    
    # Ensure we always include endpoints
    corners[0] = True
    corners[-1] = True
    
    # Always include corner points in keypoints
    corner_indices = np.where(corners)[0]
    
    # Adjust curvature weighting - higher emphasis on curved regions
    curvature_weight = 1 + 5 * (curvatures / (np.max(curvatures) + 1e-8))
    
    # Calculate cumulative distance along path with curvature weighting
    cum_dist = [0]
    for i in range(1, len(ordered_pts)):
        pt1 = ordered_pts[i-1]
        pt2 = ordered_pts[i]
        dist = np.linalg.norm(pt2 - pt1)
        # Weight by curvature - more "distance" in curved regions
        weighted_dist = dist * curvature_weight[i]
        cum_dist.append(cum_dist[-1] + weighted_dist)
    
    total_dist = cum_dist[-1]
    
    # Determine how many additional keypoints to place between corners
    remaining_keypoints = num_keypoints - len(corner_indices)
    
    if remaining_keypoints <= 0:
        # If we have more corners than allowed keypoints, use the most significant ones
        # Sort corners by curvature and keep the top num_keypoints
        corner_curvatures = curvatures[corner_indices]
        sorted_indices = np.argsort(corner_curvatures)[::-1]  # Descending order
        
        # Always keep endpoints (first and last)
        endpoints = np.array([0, len(ordered_pts)-1])
        # Filter out endpoints from sorted corners
        interior_corners = sorted_indices[~np.isin(corner_indices[sorted_indices], endpoints)]
        
        # Take top N-2 interior corners + 2 endpoints
        top_corners = np.concatenate([
            endpoints, 
            corner_indices[interior_corners[:num_keypoints-2]] if len(interior_corners) > 0 else []
        ])
        
        # Sort by position along path
        top_corners.sort()
        
        # Return the keypoints at these indices
        return ordered_pts[top_corners]
    
    # Generate remaining keypoints with adaptive spacing
    keypoints = []
    
    # Add all corner points first
    for idx in corner_indices:
        keypoints.append(ordered_pts[idx])
    
    # Sample additional points with even spacing in weighted distance
    for i in range(remaining_keypoints):
        target_dist = (i + 1) * total_dist / (remaining_keypoints + 1)
        
        # Find closest point or interpolate
        idx = np.searchsorted(cum_dist, target_dist)
        
        if idx == 0:
            pt = ordered_pts[0]
        elif idx >= len(ordered_pts):
            pt = ordered_pts[-1]
        else:
            # Linear interpolation
            prev_idx = idx - 1
            prev_dist = cum_dist[prev_idx]
            curr_dist = cum_dist[idx]
            
            if curr_dist == prev_dist:
                t = 0
            else:
                t = (target_dist - prev_dist) / (curr_dist - prev_dist)
                
            pt = (1-t) * ordered_pts[prev_idx] + t * ordered_pts[idx]
        
        # Check if this point is too close to an existing corner point
        min_dist_to_corner = float('inf')
        for corner_pt in [ordered_pts[i] for i in corner_indices]:
            dist = np.linalg.norm(pt - corner_pt)
            min_dist_to_corner = min(min_dist_to_corner, dist)
        
        # Only add if not too close to a corner
        if min_dist_to_corner > 1.0:
            keypoints.append(pt)
    
    # Make sure we have endpoints
    if np.linalg.norm(keypoints[0] - ordered_pts[0]) > 1e-8:
        keypoints.insert(0, ordered_pts[0])
    if np.linalg.norm(keypoints[-1] - ordered_pts[-1]) > 1e-8:
        keypoints.append(ordered_pts[-1])
    
    # Sort keypoints by position along the path
    # Project each keypoint onto the path and get its parameter value
    path_params = []
    for kp in keypoints:
        min_dist = float('inf')
        min_idx = 0
        
        # Find closest point on the path
        for i in range(len(ordered_pts)):
            dist = np.linalg.norm(kp - ordered_pts[i])
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        # Parameter is the index normalized to [0, 1]
        path_params.append(min_idx / (len(ordered_pts) - 1))
    
    # Sort keypoints by path parameter
    sorted_indices = np.argsort(path_params)
    sorted_keypoints = [keypoints[i] for i in sorted_indices]
    
    # Limit to requested number of keypoints
    if len(sorted_keypoints) > num_keypoints:
        # Keep endpoints and select interior points evenly
        interior_points = sorted_keypoints[1:-1]
        step = len(interior_points) / (num_keypoints - 2)
        
        selected_indices = [0]  # First point
        for i in range(num_keypoints - 2):
            idx = int(i * step)
            idx = min(idx, len(interior_points) - 1)  # Safety check
            selected_indices.append(idx + 1)  # +1 because we skipped first point
        selected_indices.append(len(sorted_keypoints) - 1)  # Last point
        
        sorted_keypoints = [sorted_keypoints[i] for i in selected_indices]
    
    return np.array(sorted_keypoints)

def compute_crack_descriptors(ordered_pts, keypoint_count=20, is_new=0):
    """
    Compute descriptors with improved orientation calculation
    """
    if len(ordered_pts) < 2:
        return None
    
    # Basic descriptor components
    row_start, col_start = ordered_pts[0]
    row_end, col_end = ordered_pts[-1]
    path_length = np.sum(np.linalg.norm(np.diff(ordered_pts, axis=0), axis=1))
    
    # IMPROVED: Calculate both endpoint-based orientation and local orientations at start/end
    overall_orientation = np.arctan2(row_end - row_start, col_end - col_start)
    
    # Calculate local orientation at start (using first few points)
    start_window = min(5, len(ordered_pts) - 1)
    dy_start = ordered_pts[start_window][0] - ordered_pts[0][0]
    dx_start = ordered_pts[start_window][1] - ordered_pts[0][1]
    start_orientation = np.arctan2(dy_start, dx_start)
    
    # Calculate local orientation at end (using last few points)
    end_window = max(0, len(ordered_pts) - 6)
    dy_end = ordered_pts[-1][0] - ordered_pts[end_window][0]
    dx_end = ordered_pts[-1][1] - ordered_pts[end_window][1]
    end_orientation = np.arctan2(dy_end, dx_end)
    
    # Calculate orientation difference between start and end
    orientation_diff = abs(start_orientation - end_orientation)
    orientation_diff = min(orientation_diff, 2*np.pi - orientation_diff)  # Handle circular wrapping
    
    # Calculate curvature as before
    curvature_vals = []
    for i in range(1, len(ordered_pts) - 1):
        v1 = ordered_pts[i] - ordered_pts[i - 1]
        v2 = ordered_pts[i + 1] - ordered_pts[i]
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-8 or v2_norm < 1e-8:
            continue
            
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        curvature_vals.append(np.arccos(dot))
    
    avg_curvature = np.mean(curvature_vals) if curvature_vals else 0.0
    
    # Extract keypoints along the path with improved sampling
    keypoints = compute_keypoints(ordered_pts, keypoint_count)
    
    # Create base descriptor with orientation difference and local orientations
    base_descriptor = np.array([
        row_start, col_start, row_end, col_end, 
        path_length, overall_orientation, avg_curvature, 
        len(ordered_pts), is_new,  # Original 9 elements
        start_orientation, end_orientation, orientation_diff  # New orientation metrics
    ])
    
    # Flatten keypoints into a 1D array
    keypoints_flat = keypoints.flatten()
    
    # Combine base descriptor with keypoints
    full_descriptor = np.concatenate([base_descriptor, keypoints_flat])
    
    return full_descriptor

def create_rasterized_mask(descriptors, image_shape):
    """
    Create a rasterized mask from crack descriptors.
    This is used to identify pixels that belong to cracks in a given test.
    
    Args:
        descriptors: List of crack descriptors
        image_shape: Shape of the output mask
        
    Returns:
        A binary mask where 1 indicates crack pixels
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    if not descriptors:
        return mask
    
    for desc in descriptors:
        if len(desc) < 10:  # Need at least one keypoint after base descriptor
            continue
            
        # Extract keypoints
        keypoints_flat = desc[9:]
        
        # Handle case where keypoints_flat has odd length
        if len(keypoints_flat) % 2 != 0:
            keypoints_flat = np.append(keypoints_flat, keypoints_flat[-1])
        
        num_keypoints = len(keypoints_flat) // 2
        keypoints = keypoints_flat.reshape(num_keypoints, 2)
        
        # Draw lines between consecutive keypoints
        for i in range(len(keypoints) - 1):
            pt1 = keypoints[i].astype(int)
            pt2 = keypoints[i + 1].astype(int)
            
            if (0 <= pt1[0] < image_shape[0] and 0 <= pt1[1] < image_shape[1] and
                0 <= pt2[0] < image_shape[0] and 0 <= pt2[1] < image_shape[1]):
                cv2.line(mask, (pt1[1], pt1[0]), (pt2[1], pt2[0]), 1, 1)
    
    return mask

def is_similar_crack(desc1, desc2, position_threshold=12, angle_threshold=0.5):
    """
    Check if two crack descriptors represent the same crack.
    Now handles extended descriptors with local orientations.
    """
    # Safety check for descriptor format - ensure we have at least the basic descriptor
    base_desc_length = 12  # Updated to include the new orientation elements
    if len(desc1) < base_desc_length or len(desc2) < base_desc_length:
        # Fallback to original format if the new fields aren't available
        return is_similar_crack_original(desc1, desc2, position_threshold, angle_threshold)
    
    # Extract base descriptors
    base1 = desc1[:base_desc_length]
    base2 = desc2[:base_desc_length]
    
    # Extract endpoints
    start1 = np.array([base1[0], base1[1]])
    end1 = np.array([base1[2], base1[3]])
    start2 = np.array([base2[0], base2[1]])
    end2 = np.array([base2[2], base2[3]])
    
    # Check distances between endpoints
    dist_start_start = np.linalg.norm(start1 - start2)
    dist_start_end = np.linalg.norm(start1 - end2)
    dist_end_start = np.linalg.norm(end1 - start2)
    dist_end_end = np.linalg.norm(end1 - end2)
    
    min_dist = min(dist_start_start, dist_start_end, dist_end_start, dist_end_end)
    
    # Compare local orientations and orientation differences
    # This helps with curved cracks like half-circles
    orientation_diff1 = base1[11]  # orientation_diff from first descriptor
    orientation_diff2 = base2[11]  # orientation_diff from second descriptor
    
    # Similar curvature patterns should have similar orientation differences
    curve_pattern_match = abs(orientation_diff1 - orientation_diff2) < angle_threshold * 2
    
    # For curved segments, also check if keypoints overlap
    curve_overlap = check_keypoint_overlap(desc1, desc2, position_threshold)
    
    # Match criteria: endpoints are close OR keypoints overlap, AND curve patterns match
    return ((min_dist < position_threshold or curve_overlap) and curve_pattern_match)

def is_similar_crack_original(desc1, desc2, position_threshold=12, angle_threshold=0.5):
    """
    Improved crack comparison with better handling of curved segments.
    Considers both endpoints and midpoints for matching.
    
    Fixed to handle different descriptor lengths safely.
    """
    # Safety check for descriptor format - ensure we have at least the base descriptor
    if len(desc1) < 9 or len(desc2) < 9:
        return False
    
    # Extract base descriptors (first 9 elements)
    base1 = desc1[:9]
    base2 = desc2[:9]
    
    # Extract endpoints
    start1 = np.array([base1[0], base1[1]])
    end1 = np.array([base1[2], base1[3]])
    start2 = np.array([base2[0], base2[1]])
    end2 = np.array([base2[2], base2[3]])
    
    # Check distances between endpoints
    dist_start_start = np.linalg.norm(start1 - start2)
    dist_start_end = np.linalg.norm(start1 - end2)
    dist_end_start = np.linalg.norm(end1 - start2)
    dist_end_end = np.linalg.norm(end1 - end2)
    
    min_dist = min(dist_start_start, dist_start_end, dist_end_start, dist_end_end)
    
    # For curved segments, also check if keypoints overlap
    curve_overlap = False
    
    # Make sure we have keypoints and they are of even length (pairs of coordinates)
    if (len(desc1) > 9 and len(desc2) > 9 and 
        (len(desc1) - 9) % 2 == 0 and (len(desc2) - 9) % 2 == 0):
        
        # Reshape safely with the correct number of points
        keypoints1 = desc1[9:].reshape(-1, 2)
        keypoints2 = desc2[9:].reshape(-1, 2)
        
        # Check middle keypoint distance (helps with curved segments)
        if len(keypoints1) > 0 and len(keypoints2) > 0:
            mid1 = keypoints1[len(keypoints1)//2]
            
            # Find closest point in the other crack
            min_mid_dist = float('inf')
            for kp2 in keypoints2:
                dist = np.linalg.norm(mid1 - kp2)
                min_mid_dist = min(min_mid_dist, dist)
            
            # If middle points are close, consider it a match
            if min_mid_dist < position_threshold:
                curve_overlap = True
    
    # Check if orientations are similar
    angle_match = True
    if min_dist >= position_threshold and not curve_overlap:
        orientation1, orientation2 = base1[5], base2[5]
        angle_diff = abs(orientation1 - orientation2)
        angle_diff = min(angle_diff, np.pi - angle_diff)  # Handle circular wrapping
        
        angle_match = angle_diff < angle_threshold
    
    # Consider it a match if endpoints are close OR we have curve overlap
    return (min_dist < position_threshold or curve_overlap) and angle_match

def check_keypoint_overlap(desc1, desc2, position_threshold):
    """Helper function to check if keypoints of two cracks overlap"""
    # Make sure we have keypoints and they are of even length (pairs of coordinates)
    if (len(desc1) > 12 and len(desc2) > 12 and 
        (len(desc1) - 12) % 2 == 0 and (len(desc2) - 12) % 2 == 0):
        
        # Reshape safely with the correct number of points
        keypoints1 = desc1[12:].reshape(-1, 2)
        keypoints2 = desc2[12:].reshape(-1, 2)
        
        # Check middle keypoint distance (helps with curved segments)
        if len(keypoints1) > 0 and len(keypoints2) > 0:
            mid1 = keypoints1[len(keypoints1)//2]
            
            # Find closest point in the other crack
            min_mid_dist = float('inf')
            for kp2 in keypoints2:
                dist = np.linalg.norm(mid1 - kp2)
                min_mid_dist = min(min_mid_dist, dist)
            
            # If middle points are close, consider it a match
            if min_mid_dist < position_threshold:
                return True
    
    return False

def process_test_cracks(mask, prev_test_skeleton=None, keypoint_count=20, max_gap=10):
    """
    Process cracks with improved new crack identification using skeleton subtraction.
    
    Args:
        mask: Binary mask of cracks
        prev_test_skeleton: Skeleton from previous test (None for first test)
        keypoint_count: Number of keypoints for each crack
        max_gap: Maximum gap to connect in skeleton enhancement
        
    Returns:
        descriptors: List of crack descriptors
    """
    # Create skeleton from mask
    skeleton = skeletonize_mask(mask)
    
    # Enhanced skeleton - try to connect broken segments
    enhanced_skeleton = enhance_skeleton(skeleton, max_gap=max_gap, curving_preference=True)
    
    # Identify new vs. continued cracks using skeleton subtraction
    new_descriptors, continued_descriptors = identify_new_cracks(
        enhanced_skeleton, prev_test_skeleton, keypoint_count)
    
    # Combine all descriptors
    all_descriptors = new_descriptors + continued_descriptors
    
    return all_descriptors, enhanced_skeleton  # Return skeleton for next test

def process_test_cracks_original(mask, prev_test_descriptors=None, keypoint_count=20, max_gap=10, 
                       curved_threshold=10, curved_angle_threshold=85, 
                       straight_angle_threshold=20, min_segment_length=2):
    """
    Process cracks without extension tracking.
    
    Args:
        mask: Binary mask of cracks
        prev_test_descriptors: Descriptors from previous test 
        keypoint_count: Number of keypoints for each crack
        max_gap: Maximum gap to connect in skeleton enhancement
        curved_threshold: Threshold to identify curved regions
        curved_angle_threshold: Angle threshold for curved regions
        straight_angle_threshold: Angle threshold for straight regions
        min_segment_length: Minimum length of crack segments to keep
        
    Returns:
        descriptors: List of crack descriptors
    """
    # Create skeleton from mask
    skeleton = skeletonize_mask(mask)
    
    # Enhanced skeleton - try to connect broken segments
    enhanced_skeleton = enhance_skeleton(
        skeleton, 
        max_gap=max_gap, 
        curving_preference=True
    )
    
    # Extract connected components
    components = skeleton_to_components(enhanced_skeleton)
    
    # Store results
    crack_descriptors = []
    
    # Process each component
    for component in components:
        if len(component) < 3:
            continue
            
        # Order points along the curve
        ordered_pts = order_points_along_path(component)
        
        # Split into segments with improved handling
        segments = split_crack_segments(
            ordered_pts, 
            min_segment_length=min_segment_length,
            curved_threshold=curved_threshold,
            curved_angle_threshold=curved_angle_threshold,
            straight_angle_threshold=straight_angle_threshold
        )
        
        for segment in segments:
            if len(segment) < 3:
                continue
                
            # Default to new crack
            is_new = 1
            
            # If not a previous extension, check against regular existing cracks
            if prev_test_descriptors is not None and len(prev_test_descriptors) > 0:
                # Check for similarity with existing cracks
                temp_desc = compute_crack_descriptors(segment, keypoint_count, is_new=0)
                
                if temp_desc is None:
                    continue
                    
                # Check if it's an existing crack
                for prev_desc in prev_test_descriptors:
                    if is_similar_crack(temp_desc, prev_desc, position_threshold=12):
                        is_new = 0
                        break
            
            # Compute final descriptor
            descriptor = compute_crack_descriptors(segment, keypoint_count, is_new)
            
            if descriptor is not None:
                crack_descriptors.append(descriptor)
    
    return crack_descriptors

def identify_new_cracks(current_skeleton, previous_skeleton, keypoint_count=20):
    """
    Identify new cracks by subtracting previous skeleton from current.
    Handles both entirely new cracks and extensions of existing cracks.
    
    Args:
        current_skeleton: Binary skeleton image from current test
        previous_skeleton: Binary skeleton image from previous test
        keypoint_count: Number of keypoints for descriptor generation
        
    Returns:
        List of descriptors for new cracks, List of descriptors for continued cracks
    """
    if previous_skeleton is None:
        # If no previous skeleton, all cracks are new
        components = skeleton_to_components(current_skeleton)
        new_crack_descriptors = []
        
        for component in components:
            if len(component) < 3:
                continue
                
            ordered_pts = order_points_along_path(component)
            segments = split_crack_segments(ordered_pts)
            
            for segment in segments:
                if len(segment) < 3:
                    continue
                    
                descriptor = compute_crack_descriptors(segment, keypoint_count, is_new=1)
                if descriptor is not None:
                    new_crack_descriptors.append(descriptor)
                    
        return new_crack_descriptors, []
    
    # Create a dilated version of previous skeleton to better match pixel locations
    dilated_previous = cv2.dilate(previous_skeleton, np.ones((3,3), np.uint8))
    
    # 1. Find completely new cracks (not present in previous test)
    new_regions = np.logical_and(current_skeleton > 0, dilated_previous == 0).astype(np.uint8)
    
    # 2. Find continued/existing cracks (present in both tests)
    continued_regions = np.logical_and(current_skeleton > 0, dilated_previous > 0).astype(np.uint8)
    
    # Process new crack regions
    new_components = skeleton_to_components(new_regions)
    new_crack_descriptors = []
    
    for component in new_components:
        if len(component) < 3:
            continue
            
        ordered_pts = order_points_along_path(component)
        segments = split_crack_segments(ordered_pts)
        
        for segment in segments:
            if len(segment) < 3:
                continue
                
            descriptor = compute_crack_descriptors(segment, keypoint_count, is_new=1)
            if descriptor is not None:
                new_crack_descriptors.append(descriptor)
    
    # Process continued crack regions
    continued_components = skeleton_to_components(continued_regions)
    continued_crack_descriptors = []
    
    for component in continued_components:
        if len(component) < 3:
            continue
            
        ordered_pts = order_points_along_path(component)
        segments = split_crack_segments(ordered_pts)
        
        for segment in segments:
            if len(segment) < 3:
                continue
                
            descriptor = compute_crack_descriptors(segment, keypoint_count, is_new=0)
            if descriptor is not None:
                continued_crack_descriptors.append(descriptor)
    
    # 3. Handle special case: crack extensions
    # Check if any new crack connects to a continued crack
    # This identifies cracks that have grown from existing ones
    extension_descriptors = []
    indices_to_remove = []  # Track indices instead of removing items during iteration
    
    for i, new_desc in enumerate(new_crack_descriptors):
        is_extension = False
        
        for cont_desc in continued_crack_descriptors:
            # Extract endpoints
            new_start = np.array([new_desc[0], new_desc[1]])
            new_end = np.array([new_desc[2], new_desc[3]])
            cont_start = np.array([cont_desc[0], cont_desc[1]])
            cont_end = np.array([cont_desc[2], cont_desc[3]])
            
            # Check if any endpoint is close to an endpoint of a continued crack
            dist_start_start = np.linalg.norm(new_start - cont_start)
            dist_start_end = np.linalg.norm(new_start - cont_end)
            dist_end_start = np.linalg.norm(new_end - cont_start)
            dist_end_end = np.linalg.norm(new_end - cont_end)
            
            # Consider it an extension if one endpoint is close
            min_dist = min(dist_start_start, dist_start_end, dist_end_start, dist_end_end)
            if min_dist < 8:  # Threshold for connecting points
                is_extension = True
                # This is an extension part (keep it marked as new)
                extension_descriptors.append(new_desc)
                indices_to_remove.append(i)
                break
    
    # Remove items from original list in reverse order to maintain valid indices
    for idx in sorted(indices_to_remove, reverse=True):
        new_crack_descriptors.pop(idx)
                
    # Add extension descriptors back to new crack descriptors
    new_crack_descriptors.extend(extension_descriptors)
                
    return new_crack_descriptors, continued_crack_descriptors

######################################
# Reconstruction
######################################
def reconstruct_mask_from_descriptors(descriptors, image_shape, line_thickness=1):
    """
    Reconstructs a binary mask with thickness that matches the original mask.
    Updated to handle the extended descriptor format.
    """
    # Create empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    if not descriptors:
        return mask
    
    for desc in descriptors:
        # Check if we have the extended descriptor format (12 base elements)
        base_length = 12 if len(desc) >= 12 else 9
        
        # Extract keypoints
        keypoints_flat = desc[base_length:]
        
        # Handle case where keypoints_flat has odd length
        if len(keypoints_flat) % 2 != 0:
            # Pad with a duplicate of the last valid coordinate
            keypoints_flat = np.append(keypoints_flat, keypoints_flat[-1])
        
        num_keypoints = len(keypoints_flat) // 2
        keypoints = keypoints_flat.reshape(num_keypoints, 2)
        
        # Draw lines between consecutive keypoints
        for i in range(len(keypoints) - 1):
            pt1 = keypoints[i].astype(int)
            pt2 = keypoints[i + 1].astype(int)
            
            if (0 <= pt1[0] < image_shape[0] and 0 <= pt1[1] < image_shape[1] and
                0 <= pt2[0] < image_shape[0] and 0 <= pt2[1] < image_shape[1]):
                # Draw with the specified thickness
                cv2.line(mask, (pt1[1], pt1[0]), (pt2[1], pt2[0]), 255, line_thickness)
    
    return mask

def validate_crack_detection(test_ids, crack_dict, binary_masks, skeletons, output_dir="improved_validation"):
    """
    Improved validation function with simplified visualization focusing on:
    - New damage (green)
    - Old damage (gray)
    - False negatives (red)
    - False positives (blue)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # For each consecutive pair of test IDs
    for i in range(len(test_ids) - 1):
        curr_id = test_ids[i]
        next_id = test_ids[i + 1]
        
        # Skip if either doesn't have cracks
        if curr_id not in crack_dict or next_id not in crack_dict:
            continue
            
        curr_cracks = crack_dict[curr_id]
        next_cracks = crack_dict[next_id]
        
        if not curr_cracks or not next_cracks:
            continue
        
        # Create visualization with black background
        shape = binary_masks[curr_id].shape
        vis_img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        
        # Create a plain black background
        # (no original mask or skeleton as requested)
        
        # Detect extended descriptor format
        base_length = 12 if len(next_cracks[0]) >= 12 else 9
        
        # Create reconstructed mask for calculating false positives/negatives
        reconstructed = np.zeros_like(binary_masks[next_id])
        
        # Draw new and existing cracks with different colors
        new_count = 0
        existing_count = 0
        
        for desc in next_cracks:
            if len(desc) < base_length:
                continue
                
            # Extract keypoints properly
            keypoints_flat = desc[base_length:]
            
            # Handle case where keypoints_flat has odd length
            if len(keypoints_flat) % 2 != 0:
                keypoints_flat = np.append(keypoints_flat, keypoints_flat[-1])
            
            keypoints = keypoints_flat.reshape(-1, 2)
            
            # Check if it's flagged as new
            is_new = desc[8]
            
            if is_new > 0.5:  # New crack
                new_count += 1
                color = (0, 255, 0)  # Green for new damage
            else:  # Existing crack
                existing_count += 1
                color = (128, 128, 128)  # Gray for old damage
                
            # Draw on visualization image
            for i in range(len(keypoints) - 1):
                pt1 = keypoints[i].astype(int)
                pt2 = keypoints[i + 1].astype(int)
                
                if (0 <= pt1[0] < shape[0] and 0 <= pt1[1] < shape[1] and
                    0 <= pt2[0] < shape[0] and 0 <= pt2[1] < shape[1]):
                    cv2.line(vis_img, (pt1[1], pt1[0]), (pt2[1], pt2[0]), color, 1)
                    
                    # Also draw on reconstructed mask (for FP/FN calculation)
                    cv2.line(reconstructed, (pt1[1], pt1[0]), (pt2[1], pt2[0]), 255, 1)
        
        # Calculate false negatives (in original but not in reconstruction)
        fn_mask = np.logical_and(binary_masks[next_id] > 0, reconstructed == 0)
        vis_img[fn_mask] = [0, 0, 255]  # Red for false negatives
        
        # Calculate false positives (in reconstruction but not in original)
        fp_mask = np.logical_and(binary_masks[next_id] == 0, reconstructed > 0)
        vis_img[fp_mask] = [255, 0, 0]  # Blue for false positives
        
        # Calculate IoU 
        intersection = np.logical_and(reconstructed > 0, binary_masks[next_id] > 0)
        union = np.logical_or(reconstructed > 0, binary_masks[next_id] > 0)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        
        # Create simplified legend and info section
        legend_height = 160
        info_img = np.ones((legend_height, shape[1], 3), dtype=np.uint8) * 255
        
        # Title
        cv2.putText(info_img, f"Test {curr_id} -> {next_id}  (IoU: {iou:.4f})", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw color swatches and labels
        swatch_size = 15
        text_offset = swatch_size + 5
        y_start = 40
        y_step = 25
        
        # New damage (green)
        cv2.rectangle(info_img, (10, y_start), (10+swatch_size, y_start+swatch_size), (0, 255, 0), -1)
        cv2.putText(info_img, f"New damage ({new_count})", (10+text_offset, y_start+swatch_size//2+4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Old damage (gray)
        cv2.rectangle(info_img, (10, y_start+y_step), (10+swatch_size, y_start+y_step+swatch_size), (128, 128, 128), -1)
        cv2.putText(info_img, f"Old damage ({existing_count})", (10+text_offset, y_start+y_step+swatch_size//2+4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # False negatives (red)
        cv2.rectangle(info_img, (10, y_start+2*y_step), (10+swatch_size, y_start+2*y_step+swatch_size), (0, 0, 255), -1)
        cv2.putText(info_img, f"Missed damage (false negatives)", (10+text_offset, y_start+2*y_step+swatch_size//2+4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # False positives (blue)
        cv2.rectangle(info_img, (10, y_start+3*y_step), (10+swatch_size, y_start+3*y_step+swatch_size), (255, 0, 0), -1)
        cv2.putText(info_img, f"Extra damage (false positives)", (10+text_offset, y_start+3*y_step+swatch_size//2+4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Statistics
        stat_x = shape[1] // 2 + 20
        cv2.putText(info_img, f"Total detected cracks: {len(next_cracks)}", (stat_x, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Show percentages only if there are cracks
        if len(next_cracks) > 0:
            new_percent = 100 * new_count / len(next_cracks)
            existing_percent = 100 * existing_count / len(next_cracks)
            cv2.putText(info_img, f"New: {new_count} ({new_percent:.1f}%)", (stat_x, y_start+y_step), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(info_img, f"Existing: {existing_count} ({existing_percent:.1f}%)", (stat_x, y_start+2*y_step), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(info_img, f"IoU with original mask: {iou:.4f}", (stat_x, y_start+3*y_step), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw divider line
        cv2.line(info_img, (0, legend_height-1), (shape[1], legend_height-1), (0, 0, 0), 1)
        
        # Combine info and visualization
        combined = np.vstack([info_img, vis_img])
        
        # Save result
        cv2.imwrite(os.path.join(output_dir, f"validate_{curr_id}_to_{next_id}.png"), combined)
        
    print(f"Improved validation images saved to {output_dir}")

def print_descriptor_guide():
    """
    Prints a guide explaining the crack descriptor format and how to tune parameters.
    """
    guide = """
    ==============================================================================================
    CRACK DESCRIPTOR REFERENCE GUIDE
    ==============================================================================================
    
    == DESCRIPTOR FORMAT ==
    Each crack is represented by a descriptor with the following components:
    
    1. BASE DESCRIPTOR (9 elements):
       - [0] row_start    : Starting row coordinate (y-coordinate of start point)
       - [1] col_start    : Starting column coordinate (x-coordinate of start point)
       - [2] row_end      : Ending row coordinate (y-coordinate of end point)
       - [3] col_end      : Ending column coordinate (x-coordinate of end point)
       - [4] path_length  : Total length of the crack path in pixels
       - [5] orientation  : Overall orientation angle in radians
       - [6] avg_curvature: Average curvature along the path
       - [7] point_count  : Number of points in the original path
       - [8] is_new       : Flag indicating if this is a new crack (1) or continuation (0)
    
    2. KEYPOINTS (variable length, default 18 points = 36 elements):
       - Elements [9:] contain flattened keypoint coordinates 
       - Each keypoint is a (row, col) pair
       - Keypoints follow the path of the crack
       - More keypoints = better reconstruction but larger descriptors
    
    == PARAMETER TUNING GUIDE ==
    
    For LONG FALSE NEGATIVES (missing cracks), try:
    
    1. SKELETON ENHANCEMENT:
       - Increase max_gap in enhance_skeleton (default: 8)
         This connects more distant endpoints, capturing broken curves
       - Example: enhance_skeleton(skeleton, max_gap=10, curving_preference=True)
    
    2. CURVED CRACK DETECTION:
       - Increase curved_threshold in split_crack_segments (default: 10°)
         Higher values identify more regions as curved
       - Example: Change threshold from 10 to 15 degrees
    
    3. KEYPOINT COUNT:
       - Increase keypoint_count in process_test_cracks (default: 18)
         More keypoints capture complex paths better
       - Try values between 18-25 for complex cracks
    
    4. ANGLE THRESHOLDS:
       - Adjust angle thresholds in split_crack_segments
         - Curved regions: 75° (higher = fewer splits in curves)
         - Straight regions: 25° (lower = more sensitive to sharp turns)
       - For ladder structures, lower straight region threshold (try 20°)
    
    For FALSE POSITIVES (extra pixels), try:
    
    1. MINIMUM SEGMENT LENGTH:
       - Increase min_segment_length in split_crack_segments (default: 3)
         This ignores very short segments that may be noise
       - Example: split_crack_segments(ordered_pts, min_segment_length=4)
    
    2. SIMILARITY DETECTION:
       - Adjust position_threshold in is_similar_crack (default: 12)
         Lower values are stricter when matching cracks between tests
       - Example: is_similar_crack(desc1, desc2, position_threshold=10)
    
    For balancing CRACK FLAGGING (new vs. existing), try:
    
    1. MATCHING SENSITIVITY:
       - position_threshold in is_similar_crack controls how close endpoints
         need to be to consider cracks as the same/continued
       - Lower values (8-10px) = more cracks flagged as new
       - Higher values (12-15px) = more cracks flagged as continuations
    
    ==============================================================================================
    """
    print(guide)
    return guide

######################################
# Main Data Loader with Progressive Tracking
######################################
def load_data(params):
    """
    Loads accelerometer data and processes cracks with improved new crack identification.
    
    Args:
        params: Dictionary of parameters for crack detection and processing
        
    Returns:
        accel_dict: Dictionary of accelerometer data by test_id
        crack_dict: Dictionary of crack descriptors by test_id
        binary_masks: Dictionary of binary masks by test_id
        skeletons: Dictionary of enhanced skeletons by test_id
    """
    accel_dict = load_accelerometer_data(DATA_DIR, SKIP_TESTS)
    crack_dict = {}
    binary_masks = {}
    skeletons = {}
    
    # Get all test IDs and sort them
    test_ids = sorted(list(accel_dict.keys()))
    
    # Process each test in order
    previous_skeleton = None
    
    for test_id in test_ids:
        if test_id in SKIP_TESTS:
            continue
            
        # Load image and create binary mask
        combined_image = load_combined_label(test_id, LABELS_DIR, IMAGE_SHAPE)
        binary_mask = compute_binary_mask(combined_image)
        
        # Store the binary mask for validation
        binary_masks[test_id] = binary_mask
        
        # Process cracks with improved identification
        descriptors, current_skeleton = process_test_cracks(
            binary_mask, 
            previous_skeleton,
            keypoint_count=params['keypoint_count'],
            max_gap=params['max_gap']
        )
        
        # Store descriptors and skeleton
        crack_dict[test_id] = descriptors
        skeletons[test_id] = current_skeleton
        
        # Update for next test
        previous_skeleton = current_skeleton
    
    return accel_dict, crack_dict, binary_masks, skeletons
######################################
# For testing and validation
######################################
def main():
    """
    Main function with centralized parameter control to test the improved crack tracking.
    Added additional debugging to troubleshoot IoU issues.
    """
    print("INFO: Loading accelerometer data and processing cracks with improved methods...")
    
    # TUNABLE PARAMETERS
    params = {
        'keypoint_count': 20,
        'max_gap': 3,
        'curved_threshold': 10,
        'curved_angle_threshold': 85,
        'straight_angle_threshold': 20,
        'min_segment_length': 2,
        'line_thickness': 1,
    }
    
    print("\nCurrent parameter settings:")
    for param, value in params.items():
        print(f"  {param}: {value}")
    
    # Load data with improved methods that use skeleton subtraction
    accel_data, crack_dict, binary_masks, skeletons = load_data(params)
    
    # Validation: Check if we can reconstruct masks from descriptors
    print(f"\nINFO: Validating crack representation for {len(crack_dict)} tests...")
    
    total_iou = 0.0
    test_count = 0
    
    # Get test IDs for validation
    test_ids = sorted(list(crack_dict.keys()))
    
    # Create debug directory
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    
    for test_id, descriptors in crack_dict.items():
        # Skip if no descriptors
        if len(descriptors) == 0:
            print(f"  Test {test_id}: No cracks found")
            continue
            
        # Verify descriptor format
        extended_format = any(len(desc) >= 12 for desc in descriptors)
        base_length = 12 if extended_format else 9
        
        # Count new vs. existing cracks
        new_count = sum(1 for desc in descriptors if len(desc) > 8 and desc[8] > 0.5)
        existing_count = len(descriptors) - new_count
        
        print(f"  Test {test_id}: {len(descriptors)} cracks ({new_count} new, {existing_count} existing)")
        
        # Save original skeleton and mask
        if test_id in skeletons:
            cv2.imwrite(os.path.join(debug_dir, f"test_{test_id}_skeleton.png"), skeletons[test_id])
        
        # Reconstruct mask from descriptors
        reconstructed = reconstruct_mask_from_descriptors(
            descriptors, binary_masks[test_id].shape, line_thickness=params['line_thickness'])
        
        # Compare with original mask
        original = binary_masks[test_id]
        
        # Calculate non-zero pixels in each mask
        orig_nonzero = np.sum(original > 0)
        recon_nonzero = np.sum(reconstructed > 0)
        
        print(f"    Original mask pixels: {orig_nonzero}")
        print(f"    Reconstructed mask pixels: {recon_nonzero}")
        
        intersection = np.logical_and(reconstructed > 0, original > 0)
        union = np.logical_or(reconstructed > 0, original > 0)
        
        intersection_count = np.sum(intersection)
        union_count = np.sum(union)
        
        print(f"    Intersection pixels: {intersection_count}")
        print(f"    Union pixels: {union_count}")
        
        # Calculate IoU (Intersection over Union)
        iou = intersection_count / union_count if union_count > 0 else 0
        total_iou += iou
        test_count += 1
        
        print(f"    Reconstruction IoU: {iou:.4f}")
        
        # Save debug images
        cv2.imwrite(os.path.join(debug_dir, f"test_{test_id}_original.png"), original)
        cv2.imwrite(os.path.join(debug_dir, f"test_{test_id}_reconstructed.png"), reconstructed)
        
        # Create overlay visualization
        diff_mask = np.zeros((original.shape[0], original.shape[1], 3), dtype=np.uint8)
        diff_mask[intersection > 0] = [0, 255, 0]  # True positive (green)
        diff_mask[np.logical_and(reconstructed > 0, original == 0)] = [255, 0, 0]  # False positive (red)
        diff_mask[np.logical_and(reconstructed == 0, original > 0)] = [0, 0, 255]  # False negative (blue)
        
        cv2.imwrite(os.path.join(debug_dir, f"test_{test_id}_diff.png"), diff_mask)
    
    # Calculate average IoU
    avg_iou = total_iou / test_count if test_count > 0 else 0
    print(f"\nAverage IoU across all tests: {avg_iou:.4f}")
    
    # Run improved validation
    validate_crack_detection(test_ids, crack_dict, binary_masks, skeletons, "improved_validation")
    
    print("\nINFO: Processing complete.")

if __name__ == "__main__":
    main()