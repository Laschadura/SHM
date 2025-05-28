"""
High-level convenience loader that glues IO ⇢ preprocessing ⇢ spectrograms.
"""

import os, numpy as np, cv2
import pickle
from pathlib import Path
from .config      import DATA_DIR, LABELS_DIR
from .io          import load_accelerometer_data, load_combined_label
from .preprocess  import segment_and_transform, mask_to_heatmap, compute_binary_mask, compute_complex_spectrogram

def load_data(segment_duration: float = 4.0,
              nperseg: int = 256,
              noverlap: int = 224,
              sample_rate: int = 200,
              recompute: bool = False,
              cache_dir: str = "cache"):

    accel_dict = load_accelerometer_data()
    binary_masks, heatmaps = {}, {}

    for tid in sorted(accel_dict):
        comb  = load_combined_label(tid)
        bmask = compute_binary_mask(comb)
        binary_masks[tid] = bmask
        heatmaps[tid]     = mask_to_heatmap(bmask, target_size=(32,96), binarize=True)[...,None]

    # ---------- cache handling ----------
    os.makedirs(cache_dir, exist_ok=True)
    tag        = f"{segment_duration:.2f}s_{nperseg}_{noverlap}"
    seg_path   = Path(cache_dir) / f"segments_{tag}.npy"
    spec_path  = Path(cache_dir) / f"specs_{tag}.npy"
    ids_path   = Path(cache_dir) / f"segIDs_{tag}.npy"
    bmask_path = Path(cache_dir) / f"binary_masks_{tag}.npy"
    hmap_path  = Path(cache_dir) / f"heatmaps_{tag}.npy"
    meta_path = Path(cache_dir) / f"segment_metadata_{tag}.pkl"
    stats_path = Path(cache_dir) / f"segStats_{tag}.pkl"

    if not recompute and all(p.exists() for p in [seg_path, spec_path, ids_path, bmask_path, hmap_path, meta_path, seg_path]):
        segments      = np.load(seg_path,  mmap_mode="r")
        spectrograms  = np.load(spec_path, mmap_mode="r")
        test_ids      = np.load(ids_path,  mmap_mode="r")
        binary_masks  = np.load(bmask_path, allow_pickle=True).item()
        heatmaps      = np.load(hmap_path,  allow_pickle=True).item()
        with open(meta_path, "rb") as f:
                segment_metadata = pickle.load(f)
        with open(stats_path, "rb") as f:
                seg_stats = pickle.load(f)
        return accel_dict, binary_masks, heatmaps, segments, spectrograms, test_ids, segment_metadata, seg_stats

    # ---------- compute fresh ----------
    segments, _, test_ids, segment_metadata, seg_stats = segment_and_transform(
        accel_dict, heatmaps, sample_rate, segment_duration
    )
    spectrograms = compute_complex_spectrogram(segments, sample_rate, nperseg, noverlap)

    np.save(seg_path,  segments)
    np.save(spec_path, spectrograms)
    np.save(ids_path,  test_ids)
    np.save(bmask_path, binary_masks, allow_pickle=True)
    np.save(hmap_path,  heatmaps,     allow_pickle=True)
    with open(meta_path, "wb") as f:
        pickle.dump(segment_metadata, f)
    with open(stats_path, "wb") as f:
        pickle.dump(seg_stats, f)




    return accel_dict, binary_masks, heatmaps, segments, spectrograms, test_ids, segment_metadata, seg_stats

