# tests/test_masks.py
import matplotlib.pyplot as plt
import numpy as np
import cv2
from bridge_data.loader import load_data
from bridge_data.preprocess import mask_to_heatmap
from bridge_data.postprocess import mask_recon

# Test and visualize mask downsampling/upsampling
def test_mask_processing():
    _, binary_masks, _, *_ = load_data(recompute=False)
    test_id = 25
    highres_mask = binary_masks[test_id]  # (256, 768)

    # Downsample
    heatmap_soft = mask_to_heatmap(highres_mask, (32, 96), interpolation=cv2.INTER_AREA)
    heatmap_bin = mask_to_heatmap(highres_mask, (32, 96), interpolation=cv2.INTER_NEAREST)
    heatmap_thresh = mask_to_heatmap(highres_mask, (32, 96), interpolation=cv2.INTER_AREA, binarize=True, threshold=0.03)

    # Upsample
    heatmap_thresh_exp = heatmap_thresh[None, ..., None]
    up_thresh = mask_recon(heatmap_thresh_exp)[0]

    # Plot
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    ax[0].imshow(highres_mask, cmap="gray"); ax[0].set_title("High-res GT")
    ax[1].imshow(heatmap_soft, cmap="gray"); ax[1].set_title("Soft Heatmap (INTER_AREA)")
    ax[2].imshow(heatmap_thresh, cmap="gray"); ax[2].set_title("Thresholded Binary")
    ax[3].imshow(up_thresh, cmap="gray"); ax[3].set_title("Upsampled")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    test_mask_processing()
