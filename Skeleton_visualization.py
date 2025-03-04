import os
import pickle
import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.io as pio
import data_loader

OUTPUT_DIR = "processed_masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_skeleton_on_image(test_id, image_rgb, skeleton):
    """
    Overlays a skeleton (binary 0/1 array) on the original combined RGB image.
    Uses Plotly to produce an interactive HTML plot.
    """
    # image_rgb: shape (H, W, 3)
    # skeleton:  shape (H, W) with 0/1

    # We'll plot the original image as a 'go.Image'
    fig = go.Figure()
    fig.add_trace(go.Image(z=image_rgb))

    # Now overlay the skeleton pixels as a scatter
    y_coords, x_coords = np.where(skeleton > 0)

    # Because Plotly's coordinate system for images starts at top-left,
    # we want (x, y) to correspond to (col, row).
    # We'll just overlay them with some visible color & small size.
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="markers",
        marker=dict(color="red", size=3),
        name="Skeleton"
    ))

    # Aesthetic adjustments: fix the aspect ratio to 'image' so it doesn't stretch
    fig.update_layout(
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    )

    # Reverse the y-axis if you want the top-left to be (0,0) visually:
    # But typically go.Image already sets origin at top-left. 
    # If needed, do:
    # fig.update_yaxes(autorange="reversed")

    # Save to HTML
    output_path = os.path.join(OUTPUT_DIR, f"test_{test_id}_skeleton.html")
    pio.write_html(fig, output_path, auto_open=False)
    print(f"✅ Saved skeleton overlay for Test {test_id} to {output_path}")

def main():
    print("📥 Loading data...")
    accel_dict, mask_dict = data_loader.load_data()  
    # -> mask_dict[test_id] should be a skeleton or any other representation 
    #    depending on how data_loader is implemented.

    test_ids = sorted(mask_dict.keys())
    
    for test_id in test_ids:
        print(f"🔍 Visualizing Test ID {test_id}...")

        # 1) Get the original combined image from data_loader
        #    We'll call the function that loads the combined perspectives *directly*:
        #    (If your data_loader doesn't expose this function, just copy it in here.)
        image_rgb = data_loader.load_combined_label(test_id, data_loader.LABELS_DIR, data_loader.IMAGE_SHAPE)
        if image_rgb is None or image_rgb.size == 0:
            print(f"Warning: Could not load original image for Test {test_id}, skipping.")
            continue

        # 2) Retrieve skeleton from mask_dict
        skeleton = mask_dict[test_id]
        if skeleton is None or skeleton.size == 0:
            print(f"Warning: Skeleton for Test {test_id} is empty, skipping visualization.")
            continue
        
        # 3) Visualize
        visualize_skeleton_on_image(test_id, image_rgb, skeleton)

    print("✅ All visualizations saved.")

if __name__ == "__main__":
    main()
