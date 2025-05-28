# bridge_data/tests/test_end_to_end_pipeline.py
from bridge_data.loader import load_data

def test_end_to_end_pipeline():
    print("🔄 Recomputing and caching everything...")
    accel_dict, binary_masks, heatmaps, segments, spectrograms, seg_ids, segment_metadata, seg_stats = load_data(recompute=True)

    print(f"✅ Loaded {len(accel_dict)} test IDs")
    print(f"📐 Segments shape:     {segments.shape}")
    print(f"📐 Spectrograms shape: {spectrograms.shape}")
    print(f"📐 Test ID array:      {seg_ids.shape}")
    print(f"📎 Metadata entry example: {segment_metadata[0]}")
    print(f"📊 Segment statistics: {seg_stats[0]}")

    # Check masks
    example_id = seg_ids[0]
    print(f"🧪 Example ID: {example_id}")
    print(f"   • Binary mask shape: {binary_masks[example_id].shape}")
    print(f"   • Heatmap shape:     {heatmaps[example_id].shape}")

if __name__ == "__main__":
    test_end_to_end_pipeline()
