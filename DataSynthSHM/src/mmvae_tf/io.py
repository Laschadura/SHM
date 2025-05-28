import numpy as np
import tensorflow as tf

from bridge_data.tf_wrappers import augment_fn

# ----- Data Processing and Dataset Creation -----
def create_tf_dataset(
    spectrograms, mask_array, test_id_array, segments,
    batch_size=32, shuffle=True, debug_mode=False, debug_samples=500, augment=True
    ):
    """
    Create a TensorFlow Dataset that loads data from memory.
    
    Args:
        spectrograms: Array or path to file containing spectrograms.
        mask_array: Array or path to file containing binary masks.
        test_id_array: Array of test IDs.
        batch_size: Batch size for the dataset.
        shuffle: Whether to shuffle the dataset.
        debug_mode: If True, only a limited number of samples are used.
        debug_samples: Number of samples to use in debug mode.
        
    Returns:
        A tf.data.Dataset yielding tuples (spectrogram, mask, test_id).
    """
    # Convert file path to memory-mapped array if a string is provided
    if isinstance(spectrograms, str):
        spectrograms = np.load(spectrograms, mmap_mode='r')
    if isinstance(mask_array, str):
        mask_array = np.load(mask_array, mmap_mode='r')
    if isinstance(segments, str):
        segments = np.load(segments, mmap_mode='r')
    
    # Apply debug mode limit
    if debug_mode:
        print(f"⚠️ Debug Mode ON: Using only {debug_samples} samples for quick testing!")
        spectrograms = spectrograms[:debug_samples]
        mask_array = mask_array[:debug_samples]
        test_id_array = test_id_array[:debug_samples]
        segments = segments[:debug_samples]
    else:
        print(f"✅ Full dataset loaded: {len(mask_array)} samples.")
    
    dataset = tf.data.Dataset.from_tensor_slices((spectrograms, mask_array, test_id_array, segments))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(mask_array))

    if augment:
        dataset = dataset.map(augment_fn,
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = (dataset
               .batch(batch_size, drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE))
    return dataset
