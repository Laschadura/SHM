from __future__ import annotations
import tensorflow as tf

def configure_tf(intra=2, inter=1, *, mem_growth=True, verbose=True):
    tf.config.threading.set_intra_op_parallelism_threads(intra)
    tf.config.threading.set_inter_op_parallelism_threads(inter)

    if mem_growth:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

    if verbose:
        gpus = tf.config.list_logical_devices("GPU")
        msg  = f"✅  {len(gpus)} GPU(s) – memory-growth ON" if gpus else "⚠️  CPU-only"
        print(msg)

__all__ = ["configure_tf"]
