import tensorflow as tf
import numpy as np

# ────────────  A U G M E N T A T I O N  ──────────────────────────
_FRAME_STEP  = 32        #  nperseg 256 – noverlap 224  ⇒ 256-224 = 32
_MAX_SHIFT   = 160       #  ≤0.8 s at 200 Hz  (tweakable)
_PINK_ALPHA  = 0.5       #  0→white, 0.5→pink, 1→brown

def _make_pink(shape, alpha=_PINK_ALPHA):
    """TensorFlow-compatible 1/f^alpha pink noise generator (approx)."""
    T = shape[0]
    C = shape[1]

    # Ensure FFT length is next power of 2 for efficiency
    fft_len = tf.cast(tf.math.pow(2.0, tf.math.ceil(tf.math.log(tf.cast(T, tf.float32)) / tf.math.log(2.0))), tf.int32)

    # Frequency shaping
    freqs = tf.cast(tf.range(1, fft_len // 2 + 1), tf.float32)
    scale = tf.pow(freqs, -alpha)[:, tf.newaxis]  # (F,1)

    # Random phase and magnitude
    real = tf.random.normal([fft_len // 2, C])
    imag = tf.random.normal([fft_len // 2, C])
    comp = tf.complex(real * scale, imag * scale)

    # Mirror to get full spectrum (DC + pos + neg freq)
    comp_full = tf.concat(
        [tf.complex(tf.zeros([1, C]), tf.zeros([1, C])), comp, tf.reverse(tf.math.conj(comp), axis=[0])],
        axis=0
    )

    # IFFT to time-domain
    pink = tf.signal.ifft(comp_full)
    pink = tf.math.real(pink[:T])  # (T,C)
    return tf.cast(pink, tf.float32)

def augment_fn(spec, mask, tid, wave):
    """Synchronised spec + wave augmentation."""
    # 1. random time-shift
    samp_shift  = tf.random.uniform([], -_MAX_SHIFT, _MAX_SHIFT, tf.int32)
    frame_shift = samp_shift // _FRAME_STEP
    wave = tf.roll(wave,  samp_shift,  axis=0)
    spec = tf.roll(spec, frame_shift, axis=1)

    # 2. random gain  ±3 dB
    gain = tf.pow(10.0, tf.random.uniform([], -3.0, 3.0) / 20.0)
    wave = wave * gain
    mag  = spec[..., 0::2] * gain               # scale magnitude only
    spec = tf.concat([mag, spec[..., 1::2]], -1)

    # 3. sensor-axis sign-flip  (25 %)
    flip = tf.random.uniform([]) > 0.25
    wave = tf.where(flip, -wave, wave)
    phase = spec[..., 1::2] + tf.where(flip, np.pi, 0.0)
    spec  = tf.concat([spec[..., 0::2], phase], -1)

    # 4. pink noise, σ≈0.005
    noise = _make_pink(tf.shape(wave)) * 0.005
    wave  = wave + noise

    # 5. SpecAug-style frequency drop (1–3 bins), applied with 30% probability
    if tf.random.uniform([]) < 0.3:
        f_drop = tf.random.uniform([], 1, 4, tf.int32)
        f0     = tf.random.uniform([], 0, tf.shape(spec)[0] - f_drop, tf.int32)
        zeros  = tf.zeros_like(spec[f0:f0+f_drop, :, 0::2])
        spec = tf.tensor_scatter_nd_update(
            spec,
            indices=tf.range(f0, f0 + f_drop)[:, None],
            updates=tf.concat([zeros,
                               spec[f0:f0 + f_drop, :, 1::2]], -1))


    return spec, mask, tid, wave

