import tensorflow as tf
import numpy as np

################################################################################
#---------losses---------------------------------------------------------------
################################################################################
def complex_spectrogram_loss(y_true, y_pred):
    total_channels = tf.shape(y_true)[-1]
    
    mag_indices = tf.range(0, total_channels, delta=2)
    phase_indices = tf.range(1, total_channels, delta=2)
    
    mag_true = tf.gather(y_true, mag_indices, axis=-1)
    mag_pred = tf.gather(y_pred, mag_indices, axis=-1)
    
    phase_true = tf.gather(y_true, phase_indices, axis=-1)
    phase_pred = tf.gather(y_pred, phase_indices, axis=-1)

    # L2 magnitude loss (can replace with L1)
    mag_loss = tf.reduce_mean(tf.square(mag_true - mag_pred))
    
    # Phase cosine distance
    phase_true_complex = tf.complex(tf.cos(phase_true), tf.sin(phase_true))
    phase_pred_complex = tf.complex(tf.cos(phase_pred), tf.sin(phase_pred))
    phase_diff_cos = tf.math.real(phase_true_complex * tf.math.conj(phase_pred_complex))
    phase_loss = tf.reduce_mean(1.0 - phase_diff_cos)

    # Instantaneous frequency loss (optional but useful)
    if_loss = tf.reduce_mean(tf.abs(phase_pred[:,1:] - phase_pred[:,:-1] -
                                    phase_true[:,1:] + phase_true[:,:-1]))

    return 0.5 * mag_loss + 0.3 * phase_loss + 0.2 * if_loss

#--------- Mask Losses ---------------------------------------------------------
def custom_mask_loss(y_true, y_pred, 
                       weight_bce=0.2, 
                       weight_dice=0.3, 
                       weight_focal=0.5, 
                       gamma=2.0, # Focal loss focusing parameter
                       alpha=0.3 # Focal loss balancing parameter
                       ):
    """
    Computes a weighted combination of Binary Cross-Entropy (BCE), Dice, and Focal losses.
    
    Args:
        y_true (tf.Tensor): Ground truth mask with shape (batch, height, width, 1) and values in [0,1].
        y_pred (tf.Tensor): Predicted mask with shape (batch, height, width, 1) and values in [0,1].
        weight_bce (float): Weight for the BCE loss component.
        weight_dice (float): Weight for the Dice loss component.
        weight_focal (float): Weight for the Focal loss component.
        gamma (float): Focusing parameter for focal loss.
        alpha (float): Balancing parameter for focal loss.
        
    Returns:
        tf.Tensor: The combined loss scalar.
    """
    # Get a small constant for numerical stability.
    epsilon = tf.keras.backend.epsilon()
    
    # --- BCE Loss ---
    y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    bce = -(y_true * tf.math.log(y_pred_clipped) + (1.0 - y_true) * tf.math.log(1.0 - y_pred_clipped))
    bce_loss = tf.reduce_mean(bce)
    # Ensure the BCE loss is positive.
    bce_loss = tf.abs(bce_loss)
    
    # --- Dice Loss ---
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    y_pred_flat = tf.clip_by_value(y_pred_flat, epsilon, 1.0 - epsilon)
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    smooth = 1.0
    dice_coef = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_coef
    
    # --- Focal Loss ---
    y_pred_focal = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(tf.equal(y_true, 1), y_pred_focal, 1 - y_pred_focal)
    focal_weight = tf.pow(1.0 - pt, gamma)
    alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    focal_bce = -tf.math.log(pt)
    focal_loss = tf.reduce_mean(alpha_weight * focal_weight * focal_bce)
    
    # --- Combined Loss ---
    combined = weight_bce * bce_loss + weight_dice * dice_loss + weight_focal * focal_loss
    return tf.cast(combined, tf.float32)

#--------- Waveform Losses -----------------------------------------------------
def multi_channel_mrstft_loss(
        y_true, y_pred, fft_sizes=(256, 512, 1024), eps=1e-7):
    n_ch = tf.shape(y_true)[-1]
    loss = 0.0

    for ch in range(n_ch):
        for n_fft in fft_sizes:
            hop = n_fft // 4
            win_fn = lambda L, dtype=tf.float32: tf.signal.hann_window(L, dtype=dtype)

            # complex STFTs
            S_t = tf.signal.stft(y_true[..., ch], n_fft, hop,
                                 window_fn=win_fn, pad_end=True)
            S_p = tf.signal.stft(y_pred[..., ch], n_fft, hop,
                                 window_fn=win_fn, pad_end=True)

            # ★ log-magnitude distance
            logmag_t = tf.math.log1p(tf.abs(S_t))
            logmag_p = tf.math.log1p(tf.abs(S_p))
            loss += tf.reduce_mean(tf.abs(logmag_t - logmag_p))

    # divide *inside* ⇒ each (channel, FFT) contributes equally
    return loss / tf.cast(n_ch * len(fft_sizes), tf.float32)

def waveform_si_l1_loss(y_true, y_pred, eps=1e-7):
    """Scale-invariant L1 in the time domain."""
    diff = tf.reduce_mean(tf.abs(y_true - y_pred))
    denom = tf.reduce_mean(tf.abs(y_true)) + eps
    return diff / denom

def waveform_l1_loss(y_true, y_pred):
    """Full L1 waveform loss over all channels"""
    return tf.reduce_mean(tf.abs(y_true - y_pred))

#--------- Magnitude Losses ----------------------------------------------------
def magnitude_l1_loss(spec_true, spec_pred):
    """Computes L1 loss between only the magnitude channels of a complex spectrogram."""
    C = tf.shape(spec_true)[-1] // 2
    mag_true = spec_true[..., :C]
    mag_pred = spec_pred[..., :C]
    return tf.reduce_mean(tf.abs(mag_true - mag_pred))

#--------- Phase Losses --------------------------------------------------------
def extract_phase_channels(spec):
    """Extract only phase channels: spec[..., 1::2]"""
    return spec[..., 1::2]

def gradient_loss_phase_only(spec_true, spec_pred):
    phase_true = extract_phase_channels(spec_true)
    phase_pred = extract_phase_channels(spec_pred)

    dx_true = phase_true[..., 1:, :] - phase_true[..., :-1, :]
    dx_pred = phase_pred[..., 1:, :] - phase_pred[..., :-1, :]

    dy_true = phase_true[..., :, 1:] - phase_true[..., :, :-1]
    dy_pred = phase_pred[..., :, 1:] - phase_pred[..., :, :-1]

    loss_dx = tf.reduce_mean(tf.abs(dx_true - dx_pred))
    loss_dy = tf.reduce_mean(tf.abs(dy_true - dy_pred))

    return loss_dx + loss_dy

def laplacian_loss_phase_only(spec_true, spec_pred):
    phase_true = extract_phase_channels(spec_true)
    phase_pred = extract_phase_channels(spec_pred)

    laplace = lambda x: (
        -4 * x[..., 1:-1, 1:-1] +
        x[..., :-2, 1:-1] + x[..., 2:, 1:-1] +
        x[..., 1:-1, :-2] + x[..., 1:-1, 2:]
    )

    lap_true = laplace(phase_true)
    lap_pred = laplace(phase_pred)
    return tf.reduce_mean(tf.abs(lap_true - lap_pred))

