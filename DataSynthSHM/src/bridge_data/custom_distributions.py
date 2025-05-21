import tensorflow as tf
import numpy as np

# Constants for numerical stability
EPSILON = 1e-6
LOGVAR_MIN = -20.0
LOGVAR_MAX = 8.0

def reparameterize(mean, logvar):
    """
    Reparameterization trick for sampling from a Gaussian distribution.
    Added clipping for numerical stability.
    
    Args:
        mean: Mean of the Gaussian
        logvar: Log variance of the Gaussian
        
    Returns:
        Sampled latent vector
    """
    # Clip logvar to avoid numerical issues
    logvar = tf.cast(tf.clip_by_value(logvar, LOGVAR_MIN, LOGVAR_MAX), dtype=tf.float32)
    mean = tf.cast(mean, dtype=tf.float32)
    
    eps = tf.random.normal(shape=tf.shape(mean), dtype=tf.float32)
    std = tf.exp(0.5 * logvar)
    std = tf.maximum(std, EPSILON)
    return mean + eps * std

def compute_kl_divergence(mu_q, logvar_q, mu_p=None, logvar_p=None):
    """
    Compute KL divergence between two Gaussian distributions: q(z|x) and p(z).
    Added clipping and epsilon for numerical stability.
    
    Args:
        mu_q: Mean of q distribution
        logvar_q: Log variance of q distribution
        mu_p: Mean of p distribution (None means standard normal)
        logvar_p: Log variance of p distribution (None means standard normal)
        
    Returns:
        KL divergence
    """
    # Clip logvar for stability and ensure it's a float32
    logvar_q = tf.cast(tf.clip_by_value(logvar_q, LOGVAR_MIN, LOGVAR_MAX), dtype=tf.float32)
    mu_q = tf.cast(mu_q, dtype=tf.float32)
    
    if mu_p is None:
        # KL with standard normal
        kl = -0.5 * tf.reduce_sum(
            1 + logvar_q - tf.square(mu_q) - tf.exp(logvar_q),
            axis=-1
        )
    else:
        # ensure dtype consistency
        mu_p = tf.cast(mu_p, dtype=tf.float32)
        logvar_p = tf.cast(logvar_p, dtype=tf.float32)
        # Clip p distribution logvar as well
        logvar_p = tf.clip_by_value(logvar_p, LOGVAR_MIN, LOGVAR_MAX)
        
        # compute the variances
        var_q = tf.exp(logvar_q)
        var_p = tf.exp(logvar_p)

        # clamp them to be at least EPSILON
        var_q = tf.maximum(var_q, EPSILON)
        var_p = tf.maximum(var_p, EPSILON)

        # now safe to do the KL formula
        kl = 0.5 * tf.reduce_sum(
            logvar_p - logvar_q +
            (var_q + tf.square(mu_q - mu_p)) / var_p -
            1,
            axis=-1
        )

    # Apply tf.maximum to avoid extreme negative values
    return tf.reduce_mean(tf.maximum(kl, 0.0))

def compute_mixture_prior(mus, logvars):
    """
    Compute the Mixture-of-Experts prior parameters.
    The prior is a mixture of the unimodal posteriors.
    Added clipping and epsilon for numerical stability.
    
    Args:
        mus: List of means from each modality encoder
        logvars: List of log variances from each modality encoder
        
    Returns:
        Tuple of (mixture_mu, mixture_logvar)
    """
    # Check for empty list
    if not mus or len(mus) == 0:
        raise ValueError("Empty list of means provided to compute_mixture_prior")
    
    # Clip logvars for stability
    logvars = [tf.clip_by_value(lv, LOGVAR_MIN, LOGVAR_MAX) for lv in logvars]
    
    # Stack means and variances
    all_mus = tf.stack(mus, axis=0)      # [num_modalities, batch_size, latent_dim]
    all_logvars = tf.stack(logvars, axis=0)  # [num_modalities, batch_size, latent_dim]

    # Ensure all_mus and all_logvars are float32
    all_mus = tf.cast(all_mus, dtype=tf.float32)
    all_logvars = tf.cast(all_logvars, dtype=tf.float32)
    
    # Compute mixture parameters (average over modalities)
    mixture_mu = tf.reduce_mean(all_mus, axis=0)  # [batch_size, latent_dim]
    
    # For the mixture variance, we need to account for both the mean of variances
    # and the variance of means (law of total variance)
    exp_logvars = tf.exp(all_logvars)  # [num_modalities, batch_size, latent_dim]
    mean_var = tf.reduce_mean(exp_logvars, axis=0)  # [batch_size, latent_dim]
    
    # Compute variance of means
    mu_diff_sq = tf.reduce_mean(tf.square(all_mus - mixture_mu[tf.newaxis, :, :]), axis=0)
    
    # Total variance is mean of variances plus variance of means, add epsilon for stability
    mixture_var = mean_var + mu_diff_sq
    mixture_var = tf.maximum(mixture_var, EPSILON)      # enforce > 0
    mixture_logvar = tf.math.log(mixture_var)
    
    # Final clip on the mixture logvar
    mixture_logvar = tf.clip_by_value(mixture_logvar, LOGVAR_MIN, LOGVAR_MAX)
    
    return mixture_mu, mixture_logvar

def compute_js_divergence(mus, logvars):
    """
    Compute the Jensen-Shannon divergence among multiple Gaussian distributions.
    This follows the Unity by Diversity paper approach where JS is computed between
    each unimodal posterior and the mixture-of-experts prior.
    Added clipping and error handling for stability.
    
    Args:
        mus: List of means from each modality encoder
        logvars: List of log variances from each modality encoder
        
    Returns:
        JS divergence (scaled by number of modalities)
    """
    # ensure all inputs are float32
    mus = [tf.cast(m, tf.float32) for m in mus]
    logvars = [tf.cast(lv, tf.float32) for lv in logvars]

    num_modalities = len(mus)
    if num_modalities <= 1:
        return tf.constant(0.0, dtype=tf.float32)

    try:
        mixture_mu, mixture_logvar = compute_mixture_prior(mus, logvars)
    except ValueError:
        return tf.constant(0.0, dtype=tf.float32)

    kl_divs = []
    for i in range(num_modalities):
        mu_i = tf.clip_by_norm(mus[i], 10.0, axes=-1)
        logvar_i = tf.clip_by_value(logvars[i], LOGVAR_MIN, LOGVAR_MAX)
        kl = compute_kl_divergence(mu_i, logvar_i, mixture_mu, mixture_logvar)
        kl_divs.append(kl)

    kl_divs = tf.stack(kl_divs)
    is_finite = tf.math.is_finite(kl_divs)
    safe_kl_divs = tf.boolean_mask(kl_divs, is_finite)

    num_masked = tf.reduce_sum(tf.cast(tf.logical_not(is_finite), tf.int32))
    tf.print("JS divergence: masked", num_masked, "non-finite KL terms")

    if tf.size(safe_kl_divs) > 0:
        js_div = tf.reduce_mean(safe_kl_divs)
    else:
        tf.print("JS divergence: all KL terms were invalid, returning 0")
        js_div = tf.constant(0.0, dtype=tf.float32)

    return tf.clip_by_value(js_div, 0.0, 1000.0)


def sample_from_mixture_prior(mus, logvars):
    """
    Sample from the mixture-of-experts prior.
    Added error handling for stability.
    
    Args:
        mus: List of means from each modality encoder
        logvars: List of log variances from each modality encoder
        
    Returns:
        Sampled latent vector
    """
    num_modalities = len(mus)
    if num_modalities == 0:
        raise ValueError("Cannot sample from empty mixture")
        
    batch_size = tf.shape(mus[0])[0]
    
    # Randomly select which mixture component to sample from for each example in batch
    # This implements a proper mixture model sampling
    component_indices = tf.random.uniform(
        shape=[batch_size], 
        minval=0, 
        maxval=num_modalities,
        dtype=tf.int32
    )
    
    # Stack all means and logvars
    all_mus = tf.stack(mus, axis=1)      # [batch_size, num_modalities, latent_dim]
    
    # Clip logvars for stability before stacking
    clipped_logvars = [tf.clip_by_value(lv, LOGVAR_MIN, LOGVAR_MAX) for lv in logvars]
    all_logvars = tf.stack(clipped_logvars, axis=1)  # [batch_size, num_modalities, latent_dim]
    
    # Create batch indices
    batch_indices = tf.range(batch_size)
    
    # Select the means and logvars for the sampled components
    selected_mus = tf.gather_nd(all_mus, tf.stack([batch_indices, component_indices], axis=1))
    selected_logvars = tf.gather_nd(all_logvars, tf.stack([batch_indices, component_indices], axis=1))
    
    # Sample using reparameterization trick (which has its own clipping)
    return reparameterize(selected_mus, selected_logvars)

def impute_missing_modality(present_mus, present_logvars, missing_idx, num_total_modalities):
    """
    Impute a missing modality by sampling from the mixture of available modalities.
    Added error handling and stability measures.
    
    Args:
        present_mus: List of means from available modality encoders
        present_logvars: List of log variances from available modality encoders
        missing_idx: Index of the missing modality
        num_total_modalities: Total number of modalities in the model
        
    Returns:
        Sampled latent vector for the missing modality
    """
    # Check if there are any available modalities
    if not present_mus or len(present_mus) == 0:
        raise ValueError("No modalities available for imputation")
    
    # Compute mixture prior parameters from available modalities
    mixture_mu, mixture_logvar = compute_mixture_prior(present_mus, present_logvars)
    
    # Sample from the mixture prior using the stabilized reparameterization
    imputed_z = reparameterize(mixture_mu, mixture_logvar)
    
    return imputed_z