"""
Custom implementation of distribution-related functions to replace TensorFlow Probability.
This module provides simplified Gaussian distribution operations for computing JS divergence.
"""

import tensorflow as tf

class GaussianDistribution:
    """
    Simple Gaussian distribution with diagonal covariance.
    """
    def __init__(self, mu, logvar):
        """
        Initialize a Gaussian distribution.
        
        Args:
            mu: Mean vector (batch_size, latent_dim)
            logvar: Log variance vector (batch_size, latent_dim)
        """
        self.mu = mu
        self.logvar = tf.clip_by_value(logvar, -10.0, 8.0)  # Clip for numerical stability
        self.var = tf.exp(self.logvar)
        self.stddev = tf.sqrt(self.var)
    
    def kl_divergence(self, other):
        """
        Compute KL(self || other) for two Gaussian distributions.
        
        Args:
            other: Another GaussianDistribution instance
            
        Returns:
            KL divergence per batch element (batch_size,)
        """
        # KL divergence between two multivariate Gaussians with diagonal covariance:
        # 0.5 * sum(log(var_other/var_self) + (var_self + (mu_self - mu_other)²)/var_other - 1)
        kl = 0.5 * tf.reduce_sum(
            other.logvar - self.logvar + 
            (self.var + tf.square(self.mu - other.mu)) / other.var - 1.0,
            axis=-1
        )
        return kl

def compute_mixture_distribution(distributions, weights=None):
    """
    Create a mixture distribution from a list of distributions.
    
    Args:
        distributions: List of GaussianDistribution instances
        weights: Optional weights for the mixture (defaults to uniform)
        
    Returns:
        A simplified representation of the mixture suitable for KL calculations
    """
    n_components = len(distributions)
    
    if weights is None:
        weights = [1.0 / n_components] * n_components
    
    # For simplicity, we'll approximate the mixture's properties
    # This is not exact but works well enough for our JS divergence calculation
    weighted_mu = tf.zeros_like(distributions[0].mu)
    weighted_var = tf.zeros_like(distributions[0].var)
    
    for i, dist in enumerate(distributions):
        weighted_mu += weights[i] * dist.mu
        # Add variance plus squared difference from mixture mean
        weighted_var += weights[i] * (dist.var + tf.square(dist.mu))
    
    # Correct the variance by subtracting squared mixture mean
    weighted_var -= tf.square(weighted_mu)
    
    # Create a new distribution with these properties
    weighted_logvar = tf.math.log(tf.maximum(weighted_var, 1e-8))
    return GaussianDistribution(weighted_mu, weighted_logvar)

def compute_js_divergence(mus, logvars):
    """
    Compute JS divergence among a set of Gaussian distributions.
    
    Args:
        mus: List of mean vectors [tensor1, tensor2, ...], each with shape (batch, latent_dim)
        logvars: List of log variance vectors, matching mus
        
    Returns:
        JS divergence (scalar)
    """
    M = len(mus)
    if M <= 1:
        return tf.constant(0.0, dtype=tf.float32)
    
    # Create distributions
    distributions = [GaussianDistribution(mu, logvar) for mu, logvar in zip(mus, logvars)]
    
    # Create the mixture distribution (uniform weighting)
    mixture = compute_mixture_distribution(distributions)
    
    # Compute KL(dist_i || mixture) for each component
    kl_terms = []
    for dist in distributions:
        kl_i = dist.kl_divergence(mixture)
        kl_terms.append(kl_i)
    
    # Average the KL terms
    kl_stack = tf.stack(kl_terms, axis=0)
    kl_mean = tf.reduce_mean(kl_stack)
    
    # JS = (1/M) * sum(KL(p_i || mixture))
    return kl_mean

def reparameterize(mu, logvar):
    """
    Reparameterization trick for sampling from a Gaussian.
    
    Args:
        mu: Mean vector
        logvar: Log variance vector
        
    Returns:
        Sampled vector
    """
    sigma = tf.exp(0.5 * logvar)
    eps = tf.random.normal(tf.shape(sigma))
    return mu + eps * sigma