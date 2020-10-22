"""Module with functions to compute stuff relevant for Prior Networks"""
import numpy as np

def pn_logits_to_probs(logits):
    """Converts the raw logits of a Prior Network to predictive probabilities.
    Assumes a Dirichlet Distribution.

    Arguments:

    logits: A (N_datapoints * N_classes) vector of raw output

    Returns:
    probabilities: A (N_datapoints * N_classes) vector of probabilites. 
    """
    
    alpha = np.exp(logits)
    alpha_0 = np.sum(alpha, axis = 1, keepdims = True)
    probs = alpha / alpha_0

    return probs