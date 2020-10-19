"""Module with functions to compute some measures."""
import tensorflow as tf
import numpy as np
import sklearn.metrics
from uncertainty_metrics.numpy.general_calibration_error import ece


def _probs_to_classes(probs):
    """Return chosen class as an integer."""
    return np.argmax(probs, axis=-1)


def calc_err(probs, labels):
    """Return mean classification error (ERR)."""
    preds = _probs_to_classes(probs)
    return 1 - sklearn.metrics.accuracy_score(labels, preds)


def calc_prr(probs, labels):
    """Return prediction rejection rate (PRR).

    See Appendix B in (Malinin, 2020) for details.
    """
    # TODO: Get curves
    err = calc_err(probs, labels)
    x_rnd, y_rnd = _calc_rnd_curve(err)
    x_orc, y_orc = _calc_orc_curve(err)
    x_uns, y_uns = _calc_uns_curve(probs, labels, n_threshold_steps=100)
    ar_uns = _calc_auc_difference(x_rnd, y_rnd, x_uns, y_uns)
    ar_orc = _calc_auc_difference(x_rnd, y_rnd, x_orc, y_orc)
    return ar_uns / ar_orc


def _calc_rnd_curve(err):
    """Return baseline curve."""
    x_rnd = [0, 1]
    y_rnd = [err, 0]
    return x_rnd, y_rnd


def _calc_orc_curve(err):
    """Return oracle curve."""
    x_orc = [0, err]
    y_orc = [err, 0]
    return x_orc, y_orc


def _calc_uns_curve(probs, labels, n_threshold_steps=10):
    """Return uncertainty curve."""
    preds = _probs_to_classes(probs)
    max_probs = probs.max(axis=1)

    errs = []
    rejection_rates = []
    thresholds = np.linspace(0, 1, n_threshold_steps, endpoint=True)
    for threshold in thresholds:
        rejection_rates.append(_calc_rejection_rate(max_probs, threshold))
        errs.append(_calc_not_rejected_err(max_probs, preds, labels, threshold))
    return rejection_rates, errs


def _calc_rejection_rate(max_probs, threshold):
    """Return rejection rate."""
    n_total = len(max_probs)
    n_rejected = np.sum((max_probs > threshold) == False)
    return n_rejected / n_total


def _calc_not_rejected_err(max_probs, preds, labels, threshold):
    """Return classification error for non-rejected samples."""
    not_rejected = max_probs > threshold
    if not not_rejected.any():

        return 0
    preds_subset = preds[not_rejected]
    labels_subset = labels[not_rejected]
    err_subset = 1 - sklearn.metrics.accuracy_score(labels_subset, preds_subset)
    return err_subset


def _calc_auc_difference(x_1, y_1, x_2, y_2):
    """Return difference in area under two curves."""
    auc_1 = _calc_auc(x_1, y_1)
    auc_2 = _calc_auc(x_2, y_2)
    return auc_1 - auc_2


def _calc_auc(x, y):
    """Return area under curve."""
    return sklearn.metrics.auc(x, y)


def calc_ece(probs, labels, n_bins=100):
    """Return expected calibration error (ECE).

    Uses the uncertainty_metrics library found at
    https://github.com/google/uncertainty-metrics/.
    """
    return ece(labels, probs, num_bins=n_bins)


def calc_nll(probs, labels):
    """Return negative log-likelihood (NLL)."""
    return sklearn.metrics.log_loss(labels, probs)
