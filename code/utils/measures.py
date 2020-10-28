"""Module with functions to compute some measures."""
import tensorflow as tf
import numpy as np
import sklearn.metrics
from uncertainty_metrics.numpy.general_calibration_error import ece
import scipy.special
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def entropy_of_expected(raw, logits):
    '''Calcs entropy of expected <-> total uncertainty

    args:
    raw - A (N_models, N_data_points, N_classes) vector or (N_data_points, N_classes)
    logits - Boolean, if true, we take softmax, otherwise, treat as probabilities.

    return:
    A N_data_points vector'''

    if logits:
        probabilities = scipy.special.softmax(raw, axis=-1)
    else:
        probabilities = raw

    if len(probabilities.shape) == 3:
        means = np.mean(probabilities, axis=0)

    elif len(probabilities.shape) == 2:
        means = probabilities
    else:
        raise ValueError(
            'Logits must be (N_models, N_data_points, N_classes) or (N_data_points, N_classes)')

    return np.sum(-means * np.log(means + 1e-12), axis=1)


def expected_entropy(raw, logits):
    '''Calcs expected entropy <-> data uncertainty

    args:
    raw - A (N_models, N_data_points, N_classes) vector
    logits - Boolean, if true, we take softmax, otherwise, treat as probabilities.

    return:
    A N_data_points vector'''

    probabilities = 0
    if logits:
        probabilities = scipy.special.softmax(raw, axis=-1)
    else:
        probabilities = raw

    return np.mean(np.sum(-probabilities * np.log(probabilities + 1e-12), axis=2), axis=0)


def expected_entropy_pn(logits):
    """ Calculated expected entropy (data uncertainty) for a prior network.
    Assumes dirichlet distribution.

    Args:
        logits - A (N_data_points, N_classes) - vector

    Outputs:
        A (N_data_points, N_classes) - vector
    """
    alpha = np.exp(logits)
    alpha_0 = np.sum(alpha, axis=1, keepdims=True)
    probs = alpha / alpha_0


    return np.sum(-probs * (scipy.special.digamma(alpha + 1) - scipy.special.digamma(alpha_0 + 1)),
                  axis=1)


def calc_tot_unc_auc_roc(in_probs, out_probs, plot=False):
    in_labels = np.zeros(in_probs.shape[0])
    out_labels = np.ones(out_probs.shape[0])
    all_labels = np.concatenate([in_labels, out_labels], axis=0)

    in_t_unc = entropy_of_expected(in_probs, logits=False)
    out_t_unc = entropy_of_expected(out_probs, logits=False)
    all_t_unc = np.concatenate([in_t_unc, out_t_unc], axis=0)

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(all_labels, all_t_unc)
    auc_roc = sklearn.metrics.auc(recall, precision)

    if plot:
        plt.plot(recall, precision, label='Total uncertainty')
        plt.title('Total uncertainty OOD detection ROC curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()

    return auc_roc


def calc_ensemble_know_unc_auc_roc(in_probs, out_probs, plot=False):
    in_labels = np.zeros(in_probs.shape[1])
    out_labels = np.ones(out_probs.shape[1])
    all_labels = np.concatenate([in_labels, out_labels], axis=0)

    in_t_unc = entropy_of_expected(in_probs, logits=False)
    out_t_unc = entropy_of_expected(out_probs, logits=False)

    in_d_unc = expected_entropy(in_probs, logits=False)
    out_d_unc = expected_entropy(out_probs, logits=False)

    in_k_unc = in_t_unc - in_d_unc
    out_k_unc = out_t_unc - out_d_unc
    all_k_unc = np.concatenate([in_k_unc, out_k_unc], axis=0)

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(all_labels, all_k_unc)
    auc_roc = sklearn.metrics.auc(recall, precision)

    if plot:
        plt.plot(recall, precision, label='Knowledge uncertainty')
        plt.title('Knowledge uncertainty OOD detection ROC curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()
        auc_roc = sklearn.metrics.auc(recall, precision)

    return auc_roc


def calc_pn_know_unc_auc_roc(in_preds, out_preds, plot=False):
    in_labels = np.zeros(in_preds.shape[0])
    out_labels = np.ones(out_preds.shape[0])
    all_labels = np.concatenate([in_labels, out_labels], axis=0)

    in_t_unc = entropy_of_expected(in_preds, logits=False)
    out_t_unc = entropy_of_expected(out_preds, logits=False)

    in_d_unc = expected_entropy_pn(in_preds)
    out_d_unc = expected_entropy_pn(out_preds)

    in_k_unc = in_t_unc - in_d_unc
    out_k_unc = out_t_unc - out_d_unc
    all_k_unc = np.concatenate([in_k_unc, out_k_unc], axis=0)

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(all_labels, all_k_unc)
    auc_roc = sklearn.metrics.auc(recall, precision)

    if plot:
        plt.plot(recall, precision, label='Knowledge uncertainty')
        plt.title('Knowledge uncertainty OOD detection ROC curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()
        auc_roc = sklearn.metrics.auc(recall, precision)

    return auc_roc


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
