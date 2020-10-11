import sys
import os
sys.path.append("/home/lennelov/Repositories/endd-reproduce/code")

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sklearn.metrics

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import settings
from code.utils import saveload
from code.utils import datasets
from code.models import ensemble as ensembles


def probs_to_classes(probs):
    return np.argmax(probs, axis=-1)


def calc_err(probs, labels):
    """Return mean classification error (ERR)."""
    preds = probs_to_classes(probs)
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
    x_rnd = [0, 1]
    y_rnd = [err, 0]
    return x_rnd, y_rnd


def _calc_orc_curve(err):
    x_orc = [0, err]
    y_orc = [err, 0]
    return x_orc, y_orc


def _calc_uns_curve(probs, labels, n_threshold_steps=10):
    preds = probs_to_classes(probs)
    max_probs = probs.max(axis=1)

    errs = []
    rejection_rates = []
    thresholds = np.linspace(0, 1, n_threshold_steps, endpoint=True)
    for threshold in thresholds:
        rejection_rates.append(_calc_rejection_rate(max_probs, threshold))
        errs.append(_calc_not_rejected_err(max_probs, preds, labels, threshold))
    return rejection_rates, errs

def _calc_rejection_rate(max_probs, threshold):
    n_total = len(max_probs)
    n_rejected = np.sum((max_probs > threshold) == False)
    return n_rejected / n_total


def _calc_not_rejected_err(max_probs, preds, labels, threshold):
    not_rejected = max_probs > threshold
    if not not_rejected.any():

        return 0
    preds_subset = preds[not_rejected]
    labels_subset = labels[not_rejected]
    err_subset = 1 - sklearn.metrics.accuracy_score(labels_subset, preds_subset)
    return err_subset


def _calc_auc_difference(x_1, y_1, x_2, y_2):
    auc_1 = _calc_auc(x_1, y_1)
    auc_2 = _calc_auc(x_2, y_2)
    return auc_1 - auc_2


def _calc_auc(x, y):
    """Return area under curve."""
    return sklearn.metrics.auc(x, y)


def calc_ece(probs, labels):
    """Return expected calibration error (ECE)."""
    tfp.stats.expected_calibration_error(10, )


def calc_nll(probs, labels):
    """Return negative log-likelihood (NLL)."""
    return sklearn.metrics.log_loss(labels, probs)


class EnsembleClassifier:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        ensemble_logits = self.predict_logits(x)
        ensemble_probs = []
        for logits in ensemble_logits:
            ensemble_probs.append(tf.nn.softmax(logits))
        mean_probs = np.array(ensemble_probs).mean(axis=0)
        return mean_probs

    def predict_logits(self, x):
        return self.model.predict(x)



class IndividualClassifier:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        logits = self.predict_logits(x)
        probs = np.array(tf.nn.softmax(logits))
        return probs

    def predict_logits(self, x):
        return self.model.predict(x)



MODEL_NAME = 'basic_cnn'
DATASET_NAME = 'cifar10'


# Prepare classifiers
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[MODEL_NAME][DATASET_NAME][:3]
models = [ensembles.KerasLoadsWhole(name) for name in model_names]
ensemble = ensembles.Ensemble(models)
ensm = EnsembleClassifier(ensemble)
ind = IndividualClassifier(ensemble.get_model(model_names[0]).get_model())

# Load data
_, (images, labels) = datasets.get_dataset(DATASET_NAME)

# Preprocess data
labels = labels.reshape(-1)

# Make predictions
probs_ind = ind.predict(images)
probs_ensm = ensm.predict(images)

# Compute ERR
err_ind = calc_err(probs_ind, labels)
err_ensm = calc_err(probs_ensm, labels)

# Compute PRR
prr_ind = calc_prr(probs_ind, labels)
prr_ensm = calc_prr(probs_ensm, labels)

# Compute ECE
# TODO

# Compute NLL
nll_ind = calc_nll(probs_ind, labels)
nll_ensm = calc_nll(probs_ensm, labels)
