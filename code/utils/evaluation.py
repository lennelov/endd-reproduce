import sys
import os
sys.path.append("/home/lennelov/Repositories/endd-reproduce/code")

import tensorflow as tf
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
    # TODO: Implement


def calc_ece(probs, labels):
    """Return expected calibration error (ECE)."""
    # TODO: Implement


def calc_nll(probs, labels):
    """Return negative log-likelihood (NLL)."""
    return sklearn.metrics.log_loss(labels, probs)
    # TODO: Implement


class EnsembleClassifier:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        ensemble_preds = self.model.predict(x)
        mean_preds = ensemble_preds.mean(axis=0)
        return mean_preds




MODEL_NAME = 'basic_cnn'
DATASET_NAME = 'cifar10'


# Prepare classifiers
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[MODEL_NAME][DATASET_NAME][:3]
models = [ensembles.KerasLoadsWhole(name) for name in model_names]
ensemble = ensembles.Ensemble(models)
ensm = EnsembleClassifier(ensemble)
ind = ensemble.get_model(model_names[0]).get_model()

# Load data
_, (images, labels) = datasets.get_dataset(DATASET_NAME)

# Preprocess data
labels = labels.reshape(-1)

# Make predictions
probs_ensm = ensm.predict(images)
probs_ind = ind.predict(images)

# Compute ERR
err_ensm = calc_err(probs_ensm, labels)
err_ind = calc_err(probs_ind, labels)

# Compute PRR
# TODO

# Compute ECE
# TODO

# Compute NLL
nll_ensm = calc_nll(probs_ensm, labels)
nll_ind = calc_nll(probs_ind, labels)
