import sys
import os
sys.path.append("/home/lennelov/Repositories/endd-reproduce/code")

import tensorflow as tf
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import settings

from code.utils import measures
from code.utils import saveload
from code.utils import datasets
from code.models import ensemble as ensembles

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
err_ind = measures.calc_err(probs_ind, labels)
err_ensm = measures.calc_err(probs_ensm, labels)

# Compute PRR
prr_ind = measures.calc_prr(probs_ind, labels)
prr_ensm = measures.calc_prr(probs_ensm, labels)

# Compute ECE
ece_ind = measures.calc_ece(probs_ind, labels)
ece_ensm = measures.calc_ece(probs_ensm, labels)

# Compute NLL
nll_ind = measures.calc_nll(probs_ind, labels)
nll_ensm = measures.calc_nll(probs_ensm, labels)
