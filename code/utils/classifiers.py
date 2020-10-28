"""Provides some wrapper classes for creating a common classifier interface."""

import numpy as np
import tensorflow as tf


class EnsembleClassifier:
    """Wraps an ensemble model predicting a list of logits."""

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


class EnsembleClassifierOOD:
    """Wraps an ensemble model predicting a list of logits for OOD evaluation."""

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        ensemble_logits = self.predict_logits(x)
        ensemble_probs = []
        for logits in ensemble_logits:
            ensemble_probs.append(tf.nn.softmax(logits))
        return np.stack(ensemble_probs, axis=0)

    def predict_logits(self, x):
        return self.model.predict(x)



class IndividualClassifier:
    """Wraps a model predicting logits."""

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        logits = self.predict_logits(x)
        probs = np.array(tf.nn.softmax(logits))
        return probs

    def predict_logits(self, x):
        return self.model.predict(x)


class PriorNetClassifier:
    """Wraps a model predicting logits."""

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        logits = self.predict_logits(x)
        return logits

    def predict_logits(self, x):
        return self.model.predict(x)
