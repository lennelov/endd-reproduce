"""Module containing classes used for building ensembles.

The functionality provided by this module is demonstrated in scripts.ensemble_example_usage.py."""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from utils import saveload
from abc import ABC, abstractmethod


class ModelWrapper(ABC):
    """Base class for model wrappers.

    Subclasses of this class can be used to wrap models of various types,
    implementing logic that allows for handling a model out-of-memory.

    The primary purpose is to wrap models that can then be used by the Ensemble
    class.
    """

    def __init__(self, name=None, type=None):
        self.type = type
        self.name = name

    @abstractmethod
    def predict(self, x):
        """Load model and return predicted values."""
        pass

    @abstractmethod
    def get_model(self):
        """Load model into memory and return."""
        pass


class KerasLoadsWhole(ModelWrapper):

    def __init__(self, model_load_name, name=None):
        """Wrap a saved Keras model.

        This subclass of ModelWrapper should be used with Keras models
        that have been saved using utils.saveload.save_tf_model. If
        only weights have been saved, KerasLoadsWeights (not implemented)
        should be used instead.

        Args:
            model_load_name (str): The name used when saving the model using
                                   utils.saveload.save_tf_model.
            name (str): Reference name of the model wrapper (used when retrieving
                        models from an Ensemble). If no name is given, model_load_name
                        is used by default.
        """
        if not name:
            name = model_load_name
        super().__init__(name=name, type="keras")
        self.model_load_name = model_load_name

    def get_model(self):
        """Return the loaded Keras model."""
        model = saveload.load_tf_model(self.model_load_name)
        return model

    def predict(self, x):
        """Take inputs x and return predictions from all models in the ensemble.

        Args:
            x (np.Array): input data of shape (N, M), where N is the number of samples
                          and M is the number of features in each sample.

        Returns:
            (np.Array): predictions of shape (N, K) where N is the number of input samples,
                        K is the number of output features.
        """
        try:
            model = self.get_model()
            y = model.predict(x)
            return y
        finally:
            keras.backend.clear_session()


class Ensemble:
    """Class representing an ensemble of models.

    The ensemble contains models wrapped using a subclass of the ModelWrapper
    superclass. Evaluation is done by sequentially loading the models into
    memory. This allows evaluation to take place without needing all models
    in memory at the same time.
    """

    def __init__(self, models):
        """Constructor.

        Args:
            models (List[ModelWrapper]): list of ModelWrappers wrapping the models of the ensemble
        """
        self.models = models

    def add_model(self, model):
        """Add model to list of models.

        Args:
            model (ModelWrapper): Wrapped model to be added.
        """
        self.models.append(model)

    def get_model(self, name):
        """Return model with given name.

        If multiple models with the same name exists, only the first one is returned.

        Args:
            name (str): Name of model.

        Returns:
            (ModelWrapper) with name attribute equal to given name.
        """
        for model in self.models:
            if model.name == name:
                return model

    def get_models(self):
        """Return the list of model wrappers."""
        return self.models

    def predict(self, x):
        """Take inputs x and return predictions from all models in the ensemble.

        Args:
            x (np.Array): input data of shape (N, M), where N is the number of samples
                          and M is the number of features in each sample.

        Returns:
            (np.Array)  predictions of shape (P, N, K) where N is the number of input samples,
                        K is the number of output features, and P is the number of models in
                        the ensemble.
        """
        preds = []
        for model in self.models:
            preds.append(model.predict(x))
        return np.stack(preds)

    def __str__(self):
        """Return string representation of the ensemble."""
        s = "Ensemble consisting of {} models:\n".format(len(self.models))
        for i, model in enumerate(self.models):
            s += "    {}. {} - {}\n".format(i, model.name, model.type)
        return s
