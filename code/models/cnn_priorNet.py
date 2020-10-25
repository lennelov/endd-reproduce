from utils import losses, saveload
from models import cnn, vgg
import tensorflow.keras as keras


def get_model(base_model,
              dataset_name=False,
              compile=True,
              weights=None,
              dropout_rate=0.3,
              epsilon = 1e-8,
              softmax = False):
    """Take an uncompiled model and return model compiled for PN.
    Warning: This function works in place. Model is returned only for
    conveniance.
    """
    if isinstance(base_model, str):
        if not dataset_name:
            raise ValueError('dataset_name must be provided if base_model is given by name.')
        if base_model == 'cnn':
            base_model = cnn.get_model(dataset_name, compile=False, softmax=False)
        elif base_model == 'vgg':
            base_model = vgg.get_model(dataset_name, compile=False, softmax=False,dropout_rate=dropout_rate)
        else:
            raise ValueError("""Base model {} not recognized, make sure it has been added
                              to endd.py, or pass a Keras model object as base model instead.""")

    if weights:
        saveload.load_weights(base_model, weights)

    if not compile:
        return base_model

    base_model.compile(optimizer='adam', loss=losses.DirichletKL())

    return base_model
