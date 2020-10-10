import tensorflow as tf
from tensorflow.keras import layers, models
import sys
import pathlib
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
from settings_prior import *
from settings import *
from utils.DirichletKL import DirichletKL


def get_model(dataset_name, n_classes, compile=True, weights=None):
    """Take dataset name and return corresponding untrained CNN model.
	    Args:
		dataset_name (str): Name of the dataset that the model will be used on,
		                    must be listed in settings.py.
		compile (bool): If False, an uncompiled model is returned. Default is True.
		weights (str): Name of saved weights. If provided, returned model will
		               be loaded with saved weights. Default is None.
	    Returns:
		keras Model object
	    If compile=True, model will be compiled with adam optimizer, categorical cross
	    entropy loss, and accuracy metric.
	"""
    if dataset_name not in DATASET_NAMES:
        raise ValueError("""Dataset {} not recognized, please make sure it has been listed in
                            settings.py""".format(dataset_name))

    input_shape = DATASET_INPUT_SHAPES[dataset_name]
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_classes, activation='exponential'))

    if weights:
        saveload.load_weights(model, weights)

    if not compile:
        return model

    KL = DirichletKL()
    model.compile(optimizer='adam', loss=KL)

    return model
