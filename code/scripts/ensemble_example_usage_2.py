"""This script demonstrates some of the functionality provided in ensemble.py.

Note: in this example, it is assumed scripts/train_ensemble.py has been run,
with ENSEMBLE_SAVE_NAME = 'basic_cnn' and DATASET_NAME = 'cifar10'
"""

ENSEMBLE_SAVE_NAME = 'basic_cnn'  # Name that the ensemble models will be saved with
DATASET_NAME = 'cifar10'  # Name of dataset models were trained with

# Add parent dir to path to allow for parallel imports
import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

import models.ensemble
from utils import datasets
from utils import saveload


# Need these settings for GPU to work on my computer /Einar
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Example usage
if __name__ == "__main__":
    # Fetch and wrap previously trained ensemble models
    ensemble_model_names = saveload.get_ensemble_model_names()
    model_names = ensemble_model_names[ENSEMBLE_SAVE_NAME][DATASET_NAME]
    wrapped_models = [models.ensemble.KerasLoadsWhole(name) for name in model_names]

    # Build ensemble
    ensemble = models.ensemble.Ensemble(wrapped_models)

    # Load data
    (train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(DATASET_NAME)

    # Predict with ensemble
    ensemble_preds = ensemble.predict(test_images)
    print("Ensemble preds shape: {}".format(ensemble_preds.shape))
