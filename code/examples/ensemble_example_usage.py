"""This script demonstrates some of the functionality provided in ensemble.py."""

# Add parent dir to path to allow for parallel imports
import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

from models import ensemble
from tensorflow.keras import datasets

# Need these settings for GPU to work on my computer /Einar
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Example usage
if __name__ == "__main__":
    # Note: in below example, it is assumed that there is a trained Keras model
    # saved with saveload.save_model, saved using the name 'cnn'

    # Wrap models
    model1 = ensemble.KerasLoadsWhole(model_load_name="cnn", name="cnn_1")
    model2 = ensemble.KerasLoadsWhole(model_load_name="cnn", name="cnn_2")
    model3 = ensemble.KerasLoadsWhole(model_load_name="cnn", name="cnn_3")
    model4 = ensemble.KerasLoadsWhole(model_load_name="cnn", name="cnn_4")

    # Build ensemble
    cnn_models = [model1, model2, model3, model4]
    cnn_ensemble = ensemble.Ensemble(cnn_models)
    print(cnn_ensemble)

    # Load data
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Predict with ensemble
    ensemble_preds = cnn_ensemble.predict(test_images)
    print("Ensemble preds shape: {}".format(ensemble_preds.shape))

    # Retrieve models from ensemble
    cnn_1 = cnn_ensemble.get_model("cnn_1")
    cnn_1_preds = cnn_1.predict(test_images)
    print("CNN preds shape: {}".format(cnn_1_preds.shape))

    # Check that retrieved model is a ModelWrapper
    assert isinstance(cnn_1, ensemble.ModelWrapper)

    # Check that ensemble predictions consist of individual model predictions
    cnn_2 = cnn_ensemble.get_model("cnn_2")
    cnn_2_preds = cnn_2.predict(test_images)
    assert (ensemble_preds[0, :, :] == cnn_1_preds).all()
    assert (ensemble_preds[1, :, :] == cnn_2_preds).all()
