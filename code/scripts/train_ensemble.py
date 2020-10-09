"""Script for training an ensemble of models.

Instructions for training:
    1. Set global parameters below.
    2. Run script from repository root.
    3. Saved models are now found in SAVED_MODELS_PATH with names
       {ENSEMBLE_SAVE_NAME}_{DATASET_NAME}_{model_number}, and are
       also listed in SAVED_MODELS_PATH/ensemble_model_names.json

Instructions for building ensemble with trained models (see also ensemble_example_usage_2.py):
    1. Fetch record of all ensembles with
           ensemble_model_names = utils.saveload.get_ensemble_model_names()
    2. Get relevant model_names with
           model_names = ensemble_model_names[ENSEMBLE_SAVE_NAME][DATASET_NAME]
    3. Wrap models with
           models = [models.ensemble.KerasLoadsWhole(name) for name in model_names]
    4. Build ensemble of models with
           ensemble = models.ensemble.Ensemble(models)
    5. Make ensemble prediction with
           ensemble.predict(x)
"""

MODEL_TYPE = 'cnn'  # Name of model module {cnn, vgg}  Note: probably won't work with vgg
ENSEMBLE_SAVE_NAME = 'basic_cnn'  # Name that the ensemble models will be saved with
NAME_START_NUMBER = 0  # Start number for model naming (set to 0 unless continuing previous training)
DATASET_NAME = 'cifar10'  # Name of dataset {cifar10, cifar100, mnist}
N_MODELS = 20  # Number of models to train
N_EPOCHS = 15  # Number of epochs to train for

# Add parent dir to path to allow for parallel imports
import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

# Imports
import tensorflow as tf
import settings
from models import cnn
from utils import saveload
from utils import datasets

# Need these settings for GPU to work on my computer /Einar
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load data
(train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(DATASET_NAME)

# Preprocess
train_labels = tf.one_hot(train_labels.reshape((-1,)), settings.DATASET_N_CLASSES[DATASET_NAME])
test_labels = tf.one_hot(test_labels.reshape((-1,)), settings.DATASET_N_CLASSES[DATASET_NAME])

# Get model module (python file with get_model function)
model_module = settings.MODEL_MODULES[MODEL_TYPE]

# Train and save ensemble
# Try-finally construct ensures that the list of trained models in the ensemble is
# update correctly even if script fails before all models have been trained and saved.
try:
    saved_model_names = []
    for i in range(NAME_START_NUMBER, N_MODELS+NAME_START_NUMBER):
        print("Training model {}...".format(i))
        # Get model
        model = model_module.get_model(dataset_name=DATASET_NAME, compile=True)

        # Train model
        model.fit(train_images, train_labels, epochs=N_EPOCHS, validation_data=(test_images, test_labels))
        print("Model {} finished training.".format(i))

        # Save model
        model_name = "{}_{}_{}".format(ENSEMBLE_SAVE_NAME, DATASET_NAME, i)
        saveload.save_tf_model(model, model_name)
        saved_model_names.append(model_name)
finally:
    # Updates ensemble record with list of saved models
    append_model_names = NAME_START_NUMBER > 0
    saveload.update_ensemble_names(ENSEMBLE_SAVE_NAME,
                                   DATASET_NAME,
                                   saved_model_names,
                                   append=append_model_names)
