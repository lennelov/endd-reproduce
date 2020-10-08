"""Script for training a simple CNN model on CIFAR10.

This script will probably be generalized to a unified
training script."""

MODEL_TYPE = 'cnn'
MODEL_SAVE_NAME = 'basic_cnn'
DATASET_NAME = 'cifar10'
N_MODELS = 10
N_EPOCHS = 15

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
train_labels = tf.one_hot(train_labels.reshape((-1,)), 10)
test_labels = tf.one_hot(test_labels.reshape((-1,)), 10)

# Get model module (python file with get_model function)
model_module = settings.MODEL_MODULES[MODEL_TYPE]

# Get model
for i in range(N_MODELS):
    print("Training model {}...".format(i))
    # Get model
    model = model_module.get_model(dataset_name=DATASET_NAME, compile=True)

    # Train model
    model.fit(train_images, train_labels, epochs=N_EPOCHS, validation_data=(test_images, test_labels))
    print("Model {} finished training.")
    # Save model
    model_name = "{}_{}_{}".format(MODEL_SAVE_NAME, DATASET_NAME, i)
    saveload.save_tf_model(model, model_name)
