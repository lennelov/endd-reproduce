"""Script for training a simple CNN model on CIFAR10.

This script will probably be generalized to a unified
training script."""

DATASET_NAME = 'cifar10'
N_EPOCHS = 10

# Add parent dir to path to allow for parallel imports
import sys
import pathlib
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

# Imports
import tensorflow as tf
import tensorflow.keras as keras
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

# Get model
model = cnn.get_model(dataset_name=DATASET_NAME, compile=True)

# Train
model.fit(train_images, train_labels, epochs=N_EPOCHS, validation_data=(test_images, test_labels))

# Save weights
saveload.save_tf_model(model, "cnn")
