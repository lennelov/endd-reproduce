"""Script for training a VGG16 model on CIFAR10.

This script will probably be generalized to a unified
training script."""

# Add parent dir to path to allow for parallel imports
import sys
import pathlib
import os
parent_dir_path = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(parent_dir_path)

# Imports
import models.vgg as vgg
import tensorflow as tf
import tensorflow.keras as keras
import category_encoders as ce
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras.applications.vgg16 import preprocess_input

# Need these settings for GPU to work on my computer /Einar
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Preprocess
train_images = preprocess_input(train_images)
test_images = preprocess_input(test_images)
train_labels = tf.one_hot(train_labels.reshape((-1,)), 10)
test_labels = tf.one_hot(test_labels.reshape((-1,)), 10)

# Get model
model = vgg.get_model(dataset_name='cifar10', compile=True)

# Train
model.fit(train_images, train_labels, epochs=10)
