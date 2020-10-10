"""Script for training a VGG16 model on CIFAR10.

This script will probably be generalized to a unified
training script."""

# Add parent dir to path to allow for parallel imports
import sys
import pathlib
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

# Imports
import models.vgg as vgg
import utils.saveload as saveload
from utils.OneCycleLRPolicy import OneCycleLRPolicy
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import datasets
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import datetime

# Need these settings for GPU to work on my computer /Einar
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Preprocess
#train_images = preprocess_input(train_images)
#test_images = preprocess_input(test_images)

normalization = "-1to1"

# Normalize images
if normalization == "gaussian":
    mean = np.mean(train_images)
    std = np.std(train_images)
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

elif normalization == "-1to1":
    train_images = train_images / 127.5
    train_images = train_images - 1.0
    test_images = test_images / 127.5
    test_images = test_images - 1.0

train_labels = tf.one_hot(train_labels.reshape((-1,)), 10)
test_labels = tf.one_hot(test_labels.reshape((-1,)), 10)

# Image augmentation
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                                 horizontal_flip=True,
                                                                 width_shift_range=4,
                                                                 height_shift_range=4,
                                                                 fill_mode='nearest')

# Get model
model = vgg.get_model(dataset_name='cifar10', compile=True)

model.summary()

# Set-up tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train

epochs = 45
init_lr = 0.001
olp_callback = OneCycleLRPolicy(init_lr = init_lr, max_lr = 10*init_lr, min_lr = init_lr/1000, cycle_length = 30, epochs = epochs)
model.fit(x = data_generator.flow(train_images, train_labels, batch_size = 128), 
	epochs = epochs, 
	validation_data = (test_images, test_labels),
	callbacks = [tensorboard_callback, olp_callback])

# Save weights
saveload.save_weights(model, "vgg")
