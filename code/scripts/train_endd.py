'''
Creates and trains a priornet on the EnDD data.
'''
import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Imports
import numpy as np
import tensorflow.keras as keras
import settings

from models import vgg, cnn, endd, ensemble
from utils import evaluation, preprocessing, saveload, simplex, datasets

# Set names for loading and saving
ENSEMBLE_LOAD_NAME = 'vgg'  # Name of ensemble to use for training
DATASET_NAME = 'cifar10'  # Name of dataset to use (ensemble must be trained on this dataset)
MODEL_SAVE_NAME = 'endd_vgg_cifar10'  # Name to use when saving model

# Set training parameters
N_MODELS = 30  # Number of models to include in ensemble
N_EPOCHS = 40  # Number of epochs to train for
BATCH_SIZE = 500  # Batch size
NORMALIZATION = "-1to1"  # Normalization scheme to use {'-1to1', 'gaussian', None}
# WARNING: It is important that normalization matches the normalization used when training
#          the ensemble models.

# Load ensemble models
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSEMBLE_LOAD_NAME][DATASET_NAME]
model_names = model_names[:N_MODELS]
wrapped_models = [ensemble.KerasLoadsWhole(name) for name in model_names]

# Build ensemble
ensemble_model = ensemble.Ensemble(wrapped_models)

# Load ensemble dataset
train_set, test_set = datasets.get_dataset(DATASET_NAME)
train_images, train_labels = train_set
test_images, test_labels = test_set

# Normalize data
if NORMALIZATION == "-1to1":
    train_images, min, max = preprocessing.normalize_minus_one_to_one(train_images)
    test_images = preprocessing.normalize_minus_one_to_one(test_images, min, max)
elif NORMALIZATION == 'gaussian':
    train_images, mean, std = preprocessing.normalize_gaussian(train_images)
    test_images = preprocessing.normalize_gaussian(test_images, mean, std)

# Get ensemble preds
train_ensemble_preds = datasets.get_ensemble_preds(ensemble_model, train_images)
test_ensemble_preds = datasets.get_ensemble_preds(ensemble_model, test_images)

# Build ENDD model
base_model = vgg.get_model(DATASET_NAME, compile=False)
endd_model = endd.get_model(base_model)

# Train ENDD model
endd_model.fit(train_images, train_ensemble_preds, batch_size=BATCH_SIZE, epochs=N_EPOCHS)
if MODEL_SAVE_NAME:
    # Note: There seems to be some difficulties when trying to load whole model with custom loss
    saveload.save_tf_model(endd_model, MODEL_SAVE_NAME)
    saveload.save_weights(endd_model, MODEL_SAVE_NAME)

# # Predict and show outputs
# logits = endd_model.predict(test_images)
# alphas = tf.math.exp(logits)
# predictions = tf.cast(tf.math.argmax(tf.squeeze(logits), axis=1), dtype=tf.float32)
# test_labels_tf = tf.cast(tf.squeeze(test_labels.copy()), dtype=tf.float32)
#
# score = tf.math.reduce_sum(tf.cast(tf.math.equal(predictions, test_labels_tf),
#                                    tf.float32)) / len(test_labels_tf)
# print('alphas for picture 1: ' + str(alphas[0, :]))
# print('alphas for picture 1: ' + str(alphas[1, :]))
# print('alphas for picture 1: ' + str(alphas[2, :]))
# print('mean of 5 ensembles for picture 1: ' +
#       str(tf.math.reduce_mean(tf.nn.softmax(train_ensemble_preds[0, :, :], axis=1), axis=0)))
# print('score: ' + str(score))

# Evaluate
measures = evaluation.calc_classification_measures(endd_model,
                                                   test_images,
                                                   test_labels,
                                                   wrapper_type='individual')

results = evaluation.format_results(['endd'], [measures])
print(results)
