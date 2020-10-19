'''
Creates and trains a priornet on the EnDD data.
'''
import sys
import os
# parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(parent_dir_path)
sys.path.append("/home/lennelov/Repositories/endd-reproduce/code")

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Imports
import pickle
import numpy as np
import tensorflow.keras as keras
import settings

from models import vgg, cnn, endd, ensemble
from utils import evaluation, preprocessing, saveload, simplex, datasets, callbacks

# Set names for loading and saving
ENSEMBLE_LOAD_NAME = 'vgg'  # Name of ensemble to use for training
DATASET_NAME = 'cifar10'  # Name of dataset to use (ensemble must be trained on this dataset)
MODEL_SAVE_NAME = 'endd_vgg_cifar10_extended'  # Name to use when saving model

# Set training parameters
N_MODELS = 30  # Number of models to include in ensemble
N_EPOCHS = 90  # Number of epochs to train for (90)
BATCH_SIZE = 128  # Batch size
NORMALIZATION = "-1to1"  # Normalization scheme to use {'-1to1', 'gaussian', None}
# WARNING: It is important that normalization matches the normalization used when training
#          the ensemble models.

TEMP_ANNEALING = True
ONE_CYCLE_LR_POLICY = True
CYCLE_LENGTH = 60  # (90)
INIT_LR = 0.001  # (0.001)
DROPOUT_RATE = 0.3  # (0.3)
INIT_TEMP = 10  # (10)

# Load ensemble models
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSEMBLE_LOAD_NAME][DATASET_NAME]
model_names = model_names[:N_MODELS]
wrapped_models = [ensemble.KerasLoadsWhole(name) for name in model_names]

# Build ensemble
ensemble_model = ensemble.Ensemble(wrapped_models)

# Load dataset
(train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(DATASET_NAME)

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

# Save / Load pickled data. Generating ensemble preds takes a long time, so saving and
# loading can make testing much more efficient.

with open('train.pkl', 'wb') as file:
    pickle.dump((train_images, train_labels, train_ensemble_preds), file)
with open('test.pkl', 'wb') as file:
    pickle.dump((test_images, test_labels, test_ensemble_preds), file)
# with open('train.pkl', 'rb') as file:
#     train_images, train_labels, train_ensemble_preds = pickle.load(file)
# with open('test.pkl', 'rb') as file:
#     test_images, test_labels, test_ensemble_preds = pickle.load(file)


# Image augmentation
data_generator = preprocessing.make_augmented_generator(
    train_images, train_ensemble_preds, BATCH_SIZE)

# Callbacks
endd_callbacks = []
if ONE_CYCLE_LR_POLICY:
    olp_callback = callbacks.OneCycleLRPolicy(
        init_lr=INIT_LR, max_lr=INIT_LR * 10, min_lr=INIT_LR / 1000,
        cycle_length=CYCLE_LENGTH, epochs=N_EPOCHS)
    endd_callbacks.append(olp_callback)

if TEMP_ANNEALING:
    temp_callback = callbacks.TemperatureAnnealing(
        init_temp=INIT_TEMP, cycle_length=CYCLE_LENGTH, epochs=N_EPOCHS)
    endd_callbacks.append(temp_callback)

if not endd_callbacks:
    endd_callbacks = None

# Build ENDD model
base_model = vgg.get_model(DATASET_NAME, compile=False, dropout_rate=DROPOUT_RATE)
endd_model = endd.get_model(base_model, init_temp=INIT_TEMP, teacher_epsilon=1e-4)

# Train ENDD model
endd_model.fit(data_generator,
               epochs=N_EPOCHS,
               callbacks=endd_callbacks)

if MODEL_SAVE_NAME:
    # Note: There seems to be some difficulties when trying to load whole model with custom loss
    saveload.save_tf_model(endd_model, MODEL_SAVE_NAME)
    saveload.save_weights(endd_model, MODEL_SAVE_NAME)

# Evaluate
measures = evaluation.calc_classification_measures(endd_model,
                                                   test_images,
                                                   test_labels,
                                                   wrapper_type='individual')

results = evaluation.format_results(['endd'], [measures])
print(results)
