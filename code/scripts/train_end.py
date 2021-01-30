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
import pickle
import numpy as np
import tensorflow.keras as keras
import settings

from models import endd, ensemble
from utils import saveload, training, evaluation, datasets, preprocessing

# Set names for loading and saving
ENSEMBLE_LOAD_NAME = 'vgg'  # Name of ensemble to use for training
DATASET_NAME = 'cifar10'  # Name of dataset to use (ensemble must be trained on this dataset)
AUX_DATASET_NAME = 'cifar100'  # Name of auxiliary dataset to use (None if no AUX data)
MODEL_SAVE_NAME = 'end_vgg_cifar10_100_aux_3'  # Name to use when saving model (None if no saving)

# Set training parameters
N_MODELS = 100  # Number of models to include in ensemble, set to None if all should be included
N_EPOCHS = 90  # Number of epochs to train for (90)
BATCH_SIZE = 128  # Batch size (128)
NORMALIZATION = "-1to1"  # Normalization scheme to use {'-1to1', 'gaussian', None} ('-1to1')
# WARNING: It is important that normalization matches the normalization used when training
#          the ensemble models.

#tf.random.set_seed(345)
#tf.random.set_seed(54323323)
tf.random.set_seed(3444)
TEMP_ANNEALING = False
ONE_CYCLE_LR_POLICY = True
CYCLE_LENGTH = 60  # (60)
INIT_LR = 0.001  # (0.001)
DROPOUT_RATE = 0.3  # (0.3)
INIT_TEMP = 1  # (10)

# Load ensemble models
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSEMBLE_LOAD_NAME][DATASET_NAME]
if N_MODELS is not None:
    model_names = model_names[:N_MODELS]
wrapped_models = [ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_names]

# Build ensemble
ensemble_model = ensemble.Ensemble(wrapped_models)

# Load dataset
(train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(DATASET_NAME)

if AUX_DATASET_NAME:
    (aux_images, _), _ = datasets.get_dataset(AUX_DATASET_NAME)
    train_images = np.concatenate((train_images, aux_images), axis=0)

# Normalize data
if NORMALIZATION == "-1to1":
    train_images, min, max = preprocessing.normalize_minus_one_to_one(train_images)
    test_images = preprocessing.normalize_minus_one_to_one(test_images, min, max)
elif NORMALIZATION == 'gaussian':
    train_images, mean, std = preprocessing.normalize_gaussian(train_images)
    test_images = preprocessing.normalize_gaussian(test_images, mean, std)

end_model = training.train_vgg_end(train_images=train_images,
                                    ensemble_model=ensemble_model,
                                    dataset_name=DATASET_NAME,
                                    batch_size=BATCH_SIZE,
                                    n_epochs=N_EPOCHS,
                                    one_cycle_lr_policy=ONE_CYCLE_LR_POLICY,
                                    init_lr=INIT_LR,
                                    cycle_length=CYCLE_LENGTH,
                                    temp_annealing=TEMP_ANNEALING,
                                    init_temp=INIT_TEMP,
                                    dropout_rate=DROPOUT_RATE,
                                    save_end_dataset=False,
                                    load_previous_end_dataset=True)

measures = evaluation.calc_classification_measures(end_model,
                                                   test_images,
                                                   test_labels,
                                                   wrapper_type='individual')

if MODEL_SAVE_NAME:
    # Note: There seems to be some difficulties when trying to load whole model with custom loss
    saveload.save_tf_model(end_model, MODEL_SAVE_NAME)
    saveload.save_weights(end_model, MODEL_SAVE_NAME)

results = evaluation.format_results(['end'], [measures])
print(results)
