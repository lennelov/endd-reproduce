import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Imports
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from models import endd, ensemble
from utils import saveload, training, evaluation, datasets, preprocessing

plt.style.use('ggplot')

# Set names for loading and saving
ENSEMBLE_LOAD_NAME = 'vgg'  # Name of ensemble to use for training
DATASET_NAME = 'cifar10'  # Name of dataset to use (ensemble must be trained on this dataset)
AUX_DATASET_NAME = 'cifar100'  # Name of auxiliary dataset to use (None if no AUX data)
MODEL_BASE_SAVE_NAME = 'cifar10_vgg_endd_aux_1'  # Name to use when saving model (None if no saving)
N_MODELS = 100
INIT_TEMP_LIST = [1, 2, 5, 10, 20]

# Set training parameters
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

# Load ensemble models
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSEMBLE_LOAD_NAME][DATASET_NAME]
model_name_subset = model_names[:N_MODELS]
wrapped_models = [ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_name_subset]

load_previous_dataset = False
measures = defaultdict(list)
for init_temp in INIT_TEMP_LIST:
    # Build ensemble
    ensm_model = ensemble.Ensemble(wrapped_models)

    # Train ENDD
    endd_model = training.train_vgg_endd(train_images=train_images,
                                         ensemble_model=ensm_model,
                                         dataset_name=DATASET_NAME,
                                         batch_size=BATCH_SIZE,
                                         n_epochs=N_EPOCHS,
                                         one_cycle_lr_policy=ONE_CYCLE_LR_POLICY,
                                         init_lr=INIT_LR,
                                         cycle_length=CYCLE_LENGTH,
                                         temp_annealing=TEMP_ANNEALING,
                                         init_temp=init_temp,
                                         dropout_rate=DROPOUT_RATE,
                                         save_endd_dataset=(not load_previous_dataset),
                                         load_previous_endd_dataset=load_previous_dataset)
    load_previous_dataset = True  # Load previous to make training faster
    endd_measures = evaluation.calc_classification_measures(endd_model,
                                                            test_images,
                                                            test_labels,
                                                            wrapper_type='individual')
    for measure, value in endd_measures.items():
        measures[measure].append(value)

    if MODEL_BASE_SAVE_NAME:
        saveload.save_tf_model(endd_model, MODEL_BASE_SAVE_NAME + '_TEMP={}'.format(init_temp))
        saveload.save_weights(endd_model, MODEL_BASE_SAVE_NAME + '_TEMP={}'.format(init_temp))

print(measures)

# Plot results
plt.subplot(2, 2, 1)
plt.plot(INIT_TEMP_LIST, measures['err'], label='ENDD+AUX')
plt.xlabel("Initial temperature")
plt.ylabel("Prediction Error")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(INIT_TEMP_LIST, measures['prr'], label='ENDD+AUX')
plt.xlabel("Initial temperature")
plt.ylabel("Prediction Rejection Rate")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(INIT_TEMP_LIST, measures['ece'], label='ENDD+AUX')
plt.xlabel("Initial temperature")
plt.ylabel("Expected Calibration Error")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(INIT_TEMP_LIST, measures['nll'], label='ENDD+AUX')
plt.xlabel("Initial temperature")
plt.ylabel("Negative Log-Likelihood")
plt.legend()

plt.show()
