import sys
import os
# parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(parent_dir_path)
sys.path.append("/home/lennelov/Repositories/endd-reproduce/code")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from collections import defaultdict
from utils import evaluation, datasets, saveload, preprocessing
from models import ensemble, endd

# Model loading parameters
N_MODELS_BASE_NAMES = [
    'cifar10_vgg_endd_aux_mini_1',
    'cifar10_vgg_endd_aux_mini_2'
]
# Should be set to the same configuration as when running ensemble_size_ablation_study.py
ENDD_AUX_BASE_MODEL = 'vgg'
ENSEMBLE_LOAD_NAME = 'vgg'
N_MODELS_LIST = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30]

# Dataset parameters
DATASET_NAME = 'cifar10'
NORMALIZATION = '-1to1'

def get_dataset(dataset_name, normalization):
    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(dataset_name)

    # Normalize data
    if normalization == "-1to1":
        train_images, min, max = preprocessing.normalize_minus_one_to_one(train_images)
        test_images = preprocessing.normalize_minus_one_to_one(test_images, min, max)
    elif normalization == 'gaussian':
        train_images, mean, std = preprocessing.normalize_gaussian(train_images)
        test_images = preprocessing.normalize_gaussian(test_images, mean, std)

    return (train_images, train_labels), (test_images, test_labels)

# Get ensemble measures
def get_ensm_measures(model_names, n_models_list, test_images, test_labels):
    ensm_measures = defaultdict(list)
    for n_models in n_models_list:
        model_name_subset = model_names[:n_models]
        wrapped_models = [ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_name_subset]
        ensm_model = ensemble.Ensemble(wrapped_models)
        evaluation_result = evaluation.calc_classification_measures(ensm_model,
                                                                    test_images,
                                                                    test_labels,
                                                                    wrapper_type='ensemble')
        for measure, value in evaluation_result.items():
            ensm_measures[measure].append(value)
    return ensm_measures

# Get ENDD measures
def get_endd_measures(n_models_base_names, n_models_list, endd_base_model, dataset_name,
                      test_images, test_labels):
    endd_measures_list = []
    for base_name in n_models_base_names:
        endd_measures = defaultdict(list)
        for n_models in N_MODELS_LIST:
            endd_model_name = base_name + '_N_MODELS={}'.format(n_models)
            uncompiled_model = saveload.load_tf_model(endd_model_name, compile=False)
            endd_model = endd.get_model(uncompiled_model,
                                        dataset_name=dataset_name,
                                        compile=True)

            evaluation_result = evaluation.calc_classification_measures(endd_model,
                                                                        test_images,
                                                                        test_labels,
                                                                        wrapper_type='individual')
            for measure, value in evaluation_result.items():
                endd_measures[measure].append(value)
        endd_measures_list.append(endd_measures)
    return endd_measures_list


def plot_with_error_fields(ensm_measures, endd_measures_list, measure, ylabel):
    stack = np.stack([endd_measures[measure] for endd_measures in endd_measures_list])
    means = stack.mean(axis=0)
    stds = stack.std(axis=0)
    plt.plot(N_MODELS_LIST, ensm_measures[measure], label='ENSM', color='xkcd:dull blue', linestyle='dashed')
    plt.plot(N_MODELS_LIST, means, label='ENDD+AUX', color='xkcd:dusty orange')
    plt.fill_between(N_MODELS_LIST, means-2*stds, means+2*stds, color='xkcd:dusty orange', alpha=0.4)
    plt.xlabel("Number of models")
    plt.ylabel(ylabel)
    plt.legend()


# Load ensemble model names
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSEMBLE_LOAD_NAME][DATASET_NAME]

# Get dataset
_, (test_images, test_labels) = get_dataset(DATASET_NAME, NORMALIZATION)

# Get ENSM measures
ensm_measures = get_ensm_measures(model_names, N_MODELS_LIST, test_images, test_labels)

# Get ENDD measures
endd_measures_list = get_endd_measures(N_MODELS_BASE_NAMES, N_MODELS_LIST, ENDD_AUX_BASE_MODEL,
                                       DATASET_NAME, test_images, test_labels)

print(ensm_measures)
print(endd_measures_list)

# Plot results
plt.style.use('ggplot')

plt.subplot(2, 2, 1)
plot_with_error_fields(ensm_measures, endd_measures_list, 'err', 'Prediction Error')

plt.subplot(2, 2, 2)
plot_with_error_fields(ensm_measures, endd_measures_list, 'nll', 'Negative Log-Likelihood')

plt.subplot(2, 2, 3)
plot_with_error_fields(ensm_measures, endd_measures_list, 'ece', 'Expected Calibration Error')

plt.subplot(2, 2, 4)
plot_with_error_fields(ensm_measures, endd_measures_list, 'prr', 'Prediction Rejection Rate')

plt.show()
