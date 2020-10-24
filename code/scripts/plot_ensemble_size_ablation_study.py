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
ENDD_AUX_BASE_MODEL = 'vgg'
N_MODELS_BASE_NAMES = [
    'cifar10_vgg_endd_aux_mini_1',
    # 'cifar10_vgg_endd_aux_mini_2'
]
ENSEMBLE_LOAD_NAME = 'vgg'
N_MODELS_LIST = [1, 2]

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
            endd_model = endd.get_model(endd_base_model,
                                        dataset_name=dataset_name,
                                        compile=True,
                                        weights=endd_model_name)

            evaluation_result = evaluation.calc_classification_measures(endd_model,
                                                                        test_images,
                                                                        test_labels,
                                                                        wrapper_type='individual')
            for measure, value in evaluation_result.items():
                endd_measures[measure].append(value)
        endd_measures_list.append(endd_measures)
    return endd_measures_list


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

endd_errs = np.stack(endd_measures['err'] for endd_measures in endd_measures_list)
endd_err_means = endd_errs.mean(axis=0)
endd_err_std = endd_errs.std(axis=0)
plt.subplot(2, 2, 1)
plt.errorbar(N_MODELS_LIST, endd_err_means, 2*endd_err_std, label='ENDD+AUX')
plt.plot(N_MODELS_LIST, ensm_measures['err'], label='ENSM')
plt.xlabel("Number of models")
plt.ylabel("Prediction Error")
plt.legend()

endd_prrs = np.stack(endd_measures['prr'] for endd_measures in endd_measures_list)
endd_prr_means = endd_prrs.mean(axis=0)
endd_prr_std = endd_prrs.std(axis=0)
plt.subplot(2, 2, 2)
plt.errorbar(N_MODELS_LIST, endd_prr_means, 2*endd_prr_std, label='ENDD+AUX')
plt.plot(N_MODELS_LIST, ensm_measures['prr'], label='ENSM')
plt.xlabel("Number of models")
plt.ylabel("Prediction Rejection Rate")
plt.legend()

endd_eces = np.stack(endd_measures['ece'] for endd_measures in endd_measures_list)
endd_ece_means = endd_eces.mean(axis=0)
endd_ece_std = endd_eces.std(axis=0)
plt.subplot(2, 2, 3)
plt.errorbar(N_MODELS_LIST, endd_ece_means, 2*endd_ece_std, label='ENDD+AUX')
plt.plot(N_MODELS_LIST, ensm_measures['ece'], label='ENSM')
plt.xlabel("Number of models")
plt.ylabel("Expected Calibration Error")
plt.legend()

endd_nlls = np.stack(endd_measures['nll'] for endd_measures in endd_measures_list)
endd_nll_means = endd_nlls.mean(axis=0)
endd_nll_std = endd_nlls.std(axis=0)
plt.subplot(2, 2, 4)
plt.errorbar(N_MODELS_LIST, endd_nll_means, 2*endd_nll_std, label='ENDD+AUX')
plt.plot(N_MODELS_LIST, ensm_measures['nll'], label='ENSM')
plt.xlabel("Number of models")
plt.ylabel("Negative Log-Likelihood")
plt.legend()

plt.show()
