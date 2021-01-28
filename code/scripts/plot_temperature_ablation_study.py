import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from collections import defaultdict
from utils import evaluation, datasets, saveload, preprocessing
from models import ensemble, endd
import pickle

# Model loading parameters
N_MODELS_BASE_NAMES = ['cifar10_vgg_endd_aux_0', 'cifar10_vgg_endd_aux_1', 'cifar10_vgg_endd_aux_2']
# Should be set to the same configuration as when running ensemble_size_ablation_study.py
ENDD_AUX_BASE_MODEL = 'vgg'
INIT_TEMP_LIST = [1, 2, 3, 4, 5, 7.5, 10, 15, 20]
ORIG_TEMP_LIST = [1, 2, 5, 10, 20]

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


# Get ENDD measures
def get_endd_measures(n_models_base_names, temp_list, endd_base_model, dataset_name, test_images,
                      test_labels):
    endd_measures_list = []
    for base_name in n_models_base_names:
        endd_measures = defaultdict(list)
        for temp in temp_list:
            print("{}/{}".format(base_name, temp))
            endd_model_name = base_name + '_TEMP={}'.format(temp)
            uncompiled_model = saveload.load_tf_model(endd_model_name, compile=False)
            endd_model = endd.get_model(uncompiled_model, dataset_name=dataset_name, compile=True)

            evaluation_result = evaluation.calc_classification_measures(endd_model,
                                                                        test_images,
                                                                        test_labels,
                                                                        wrapper_type='individual')
            for measure, value in evaluation_result.items():
                endd_measures[measure].append(value)
        endd_measures_list.append(endd_measures)
    return endd_measures_list


def plot_with_error_fields(init_temp_list, endd_measures_list, measure, ylabel):
    stack = np.stack([endd_measures[measure] for endd_measures in endd_measures_list])
    means = stack.mean(axis=0)
    stds = stack.std(axis=0)
    plt.plot(init_temp_list, means, '.-', label=r'END$^2_{+AUX}$', color='xkcd:dusty orange')
    plt.fill_between(init_temp_list,
                     means - 2 * stds,
                     means + 2 * stds,
                     color='xkcd:dusty orange',
                     alpha=0.4)
    plt.xlabel("Initial temperature")
    plt.ylabel(ylabel)
    plt.legend()


def plot_with_error_fields_paper(init_temp_list, endd_measures_list, measure, ylabel):
    stack = np.stack([endd_measures[measure] for endd_measures in endd_measures_list])
    means = stack.mean(axis=0)
    minimum = stack.min(axis=0)
    maximum = stack.max(axis=0)
    plt.plot(init_temp_list, means, '.-', label=r'END$^2_{+AUX}$ Paper', color='xkcd:dull blue')
    plt.fill_between(init_temp_list, minimum, maximum, color='xkcd:dull blue', alpha=0.4)
    plt.xlabel("Initial temperature")
    plt.ylabel(ylabel)
    plt.legend()


'''
# Get dataset
_, (test_images, test_labels) = get_dataset(DATASET_NAME, NORMALIZATION)

# Get ENDD measures
endd_measures_list = get_endd_measures(N_MODELS_BASE_NAMES, INIT_TEMP_LIST, ENDD_AUX_BASE_MODEL,
                                       DATASET_NAME, test_images, test_labels)

with open("temp_measure.pkl", "wb") as file:
    pickle.dump((endd_measures_list), file)

print("saved")

'''

with open("temp_measure.pkl", "rb") as file:
    endd_measures_list = pickle.load(file)

print(endd_measures_list)

# The paper's values
paper_measures_list = []
paper_measures = defaultdict(list)
paper_measures["err"].append(22.96 / 100)
paper_measures["err"].append(8.67 / 100)
paper_measures["err"].append(6.93 / 100)
paper_measures["err"].append(6.875 / 100)
paper_measures["err"].append(6.875 / 100)

paper_measures["nll"].append(0.7796)
paper_measures["nll"].append(0.2858)
paper_measures["nll"].append(0.2415)
paper_measures["nll"].append(0.2403)
paper_measures["nll"].append(0.23795)

paper_measures["ece"].append(12.21 / 100)
paper_measures["ece"].append(1.53 / 100)
paper_measures["ece"].append(2.23 / 100)
paper_measures["ece"].append(2.28 / 100)
paper_measures["ece"].append(2.41 / 100)

paper_measures["prr"].append(73.73 / 100)
paper_measures["prr"].append(84.12 / 100)
paper_measures["prr"].append(86.09 / 100)
paper_measures["prr"].append(85.71 / 100)
paper_measures["prr"].append(85.93 / 100)
paper_measures_list.append(paper_measures)

paper_measures = defaultdict(list)
paper_measures["err"].append(38.44 / 100)
paper_measures["err"].append(8.67 / 100)
paper_measures["err"].append(6.93 / 100)
paper_measures["err"].append(6.875 / 100)
paper_measures["err"].append(6.875 / 100)

paper_measures["nll"].append(1.2799)
paper_measures["nll"].append(0.2858)
paper_measures["nll"].append(0.2415)
paper_measures["nll"].append(0.2403)
paper_measures["nll"].append(0.23795)

paper_measures["ece"].append(20.27 / 100)
paper_measures["ece"].append(1.53 / 100)
paper_measures["ece"].append(2.23 / 100)
paper_measures["ece"].append(2.28 / 100)
paper_measures["ece"].append(2.41 / 100)

paper_measures["prr"].append(86.42 / 100)
paper_measures["prr"].append(84.61 / 100)
paper_measures["prr"].append(86.09 / 100)
paper_measures["prr"].append(85.71 / 100)
paper_measures["prr"].append(85.93 / 100)
paper_measures_list.append(paper_measures)

paper_measures = defaultdict(list)
paper_measures["err"].append(5.62 / 100)
paper_measures["err"].append(8.67 / 100)
paper_measures["err"].append(6.93 / 100)
paper_measures["err"].append(6.875 / 100)
paper_measures["err"].append(6.875 / 100)

paper_measures["nll"].append(0.2473)
paper_measures["nll"].append(0.2858)
paper_measures["nll"].append(0.2415)
paper_measures["nll"].append(0.2403)
paper_measures["nll"].append(0.23795)

paper_measures["ece"].append(3.64 / 100)
paper_measures["ece"].append(1.53 / 100)
paper_measures["ece"].append(2.23 / 100)
paper_measures["ece"].append(2.28 / 100)
paper_measures["ece"].append(2.41 / 100)

paper_measures["prr"].append(61.31 / 100)
paper_measures["prr"].append(83.51 / 100)
paper_measures["prr"].append(86.09 / 100)
paper_measures["prr"].append(85.71 / 100)
paper_measures["prr"].append(85.93 / 100)
paper_measures_list.append(paper_measures)

print(paper_measures_list)

# Plot results
plt.style.use('ggplot')

plt.subplot(2, 2, 1)
plot_with_error_fields(INIT_TEMP_LIST, endd_measures_list, 'err', 'Prediction Error')
plot_with_error_fields_paper(ORIG_TEMP_LIST, paper_measures_list, 'err', 'Prediction Error')

plt.subplot(2, 2, 2)
plot_with_error_fields(INIT_TEMP_LIST, endd_measures_list, 'nll', 'Negative Log-Likelihood')
plot_with_error_fields_paper(ORIG_TEMP_LIST, paper_measures_list, 'nll', 'Negative Log-Likelihood')

plt.subplot(2, 2, 3)
plot_with_error_fields(INIT_TEMP_LIST, endd_measures_list, 'ece', 'Expected Calibration Error')
plot_with_error_fields_paper(ORIG_TEMP_LIST, paper_measures_list, 'ece',
                             'Expected Calibration Error')

plt.subplot(2, 2, 4)
plot_with_error_fields(INIT_TEMP_LIST, endd_measures_list, 'prr', 'Prediction Rejection Rate')
plot_with_error_fields_paper(ORIG_TEMP_LIST, paper_measures_list, 'prr',
                             'Prediction Rejection Rate')

plt.show()
