"""Example evaluation on individual and ensemble models."""

# Add parent dir to path to allow for parallel imports
import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from utils import evaluation, datasets, saveload, preprocessing
from models import ensemble, endd, cnn_priorNet

# Choose models
IND_MODEL_NAME = 'vgg_cifar10_cifar10_0'
ENSM_MODEL_NAME, ENSM_N_MODELS = 'vgg', 30
ENDD_MODEL_NAME, ENDD_BASE_MODEL = 'endd_vgg_cifar10_extended', 'vgg'
ENDD_AUX_MODEL_NAME, ENDD_AUX_BASE_MODEL = 'endd_vgg_cifar10_aux', 'vgg'
PN_AUX_MODEL_NAME, PN_AUX_BASE_MODEL = 'PN_vgg_cifar10_aux_c', 'vgg'
# Choose dataset
DATASET_NAME = 'cifar10'

# # Prepare IND model
# ind_model = saveload.load_tf_model(IND_MODEL_NAME)
# ind_wrapper_type = 'individual'
#
# # Prepare ENSM model
# ensemble_model_names = saveload.get_ensemble_model_names()
# model_names = ensemble_model_names[ENSM_MODEL_NAME][DATASET_NAME][:ENSM_N_MODELS]
# models = [ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_names]
# ensm_model = ensemble.Ensemble(models)
# ensm_wrapper_type = 'ensemble'
#
# # Prepare END model
# # TODO: Add END model
#
# # Prepare ENDD model
# endd_model = endd.get_model(ENDD_BASE_MODEL,
#                             dataset_name=DATASET_NAME,
#                             compile=True,
#                             weights=ENDD_MODEL_NAME)
# endd_wrapper_type = 'individual'
#
# # Prepare ENDD+AUX model
# endd_aux_model = endd.get_model(ENDD_AUX_BASE_MODEL,
#                                 dataset_name=DATASET_NAME,
#                                 compile=True,
#                                 weights=ENDD_AUX_MODEL_NAME)
# endd_aux_wrapper_type = 'individual'

pn_base_model = saveload.load_tf_model(PN_AUX_MODEL_NAME, compile=False)
pn_aux_model = cnn_priorNet.get_model(pn_base_model,
                                      dataset_name=DATASET_NAME,
                                      compile=True)
pn_aux_wrapper_type = 'individual'


# Load data
_, (test_images, test_labels) = datasets.get_dataset(DATASET_NAME)

# Preprocess data
test_images = preprocessing.normalize_minus_one_to_one(test_images, min=0, max=255)

# # Calculate measures
# print("Evaluating IND...")
# ind_measures = evaluation.calc_classification_measures(ind_model,
#                                                        test_images,
#                                                        test_labels,
#                                                        wrapper_type=ind_wrapper_type)
# print("Evaluating ENSM...")
# ensm_measures = evaluation.calc_classification_measures(ensm_model,
#                                                         test_images,
#                                                         test_labels,
#                                                         wrapper_type=ensm_wrapper_type)
# print("Evaluating ENDD...")
# endd_measures = evaluation.calc_classification_measures(endd_model,
#                                                         test_images,
#                                                         test_labels,
#                                                         wrapper_type=endd_wrapper_type)
# print("Evaluating ENDD+AUX...")
# endd_aux_measures = evaluation.calc_classification_measures(endd_aux_model,
#                                                             test_images,
#                                                             test_labels,
#                                                             wrapper_type=endd_aux_wrapper_type)
print("Evaluating PN+AUX...")
pn_aux_measures = evaluation.calc_classification_measures(pn_aux_model,
                                                          test_images,
                                                          test_labels,
                                                          wrapper_type=pn_aux_wrapper_type)
print("Evaluations complete.")

# # Format and print results
# summary = evaluation.format_results(
#     ['IND', 'ENSM', 'ENDD', 'ENDD+AUX', 'PN_AUX'],
#     [ind_measures, ensm_measures, endd_measures, endd_aux_measures, pn_aux_measures],
#     dataset_name=DATASET_NAME)

# Format and print results
summary = evaluation.format_results(
    ['PN_AUX'],
    [pn_aux_measures],
    dataset_name=DATASET_NAME)

print(summary)
