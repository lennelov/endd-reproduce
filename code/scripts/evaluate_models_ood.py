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
from models import ensemble, endd

# Choose models
IND_MODEL_NAME = 'vgg_cifar10_cifar10_0'
ENSM_MODEL_NAME, ENSM_N_MODELS = 'vgg', 2
ENDD_MODEL_NAME, ENDD_BASE_MODEL = 'endd_vgg_cifar10_extended', 'vgg'
ENDD_AUX_MODEL_NAME, ENDD_AUX_BASE_MODEL = 'endd_vgg_cifar10_aux', 'vgg'

# Choose dataset
DATASET_NAME = 'cifar10'
OUT_DATASET_NAME = 'lsun'

# # Prepare IND model
# ind_model = saveload.load_tf_model(IND_MODEL_NAME)
# ind_tot_wrapper_type = 'individual'
# ind_know_wrapper_type = None

# Prepare ENSM model
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSM_MODEL_NAME][DATASET_NAME][:ENSM_N_MODELS]
models = [ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_names]
ensm_model = ensemble.Ensemble(models)
ensm_tot_wrapper_type = 'ensemble'
ensm_know_wrapper_type = 'ensemble_ood'

# Prepare END model
# TODO: Add END model

# Prepare ENDD model
endd_model = endd.get_model(ENDD_BASE_MODEL,
                            dataset_name=DATASET_NAME,
                            compile=True,
                            weights=ENDD_MODEL_NAME)
endd_tot_wrapper_type = 'individual'
endd_know_wrapper_type = 'priornet'

# Prepare ENDD+AUX model
endd_aux_model = endd.get_model(ENDD_AUX_BASE_MODEL,
                                dataset_name=DATASET_NAME,
                                compile=True,
                                weights=ENDD_AUX_MODEL_NAME)
endd_aux_tot_wrapper_type = 'individual'
endd_aux_know_wrapper_type = 'priornet'

# Load data
_, (in_images, _) = datasets.get_dataset(DATASET_NAME)
_, out_images = datasets.get_dataset(OUT_DATASET_NAME)

# Preprocess data
in_images = preprocessing.normalize_minus_one_to_one(in_images, min=0, max=255)
out_images = preprocessing.normalize_minus_one_to_one(out_images, min=0, max=255)

# # Calculate measures
# print("Evaluating IND...")
# ind_measures = evaluation.calc_ood_measures(ind_model, in_images, out_images,
#                                             tot_wrapper_type=ind_tot_wrapper_type,
#                                             know_wrapper_type=ind_know_wrapper_type)
# print(ind_measures)

# print("Evaluating ENSM...")
# ensm_measures = evaluation.calc_ood_measures(ensm_model, in_images, out_images,
#                                              tot_wrapper_type=ensm_tot_wrapper_type,
#                                              know_wrapper_type=ensm_know_wrapper_type)
# print(ensm_measures)

print("Evaluating ENDD...")
endd_measures = evaluation.calc_ood_measures(endd_model, in_images, out_images,
                                             tot_wrapper_type=endd_tot_wrapper_type,
                                             know_wrapper_type=endd_know_wrapper_type)
print(endd_measures)


# print("Evaluating ENDD+AUX...")
# endd_aux_measures = evaluation.calc_ood_measures(endd_aux_model, in_images, out_images,
#                                                  tot_wrapper_type=endd_aux_tot_wrapper_type,
#                                                  know_wrapper_type=endd_aux_know_wrapper_type,
#                                                  classifier_is_pn=True)
#
#
# print("Evaluations complete.")

# Format and print results
summary = evaluation.format_results(['IND', 'ENSM', 'ENDD', 'ENDD+AUX'],
                                    [ind_measures, ensm_measures, endd_measures, endd_aux_measures],
                                    dataset_name=DATASET_NAME)

print(summary)
