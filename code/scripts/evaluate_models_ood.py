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
ENSM_MODEL_NAME, ENSM_N_MODELS = 'vgg_a', 100
ENDD_MODEL_NAME, ENDD_BASE_MODEL = 'endd_vgg_cifar10_a', 'vgg'
ENDD_AUX_MODEL_NAME, ENDD_AUX_BASE_MODEL = 'endd_vgg_cifar10_aux_final', 'vgg'
PN_AUX_MODEL_NAME = 'PN_vgg_cifar10_aux_c'

# Choose dataset
DATASET_NAME = 'cifar10'
OUT_DATASET_NAME = 'lsun'

# Prepare IND model
ind_model = saveload.load_tf_model(IND_MODEL_NAME)
ind_tot_wrapper_type = 'individual'
ind_know_wrapper_type = None

# Prepare ENSM model
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSM_MODEL_NAME][DATASET_NAME][:ENSM_N_MODELS]
models = [ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_names]
ensm_model = ensemble.Ensemble(models)
ensm_tot_wrapper_type = 'ensemble'
ensm_know_wrapper_type = 'ensemble_ood'

# Prepare END model
# TODO: Add END model
end_tot_wrapper_type = 'individual'
end_know_wrapper_type = 'individual'

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

# Prepare PN+AUX model
pn_base_model = saveload.load_tf_model(PN_AUX_MODEL_NAME, compile=False)
pn_aux_model = cnn_priorNet.get_model(pn_base_model,
                                      dataset_name=DATASET_NAME,
                                      compile=True)
pn_aux_tot_wrapper_type = 'individual'
pn_aux_know_wrapper_type = 'priornet'


# Load data
_, (in_images, _) = datasets.get_dataset(DATASET_NAME)
_, out_images = datasets.get_dataset(OUT_DATASET_NAME)

# Preprocess data
in_images = preprocessing.normalize_minus_one_to_one(in_images, min=0, max=255)
out_images = preprocessing.normalize_minus_one_to_one(out_images, min=0, max=255)

# Calculate measures
# print("Evaluating IND...")
# ind_measures = evaluation.calc_ood_measures(ind_model,
#                                             in_images,
#                                             out_images,
#                                             tot_wrapper_type=ind_tot_wrapper_type,
#                                             know_wrapper_type=ind_know_wrapper_type)

print("Evaluating ENSM...")
ensm_measures = evaluation.calc_ood_measures(ensm_model,
                                             in_images,
                                             out_images,
                                             tot_wrapper_type=ensm_tot_wrapper_type,
                                             know_wrapper_type=ensm_know_wrapper_type,
                                             preds_save_name="ensm_ood")

print("Evaluating ENDD...")
endd_measures = evaluation.calc_ood_measures(endd_model,
                                             in_images,
                                             out_images,
                                             tot_wrapper_type=endd_tot_wrapper_type,
                                             know_wrapper_type=endd_know_wrapper_type,
                                             preds_save_name="endd_ood")

print("Evaluating ENDD+AUX...")
endd_aux_measures = evaluation.calc_ood_measures(endd_aux_model,
                                                 in_images,
                                                 out_images,
                                                 tot_wrapper_type=endd_aux_tot_wrapper_type,
                                                 know_wrapper_type=endd_aux_know_wrapper_type,
                                                 preds_save_name="endd_aux_ood")

# print("Evaluating PN+AUX...")
# pn_aux_measures = evaluation.calc_ood_measures(pn_aux_model,
#                                                  in_images,
#                                                  out_images,
#                                                  tot_wrapper_type=pn_aux_tot_wrapper_type,
#                                                  know_wrapper_type=pn_aux_know_wrapper_type)

# print("Evaluations complete.")
#
# # Format and print results
# summary = evaluation.format_ood_results(
#     ['IND', 'ENSM', 'ENDD', 'ENDD+AUX'],
#     [ind_measures, ensm_measures, endd_measures, endd_aux_measures],
#     in_dataset_name=DATASET_NAME,
#     out_dataset_name=OUT_DATASET_NAME)
#
#
# print(summary)
