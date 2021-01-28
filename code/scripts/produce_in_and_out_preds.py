"""Saves in and out preds from ENSM and ENDD."""

# Add parent dir to path to allow for parallel imports
import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

import pickle
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from utils import evaluation, datasets, saveload, preprocessing
from models import ensemble, endd, cnn_priorNet

# Choose models
ENSM_MODEL_NAME, ENSM_N_MODELS = 'vgg_a', 100
ENDD_MODEL_NAME, ENDD_BASE_MODEL = 'endd_vgg_cifar10_a', 'vgg'
ENDD_AUX_MODEL_NAME, ENDD_AUX_BASE_MODEL = 'new_cifar10_vgg_endd_aux_0_TEMP=10', 'vgg'

# Choose dataset
DATASET_NAME = 'cifar10'
OUT_DATASET_NAME = 'lsun'

# Prepare ENSM model
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSM_MODEL_NAME][DATASET_NAME][:ENSM_N_MODELS]
models = [ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_names]
ensm_model = ensemble.Ensemble(models)

# Prepare ENDD model
endd_model = endd.get_model(ENDD_BASE_MODEL,
                            dataset_name=DATASET_NAME,
                            compile=True,
                            weights=ENDD_MODEL_NAME)

# Prepare ENDD+AUX model
endd_aux_model = endd.get_model(ENDD_AUX_BASE_MODEL,
                                dataset_name=DATASET_NAME,
                                compile=True,
                                weights=ENDD_AUX_MODEL_NAME)

# Load data
_, (in_images, _) = datasets.get_dataset(DATASET_NAME)
_, out_images = datasets.get_dataset(OUT_DATASET_NAME)

# Preprocess data
in_images = preprocessing.normalize_minus_one_to_one(in_images, min=0, max=255)
out_images = preprocessing.normalize_minus_one_to_one(out_images, min=0, max=255)

print("Evaluating ENSM...")
ensm_preds_in = ensm_model.predict(in_images)
ensm_preds_out = ensm_model.predict(out_images)
with open("code/preds/ensm_preds_in.pkl", 'wb') as file:
    pickle.dump((ensm_preds_in), file)
with open("code/preds/ensm_preds_out.pkl", 'wb') as file:
    pickle.dump((ensm_preds_out), file)

print("Evaluating ENDD...")
endd_preds_in = endd_model.predict(in_images)
endd_preds_out = endd_model.predict(out_images)
with open("code/preds/endd_preds_in.pkl", 'wb') as file:
    pickle.dump((endd_preds_in), file)
with open("code/preds/endd_preds_out.pkl", 'wb') as file:
    pickle.dump((endd_preds_out), file)

print("Evaluating ENDD+AUX...")
endd_aux_preds_in = endd_aux_model.predict(in_images)
endd_aux_preds_out = endd_aux_model.predict(out_images)
with open("code/preds/endd_aux_preds_in.pkl", 'wb') as file:
    pickle.dump((endd_aux_preds_in), file)
with open("code/preds/endd_aux_preds_out.pkl", 'wb') as file:
    pickle.dump((endd_aux_preds_out), file)

print("Evaluations complete.")
