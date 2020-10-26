'''
Plots a simplex for each of the models on cifar10 data with three classes ID and seven classes OOD. 
'''
import sys
import pathlib
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets

import settings
from models import ensemble, endd, cnn_priorNet
from utils import preprocessing, saveload, simplex, losses, training
from utils.losses import DirichletKL

(_, _), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = preprocessing.normalize_minus_one_to_one(test_images, min=0, max=255)
ID_index = np.where(test_labels < 3)
ID_index = ID_index[0]
ID_index = ID_index[0:6]
OOD_index = np.where(test_labels > 2)
OOD_index = OOD_index[0]
test_image_ID = test_images[ID_index[0], :, :, :]
test_image_OOD = test_images[OOD_index[0], :, :, :]
test_image_ID_uncertain = (test_images[ID_index[0], :, :, :] + test_images[ID_index[1], :, :, :] +
                           test_images[ID_index[5], :, :, :]) / 3  #0,1,5 are all different classes

test_image_ID = tf.expand_dims(test_image_ID, axis=0)
test_image_OOD = tf.expand_dims(test_image_OOD, axis=0)
test_image_ID_uncertain = tf.expand_dims(test_image_ID_uncertain, axis=0)

ENSM_MODEL_NAME, ENSM_N_MODELS = 'vgg_3_class', 5
DATASET_NAME = 'cifar10_3_class'
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSM_MODEL_NAME][DATASET_NAME][:ENSM_N_MODELS]
models = [ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_names]
ensm_model = ensemble.Ensemble(models)

ensemble_alphas_ID = np.squeeze(tf.math.exp(ensm_model.predict(test_image_ID)))
ensemble_alphas_OOD = np.squeeze(tf.math.exp(ensm_model.predict(test_image_OOD)))
ensemble_alphas_ID_uncertain = np.squeeze(tf.math.exp(ensm_model.predict(test_image_ID_uncertain)))
ensemble_alphas_ID = ensemble_alphas_ID / np.expand_dims(np.sum(ensemble_alphas_ID, axis=1), axis=1)
ensemble_alphas_OOD = ensemble_alphas_OOD / np.expand_dims(np.sum(ensemble_alphas_OOD, axis=1),
                                                           axis=1)
ensemble_alphas_ID_uncertain = ensemble_alphas_ID_uncertain / np.expand_dims(
    np.sum(ensemble_alphas_ID_uncertain, axis=1), axis=1)
simplex.compare_simplex(ensemble_alphas_ID_uncertain, ensemble_alphas_OOD, ensemble_alphas_ID)
