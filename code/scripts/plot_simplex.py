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


(_,_), (test_images, _) = datasets.cifar10.load_data()
test_images = preprocessing.normalize_minus_one_to_one(test_images, min=0, max=255)
test_image = test_images[0,:,:,:]

#PN_AUX_MODEL_NAME, PN_AUX_BASE_MODEL = 'PN_vgg_cifar10_aux','vgg'
ENSM_MODEL_NAME, ENSM_N_MODELS = 'vgg', 5
DATASET_NAME = 'cifar10'
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSM_MODEL_NAME][DATASET_NAME][:ENSM_N_MODELS]
models = [ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_names]
ensm_model = ensemble.Ensemble(models)


# Prepare PN+AUX model
#pn_aux_model = cnn_priorNet.get_model(PN_AUX_BASE_MODEL,
#                                      dataset_name = DATASET_NAME,
#                                      compile = True,
#                                      weights = PN_AUX_MODEL_NAME)







#pn_alphas = tf.math.exp(pn_model.predict(test_image))
ensemble_alphas = tf.math.exp(ensm_model.predict(tf.expand_dims(test_image,axis = 0)))
#endd_alphas = tf.math.exp(endd_model.predict(test_image))

#top3_classes = tf.nn.top_k(pn_alphas, k=3, sorted=True, name=None)

print(ensemble_alphas.shape)
#pn_alphas = pn_alphas[top3_classes]
ensemble_alphas = np.squeeze(ensemble_alphas)
index = np.argpartition(np.sum(ensemble_alphas,axis = 0), -3)[-3:]
ensemble_alphas = ensemble_alphas[:,index]

simplex.compare_simplex(ensemble_alphas,ensemble_alphas,ensemble_alphas)
