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


(_,_), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = preprocessing.normalize_minus_one_to_one(test_images, min=0, max=255)
test_image_ID = test_images[1,:,:,:]
test_image_OOD = np.random.normal(0,0.1,size =(32,32,3))
test_image_ID_uncertain = (test_images[0,:,:,:]+test_images[10,:,:,:]+test_images[20,:,:,:])/3
print(test_labels[0])
print(test_labels[10])
print(test_labels[20])
test_image_ID = tf.expand_dims(test_image_ID,axis = 0)
test_image_OOD = tf.expand_dims(test_image_OOD,axis = 0)
test_image_ID_uncertain = tf.expand_dims(test_image_ID_uncertain,axis = 0)
PN_AUX_MODEL_NAME, PN_AUX_BASE_MODEL = 'PN_vgg_cifar10_aux','vgg'
ENSM_MODEL_NAME, ENSM_N_MODELS = 'vgg', 5
DATASET_NAME = 'cifar10'
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSM_MODEL_NAME][DATASET_NAME][:ENSM_N_MODELS]
models = [ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_names]
ensm_model = ensemble.Ensemble(models)


# Prepare PN+AUX model
pn_aux_model = saveload.load_tf_model(PN_AUX_MODEL_NAME,compile = False)
pn_aux_model.compile(optimizer = 'adam',loss = DirichletKL())







pn_alphas = tf.math.exp(pn_aux_model.predict(test_image_ID))
ensemble_alphas_ID = tf.math.exp(ensm_model.predict(test_image_ID))
ensemble_alphas_OOD = tf.math.exp(ensm_model.predict(test_image_OOD))
ensemble_alphas_ID_uncertain = tf.math.exp(ensm_model.predict(test_image_ID_uncertain))
#endd_alphas = tf.math.exp(endd_model.predict(test_image))

top3_classes, ind = tf.nn.top_k(pn_alphas, k=3, sorted=True, name=None)

pn_alphas =np.squeeze(top3_classes.numpy())
ensemble_alphas_ID = np.squeeze(ensemble_alphas_ID)
ensemble_alphas_OOD = np.squeeze(ensemble_alphas_OOD)
ensemble_alphas_ID_uncertain = np.squeeze(ensemble_alphas_ID_uncertain)
index = np.argpartition(np.sum(ensemble_alphas_ID,axis = 0), -3)[-3:]
index = index[::-1]
ensemble_alphas_ID = ensemble_alphas_ID[:,index]

index = np.argpartition(np.sum(ensemble_alphas_OOD,axis = 0), -3)[-3:]
index = index[::-1]
ensemble_alphas_OOD = ensemble_alphas_OOD[:,index]

index = np.argpartition(np.sum(ensemble_alphas_ID_uncertain,axis = 0), -3)[-3:]
index = index[::-1]
ensemble_alphas_ID_uncertain = ensemble_alphas_ID_uncertain[:,index]

summ =np.sum(ensemble_alphas_ID,axis = 1).reshape((5,1))
ensemble_alphas_ID = ensemble_alphas_ID/summ
summ =np.sum(ensemble_alphas_OOD,axis = 1).reshape((5,1))
ensemble_alphas_OOD = ensemble_alphas_OOD/summ
summ =np.sum(ensemble_alphas_ID_uncertain,axis = 1).reshape((5,1))
ensemble_alphas_ID_uncertain = ensemble_alphas_ID_uncertain/summ
print(ensemble_alphas_ID_uncertain)

simplex.compare_simplex(ensemble_alphas_ID_uncertain,ensemble_alphas_OOD,ensemble_alphas_ID)
