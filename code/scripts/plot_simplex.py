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

from utils import preprocessing, saveload, simplex, losses, training
from models import cnn_priorNet

(_,_), (test_images, _) = datasets.cifar10.load_data()
'''
test_image = test_images[0,:,:,:]
pn_model = saveload.load_tf_model("PN_vgg_cifar_3_class")
ensemble_model = saveload.load_tf_model("ensemble_vgg_cifar_3_class")
endd_model = saveload.load_tf_model("EnDD_vgg_cifar_3_class")


pn_alphas = tf.math.exp(pn_model.predict(test_image))
ensemble_alphas = tf.math.exp(ensemble_model.predict(test_image))
endd_alphas = tf.math.exp(endd_model.predict(test_image))
'''
pn_alphas = [1, 2, 1]
ensemble_alphas = [2, 2, 1]
endd_alphas = [1, 3, 5]
simplex.compare_simplex(pn_alphas,ensemble_alphas,endd_alphas)
