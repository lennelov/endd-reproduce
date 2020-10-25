'''
Creates and trains a conv model on the cifar10 dataset
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
MODEL = 'vgg'
DATASET = 'cifar10'
PLOT_SIMPLEX = False
SAVE_WEIGHTS = True
NORMALIZATION = "-1to1"
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
(OOD_images, _), (_,_) = datasets.cifar100.load_data()
OOD_images = OOD_images[0:5000,:,:,:]
train_images, train_alphas, test_images, test_alphas = preprocessing.preprocess_cifar_for_priornet(
    train_images, train_labels, test_images, test_labels,normalization = NORMALIZATION,OOD_images = OOD_images)

model = training.train_pn(train_images,train_alphas,DATASET,MODEL)

if SAVE_WEIGHTS:
    saveload.save_tf_model(model, "PN_vgg_cifar10_aux")
alphas = tf.math.exp(model.predict(test_images))

predictions = tf.math.argmax(tf.squeeze(alphas), axis=1)
real = tf.math.argmax(tf.squeeze(test_alphas), axis=1)
print(real[0:20])
print(predictions[0:20])
score = tf.math.reduce_sum(tf.cast(predictions == real, tf.float32)) / len(real)
print('score: ' + str(score))
if PLOT_SIMPLEX:
    simplex.plot_simplex(alphas)
