'''
Creates and trains a conv model on the cifar10 dataset
'''
import sys
import pathlib
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import tensorflow as tf

from tensorflow.keras import datasets

import settings
from utils import preprocessing, saveload, simplex, losses
from models import cnn_priorNet
MODEL = 'vgg'
DATASET = 'cifar10'
PLOT_SIMPLEX = False
SAVE_WEIGHTS = True
BATCH_SIZE = 100
EPOCHS = 5
NORMALIZATION = "-1to1"
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
(OOD_images, _), (_,_) = datasets.cifar100.load_data()
OOD_images = OOD_images[0:5000,:,:,:]
train_images, train_logits, test_images, test_logits = preprocessing.preprocess_cifar_for_priornet(
    train_images, train_labels, test_images, test_labels,normalization = NORMALIZATION,OOD_images = OOD_images)

train_images = train_images[0:5000,:,:,:]
train_logits = train_logits[0:5000,:]
print(train_images[4900:4920,0,0,0])
model = cnn_priorNet.get_model(MODEL,DATASET)
model.fit(train_images, train_logits, batch_size=BATCH_SIZE, epochs=EPOCHS)
if SAVE_WEIGHTS:
    saveload.save_tf_model(model, "PN_vgg_cifar10_aux")

logits = model.predict(test_images)
predictions = tf.math.argmax(tf.squeeze(logits), axis=1)
real = tf.math.argmax(tf.squeeze(test_logits), axis=1)
score = tf.math.reduce_sum(tf.cast(predictions == real, tf.float32)) / len(real)
print('score: ' + str(score))
if PLOT_SIMPLEX:
    simplex.plot_simplex(logits)
