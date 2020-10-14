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
from utils.simplex import plot_simplex
from utils.DirichletKL import DirichletKL
from models.cnn_priorNet import get_model
from utils import preprocessing
from utils import saveload

DATASET = 'cifar10_PN'
PLOT_SIMPLEX = False
SAVE_WEIGHTS = False
BATCH_SIZE = 100
EPOCHS = 2

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, train_logits, test_images, test_logits = preprocessing.preprocess_cifar_for_priornet(
    train_images, train_labels, test_images, test_labels)
model = get_model(DATASET)
model.fit(train_images, train_logits, batch_size=BATCH_SIZE, epochs=EPOCHS)
if SAVE_WEIGHTS:
    saveload.save_tf_model(model, "cnn_priorNet")

logits = model.predict(test_images)
predictions = tf.math.argmax(tf.squeeze(logits), axis=1)
real = tf.math.argmax(tf.squeeze(test_logits), axis=1)
score = tf.math.reduce_sum(tf.cast(predictions == real, tf.float32)) / len(real)
print('score: ' + str(score))
if PLOT_SIMPLEX:
    plot_simplex(logits)
