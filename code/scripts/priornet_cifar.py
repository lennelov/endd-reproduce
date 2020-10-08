'''
Creates and trains a conv model on the 
'''
import sys
import pathlib
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.math import lgamma,digamma
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from tensorflow.keras import datasets
from datetime import datetime
from packaging import version

from settings_prior import *
from settings import *
from utils.simplex_plot_function import *
from utils.DirichletKL import DirichletKL
from models.cnn_priorNet import get_model
from utils.preprocess_priornet_cifar import preprocess


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, train_logits, test_images, test_logits = preprocess(train_images, train_labels, test_images, test_labels)
model = get_model('cifar10',3)
model.fit(train_images,train_logits,batch_size = 100,epochs = 2)
logits = model.predict(test_images)
predictions = tf.math.argmax(tf.squeeze(logits),axis = 1)
real = tf.math.argmax(tf.squeeze(test_logits),axis = 1)
score = tf.math.reduce_sum(tf.cast(predictions == real,tf.float32))/len(real)
print('score: ' + str(score))
plot_simplex(logits)
