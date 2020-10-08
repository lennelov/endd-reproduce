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
from datetime import datetime
from packaging import version

from settings_prior import *
from settings import *
from utils.simplex_plot_function import *
from scripts.DirichletKL import DirichletKL
from models.create_image_model import create_image_model
from scripts.train_priornet_toy_dataset import train_priornet_toy_dataset
from utils.data_creation_cifar10 import data_creation_cifar10
from utils.plot_simplex import plot_simplex

print(DATASET_NAMES)
train_images, train_logits, test_images, test_logits = data_creation_cifar10()
model = create_image_model('cifar10')
KL = DirichletKL()
model.compile(optimizer = 'adam',loss = KL,run_eagerly=False)
model = train_priornet_toy_dataset(train_images,train_logits,model,batch = 100,n_epochs = 2)
logits = model.predict(test_images)
predictions = tf.math.argmax(tf.squeeze(logits),axis = 1)
real = tf.math.argmax(tf.squeeze(test_logits),axis = 1)
score = tf.math.reduce_sum(tf.cast(predictions == real,tf.float32))/len(real)
print('score: ' + str(score))
plot_simplex(logits)
