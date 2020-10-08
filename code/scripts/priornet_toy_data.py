# Add parent dir to path to allow for parallel imports
import sys
import pathlib
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

#packages
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.math import lgamma,digamma
import numpy as np
import torch # only used for KaosEngineers dataset
from torch.utils.data import Dataset, DataLoader # only used for KaosEngineers dataset
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from datetime import datetime
from packaging import version
#other functions
from settings_prior import *
from utils.simplex_plot_function import *
from utils.preprocess_toy_dataset import preprocess
from utils.DirichletKL import DirichletKL
from models.dense_priornet import get_model
from utils.create_toy_data import SpiralDataset, OODSpiralDataset

Spiral = SpiralDataset(SAMPLES_PER_CLASS,NOISE,N_CLASSES)
OOD = OODSpiralDataset(SAMPLES_OOD)

x_train,logits_train,x_test,y_test = preprocess(Spiral,OOD)
model = get_model(N_CLASSES,N_LAYERS,N_NEURONS,activations = ACTIVATION)

model.fit(
            x = x_train,
            y = logits_train,
            batch_size = BATCH_SIZE,epochs = N_EPOCHS)

logits = model.predict(x_test)
predictions = tf.math.argmax(logits,axis = 1)
real = tf.math.argmax(y_test,axis = 1)

if PLOT_SIMPLEX and N_CLASSES == 2:
    plot_simplex(logits)

      
