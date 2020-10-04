#packages
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.math import lgamma
from tensorflow.math import digamma
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

from preprocess_toy_dataset import preprocess_toy_dataset
from DirichletKL import DirichletKL
from create_toy_dataset_model import create_toy_dataset_model
from train_priornet_toy_dataset import train_priornet_toy_dataset
from dataset_creation_KaosEngineer import SpiralDataset
from dataset_creation_KaosEngineer import OODSpiralDataset

def main():
      n_classes = 3
      Spiral = SpiralDataset(10000,0.2,n_classes)
      OOD = OODSpiralDataset(10000)
      x_train,y_train,x_test,y_test = preprocess_toy_dataset(Spiral,OOD)
      model = create_toy_dataset_model(n_classes,n_layers = 5, n_neurons = 64,activations = 'relu')
      KL = DirichletKL()
      model.compile(optimizer='adam',loss = KL)
      model = train_priornet_toy_dataset(x_train,y_train,model)
      logits = model.predict(x_test)
      predictions = tf.math.argmax(logits,axis = 1)
      real = tf.math.argmax(y_test,axis = 1)
      print(tf.reduce_sum(tf.cast(predictions ==real, tf.float32))/len(real))
main()