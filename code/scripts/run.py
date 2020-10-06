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
import settings
import utils.simplex_plot_function
from scripts.preprocess_toy_dataset import preprocess_toy_dataset
from scripts.DirichletKL import DirichletKL
from models.create_toy_dataset_model import create_toy_dataset_model
from scripts.train_priornet_toy_dataset import train_priornet_toy_dataset
from utils.dataset_creation_KaosEngineer import SpiralDataset
from utils.dataset_creation_KaosEngineer import OODSpiralDataset

Spiral = SpiralDataset(SAMPLES_PER_CLASS,NOISE,N_CLASSES)
OOD = OODSpiralDataset(SAMPLES_OOD)
x_train,y_train,x_test,y_test = preprocess_toy_dataset(Spiral,OOD)
model = create_toy_dataset_model(n_classes,N_LAYERS,N_NEURONS,activations = ACTIVATION)

KL = DirichletKL()
model.compile(optimizer='adam',loss = KL)
model = train_priornet_toy_dataset(x_train,y_train,model,BATCH_SIZE,N_EPOCHS)
logits = model.predict(x_test)
predictions = tf.math.argmax(logits,axis = 1)
real = tf.math.argmax(y_test,axis = 1)
if PLOT_SIMPLEX:
    import seaborn as sn
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 16,
            }
    plt.style.use('seaborn-white')
    plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

    for i in range(0,6):
        plt.subplot(2, 3, i+1)
        plt.title("logits: " + str(np.around(logits[i,:],decimals =1)) ,
                fontsize=18, ha='center')
        plot_logits = logits[i,:]
        draw_pdf_contours(Dirichlet(plot_logits))

      
