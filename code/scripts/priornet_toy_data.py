# Add parent dir to path to allow for parallel imports
import sys
import pathlib
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

#packages
import tensorflow as tf
import numpy as np
import keras
#other functions
import settings
from utils.simplex_plot_function import plot_simplex
from utils.preprocess_toy_dataset import preprocess
from models.dense_priornet import get_model
from utils.create_toy_data import create_mixed_data
import utils.saveload as saveload

dataset = "synthetic"
BATCH_SIZE = 100
N_EPOCHS = 20
PLOT_SIMPLEX = False
SAVE_WEIGHTS = False

X,Y = create_mixed_data(settings.SAMPLES_PER_CLASS,settings.SAMPLES_OOD,
	settings.DATASET_N_CLASSES[dataset],radius=settings.RADIUS,
	ID_noise=settings.ID_NOISE,OOD_noise=settings.OOD_NOISE)

x_train,logits_train,x_test,logits_test = preprocess(X,Y,0.8)
model = get_model(dataset,compile = True)

model.fit(
            x = x_train,
            y = logits_train,
            batch_size = BATCH_SIZE,epochs = N_EPOCHS)
if SAVE_WEIGHTS:
	saveload.save_tf_model(model, "dense_priornet")

logits = model.predict(x_test)
predictions = tf.math.argmax(logits,axis = 1)
real = tf.math.argmax(logits_test,axis = 1)
if PLOT_SIMPLEX and settings.DATASET_N_CLASSES[dataset] == 3:
    plot_simplex(logits)

      
