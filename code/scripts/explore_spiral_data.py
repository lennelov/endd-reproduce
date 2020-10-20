# Add parent dir to path to allow for parallel imports
import sys
import pathlib
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
#other functions
import settings
from utils.simplex import plot_simplex
from utils import preprocessing, datasets
from models.small_net import get_model
from utils.create_toy_data import create_mixed_data
import utils.saveload as saveload

#####################################################
####             Utility functions				####
####################################################

def plot_dataset(X, Y, aux = False):

    if aux:
        classes = (-1, 0, 1, 2)
        lim = 1300
        marker_size = 5
    else:
        classes = (0, 1, 2)
        lim = 475
        marker_size = 20
        
    colors = {0: (245/255, 113/255, 137/255),
              1: (80/255, 174/255, 50/255),
              2: (59/255, 161/255, 234/255),
              -1: (254/255, 203/255, 82/255)}

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    
    

    for i in classes:
        idx = np.where(Y == i)
        sns.scatterplot(X[idx, 0].flatten(), X[idx, 1].flatten(), 
                        marker = "s",
                        s = marker_size,
                        alpha = 0.35,
                        color = colors[i],
                        ax = ax,
                        edgecolor = "none")
        
    plt.show()



#####################################################
####             Experiments					####
####################################################

def generate_figure_2():
	''' Reproduces the Figure 2 in Malinin (2020)'''

	(x_train, y_train), _ = datasets.get_dataset("spiral")

	plot_dataset(x_train, y_train, aux = True)
	plot_dataset(x_train, y_train, aux = False)

def train_models():
	'''Trains an ensemble of models on the spiral dataset.'''

	MODEL_TYPE = "small_net"
	ENSEMBLE_SAVE_NAME = "small_net"
	DATASET_NAME = "spiral"
	NAME_START_NUMBER = 0
	N_MODELS = 100
	N_EPOCHS = 85

	# Import data
	(x_train, y_train), (x_test, y_test) = datasets.get_dataset(DATASET_NAME)
	y_train_one_hot = tf.one_hot(y_train.reshape((-1,)), settings.DATASET_N_CLASSES[DATASET_NAME])
	y_test_one_hot = tf.one_hot(y_test.reshape((-1,)), settings.DATASET_N_CLASSES[DATASET_NAME])
	
	# Train models
	model_module = settings.MODEL_MODULES[MODEL_TYPE]
	saved_model_names = []
	try:
		for i in range(N_MODELS):
			# Get model
			model = model_module.get_model(dataset_name = DATASET_NAME, compile = True)

			# Train model
			model.fit(x_train, y_train_one_hot, 
		          validation_data = (x_test, y_test_one_hot), 
		          epochs = N_EPOCHS,
		          verbose = 0)
			print("Model {} finished training.".format(i))

			# Save model
			model_name = "{}_{}_{}".format(ENSEMBLE_SAVE_NAME, DATASET_NAME, i)
			saveload.save_tf_model(model, model_name)
			saved_model_names.append(model_name)

	finally:
		    append_model_names = NAME_START_NUMBER > 0
		    saveload.update_ensemble_names(ENSEMBLE_SAVE_NAME,
		                                   DATASET_NAME,
		                                   saved_model_names,
		                                   append=append_model_names)


	





#####################################################
####             Main 							####
####################################################


if __name__ == '__main__':
	#generate_figure_2()
	train_models()


