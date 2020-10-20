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
from models.dense_priornet import get_model
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


#####################################################
####             Main 							####
####################################################

generate_figure_2()


