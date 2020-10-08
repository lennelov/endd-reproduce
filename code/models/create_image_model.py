import tensorflow as tf
from tensorflow.keras import layers, models
import sys
import pathlib
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
from settings_prior import *
from settings import *
def create_image_model(dataset_name):
        '''
        Take dataset name and return corresponding untrained 
        CNN model with exponential outputs to be trained as a prior network.
        
        Inputs:
                dataset_name, name of the image data-set
                n_neurons = 64, neurons per layer
                activations = 'relu', activation functions at all layers apart from the last which is exponential
        Output:
                Class logits
        '''
        if dataset_name not in DATASET_NAMES:
            raise ValueError("""Dataset {} not recognized, please make sure it has been listed in
                            settings.py""".format(dataset_name))

        input_shape = DATASET_INPUT_SHAPES[dataset_name]
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(3,activation = 'exponential'))
        return model
