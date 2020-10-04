import tensorflow as tf
from tensorflow.keras import layers, models
def create_toy_dataset_model(n_classes,n_layers = 4, n_neurons = 64,activations = 'relu'):
        '''
        Creates a dense-net for the toy data-set with inputs in 2D,
        relu activations and exponential activation
        at the output layer.
        
        Inputs:
                n_classes, nr of classes in the toy data-set
                n_layers = 4, the number of hidden layers
                n_neurons = 64, neurons per layer
                activations = 'relu', activation functions at all layers apart from the last which is exponential
        Output:
                Class logits
        '''
        model = models.Sequential()
        model.add(tf.keras.Input(shape=(2,)))
        for hidden_layer in range(n_layers):
                model.add(layers.Dense(n_neurons,activation = activations))
        model.add(layers.Dense(n_classes,activation = 'exponential'))
        return model