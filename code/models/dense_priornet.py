import tensorflow as tf
from tensorflow.keras import layers, models
from utils.DirichletKL import DirichletKL
def get_model(n_classes,n_layers = 4, n_neurons = 64,activations = 'relu',weights = None):
        """Take n_classes and return corresponding untrained dense model.
         Args:
                n_classes (int): nr of classes in the toy data-set
                n_layers = 4 (int): the number of hidden layers
                n_neurons = 64 (int): neurons per layer
                activations = 'relu' (str): activation functions at all layers apart from the last which is exponential
		compile (bool): If False, an uncompiled model is returned. Default is True.
		weights (str): Name of saved weights. If provided, returned model will
		               be loaded with saved weights. Default is None.			
        Output:
		Returns:
		keras Model object
		If compile=True, model will be compiled with adam optimizer, categorical cross
		entropy loss, and accuracy metric.
        """	
        model = models.Sequential()
        model.add(tf.keras.Input(shape=(2,)))
        for hidden_layer in range(n_layers):
                model.add(layers.Dense(n_neurons,activation = activations))
        model.add(layers.Dense(n_classes,activation = 'exponential'))
        if weights:
                saveload.load_weights(model, weights)

        if not compile:
                return model
		
        KL = DirichletKL()
        model.compile(optimizer='adam',loss = KL)
        return model
