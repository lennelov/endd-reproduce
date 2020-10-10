import tensorflow as tf
from tensorflow.keras import layers, models
from utils.DirichletKL import DirichletKL
import settings 
import utils.saveload as saveload
def get_model(dataset_name, compile=True, weights=None):
        """Take dataset name and return corresponding untrained DPN model.
        Args:
            dataset_name (str): Name of the dataset that the model will be used on,
                                must be listed in settings.py.
            compile (bool): If False, an uncompiled model is returned. Default is True.
            weights (str): Name of saved weights. If provided, returned model will
                           be loaded with saved weights. Default is None.
        Returns:
            keras Model object
        If compile=True, model will be compiled with adam optimizer, dirichlet KL
         loss
        """
        if dataset_name not in settings.DATASET_NAMES:
            raise ValueError("""Dataset {} not recognized, please make sure it has been listed in
                            settings.py""".format(dataset_name))
        input_shape = settings.DATASET_INPUT_SHAPES[dataset_name]
        model = models.Sequential()
        model.add(tf.keras.Input(shape=input_shape))
        model.add(layers.Dense(64,activation = 'relu'))
        model.add(layers.Dense(64,activation = 'relu'))
        model.add(layers.Dense(64,activation = 'relu')) 
        model.add(layers.Dense(64,activation = 'relu'))
        model.add(layers.Dense(settings.DATASET_N_CLASSES[dataset_name],activation = 'exponential'))
        if weights:
                saveload.load_weights(model, weights)

        if not compile:
                return model
		
        KL = DirichletKL()
        model.compile(optimizer='adam',loss = KL)
        return model
