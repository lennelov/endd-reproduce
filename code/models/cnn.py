import tensorflow as tf
import tensorflow.keras as keras
import models.model_globals as model_globals


def get_model(dataset_name, compile=True):
    """Take dataset name and return corresponding untrained CNN model.

    Args:
        dataset_name (str): Name of the dataset that the model will be used on,
                            must be listed in model_globals.py.
        compile (bool): If False, an uncompiled model is returned. Default is True.

    Returns:
        keras Model object

    If compile=True, model will be compiled with adam optimizer, categorical cross
    entropy loss, and accuracy metric.
    """
    if dataset_name not in model_globals.DATASET_INPUT_SHAPES:
        raise ValueError("Dataset {} not recognized".format(dataset_name))

    input_shape = model_globals.DATASET_INPUT_SHAPES[dataset_name]

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(model_globals.DATASET_N_CLASSES[dataset_name]))

    if not compile:
        return model

    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model
