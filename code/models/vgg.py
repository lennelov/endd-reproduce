import tensorflow as tf
import tensorflow.keras as keras
import models.model_globals as model_globals


def get_model(dataset_name, compile=True):
    """Take dataset name and return corresponding untrained VGG16 model.

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

    model = tf.keras.applications.VGG16(
        include_top=True,
        weights=None,
        input_shape=model_globals.DATASET_INPUT_SHAPES[dataset_name],
        classes=model_globals.DATASET_N_CLASSES[dataset_name])

    if not compile:
        return model

    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model
