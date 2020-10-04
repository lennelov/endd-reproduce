import tensorflow as tf
import tensorflow.keras as keras
import settings
import utils.saveload as saveload


def get_model(dataset_name, compile=True, weights=None):
    """Take dataset name and return corresponding untrained VGG16 model.

    Args:
        dataset_name (str): Name of the dataset that the model will be used on,
                            must be listed in settings.py.
        compile (bool): If False, an uncompiled model is returned. Default is True.
        weights (str): Name of saved weights. If provided, returned model will
                       be loaded with saved weights. Default is None.

    Returns:
        keras Model object

    If compile=True, model will be compiled with adam optimizer, categorical cross
    entropy loss, and accuracy metric.
    """
    if dataset_name not in settings.DATASET_NAMES:
        raise ValueError("""Dataset {} not recognized, please make sure it has been listed in
                            settings.py""".format(dataset_name))

    model = tf.keras.applications.VGG16(include_top=True,
                                        weights=None,
                                        input_shape=settings.DATASET_INPUT_SHAPES[dataset_name],
                                        classes=settings.DATASET_N_CLASSES[dataset_name])

    if weights:
        saveload.load_weights(model, weights)

    if not compile:
        return model

    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model
