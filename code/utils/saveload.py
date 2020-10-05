import os
import tensorflow.keras as keras

SAVED_WEIGHTS_PATH = "code/models/saved_weights"
SAVED_MODELS_PATH = "code/models/saved_models"


def save_weights(model, name, verbose=True):
    """Take model and save weights with given name.

    Args:
        model (keras.Model): Model with weights to be saved.
        name (str): Name that will be used when loading. Shoud not contain path
                    or file ending.
        verbose (str): If True a message is printed upon success.
    """
    weights_name = "weights_" + name
    weights_path = os.path.join(SAVED_WEIGHTS_PATH, weights_name, weights_name)
    model.save_weights(weights_path)
    print("Weights succesfully saved as {}".format(weights_path))


def load_weights(model, name):
    """Take model and load weights with given name (in place)."""
    weights_name = "weights_" + name
    weights_path = os.path.join(SAVED_WEIGHTS_PATH, weights_name, weights_name)
    model.load_weights(weights_path)


def save_tf_model(model, name, verbose=True):
    """Take model and save with given name.

    Args:
        model (keras.Model): Model to be saved.
        name (str): Name that will be used when loading. Shoud not contain path
                    or file ending.
        verbose (str): If True a message is printed upon success.
    """
    model_path = os.path.join(SAVED_MODELS_PATH, "model_" + name)
    model.save(model_path)
    print("Model succesfully saved as {}".format(model_path))


def load_tf_model(name):
    """Take saved model name and return model."""
    model_path = os.path.join(SAVED_MODELS_PATH, "model_" + name)
    model = keras.models.load_model(model_path)
    return model
