import os
import json
import tensorflow.keras as keras
import settings

ENSEMBLE_MODEL_NAMES_PATH = os.path.join(settings.SAVED_MODELS_PATH, "ensemble_model_names.json")
ENSEMBLE_WEIGHT_NAMES_PATH = os.path.join(settings.SAVED_WEIGHTS_PATH, "ensemble_weight_names.json")


def save_weights(model, name, verbose=True):
    """Take model and save weights with given name.

    Args:
        model (keras.Model): Model with weights to be saved.
        name (str): Name that will be used when loading. Shoud not contain path
                    or file ending.
        verbose (str): If True a message is printed upon success.
    """
    weights_name = "weights_" + name
    weights_path = os.path.join(settings.SAVED_WEIGHTS_PATH, weights_name, weights_name)
    model.save_weights(weights_path)
    print("Weights succesfully saved as {}".format(weights_path))


def load_weights(model, name):
    """Take model and load weights with given name (in place)."""
    weights_name = "weights_" + name
    weights_path = os.path.join(settings.SAVED_WEIGHTS_PATH, weights_name, weights_name)
    model.load_weights(weights_path)


def save_tf_model(model, name, verbose=True):
    """Take model and save with given name.

    Args:
        model (keras.Model): Model to be saved.
        name (str): Name that will be used when loading. Shoud not contain path
                    or file ending.
        verbose (str): If True a message is printed upon success.
    """
    model_path = os.path.join(settings.SAVED_MODELS_PATH, "model_" + name)
    model.save(model_path)
    print("Model succesfully saved as {}".format(model_path))


def load_tf_model(name):
    """Take saved model name and return model."""
    model_path = os.path.join(settings.SAVED_MODELS_PATH, "model_" + name)
    model = keras.models.load_model(model_path)
    return model


def get_ensemble_model_names():
    """Return dictionary mapping ensemble names to lists of saved model names.

    Returns:
        (Dict[Dict[str]]) with the following format:

            ensemble_model_names = {
                'cnn': {
                    'cifar10': [cnn_cifar10_0, cnn_cifar10_1, ...]
                    'cifar100': [cnn_cifar100_0, cnn_cifar100_1, ...],
                    ...
                },
                'vgg': {...}
            }
    """
    if not os.path.isfile(ENSEMBLE_MODEL_NAMES_PATH):
        return None

    with open(ENSEMBLE_MODEL_NAMES_PATH, 'r') as file:
        ensemble_model_names = json.load(file)
    return ensemble_model_names


def update_ensemble_names(ensemble_name, dataset_name, model_names, append=False):
    """Add or update key ensemble_name with value model_names.

    Args:
        ensemble_name (str): Name of ensemble.
        model_names List[str]: List of model names belonging to ensemble.
        append (bool): If True, names will append instead of overwrite.
    """
    ensemble_model_names = get_ensemble_model_names()
    if ensemble_model_names is None:
        ensemble_model_names = {ensemble_name: {}}

    if append:
        current_names = ensemble_model_names[ensemble_name][dataset_name]
        ensemble_model_names[ensemble_name][dataset_name] = sorted(
            list(set(current_names) | set(model_names)))
    else:
        ensemble_model_names[ensemble_name][dataset_name] = model_names

    with open(ENSEMBLE_MODEL_NAMES_PATH, 'w') as file:
        json.dump(ensemble_model_names, file)
