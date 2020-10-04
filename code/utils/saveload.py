import os
import tensorflow.keras as keras

SAVED_WEIGHTS_PATH = "code/models/saved_weights"
SAVED_MODELS_PATH = "code/models/saved_models"


def save_weights(model, name, verbose=True):
    weights_path = os.path.join(SAVED_WEIGHTS_PATH, "weights_" + name)
    model.save_weights(weights_path)
    print("Weights succesfully saved as {}".format(weights_path))


def load_weights(model, name):
    weights_path = os.path.join(SAVED_WEIGHTS_PATH, "weights_" + name)
    model.load_weights(weights_path)


def save_tf_model(model, name, verbose=True):
    model_path = os.path.join(SAVED_MODELS_PATH, "model_" + name)
    model.save(model_path)
    print("Model succesfully saved as {}".format(model_path))


def load_tf_model(name):
    model_path = os.path.join(SAVED_MODELS_PATH, "model_" + name)
    model = keras.models.load_model(model_path)
    return model
