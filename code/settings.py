# Paths
SAVED_WEIGHTS_PATH = "code/models/saved_weights"
SAVED_MODELS_PATH = "code/models/saved_models"

# Dataset metadata
DATASET_NAMES = ["cifar10", "cifar100", "mnist", "synthetic"]
DATASET_INPUT_SHAPES = {"cifar10": (32, 32, 3),
                        "cifar100": (32, 32, 3),
                        "mnist": (28, 28, 1),
                        "synthetic": (2)}
DATASET_N_CLASSES = {"cifar10": 10, "cifar100": 100, "mnist": 10, "synthetic": 3}

# Model metadata
from models import cnn, vgg, dense_priornet
MODEL_MODULES = {
    'cnn': cnn,
    'vgg': vgg,
    'dense_priornet': dense_priornet
}

# synthetic data hyperparams


SAMPLES_PER_CLASS = 10000
SAMPLES_OOD = 10000
NOISE_ID = 0.3
NOISE_OOD = 0.5
RADIUS=410
ID_NOISE=0.1
OOD_NOISE=0.1

# preprocess data for PN

ID_LOGIT = 100
OOD_LOGIT = 1
TRAIN_RATIO = 0.7
