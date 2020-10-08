# Paths
SAVED_WEIGHTS_PATH = "code/models/saved_weights"
SAVED_MODELS_PATH = "code/models/saved_models"

# Dataset metadata
DATASET_NAMES = ["cifar10", "cifar100", "mnist"]
DATASET_INPUT_SHAPES = {"cifar10": (32, 32, 3),
                        "cifar100": (32, 32, 3),
                        "mnist": (28, 28, 1)}
DATASET_N_CLASSES = {"cifar10": 10, "cifar100": 100, "mnist": 10}

# Model metadata
from models import cnn, vgg
MODEL_MODULES = {
    'cnn': cnn,
    'vgg': vgg
}
