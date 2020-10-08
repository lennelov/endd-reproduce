from models import cnn, vgg

# Paths
SAVED_WEIGHTS_PATH = "code/models/saved_weights"
SAVED_MODELS_PATH = "code/models/saved_models"

# Dataset metadata
DATASET_NAMES = ["cifar10", "mnist"]
DATASET_INPUT_SHAPES = {"cifar10": (32, 32, 3), "mnist": (32, 32, 1)}
DATASET_N_CLASSES = {"cifar10": 10, "mnist": 10}

# Model metadata
MODEL_MODULES = {
    'cnn': cnn,
    'vgg': vgg
}
