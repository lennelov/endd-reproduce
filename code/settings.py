# Paths
SAVED_WEIGHTS_PATH = "code/models/saved_weights"
SAVED_MODELS_PATH = "code/models/saved_models"
DATA_PATH = "data"

# Dataset metadata
DATASET_NAMES = ["cifar10", "cifar100", "mnist", "spiral", "cifar10_3_classes"]
DATASET_INPUT_SHAPES = {
    "cifar10": (32, 32, 3),
    "cifar100": (32, 32, 3),
    "mnist": (28, 28, 1),
    "spiral": (2),
    "cifar10_3_classes": (32, 32, 3)
}
DATASET_N_CLASSES = {"cifar10": 10, "cifar100": 100, "mnist": 10, "spiral": 3, "cifar10_3_classes": 3}

# Model metadata
from models import cnn, vgg, cnn_priorNet, dense_priornet, small_net
MODEL_MODULES = {
    'cnn': cnn,
    'vgg': vgg,
    'cnn_priorNet': cnn_priorNet,
    'dense_priornet': dense_priornet,
    'small_net': small_net
}

# Synthetic data hyperparams
SAMPLES_PER_CLASS_PN = 10000
SAMPLES_OOD_PN = 10000
NOISE_ID_PN = 0.3
NOISE_OOD_PN = 0.5
RADIUS_PN = 410
ID_NOISE_PN = 0.1
OOD_NOISE_PN = 0.1

SAMPLES_PER_CLASS_ENDD = 1000
SAMPLES_OOD_ENDD = 1000
NOISE_ID_ENDD = 0.25
NOISE_OOD_ENDD = 100
RADIUS_ENDD = 800
SEED_TRAIN = 84329
SEED_TEST = 749832

# Preprocess data for PN
ID_LOGIT = 100
OOD_LOGIT = 1
TRAIN_RATIO = 0.7
