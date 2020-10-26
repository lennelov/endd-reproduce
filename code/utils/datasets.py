"""Module collecting datasets for loading.

The _get_dataset functions should return (x_train, y_train), (x_test, y_test)
where the data is formated as numpy arrays.

Once a function as been added it should also be added to the DATASET_GETTERS list in
the get_dataset function.
"""
import numpy as np
from tensorflow.keras import datasets
from utils.create_toy_data import create_mixed_data
import settings


def _get_cifar10():
    return datasets.cifar10.load_data()


def _get_cifar100():
    return datasets.cifar10.load_data()


def _get_mnist():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)


def _get_spiral():

    x_train, y_train = create_mixed_data(ID_points=settings.SAMPLES_PER_CLASS_ENDD,
                                         OOD_points=0,
                                         n_spirals=settings.DATASET_N_CLASSES["spiral"],
                                         radius=settings.RADIUS_ENDD,
                                         ID_noise=settings.NOISE_ID_ENDD,
                                         OOD_noise=settings.NOISE_OOD_ENDD,
                                         seed=settings.SEED_TRAIN)

    x_test, y_test = create_mixed_data(ID_points=settings.SAMPLES_PER_CLASS_ENDD,
                                       OOD_points=0,
                                       n_spirals=settings.DATASET_N_CLASSES["spiral"],
                                       radius=settings.RADIUS_ENDD,
                                       ID_noise=settings.NOISE_ID_ENDD,
                                       OOD_noise=settings.NOISE_OOD_ENDD,
                                       seed=settings.SEED_TEST)

    return (x_train, y_train), (x_test, y_test)


def _get_spiral_aux():

    x_train, y_train = create_mixed_data(ID_points=0,
                                         OOD_points=settings.SAMPLES_OOD_ENDD,
                                         n_spirals=settings.DATASET_N_CLASSES["spiral"],
                                         radius=settings.RADIUS_ENDD,
                                         ID_noise=settings.NOISE_ID_ENDD,
                                         OOD_noise=settings.NOISE_OOD_ENDD,
                                         seed=settings.SEED_TRAIN)

    x_test, y_test = create_mixed_data(ID_points=0,
                                       OOD_points=settings.SAMPLES_OOD_ENDD,
                                       n_spirals=settings.DATASET_N_CLASSES["spiral"],
                                       radius=settings.RADIUS_ENDD,
                                       ID_noise=settings.NOISE_ID_ENDD,
                                       OOD_noise=settings.NOISE_OOD_ENDD,
                                       seed=settings.SEED_TEST)

    return (x_train, y_train), (x_test, y_test)


DATASET_GETTERS = {
    'cifar10': _get_cifar10,
    'cifar100': _get_cifar100,
    'mnist': _get_mnist,
    "spiral": _get_spiral,
    "spiral_aux": _get_spiral_aux
}


def get_dataset(dataset_name):
    """Take dataset name and return dataset.

    Currently available datasets:
        'cifar10'
        'cifar100'
        'mnist'

    Args:
        dataset_name (str): Name of dataset.

    Returns:
        (x_train, y_train), (x_test, y_test) as tuples of numpy arrays.
    """

    if dataset_name not in DATASET_GETTERS:
        raise ValueError("""Dataset {} not recognized, please make sure it has been added
                            to datasets.py""".format(dataset_name))

    train_set, test_set = DATASET_GETTERS[dataset_name]()
    return train_set, test_set


def get_ensemble_dataset(ensemble, dataset_name):
    """WARNING: NO PREPROCESSING IS APPLIED, THIS FUNCITON IS USELESS AT THE MOMENT

    Take ensemble model and dataset name and return ensemble dataset for use with ENDD.

    Args:
        ensemble (models.ensemble.Ensemble): Ensemble model that will provide predictions.
        dataset_name (str): Name of dataset.
    Returns: Two tuples of numpy arrays with format
        ((train_images, train_labels, train_ensemble_preds),
         (test_images, test_labels, test_ensemble_preds))
    """
    if dataset_name not in DATASET_GETTERS:
        raise ValueError("""Dataset {} not recognized, please make sure it has been added
                            to datasets.py""".format(dataset_name))

    (train_images, train_labels), (test_images, test_labels) = DATASET_GETTERS[dataset_name]()
    train_ensemble_preds = get_ensemble_preds(ensemble, train_images)
    test_ensemble_preds = get_ensemble_preds(ensemble, test_images)
    train_set = (train_images, train_labels, train_ensemble_preds)
    test_set = (test_images, test_labels, test_ensemble_preds)
    return train_set, test_set


def get_ensemble_preds(ensemble, images):
    """Take an Ensemble model and images, and return ensemble preds for use with ENDD."""
    ensemble_preds = ensemble.predict(images)
    ensemble_preds = np.transpose(ensemble_preds, (1, 0, 2))
    return ensemble_preds
