"""Module collecting datasets for loading.

The _get_dataset functions should return (x_train, y_train), (x_test, y_test)
where the data is formated as numpy arrays.

Once a function as been added it should also be added to the DATASET_GETTERS list in
the get_dataset function.
"""
from tensorflow.keras import datasets


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

    DATASET_GETTERS = {
        'cifar10': _get_cifar10,
        'cifar100': _get_cifar100,
        'mnist': _get_mnist
    }

    if dataset_name not in DATASET_GETTERS:
        raise ValueError("""Dataset {} not recognized, please make sure it has been added
                            to datasets.py""".format(dataset_name))


    train_set, test_set = DATASET_GETTERS[dataset_name]()
    return train_set, test_set


def _get_cifar10():
    return datasets.cifar10.load_data()


def _get_cifar100():
    return datasets.cifar10.load_data()


def _get_mnist():
    return datasets.mnist.load_data()
