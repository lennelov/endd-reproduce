import pickle
import numpy as np
import tensorflow.keras as keras
import settings

from models import vgg, cnn, endd, ensemble
from utils import evaluation, preprocessing, saveload, simplex, datasets, callbacks

def train_vgg_endd(
        ensemble_model,
        dataset_name,
        aux_dataset_name=None,
        normalization='-1to1',
        batch_size=128,
        n_epochs=90,
        one_cycle_lr_policy=True,
        init_lr=0.001,
        cycle_length=60,
        temp_annealing=True,
        init_temp=10,
        dropout_rate=0.3
        ):

    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(dataset_name)

    if aux_dataset_name:
        (aux_images, _), _ = datasets.get_dataset(aux_dataset_name)
        train_images = np.concatenate((train_images, aux_images), axis=0)

    # Normalize data
    if normalization == "-1to1":
        train_images, min, max = preprocessing.normalize_minus_one_to_one(train_images)
        test_images = preprocessing.normalize_minus_one_to_one(test_images, min, max)
    elif normalization == 'gaussian':
        train_images, mean, std = preprocessing.normalize_gaussian(train_images)
        test_images = preprocessing.normalize_gaussian(test_images, mean, std)

    # Get ensemble preds
    train_ensemble_preds = datasets.get_ensemble_preds(ensemble_model, train_images)
    test_ensemble_preds = datasets.get_ensemble_preds(ensemble_model, test_images)

    # Save / Load pickled data. Generating ensemble preds takes a long time, so saving and
    # loading can make testing much more efficient.
    with open('train.pkl', 'wb') as file:
        pickle.dump((train_images, train_labels, train_ensemble_preds), file)
    with open('test.pkl', 'wb') as file:
        pickle.dump((test_images, test_labels, test_ensemble_preds), file)

    # with open('train.pkl', 'rb') as file:
    #     train_images, train_labels, train_ensemble_preds = pickle.load(file)
    # with open('test.pkl', 'rb') as file:
    #     test_images, test_labels, test_ensemble_preds = pickle.load(file)

    # Image augmentation
    data_generator = preprocessing.make_augmented_generator(train_images, train_ensemble_preds,
                                                            batch_size)

    # Callbacks
    endd_callbacks = []
    if one_cycle_lr_policy:
        olp_callback = callbacks.OneCycleLRPolicy(init_lr=init_lr,
                                                  max_lr=init_lr * 10,
                                                  min_lr=init_lr / 1000,
                                                  cycle_length=cycle_length,
                                                  epochs=n_epochs)
        endd_callbacks.append(olp_callback)

    if temp_annealing:
        temp_callback = callbacks.TemperatureAnnealing(init_temp=init_temp,
                                                       cycle_length=cycle_length,
                                                       epochs=n_epochs)
        endd_callbacks.append(temp_callback)

    if not endd_callbacks:
        endd_callbacks = None

    # Build ENDD model
    base_model = vgg.get_model(dataset_name, compile=False, dropout_rate=dropout_rate)
    endd_model = endd.get_model(base_model, init_temp=init_temp, teacher_epsilon=1e-4)
    return endd_model
