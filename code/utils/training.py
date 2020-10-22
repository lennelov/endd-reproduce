import pickle
import numpy as np
import tensorflow.keras as keras
import settings

from models import vgg, cnn, endd, ensemble
from utils import evaluation, preprocessing, saveload, simplex, datasets, callbacks


def train_vgg_endd(train_images,
                   ensemble_model,
                   dataset_name,
                   batch_size=128,
                   n_epochs=90,
                   one_cycle_lr_policy=True,
                   init_lr=0.001,
                   cycle_length=60,
                   temp_annealing=True,
                   init_temp=10,
                   dropout_rate=0.3,
                   save_endd_dataset=False,
                   load_previous_endd_dataset=False):
    """Return a trained VGG ENDD model.

    The save_endd_dataset and load_previous_endd_dataset arguments are useful to avoid having to
    re-create the ensemble predictions.

    Args:
        train_images (np.array): Normalized train images, potentially including AUX data.
        ensemble_model (models.ensemble.Ensemble): Ensemble to distill.
        dataset_name (str): Name of dataset (required for loading correct model settings).
        batch_size (int): Batch size to use while training. Default 128,
        n_epochs (int): Number of epochs to train. Default 90.
        one_cycle_lr_policy (bool): True if one cycle LR policy should be used. Default True.
        init_lr (float): Initial learning rate for one cycle LR. Default 0.001.
        cycle_length (int): Epoch length in number of cycles. Default 60.
        temp_annealing (bool): True if temperature annealing should be used. Default True.
        init_temp (float): Initial temperature. Default 10.
        dropout_rate (float): Probability to drop node. Default 0.3.
        save_endd_dataset (bool): True if ENDD dataset should be saved (useful for speeding up
                                  repeated training with the same ensemble. Default False.
        load_previous_endd_dataset (bool): True if ENDD dataset should be loaded. The dataset loaded
                                           is the one saved the last time the function was run with
                                           save_endd_dataset=True.

    Returns:
        (keras.Model): Trained VGG ENDD model.
    """
    if load_previous_endd_dataset:
        with open('train_endd_dataset.pkl', 'rb') as file:
            train_images, train_ensemble_preds = pickle.load(file)
    else:
        # Get ensemble preds
        train_ensemble_preds = datasets.get_ensemble_preds(ensemble_model, train_images)

    # Save / Load pickled data. Generating ensemble preds takes a long time, so saving and
    # loading can make testing much more efficient.
    if save_endd_dataset:
        with open('train_endd_dataset.pkl', 'wb') as file:
            pickle.dump((train_images, train_ensemble_preds), file)

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
    base_model = vgg.get_model(dataset_name,
                               compile=False,
                               dropout_rate=dropout_rate,
                               softmax=False)
    endd_model = endd.get_model(base_model, init_temp=init_temp, teacher_epsilon=1e-4)

    # Train model
    endd_model.fit(data_generator, epochs=n_epochs, callbacks=endd_callbacks)

    return endd_model