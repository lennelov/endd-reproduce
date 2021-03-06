# Add parent dir to path to allow for parallel imports
import sys
import pathlib
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
#other functions
import settings
from utils import preprocessing, datasets, measures, pn_utils, callbacks
from models.small_net import get_model
import models.ensemble
from models import end, endd
from utils.create_toy_data import create_mixed_data
import utils.saveload as saveload
import pickle
import sklearn.metrics
import time

#####################################################
####             Helper functions               ####
####################################################


def plot_dataset(X, Y, aux=False, filename=None):
    """ Helper function for plotting the dataset in a similar way as Figure 2 in Malinin 2020"""

    if aux:
        classes = (-1, 0, 1, 2)
        lim = 1300
        marker_size = 5
    else:
        classes = (0, 1, 2)
        lim = 475
        marker_size = 20

    colors = {
        0: (245 / 255, 113 / 255, 137 / 255),
        1: (80 / 255, 174 / 255, 50 / 255),
        2: (59 / 255, 161 / 255, 234 / 255),
        -1: (254 / 255, 203 / 255, 82 / 255)
    }

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])

    for i in classes:
        idx = np.where(Y == i)
        sns.scatterplot(X[idx, 0].flatten(),
                        X[idx, 1].flatten(),
                        marker="s",
                        s=marker_size,
                        alpha=0.35,
                        color=colors[i],
                        ax=ax,
                        edgecolor="none")

    if filename is not None:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()


def grid_plot_helper(values,
                     v=None,
                     extent=(-2000, 2000, -2000, 2000),
                     overlay_data=None,
                     filename=None):
    """ Helper function for creating individual plots in the style of Figure 3 in Malinin 2020"""

    fig, ax = plt.subplots(figsize=(5, 5))

    if v is None:
        im = ax.imshow(values, extent=extent, origin='lower')
    else:
        im = ax.imshow(values, extent=extent, origin='lower', vmin=v[0], vmax=v[1])

    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.grid(False)

    if overlay_data is not None:
        sns.scatterplot(overlay_data[:, 0].flatten(),
                        overlay_data[:, 1].flatten(),
                        marker="s",
                        s=20,
                        edgecolor="None",
                        alpha=0.35,
                        color=(1, 1, 1),
                        ax=ax)

    if filename is not None:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()


def get_grid(size=2000, steps=1000):
    """ Creates a grid between -size and size, with steps in between, in the format [[-1, -1], [-1, -0.5], [-1, 0], [-1, 0.5], [-1, 1], [-0.5, -1], ...]"""
    x_grid = np.linspace(-size, size, steps)
    y_grid = np.linspace(-size, size, steps)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.array((xx.ravel(), yy.ravel())).T

    return grid


#####################################################
####             Experiments                    ####
####################################################


def generate_figure_2():
    ''' Reproduces the Figure 2 in Malinin (2020)'''

    (x_train, y_train), _ = datasets.get_dataset("spiral")
    (x_aux, y_aux), (_, _) = datasets.get_dataset("spiral_aux")

    x_train_aux = np.concatenate((x_train, x_aux), axis=0)
    y_train_aux = np.concatenate((y_train, y_aux), axis=0)

    plot_dataset(x_train, y_train, aux=False, filename="plots/" + str(i) + "/2a.png")
    plot_dataset(x_train_aux, y_train_aux, aux=True, filename="plots/" + str(i) + "/2b.png")


def train_models():
    '''Trains an ensemble of models on the spiral dataset.'''

    MODEL_TYPE = "small_net"
    ENSEMBLE_SAVE_NAME = "small_net"
    DATASET_NAME = "spiral"
    NAME_START_NUMBER = 0
    N_MODELS = 100
    N_EPOCHS = 85

    # Import data
    (x_train, y_train), (x_test, y_test) = datasets.get_dataset(DATASET_NAME)
    y_train_one_hot = tf.one_hot(y_train.reshape((-1,)), settings.DATASET_N_CLASSES[DATASET_NAME])
    y_test_one_hot = tf.one_hot(y_test.reshape((-1,)), settings.DATASET_N_CLASSES[DATASET_NAME])

    # Train models
    model_module = settings.MODEL_MODULES[MODEL_TYPE]
    saved_model_names = []
    try:
        for i in range(N_MODELS):
            # Get model
            model = model_module.get_model(dataset_name=DATASET_NAME, compile=True)

            # Train model
            model.fit(x_train,
                      y_train_one_hot,
                      validation_data=(x_test, y_test_one_hot),
                      epochs=N_EPOCHS,
                      verbose=2)
            print("Model {} finished training.".format(i))

            # Save model
            model_name = "{}_{}_{}".format(ENSEMBLE_SAVE_NAME, DATASET_NAME, i)
            saveload.save_tf_model(model, model_name)
            saved_model_names.append(model_name)

    finally:
        append_model_names = NAME_START_NUMBER > 0
        saveload.update_ensemble_names(ENSEMBLE_SAVE_NAME,
                                       DATASET_NAME,
                                       saved_model_names,
                                       append=append_model_names)


def predict_ensemble():
    """Predicts and saves the predictions from the ensemble"""

    # Import models
    ENSEMBLE_SAVE_NAME = 'small_net'  # Name that the ensemble models will be saved with
    DATASET_NAME = 'spiral'  # Name of dataset models were trained with
    AUX_DATASET_NAME = "spiral_aux"
    ensemble_model_names = saveload.get_ensemble_model_names()
    model_names = ensemble_model_names[ENSEMBLE_SAVE_NAME][DATASET_NAME]
    wrapped_models = [models.ensemble.KerasLoadsWhole(name) for name in model_names]

    # Build ensemble
    ensemble = models.ensemble.Ensemble(wrapped_models)
    print(ensemble)

    # Load data
    (x_train, y_train), (x_test, y_test) = datasets.get_dataset(DATASET_NAME)
    (x_aux, y_aux), (_, _) = datasets.get_dataset(AUX_DATASET_NAME)
    x_train_aux = np.concatenate((x_train, x_aux), axis=0)
    y_train_aux = np.concatenate((y_train, y_aux), axis=0)
    x_train_aux_20 = np.concatenate((x_train, x_aux[:20]), axis=0)
    y_train_aux_20 = np.concatenate((y_train, y_aux[:20]), axis=0)
    grid = get_grid(size=2000, steps=1000)

    # Predict with ensemble
    ensemble_logits_train_aux = ensemble.predict(x_train_aux)
    ensemble_logits_train_aux_20 = ensemble.predict(x_train_aux_20)
    ensemble_logits_train = ensemble.predict(x_train)
    ensemble_logits_test = ensemble.predict(x_test)
    ensemble_logits_grid = ensemble.predict(grid)

    # Save to file
    with open('train_aux_small_net_spiral.pkl', 'wb') as file:
        pickle.dump((x_train_aux, y_train_aux, ensemble_logits_train_aux), file)
    with open('train_aux_20_small_net_spiral.pkl', 'wb') as file:
        pickle.dump((x_train_aux_20, y_train_aux_20, ensemble_logits_train_aux_20), file)
    with open('train_small_net_spiral.pkl', 'wb') as file:
        pickle.dump((x_train, y_train, ensemble_logits_train), file)
    with open('test_small_net_spiral.pkl', 'wb') as file:
        pickle.dump((x_test, y_test, ensemble_logits_test), file)
    with open('grid_small_net_spiral_1000.pkl', 'wb') as file:
        pickle.dump((grid, 0, ensemble_logits_grid), file, protocol=4)


def plot_decision_boundary():
    """Plots a decision boundary using the grid."""

    # Load data
    with open('grid_small_net_spiral_1000.pkl', 'rb') as file:
        x_grid, _, ensemble_logits_grid = pickle.load(file)
    grid_size = int(np.sqrt(x_grid.shape[0]))

    (x_train, y_train), _ = datasets.get_dataset("spiral")

    # For ensemble
    prediction_grid = np.reshape(np.argmax(np.mean(ensemble_logits_grid, axis=0), axis=1),
                                 (grid_size, grid_size))

    # Without ensemble
    #prediction_grid = np.reshape(np.argmax(ensemble_logits_grid, axis = 1), (grid_size, grid_size))

    # Plot decision boundary

    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(prediction_grid, extent=(-2000, 2000, -2000, 2000), origin='lower')
    fig.colorbar(im)

    colors = {
        0: (245 / 255, 113 / 255, 137 / 255),
        1: (80 / 255, 174 / 255, 50 / 255),
        2: (59 / 255, 161 / 255, 234 / 255),
        -1: (254 / 255, 203 / 255, 82 / 255)
    }

    for i in range(3):
        idx = np.where(y_train == i)
        sns.scatterplot(x_train[idx, 0].flatten(),
                        x_train[idx, 1].flatten(),
                        marker="s",
                        s=20,
                        alpha=0.35,
                        color=colors[i],
                        ax=ax)

    ax.set_xlim((-500, 500))
    ax.set_ylim((-500, 500))
    plt.show()


def train_end():
    """Trains an END and and END_AUX model on the ensemble predictions"""

    # Name
    ENSEMBLE_SAVE_NAME = 'small_net'  # Name that the ensemble models will be saved with
    DATASET_NAME = 'spiral'  # Name of dataset models were trained with
    MODEL_SAVE_NAME = "end_small_net_spiral"
    MODEL_SAVE_NAME_AUX = "end_AUX_small_net_spiral"

    # Load data
    with open('train_small_net_spiral.pkl', 'rb') as file:
        x_train, y_train, ensemble_logits_train = pickle.load(file)
    with open('train_aux_small_net_spiral.pkl', 'rb') as file:
        x_train_aux, y_train_aux, ensemble_logits_train_aux = pickle.load(file)

    # Build ENDD model
    base_model = get_model(DATASET_NAME, compile=False)
    end_model = end.get_model(base_model, init_temp=1)

    base_model_AUX = get_model(DATASET_NAME, compile=False)
    end_model_AUX = end.get_model(base_model_AUX, init_temp=1)

    # Train model
    end_model.fit(x_train, np.transpose(ensemble_logits_train, (1, 0, 2)), epochs=150)
    end_model_AUX.fit(x_train_aux, np.transpose(ensemble_logits_train_aux, (1, 0, 2)), epochs=500)

    # Save model
    saveload.save_tf_model(end_model, MODEL_SAVE_NAME)
    saveload.save_tf_model(end_model_AUX, MODEL_SAVE_NAME_AUX)

    # Evaluate model
    _, (x_test, y_test) = datasets.get_dataset("spiral")
    logits = end_model.predict(x_test)
    logits_aux = end_model_AUX.predict(x_test)
    print(np.argmax(logits, axis=1))
    print(np.argmax(logits_aux, axis=1))
    print(y_test)
    print(sklearn.metrics.accuracy_score(y_test, np.argmax(logits, axis=1)))
    print(sklearn.metrics.accuracy_score(y_test, np.argmax(logits_aux, axis=1)))


def train_endd():
    """Trains an ENDD and and ENDD_AUX model on the ensemble predictions"""

    # Name
    ENSEMBLE_SAVE_NAME = 'small_net'  # Name that the ensemble models will be saved with
    DATASET_NAME = 'spiral'  # Name of dataset models were trained with
    MODEL_SAVE_NAME = "endd_small_net_spiral"
    MODEL_SAVE_NAME_AUX = "endd_AUX_small_net_spiral"
    MODEL_SAVE_NAME_AUX_20 = "endd_AUX_20_small_net_spiral"
    MODEL_SAVE_NAME_AUX_ANN = "endd_AUX_ANN_small_net_spiral"
    MODEL_SAVE_NAME_AUX_T25 = "endd_AUX_T25_small_net_spiral"

    # Load data
    with open('train_small_net_spiral.pkl', 'rb') as file:
        x_train, y_train, ensemble_logits_train = pickle.load(file)
    with open('train_aux_small_net_spiral.pkl', 'rb') as file:
        x_train_aux, y_train_aux, ensemble_logits_train_aux = pickle.load(file)
    with open('train_aux_20_small_net_spiral.pkl', 'rb') as file:
        x_train_aux_20, y_train_aux_20, ensemble_logits_train_aux_20 = pickle.load(file)

    # Build ENDD model
    base_model = get_model(DATASET_NAME, compile=False)
    endd_model = endd.get_model(base_model, init_temp=1, teacher_epsilon=1e-4)

    base_model_AUX = get_model(DATASET_NAME, compile=False)
    base_model_AUX_20 = get_model(DATASET_NAME, compile=False)
    base_model_AUX_ANN = get_model(DATASET_NAME, compile=False)
    base_model_AUX_T25 = get_model(DATASET_NAME, compile=False)

    endd_model_AUX = endd.get_model(base_model_AUX, init_temp=1, teacher_epsilon=1e-4)
    endd_model_AUX_20 = endd.get_model(base_model_AUX_20, init_temp=1, teacher_epsilon=1e-4)
    endd_model_AUX_ANN = endd.get_model(base_model_AUX_ANN, init_temp=2.5, teacher_epsilon=1e-4)
    endd_model_AUX_T25 = endd.get_model(base_model_AUX_T25, init_temp=2.5, teacher_epsilon=1e-4)

    # Train model
    #endd_model.fit(x_train, np.transpose(ensemble_logits_train, (1, 0, 2)), epochs=500)
    #endd_model_AUX.fit(x_train_aux, np.transpose(ensemble_logits_train_aux, (1, 0, 2)), epochs=500)
    #endd_model_AUX_20.fit(x_train_aux_20, np.transpose(ensemble_logits_train_aux_20, (1, 0, 2)), epochs=500)
    endd_model_AUX_ANN.fit(x_train_aux, np.transpose(ensemble_logits_train_aux, (1, 0, 2)), epochs=500, 
        callbacks = [callbacks.TemperatureAnnealing(init_temp = 2.5, cycle_length = 400, epochs = 500)])
    endd_model_AUX_T25.fit(x_train_aux, np.transpose(ensemble_logits_train_aux, (1, 0, 2)), epochs=500)
    
    # Save model
    saveload.save_tf_model(endd_model, MODEL_SAVE_NAME)
    saveload.save_tf_model(endd_model_AUX, MODEL_SAVE_NAME_AUX)
    saveload.save_tf_model(endd_model_AUX_20, MODEL_SAVE_NAME_AUX_20)
    saveload.save_tf_model(endd_model_AUX_ANN, MODEL_SAVE_NAME_AUX_ANN)
    saveload.save_tf_model(endd_model_AUX_T25, MODEL_SAVE_NAME_AUX_T25)

    # Evaluate model
    _, (x_test, y_test) = datasets.get_dataset("spiral")
    logits = endd_model_AUX.predict(x_test)
    print(np.argmax(logits, axis=1))
    print(y_test)
    print(sklearn.metrics.accuracy_score(y_test, np.argmax(logits, axis=1)))


def predict_endd():
    """Predicts and saves the predictions of the ENDD-models to file"""

    # Load model
    MODEL_SAVE_NAMES = ["endd_small_net_spiral", 
    "endd_AUX_small_net_spiral", 
    "endd_AUX_20_small_net_spiral", 
    "endd_AUX_ANN_small_net_spiral",
    "endd_AUX_T25_small_net_spiral"]
    PREDICT_SAVE_NAMES = ["endd", "endd_AUX", "endd_AUX_20", "endd_AUX_ANN", "endd_AUX_T25"]

    # Loop for aux or no aux
    for i in range(len(MODEL_SAVE_NAMES)):
        print(i)

        MODEL_SAVE_NAME = MODEL_SAVE_NAMES[i]
        PREDICT_SAVE_NAME = PREDICT_SAVE_NAMES[i]

        endd_model = saveload.load_tf_model(MODEL_SAVE_NAME, compile=False)
        endd_model = endd.get_model(endd_model, init_temp=1, teacher_epsilon=1e-4)

        # Load data
        (x_train, y_train), (x_test, y_test) = datasets.get_dataset("spiral")
        grid = get_grid(size=2000, steps=1000)

        # Predict
        endd_logits_train = endd_model.predict(x_train)
        endd_logits_test = endd_model.predict(x_test)
        endd_logits_grid = endd_model.predict(grid)

        with open('train_small_net_spiral_{}.pkl'.format(PREDICT_SAVE_NAME), 'wb') as file:
            pickle.dump((x_train, y_train, endd_logits_train), file)
        with open('test_small_net_spiral_{}.pkl'.format(PREDICT_SAVE_NAME), 'wb') as file:
            pickle.dump((x_test, y_test, endd_logits_test), file)
        with open('grid_small_net_spiral_{}.pkl'.format(PREDICT_SAVE_NAME), 'wb') as file:
            pickle.dump((grid, 0, endd_logits_grid), file)


def predict_end():
    """Predicts and saves the predictions of the END-models to file"""

    # Load model
    MODEL_SAVE_NAMES = ["end_small_net_spiral", "end_AUX_small_net_spiral"]
    PREDICT_SAVE_NAMES = ["end", "end_AUX"]

    # Loop for aux or no aux
    for i in range(2):
        print(i)

        MODEL_SAVE_NAME = MODEL_SAVE_NAMES[i]
        PREDICT_SAVE_NAME = PREDICT_SAVE_NAMES[i]

        endd_model = saveload.load_tf_model(MODEL_SAVE_NAME, compile=False)
        endd_model = endd.get_model(endd_model, init_temp=1, teacher_epsilon=1e-4)

        # Load data
        (x_train, y_train), (x_test, y_test) = datasets.get_dataset("spiral")
        grid = get_grid(size=2000, steps=1000)

        # Predict
        endd_logits_train = endd_model.predict(x_train)
        endd_logits_test = endd_model.predict(x_test)
        endd_logits_grid = endd_model.predict(grid)

        with open('train_small_net_spiral_{}.pkl'.format(PREDICT_SAVE_NAME), 'wb') as file:
            pickle.dump((x_train, y_train, endd_logits_train), file)
        with open('test_small_net_spiral_{}.pkl'.format(PREDICT_SAVE_NAME), 'wb') as file:
            pickle.dump((x_test, y_test, endd_logits_test), file)
        with open('grid_small_net_spiral_{}.pkl'.format(PREDICT_SAVE_NAME), 'wb') as file:
            pickle.dump((grid, 0, endd_logits_grid), file)

def plot_grids(i):
    """Function for recreating Figure 3 in Malinin 2020"""

    # Plotting settings
    v = [0, 1.3]

    # First plot ensemble

    with open('grid_small_net_spiral_1000.pkl', 'rb') as file:
        x_grid, _, ensemble_logits_grid = pickle.load(file)
    grid_size = int(np.sqrt(x_grid.shape[0]))

    unct_tot = np.reshape(measures.entropy_of_expected(ensemble_logits_grid, logits=True),
                          (grid_size, grid_size))
    unct_data = np.reshape(measures.expected_entropy(ensemble_logits_grid, logits=True),
                           (grid_size, grid_size))
    unct_know = unct_tot - unct_data

    grid_plot_helper(unct_tot, v=v, filename="plots/" + str(i) + "/3a.png")
    grid_plot_helper(unct_data, v=v, filename="plots/" + str(i) + "/3b.png")
    grid_plot_helper(unct_know, v=v, filename="plots/" + str(i) + "/3c.png")

    # Then plot ENDD

    with open('grid_small_net_spiral_endd.pkl', 'rb') as file:
        x_grid, _, endd_logits_grid = pickle.load(file)
    grid_size = int(np.sqrt(x_grid.shape[0]))

    endd_probs_grid = pn_utils.pn_logits_to_probs(endd_logits_grid)
    unct_tot = np.reshape(measures.entropy_of_expected(endd_probs_grid, logits=False),
                          (grid_size, grid_size))
    unct_data = np.reshape(measures.expected_entropy_pn(endd_logits_grid), (grid_size, grid_size))
    unct_know = unct_tot - unct_data

    grid_plot_helper(unct_tot, v=v, filename="plots/" + str(i) + "/3d.png")
    grid_plot_helper(unct_data, v=v, filename="plots/" + str(i) + "/3e.png")
    grid_plot_helper(unct_know, v=v, filename="plots/" + str(i) + "/3f.png")

    # Then plot ENDD_AUX

    with open('grid_small_net_spiral_endd_AUX.pkl', 'rb') as file:
        x_grid, _, endd_logits_grid = pickle.load(file)
    grid_size = int(np.sqrt(x_grid.shape[0]))

    endd_probs_grid = pn_utils.pn_logits_to_probs(endd_logits_grid)
    unct_tot = np.reshape(measures.entropy_of_expected(endd_probs_grid, logits=False),
                          (grid_size, grid_size))
    unct_data = np.reshape(measures.expected_entropy_pn(endd_logits_grid), (grid_size, grid_size))
    unct_know = unct_tot - unct_data

    grid_plot_helper(unct_tot, v=v, filename="plots/" + str(i) + "/3g.png")
    grid_plot_helper(unct_data, v=v, filename="plots/" + str(i) + "/3h.png")
    grid_plot_helper(unct_know, v=v, filename="plots/" + str(i) + "/3i.png")

    # Then plot ENDD_AUX_ANN

    with open('grid_small_net_spiral_endd_AUX_ANN.pkl', 'rb') as file:
        x_grid, _, endd_logits_grid = pickle.load(file)
    grid_size = int(np.sqrt(x_grid.shape[0]))

    endd_probs_grid = pn_utils.pn_logits_to_probs(endd_logits_grid)
    unct_tot = np.reshape(measures.entropy_of_expected(endd_probs_grid, logits=False),
                          (grid_size, grid_size))
    unct_data = np.reshape(measures.expected_entropy_pn(endd_logits_grid), (grid_size, grid_size))
    unct_know = unct_tot - unct_data

    grid_plot_helper(unct_tot, v=v, filename="plots/" + str(i) + "/3j.png")
    grid_plot_helper(unct_data, v=v, filename="plots/" + str(i) + "/3k.png")
    grid_plot_helper(unct_know, v=v, filename="plots/" + str(i) + "/3l.png")

    # Then plot ENDD_AUX_T25

    with open('grid_small_net_spiral_endd_AUX_T25.pkl', 'rb') as file:
        x_grid, _, endd_logits_grid = pickle.load(file)
    grid_size = int(np.sqrt(x_grid.shape[0]))

    endd_probs_grid = pn_utils.pn_logits_to_probs(endd_logits_grid)
    unct_tot = np.reshape(measures.entropy_of_expected(endd_probs_grid, logits=False),
                          (grid_size, grid_size))
    unct_data = np.reshape(measures.expected_entropy_pn(endd_logits_grid), (grid_size, grid_size))
    unct_know = unct_tot - unct_data

    grid_plot_helper(unct_tot, v=v, filename="plots/" + str(i) + "/3m.png")
    grid_plot_helper(unct_data, v=v, filename="plots/" + str(i) + "/3n.png")
    grid_plot_helper(unct_know, v=v, filename="plots/" + str(i) + "/3o.png")

    # Then plot ENDD_AUX_20

    with open('grid_small_net_spiral_endd_AUX_20.pkl', 'rb') as file:
        x_grid, _, endd_logits_grid = pickle.load(file)
    grid_size = int(np.sqrt(x_grid.shape[0]))

    endd_probs_grid = pn_utils.pn_logits_to_probs(endd_logits_grid)
    unct_tot = np.reshape(measures.entropy_of_expected(endd_probs_grid, logits=False),
                          (grid_size, grid_size))
    unct_data = np.reshape(measures.expected_entropy_pn(endd_logits_grid), (grid_size, grid_size))
    unct_know = unct_tot - unct_data

    grid_plot_helper(unct_tot, v=v, filename="plots/" + str(i) + "/3p.png")
    grid_plot_helper(unct_data, v=v, filename="plots/" + str(i) + "/3q.png")
    grid_plot_helper(unct_know, v=v, filename="plots/" + str(i) + "/3r.png")

def get_metrics():
    """Calculates some interesting metrics of the ensemble, for recreating table 2 in Malinin 2020."""

    # Load data
    with open('train_small_net_spiral.pkl', 'rb') as file:
        x_train, y_train, ensemble_logits_train = pickle.load(file)
    with open('test_small_net_spiral.pkl', 'rb') as file:
        x_test, y_test, ensemble_logits_test = pickle.load(file)

    # Calculate error
    ensemble_preds_train = np.argmax(ensemble_logits_train, axis=-1)
    ensemble_preds_test = np.argmax(ensemble_logits_test, axis=-1)

    nr_models = ensemble_preds_train.shape[0]

    err_train = np.zeros(nr_models)
    err_test = np.zeros(nr_models)

    for i in range(nr_models):
        err_train[i] = 1 - sklearn.metrics.accuracy_score(y_train, ensemble_preds_train[i, :])
        err_test[i] = 1 - sklearn.metrics.accuracy_score(y_test, ensemble_preds_test[i, :])

    # Plot histogram
    '''
    plt.hist(err_train, bins = 60, alpha = 0.5, density = True)
    plt.hist(err_test, bins = 60, alpha = 0.5, density = True)
    plt.xlabel("Classification error")
    plt.xlabel("Probability density")
    plt.xlim([0, 0.25])
    plt.legend(["Train", "Test"])
    plt.title("Small Net on spiral data, classification error")
    plt.show()
    '''

    # Print ind-metric
    print("# Ind metrics")
    names = ["Train", "Test"]
    for i, err in enumerate((err_train, err_test)):
        print(names[i])
        print("Mean: {}".format(round(100 * np.mean(err), 4)))
        print("Plus-minus: {}".format(round(100 * 1.96 * np.std(err) / np.sqrt(nr_models), 4)))
        print("Min-range: {}".format(
            round(100 * (np.mean(err) + 1.96 * np.std(err) / np.sqrt(nr_models)), 4)))
        print("Max-range: {}".format(
            round(100 * (np.mean(err) - 1.96 * np.std(err) / np.sqrt(nr_models)), 4)))
        print()

    # Print ensemble-metrics
    # Do you need to softmax before this? Probably no significant difference

    ensemble_mean_preds_train = np.argmax(np.mean(ensemble_logits_train, axis=0), axis=1)
    ensemble_mean_preds_test = np.argmax(np.mean(ensemble_logits_test, axis=0), axis=1)

    ensemble_err_train = 1 - sklearn.metrics.accuracy_score(y_train, ensemble_mean_preds_train)
    ensemble_err_test = 1 - sklearn.metrics.accuracy_score(y_test, ensemble_mean_preds_test)

    # Ensemble metrics
    print("# Ensemble metrics")
    print("Train: {}".format(round(100 * ensemble_err_train, 4)))
    print("Test: {}".format(round(100 * ensemble_err_test, 4)))
    print()

    # END metrics
    print("# END metrics")

    with open('train_small_net_spiral_end.pkl', 'rb') as file:
        x_train, y_train, end_logits_train = pickle.load(file)

    with open('test_small_net_spiral_end.pkl', 'rb') as file:
        x_test, y_test, end_logits_test = pickle.load(file)

    print("Train: {}".format(
        round(
            100 -
            100 * sklearn.metrics.accuracy_score(y_train, np.argmax(end_logits_train, axis=1)), 4)))
    print("Test: {}".format(
        round(
            100 - 100 * sklearn.metrics.accuracy_score(y_test, np.argmax(end_logits_test, axis=1)),
            4)))

    print()

    # END_AUX metrics
    print("# END_AUX metrics")

    with open('train_small_net_spiral_end_AUX.pkl', 'rb') as file:
        x_train, y_train, end_AUX_logits_train = pickle.load(file)

    with open('test_small_net_spiral_end_AUX.pkl', 'rb') as file:
        x_test, y_test, end_AUX_logits_test = pickle.load(file)

    print("Train: {}".format(
        round(
            100 -
            100 * sklearn.metrics.accuracy_score(y_train, np.argmax(end_AUX_logits_train, axis=1)),
            4)))
    print("Test: {}".format(
        round(
            100 -
            100 * sklearn.metrics.accuracy_score(y_test, np.argmax(end_AUX_logits_test, axis=1)),
            4)))

    print()

    # ENDD metrics
    print("# ENDD metrics")

    with open('train_small_net_spiral_endd.pkl', 'rb') as file:
        x_train, y_train, endd_logits_train = pickle.load(file)

    with open('test_small_net_spiral_endd.pkl', 'rb') as file:
        x_test, y_test, endd_logits_test = pickle.load(file)

    print("Train: {}".format(
        round(
            100 -
            100 * sklearn.metrics.accuracy_score(y_train, np.argmax(endd_logits_train, axis=1)),
            4)))
    print("Test: {}".format(
        round(
            100 - 100 * sklearn.metrics.accuracy_score(y_test, np.argmax(endd_logits_test, axis=1)),
            4)))

    print()

    # ENDD_AUX metrics
    print("# ENDD_AUX metrics")

    with open('train_small_net_spiral_endd_AUX.pkl', 'rb') as file:
        x_train, y_train, endd_AUX_logits_train = pickle.load(file)

    with open('test_small_net_spiral_endd_AUX.pkl', 'rb') as file:
        x_test, y_test, endd_AUX_logits_test = pickle.load(file)

    print("Train: {}".format(
        round(
            100 -
            100 * sklearn.metrics.accuracy_score(y_train, np.argmax(endd_AUX_logits_train, axis=1)),
            4)))
    print("Test: {}".format(
        round(
            100 -
            100 * sklearn.metrics.accuracy_score(y_test, np.argmax(endd_AUX_logits_test, axis=1)),
            4)))

    print()

    # ENDD_AUX metrics
    print("# ENDD_AUX_ANN metrics")

    with open('train_small_net_spiral_endd_AUX_ANN.pkl', 'rb') as file:
        x_train, y_train, endd_AUX_logits_train = pickle.load(file)

    with open('test_small_net_spiral_endd_AUX_ANN.pkl', 'rb') as file:
        x_test, y_test, endd_AUX_logits_test = pickle.load(file)

    print("Train: {}".format(
        round(
            100 -
            100 * sklearn.metrics.accuracy_score(y_train, np.argmax(endd_AUX_logits_train, axis=1)),
            4)))
    print("Test: {}".format(
        round(
            100 -
            100 * sklearn.metrics.accuracy_score(y_test, np.argmax(endd_AUX_logits_test, axis=1)),
            4)))

    print()

    # ENDD_AUX metrics
    print("# ENDD_AUX_T25 metrics")

    with open('train_small_net_spiral_endd_AUX_T25.pkl', 'rb') as file:
        x_train, y_train, endd_AUX_logits_train = pickle.load(file)

    with open('test_small_net_spiral_endd_AUX_T25.pkl', 'rb') as file:
        x_test, y_test, endd_AUX_logits_test = pickle.load(file)

    print("Train: {}".format(
        round(
            100 -
            100 * sklearn.metrics.accuracy_score(y_train, np.argmax(endd_AUX_logits_train, axis=1)),
            4)))
    print("Test: {}".format(
        round(
            100 -
            100 * sklearn.metrics.accuracy_score(y_test, np.argmax(endd_AUX_logits_test, axis=1)),
            4)))

    print()

    # ENDD_AUX metrics
    print("# ENDD_AUX_20 metrics")

    with open('train_small_net_spiral_endd_AUX_20.pkl', 'rb') as file:
        x_train, y_train, endd_AUX_logits_train = pickle.load(file)

    with open('test_small_net_spiral_endd_AUX_20.pkl', 'rb') as file:
        x_test, y_test, endd_AUX_logits_test = pickle.load(file)

    print("Train: {}".format(
        round(
            100 -
            100 * sklearn.metrics.accuracy_score(y_train, np.argmax(endd_AUX_logits_train, axis=1)),
            4)))
    print("Test: {}".format(
        round(
            100 -
            100 * sklearn.metrics.accuracy_score(y_test, np.argmax(endd_AUX_logits_test, axis=1)),
            4)))

    print()


#####################################################
####             Main                           ####
####################################################

if __name__ == '__main__':
    start = time.time()

    #generate_figure_2()
    #train_models()
    #predict_ensemble()
    #plot_decision_boundary()
    for i in range(20):
        #train_end()
        train_endd()
        predict_endd()
        predict_end()
        plot_grids(i)
        get_metrics()

    end = time.time()
    print("Time elapsed: ", end - start)
