import sys
import pathlib
import os
#parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("/home/lennelov/Repositories/endd-reproduce/code")
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np

import settings
from models import ensemble, endd, cnn_priorNet
from utils import preprocessing, saveload, simplex, losses, training, measures
from utils.losses import DirichletKL
from math import gamma
from utils import datasets
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ======== PARAMETERS ========

DATASET_NAME = 'cifar10'
THREE_CLASS_DATASET_NAME = 'cifar10_3_class'
ENSM_MODEL_NAME, ENSM_N_MODELS = 'vgg_3_class', 100
ENDD_MODEL_NAME  = "endd_vgg_cifar10_3_class"
LOAD_PREVIOUS_ENSM_PREDS = True
LOAD_PREVIOUS_ENDD_PREDS = True

PLOT_COLLAGE = True
PLOT_SELECTED = False


# ======== FUNCTIONS =========

def prepare_ensemble_prediction(x):
    x = np.exp(np.float64(x))
    x_2 = np.expand_dims(np.sum(x, axis=1), axis=1)
    return x / x_2


def prepare_prediction(x):
    x = np.exp(np.float64(x))
    x_2 = np.expand_dims(np.sum(x, axis=1), axis=1)
    return x / x_2


def plot_img(img):
    im = plt.imshow(img)
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)


def plot_points(ensm_output):
    alphas = prepare_ensemble_prediction(ensm_output)
    simplex.plot_points(alphas, alpha=0.3, clip_on=False)


def plot_pdf(endd_output):
    logits = np.float64(endd_output)
    exped = np.exp(np.array(logits))
    exped[exped > 100] = 100
    exped = exped.reshape(3,)
    simplex.draw_pdf_contours(simplex.Dirichlet(np.float64(exped)), nlevels=200, subdiv=3, log_probs=True)


def compare_simplex(ensm_data_uncertain, ensm_know_uncertain, ensm_certain, ensm_noise,
                    endd_data_uncertain, endd_know_uncertain, endd_certain, endd_noise,
                    imgs_in, filename = None):
    ensm_output = [
        ensm_data_uncertain,
        ensm_know_uncertain,
        ensm_certain,
        ensm_noise
    ]
    endd_output = [
        endd_data_uncertain,
        endd_know_uncertain,
        endd_certain,
        endd_noise
    ]
    font = {
        'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 16,
    }
    plt.style.use('seaborn-white')
    plt.figure(num=None, figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    models = ["data uncertain deer", "knowledge uncertain plane", "certain deer", "random noise"]

    plt.axis('off')
    for i in range(0, 4):
        plt.subplot(3, 4, i+1)
        plt.title(models[i], fontsize=18, ha='center')
        plot_img(imgs_in[i])

        plt.subplot(3, 4, i + 5)
        plot_points(ensm_output[i])

        plt.subplot(3, 4, i + 9)
        plot_pdf(endd_output[i])

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.savefig('compare_plot.png')
    plt.show()


def plot_collage(images, ensm_predictions, endd_predictions):
    n_cols = 10
    n_imgs = len(images)
    indices = np.random.randint(0, high=n_imgs, size=(n_cols,))
    subplot_size = 2
    fig = plt.figure(figsize=(n_cols*subplot_size, 3*subplot_size*0.8))
    gs1 = gridspec.GridSpec(3, n_cols)
    gs1.update(wspace=0.1, hspace = 0.025)
    plt.margins(0.5)
    for i, index in enumerate(indices):
        img = (images[index, :, :] + 1) / 2
        plt.subplot(gs1[i])
        plot_img(img)

        ensm_prediction = ensm_predictions[:, index, :]
        plt.subplot(gs1[i + n_cols])
        plot_points(ensm_prediction)

        endd_prediction = endd_predictions[index, :]
        plt.subplot(gs1[i + 2*n_cols])
        plot_pdf(endd_prediction)


# ======== PREPARE IMAGES =========

# Load test images
(_, _), (test_images, test_labels) = datasets.get_dataset(DATASET_NAME)
raw_test_images = test_images
test_images = preprocessing.normalize_minus_one_to_one(test_images, min=0, max=255)

(aux_images, _), (_, _) = datasets.get_dataset('cifar100')
aux_images = preprocessing.normalize_minus_one_to_one(aux_images, min=0, max=255)

noise_img = np.random.randn(1, 32, 32, 3)

# ======== PREDICTIONS =========

if LOAD_PREVIOUS_ENSM_PREDS:
    with open("ensm_preds.pkl", 'rb') as file:
        ensm_preds = pickle.load(file)
    with open("ensm_preds_noise.pkl", 'rb') as file:
        ensm_preds_noise = pickle.load(file)
else:
    # Load ensemble
    ensemble_model_names = saveload.get_ensemble_model_names()
    model_names = ensemble_model_names[ENSM_MODEL_NAME][THREE_CLASS_DATASET_NAME][:ENSM_N_MODELS]
    models = [ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_names]
    ensm_model = ensemble.Ensemble(models)

    # Predict ensemble
    ensm_preds = ensm_model.predict(test_images)
    with open("ensm_preds.pkl", 'wb') as file:
        pickle.dump((ensm_preds), file)
    ensm_preds_noise = ensm_model.predict(noise_img)
    with open("ensm_preds_noise.pkl", 'wb') as file:
        pickle.dump((ensm_preds_noise), file)

if LOAD_PREVIOUS_ENDD_PREDS:
    with open("endd_preds.pkl", 'rb') as file:
        endd_preds = pickle.load(file)
    with open("endd_preds_noise.pkl", 'rb') as file:
        endd_preds_noise = pickle.load(file)
else:
    # Load endd
    endd_base_model = saveload.load_tf_model(ENDD_MODEL_NAME, compile=False)
    endd_model = endd.get_model(endd_base_model, init_temp=1, teacher_epsilon=1e-4)

    # Predict endd
    endd_preds = endd_model.predict(test_images)
    with open("endd_preds.pkl", 'wb') as file:
        pickle.dump((endd_preds), file)
    endd_preds_noise = endd_model.predict(noise_img)
    with open("endd_preds_noise.pkl", 'wb') as file:
        pickle.dump((endd_preds_noise), file)


# Plot random images
if PLOT_COLLAGE:
    in_indices = np.where((test_labels == 4) | (test_labels == 5) | (test_labels == 7))[0]
    out_indices = np.where((test_labels != 4) & (test_labels != 5) & (test_labels != 7))[0]
    print(in_indices)
    print(test_images.shape)
    print(ensm_preds.shape)
    print(endd_preds.shape)
    plot_collage(test_images[in_indices, :, :, :], ensm_preds[:, in_indices, :], endd_preds[in_indices, :])
    plt.show()
    plot_collage(test_images[out_indices, :, :, :], ensm_preds[:, out_indices, :], endd_preds[out_indices, :])
    plt.show()

if PLOT_SELECTED:
    # Pick out plane images and preds
    indx_plane = test_labels == 0
    indx_plane = indx_plane.flatten()
    imgs_plane = raw_test_images[indx_plane]
    ensm_preds_plane = ensm_preds[:, indx_plane, :]
    endd_preds_plane = endd_preds[indx_plane, :]

    # Pick out deer images and preds
    indx_deer = test_labels == 4
    indx_deer = indx_deer.flatten()
    imgs_deer = raw_test_images[indx_deer]
    ensm_preds_deer = ensm_preds[:, indx_deer, :]
    endd_preds_deer = endd_preds[indx_deer, :]

    unct_tot_deer = measures.entropy_of_expected(ensm_preds_deer, True)
    unct_data_deer = measures.expected_entropy(ensm_preds_deer, True)
    unct_know_deer = unct_tot_deer - unct_data_deer
    print("Five most certain deer: {}".format(np.argsort(unct_tot_deer)[:5]))  # Five most certain
    print("Five most data uncertain deer: {}".format(np.argsort(unct_data_deer)[-5:]))  # Five most data uncertain
    print("Five most knowledge uncertain deer: {}".format(np.argsort(unct_know_deer)[-5:]))  # Five most knowledge uncertain

    unct_tot_plane = measures.entropy_of_expected(ensm_preds_plane, True)
    unct_data_plane = measures.expected_entropy(ensm_preds_plane, True)
    unct_know_plane = unct_tot_plane - unct_data_plane
    print("Five most certain plain: {}".format(np.argsort(unct_tot_plane)[:5]))  # Five most certain
    print("Five most data uncertain plane: {}".format(np.argsort(unct_data_plane)[-5:]))  # Five most data uncertain
    print("Five most knowledge uncertain plane: {}".format(np.argsort(unct_know_plane)[-5:]))  # Five most knowledge uncertain

    # Hard coded image indices chosen based on above printouts
    data_uncertain_deer = 799
    knowledge_uncertain_plane = 653
    certain_deer = 682

    ensm_data_uncertain = ensm_preds_deer[:, data_uncertain_deer, :]
    ensm_know_uncertain = ensm_preds_plane[:, knowledge_uncertain_plane, :]
    ensm_certain = ensm_preds_deer[:, certain_deer, :]
    ensm_noise = ensm_preds_noise[:, 0, :]
    endd_data_uncertain = endd_preds_deer[data_uncertain_deer, :]
    endd_know_uncertain = endd_preds_plane[knowledge_uncertain_plane, :]
    endd_certain = endd_preds_deer[certain_deer, :]
    endd_noise = endd_preds_noise[0, :].flatten()


    imgs_in = [imgs_deer[data_uncertain_deer], imgs_plane[knowledge_uncertain_plane], imgs_deer[certain_deer], noise_img[0]]
    compare_simplex(ensm_data_uncertain, ensm_know_uncertain, ensm_certain, ensm_noise,
                    endd_data_uncertain, endd_know_uncertain, endd_certain, endd_noise,
                    imgs_in)
