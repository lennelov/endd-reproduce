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

DATASET_NAME = 'cifar10'

# # ======== SETUP =========
#
# # Load test images
# (train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(DATASET_NAME)
# test_images = preprocessing.normalize_minus_one_to_one(test_images, min=0, max=255)
#
# # Load ensemble
# ENSM_MODEL_NAME, ENSM_N_MODELS = 'vgg_3_class', 100
# DATASET_NAME = 'cifar10_3_class'
# ensemble_model_names = saveload.get_ensemble_model_names()
# model_names = ensemble_model_names[ENSM_MODEL_NAME][DATASET_NAME][:ENSM_N_MODELS]
# models = [ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_names]
# ensm_model = ensemble.Ensemble(models)
#
# # Predict with ensemble
# # out = ensm_model.predict(test_images)
#
# # Predict ensemble
# ensm_out = ensm_model.predict(test_images)
#
# # Save ensemble
# with open("ensemble_out.pkl", 'wb') as file:
#     pickle.dump((ensm_out), file)
#
# # Load endd
# endd_model = saveload.load_tf_model("endd_vgg_cifar10_3class_aux", compile = False)
# endd_model = endd.get_model(endd_model, teacher_epsilon = 1e-4)
#
# # Predict endd
# endd_out = endd_model.predict(test_images)
#
# # Save endd
# with open("endd_out.pkl", 'wb') as file:
#     pickle.dump((endd_out), file)


# ======== MAIN SCRIPT =========

with open("ensemble_out.pkl", 'rb') as file:
    ens_out = pickle.load(file)

with open("endd_out.pkl", 'rb') as file:
    endd_out = np.exp(pickle.load(file))


def prepare_prediction(x):
    x = np.exp(np.float64(x))
    x_2 = np.expand_dims(np.sum(x, axis=1), axis=1)
    return x / x_2


def compare_simplex(data_logits, know_logits, certain_logits, d, e, f, filename = None):
    alphas = [prepare_prediction(data_logits),
              prepare_prediction(know_logits),
              prepare_prediction(certain_logits)]
    exped = [np.float64(d),
            np.float64(e),
            np.float64(f)]

    font = {
        'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 16,
    }
    plt.style.use('seaborn-white')
    plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
    models = ["data_uncertainty", "knowledge uncertainty", "certain"]

    for i in range(0, 3):
        plt.subplot(2, 3, i + 1)
        plt.title(models[i], fontsize=18, ha='center')
        plot_alphas = alphas[i]
        plot_logits = np.array(exped[i])
        # print(plot_logits)

        simplex.plot_points(plot_alphas)

        plt.subplot(2, 3, i + 4)

        # print(plot_logits.shape)
        # print(np.sum(plot_logits))
        # print(np.multiply.reduce([gamma(a) for a in plot_logits]))
        print(simplex.Dirichlet(np.float64(plot_logits))._alpha)
        simplex.draw_pdf_contours(simplex.Dirichlet(np.float64(plot_logits)), nlevels=200, subdiv=5)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.savefig('compare_plot.png')
    plt.show()


# unct_tot = measures.entropy_of_expected(ens_out, True)
# unct_data = measures.expected_entropy(ens_out, True)
# unct_know = unct_tot - unct_data
#
# print(np.argsort(unct_tot))
# print(np.flip(np.argsort(unct_data)[-100:]))
# print(np.argsort(unct_know))

img_1 = 112
img_2 = 113  # 2
img_3 = 114  # 3

# Pick out selected images
a = ens_out[:, img_1, :]
b = ens_out[:, img_2, :]
c = ens_out[:, img_3, :]
d = endd_out[img_1, :]
e = endd_out[img_2, :]
f = endd_out[img_3, :]

# Pick out specific class
indx = test_labels == 1
indx = indx.flatten()
ens_class = ens_out[:, indx, :]
endd_class = endd_out[indx, :]

ens_class_1 = ens_class[:, 1, :]
ens_class_2 = ens_class[:, 2, :]
ens_class_3 = ens_class[:, 3, :]
endd_class_1 = endd_class[1, :]
endd_class_2 = endd_class[2, :]
endd_class_3 = endd_class[3, :]

np.min(np.exp(a))
compare_simplex(ens_class_1, ens_class_2, ens_class_3, endd_class_1, endd_class_2, endd_class_3)

#simplex.plot_points(a, barycentric = True, border = True)
