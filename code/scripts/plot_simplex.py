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

# Load test images
(train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(DATASET_NAME)
raw_test_images = test_images
test_images = preprocessing.normalize_minus_one_to_one(test_images, min=0, max=255)
#
# Load ensemble
ENSM_MODEL_NAME, ENSM_N_MODELS = 'vgg_3_class', 100
DATASET_NAME = 'cifar10_3_class'
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSM_MODEL_NAME][DATASET_NAME][:ENSM_N_MODELS]
models = [ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_names]
ensm_model = ensemble.Ensemble(models)

# Predict ensemble
ensm_out = ensm_model.predict(test_images)

# Save ensemble
with open("ensemble_out.pkl", 'wb') as file:
    pickle.dump((ensm_out), file)

# Load endd
endd_model = saveload.load_tf_model("endd_vgg_cifar10_3class_aux", compile=False)
endd_model = endd.get_model(endd_model, init_temp=1, teacher_epsilon=1e-4)

# Predict endd
endd_out = endd_model.predict(test_images)

# Save endd
with open("endd_out.pkl", 'wb') as file:
    pickle.dump((endd_out), file)

noise_img = np.random.randn(1, 32, 32, 3)
noise_ens = ensm_model.predict(noise_img)
noise_endd = endd_model.predict(noise_img)

# ======== MAIN SCRIPT =========

with open("ensemble_out.pkl", 'rb') as file:
    ens_out = pickle.load(file)

with open("endd_out.pkl", 'rb') as file:
    endd_out = pickle.load(file)


def prepare_prediction(x):
    x = np.exp(np.float64(x))
    x_2 = np.expand_dims(np.sum(x, axis=1), axis=1)
    return x / x_2


def compare_simplex(data_logits, know_logits, certain_logits, d, e, f, ens_noise, endd_noise, imgs_in, filename = None):
    alphas = [prepare_prediction(data_logits),
              prepare_prediction(know_logits),
              prepare_prediction(certain_logits),
              prepare_prediction(ens_noise)]
    exped = [np.float64(d),
            np.float64(e),
            np.float64(f),
            np.float64(endd_noise)]

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
        im = plt.imshow(imgs_in[i])
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)

        plt.subplot(3, 4, i + 5)
        plot_alphas = alphas[i]
        plot_logits = np.exp(np.array(exped[i]))
        plot_logits[plot_logits > 100] = 100
        # print(plot_logits)

        simplex.plot_points(plot_alphas, alpha=0.3)

        plt.subplot(3, 4, i + 9)

        # print(plot_logits.shape)
        # print(np.sum(plot_logits))
        # print(np.multiply.reduce([gamma(a) for a in plot_logits]))
        try:
            simplex.draw_pdf_contours(simplex.Dirichlet(np.float64(plot_logits)), nlevels=200, subdiv=3)
        except:
            pass

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.savefig('compare_plot.png')
    plt.show()





# img_1 = 112
# img_2 = 113  # 2
# img_3 = 114  # 3
#
# # Pick out selected images
# a = ens_out[:, img_1, :]
# b = ens_out[:, img_2, :]
# c = ens_out[:, img_3, :]
# d = endd_out[img_1, :]
# e = endd_out[img_2, :]
# f = endd_out[img_3, :]

# Pick out specific class
indx = test_labels == 0
indx = indx.flatten()
imgs_plane = raw_test_images[indx]
ens_plane = ens_out[:, indx, :]
endd_plane = endd_out[indx, :]

indx = test_labels == 4
indx = indx.flatten()
imgs_deer = raw_test_images[indx]
ens_deer = ens_out[:, indx, :]
endd_deer = endd_out[indx, :]

deer_unct_tot = measures.entropy_of_expected(ens_deer, True)
deer_unct_data = measures.expected_entropy(ens_deer, True)
deer_unct_know = deer_unct_tot - deer_unct_data
print("Five most certain deer: {}".format(np.argsort(deer_unct_tot)[:5]))  # Five most certain
print("Five most data uncertain deer: {}".format(np.argsort(deer_unct_data)[-5:]))  # Five most data uncertain
print("Five most knowledge uncertain deer: {}".format(np.argsort(deer_unct_know)[-5:]))  # Five most knowledge uncertain

plane_unct_tot = measures.entropy_of_expected(ens_plane, True)
plane_unct_data = measures.expected_entropy(ens_plane, True)
plane_unct_know = plane_unct_tot - plane_unct_data
print("Five most certain plain: {}".format(np.argsort(plane_unct_tot)[:5]))  # Five most certain
print("Five most data uncertain plane: {}".format(np.argsort(plane_unct_data)[-5:]))  # Five most data uncertain
print("Five most knowledge uncertain plane: {}".format(np.argsort(plane_unct_know)[-5:]))  # Five most knowledge uncertain

data_uncertain_deer = 799
knowledge_uncertain_plane = 653
certain_deer = 682  # 3

ens_class_1 = ens_deer[:, data_uncertain_deer, :]
ens_class_2 = ens_plane[:, knowledge_uncertain_plane, :]
ens_class_3 = ens_deer[:, certain_deer, :]
ens_noise = noise_ens[:, 0, :]
endd_class_1 = endd_deer[data_uncertain_deer, :]
endd_class_2 = endd_plane[knowledge_uncertain_plane, :]
endd_class_3 = endd_deer[certain_deer, :]
endd_noise = noise_endd[0, :]

imgs_in = [imgs_deer[data_uncertain_deer], imgs_plane[knowledge_uncertain_plane], imgs_deer[certain_deer], noise_img[0]]
# ens_outs
# endd_outs =
compare_simplex(ens_class_1, ens_class_2, ens_class_3, endd_class_1, endd_class_2, endd_class_3, ens_noise, endd_noise, imgs_in)

#simplex.plot_points(a, barycentric = True, border = True)
