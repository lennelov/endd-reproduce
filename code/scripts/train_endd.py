'''
Creates and trains a priornet on the EnDD data.
'''
import sys
import os
# parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(parent_dir_path)
sys.path.append('/home/lennelov/Repositories/endd-reproduce/code')
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Imports
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import settings

from models import cnn, endd, ensemble
from utils import evaluation, preprocessing, saveload, simplex, datasets

# Set names for loading and saving
ENSEMBLE_LOAD_NAME = 'basic_cnn'
DATASET_NAME = 'cifar10'
MODEL_SAVE_NAME = 'endd'

# Set training parameters
N_MODELS = 20
N_EPOCHS = 40
BATCH_SIZE = 500
NORMALIZATION = "-1to1"
RUN_EAGERLY = False

# Load ensemble models
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSEMBLE_LOAD_NAME][DATASET_NAME]
model_names = model_names[:N_MODELS]
wrapped_models = [ensemble.KerasLoadsWhole(name) for name in model_names]

# Build ensemble
cnn_ensemble = ensemble.Ensemble(wrapped_models)

# Load ensemble dataset
train_set, test_set = datasets.get_ensemble_dataset(cnn_ensemble, DATASET_NAME)
train_images, train_labels, train_ensemble_preds = train_set
test_images, test_labels, test_ensemble_preds = test_set

# Normalize data
if NORMALIZATION == "-1to1":
    train_images, min, max = preprocessing.normalize_minus_one_to_one(train_images)
    test_images = preprocessing.normalize_minus_one_to_one(test_images, min, max)
elif NORMALIZATION == 'gaussian':
    train_images, mean, std = preprocessing.normalize_gaussian(train_images)
    test_images = preprocessing.normalize_gaussian(test_images, mean, std)

# Build ENDD model
base_model = cnn.get_model(DATASET_NAME, compile=False)
endd_model = endd.get_model(base_model)

# Train ENDD model
endd_model.fit(train_images, train_ensemble_preds, batch_size=BATCH_SIZE, epochs=N_EPOCHS)
if MODEL_SAVE_NAME:
    saveload.save_tf_model(endd_model, MODEL_SAVE_NAME)

# Predict and show outputs
logits = endd_model.predict(test_images)
alphas = tf.math.exp(logits)
predictions = tf.cast(tf.math.argmax(tf.squeeze(logits), axis=1), dtype=tf.float32)
test_labels_tf = tf.cast(tf.squeeze(test_labels.copy()), dtype=tf.float32)

score = tf.math.reduce_sum(tf.cast(tf.math.equal(predictions, test_labels_tf),
                                   tf.float32)) / len(test_labels_tf)
print('alphas for picture 1: ' + str(alphas[0, :]))
print('alphas for picture 1: ' + str(alphas[1, :]))
print('alphas for picture 1: ' + str(alphas[2, :]))
print('mean of 5 ensembles for picture 1: ' +
      str(tf.math.reduce_mean(tf.nn.softmax(train_ensemble_preds[0, :, :], axis=1), axis=0)))
print('score: ' + str(score))

# Evaluate
measures = evaluation.calc_classification_measures(endd_model,
                                                   test_images,
                                                   test_labels,
                                                   wrapper_type='individual')

results = evaluation.format_results(['endd'], [measures])
print(results)
