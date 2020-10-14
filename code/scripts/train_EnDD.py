'''
Creates and trains a priornet on the EnDD data.
'''
import sys
import pathlib
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import tensorflow as tf

from models import ensemble
from tensorflow.keras import datasets
from utils.NLLensemble import DirichletEnDDLoss


from tensorflow.keras import datasets

import settings
from utils.simplex_plot_function import plot_simplex

#from models.cnn_priorNet import get_model
from models.cnn import get_model

from utils.NLLensemble import DirichletEnDDLoss
from utils import saveload

DATASET = 'cifar10' 
PLOT_SIMPLEX = False
SAVE_WEIGHTS = False
BATCH_SIZE = 500
EPOCHS = 40
RUN_EAGERLY = False #debugging
#load ensemble
if __name__ == "__main__":
    # Note: in below example, it is assumed that there is a trained Keras model
    # saved with saveload.save_model, saved using the name 'cnn'

    # Wrap models
    model1 = ensemble.KerasLoadsWhole(model_load_name="vgg_cifar10_cifar10_71", name="cnn_1")
    model2 = ensemble.KerasLoadsWhole(model_load_name="vgg_cifar10_cifar10_87", name="cnn_2")
    model3 = ensemble.KerasLoadsWhole(model_load_name="vgg_cifar10_cifar10_18", name="cnn_3")
    model4 = ensemble.KerasLoadsWhole(model_load_name="vgg_cifar10_cifar10_50", name="cnn_4")
    model5 = ensemble.KerasLoadsWhole(model_load_name="vgg_cifar10_cifar10_86", name="cnn_5")


    # Build ensemble
    cnn_models = [model1, model2,model3,model4,model5]
    cnn_ensemble = ensemble.Ensemble(cnn_models) 

    # Load data
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    #normalize data
    normalization = "-1to1"
    if normalization == "-1to1":
	    train_images = train_images / 127.5
	    train_images = train_images - 1.0
	    test_images = test_images / 127.5
	    test_images = test_images - 1.0

    # Predict with ensemble
    ensemble_preds = cnn_ensemble.predict(train_images)
    print("Ensemble preds shape: {}".format(ensemble_preds.shape))

ensemble_preds = tf.transpose(ensemble_preds,[1,0,2])
if RUN_EAGERLY:
	print(ensemble_preds.shape)
	print("pred 1")
	print(tf.nn.softmax(ensemble_preds[0,0,:],axis = 0))
	print("pred 2")
	print(tf.nn.softmax(ensemble_preds[0,1,:],axis = 0))


model = get_model(DATASET,compile = False) #change this to a vgg model for better accuracy I can not run this on my computer
model.compile(optimizer='adam',
                  loss= DirichletEnDDLoss(),run_eagerly = RUN_EAGERLY)

model.fit(train_images, ensemble_preds, batch_size=BATCH_SIZE, epochs=EPOCHS)
if SAVE_WEIGHTS:
    saveload.save_tf_model(model, "EnDD")

logits = model.predict(test_images)
alphas = tf.math.exp(logits)
predictions = tf.cast(tf.math.argmax(tf.squeeze(logits), axis=1),dtype = tf.float32)
test_labels = tf.cast(tf.squeeze(test_labels),dtype = tf.float32)

score = tf.math.reduce_sum(tf.cast(tf.math.equal(predictions, test_labels), tf.float32)) / len(test_labels)
print('alphas for picture 1: ' + str(alphas[0,:]))
print('alphas for picture 1: ' + str(alphas[1,:]))
print('alphas for picture 1: ' + str(alphas[2,:]))
print('mean of 5 ensembles for picture 1: ' + str(tf.math.reduce_mean(tf.nn.softmax(ensemble_preds[0,:,:],axis = 1),axis = 0)))
print('score: ' + str(score))
if PLOT_SIMPLEX:
    plot_simplex(logits)
