import sys
import pathlib
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import tensorflow as tf
from settings_prior import *
def preprocess(spiral,OOD,train_ratio = 0.7):
      '''
      ---This function should be updated using some built in functions---
      divides the spiral dataset into training and testing and adds OOD data to the training data.
      labels are converted to class logits with value 1 for incorrect class and 1000 for correct class
      Inputs:
            spiral, toy_dataset given by SpiralDataset
            OOD, Custom OOD-data
      Outputs:
      '''
      classes = tf.math.reduce_max(spiral.y[:])
      classes = tf.cast(classes,dtype = tf.int32)
      logits = tf.one_hot(spiral.y,3)*ID_LOGIT +OOD_LOGIT
      tf.random.set_seed(123) # seed needs to be set for each random in tf.
      X =tf.random.shuffle(spiral.x,seed = 1)
      tf.random.set_seed(123) # seed needs to be set for each random in tf.
      Y =tf.random.shuffle(logits,seed = 1)
      train_ratio = tf.constant(train_ratio,dtype=tf.double)
      size = tf.cast(tf.size(X[:,0]),dtype=tf.double)
      train_size = tf.cast(size*train_ratio, dtype = tf.int32, name=None)



      x_train = X[0:train_size,:]
      y_train = Y[0:train_size,:]
      x_test = X[train_size:,:]
      y_test = Y[train_size:,:]
      #Add OOD data
      x_train =tf.concat([x_train,OOD.x],axis = 0)
      #normalize_data this needs to be updated

      avg_norm = (tf.math.reduce_sum(tf.linalg.norm(
            x_train, ord='euclidean', axis=1, name=None)/len(x_train[:,0])))
      x_train =  x_train/avg_norm
      x_test = x_test/avg_norm # to do: fix the standardization

      #add OOD logits (ones)
      y_train = tf.concat([y_train,OOD_LOGIT*tf.ones([OOD.x.shape[0],y_train.shape[1]])],axis = 0)

      #shuffle the OOD data into the training set
      tf.random.set_seed(1234)
      x_train =tf.random.shuffle(x_train,seed = 2) 
      tf.random.set_seed(1234)
      y_train =tf.random.shuffle(y_train,seed = 2)
      return x_train,y_train,x_test,y_test
