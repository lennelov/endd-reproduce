import tensorflow as tf
from tensorflow.keras import  datasets
from settings_prior import *
def data_creation_cifar10():
        '''
        Creates and preprocesses train and test data from cifar10 for a prior net with 3 classes as ID and remaining 7 as OOD.

        Returns:
        	standardized train and test images with corresponding logits
        '''
        ID_classes = 3
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        ID_test_index = tf.squeeze(tf.where(test_labels <=ID_classes-1))
        ID_test_index = ID_test_index[:,0]

        test_images = test_images[ID_test_index,:,:,:]

        test_labels = test_labels[ID_test_index]
        train_logits = tf.one_hot(train_labels,3)*100+1
        test_logits = tf.one_hot(test_labels,3)*100+1

        train_images = tf.image.per_image_standardization(tf.cast(train_images,dtype=tf.float32))
        test_images = tf.image.per_image_standardization(tf.cast(test_images,dtype=tf.float32))
        train_logits = tf.squeeze(train_logits)
        test_images = tf.squeeze(test_images)
        return train_images, train_logits, test_images, test_logits 