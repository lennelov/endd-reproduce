import tensorflow as tf
import category_encoders as ce
import pandas as pd
from tensorflow.keras import datasets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_labels = train_labels.reshape((-1,))
train_labels = tf.one_hot(train_labels, 10)

# TODO Preprocess

model = tf.keras.applications.VGG16(include_top=True, weights=None, input_shape=(32, 32, 3), classes=10)

# TODO Add metrics
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(train_images, train_labels)
