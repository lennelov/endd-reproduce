import tensorflow as tf
import category_encoders as ce
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras.applications.vgg16 import preprocess_input

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Preprocess
train_images = preprocess_input(train_images)
train_labels = train_labels.reshape((-1, ))
train_labels = tf.one_hot(train_labels, 10)

# Build model
model = tf.keras.applications.VGG16(include_top=True,
                                    weights=None,
                                    input_shape=(32, 32, 3),
                                    classes=10)


preds = model.predict(train_images)

# preds.shape
# train_labels.shape
# preds[:5, :]
# train_labels[:5, :]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[
              tf.keras.metrics.Accuracy(name='accuracy')
              ])

# Train
model.fit(train_images, train_labels, epochs=1)

model.summary()


# print(preds[:5, :])
# print(train_labels[:5, :])
