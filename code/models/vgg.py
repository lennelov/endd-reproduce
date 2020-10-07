import tensorflow as tf
import tensorflow.keras as keras
import settings
import utils.saveload as saveload
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.models import Sequential
from keras import regularizers


def get_vgg_model(input_shape, classes, dropout_rate=0.5, alpha=0.1, batch_norm=True):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    model = Sequential()

    # Block 1

    model.add(
        Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 2

    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 3

    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 4

    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 5

    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal'))
    #model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # fc classfier

    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())

    model.add(Dropout(dropout_rate))
    model.add(Dense(classes, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))
    return model


def get_model(dataset_name, compile=True, weights=None):
    """Take dataset name and return corresponding untrained VGG16 model.

    Args:
        dataset_name (str): Name of the dataset that the model will be used on,
                            must be listed in settings.py.
        compile (bool): If False, an uncompiled model is returned. Default is True.
        weights (str): Name of saved weights. If provided, returned model will
                       be loaded with saved weights. Default is None.

    Returns:
        keras Model object

    If compile=True, model will be compiled with adam optimizer, categorical cross
    entropy loss, and accuracy metric.
    """
    if dataset_name not in settings.DATASET_NAMES:
        raise ValueError("""Dataset {} not recognized, please make sure it has been listed in
                            settings.py""".format(dataset_name))

    model = get_vgg_model(input_shape=settings.DATASET_INPUT_SHAPES[dataset_name],
                          classes=settings.DATASET_N_CLASSES[dataset_name],
                          batch_norm=True)

    if weights:
        saveload.load_weights(model, weights)

    if not compile:
        return model

    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model
