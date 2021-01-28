import numpy as np
import tensorflow as tf
import settings
import random


def make_batch_generator(images, targets, batch_size):
    """Simple generator with no augmentation or batching."""
    n_samples = len(images)
    i = 0
    while i < n_samples - batch_size:
        image_batch = images[i:i + batch_size]
        target_batch = targets[i:i + batch_size]
        i += batch_size
        yield (image_batch, target_batch)
    yield (images[i:], targets[i:])


def make_plain_generator(images, targets):
    """Simple generator with no augmentation or batching."""
    return zip(images, targets)


def make_augmented_endd_generator(images, ensemble_preds, batch_size):
    """Wraps the ImageDataGenerator to allow for using preds as targets."""
    n_samples = len(images)
    ids = list(range(n_samples))
    augmentator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                                  horizontal_flip=True,
                                                                  width_shift_range=4,
                                                                  height_shift_range=4,
                                                                  fill_mode='nearest')
    id_generator = augmentator.flow(images, ids, batch_size=batch_size)
    id_to_preds = {id: preds for id, preds in enumerate(ensemble_preds)}

    for batch in id_generator:
        batch_imgs = batch[0]
        ids = batch[1]
        batch_preds = np.stack([id_to_preds[id] for id in ids], axis=0)
        yield batch_imgs, batch_preds


def make_augmented_generator(images, targets, batch_size):
    """Wraps the ImageDataGenerator to allow for using preds as targets."""
    augmentator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                                  horizontal_flip=True,
                                                                  width_shift_range=4,
                                                                  height_shift_range=4,
                                                                  fill_mode='nearest')
    generator = augmentator.flow(images, targets, batch_size=batch_size)
    return generator


def normalize_gaussian(images, mean=None, std=None):
    """Take images and normalize as gaussian.

    Optionally take precomputed mean and standard deviation. This allows
    the same normalization to be applied to both train set and test set.

    Args:
        images (np.array): Images to normalize.
        mean (float): Optional, precomputed mean.
        std (flat): Optional, precomuted standard deviation.
    Returns:
        If mean and std are provided, only the new images are returned.
        If mean and std are not provided, a tuple of (new_images, mean, std)
        are returned.
    """
    if mean is not None or std is not None:
        return (images - mean) / std

    mean = np.mean(images)
    std = np.std(images)
    new_images = (images - mean) / std
    return new_images, mean, std


def normalize_minus_one_to_one(images, min=None, max=None):
    """Take images and normalize images between -1 and 1.

    Optionally take precomputed min and max values. This allows
    the same normalization to be applied to both train set and test set.

    Normalization is performed set-wise.

    Args:
        images (np.array): Images to normalize.
        min (float): Optional, precomputed min.
        max (flat): Optional, precomuted max.
    Returns:
        If mean and std are provided, only the new images are returned.
        If mean and std are not provided, a tuple of (new_images, min, max)
        are returned.
    """
    if min is not None or max is not None:
        if min < 0:
            images = images + (-min)
        return images / (max / 2) - 1

    min = images.min()
    if min < 0:
        images = images + (-min)
    max = images.max()
    images = images / (max / 2) - 1
    return images, min, max


def normalize_minus_one_to_one_individually(images):
    """Take images and normalize images between -1 and 1.

    Each image is normalized individually.

    Args:
        images (np.array): Images to normalize.
    Returns:
        (np.array) containing normalized images.
    """
    new_images = []
    for img in images:
        min = img.min()
        if min < 0:
            img = img + (-min)
        max = img.max()
        img = img / (max / 2) - 1
        new_images.append(img)
    return np.stack(new_images, axis=0)


def preprocess_toy_dataset(X, Y, training_ratio=0.8, norm=None):
    '''
	normalizes data and splits it into training/testing
        '''
    #shuffle
    XY = np.concatenate((X, np.expand_dims(Y, axis=1)), axis=1)
    np.random.shuffle(XY)
    X = XY[:, :-1]
    Y = XY[:, -1]
    #normalize
    X = np.squeeze(X)
    row_norm = np.linalg.norm(X, axis=1)
    min_norm = np.amin(row_norm)
    max_norm = np.amax(row_norm)
    X = (X - min_norm) / (max_norm - min_norm)
    #split data
    Y = np.squeeze(Y)
    X_train, X_test = X[:int(training_ratio * len(X[:, 0])), :], X[int(training_ratio *
                                                                       len(X[:, 0])):, :]
    Y_train, Y_test = Y[:int(training_ratio * len(X[:, 0]))], Y[int(training_ratio * len(X[:, 0])):]

    #remove OOD from testing data
    index = Y_test >= 0
    X_test = X_test[index]
    Y_test = Y_test[index]

    #Create logits from labels
    index = Y_train < 0
    Y_train[index] = Y_train.max(
    ) + 1  #let the ood data contain class n_classes+1 which we later remove
    logits_train = np.zeros((Y_train.size, int(Y_train.max() + 1)))
    logits_train[np.arange(Y_train.size).astype(int), Y_train.astype(int)] = 100
    logits_train = logits_train + 1
    logits_train = np.delete(logits_train, -1, axis=1)  #delete last column
    logits_test = np.zeros((Y_test.size, int(Y_test.max() + 1)))
    logits_test[np.arange(Y_test.size).astype(int), Y_test.astype(int)] = 100
    logits_test = logits_test + 1
    return X_train, logits_train, X_test, logits_test


def preprocess_cifar_for_priornet(train_images,
                                  train_labels,
                                  test_images,
                                  test_labels,
                                  normalization,
                                  OOD_images=None,
                                  ID_classes=10):
    '''
        preprocesses train and test data from cifar10 for a prior net by taking the first ID_classes classes as ID and remaining as OOD.
	Args:
		train_images (ndarray),
		train_labels (ndarray),
		test_images (ndarray),
		test_labels (ndarray),
		ID_classes (int), nr of classes used for the
        Returns:
		train_images (ndarray),
		train_logits (ndarray),
		test_images (ndarray),
		test_logits (ndarray),
        '''
    test_labels = np.squeeze(test_labels)
    ID_test_index = np.squeeze(np.where(test_labels <= ID_classes - 1))
    test_images = test_images[ID_test_index, :, :, :]

    test_labels = test_labels[ID_test_index]
    train_logits = tf.one_hot(train_labels, ID_classes) * 100 + 1
    test_logits = tf.one_hot(test_labels, ID_classes) * 100 + 1
    if OOD_images is not None:
        train_images = np.concatenate((train_images, OOD_images), axis=0)
        train_logits = tf.concat(
            [tf.squeeze(train_logits),
             tf.ones([OOD_images.shape[0], train_logits.shape[2]])],
            axis=0)

    train_images = np.array(train_images)
    train_logits = np.array(train_logits)

    if normalization == "-1to1":
        train_images, min, max = normalize_minus_one_to_one(train_images)
        test_images = normalize_minus_one_to_one(test_images, min, max)
    elif normalization == 'gaussian':
        train_images, mean, std = normalize_gaussian(train_images)
        test_images = normalize_gaussian(test_images, mean, std)
    if OOD_images is not None:  #shuffle the images
        indices = list(range(len(train_images)))
        random.shuffle(indices)
        train_images = train_images[indices, :, :, :]
        train_logits = train_logits[indices, :]
    train_logits = np.squeeze(train_logits)
    test_images = np.squeeze(test_images)
    return train_images, train_logits, test_images, test_logits
