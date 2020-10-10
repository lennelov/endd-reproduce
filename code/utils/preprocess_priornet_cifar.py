import tensorflow as tf

import settings

def preprocess(train_images, train_labels, test_images, test_labels, ID_classes=3):
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

    ID_test_index = tf.squeeze(tf.where(test_labels <= ID_classes - 1))
    ID_test_index = ID_test_index[:, 0]

    test_images = test_images[ID_test_index, :, :, :]

    test_labels = test_labels[ID_test_index]
    train_logits = tf.one_hot(train_labels, ID_classes) * 100 + 1
    test_logits = tf.one_hot(test_labels, ID_classes) * 100 + 1

    train_images = tf.image.per_image_standardization(tf.cast(train_images, dtype=tf.float32))
    test_images = tf.image.per_image_standardization(tf.cast(test_images, dtype=tf.float32))
    train_logits = tf.squeeze(train_logits)
    test_images = tf.squeeze(test_images)
    return train_images, train_logits, test_images, test_logits
