from utils import losses, saveload
from models import cnn, vgg


def get_model(base_model,
              dataset_name=False,
              compile=True,
              weights=None,
              epsilon=1e-8,
              teacher_epsilon=1e-3,
              init_temp=2.5):
    """Take an uncompiled model and return model compiled for ENDD.

    Warning: This function works in place. Model is returned only for
    conveniance.
    """
    if isinstance(base_model, str):
        if not dataset_name:
            raise ValueError('dataset_name must be provided if base_model is given by name.')
        if base_model == 'cnn':
            base_model = cnn.get_model(dataset_name, compile=False, softmax=False)
        elif base_model == 'vgg':
            base_model = vgg.get_model(dataset_name, compile=False, softmax=False)
        else:
            raise ValueError("""Base model {} not recognized, make sure it has been added
                              to endd.py, or pass a Keras model object as base model instead.""")

    if weights:
        saveload.load_weights(base_model, weights)

    if compile:
        base_model.compile(optimizer='adam',
                           loss=losses.DirichletEnDDLoss(init_temp=init_temp,
                                                         epsilon=epsilon,
                                                         ensemble_epsilon=teacher_epsilon))
    return base_model
