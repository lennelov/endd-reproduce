"""This module features basic evaluation functionality.

See scripts/evaluation_example_usage.py for an example of how the module can be used.
"""

import tensorflow as tf
import numpy as np
import settings
from utils import measures
from utils import classifiers


CLASSIFIER_WRAPPERS = {'ensemble': classifiers.EnsembleClassifier,
                       'individual': classifiers.IndividualClassifier}


def calc_classification_measures(model, images, labels, wrapper_type=None):
    """Return dict containing classification measures.

    If model.predict() returns a normalized probability distribution, wrapper_type can
    be omitted. If it returns a single array of logits, 'individual' should be used.
    If the model is an ensemble and returns a list of logits, 'ensemble' should be used.

    Args:
        model: Trained classification model to be evaluated.
        images (np.Array): Test images.
        labels (np.Array): Test labels with onehot encoding.
        wrapper_type (str): Type of wrapper needed by the classifier, must be listed in
                            CLASSIFIER_WRAPPERS in evaluation.py if provided (default None).

    Returns:
        (dict) with format
            results = {
                'err': float,
                'prr': float,
                'ece': float,
                'nll': float
            }
    """

    if wrapper_type is not None and wrapper_type not in CLASSIFIER_WRAPPERS:
        raise ValueError("""wrapper_type {} not recognized, make sure it has been added to
                            CLASSIFIER_WRAPPERS in evaluation.py and that a corresponding
                            wrapper exists.""")

    if wrapper_type:
        clf = CLASSIFIER_WRAPPERS[wrapper_type](model)

    labels = labels.reshape((-1,))
    probs = clf.predict(images)
    err = measures.calc_err(probs, labels)
    prr = measures.calc_prr(probs, labels)
    ece = measures.calc_ece(probs, labels)
    nll = measures.calc_nll(probs, labels)

    output = {'err': err, 'prr': prr, 'ece': ece, 'nll': nll}

    return output


def format_results(model_names, model_measures):
    """Format results into readable string."""
    s = "== EVALUATION RESULTS ==\n"
    s += "ERR (classification error)\n"
    for name, measures in zip(model_names, model_measures):
        s += "    {}: {}\n".format(name, measures['err'])
    s += "PRR (prediction rejection rate)\n"
    for name, measures in zip(model_names, model_measures):
        s += "    {}: {}\n".format(name, measures['prr'])
    s += "ECE (expected calibration error)\n"
    for name, measures in zip(model_names, model_measures):
        s += "    {}: {}\n".format(name, measures['ece'])
    s += "NLL (negative log-likelihood)\n"
    for name, measures in zip(model_names, model_measures):
        s += "    {}: {}\n".format(name, measures['nll'])

    return s
