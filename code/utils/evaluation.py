"""This module features basic evaluation functionality.

See scripts/evaluation_example_usage.py for an example of how the module can be used.
"""

import tensorflow as tf
import numpy as np
import settings
from utils import measures
from utils import classifiers

CLASSIFIER_WRAPPERS = {
    'ensemble': classifiers.EnsembleClassifier,
    'ensemble_ood': classifiers.EnsembleClassifierOOD,
    'individual': classifiers.IndividualClassifier,
    'priornet': classifiers.PriorNetClassifier
}


def calc_ood_measures(model, in_images, out_images, tot_wrapper_type, know_wrapper_type):
    # if wrapper_type is not None and wrapper_type not in CLASSIFIER_WRAPPERS:
    #     raise ValueError("""wrapper_type {} not recognized, make sure it has been added to
    #                         CLASSIFIER_WRAPPERS in evaluation.py and that a corresponding
    #                         wrapper exists.""")

    tot_clf = CLASSIFIER_WRAPPERS[tot_wrapper_type](model)
    in_preds = tot_clf.predict(in_images)
    out_preds = tot_clf.predict(out_images)
    tot_unc_roc_auc = measures.calc_tot_unc_auc_roc(in_preds, out_preds)

    if know_wrapper_type:
        know_clf = CLASSIFIER_WRAPPERS[know_wrapper_type](model)
        in_preds = know_clf.predict(in_images)
        out_preds = know_clf.predict(out_images)
        if know_wrapper_type == 'priornet':
            know_unc_roc_auc = measures.calc_pn_know_unc_auc_roc(in_preds, out_preds)
        elif know_wrapper_type == 'ensemble_ood':
            know_unc_roc_auc = measures.calc_ensemble_know_unc_auc_roc(in_preds, out_preds)
    else:
        know_unc_roc_auc = 0

    output = {'tot_unc_roc_auc': tot_unc_roc_auc, 'know_unc_roc_auc': know_unc_roc_auc}
    return output


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


def format_results(model_names, model_measures, dataset_name=None):
    """Format results into readable string.

    Args:
        model_names (List[str]): List of names.
        model_measures List[Dict]: List of measures dicts.
        dataset_name (str): Name of dataset (optional).
    """
    s = "== EVALUATION RESULTS ==\n"
    if dataset_name:
        s += "Dataset: {}\n".format(dataset_name)
    s += "ERR (classification error)\n"
    for name, measures in zip(model_names, model_measures):
        s += "    {}: {:.1f}%\n".format(name, 100 * measures['err'])
    s += "PRR (prediction rejection rate)\n"
    for name, measures in zip(model_names, model_measures):
        s += "    {}: {:.1f}%\n".format(name, 100 * measures['prr'])
    s += "ECE (expected calibration error)\n"
    for name, measures in zip(model_names, model_measures):
        s += "    {}: {:.3f}%\n".format(name, 100 * measures['ece'])
    s += "NLL (negative log-likelihood)\n"
    for name, measures in zip(model_names, model_measures):
        s += "    {}: {:.3f}\n".format(name, measures['nll'])

    return s


def format_ood_results(model_names, model_measures, in_dataset_name=None, out_dataset_name=None):
    s = "== EVALUATION RESULTS ==\n"
    if in_dataset_name:
        s += "In dataset: {}\n".format(in_dataset_name)
    if out_dataset_name:
        s += "Out dataset: {}\n".format(out_dataset_name)
    s += "Total uncertainty AUC-ROC\n"
    for name, measures in zip(model_names, model_measures):
        s += "    {}: {:.1f}%\n".format(name, 100 * measures['tot_unc_roc_auc'])
    s += "Knowledge uncertainty AUC-ROC\n"
    for name, measures in zip(model_names, model_measures):
        s += "    {}: {:.1f}%\n".format(name, 100 * measures['know_unc_roc_auc'])

    return s
