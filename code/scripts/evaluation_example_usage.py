"""Example evaluation on individual and ensemble models."""

# Add parent dir to path to allow for parallel imports
import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

from utils import evaluation
from utils import datasets
from utils import saveload
from models import ensemble

ENSEMBLE_NAME = 'basic_cnn'
DATASET_NAME = 'cifar10'

# Load ensemble model
ensemble_model_names = saveload.get_ensemble_model_names()
model_names = ensemble_model_names[ENSEMBLE_NAME][DATASET_NAME][:3]
models = [ensemble.KerasLoadsWhole(name) for name in model_names]
ensm = ensemble.Ensemble(models)
ensm_wrapper_type = 'ensemble'

# Load individual model
ind = saveload.load_tf_model(model_names[0])
ind_wrapper_type = 'individual'

# Load data
_, (test_images, test_labels) = datasets.get_dataset(DATASET_NAME)

# Preprocess data
test_labels = test_labels.reshape(-1)

# Calculate measures
ensm_measures = evaluation.calc_classification_measures(ensm, test_images, test_labels,
                                                        wrapper_type=ensm_wrapper_type)

ind_measures = evaluation.calc_classification_measures(ind, test_images, test_labels,
                                                       wrapper_type=ind_wrapper_type)

# Format and print results
summary = evaluation.format_results(['ENSM', 'IND'], [ensm_measures, ind_measures])
print(summary)
