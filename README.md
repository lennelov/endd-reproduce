# endd-reproduce

This is the official respository for our reproduction of Ensemble Distribution Distillation submitted to the ML Reproducibility Challenge 2020. Follow the instructions below to reproduce our findings. Note that reproduction involves the training of a large ensemble of deep neurla networks, and that this will a few days on a consumer GPU.

## Instructions

Note: None of the scripts feature command line arguments. Settings are instead set using global variables (named in ALL_CAPS). 

### Reproducing the main results

1. First, ensure that Tensorflow 2, Numpy, Scikit-learn and Matplotlib are installed and up to date.
2. To train the base VGG ensemble on CIFAR-10, run train_ensemble_vgg.py. 
3. To train an ENDD on this ensemble, run train_endd.py.
4. To train an END on this ensemble, run train_end.py.
5. To To train a PN on CIFAR-10, run train_pn.py.
6. To evaluate on classification, run evaluate_models.py.
7. To evaluate on out-of-distribution-detection, run evaluate_models_ood.py.
8. To perform ensemble size ablation study, run ensemble_size_ablation_study.py.
9. To perform temperature ablation study, run temperature_ablation_study.py.
10. To plot the ensemble studies, use plot_ensemble_size_ablation_study.py and plot_temperature_ablation_study.py. NOTE: In order to plot with error bars, step 8 and step 9 must be repeated with different save names, and these names entered in the MODE_BASE_NAMES variables in the plotting scripts.

### Reproducing the simplex plots
1. To train an ensemble on the 3-class CIFAR dataset, run train_ensemble_vgg_3_class.py.
2. To train an ENDD on this ensemble, set the DATASET_NAME='cifar10_3_class' in train_endd.py and run the script.
3. To visualize the simplex plots, run plot_simplex.py.
