import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from collections import defaultdict
from utils import evaluation, datasets, saveload, preprocessing
from models import ensemble, endd
import pickle

# Model loading parameters
N_MODELS_BASE_NAMES = ['new_cifar10_vgg_endd_aux_0']
N_MODELS_BASE_NAMES_SAMPLED = ['cifar10_vgg_endd_aux_sampled_0', 'cifar10_vgg_endd_aux_sampled_1', 'cifar10_vgg_endd_aux_sampled_2']
# Should be set to the same configuration as when running ensemble_size_ablation_study.py
ENDD_AUX_BASE_MODEL = 'vgg'
ENSEMBLE_LOAD_NAME = 'vgg_a'
N_MODELS_LIST = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 45, 60, 75, 100]
N_MODELS_LIST_ORIG = [5, 20, 50, 100]

# Dataset parameters
DATASET_NAME = 'cifar10'
NORMALIZATION = '-1to1'


def get_dataset(dataset_name, normalization):
    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(dataset_name)

    # Normalize data
    if normalization == "-1to1":
        train_images, min, max = preprocessing.normalize_minus_one_to_one(train_images)
        test_images = preprocessing.normalize_minus_one_to_one(test_images, min, max)
    elif normalization == 'gaussian':
        train_images, mean, std = preprocessing.normalize_gaussian(train_images)
        test_images = preprocessing.normalize_gaussian(test_images, mean, std)

    return (train_images, train_labels), (test_images, test_labels)


# Get ensemble measures
def get_ensm_measures(model_names, n_models_list, test_images, test_labels):
    ensm_measures = defaultdict(list)
    for n_models in n_models_list:
        print("############ ensm {}".format(n_models))
        model_name_subset = model_names[:n_models]
        print(model_name_subset)
        wrapped_models = [
            ensemble.KerasLoadsWhole(name, pop_last=True) for name in model_name_subset
        ]
        ensm_model = ensemble.Ensemble(wrapped_models)
        evaluation_result = evaluation.calc_classification_measures(ensm_model,
                                                                    test_images,
                                                                    test_labels,
                                                                    wrapper_type='ensemble')
        for measure, value in evaluation_result.items():
            ensm_measures[measure].append(value)
    return ensm_measures


# Get ENDD measures
def get_endd_measures(n_models_base_names, n_models_list, endd_base_model, dataset_name,
                      test_images, test_labels):
    endd_measures_list = []
    for base_name in n_models_base_names:
        endd_measures = defaultdict(list)
        for n_models in n_models_list:
            #print("{}/{}".format(base_name, n_models))
            #print(n_models)
            endd_model_name = base_name + '_N_MODELS={}'.format(n_models)
            print(endd_model_name)
            uncompiled_model = saveload.load_tf_model(endd_model_name, compile=False)
            endd_model = endd.get_model(uncompiled_model, dataset_name=dataset_name, compile=True)

            evaluation_result = evaluation.calc_classification_measures(endd_model,
                                                                        test_images,
                                                                        test_labels,
                                                                        wrapper_type='individual')
            #print("############# Measures")
            for measure, value in evaluation_result.items():
                endd_measures[measure].append(value)
                #print("{}={}".format(measure, value))
            #print()
        endd_measures_list.append(endd_measures)
    return endd_measures_list


def plot_with_error_fields(n_models_list, ensm_measures, endd_measures_list, measure, ylabel):
    stack = np.stack([endd_measures[measure] for endd_measures in endd_measures_list])
    means = stack.mean(axis=0)
    stds = stack.std(axis=0)
    plt.plot(n_models_list,
             ensm_measures[measure],
             label='ENSM',
             color='xkcd:dull blue',
             linestyle='solid',
             marker='.')
    plt.plot(n_models_list, means, '.-', label=r'END$^2_{+AUX}$', color='xkcd:dusty orange')
    plt.fill_between(n_models_list,
                     means - 2 * stds,
                     means + 2 * stds,
                     color='xkcd:dusty orange',
                     alpha=0.4)
    plt.xlabel("Number of models")
    plt.ylabel(ylabel)
    plt.legend()

def plot_with_error_fields_sampling(n_models_list, ensm_measures_list, endd_measures_list, measure, ylabel, endd_measures_list_repeat = None):

    stack = np.stack([ensm_measures[measure] for ensm_measures in ensm_measures_list])
    means = stack.mean(axis=0)
    stds = stack.std(axis=0)
    plt.plot(n_models_list, means, '.-', label=r'ENSM', color='xkcd:dull blue')
    plt.fill_between(n_models_list,
                     means - 2 * stds,
                     means + 2 * stds,
                     color='xkcd:dull blue',
                     alpha=0.4)


    stack = np.stack([endd_measures[measure] for endd_measures in endd_measures_list])
    means = stack.mean(axis=0)
    stds = stack.std(axis=0)
    plt.plot(n_models_list, means, '.-', label=r'END$^2_{+AUX}$', color='xkcd:dusty orange')
    plt.fill_between(n_models_list,
                     means - 2 * stds,
                     means + 2 * stds,
                     color='xkcd:dusty orange',
                     alpha=0.4)

    if endd_measures_list_repeat is not None:
        stack = np.stack([endd_measures[measure] for endd_measures in endd_measures_list_repeat])
        stds = stack.std(axis=0)
        plt.fill_between(n_models_list,
                         means - 2 * stds,
                         means + 2 * stds,
                         label = 'no ensm. var.',
                         color='xkcd:maroon',
                         alpha=0.4)


    plt.xlabel("Number of models")
    plt.ylabel(ylabel)
    plt.legend()


def plot_with_error_fields_paper(n_models_list, ensm_measures, endd_measures_list, measure, ylabel):
    stack = np.stack([endd_measures[measure] for endd_measures in endd_measures_list])
    means = stack.mean(axis=0)
    minimum = stack.min(axis=0)
    maximum = stack.max(axis=0)
    plt.plot(n_models_list,
             ensm_measures[measure],
             label='ENSM Paper',
             color='xkcd:dull blue',
             linestyle='dashed',
             marker = 'x')
    plt.plot(n_models_list, means,
      label=r'END$^2_{+AUX}$ Paper',
      color='xkcd:dusty orange',
      linestyle = 'dashed',
      marker = 'x')
    plt.fill_between(n_models_list,
                     minimum,
                     maximum,
                     color='xkcd:dusty orange',
                     linestyle = 'dashed',
                     alpha=0.4)
    plt.xlabel("Number of models")
    plt.ylabel(ylabel)
    plt.legend()

def get_paper_measures():

    paper_ensm_measures = defaultdict(list)

    paper_ensm_measures["err"].append(6.703/100)
    paper_ensm_measures["err"].append(6.400/100)
    paper_ensm_measures["err"].append(6.204/100)
    paper_ensm_measures["err"].append(6.203/100)

    paper_ensm_measures["nll"].append(0.2)
    paper_ensm_measures["nll"].append(0.19)
    paper_ensm_measures["nll"].append(0.19)
    paper_ensm_measures["nll"].append(0.19)

    paper_ensm_measures["ece"].append(0.85/100)
    paper_ensm_measures["ece"].append(1.086/100)
    paper_ensm_measures["ece"].append(1.337/100)
    paper_ensm_measures["ece"].append(1.31/100)

    paper_ensm_measures["prr"].append(0.865)
    paper_ensm_measures["prr"].append(0.873)
    paper_ensm_measures["prr"].append(0.8670)
    paper_ensm_measures["prr"].append(0.8680)



    paper_endd_measures_list = []



    # Mid
    paper_endd_measures = defaultdict(list)

    paper_endd_measures['err'].append(7.1/100)
    paper_endd_measures['err'].append(6.96/100)
    paper_endd_measures['err'].append(7.07/100)
    paper_endd_measures['err'].append(6.93/100)

    paper_endd_measures['nll'].append(0.243)
    paper_endd_measures['nll'].append(0.24)
    paper_endd_measures['nll'].append(0.24)
    paper_endd_measures['nll'].append(0.24)

    paper_endd_measures['ece'].append(2.086/100)
    paper_endd_measures['ece'].append(2.127/100)
    paper_endd_measures['ece'].append(1.9862/100)
    paper_endd_measures['ece'].append(2.2/100)

    paper_endd_measures['prr'].append(0.8583)
    paper_endd_measures['prr'].append(0.8583)
    paper_endd_measures['prr'].append(0.8627)
    paper_endd_measures['prr'].append(0.8570)

    paper_endd_measures_list.append(paper_endd_measures)

    # High
    paper_endd_measures = defaultdict(list)

    paper_endd_measures['err'].append(7.23/100)
    paper_endd_measures['err'].append(7.01/100)
    paper_endd_measures['err'].append(7.19/100)
    paper_endd_measures['err'].append(7.02/100)

    paper_endd_measures['nll'].append(0.2474)
    paper_endd_measures['nll'].append(0.24)
    paper_endd_measures['nll'].append(0.24)
    paper_endd_measures['nll'].append(0.24)

    paper_endd_measures['ece'].append(2.25/100)
    paper_endd_measures['ece'].append(2.19/100)
    paper_endd_measures['ece'].append(2.16/100)
    paper_endd_measures['ece'].append(2.37/100)

    paper_endd_measures['prr'].append(0.86094)
    paper_endd_measures['prr'].append(0.861991)
    paper_endd_measures['prr'].append(0.8672)
    paper_endd_measures['prr'].append(0.8586)

    paper_endd_measures_list.append(paper_endd_measures)

    # Low
    paper_endd_measures = defaultdict(list)

    paper_endd_measures['err'].append(6.96/100)
    paper_endd_measures['err'].append(6.92/100)
    paper_endd_measures['err'].append(6.94/100)
    paper_endd_measures['err'].append(6.84/100)

    paper_endd_measures['nll'].append(0.238)
    paper_endd_measures['nll'].append(0.24)
    paper_endd_measures['nll'].append(0.24)
    paper_endd_measures['nll'].append(0.24)

    paper_endd_measures['ece'].append(1.91/100)
    paper_endd_measures['ece'].append(2.05/100)
    paper_endd_measures['ece'].append(1.806/100)
    paper_endd_measures['ece'].append(2.013/100)

    paper_endd_measures['prr'].append(0.85582)
    paper_endd_measures['prr'].append(0.854609)
    paper_endd_measures['prr'].append(0.8580)
    paper_endd_measures['prr'].append(0.8555)

    paper_endd_measures_list.append(paper_endd_measures)

    return paper_ensm_measures, paper_endd_measures_list

def get_ens_sampled_measures():
    sampled_ens_measures_list = []

    ## Rep 1
    sampled_ens_measures = defaultdict(list)

    sampled_ens_measures['err'].append(0.09770000000000001)
    sampled_ens_measures['err'].append(0.0897)
    sampled_ens_measures['err'].append(0.09040000000000004)
    sampled_ens_measures['err'].append(0.09230000000000005)
    sampled_ens_measures['err'].append(0.09040000000000004)
    sampled_ens_measures['err'].append(0.08799999999999997)
    sampled_ens_measures['err'].append(0.08679999999999999)
    sampled_ens_measures['err'].append(0.08989999999999998)
    sampled_ens_measures['err'].append(0.0887)
    sampled_ens_measures['err'].append(0.08630000000000004)
    sampled_ens_measures['err'].append(0.08799999999999997)
    sampled_ens_measures['err'].append(0.08730000000000004)
    sampled_ens_measures['err'].append(0.08730000000000004)
    sampled_ens_measures['err'].append(0.08850000000000002)
    sampled_ens_measures['err'].append(0.08750000000000002)
    sampled_ens_measures['err'].append(0.08760000000000001)

    sampled_ens_measures['prr'].append(0.7628538235096157)
    sampled_ens_measures['prr'].append(0.7728590069270793)
    sampled_ens_measures['prr'].append(0.7737677415383977)
    sampled_ens_measures['prr'].append(0.7952224006940192)
    sampled_ens_measures['prr'].append(0.7935340034465063)
    sampled_ens_measures['prr'].append(0.7933693291250025)
    sampled_ens_measures['prr'].append(0.8006413715436153)
    sampled_ens_measures['prr'].append(0.803175274912816)
    sampled_ens_measures['prr'].append(0.8013109280240819)
    sampled_ens_measures['prr'].append(0.8010402508294245)
    sampled_ens_measures['prr'].append(0.7993227670544046)
    sampled_ens_measures['prr'].append(0.7992825217018691)
    sampled_ens_measures['prr'].append(0.8038484924241177)
    sampled_ens_measures['prr'].append(0.8057395979129228)
    sampled_ens_measures['prr'].append(0.800504952986585)
    sampled_ens_measures['prr'].append(0.8022540645813261)

    sampled_ens_measures['ece'].append(0.03139433637410407)
    sampled_ens_measures['ece'].append(0.01785618839263919)
    sampled_ens_measures['ece'].append(0.01998816794455055)
    sampled_ens_measures['ece'].append(0.018293927524983912)
    sampled_ens_measures['ece'].append(0.017090723612904514)
    sampled_ens_measures['ece'].append(0.018696480201184733)
    sampled_ens_measures['ece'].append(0.01856758460253477)
    sampled_ens_measures['ece'].append(0.018282189263403463)
    sampled_ens_measures['ece'].append(0.016700169710814908)
    sampled_ens_measures['ece'].append(0.017365400521457203)
    sampled_ens_measures['ece'].append(0.018816743324697016)
    sampled_ens_measures['ece'].append(0.01791400345414878)
    sampled_ens_measures['ece'].append(0.015452068389952209)
    sampled_ens_measures['ece'].append(0.016553349840641034)
    sampled_ens_measures['ece'].append(0.017656126230955138)
    sampled_ens_measures['ece'].append(0.015145644748210908)

    sampled_ens_measures['nll'].append(0.2944625885303713)
    sampled_ens_measures['nll'].append(0.2665589616811817)
    sampled_ens_measures['nll'].append(0.26560673127263934)
    sampled_ens_measures['nll'].append(0.260220236286204)
    sampled_ens_measures['nll'].append(0.2587900142429449)
    sampled_ens_measures['nll'].append(0.2530800458324652)
    sampled_ens_measures['nll'].append(0.24967668012247754)
    sampled_ens_measures['nll'].append(0.25096511095779417)
    sampled_ens_measures['nll'].append(0.24830588520774863)
    sampled_ens_measures['nll'].append(0.24665661704275677)
    sampled_ens_measures['nll'].append(0.24941750200252671)
    sampled_ens_measures['nll'].append(0.24782753053747525)
    sampled_ens_measures['nll'].append(0.2480706390366362)
    sampled_ens_measures['nll'].append(0.24902406007829705)
    sampled_ens_measures['nll'].append(0.24736762882971394)
    sampled_ens_measures['nll'].append(0.24771360189519878)


    sampled_ens_measures_list.append(sampled_ens_measures)

    ## Rep 2
    sampled_ens_measures = defaultdict(list)

    sampled_ens_measures['err'].append(0.09219999999999995)
    sampled_ens_measures['err'].append(0.09250000000000003)
    sampled_ens_measures['err'].append(0.08960000000000001)
    sampled_ens_measures['err'].append(0.08819999999999995)
    sampled_ens_measures['err'].append(0.08809999999999996)
    sampled_ens_measures['err'].append(0.08989999999999998)
    sampled_ens_measures['err'].append(0.08940000000000003)
    sampled_ens_measures['err'].append(0.08740000000000003)
    sampled_ens_measures['err'].append(0.08650000000000002)
    sampled_ens_measures['err'].append(0.08809999999999996)
    sampled_ens_measures['err'].append(0.08709999999999996)
    sampled_ens_measures['err'].append(0.08779999999999999)
    sampled_ens_measures['err'].append(0.0887)
    sampled_ens_measures['err'].append(0.08650000000000002)
    sampled_ens_measures['err'].append(0.08720000000000006)
    sampled_ens_measures['err'].append(0.08809999999999996)

    sampled_ens_measures['prr'].append(0.7623497549050586)
    sampled_ens_measures['prr'].append(0.7794355509280461)
    sampled_ens_measures['prr'].append(0.7806232476401972)
    sampled_ens_measures['prr'].append(0.796597458074804)
    sampled_ens_measures['prr'].append(0.7894564685160987)
    sampled_ens_measures['prr'].append(0.8019017767564024)
    sampled_ens_measures['prr'].append(0.7959726893258201)
    sampled_ens_measures['prr'].append(0.8011093246873126)
    sampled_ens_measures['prr'].append(0.7884904226235272)
    sampled_ens_measures['prr'].append(0.8042727383968391)
    sampled_ens_measures['prr'].append(0.7925923421386476)
    sampled_ens_measures['prr'].append(0.8004331760002295)
    sampled_ens_measures['prr'].append(0.8042847522536748)
    sampled_ens_measures['prr'].append(0.7956577544194414)
    sampled_ens_measures['prr'].append(0.8016926761549391)
    sampled_ens_measures['prr'].append(0.8040273175233515)

    sampled_ens_measures['ece'].append(0.03337129490673543)
    sampled_ens_measures['ece'].append(0.021568498441576944)
    sampled_ens_measures['ece'].append(0.01895388883054251)
    sampled_ens_measures['ece'].append(0.018469107010960547)
    sampled_ens_measures['ece'].append(0.019209435287117937)
    sampled_ens_measures['ece'].append(0.018132549259066634)
    sampled_ens_measures['ece'].append(0.01688504027575254)
    sampled_ens_measures['ece'].append(0.01713586966544392)
    sampled_ens_measures['ece'].append(0.016617135727405516)
    sampled_ens_measures['ece'].append(0.016202568663656686)
    sampled_ens_measures['ece'].append(0.015065063504874694)
    sampled_ens_measures['ece'].append(0.017798748302459724)
    sampled_ens_measures['ece'].append(0.016499805355071992)
    sampled_ens_measures['ece'].append(0.017372954124212244)
    sampled_ens_measures['ece'].append(0.01619368659853933)
    sampled_ens_measures['ece'].append(0.016961750987172158)

    sampled_ens_measures['nll'].append(0.2829228044085516)
    sampled_ens_measures['nll'].append(0.27068637753049696)
    sampled_ens_measures['nll'].append(0.26056061743796427)
    sampled_ens_measures['nll'].append(0.25468340642005755)
    sampled_ens_measures['nll'].append(0.25417455853313875)
    sampled_ens_measures['nll'].append(0.25311497341016226)
    sampled_ens_measures['nll'].append(0.25421198103712994)
    sampled_ens_measures['nll'].append(0.2501553440356367)
    sampled_ens_measures['nll'].append(0.24996938669564855)
    sampled_ens_measures['nll'].append(0.24923458174685967)
    sampled_ens_measures['nll'].append(0.2504395957548247)
    sampled_ens_measures['nll'].append(0.24905140825184813)
    sampled_ens_measures['nll'].append(0.2481466161912708)
    sampled_ens_measures['nll'].append(0.24828424311197286)
    sampled_ens_measures['nll'].append(0.2472108247490523)
    sampled_ens_measures['nll'].append(0.24909313550068082)

    sampled_ens_measures_list.append(sampled_ens_measures)

    ## Rep 3
    sampled_ens_measures = defaultdict(list)

    sampled_ens_measures['err'].append(0.10209999999999997)
    sampled_ens_measures['err'].append(0.09340000000000004)
    sampled_ens_measures['err'].append(0.0907)
    sampled_ens_measures['err'].append(0.09060000000000001)
    sampled_ens_measures['err'].append(0.08740000000000003)
    sampled_ens_measures['err'].append(0.09219999999999995)
    sampled_ens_measures['err'].append(0.08840000000000003)
    sampled_ens_measures['err'].append(0.08899999999999997)
    sampled_ens_measures['err'].append(0.08720000000000006)
    sampled_ens_measures['err'].append(0.08860000000000001)
    sampled_ens_measures['err'].append(0.08819999999999995)
    sampled_ens_measures['err'].append(0.08760000000000001)
    sampled_ens_measures['err'].append(0.0887)
    sampled_ens_measures['err'].append(0.08840000000000003)
    sampled_ens_measures['err'].append(0.08720000000000006)
    sampled_ens_measures['err'].append(0.08819999999999995)

    sampled_ens_measures['prr'].append(0.7644175284417839)
    sampled_ens_measures['prr'].append(0.7707975581954372)
    sampled_ens_measures['prr'].append(0.7838550144324605)
    sampled_ens_measures['prr'].append(0.792239497814315)
    sampled_ens_measures['prr'].append(0.7889089805808471)
    sampled_ens_measures['prr'].append(0.8069382875131941)
    sampled_ens_measures['prr'].append(0.7927329096511742)
    sampled_ens_measures['prr'].append(0.7989902406187526)
    sampled_ens_measures['prr'].append(0.7959288473570778)
    sampled_ens_measures['prr'].append(0.8010292435124424)
    sampled_ens_measures['prr'].append(0.801744331319597)
    sampled_ens_measures['prr'].append(0.7968256311678582)
    sampled_ens_measures['prr'].append(0.8074855119478475)
    sampled_ens_measures['prr'].append(0.8068114501214504)
    sampled_ens_measures['prr'].append(0.7957265936282049)
    sampled_ens_measures['prr'].append(0.8047697593802181)

    sampled_ens_measures['ece'].append(0.034884572990238664)
    sampled_ens_measures['ece'].append(0.020746426121890562)
    sampled_ens_measures['ece'].append(0.01766726886928081)
    sampled_ens_measures['ece'].append(0.01890465791672466)
    sampled_ens_measures['ece'].append(0.017573820623755464)
    sampled_ens_measures['ece'].append(0.015821297705173466)
    sampled_ens_measures['ece'].append(0.01683249293416739)
    sampled_ens_measures['ece'].append(0.017685826887190347)
    sampled_ens_measures['ece'].append(0.01657173907905816)
    sampled_ens_measures['ece'].append(0.015440757890045651)
    sampled_ens_measures['ece'].append(0.017390120497345937)
    sampled_ens_measures['ece'].append(0.016136639565229392)
    sampled_ens_measures['ece'].append(0.017087308491766436)
    sampled_ens_measures['ece'].append(0.017336303213238747)
    sampled_ens_measures['ece'].append(0.01690224304348232)
    sampled_ens_measures['ece'].append(0.01584664890021082)

    sampled_ens_measures['nll'].append(0.3052048919410399)
    sampled_ens_measures['nll'].append(0.2748114675652928)
    sampled_ens_measures['nll'].append(0.26339604131405125)
    sampled_ens_measures['nll'].append(0.2576284617955046)
    sampled_ens_measures['nll'].append(0.25545555798044034)
    sampled_ens_measures['nll'].append(0.2571129421456795)
    sampled_ens_measures['nll'].append(0.25317755813968806)
    sampled_ens_measures['nll'].append(0.2527470552617273)
    sampled_ens_measures['nll'].append(0.25103149417423026)
    sampled_ens_measures['nll'].append(0.25000116296753755)
    sampled_ens_measures['nll'].append(0.24919727719578913)
    sampled_ens_measures['nll'].append(0.25010861404158957)
    sampled_ens_measures['nll'].append(0.24840294385608958)
    sampled_ens_measures['nll'].append(0.24757865391553446)
    sampled_ens_measures['nll'].append(0.2490726380473154)
    sampled_ens_measures['nll'].append(0.24679787793283367)

    sampled_ens_measures_list.append(sampled_ens_measures)

    return sampled_ens_measures_list



# Get dataset
_, (test_images, test_labels) = get_dataset(DATASET_NAME, NORMALIZATION)


# # Load ensemble model names
# ensemble_model_names = saveload.get_ensemble_model_names()
# model_names = ensemble_model_names[ENSEMBLE_LOAD_NAME][DATASET_NAME]
# model_names = model_names[-100:] # 100 last, the original ones

# # Get ENSM measures
# ensm_measures = get_ensm_measures(model_names, N_MODELS_LIST, test_images, test_labels)
# print(ensm_measures)
#
# with open("ensemble_measure.pkl", "wb") as file:
#    pickle.dump((ensm_measures), file)
# print("saved")

# Get ENDD measures
endd_measures_list = get_endd_measures(N_MODELS_BASE_NAMES, N_MODELS_LIST, ENDD_AUX_BASE_MODEL,
                                      DATASET_NAME, test_images, test_labels)

with open("endd_measure.pkl", "wb") as file:
   pickle.dump((endd_measures_list), file)
print("saved")

# Get sampling ENDD measures
#endd_sampled_measures_list = get_endd_measures(N_MODELS_BASE_NAMES_SAMPLED, N_MODELS_LIST, ENDD_AUX_BASE_MODEL,
#                                       DATASET_NAME, test_images, test_labels)

#with open("endd_measure_sampled.pkl", "wb") as file:
#    pickle.dump((endd_sampled_measures_list), file)
#print("saved")

## Load data
#
# with open("ensemble_measure.pkl", "rb") as file:
#     ensm_measures, _ = pickle.load(file)
#
# with open("endd_measure.pkl", "rb") as file:
#     endd_measures_list = pickle.load(file)
#
# with open("endd_measure_sampled.pkl", "rb") as file:
#     endd_sampled_measures_list = pickle.load(file)
# endd_sampled_measures_list.append(endd_measures_list[2])
# #import pdb; pdb.set_trace()
#
#
# ensm_sampled_measures_list = get_ens_sampled_measures()
# ensm_sampled_measures_list.append(ensm_measures)
#
# paper_ensm_measures, paper_endd_measures_list = get_paper_measures()
#
#
# print(ensm_measures)
# print(endd_measures_list)
# print()
# print(ensm_sampled_measures_list)
# print(endd_sampled_measures_list)
# print()
# print(paper_ensm_measures)
# print(paper_endd_measures_list)
#
#
#
# ## Plot results
# plt.style.use('ggplot')
#
# plt.subplot(2, 2, 1)
# #plot_with_error_fields(N_MODELS_LIST, ensm_measures, endd_measures_list, 'err', 'Prediction Error')
# plot_with_error_fields_sampling(N_MODELS_LIST, ensm_sampled_measures_list, endd_sampled_measures_list, 'err', 'Prediction Error', endd_measures_list)
# plot_with_error_fields_paper(N_MODELS_LIST_ORIG, paper_ensm_measures, paper_endd_measures_list, 'err', 'Prediction Error')
#
#
#
# plt.subplot(2, 2, 2)
# #plot_with_error_fields(N_MODELS_LIST, ensm_measures, endd_measures_list, 'nll', 'Negative Log-Likelihood')
# plot_with_error_fields_sampling(N_MODELS_LIST, ensm_sampled_measures_list, endd_sampled_measures_list, 'nll', 'Negative Log-Likelihood', endd_measures_list)
# plot_with_error_fields_paper(N_MODELS_LIST_ORIG, paper_ensm_measures, paper_endd_measures_list, 'nll', 'Negative Log-Likelihood')
#
#
#
# plt.subplot(2, 2, 3)
# #plot_with_error_fields(N_MODELS_LIST, ensm_measures, endd_measures_list, 'ece', 'Expected Calibration Error')
# plot_with_error_fields_sampling(N_MODELS_LIST, ensm_sampled_measures_list, endd_sampled_measures_list, 'ece', 'Expected Calibration Error', endd_measures_list)
# plot_with_error_fields_paper(N_MODELS_LIST_ORIG, paper_ensm_measures, paper_endd_measures_list, 'ece', 'Expected Calibration Error')
#
#
#
# plt.subplot(2, 2, 4)
# #plot_with_error_fields(N_MODELS_LIST, ensm_measures, endd_measures_list, 'prr', 'Prediction Rejection Rate')
# plot_with_error_fields_sampling(N_MODELS_LIST, ensm_sampled_measures_list, endd_sampled_measures_list, 'prr', 'Prediction Rejection Rate', endd_measures_list)
# plot_with_error_fields_paper(N_MODELS_LIST_ORIG, paper_ensm_measures, paper_endd_measures_list, 'prr', 'Prediction Rejection Rate')
#
#
#
# plt.show()
