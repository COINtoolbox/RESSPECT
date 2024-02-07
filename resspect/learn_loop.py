# Copyright 2020 resspect software
# Author: The RESSPECT team
#
# created on 14 August 2020
#
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ['learn_loop', 'load_features', 'run_classification', 
          'run_evaluation', 'save_photo_ids', 'run_make_query',
          'update_alternative_label']

import copy
import logging
from typing import Union

import progressbar
from resspect import DataBase


def load_features(database_class: DataBase,
                  path_to_features: Union[str, dict],
                  survey: str, features_method: str, number_of_classes: int,
                  training_method: str, is_queryable: bool,
                  separate_files: bool = False, 
                  initial_training_samples_file: str = None) -> DataBase:
    """
    Load features according to feature extraction method

    Parameters
    ----------
    database_class
        An instance of DataBase class
    path_to_features
        Complete path to input features file.
        if dict, keywords should be 'train' and 'test',
        and values must contain the path for separate train
        and test sample files.
    survey
       'DES' or 'LSST'. Default is 'DES'.
        Name of the survey which characterizes filter set.
    features_method
        Feature extraction method. Currently only 'bazin' and "bump" are implemented.
    number_of_classes
        Number of classes to consider in the classification
        Currently only nclass == 2 is implemented.
    training_method
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia
        Default is 'original'.
    is_queryable
       If True, check if randomly chosen object is queryable.
       Default is False.
    separate_files: bool (optional)
        If True, consider train and test samples separately read
        from independent files. Default is False.
    initial_training_samples_file
        File name to save initial training samples.
        File will be saved if "training"!="original"
    """
    if isinstance(path_to_features, str):
        database_class.load_features(
            path_to_file=path_to_features, feature_extractor=features_method,
            survey=survey)
    else:
        features_set_names = ['train', 'test', 'validation', 'pool']
        for sample_name in features_set_names:
            if sample_name in path_to_features.keys():
                database_class.load_features(
                    path_to_features[sample_name], feature_extractor=features_method,
                    survey=survey, sample=sample_name)
            else:
                logging.warning(f'Path to {sample_name} not given.'
                                f' Proceeding without this sample')
                
    database_class.build_samples(
        initial_training=training_method, nclass=number_of_classes,
        queryable=is_queryable, sep_files=separate_files, 
        output_fname=initial_training_samples_file)
    
    return database_class


def run_classification(database_class: DataBase, classifier: str,
                       is_classifier_bootstrap: bool, prediction_dir: str,
                       is_save_prediction: bool, iteration_step: int,
                       **kwargs: dict) -> DataBase:
    """
    Run active learning classification model

    Parameters
    ----------
    database_class
        An instance of DataBase class
    classifier
        Machine Learning algorithm.
        Currently implemented options are 'RandomForest', 'GradientBoostedTrees',
        'KNNclassifier','MLPclassifier','SVMclassifier' and 'NBclassifier'.
        Default is 'RandomForest'.
    is_classifier_bootstrap
        if tp apply a machine learning classifier by bootstrapping
    prediction_dir
       Output directory to store prediction file for each loop.
       Only used if `save_predictions==True
    is_save_prediction
        if predictions should be saved
    iteration_step
        active learning iteration number
    kwargs
       All keywords required by the classifier function.
    -------

    """
    if is_classifier_bootstrap:
        database_class.classify_bootstrap(
            method=classifier, loop=iteration_step, pred_dir=prediction_dir,
            save_predictions=is_save_prediction, **kwargs)
    else:
        database_class.classify(
            method=classifier, pred_dir=prediction_dir, loop=iteration_step,
            save_predictions=is_save_prediction, **kwargs)
    return database_class


def run_evaluation(database_class: DataBase, metric_label: str):
    """
    Evaluates the active learning model

    Parameters
    ----------
    database_class
        An instance of DataBase class
    metric_label
        Choice of metric.
        Currently only "snpcc", "cosmo" or "snpcc_cosmo" are accepted.
        Default is "snpcc".

    """
    database_class.evaluate_classification(metric_label=metric_label)


def save_photo_ids(database_class: DataBase, is_save_photoids_to_file: bool,
                   is_save_snana_types: bool, metadata_fname: str,
                   photo_class_threshold: float, iteration_step: int,
                   file_name_prefix: str = None, file_name_suffix: str = None):
    """
    Function to save photo IDs to a file

    Parameters
    ----------
    database_class
        An instance of DataBase class
    is_save_photoids_to_file
        If true, populate the photo_Ia_list attribute. Otherwise
        write to file. Default is False.
    is_save_snana_types
        if True, translate type to SNANA codes and
        add column with original values. Default is False.
    metadata_fname
        Full path to PLAsTiCC zenodo test metadata file.
    photo_class_threshold
         Probability threshold above which an object is considered Ia.
    iteration_step
        active learning iteration number
    file_name_suffix
        suffix string for save file name with file extension
    file_name_prefix
        prefix string for save file name
    """
    if is_save_photoids_to_file or is_save_snana_types:
        file_name = file_name_prefix + '_' + str(iteration_step) + file_name_suffix
        database_class.output_photo_Ia(
            photo_class_threshold, to_file=is_save_photoids_to_file,
            filename=file_name, SNANA_types=is_save_snana_types,
            metadata_fname=metadata_fname)


def run_make_query(database_class: DataBase, strategy: str, batch_size: int,
                   is_queryable: bool):
    """
    Run active learning query process

    Parameters
    ----------
    database_class
        An instance of DataBase class
    strategy
        Strategy used to choose the most informative object.
        Current implementation accepts 'UncSampling' and
        'RandomSampling', 'UncSamplingEntropy',
        'UncSamplingLeastConfident', 'UncSamplingMargin',
        'QBDMI', 'QBDEntropy', . Default is `UncSampling`.
    batch_size
        Number of objects to be chosen in each batch query.
        Default is 1
    is_queryable
        If True, consider only queryable objects.
        Default is False.
    """
    return database_class.make_query(strategy=strategy, batch=batch_size,
                                     queryable=is_queryable)


def _save_metrics_and_queried_samples(
        database_class: DataBase, metrics_file_name: str,
        queried_file_name: str, iteration_step: int, batch: int,
        full_sample: bool, file_name_suffix: str = None):
    """
    Save metrics and queried samples details

    Parameters
    ----------
    database_class
        An instance of DataBase class
    metrics_file_name
        Full path to file to store metrics results.
    queried_file_name
        Complete path to output file.
    iteration_step
        active learning iteration number
    batch
        Number of queries in each loop.
    full_sample
        If true, write down a complete queried sample stored in
        property 'queried_sample'. Otherwise append 1 line per loop to
        'queried_sample_file'. Default is False.
    file_name_suffix
        suffix string for save file name with file extension
    """
    if file_name_suffix is not None:
        metrics_file_name = metrics_file_name.replace('.dat', file_name_suffix)
        queried_file_name = queried_file_name.replace(
            '.dat', file_name_suffix)
    database_class.save_metrics(
        loop=iteration_step, output_metrics_file=metrics_file_name,
        batch=batch, epoch=iteration_step)
    database_class.save_queried_sample(
        queried_file_name, loop=iteration_step, full_sample=full_sample,
        epoch=iteration_step, batch=batch)


def update_alternative_label(database_class_alternative: DataBase,
                             indices_to_query: list, iteration_step: int,
                             classifier: str, pred_dir: str,
                             is_save_prediction: bool, metric_label: str,
                             is_save_snana_types: bool,
                             is_save_photoids_to_file: bool,
                             meta_data_fname: str, photo_class_threshold: float,
                             photo_ids_froot: str,
                             **kwargs: dict):
    """
    Function to update active learning training with alternative label

    Parameters
    ----------
    database_class_alternative
        An instance of DataBase class for alternative label
    indices_to_query
        List of indexes identifying objects to be moved.
    iteration_step
        active learning iteration number
    classifier
        Machine Learning algorithm.
        Currently implemented options are 'RandomForest', 'GradientBoostedTrees',
        'K-NNclassifier','MLPclassifier','SVMclassifier' and 'NBclassifier'.
        Default is 'RandomForest'.
    pred_dir
        Output diretory to store prediction file for each loop.
        Only used if `save_predictions==True`.
    is_save_prediction
        if predictions should be saved
    metric_label
        Choice of metric.
        Currently only "snpcc", "cosmo" or "snpcc_cosmo" are accepted.
        Default is "snpcc".
    is_save_snana_types
        if True, translate type to SNANA codes and
        add column with original values. Default is False.
    is_save_photoids_to_file
        If true, populate the photo_Ia_list attribute. Otherwise
        write to file. Default is False.
    meta_data_fname
        Full path to PLAsTiCC zenodo test metadata file.
    photo_class_threshold
         Probability threshold above which an object is considered Ia.
    photo_ids_froot
        Output root of file name to store photo ids.
        Only used if photo_ids is True.
    kwargs
        additional arguments
    """
    database_class_alternative.update_samples(
        indices_to_query, epoch=iteration_step,
        alternative_label=True)
    
    database_class_alternative = run_classification(
        database_class_alternative, classifier, False, pred_dir,
        is_save_prediction, iteration_step, **kwargs)
    
    run_evaluation(database_class_alternative, metric_label)
    
    save_photo_ids(database_class_alternative, is_save_photoids_to_file,
                   is_save_snana_types, meta_data_fname, photo_class_threshold,
                   iteration_step, photo_ids_froot,'_alt_label.dat')
    
    return database_class_alternative


# TODO: too many arguments! refactor further and update docs
def learn_loop(nloops: int, strategy: str, path_to_features: str,
               output_metrics_file: str, output_queried_file: str,
               features_method: str = 'bazin', classifier: str = 'RandomForest',
               training: str = 'original', batch: int =1, survey: str = 'DES',
               nclass: int = 2, photo_class_thr: float = 0.5,
               photo_ids_to_file: bool = False, photo_ids_froot: str =' ',
               classifier_bootstrap: bool = False, save_predictions:
               bool = False, sep_files=False, pred_dir: str = None,
               queryable: bool = False, metric_label: str = 'snpcc',
               save_alt_class: bool = False, SNANA_types: bool = False,
               metadata_fname: str = None, bar: bool = True,
               initial_training_samples_file: str = None, **kwargs):
    """
    Perform the active learning loop. All results are saved to file.

    Parameters
    ----------
    nloops: int
        Number of active learning loops to run.
    strategy: str
        Query strategy. Options are 'UncSampling', 'RandomSampling',
        'UncSamplingEntropy', 'UncSamplingLeastConfident', 'UncSamplingMargin',
        'QBDMI' and 'QBDEntropy'.
    path_to_features: str or dict
        Complete path to input features file.
        if dict, keywords should be 'train' and 'test',
        and values must contain the path for separate train
        and test sample files.
    output_metrics_file: str
        Full path to output file to store metric values of each loop.
    output_queried_file: str
        Full path to output file to store the queried sample.
    features_method: str (optional)
        Feature extraction method. Currently only 'bazin' and 'Bump' are implemented.
    classifier: str (optional)
        Machine Learning algorithm.
        Currently implemented options are 'RandomForest', 'GradientBoostedTrees',
        'K-NNclassifier','MLPclassifier','SVMclassifier' and 'NBclassifier'.
        Default is 'RandomForest'.
    sep_files: bool (optional)
        If True, consider train and test samples separately read
        from independent files. Default is False.
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    classifier_bootstrap: bool (optional)
        Flag for bootstrapping on the classifier
        Must be true if using disagreement based strategy.
    metadata_fname: str (optional)
        Complete path to PLAsTiCC zenodo test metadata. Only used it
        SNANA_types == True. Default is None.
    metric_label: str (optional)
        Choice of metric.
        Currently only "snpcc", "cosmo" or "snpcc_cosmo" are accepted.
        Default is "snpcc".
    nclass: int (optional)
        Number of classes to consider in the classification
        Currently only nclass == 2 is implemented.
    photo_class_thr: float (optional)
        Threshold for photometric classification. Default is 0.5.
        Only used if photo_ids is True.
    photo_ids_to_file: bool (optional)
        If True, save photometric ids to file. Default is False.
    photo_ids_froot: str (optional)
        Output root of file name to store photo ids.
        Only used if photo_ids is True.
    pred_dir: str (optional)
        Output diretory to store prediction file for each loop.
        Only used if `save_predictions==True`.
    queryable: bool (optional)
        If True, check if randomly chosen object is queryable.
        Default is False.
    save_alt_class: bool (optional)
        If True, train the model and save classifications for alternative
        query label (this is necessary to calculate impact on cosmology).
        Default is False.
    save_predictions: bool (optional)
        If True, save classification predictions to file in each loop.
        Default is False.
    SNANA_types: bool (optional)
        If True, translate zenodo types to SNANA codes.
        Default is False.
    survey: str (optional)
        'DES' or 'LSST'. Default is 'DES'.
        Name of the survey which characterizes filter set.
    training: str or int (optional)
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia
        Default is 'original'.
    bar: bool (optional)
        If True, display progress bar.
    initial_training_samples_file
        File name to save initial training samples.
        File will be saved if "training"!="original"  
    kwargs: extra parameters
        All keywords required by the classifier function.
    """
    if 'QBD' in strategy and not classifier_bootstrap:
        raise ValueError(
            'Bootstrap must be true when using disagreement strategy')
        
    # initiate object
    database_class = DataBase()
    logging.info('Loading features')
    database_class = load_features(database_class, path_to_features, survey,
                                   features_method, nclass, training, queryable,
                                   sep_files, initial_training_samples_file)
    
    logging.info('Running active learning loop')
    
    if bar:
        ensemble = progressbar.progressbar(range(nloops))
    else:
        ensemble = range(nloops)
    
    for iteration_step in ensemble:
        if not bar:
            print(iteration_step)
            
        database_class = run_classification(
            database_class, classifier, classifier_bootstrap, pred_dir,
            save_predictions, iteration_step, **kwargs)
        run_evaluation(database_class, metric_label)
        save_photo_ids(database_class, photo_ids_to_file, SNANA_types,
                       metadata_fname, photo_class_thr, iteration_step,
                       photo_ids_froot, '.dat')
        indices_to_query = run_make_query(database_class, strategy, batch,
                                          queryable)
        if save_alt_class and batch == 1:
            database_class_alternative = copy.deepcopy(database_class)
            database_class_alternative = update_alternative_label(
                database_class_alternative, indices_to_query, iteration_step,
                classifier, pred_dir, save_predictions, metric_label,
                SNANA_types, photo_ids_to_file, metadata_fname, photo_class_thr,
                photo_ids_froot, **kwargs)
            _save_metrics_and_queried_samples(
                database_class_alternative, output_metrics_file,
                output_queried_file, iteration_step, batch, False,
                '_alt_label.dat')
            
        elif save_alt_class and batch > 1:
            raise ValueError('Alternative label only works with batch=1!')

        database_class.update_samples(
            indices_to_query, epoch=iteration_step,
            queryable=queryable, alternative_label=False)
        _save_metrics_and_queried_samples(database_class, output_metrics_file,
                                          output_queried_file, iteration_step, batch,
                                          False)
    return database_class


def main():
    return None


if __name__ == '__main__':
    main()

