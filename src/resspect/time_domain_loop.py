# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
# created on 14 April 2020
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

__all__ = ['time_domain_loop', 'load_dataset', 'submit_queries_to_TOM']

import os
from typing import Union, Tuple

import numpy as np
import pandas as pd
import progressbar

from resspect import DataBase
from resspect.tom_client import TomClient
from resspect.time_domain_configuration import TimeDomainConfiguration

def load_dataset(
    file_names_dict: dict, config: TimeDomainConfiguration, samples_list: list = [None], is_load_build_samples: bool = True
) -> DataBase:
    """
    Reads a data sample from file.

    Parameters
    ----------
    file_names_dict:  dict
        Path to light curve features file.
        #if "sep_files == True", dictionary keywords must contain identify
        #different samples: ['train', 'test','validation', 'pool',  None]
    config: TimeDomainConfiguration
        The configuration blob.
    samples_list: list (optional)
        If None, sample is given by a column within the given file.
        else, read independent files for 'train' and 'test'.
        Default is None.
    is_load_build_samples
        if database.build_samples method should be called
    """

    # initiate object
    database_class = DataBase()
    for sample in samples_list:
        database_class.load_features(
            file_names_dict[sample],
            survey=config.survey,
            sample=sample,
            feature_extractor=config.feature_extraction_method
        )
    if is_load_build_samples:
        database_class.build_samples(
            initial_training=config.initial_training,
            nclass=config.nclass,
            Ia_frac=config.ia_frac,
            queryable=config.queryable,
            save_samples=config.save_samples,
            sep_files=config.sep_files,
            survey=config.survey
        )
    return database_class


def _load_first_loop_and_full_data(
        first_loop_file_name: str, config: TimeDomainConfiguration) -> Tuple[DataBase, DataBase]:
    """
    Loads first loop and initial light curve training data

    Parameters
    ----------
    first_loop_file_name
        Path to light curve features file.
        #if "sep_files == True", dictionary keywords must contain identify
        #different samples: ['train', 'test','validation', 'pool']
    config
        The `TimeDomainConfiguration` dataclass that
        contains all the runtime configuration options.
    """
    if not config.sep_files:
        first_loop_file_name = {None: first_loop_file_name}
        first_loop_data = load_dataset(
            file_names_dict=first_loop_file_name,
            config=config,
        )
        light_curve_file_name = {None: config.path_to_ini_files['train']}
        light_curve_data = load_dataset(
            file_names_dict=light_curve_file_name,
            config=config,
        )
    else:
        first_loop_file_name = {'pool': first_loop_file_name}
        first_loop_data = load_dataset(
            file_names_dict=first_loop_file_name,
            config=config,
            samples_list=['pool'],
        )
        light_curve_data = load_dataset(
            file_names_dict=config.path_to_ini_files,
            config=config,
            samples_list=['train', 'test', 'validation'],
            is_load_build_samples=False,
        )
    return first_loop_data, light_curve_data


def _update_light_curve_data_val_and_test_data(
        light_curve_data: DataBase, first_loop_data: DataBase,
        is_separate_files: bool = False,
        initial_training: Union[str, int] = 'original',
        is_queryable: bool = False, number_of_classes: int = 2) -> DataBase:
    """
    Updates initial light curve validation and test data

    Parameters
    ----------
    light_curve_data
        initial light curve training data
    first_loop_data
        first loop light curve data
    is_queryable
        If True, allow queries only on objects flagged as queryable.
        Default is True.
    is_separate_files
        If True, consider samples separately read
        from independent files. Default is False.
    initial_training
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        eilf 'previous': read training and queried from previous run.
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia
        Default is 'original'.
    number_of_classes
        Number of classes to consider in the classification
        Currently only number_of_classes == 2 is implemented.
    """
    if is_separate_files:
        light_curve_data.build_samples(
            nclass=number_of_classes, queryable=is_queryable,
            sep_files=is_separate_files, initial_training=initial_training)
    else:
        light_curve_data.test_features = first_loop_data.pool_features
        light_curve_data.test_metadata = first_loop_data.pool_metadata
        light_curve_data.test_labels = first_loop_data.pool_labels

        light_curve_data.validation_features = first_loop_data.pool_features
        light_curve_data.validation_metadata = first_loop_data.pool_metadata
        light_curve_data.validation_labels = first_loop_data.pool_labels
    return light_curve_data


def _update_data_by_remove_repeated_ids(first_loop_data: DataBase,
                                        light_curve_data: DataBase,
                                        id_key_name: str,
                                        pool_labels_class: str = 'Ia') -> Tuple[
        DataBase, DataBase]:
    """
    Updates first loop and initial data by removing repetitive id indices

    Parameters
    ----------
    first_loop_data
        first loop light curve data
    light_curve_data
        initial light curve training data
    id_key_name
        object identification key name
    pool_labels_class
        pool labels class name
    """
    repeated_id_flags = np.in1d(
        first_loop_data.pool_metadata[id_key_name].values,
        light_curve_data.train_metadata[id_key_name].values)
    first_loop_data.pool_metadata = first_loop_data.pool_metadata[
        ~repeated_id_flags]
    first_loop_data.pool_features = first_loop_data.pool_features[
        ~repeated_id_flags]
    pool_labels = (
            first_loop_data.pool_metadata['type'].values == pool_labels_class)
    first_loop_data.pool_labels = pool_labels.astype(int)
    light_curve_data.pool_features = first_loop_data.pool_features
    light_curve_data.pool_metadata = first_loop_data.pool_metadata
    light_curve_data.pool_labels = first_loop_data.pool_labels
    return first_loop_data, light_curve_data


def _update_queryable_ids(light_curve_data: DataBase, id_key_name: str,
                          is_queryable: bool) -> DataBase:
    """
    Updates queryable ids

    Parameters
    ----------
    light_curve_data
        initial light curve training data
    id_key_name
        object identification key name
    is_queryable
        If True, allow queries only on objects flagged as queryable.
        Default is True.
    """
    if is_queryable:
        queryable_flags = light_curve_data.pool_metadata['queryable'].values
        light_curve_data.queryable_ids = light_curve_data.pool_metadata[
            id_key_name].values[queryable_flags]
    else:
        light_curve_data.queryable_ids = light_curve_data.pool_metadata[
            id_key_name].values
    return light_curve_data


def _update_canonical_ids(light_curve_data: DataBase,
                          canonical_file_name: str,
                          is_restrict_canonical: bool) -> Tuple[
        DataBase, DataBase]:
    """
    Updates canonical ids

    Parameters
    ----------
    light_curve_data
        initial light curve training data
    canonical_file_name
        Path to canonical sample features files.
        It is only used if "strategy==canonical".
    is_restrict_canonical
        If True, restrict the search to the canonical sample.
    """
    database_class = None
    if is_restrict_canonical:
        database_class = DataBase()
        database_class.load_features(path_to_file=canonical_file_name)
        light_curve_data.queryable_ids = database_class.queryable_ids
    return light_curve_data, database_class


def _update_initial_train_meta_data_header(
        first_loop_data: DataBase, light_curve_data: DataBase) -> DataBase:
    """
    Updates if all headers in test not exist in train

    Parameters
    ----------
    first_loop_data
        first loop light curve data
    light_curve_data
        light curve learning data

    """
    for each_name in first_loop_data.metadata_names:
        if each_name not in light_curve_data.metadata_names:
            light_curve_data.metadata_names.append(each_name)
            light_curve_data.metadata[each_name] = None
            light_curve_data.train_metadata.insert(
                len(light_curve_data.metadata_names) - 1, each_name, None, True)
    return light_curve_data


def _run_classification_and_evaluation(
        database_class: DataBase, classifier: str,
        is_classifier_bootstrap: bool, **kwargs: dict) -> DataBase:
    """
    Runs active learning classification and evaluation methods

    Parameters
    ----------
    database_class
        An instance of DataBase class
    classifier
        Machine Learning algorithm.
        Currently 'RandomForest', 'GradientBoostedTrees',
        'KNN', 'MLP', 'SVM' and 'NB' are implemented.
        Default is 'RandomForest'.
    is_classifier_bootstrap
        if tp apply a machine learning classifier by bootstrapping
    kwargs
       All keywords required by the classifier function.
    """
    if is_classifier_bootstrap:
        database_class.classify_bootstrap(method=classifier, **kwargs)
    else:
        database_class.classify(method=classifier, **kwargs)
    database_class.evaluate_classification()
    
    return database_class


def _get_indices_of_objects_to_be_queried(
        database_class: DataBase,budgets: tuple, config: TimeDomainConfiguration) -> list:
    """
    Finds indices of objects to be queried

    Parameters
    ----------
    database_class
        An instance of DataBase class
    budgets
        Budgets for each of the telescopes
    config
        The `TimeDomainConfiguration` dataclass that
        contains all the runtime configuration options.
    """
    if budgets:
        object_indices = database_class.make_query_budget(
            budgets=budgets, strategy=config.strategy
        )
    else:
        object_indices = database_class.make_query(
            strategy=config.strategy,
            batch=config.batch,
            queryable=config.queryable,
            query_thre=config.query_thre,
        )
    return list(object_indices)


def _update_samples_with_object_indices(
        database_class: DataBase, object_indices: list,
        is_queryable: bool, epoch: int) -> DataBase:
    """
    Runs database class update_samples methods with object indices

    Parameters
    ----------
    database_class
        An instance of DataBase class
    object_indices
        List of indexes identifying objects to be moved.
    is_queryable
        If True, consider queryable flag. Default is False.
    epoch
        Day since beginning of survey. Default is 20.
    """
    database_class.update_samples(
        object_indices, queryable=is_queryable, epoch=epoch)
    return database_class


def _save_metrics_and_queried_sample(
        database_class: DataBase,
        current_loop: int, output_metric_file_name: str,
        output_queried_file_name: str, batch: int, epoch: int,
        is_save_full_query: bool):
    """
    Saves metrics and queried sample data

    Parameters
    ----------
    database_class
        An instance of DataBase class
    current_loop
        Number of learning loops finished at this stage.
    output_metric_file_name
        Full path to file to store metrics results.
    output_queried_file_name
    batch
        Number of queries in each loop.
    epoch
        Days since the beginning of the survey.
    is_save_full_query
        If true, write down a complete queried sample stored in
        property 'queried_sample'. Otherwise append 1 line per loop to
        'queried_sample_file'. Default is False.
    """
    database_class.save_metrics(
        loop=current_loop, output_metrics_file=output_metric_file_name,
        batch=batch, epoch=epoch)
    if is_save_full_query:
        output_queried_file_name = (output_queried_file_name[:-4] +
                                    '_' + str(current_loop) + '.csv')
    database_class.save_queried_sample(
        output_queried_file_name, loop=current_loop,
        full_sample=is_save_full_query, epoch=epoch, batch=batch)


def _load_next_day_data(
        next_day_features_file_name: str, config: TimeDomainConfiguration):
    """
    Loads features of next day

    Parameters
    ----------
    next_day_features_file_name
        next day features file name
    is_separate_files
       If True, consider samples separately read
        from independent files. Default is False.
    is_queryable
        If True, allow queries only on objects flagged as queryable.
        Default is True.
    survey_name
        Name of survey to be analyzed. Accepts 'DES' or 'LSST'.
        Default is LSST.
    ia_training_fraction
        Fraction of Ia required in initial training sample.
        Default is 0.5.
    is_save_samples
        If True, save training and test samples to file.
        Default is False.
    """
    if config.sep_files:
        next_day_features_file_name = {'pool': next_day_features_file_name}
        next_day_data = load_dataset(
            next_day_features_file_name, config, samples_list=['pool'],
        )
    else:
        next_day_features_file_name = {None: next_day_features_file_name}
        next_day_data = load_dataset(next_day_features_file_name, config)
    return next_day_data


def _remove_old_training_features(
        light_curve_data: DataBase, light_curve_metadata: np.ndarray,
        metadata_value: int):
    """
    Removes old training features

    Parameters
    ----------
    light_curve_data
        light curve training data
    light_curve_metadata
        light curve meta data
    metadata_value
        metadata object value
    """
    current_day_object_index = list(light_curve_metadata).index(
        metadata_value)
    light_curve_data.train_metadata = light_curve_data.train_metadata.drop(
        light_curve_data.train_metadata.index[current_day_object_index])
    light_curve_data.train_labels = np.delete(
        light_curve_data.train_labels, current_day_object_index, axis=0)
    light_curve_data.train_features = np.delete(
        light_curve_data.train_features, current_day_object_index, axis=0)
    return light_curve_data


def _update_queried_sample(light_curve_data: DataBase, next_day_data: DataBase,
                           id_key_name: str, metadata_value: int) -> DataBase:
    """
    Updates queried sample in light curve data

    Parameters
    ----------
    light_curve_data
        light curve data
    next_day_data
        next day light curve data
    id_key_name
        object identification key name
    metadata_value
        metadata object value
    """
    # build query data frame
    full_header_name = (['epoch'] + light_curve_data.metadata_names
                        + light_curve_data.features_names)
    queried_sample = pd.DataFrame(light_curve_data.queried_sample,
                                  columns=full_header_name)
    # get object index in the queried sample
    queried_index = list(
        queried_sample[id_key_name].values).index(metadata_value)
    # get flag to isolate object in question
    queried_values_flag = queried_sample[id_key_name].values == metadata_value
    # get object epoch in the queried sample
    metadata_value_epoch = queried_sample['epoch'].values[queried_values_flag]
    # remove old features from queried
    queried_sample = queried_sample.drop(queried_sample.index[queried_index])
    next_day_pool_data_flag = (
            next_day_data.pool_metadata[id_key_name].values == metadata_value)
    new_query_pool_metadata = list(next_day_data.pool_metadata[
                                       next_day_pool_data_flag].values[0])
    new_query_pool_features = list(next_day_data.pool_features[
                                       next_day_pool_data_flag][0])
    new_query = ([metadata_value_epoch[0]] + new_query_pool_metadata +
                 new_query_pool_features)
    new_query = pd.DataFrame([new_query], columns=full_header_name)
    queried_sample = pd.concat([queried_sample, new_query], axis=0,
                               ignore_index=True)
    # update queried sample
    light_curve_data.queried_sample = list(queried_sample.values)
    return light_curve_data


def _update_training_data_with_new_features(
        light_curve_data: DataBase, next_day_data: DataBase, metadata_value: int,
        id_key_name: str) -> DataBase:
    """
    Updates new features of the training with new metadata value

    Parameters
    ----------
    light_curve_data
        light curve data
    next_day_data
        next day light curve data
    id_key_name
        object identification key name
    metadata_value
        metadata object value
    """
    next_day_pool_data_flag = (
            next_day_data.pool_metadata[id_key_name].values == metadata_value)
    light_curve_data.train_metadata = pd.concat(
        [light_curve_data.train_metadata,
         next_day_data.pool_metadata[next_day_pool_data_flag]],
        axis=0, ignore_index=True)
    light_curve_data.train_features = np.append(
        light_curve_data.train_features,
        next_day_data.pool_features[next_day_pool_data_flag], axis=0)
    light_curve_data.train_labels = np.append(
        light_curve_data.train_labels,
        next_day_data.pool_labels[next_day_pool_data_flag], axis=0)
    return light_curve_data


def _update_next_day_pool_data(next_day_data: DataBase,
                               next_day_pool_metadata_indices) -> DataBase:
    """
    Removes metadata value data from next day pool sample

    Parameters
    ----------
    next_day_data
        next day light curve data
    next_day_pool_metadata_indices
        indices of metadata value in next day light curve data
    """
    # remove obj from pool sample
    next_day_data.pool_metadata = next_day_data.pool_metadata.drop(
        next_day_data.pool_metadata.index[next_day_pool_metadata_indices])
    next_day_data.pool_labels = np.delete(
        next_day_data.pool_labels, next_day_pool_metadata_indices, axis=0)
    next_day_data.pool_features = np.delete(
        next_day_data.pool_features, next_day_pool_metadata_indices, axis=0)
    return next_day_data


def _update_next_day_val_and_test_data(
        next_day_data: DataBase, metadata_value: int,
        id_key_name: str) -> DataBase:
    """
    Removes metadata value data from next day validation and test samples

    Parameters
    ----------
    next_day_data
        next day light curve data
    metadata_value
        metadata object value
    id_key_name
        object identification key name
    """
    if (len(next_day_data.validation_metadata) > 0 and metadata_value
            in next_day_data.validation_metadata[id_key_name].values):
        val_indices = list(next_day_data.validation_metadata[
                               id_key_name].values).index(metadata_value)
        next_day_data.validation_metadata = (
            next_day_data.validation_metadata.drop(
                next_day_data.validation_metadata.index[val_indices]))
        next_day_data.validation_labels = np.delete(
            next_day_data.validation_labels, val_indices, axis=0)
        next_day_data.validation_features = np.delete(
            next_day_data.validation_features, val_indices, axis=0)

    if (len(next_day_data.test_metadata) > 0 and metadata_value
            in next_day_data.test_metadata[id_key_name].values):
        test_indices = list(next_day_data.test_metadata[
                                id_key_name].values).index(metadata_value)

        next_day_data.test_metadata = (
            next_day_data.test_metadata.drop(
                next_day_data.test_metadata.index[test_indices]))
        next_day_data.test_labels = np.delete(
            next_day_data.test_labels, test_indices, axis=0)
        next_day_data.test_features = np.delete(
            next_day_data.test_features, test_indices, axis=0)
    return next_day_data


def _update_light_curve_data_for_next_epoch(
        light_curve_data: DataBase, next_day_data: DataBase,
        canonical_data: DataBase, is_queryable: bool, strategy: str,
        is_separate_files: bool) -> DataBase:
    """
    Updates samples for next epoch

    Parameters
    ----------
    light_curve_data
        light curve learning data
    next_day_data
        next day light curve data
    canonical_data
        canonical strategy light curve data
    is_queryable
        If True, allow queries only on objects flagged as queryable.
        Default is True.
    strategy
        Query strategy. Options are (all can be run with budget):
        "UncSampling", "UncSamplingEntropy", "UncSamplingLeastConfident",
        "UncSamplingMargin", "QBDMI", "QBDEntropy", "RandomSampling",
    is_separate_files
        If True, consider samples separately read
        from independent files. Default is False.
    """
    light_curve_data.pool_metadata = next_day_data.pool_metadata
    light_curve_data.pool_features = next_day_data.pool_features
    light_curve_data.pool_labels = next_day_data.pool_labels

    if not is_separate_files:
        light_curve_data.test_metadata = next_day_data.test_metadata
        light_curve_data.test_features = next_day_data.test_features
        light_curve_data.test_labels = next_day_data.test_labels

        light_curve_data.validation_metadata = next_day_data.validation_metadata
        light_curve_data.validation_features = next_day_data.validation_features
        light_curve_data.validation_labels = next_day_data.validation_labels

    if strategy == 'canonical':
        light_curve_data.queryable_ids = canonical_data.queryable_ids

    if is_queryable:
        queryable_flag = light_curve_data.pool_metadata['queryable'].values
        light_curve_data.queryable_ids = light_curve_data.pool_metadata[
            'id'].values[queryable_flag]
    else:
        light_curve_data.queryable_ids = light_curve_data.pool_metadata[
            'id'].values
    return light_curve_data


# TODO: Too many arguments. Refactor and update docs
def process_next_day_loop(
    light_curve_data: DataBase,
    next_day_features_file_name: str,
    id_key_name: str,
    light_curve_train_ids: np.ndarray,
    canonical_data: DataBase,
    config: TimeDomainConfiguration,
) -> DataBase:
    """
    Runs next day active learning loop

    Parameters
    ----------
    light_curve_data
        next day light curve data
    next_day_features_file_name
        path to next day features file name
    id_key_name
        object identification key name
    light_curve_train_ids
        light curve data metadata values
    canonical_data
        canonical strategy light curve data
    config
        The `TimeDomainConfiguration` dataclass that
        contains all the runtime configuration options.
    """
    next_day_data = _load_next_day_data(next_day_features_file_name, config)
    for metadata_value in light_curve_data.train_metadata[id_key_name].values:
        next_day_pool_metadata = next_day_data.pool_metadata[id_key_name].values
        if metadata_value in next_day_pool_metadata:
            next_day_pool_metadata_indices = list(next_day_pool_metadata).index(metadata_value)
            if metadata_value not in light_curve_train_ids:
                light_curve_train_metadata = light_curve_data.train_metadata[id_key_name].values
                light_curve_data = _remove_old_training_features(
                    light_curve_data,
                    light_curve_train_metadata,
                    metadata_value
                )
                if light_curve_data.queryable_ids.shape[0] > 0:
                    light_curve_data = _update_queried_sample(
                        light_curve_data,
                        next_day_data,
                        id_key_name,
                        metadata_value
                    )
                light_curve_data = _update_training_data_with_new_features(
                    light_curve_data,
                    next_day_data,
                    metadata_value,
                    id_key_name
                )
            next_day_data = _update_next_day_pool_data(
                next_day_data,
                next_day_pool_metadata_indices
            )
        next_day_data = _update_next_day_val_and_test_data(
            next_day_data,
            metadata_value,
            id_key_name
        )
    light_curve_data = _update_light_curve_data_for_next_epoch(
        light_curve_data,
        next_day_data,
        canonical_data,
        config.queryable,
        config.strategy,
        config.sep_files
    )
    return light_curve_data


def submit_queries_to_TOM(username, passwordfile, objectids: list, priorities: list, requester: str='resspect'):
    tom = TomClient(url = "https://desc-tom-2.lbl.gov", username = username, passwordfile = passwordfile)
    req = { 'requester': requester,
            'objectids': objectids,
            'priorities': priorities}
    res = tom.request( 'POST', 'elasticc2/askforspectrum', json=req )
    dic = res.json()
    if res.status_code != 200:
        raise ValueError('Request failed, ' + res.text + ". Status code: " + str(res.status_code))
    
    if dic['status'] == 'error':
        raise ValueError('Request failed, ' + dic.json()['error'])


# TODO: Too many arguments. Refactor and update docs
def run_time_domain_active_learning_loop(
    light_curve_data: DataBase,
    id_key_name: str,
    light_curve_train_ids: np.ndarray,
    canonical_data: DataBase,
    config: TimeDomainConfiguration,
    **kwargs: dict
):
    """
    Runs time domain active learning loop

    Parameters
    ----------
    light_curve_data
        light curve learning data
    id_key_name
        object identification key name
    light_curve_train_ids
        light curve data metadata values
    canonical_data
        canonical strategy light curve data
    config
        The `TimeDomainConfiguration` dataclass that
        contains all the runtime configuration options.
    kwargs
       All keywords required by the classifier function.

    Returns
    -------

    """
    learning_days = [int(each_day) for each_day in config.days]
    
    # create dictionary with budgets
    if config.budgets is not None and len(config.budgets) not in [2, len(np.arange(config.days[0], config.days[1]))]:
        raise ValueError('There must be 1 budget per telescope or ' + \
                            '1 budget per telescope per night!')

    budgets_dict = {}
    for epoch in range(config.days[0], config.days[-1] - 1):
        budgets_dict[epoch] = config.budgets
    
    for epoch in progressbar.progressbar(
            range(learning_days[0], learning_days[-1] - 1)):
        if light_curve_data.pool_features.shape[0] > 0:
            light_curve_data = _run_classification_and_evaluation(
                light_curve_data, config.classifier, config.clf_bootstrap, **kwargs
            )
            if light_curve_data.queryable_ids.shape[0] > 0:
                object_indices = _get_indices_of_objects_to_be_queried(
                    light_curve_data,
                    budgets_dict[epoch],
                    config,
                )
                light_curve_data = _update_samples_with_object_indices(
                    light_curve_data, object_indices, config.queryable, epoch
                )
        _save_metrics_and_queried_sample(
            light_curve_data, epoch - learning_days[0],
            config.output_metrics_file,
            config.output_queried_file,
            len(object_indices),
            epoch,
            config. save_full_query,
        )
        next_day_features_file_name = f"{config.fname_pattern[0]}{epoch + 1}{config.fname_pattern[1]}"
        next_day_features_file_path = os.path.join(config.path_to_features_dir, next_day_features_file_name)
        light_curve_data = process_next_day_loop(
            light_curve_data,
            next_day_features_file_path,
            id_key_name,
            light_curve_train_ids,
            canonical_data,
            config,
        )


# TODO: Too many arguments. Refactor and update docs
def time_domain_loop(config, **kwargs):
    """
    Perform the active learning loop. All results are saved to file.

    Parameters
    ----------
    config
        The `TimeDomainConfiguration` dataclass that
        contains all the runtime configuration options.
    """

    # load features for the first obs day
    first_loop_file_name = f"{config.fname_pattern[0]}{config.days[0]}{config.fname_pattern[1]}"
    first_loop_file_path = os.path.join(
        config.path_to_features_dir,
        first_loop_file_name
    )

    first_loop_data, light_curve_data = _load_first_loop_and_full_data(
        first_loop_file_path, config
    )

    # get keyword for obj identification
    id_key_name = light_curve_data.identify_keywords()
    light_curve_train_ids = light_curve_data.train_metadata[id_key_name].values

    first_loop_data, light_curve_data = _update_data_by_remove_repeated_ids(
        first_loop_data, light_curve_data, id_key_name
    )
    light_curve_data = _update_light_curve_data_val_and_test_data(
        light_curve_data,
        first_loop_data,
        config.sep_files,
        config.initial_training,
        config.queryable
    )
    light_curve_data = _update_queryable_ids(
        light_curve_data, id_key_name, config.queryable
    )
    light_curve_data, canonical_data = _update_canonical_ids(
        light_curve_data, config.path_to_canonical, config.canonical
    )
    light_curve_data = _update_initial_train_meta_data_header(
        first_loop_data, light_curve_data
    )
    run_time_domain_active_learning_loop(
        light_curve_data,
        id_key_name,
        light_curve_train_ids,
        canonical_data,
        config,
        **kwargs
    )

def main():
    return None


if __name__ == '__main__':
    main()
