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
from resspect.loop_configuration import LoopConfiguration


def load_features(database_class: DataBase, config: LoopConfiguration) -> DataBase:
    """
    Load features according to feature extraction method

    Parameters
    ----------
    database_class: `resspect.DataBase`
        An instance of DataBase class
    config: `resspect.loop_configuration.LoopConfiguration`
        The configuration elements of the learn loop.
    """
    if isinstance(config.path_to_features, str):
        database_class.load_features(
            path_to_file=config.path_to_features,
            feature_extractor=config.features_method,
            survey=config.survey
        )
    else:
        features_set_names = ['train', 'test', 'validation', 'pool']
        for sample_name in features_set_names:
            if sample_name in config.path_to_features.keys():
                database_class.load_features(
                    config.path_to_features[sample_name],
                    feature_extractor=config.features_method,
                    survey=config.survey,
                    sample=sample_name
                )
            else:
                logging.warning(f'Path to {sample_name} not given.'
                                f' Proceeding without this sample')

    database_class.build_samples(
        initial_training=config.training,
        nclass=config.nclass,
        queryable=config.queryable,
        sep_files=config.sep_files,
        output_fname=config.initial_training_samples_file
    )

    return database_class


def run_classification(
    database_class: DataBase,
    config: LoopConfiguration,
    iteration_step: int,
    **kwargs: dict
) -> DataBase:
    """
    Run active learning classification model

    Parameters
    ----------
    database_class: `resspect.DataBase`
        An instance of DataBase class
    config: `resspect.loop_configuration.LoopConfiguration`
        An instance of the LoopConfiguration class containing
        relevant elements.
    iteration_step: int
        active learning iteration number
    kwargs: dict
       All keywords required by the classifier function.
    -------

    """
    if config.classifier_bootstrap:
        database_class.classify_bootstrap(
            method=config.classifier,
            loop=iteration_step,
            pred_dir=config.prediction_dir,
            save_predictions=config.save_predictions,
            **kwargs
        )
    else:
        database_class.classify(
            method=config.classifier,
            pred_dir=config.pred_dir,
            loop=iteration_step,
            save_predictions=config.save_predictions,
            pretrained_model_path=config.pretrained_model_path,
            **kwargs
        )
    return database_class


def run_evaluation(database_class: DataBase, metric_label: str):
    """
    Evaluates the active learning model

    Parameters
    ----------
    database_class: `resspect.DataBase`
        An instance of DataBase class
    metric_label: str
        Choice of metric.
        Currently only "snpcc", "cosmo" or "snpcc_cosmo" are accepted.
        Default is "snpcc".

    """
    database_class.evaluate_classification(metric_label=metric_label)


def save_photo_ids(
    database_class: DataBase,
    config: LoopConfiguration,
    iteration_step: int,
    file_name_suffix: str = None
):
    """
    Function to save photo IDs to a file

    Parameters
    ----------
    database_class: `resspect.DataBase`
        An instance of DataBase class
    config: `resspect.loop_configuration.LoopConfiguration`
    iteration_step: int
        active learning iteration number
    file_name_suffix: str
        suffix string for save file name with file extension
    """
    if config.photo_ids_to_file or config.SNANA_types:
        file_name = config.photo_ids_froot + '_' + str(iteration_step) + file_name_suffix
        database_class.output_photo_Ia(
            config.photo_class_thr,
            to_file=config.photo_ids_to_file,
            filename=file_name,
            SNANA_types=config.SNANA_types,
            metadata_fname=config.metadata_fname
        )


def run_make_query(
    database_class: DataBase,
    strategy: str,
    batch_size: int,
    is_queryable: bool
):
    """
    Run active learning query process

    Parameters
    ----------
    database_class: `resspect.DataBase`
        An instance of DataBase class
    strategy: str
        Strategy used to choose the most informative object.
        Current implementation accepts 'UncSampling' and
        'RandomSampling', 'UncSamplingEntropy',
        'UncSamplingLeastConfident', 'UncSamplingMargin',
        'QBDMI', 'QBDEntropy', . Default is `UncSampling`.
    batch_size: int
        Number of objects to be chosen in each batch query.
        Default is 1
    is_queryable: bool
        If True, consider only queryable objects.
        Default is False.
    """
    return database_class.make_query(
        strategy=strategy,
        batch=batch_size,
        queryable=is_queryable
    )


def _save_metrics_and_queried_samples(
    database_class: DataBase,
    metrics_file_name: str,
    queried_file_name: str,
    iteration_step: int,
    batch: int,
    full_sample: bool,
    file_name_suffix: str = None
):
    """
    Save metrics and queried samples details

    Parameters
    ----------
    database_class: `resspect.DataBase`
        An instance of DataBase class
    metrics_file_name: str
        Full path to file to store metrics results.
    queried_file_name: str
        Complete path to output file.
    iteration_step: int
        active learning iteration number
    batch: int
        Number of queries in each loop.
    full_sample: bool
        If true, write down a complete queried sample stored in
        property 'queried_sample'. Otherwise append 1 line per loop to
        'queried_sample_file'. Default is False.
    file_name_suffix: str
        suffix string for save file name with file extension
    """
    if file_name_suffix is not None:
        metrics_file_name = metrics_file_name.replace('.dat', file_name_suffix)
        queried_file_name = queried_file_name.replace(
            '.dat',
            file_name_suffix
        )
    database_class.save_metrics(
        loop=iteration_step,
        output_metrics_file=metrics_file_name,
        batch=batch, epoch=iteration_step
    )
    database_class.save_queried_sample(
        queried_file_name,
        loop=iteration_step,
        full_sample=full_sample,
        epoch=iteration_step,
        batch=batch
    )


def update_alternative_label(
    database_class_alternative: DataBase,
    indices_to_query: list,
    iteration_step: int,
    config: LoopConfiguration,
    **kwargs: dict
):
    """
    Function to update active learning training with alternative label

    Parameters
    ----------
    database_class_alternative: `resspect.DataBase`
        An instance of DataBase class for alternative label
    indices_to_query: list
        List of indexes identifying objects to be moved.
    iteration_step: int
        active learning iteration number
    config: `resspect.loop_configuration.LoopConfiguration`
    kwargs
        additional arguments
    """
    database_class_alternative.update_samples(
        indices_to_query,
        epoch=iteration_step,
        alternative_label=True)
    
    database_class_alternative = run_classification(
        database_class_alternative,
        config,
        iteration_step,
        **kwargs
    )
    
    run_evaluation(database_class_alternative, config.metric_label)
    
    save_photo_ids(
        database_class_alternative,
        config,
        iteration_step,
        '_alt_label.dat'
    )
    
    return database_class_alternative


# TODO: too many arguments! refactor further and update docs
def learn_loop(config, **kwargs):
    """
    Perform the active learning loop. All results are saved to file.

    Parameters
    ----------
    config: `resspect.LoopConfiguration`.
        The config values for the learn loop. See the `LoopConfiguration`
        documentation for more info.
    kwargs: extra parameters
        All keywords required by the classifier function.
    """
    # initiate object
    database_class = DataBase()
    logging.info('Loading features')
    database_class = load_features(
        database_class,
        config
    )
    
    logging.info('Running active learning loop')
    
    if config.bar:
        ensemble = progressbar.progressbar(range(config.nloops))
    else:
        ensemble = range(config.nloops)
    
    for iteration_step in ensemble:
        if not config.bar:
            print(iteration_step)

        database_class = run_classification(database_class, config, iteration_step, **kwargs)
        run_evaluation(database_class, config.metric_label)

        save_photo_ids(database_class, config, iteration_step,'.dat')
        indices_to_query = run_make_query(
            database_class,
            config.strategy,
            config.batch,
            config.queryable,
        )
        if config.save_alt_class and config.batch == 1:
            database_class_alternative = copy.deepcopy(database_class)
            database_class_alternative = update_alternative_label(
                database_class_alternative,
                indices_to_query,
                iteration_step,
                config,
                **kwargs
            )
            _save_metrics_and_queried_samples(
                database_class_alternative,
                config.output_metrics_file,
                config.output_queried_file,
                iteration_step,
                config.batch,
                False,
                '_alt_label.dat',
            )
            
        elif config.save_alt_class and config.batch > 1:
            raise ValueError('Alternative label only works with batch=1!')

        database_class.update_samples(
            indices_to_query,
            epoch=iteration_step,
            queryable=config.queryable,
            alternative_label=False
        )
        _save_metrics_and_queried_samples(
            database_class,
            config.output_metrics_file,
            config.output_queried_file,
            iteration_step,
            config.batch,
            False
        )
    return database_class


def main():
    return None


if __name__ == '__main__':
    main()

