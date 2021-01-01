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

__all__ = ['learn_loop']

from resspect import DataBase

import copy


def learn_loop(nloops: int, strategy: str, path_to_features: str,
               output_metrics_file: str, output_queried_file: str,
               features_method='Bazin', classifier='RandomForest',
               training='original', batch=1, screen=True, survey='DES',
               nclass=2, photo_class_thr=0.5, photo_ids=False, photo_ids_tofile = False,
               photo_ids_froot=' ', classifier_bootstrap=False, save_predictions=False,
               sep_files=False, pred_dir=None, queryable=False, 
               metric_label='snpcc', dist_loop_root=None, save_alt_class=False,
               SNANA_types=False,  **kwargs):
    """Perform the active learning loop. All results are saved to file.

    Parameters
    ----------
    nloops: int
        Number of active learning loops to run.
    strategy: str
        Query strategy. Options are 'UncSampling', 'RandomSampling', 'UncSamplingEntropy',
        'UncSamplingLeastConfident', 'UncSamplingMargin', 'QBDMI' and 'QBDEntropy'.
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
        Feature extraction method. Currently only 'Bazin' is implemented.
    classifier: str
        Machine Learning algorithm.
        Currently implemented options are 'RandomForest', 'GradientBoostedTrees',
        'K-NNclassifier','MLPclassifier','SVMclassifier' and 'NBclassifier'.
    sep_files: bool (optional)
        If True, consider train and test samples separately read
        from independent files. Default is False.    
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    bootstrap: bool (optional)
        Flag for bootstrapping on the classifier
        Must be true if using disagreement based strategy.
    dist_loop_root: str (optional)
        Pattern for file storing distances in each learn loop.
        Only used if "metric_label" is "cosmo" or "snpcc_cosmo".
    metric_label: str (optional)
        Choice of metric. 
        Currenlty only "snpcc", "cosmo" or "snpcc_cosmo" are accepted.
        Default is "snpcc".
    nclass: int (optional)
        Number of classes to consider in the classification
        Currently only nclass == 2 is implemented.
    photo_class_thr: float (optional)
        Threshold for photometric classification. Default is 0.5.
        Only used if photo_ids is True.
    photo_ids: bool (optional)
        Get photometrically classified ids. Default is False.
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
    screen: bool (optional)
        If True, print on screen number of light curves processed.
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
    kwargs: extra parameters
        All keywords required by the classifier function.
    """
    if 'QBD' in strategy and not classifier_bootstrap:
        raise ValueError('Bootstrap must be true when using ' + \
                         'disagreement strategy.')

    # initiate object
    data = DataBase()

    # load features
    if isinstance(path_to_features, str):
        data.load_features(path_to_features, method=features_method,
                           screen=screen, survey=survey)

        # separate training and test samples
        data.build_samples(initial_training=training, nclass=nclass,
                          queryable=queryable)

    else:
        for name in ['train', 'test', 'validation', 'pool']:
            if name in path_to_features.keys():
                data.load_features(path_to_features[name], method=features_method,
                                   screen=screen, survey=survey, sample=name)
            elif screen:
                print('Path to ' + name + 'not given. Proceeding without this sample.')

        data.build_samples(initial_training=training, nclass=nclass,
                           screen=screen, sep_files=True, queryable=queryable)

    for loop in range(nloops):

        if screen:
            print('Processing... ', loop)

        # classify
        if classifier_bootstrap:
            data.classify_bootstrap(method=classifier, save_predictions=save_predictions,
                                    pred_dir=pred_dir, loop=loop, screen=screen, **kwargs)            
        else:
            data.classify(method=classifier, save_predictions=save_predictions,
                          pred_dir=pred_dir, loop=loop, screen=screen, **kwargs)

        # calculate metrics
        data.evaluate_classification(metric_label=metric_label, screen=screen)
        
        # save photo ids
        if photo_ids and photo_ids_tofile:
            fname = photo_ids_froot + '_' + str(loop) + '.dat'
            data.output_photo_Ia(photo_class_thr, to_file=photo_ids_tofile,
                                 filename=fname, SNANA_types=SNANA_types)
        elif photo_ids:
            data.output_photo_Ia(photo_class_thr, to_file=False,
                                 SNANA_types=SNANA_types)

        # choose object to query
        indx = data.make_query(strategy=strategy, batch=batch, queryable=queryable,
                               screen=screen)
        
        # make copy of index (why is this necessary??)
        indx2 = copy.deepcopy(indx)

        # update training with alternative label
        if save_alt_class and batch == 1:
            # create a copy of the DataBase object
            data_alt = copy.deepcopy(data)
            # update with the less probable label
            data_alt.update_samples(indx2, epoch=loop, alternative_label=True,
                                    screen=screen)
            # classify
            data_alt.classify(method=classifier, save_predictions=save_predictions,
                              pred_dir=pred_dir, loop=loop, screen=screen, 
                              **kwargs)
            # evaluate classification
            data_alt.evaluate_classification(metric_label=metric_label, screen=screen)
            # save photo ids  
            fname_alt = photo_ids_froot + '_' + str(loop) + '_alt_label.dat'
            data_alt.output_photo_Ia(photo_class_thr, to_file=photo_ids_tofile,
                                     filename=fname_alt, SNAN_types=SNANA_types)

            # save metrics for alternate state
            output_metrics_file_alt = output_metrics_file[:-4] + '_alt_label.dat'
            data_alt.save_metrics(loop=loop, output_metrics_file=output_metrics_file_alt,
                                  batch=batch, epoch=loop)
            # save queried sample for alternate state
            output_queried_file_alt = output_queried_file[:-4] + '_alt_label.dat'
            data_alt.save_queried_sample(output_queried_file_alt, loop=loop,
                                         full_sample=False, epoch=loop, batch=batch)
            
            del data_alt
            
        elif save_alt_class and batch > 1:
            raise ValueError('Alternative label only works with batch=1!')
        
        # update training and test samples
        data.update_samples(indx, epoch=loop, screen=screen, queryable=queryable,
                            alternative_label=False)

        # save metrics for current state
        data.save_metrics(loop=loop, output_metrics_file=output_metrics_file,
                          batch=batch, epoch=loop)

        # save query sample to file
        data.save_queried_sample(output_queried_file, loop=loop,
                                 full_sample=False, epoch=loop, batch=batch)


def main():
    return None


if __name__ == '__main__':
    main()
