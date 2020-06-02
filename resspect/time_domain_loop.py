# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
#      
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

__all__ = ['time_domain_loop']

import numpy as np
import pandas as pd

from resspect import DataBase

def time_domain_loop(days: list,  output_metrics_file: str,
                     output_queried_file: str,
                     path_to_features_dir: str, strategy: str,
                     fname_pattern: list,
                     batch=1, canonical = False,  classifier='RandomForest',
                     cont=False, first_loop=20, features_method='Bazin', nclass=2,
                     Ia_frac=0.5, output_fname="", path_to_canonical="",
                     path_to_ini_train="", path_to_train="",
                     path_to_queried="", queryable=True,
                     query_thre=1.0, save_samples=False, sep_files=False,
                     screen=True, survey='LSST', initial_training='original'):
    """Perform the active learning loop. All results are saved to file.

    Parameters
    ----------
    days: list
        List of 2 elements. First and last day of observations since the
        beginning of the survey.
    output_metrics_file: str
        Full path to output file to store metrics for each loop.
    output_queried_file: str
        Full path to output file to store the queried sample.
    path_to_features_dir: str
        Complete path to directory holding features files for all days.
    strategy: str
        Query strategy. Options are 'UncSampling' and 'RandomSampling'.
    fname_pattern: str
        List of strings. Set the pattern for filename, except day of 
        survey. If file name is 'day_1_vx.dat' -> ['day_', '_vx.dat']
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    canonical: bool (optional)
        If True, restrict the search to the canonical sample.
    continue: bool (optional)
        If True, read the initial states of previous runs from file.
        Default is False.
    classifier: str (optional)
        Machine Learning algorithm.
        Currently 'RandomForest', 'GradientBoostedTrees',
        'KNN', 'MLP', 'SVM' and 'NB' are implemented.
        Default is 'RandomForest'.
    first_loop: int (optional)
        First day of the survey already calculated in previous runs.
        Only used if initial_training == 'previous'.
        Default is 20.
    features_method: str (optional)
        Feature extraction method. Currently only 'Bazin' is implemented.
    Ia_frac: float in [0,1] (optional)
        Fraction of Ia required in initial training sample.
        Default is 0.5.
    nclass: int (optional)
        Number of classes to consider in the classification
        Currently only nclass == 2 is implemented.
    path_to_canonical: str (optional)
        Path to canonical sample features files.
        It is only used if "strategy==canonical".
    path_to_ini_train: str (optional)
        Path to full light curve features file.
        Only used if "training == 'original'".
    path_to_train: str (optional)
        Path to initial training file from previous run.
        Only used if initial_training == 'previous'.
    path_to_queried: str(optional)
        Path to queried sample from previous run.
        Only used if initial_training == 'previous'.
    queryable: bool (optional)
        If True, allow queries only on objects flagged as queryable.
        Default is True.
    query_thre: float (optional)
        Percentile threshold for query. Default is 1. 
    save_samples: bool (optional)
        If True, save training and test samples to file.
        Default is False.
    screen: bool (optional)
        If True, print on screen number of light curves processed.
    sep_files: bool (optional)
        If True, consider train and test samples separately read 
        from independent files. Default is False.
    survey: str (optional)
        Name of survey to be analyzed. Accepts 'DES' or 'LSST'.
        Default is LSST.
    initial_training: str or int (optional)
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        eilf 'previous': read training and queried from previous run.
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia
        Default is 'original'.
    output_fname: str (optional)
        Complete path to output file where initial training will be stored.
        Only used if save_samples == True.
    """

    # initiate object
    data = DataBase()

    # constructs training, test, validation and pool samples
    data.load_features(path_to_ini_train, method=features_method,
                       screen=screen, survey=survey)
    
    data.build_samples(initial_training=initial_training, nclass=nclass,
                       screen=screen, Ia_frac=Ia_frac,
                       queryable=queryable, save_samples=save_samples,
                       sep_files=sep_files, survey=survey, 
                       output_fname=output_fname,
                       path_to_train=path_to_train,
                       path_to_queried=path_to_queried,
                       method=features_method)
    
    # load features for the first obs day
    path_to_features3 = path_to_features_dir + fname_pattern[0] + \
                                       str(days[0]) + fname_pattern[1]
    
    first_loop = DataBase()
    first_loop.load_features(path_to_features3)
    first_loop.build_samples(initial_training=initial_training, 
                             nclass=nclass,
                             screen=screen, Ia_frac=Ia_frac,
                             queryable=queryable, 
                             save_samples=save_samples,
                             sep_files=sep_files, survey=survey,
                             output_fname=output_fname,
                             path_to_train=path_to_train,
                             path_to_queried=path_to_queried,
                             method=features_method)
            
    # update test and pool sample
    data.pool_features = first_loop.pool_features
    data.pool_labels = first_loop.pool_labels
    data.pool_metadata = first_loop.pool_metadata
    
    if sep_files:
        pass
    else:
        data.test_features = first_loop.features
        test_labels = first_loop.metadata['type'].values == 'Ia'
        data.test_labels = test_labels.astype(int)
        data.test_metadata = first_loop.metadata
        data.validation_features = first_loop.features
        data.validation_metadata = first_loop.metadata
        data.validation_labels = data.test_labels
                
    if queryable:
        q_flag = data.pool_metadata['queryable'].values
        data.queryable_ids = data.pool_metadata['id'].values[q_flag]

    # get list of canonical ids
    if canonical:
        canonical = DataBase()
        canonical.load_features(path_to_file=path_to_canonical)
        data.queryable_ids = canonical.queryable_ids
        
    # check if all headers in test exist in train
    for name in first_loop.metadata_names:
        if name not in data.metadata_names:
            data.metadata_names.append(name)
            data.metadata[name] = None
            data.train_metadata.insert(len(data.metadata_names) - 1, 
                                       name, None, True)

    for night in range(int(days[0]), int(days[-1]) - 1):
            
        if screen:
            print('Processing night: ', night)

        # cont loop
        if initial_training == 'previous':
            loop = night - first_loop
        else:
            loop = night - int(days[0])

        if data.pool_metadata.shape[0] > 0:
            # classify
            data.classify(method=classifier)

            # calculate metrics
            data.evaluate_classification()
        
            indx = data.make_query(strategy=strategy, batch=batch, 
                                   queryable=queryable,
                                   query_thre=query_thre)

            # update training and test samples
            data.update_samples(indx, loop=loop, queryable=queryable, 
                                epoch=loop)
            
            if screen:
                print('After update_samples:')
                print('   ... train: ', data.train_metadata.shape[0])
                print('   ... pool: ', data.pool_metadata.shape[0])

        # save metrics for current state
        data.save_metrics(loop=loop, output_metrics_file=output_metrics_file,
                          batch=batch, epoch=night)

        # save query sample to file
        data.save_queried_sample(output_queried_file, loop=loop,
                                     full_sample=False)

        # load features for next day
        path_to_features2 = path_to_features_dir + fname_pattern[0] + \
                            str(night + 1) + fname_pattern[1]

        data_tomorrow = DataBase()
        data_tomorrow.load_features(path_to_features2, method=features_method,
                                    screen=screen, survey=survey)

        # identify objects in the new day which must be in training
        train_flag = np.array([item in data.train_metadata['id'].values 
                              for item in data_tomorrow.metadata['id'].values])
        train_ids = data_tomorrow.metadata['id'].values[train_flag]
        
        # keep objs who were in training but are not in the new day
        keep_flag = np.array([item not in train_ids 
                              for item in data.train_metadata['id'].values])
        
        # use new data for training (this might have extra obs points)
        data.train_metadata = pd.concat([data.train_metadata[keep_flag],
                                         data_tomorrow.metadata[train_flag]])
        data.train_features = np.append(data.train_features[keep_flag],
                                        data_tomorrow.features[train_flag],
                                        axis=0)
         
        train_labels = data.train_metadata['type'].values == 'Ia'
        data.train_labels = train_labels.astype(int)

        # use new data
        if sep_files:
            pass
        else:
            data.pool_metadata = data_tomorrow.metadata[~train_flag]
            data.pool_features = data_tomorrow.features.values[~train_flag]
            data.validation_features = data.pool_features
            data.validation_metadata = data.pool_metadata
            
            pool_labels = data.pool_metadata['type'] == 'Ia'
            data.pool_labels = pool_labels.astype(int)
            data.validation_labels = data.pool_labels

        if strategy == 'canonical':
            data.queryable_ids = canonical.queryable_ids

        if  queryable:
            queryable_flag = data.pool_metadata['queryable'].values
            data.queryable_ids = data.pool_metadata['id'].values[queryable_flag]
        else:
            data.queryable_ids = data.pool_metadata['id'].values

        if screen:
            print('Training set size: ', data.train_metadata.shape[0])
            print('Test set size: ', data.test_metadata.shape[0])
            print('Queryable set size: ', len(data.queryable_ids))


def main():
    return None


if __name__ == '__main__':
    main()
