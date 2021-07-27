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

__all__ = ['time_domain_loop', 'load_dataset']

import numpy as np
import pandas as pd

from resspect import DataBase

def load_dataset(fname: str, survey='DES',
                         screen=False, initial_training='original',
                         ia_frac=0.5, queryable=False, sep_files=False,
                         save_samples=False):
    """Read a data sample from file.

    Parameters
    ----------
    fname: str #or dict
        Path to light curve features file.
        #if "sep_files == True", dictionary keywords must contain identify
        #different samples: ['train', 'test','validation', 'pool']
    ia_frac: float in [0,1] (optional)
        Fraction of Ia required in initial training sample.
        Only used if "initial_training" is a number. Default is 0.5.
    initial_training: str or int (optional)
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        elif int: choose the required number of samples at random,
        ensuring that at least "ia_frac" are SN Ia.
        Default is 'original'.
    queryable: bool (optional)
        If True, allow queries only on objects flagged as queryable.
        Default is True.
    save_samples: bool (optional)
        If True, save training and test samples to file.
        Default is False.
    sep_files: bool (optional)
            If True, consider samples separately read
            from independent files. Default is False.
    screen: bool (optional)
        If True, print on screen number of light curves processed.
    survey: str (optional)
        Name of survey to be analyzed. Accepts 'DES' or 'LSST'.
        Default is DES.

    Returns
    -------
    resspect.DataBase object
    """

    # initiate object
    data = DataBase()

    # constructs training, test, validation and pool samples
    data.load_features(fname, screen=screen, survey=survey)

     # get identification keyword
    id_name = data.identify_keywords()

    data.build_samples(initial_training=initial_training, nclass=2,
                       screen=screen, Ia_frac=ia_frac,
                       queryable=queryable, save_samples=save_samples,
                       sep_files=sep_files, survey=survey)

    return data


def time_domain_loop(days: list,  output_metrics_file: str,
                     output_queried_file: str,
                     path_to_features_dir: str, strategy: str,
                     fname_pattern: list, path_to_ini_files: dict,
                     batch=1, canonical = False,  classifier='RandomForest',
                     clf_bootstrap=False, budgets=None ,cont=False,
                     first_loop=20, nclass=2, ia_frac=0.5, output_fname="",
                     path_to_canonical="", path_to_train="",
                     path_to_queried="", queryable=True,
                     query_thre=1.0, save_samples=False, sep_files=False,
                     screen=True, survey='LSST', initial_training='original',
                     save_full_query=False, save_batches=False, 
                     batch_outfile=None, num_batches=1, **kwargs):
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
        Query strategy. Options are (all can be run with budget):
        "UncSampling",
        "UncSamplingEntropy",
        "UncSamplingLeastConfident",
        "UncSamplingMargin",
        "QBDMI",
        "QBDEntropy",
        "RandomSampling",
    fname_pattern: str
        List of strings. Set the pattern for filename, except day of
        survey. If file name is 'day_1_vx.dat' -> ['day_', '_vx.dat'].
    path_to_ini_files: dict (optional)
        Path to initial full light curve files.
        Possible keywords are: "train", "test" and "validation".
        At least "train" is mandatory.
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    batch_outfile: str (optional)
        Name of file to save batches. Only used it save_batches == True.
    canonical: bool (optional)
        If True, restrict the search to the canonical sample.
    classifier: str (optional)
        Machine Learning algorithm.
        Currently 'RandomForest', 'GradientBoostedTrees',
        'KNN', 'MLP', 'SVM' and 'NB' are implemented.
        Default is 'RandomForest'.
    clf_bootstrap: bool (default: False)
        If true build a boostrap ensemble of the classifier.
    budgets: tuple of floats (default: None)
        Budgets for each of the telescopes
    features_method: str (optional)
        Feature extraction method. Currently only 'Bazin' is implemented.
    ia_frac: float in [0,1] (optional)
        Fraction of Ia required in initial training sample.
        Default is 0.5.
    loop: int (optional)
        Loop identification only used to store query in file.
        Only use if save_batches == True.
    num_batches: int (optional)
        Number of batches to considered. Default is 1.
        Only use if budgets are considered.
    path_to_canonical: str (optional)
        Path to canonical sample features files.
        It is only used if "strategy==canonical".
    queryable: bool (optional)
        If True, allow queries only on objects flagged as queryable.
        Default is True.
    query_thre: float (optional)
        Percentile threshold for query. Default is 1.
    save_batches: bool (optional)
        If True, save batches to file. Default is False. 
    save_samples: bool (optional)
        If True, save training and test samples to file.
        Default is False.
    save_full_query: bool (optional)
        If True, save complete queried sample to file. 
        Otherwise, save only first element. Default is False.
    screen: bool (optional)
        If True, print on screen number of light curves processed.
    sep_files: bool (optional)
        If True, consider samples separately read
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

    # load features for the first obs day
    path_to_first_loop = path_to_features_dir + fname_pattern[0] + \
                                     str(days[0]) + fname_pattern[1]

    # read data from fist loop
    if not sep_files:
        first_loop = load_dataset(fname=path_to_first_loop,
                                 survey=survey, sep_files=False,
                                 screen=screen,
                                 initial_training=0,
                                 ia_frac=ia_frac, queryable=queryable)
        
    else:
        first_loop = DataBase()
        first_loop.load_features(path_to_file=path_to_first_loop,
                                 survey=survey, screen=screen, method='Bazin',
                                 sample='pool')
        first_loop.build_samples(initial_training='original', nclass=2,
                                 screen=screen, queryable=queryable,
                                 save_samples=save_samples, sep_files=sep_files,
                                 survey=survey)
        
    if sep_files:
        # initiate object
        data = DataBase()

        # constructs training, test, validation and pool samples
        data.load_features(path_to_file=path_to_ini_files['train'],
                           screen=screen, method='Bazin', survey=survey,
                           sample='train')

        for s in ['test', 'validation']:
            data.load_features(path_to_file=path_to_ini_files[s],
                               method='Bazin', screen=screen,
                               survey=survey, sample=s)
    else:
        # read initial training
        data = load_dataset(fname=path_to_ini_files['train'],
                            survey=survey, sep_files=sep_files,
                            screen=screen,
                            initial_training=initial_training,
                            ia_frac=ia_frac, queryable=queryable)

    # get keyword for obj identification
    id_name = data.identify_keywords()

    # get ids from initial training
    ini_train_ids = data.train_metadata[id_name].values

    # remove repeated ids
    rep_ids_flag = np.array([item in data.train_metadata[id_name].values
                            for item in first_loop.pool_metadata[id_name].values])

    first_loop.pool_metadata = first_loop.pool_metadata[~rep_ids_flag]
    first_loop.pool_features = first_loop.pool_features[~rep_ids_flag]
    pool_labels = first_loop.pool_metadata['type'].values == 'Ia'
    first_loop.pool_labels = pool_labels.astype(int)

    data.pool_features = first_loop.pool_features
    data.pool_metadata = first_loop.pool_metadata
    data.pool_labels = first_loop.pool_labels

    if sep_files:
        data.build_samples(nclass=2, screen=screen,
                                queryable=queryable,
                                sep_files=True, 
                           initial_training=initial_training)
    else:
        data.test_features = first_loop.pool_features
        data.test_metadata = first_loop.pool_metadata
        data.test_labels = first_loop.pool_labels

        data.validation_features = first_loop.pool_features
        data.validation_metadata = first_loop.pool_metadata
        data.validation_labels = first_loop.pool_labels


    if queryable:
        q_flag = data.pool_metadata['queryable'].values
        data.queryable_ids = data.pool_metadata[id_name].values[q_flag]
    else:
        data.queryable_ids = data.pool_metadata[id_name].values

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


    # get validation ids
    validation_ids = data.validation_metadata[id_name].values

    for night in range(int(days[0]), int(days[-1]) - 1):

        if screen:
            print('\n ****************************')
            print(' Processing night: ', night, '\n\n')
            print(' Before update_samples:')
            print('   ... train: ', data.train_metadata.shape[0])
            print('   ... test: ', data.test_metadata.shape[0])
            print('   ... validation: ', data.validation_metadata.shape[0])
            print('   ... pool: ', data.pool_metadata.shape[0], '\n')

            # classify
            if clf_bootstrap:
                data.classify_bootstrap(method=classifier, screen=screen, **kwargs)
            else:
                data.classify(method=classifier, screen=screen, **kwargs)

            # calculate metrics
            data.evaluate_classification(screen=screen)
            
            if data.queryable_ids.shape[0] > 0:

                # get index of object to be queried
                if budgets:
                    indx = data.make_query_budget(budgets=budgets, strategy=strategy,
                                                  screen=False, num_batches=num_batches,
                                                  save_batches=save_batches, 
                                                  batch_outfile=batch_outfile, loop=night)
                else:
                    indx = data.make_query(strategy=strategy, batch=batch,
                                           queryable=queryable,
                                           query_thre=query_thre, screen=screen)

                if screen:
                    print('\n queried obj index: ', indx)
                    print('Prob [nIa, Ia]: ', data.classprob[indx[0]])
                    print('size of pool: ', data.pool_metadata.shape[0], '\n')

                # update training and test samples
                data.update_samples(indx, queryable=queryable,
                                    epoch=night)
                
                # save metrics for current state
                data.save_metrics(loop=night - days[0], output_metrics_file=output_metrics_file,
                                  batch=len(indx), epoch=night)

                if screen:
                    print('\n After update_samples:')
                    print('   ... train: ', data.train_metadata.shape[0])
                    print('   ... test: ', data.test_metadata.shape[0])
                    print('   ... validation: ', data.validation_metadata.shape[0])
                    print('   ... pool: ', data.pool_metadata.shape[0], '\n')

                # save query sample to file
                if save_full_query:
                    query_fname = output_queried_file[:-4] + '_' + str(night - days[0]) + '.dat' 
                else:
                    query_fname = output_queried_file
                
                data.save_queried_sample(query_fname, loop=night - days[0],
                                         full_sample=save_full_query, epoch=night)

            # load features for next day
            path_to_features2 = path_to_features_dir + fname_pattern[0] + \
                                str(night + 1) + fname_pattern[1]

            if sep_files:
                data_tomorrow = DataBase()
                data_tomorrow.load_features(path_to_file=path_to_features2,
                                            screen=screen, method='Bazin',
                                            survey=survey, sample='pool')
                data_tomorrow.build_samples(initial_training='original',
                                            screen=screen, queryable=queryable,
                                            save_samples=save_samples,
                                            sep_files=sep_files, survey=survey)

            else:
                data_tomorrow = load_dataset(fname=path_to_features2,
                              survey=survey,
                              screen=screen,
                              initial_training=0,
                              ia_frac=ia_frac, queryable=queryable)

            for obj in data.train_metadata[id_name].values:
                if obj in data_tomorrow.pool_metadata[id_name].values:

                    indx_tomorrow = list(data_tomorrow.pool_metadata[id_name].values).index(obj)

                    if obj not in ini_train_ids:
                        # remove old features from training
                        indx_today = list(data.train_metadata[id_name].values).index(obj)

                        flag1 = data.train_metadata[id_name].values == obj
                        data.train_metadata = data.train_metadata.drop(data.train_metadata.index[indx_today])
                        data.train_labels = np.delete(data.train_labels, indx_today, axis=0)
                        data.train_features = np.delete(data.train_features, indx_today, axis=0)
                        
                        if data.queryable_ids.shape[0] > 0:
                            # get number of queried objects
                            n = np.array(data.queried_sample).shape[0] * np.array(data.queried_sample).shape[1]
                            
                            # build query data frame
                            full_header = ['epoch'] + data.metadata_names + data.features_names
                            queried_sample = pd.DataFrame(data.queried_sample,
                                                          columns=full_header)
                        
                            # get object index in the queried sample
                            indx_queried = list(queried_sample[id_name].values).index(obj)
                        
                            # get flag to isolate object in question
                            flag2 = queried_sample[id_name].values == obj
                        
                            # get object epoch in the queried sample
                            obj_epoch = queried_sample['epoch'].values[flag2]
                        
                            # remove old features from queried
                            queried_sample = queried_sample.drop(queried_sample.index[indx_queried])
                        
                        # update new features of the training with new obs
                        flag = data_tomorrow.pool_metadata[id_name].values == obj
                        
                        if data.queryable_ids.shape[0] > 0:
                            
                            # update new features in the queried sample
                            l1 = [obj_epoch[0]] + list(data_tomorrow.pool_metadata[flag].values[0]) + \
                                 list(data_tomorrow.pool_features[flag][0])
                            new_query = pd.DataFrame([l1], columns=full_header)
                            queried_sample = pd.concat([queried_sample, new_query], axis=0,
                                                        ignore_index=True)
                        
                            # update queried sample
                            data.queried_sample = list(queried_sample.values)

                        data.train_metadata = pd.concat([data.train_metadata,
                                                         data_tomorrow.pool_metadata[flag]],
                                                         axis=0, ignore_index=True)
                        data.train_features = np.append(data.train_features,
                                                        data_tomorrow.pool_features[flag], axis=0)
                        data.train_labels = np.append(data.train_labels,
                                                      data_tomorrow.pool_labels[flag], axis=0)
                        
                    # remove obj from pool sample
                    data_tomorrow.pool_metadata = data_tomorrow.pool_metadata.drop(\
                                             data_tomorrow.pool_metadata.index[indx_tomorrow])
                    data_tomorrow.pool_labels = np.delete(data_tomorrow.pool_labels, indx_tomorrow, axis=0)
                    data_tomorrow.pool_features = np.delete(data_tomorrow.pool_features, indx_tomorrow, axis=0)

                # remove object from other samples
                if len(data_tomorrow.validation_metadata) > 0 and  obj in data_tomorrow.validation_metadata[id_name].values:
                    indx_val =  list(data_tomorrow.validation_metadata[id_name].values).index(obj)

                    data_tomorrow.validation_metadata = data_tomorrow.validation_metadata.drop(\
                                             data_tomorrow.validation_metadata.index[indx_val])
                    data_tomorrow.validation_labels = np.delete(data_tomorrow.validation_labels,
                                                                indx_val, axis=0)
                    data_tomorrow.validation_features = np.delete(data_tomorrow.validation_features,
                                                                  indx_val, axis=0)

                if len(data_tomorrow.test_metadata) > 0 and obj in data_tomorrow.test_metadata[id_name].values:
                    indx_test = list(data_tomorrow.test_metadata[id_name].values).index(obj)

                    data_tomorrow.test_metadata = data_tomorrow.test_metadata.drop(\
                                             data_tomorrow.test_metadata.index[indx_test])
                    data_tomorrow.test_labels = np.delete(data_tomorrow.test_labels,
                                                                indx_test, axis=0)
                    data_tomorrow.test_features = np.delete(data_tomorrow.test_features,
                                                                  indx_test, axis=0)


            # use new data
            data.pool_metadata = data_tomorrow.pool_metadata
            data.pool_features = data_tomorrow.pool_features
            data.pool_labels = data_tomorrow.pool_labels

            if not sep_files:
                data.test_metadata = data_tomorrow.test_metadata
                data.test_features = data_tomorrow.test_features
                data.test_labels = data_tomorrow.test_labels

                data.validation_metadata = data_tomorrow.validation_metadata
                data.validation_features = data_tomorrow.validation_features
                data.validation_labels = data_tomorrow.validation_labels

            if strategy == 'canonical':
                data.queryable_ids = canonical.queryable_ids

            if  queryable:
                queryable_flag = data.pool_metadata['queryable'].values
                data.queryable_ids = data.pool_metadata['id'].values[queryable_flag]
            else:
                data.queryable_ids = data.pool_metadata['id'].values

        if screen:
            print('\n After reading tomorrow data:')
            print('Training set size: ', data.train_metadata.shape[0])
            print('Test set size: ', data.test_metadata.shape[0])
            print('Validation set size: ', data.validation_metadata.shape[0])
            print('Pool set size: ', data.pool_metadata.shape[0])
            print('    From which queryable: ', len(data.queryable_ids))
            print('**************************** \n')

        # check if there are repeated ids
        for name in data.train_metadata['id'].values:
            if name in data.pool_metadata['id'].values:
                raise ValueError('End of time_domain_loop: ' + \
                                 'Object ', name, ' found in pool and training sample!')

        # check if all queried samples are in the training sample
        for i in range(len(data.queried_sample)):
            if data.queried_sample[i][1] not in data.train_metadata['id'].values:
                raise ValueError('End of time_domain_loop : Object '+ \
                                 str(data.queried_sample[i][1]) + \
                                 ' was queried but is missing from training!')

        # check if validation sample continues the same
        if sep_files:
            for name in data.validation_metadata[id_name].values:
                if name not in validation_ids:
                    raise ValueError('There was a change in the validation sample!')


def main():
    return None


if __name__ == '__main__':
    main()
