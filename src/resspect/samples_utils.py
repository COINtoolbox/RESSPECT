# Copyright 2023 resspect software
# Author: Emille Ishida
#
# created on 24 July 2023
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

__all__ = ['sep_samples', 'read_features_fullLC_samples']

import os
import pandas as pd
import numpy as np
import glob

def sep_samples(all_ids: np.array, n_test_val: int,
                n_train: int):
    """
    Separate train test and validation samples.

    Parameters
    ----------
    all_ids: np.array
        Array with all available object ids.
    n_test_val: int
        Number of objects to be added to test and validation.
        All the remaining ones will be assigned to 
    n_train: int
        Number of objects in training sample. This should be
        enough to allow testing of mutiple initial conditions.
        Set it to at least 10x the size of the initial sample
        within the learning loop.

    Returns
    -------
    dict
        keys are the sample names, values are the ids of
        objects in each sample.
    """
    if n_train + 2 * n_test_val > len(all_ids):
        raise ValueError(
            f"Unable to draw samples of sizes {n_train}, {n_test_val}, and {n_test_val} "
            f"from only {len(all_ids)} indices."
        )

    samples = {}
    
    # separate ids for training
    samples['train'] = np.random.choice(all_ids, size=n_train, replace=False)
    train_flag = np.isin(all_ids, samples['train'])

    #separate ids for test and validation
    samples['test'] = np.random.choice(all_ids[~train_flag], size=n_test_val, 
                                       replace=False)
    test_flag = np.isin(all_ids, samples['test'])
    test_train_flag = np.logical_or(train_flag, test_flag)

    samples['val'] = np.random.choice(all_ids[~test_train_flag], size=n_test_val,
                                      replace=False)
    val_flag = np.isin(all_ids, samples['val'])
    val_test_train_flag = np.logical_or(test_train_flag, val_flag)

    samples['query'] = all_ids[~val_test_train_flag]

    return samples   


def read_features_fullLC_samples(sample_ids: np.array, 
                              output_fname: str, path_to_features: str,
                                id_key='id'):
    """
    Create separate files for full light curve samples.

    Parameters
    ----------
    sample_ids: np.array
        Array of ids to be added to the sample.
    output_fname: str
        Filename where the sample will be saved.
        If 'path_to_features' is a directory, this should
        be pattern without extension.
    path_to_features: str
        Full path for where the features are stored.
        It can be a directory or  a file.
        All files should be csv.
    id_key: str (optional)
        String identifying the object id column. 
        Default is 'id'.

    Returns
    -------
    None
        Save samples to file.
    """

    # read features
    if os.path.isfile(path_to_features):
        data_temp = pd.read_csv(path_to_features, index_col=False)
        flag = np.isin(data_temp[id_key].values, sample_ids)
        data = data_temp[flag]
        data.to_csv(output_fname, index=False)

    elif os.path.isdir(path_to_features):        
        flist = glob.glob('*.csv')
        
        for i in range(len(flist)):
            data_temp = pd.read_csv(path_to_features + flist[i])
            flag = np.isin(data_temp[id_key].values, sample_ids)
            data = data_temp[flag]
            data.to_csv(output_fname + str(i + 1) + '.csv', index_col=False)

    return None


def main():
    return None

if __name__ == '__main__':
    main()
