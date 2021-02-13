# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
#
# created on 30 December 2020
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

import numpy as np
import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors

__all__ = ['CanonicalPLAsTiCC', 'build_plasticc_canonical']


class CanonicalPLAsTiCC(object):
    """Canonical sample object for PLAsTiCC data.
    
    Attributes
    ----------
    canonical_ids: list
        List of ids for objects in the canonical sample.
    canonical_metadata: pd.DataFrame()
        Metadata for final canonical sample.
    canonical_sample: list
        Complete data matrix for the canonical sample.
    ddf: bool
        If True, deal only with DDF objects.
    metadata_names: list
        List of keywords on which the nearest neighbors will
        be calculated. This corresponds to true redshift.
    metadata_train: pd.DataFrame
        Metadata for all objects in the PLAsTiCC zenodo 
        training set. If ddf == True this corresponds only 
        to DDF objects.
    metadata_test: pd.DataFrame
        Metadata for all objects in the PLAsTiCC zenodo test 
        set. If ddf == True this corresponds only to DDF objects.
    obj_code: dict
        Map between PLAsTiCC class code and astrophysical category.
    test_subsamples: dict
        Identify sub-samples per type within test sample.
        Keywords are the same as 'obj_code' attribute.
    train_subsamples: dict
        Identify sub-samples per type within train sample. 
        Keywords are the same as 'obj_code' attribute.
    
    
    Methods
    -------   
    read_metadata(fname: str, sample: str)
        Reads metadata from PLAsTiCC zenodo files.
    find_subsamples()
        Identify subsamples within train and test samples.
    """

    def __init__(self):
        self.canonical_ids = []
        self.canonical_sample = []
        self.canonical_metadata = pd.DataFrame()
        self.ddf = True
        self.obj_code = {15: 'TDE',     # use only extragalactic objs
                         42: 'II',
                         52: 'Iax',
                         62: 'Ibc',
                         64: 'KN',
                         67: '91bg',
                         88: 'AGN',
                         90: 'Ia',
                         95: 'SLSN'}
        self.metadata_names = ['object_id', 'true_z']
        self.metadata_train = pd.DataFrame()
        self.metadata_test = pd.DataFrame()
        self.test_subsamples = {}
        self.train_subsamples = {}
        
    def read_metadata(self, fname: str, sample: str):
        """Reads metadata built with "build_plasticc_metadata" function.
        
        Populates the attributes 'metadata_train' or 'metadata_test'.
        
        Parameters
        ----------
        fname: str
            Complete path to metadata file.
        sample: str
            Original PLAsTiCC sample, options are: 'train' or
            'test'.        
        """
        
        if sample not in ['train', 'test']:
            raise ValueError('Only accepted samples are: "train" and "test".')
        
        # read data
        data = pd.read_csv(fname)
        
        type_flag = np.array([item in self.obj_code.keys() 
                              for item in data['code_SNANA'].values])
            
        if sample == 'train':
            self.metadata_train = data[final_flag]
        else: 
            self.metadata_test = data[final_flag]
            
    def find_subsamples(self):
        """Subsamples per object type according to 'true_type' keyword.
        
        Populates attributes 'train_subsamples' and 'test_subsamples'.
        """
        
        for num in self.obj_code.keys():
            train_flag = self.metadata_train['true_target'].values == num
            self.train_subsamples[num] = \
                self.metadata_train[train_flag][self.metadata_names]
            
            test_flag = self.metadata_test['true_target'].values == num
            self.test_subsamples[num] = \
                self.metadata_test[test_flag][self.metadata_names]
        
    def find_neighbors(self, n_neighbors=5, screen=False):
        """Identify nearest neighbors in test for each object in training.

        Populates attribute: canonical_ids.

        Parameters
        ----------
        n_neighbors: int (optional)
            Number of neighbors in test to be found for each object
            in training. Default is 5.
        screen: bool (optional)
            If true, display steps info on screen. Default is False.
        """
        
        for sntype in self.obj_code.keys():
            if screen:
                print('Scanning ', self.obj_code[sntype], ' . . . ')

            # find 5x more neighbors in case there are repetitions
            n = min(n_neighbors, self.test_subsamples[sntype].shape[0])
            nbrs = NearestNeighbors(n_neighbors=n,
                                    algorithm='auto')
            
            nbrs.fit(self.test_subsamples[sntype].values[:,1:])
            
            # store indices already used
            vault = []
            
            # match with objects in training
            for i in range(self.train_subsamples[sntype].shape[0]):
                elem = np.array(self.train_subsamples[sntype].values[i][1:]).reshape(1, -1)
                indices = nbrs.kneighbors(elem, return_distance=False)[0]
                
                # only add elements which were not added in a previous loop
                for indx in indices:
                    if indx not in vault:
                        element = self.test_subsamples[sntype].iloc[indx]['object_id']
                        self.canonical_ids.append(element)
                        vault.append(indx)
                    
            if screen:
                print('     Processed: ', len(vault))
                
    def build_canonical_sample(self, path_to_test: list):
        """Gather ids and build final canonical sample.
        
        Populates attribute 'canonical_sample' and 'canonical_metadata'.
        
        Parameters
        ----------
        path_to_test: list
            Path to all test sample Bazin files.
        """
        # store all Bazin fits
        data_all = []
        
        for fname in path_to_test:
            data_temp = pd.read_csv(fname)
            data_all.append(data_temp)
            
        data = pd.concat(data_all, ignore_index=True)
        
        # identify objs
        all_ids  = data['id'].values
            
        flag = np.array([item in self.canonical_ids for item in all_ids])
        
        self.canonical_sample = data[flag]
        
        test_ids = self.metadata_test['object_id'].values
        canonical_ids = self.canonical_sample['id'].values
        flag_meta = np.array([item in canonical_ids for item in test_ids])
        self.canonical_metadata = self.metadata_test[flag_meta]

            
def build_plasticc_canonical(n_neighbors: int, path_to_metadata: dict,
                             path_to_features: list,
                             output_canonical_file: str, 
                             output_meta_file: str,
                             screen=False, plot=True, 
                             plot_fname='compare_train_canonical.png'):
    """Build canonical sample for SNPCC data.

    Parameters
    ----------
    n_neighbors: int
        Number of neighbors to identify in test for each obj in train.
    output_canonical_file: str
        Complete path to output canonical sample file.
    output_meta_file: str
        Complete path to output metadata file for canonical sample.
    path_to_metadata: dict
        Complete path to metadata files.
        Keywords must be ['train', 'test'].
    path_to_features: list
        Path to all features files for test sample.
    plot: bool (optional)
        If True, plot comparison for redshift and vspec.
        Default is True.
    plot_fname: str (optional)
        Complete path for saving plot. Default is 
        'compare_train_canonical.png'.
    features_method: str (optional)
        Method for feature extraction. Only 'Bazin' is implemented.
    screen: bool (optional)
            If true, display steps info on screen. Default is False.
    """
    
    sample = CanonicalPLAsTiCC()
    
    for name in path_to_metadata.keys():
        sample.read_metadata(fname=path_to_metadata[name], sample=name)
        
    sample.find_subsamples()
    sample.find_neighbors(n_neighbors=n_neighbors, screen=False)
    sample.build_canonical_sample(path_to_test=path_to_features)
    
    # save result to file
    sample.canonical_sample.to_csv(output_canonical_file, index=False)
    sample.canonical_metadata.to_csv(output_meta_file, index=False)
    
    if plot:
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        sns.distplot(sample.canonical_metadata['true_z'], label='canonical')
        sns.distplot(sample.metadata_train['true_z'], label='train')
        plt.legend()

        plt.subplot(1,2,2)
        sns.distplot(sample.canonical_metadata['true_vpec'], label='canonical')
        sns.distplot(sample.metadata_train['true_vpec'], label='train')
        plt.savefig(plot_fname)
    
    
def main():
    return None


if __name__ == '__main__':
    main()