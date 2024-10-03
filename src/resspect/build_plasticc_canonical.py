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
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

from resspect.build_plasticc_metadata import get_SNR_headers 
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
    filters: str
        List of LSST filters.
    galactic_codes: list
        Codes identifying galactic models.
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
    model_set: list
        List of models to be considered.
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
    build_canonical_sample(path_to_test: list)
        Gather ids and build final canonical sample.
    find_neighbors(n_neighbors: int, screen: bool)
        Identify nearest neighbors in test for each object in training.
    find_subsamples()
        Identify subsamples within train and test samples.
    read_metadata(fname: str, sample: str)
        Reads metadata from PLAsTiCC zenodo files.
    """

    def __init__(self):
        self.canonical_ids = []
        self.canonical_sample = []
        self.canonical_metadata = pd.DataFrame()
        self.ddf = True
        self.filters = ['u', 'g', 'r', 'i', 'z', 'Y']
        self.galactic_codes = [92, 65,16, 53, 6]       
        self.metadata_names = get_SNR_headers()
        self.metadata_train = pd.DataFrame()
        self.metadata_test = pd.DataFrame()
        self.model_set = []
        self.obj_code = {15: 'TDE',  
                         42: 'II',
                         52: 'Iax',
                         62: 'Ibc',
                         64: 'KN',
                         67: '91bg',
                         88: 'AGN',
                         90: 'Ia',
                         95: 'SLSN',
                         92: 'RRL',
                         65:'M-dwarf',
                         16:'EB',
                         53:'Mira',
                         6: 'm-lens'} 
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
                              for item in data['code_zenodo'].values])
            
        if sample == 'train':
            self.metadata_train = data[type_flag]
        else: 
            self.metadata_test = data[type_flag]
            
    def find_subsamples(self, models='extragal'):
        """Subsamples per object type according to 'true_type' keyword.
        
        Populates attributes 'train_subsamples' and 'test_subsamples'.
        
        Parameters
        ----------
        models: str (optional)
           Category of models to be consider. Options are 'galactic',
           'extragal' or 'all'. Default is 'extragal'.
        """
        
        if models == 'all':
            self.model_set = self.obj_code.keys()
            
        elif models == 'extragal':
            self.model_set = [item for item in list(self.obj_code.keys())
                              if item not in self.galactic_codes]
        elif models == 'galactic':
            self.model_set = self.galactic_codes
        else:
            raise ValueError('Invalid models choice. Options are ' +\
                            '"all", "extragal" and "galactic".')
            
        for num in self.model_set:
            train_flag = self.metadata_train['code_zenodo'].values == num
            self.train_subsamples[num] = \
                self.metadata_train[train_flag][self.metadata_names]

            test_flag = self.metadata_test['code_zenodo'].values == num
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
        
        for sntype in self.model_set:
            if screen:
                print('Scanning ', self.obj_code[sntype], ' . . . ')

            # find 5x more neighbors in case there are repetitions
            n = min(n_neighbors, self.test_subsamples[sntype].shape[0])
            nbrs = NearestNeighbors(n_neighbors=n,
                                    algorithm='auto')
            
            nbrs.fit(self.test_subsamples[sntype].values[:,5:12])
            
            # store indices already used
            vault = []
            
            # match with objects in training
            for i in range(self.train_subsamples[sntype].shape[0]):
                subsamp = self.train_subsamples[sntype].values[i][5:12]
                elem = np.array(subsamp.reshape(1, -1))
                indices = nbrs.kneighbors(elem, return_distance=False)[0]
                
                # only add elements which were not added in a previous loop
                for indx in indices:
                    if indx not in vault:
                        subtype = self.test_subsamples[sntype]
                        element = subtype.iloc[indx]['SNID']
                        self.canonical_ids.append(element)
                        vault.append(indx)
                    
            if screen:
                print('     Processed: ', len(vault))
                
    def build_canonical_sample(self, path_to_test: str):
        """Gather ids and build final canonical sample.
        
        Populates attribute 'canonical_sample' and 'canonical_metadata'.
        
        Parameters
        ----------
        path_to_test: list
            Path to test sample features files.
        """
        
        # store all Bazin fits
        data = pd.read_csv(path_to_test)
        
        # identify objs
        all_ids  = data['id'].values
            
        flag = np.array([item in self.canonical_ids for item in all_ids])
        
        self.canonical_sample = data[flag]
        
        test_ids = self.metadata_test['SNID'].values
        canonical_ids = self.canonical_sample['id'].values
        flag_meta = np.array([item in canonical_ids for item in test_ids])
        self.canonical_metadata = self.metadata_test[flag_meta]
        
    def clean_samples(self, input_features_files: dict, output_features_files: dict):
        """Remove repeated IDS from validation and test samples.
        
        Parameters
        ----------
        input_features_files: dict
            Dictionary with paths to features files for val and test samples.
            Keywords must be ['validation', 'test'].
            
        output_features_files: dict
            Dictionary with paths to output file names. 
            Keywords must be ['validation', 'test'].
        
        Return
        ------
            Write test and validation sample to file, eliminating objects
            present in the canonical sample.
        """
        
        # read data
        test_features = pd.read_csv(input_features_files['test'])
        validation_features = pd.read_csv(input_features_files['validation'])
    
        # remove repeated ids from other 
        canonical_ids = self.canonical_sample['id'].values
        test_ids = test_features['id'].values
        validation_ids = validation_features['id'].values

        flag_test = np.array([item not in canonical_ids for item in test_ids])
        flag_validation = np.array([item not in canonical_ids for item in validation_ids])

        new_test = test_features[flag_test]
        new_validation = validation_features[flag_validation]

        new_test.to_csv(output_features_files['test'], index=False)
        new_validation.to_csv(output_features_files['validation'], index=False)

            
def build_plasticc_canonical(n_neighbors: int, path_to_metadata: dict,
                             output_meta_file: str, input_features_files:dict,
                             output_features_files: dict,
                              plot_fname: str,
                             screen=False, plot=True, models='extragal'):
    """Build canonical sample for SNPCC data.

    Parameters
    ----------
    input_features_files: dict
        Dictionary with paths to features files for val and test samples.
        Keywords must be ['validation', 'test'].        
    n_neighbors: int
        Number of neighbors to identify in test for each obj in train.
    output_features_files: dict
        Output Bazin file name for canonical sample.
        Keywords must be ['pool', 'test', 'validation'].
    output_meta_file: str
        Complete path to output metadata file for canonical sample.
    path_to_metadata: dict
        Complete path to metadata files.
        Keywords must be ['train', 'test'].
    plot_fname: str 
        Complete path for saving plot. Default is 
    models: str (optional)
        Class of models to consider. Options are "all", "galactic"
        and "extragal". Default is "extragal".
    plot: bool (optional)
        If True, plot comparison for redshift and vspec.
        Default is True.
    features_method: str (optional)
        Method for feature extraction. Only 'Bazin' is implemented.
    screen: bool (optional)
            If true, display steps info on screen. Default is False.
    """
    
    sample = CanonicalPLAsTiCC()
    
    for name in path_to_metadata.keys():
        sample.read_metadata(fname=path_to_metadata[name], sample=name)
        
    sample.find_subsamples(models=models)
    sample.find_neighbors(n_neighbors=n_neighbors, screen=screen)
    sample.build_canonical_sample(path_to_test=input_features_files['test'])
    
    # save result to file
    sample.canonical_sample.to_csv(output_features_files['pool'], index=False)
    sample.canonical_metadata.to_csv(output_meta_file, index=False)
    
    # remove repeated ids from test and validation samples
    sample.clean_samples(input_features_files=input_features_files, 
                         output_features_files=output_features_files)    
    
    if screen:
        print('Size of original training: ', sample.metadata_train.shape[0])
        print('Size of canonical: ', sample.canonical_metadata.shape[0])
    
    if plot:
        ax = {}
        plt.figure(figsize=(20,10))

        ax[0] = plt.subplot(2,4,1)
        sns.distplot(sample.canonical_metadata['redshift'], label='canonical')
        sns.distplot(sample.metadata_train['redshift'], label='train')
        plt.legend()

        for i in range(6):
            ax[i + 1] = plt.subplot(2,4, i + 2)
            sns.distplot(sample.canonical_metadata['SIM_PEAKMAG_' + sample.filters[i]],
                         label='canonical')
            sns.distplot(sample.metadata_train['SIM_PEAKMAG_' + sample.filters[i]],
                         label='train')
        
        plt.savefig(plot_fname)
    
    return sample
    
def main():
    return None


if __name__ == '__main__':
    main()