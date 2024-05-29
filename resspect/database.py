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

import io
import os
import pandas as pd
import tarfile

from resspect.classifiers import *
from resspect.feature_extractors.bazin import BazinFeatureExtractor
from resspect.feature_extractors.bump import BumpFeatureExtractor
from resspect.feature_extractors.malanchev import MalanchevFeatureExtractor
from resspect.query_strategies import *
from resspect.query_budget_strategies import *
from resspect.metrics import get_snpcc_metric


__all__ = ['DataBase']


FEATURE_EXTRACTOR_MAPPING = {
    "bazin": BazinFeatureExtractor,
    "bump": BumpFeatureExtractor,
    "malanchev": MalanchevFeatureExtractor
}


class DataBase:
    """DataBase object, upon which the active learning loop is performed.

    Attributes
    ----------
    alt_label: bool
        Flag indicating training with less probable label.
    classifier: sklearn.classifier
        Classifier object.
    classprob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    data: pd.DataFrame
        Complete information read from features files.
    features: pd.DataFrame
        Feature matrix to be used in classification (no metadata).
    features_names: list
        Header for attribute `features`.
    metadata: pd.DataFrame
        Features matrix which will not be used in classification.
    metadata_names: list
        Header for metadata.
    metrics_list_names: list
        Values for metric elements.
    output_photo_Ia: pd.DataFrame
        Returns metadata for photometrically classified Ia.
    photo_Ia_metadata: pd.DataFrame
        Metadata for photometrically classified object ids.
    plasticc_mjd_lim: list
        [min, max] mjds for plasticc data
    predicted_class: np.array
        Predicted classes - results from ML classifier.
    queried_sample: list
        Complete information of queried objects.
    queryable_ids: np.array
        Flag for objects available to be queried.
    SNANA_types: dict
        Map between PLAsTiCC zenodo and SNANA types.
    telescope_names: list
        Name of telescopes for which costs are given.
    test_features: np.array
        Features matrix for the test sample.
    test_metadata: pd.DataFrame
        Metadata for the test sample
    test_labels: np.array
        True classification for the test sample.
    train_features: np.array
        Features matrix for the train sample.
    train_metadata: pd.DataFrame
        Metadata for the training sample.
    train_labels: np.array
        Classes for the training sample.
    validation_class: np.array
        Estimated classes for validation sample.
    validation_features: np.array
        Features matrix for the validation sample.
    validation_labels: np.array
        Classes for the validation sample.
    validation_metadata: pd.DataFrame
        Metadata for the validation sample.
    validation_prob: np.array
        Estimated probabilities for validation sample.

    Methods
    -------
    build_orig_samples()
        Construct train and test samples as given in the original data set.
    build_random_training(initial_training: int)
        Construct initial random training and corresponding test sample.
    build_previous_runs(path_to_train: str, path_to_queried: str)
        Build train, test and queryable samples from previous runs.
    build_samples(initial_training: str or int, nclass: int)
        Separate train and test samples.
    classify(method: str)
        Apply a machine learning classifier.
    classify_bootstrap(method: str)
        Apply a machine learning classifier bootstrapping the classifier
    evaluate_classification(metric_label: str)
        Evaluate results from classification.
    identify_keywords()
        Break degenerescency between keywords with equal meaning.
    load_extracted_features(path_to_features_file: str)
        Load features from file
    load_photometry_features(path_to_photometry_file:str)
        Load photometric light curves from file
    load_plasticc_mjd(path_to_data_dir: str)
        Get min and max mjds for PLAsTiCC data
    load_features_from_file(path_to_file: str, method: str)
        Load features according to the chosen feature extraction method.
    load_features(path_to_file: str, method: str)
        Load features or photometry according to the chosen feature extraction method.
    make_query(strategy: str, batch: int) -> list
        Identify new object to be added to the training sample.
    output_photo_Ia(threshold: float)
        Returns the metadata for  photometrically classified SN Ia.
    save_metrics(loop: int, output_metrics_file: str)
        Save current metrics to file.
    save_queried_sample(queried_sample_file: str, loop: int, full_sample: str)
        Save queried sample to file.
    update_samples(query_indx: list)
        Add the queried obj(s) to training and remove them from test.

    Examples
    --------
    >>> from resspect import DataBase

    Define the necessary paths

    >>> path_to_bazin_file = 'results/Bazin.dat'
    >>> metrics_file = 'results/metrics.dat'
    >>> query_file = 'results/query_file.dat'

    Initiate the DataBase object and load the data.
    >>> data = DataBase()
    >>> data.load_features(path_to_bazin_file, method='bazin')

    Separate training and test samples and classify

    >>> data.build_samples(initial_training='original', nclass=2)
    >>> data.classify(method='RandomForest')
    >>> print(data.classprob)          # check predicted probabilities
    [[0.461 0.539]
    [0.346print(data.metrics_list_names)           # check metric header
    ['acc', 'eff', 'pur', 'fom']

    >>> print(data.metrics_list_values)          # check metric values
    [0.5975434599574068, 0.9024767801857585,
    0.34684684684684686, 0.13572404702012383] 0.654]
    ...
    [0.398 0.602]
    [0.396 0.604]]

    Calculate classification metrics

    >>> data.evaluate_classification(metric_label='snpcc')
    >>>

    Make query, choose object and update samples

    >>> indx = data.make_query(strategy='UncSampling', batch=1)
    >>> data.update_samples(indx)

    Save results to file

    >>> data.save_metrics(loop=0, output_metrics_file=metrics_file)
    >>> data.save_queried_sample(loop=0, queried_sample_file=query_file,
    >>>                          full_sample=False)
    """

    def __init__(self):
        self.alt_label = False
        self.classifier = None
        self.classprob = np.array([])
        self.data = pd.DataFrame()
        self.ensemble_probs = None
        self.features = pd.DataFrame([])
        self.features_names = []
        self.metadata = pd.DataFrame()
        self.metadata_names = []
        self.metrics_list_names = []
        self.metrics_list_values = []
        self.pool_features = np.array([])
        self.pool_metadata = pd.DataFrame()
        self.pool_labels = np.array([])
        self.predicted_class = np.array([])
        self.queried_sample = []
        self.queryable_ids = np.array([])
        self.SNANA_types = {90:11, 62:{1:3, 2:13}, 42:{1:2, 2:12, 3:14},
                            67:41, 52:43, 64:51, 95:60, 994:61, 992:62,
                            993:63, 15:64, 88:70, 92:80, 65:81, 16:83,
                            53:84, 991:90, 6:{1:91, 2:93}}
        self.telescope_names = ['4m', '8m']
        self.test_features = np.array([])
        self.test_metadata = pd.DataFrame()
        self.test_labels = np.array([])
        self.train_features = np.array([])
        self.train_metadata = pd.DataFrame()
        self.train_labels = np.array([])
        self.validation_class = np.array([])
        self.validation_features = np.array([])
        self.validation_labels = np.array([])
        self.validation_metadata = pd.DataFrame()
        self.validation_prob = np.array([])

    def load_features_from_file(self, path_to_features_file: str, screen=False,
                      survey='DES', sample=None, feature_extractor: str='bazin'):

        """Load features from file.

        Populate properties: features, feature_names, metadata
        and metadata_names.

        Parameters
        ----------
        path_to_features_file: str
            Complete path to features file.
        screen: bool (optional)
            If True, print on screen number of light curves processed.
            Default is False.
        survey: str (optional)
            Name of survey. Used to infer the filter set.
            Options are DES or LSST. Default is DES.
        sample: str (optional)
            If None, sample is given by a column within the given file.
            else, read independent files for 'train' and 'test'.
            Default is None.
        feature_extractor: str (optional)
            Function used for feature extraction. Options are "bazin", 
            "bump", or "malanchev". Default is "bump".
        """

        # read matrix with features
        if '.tar.gz' in path_to_features_file:
            tar = tarfile.open(path_to_features_file, 'r:gz')
            fname = tar.getmembers()[0]
            content = tar.extractfile(fname).read()
            data = pd.read_csv(io.BytesIO(content))
            tar.close()

        else:
            data = pd.read_csv(path_to_features_file, index_col=False)

        # check if queryable is there
        if 'queryable' not in data.keys():
            data['queryable'] = [True for i in range(data.shape[0])]

        # list of features to use
        if survey == 'DES':
            if feature_extractor == "bazin":
                self.features_names = ['gA', 'gB', 'gt0', 'gtfall', 'gtrise', 'rA',
                                       'rB', 'rt0', 'rtfall', 'rtrise', 'iA', 'iB',
                                       'it0', 'itfall', 'itrise', 'zA', 'zB', 'zt0',
                                       'ztfall', 'ztrise']
            elif feature_extractor == 'bump':
                self.features_names = ['gp1', 'gp2', 'gp3', 'gmax_flux', 
                                       'rp1', 'rp2', 'rp3', 'rmax_flux', 
                                       'ip1', 'ip2', 'ip3', 'imax_flux', 
                                       'zp1', 'zp2', 'zp3', 'zmax_flux']
            elif feature_extractor == 'malanchev':
                self.features_names = ['ganderson_darling_normal','ginter_percentile_range_5',
                                       'gchi2','gstetson_K','gweighted_mean','gduration', 
                                       'gotsu_mean_diff','gotsu_std_lower', 'gotsu_std_upper',
                                       'gotsu_lower_to_all_ratio', 'glinear_fit_slope', 
                                       'glinear_fit_slope_sigma','glinear_fit_reduced_chi2',
                                       'randerson_darling_normal', 'rinter_percentile_range_5',
                                       'rchi2', 'rstetson_K', 'rweighted_mean','rduration', 
                                       'rotsu_mean_diff','rotsu_std_lower', 'rotsu_std_upper',
                                       'rotsu_lower_to_all_ratio', 'rlinear_fit_slope', 
                                       'rlinear_fit_slope_sigma','rlinear_fit_reduced_chi2',
                                       'ianderson_darling_normal','iinter_percentile_range_5',
                                       'ichi2', 'istetson_K', 'iweighted_mean','iduration', 
                                       'iotsu_mean_diff','iotsu_std_lower', 'iotsu_std_upper',
                                       'iotsu_lower_to_all_ratio', 'ilinear_fit_slope', 
                                       'ilinear_fit_slope_sigma','ilinear_fit_reduced_chi2',
                                       'zanderson_darling_normal','zinter_percentile_range_5',
                                       'zchi2', 'zstetson_K', 'zweighted_mean','zduration', 
                                       'zotsu_mean_diff','zotsu_std_lower', 'zotsu_std_upper',
                                       'zotsu_lower_to_all_ratio', 'zlinear_fit_slope', 
                                       'zlinear_fit_slope_sigma','zlinear_fit_reduced_chi2']

            self.metadata_names = ['id', 'redshift', 'type', 'code',
                                   'orig_sample', 'queryable']

            if 'last_rmag' in data.keys():
                self.metadata_names.append('last_rmag')

            for name in self.telescope_names:
                if 'cost_' + name in data.keys():
                    self.metadata_names = self.metadata_names + ['cost_' + name]

        elif survey == 'LSST':
            if feature_extractor == "bazin":
                self.features_names = ['uA', 'uB', 'ut0', 'utfall', 'utrise',
                                       'gA', 'gB', 'gt0', 'gtfall', 'gtrise',
                                       'rA', 'rB', 'rt0', 'rtfall', 'rtrise',
                                       'iA', 'iB', 'it0', 'itfall', 'itrise',
                                       'zA', 'zB', 'zt0', 'ztfall', 'ztrise',
                                       'YA', 'YB', 'Yt0', 'Ytfall', 'Ytrise']
            elif feature_extractor == "malanchev":
                self.features_names = ['uanderson_darling_normal','uinter_percentile_range_5',
                                       'uchi2','ustetson_K','uweighted_mean','uduration', 
                                       'uotsu_mean_diff','uotsu_std_lower', 'uotsu_std_upper',
                                       'uotsu_lower_to_all_ratio', 'ulinear_fit_slope', 
                                       'ulinear_fit_slope_sigma','ulinear_fit_reduced_chi2',
                                       'ganderson_darling_normal','ginter_percentile_range_5',
                                       'gchi2','gstetson_K','gweighted_mean','gduration', 
                                       'gotsu_mean_diff','gotsu_std_lower', 'gotsu_std_upper',
                                       'gotsu_lower_to_all_ratio', 'glinear_fit_slope', 
                                       'glinear_fit_slope_sigma','glinear_fit_reduced_chi2',
                                       'randerson_darling_normal', 'rinter_percentile_range_5',
                                       'rchi2', 'rstetson_K', 'rweighted_mean','rduration', 
                                       'rotsu_mean_diff','rotsu_std_lower', 'rotsu_std_upper',
                                       'rotsu_lower_to_all_ratio', 'rlinear_fit_slope', 
                                       'rlinear_fit_slope_sigma','rlinear_fit_reduced_chi2',
                                       'ianderson_darling_normal','iinter_percentile_range_5',
                                       'ichi2', 'istetson_K', 'iweighted_mean','iduration', 
                                       'iotsu_mean_diff','iotsu_std_lower', 'iotsu_std_upper',
                                       'iotsu_lower_to_all_ratio', 'ilinear_fit_slope', 
                                       'ilinear_fit_slope_sigma','ilinear_fit_reduced_chi2',
                                       'zanderson_darling_normal','zinter_percentile_range_5',
                                       'zchi2', 'zstetson_K', 'zweighted_mean','zduration', 
                                       'zotsu_mean_diff','zotsu_std_lower', 'zotsu_std_upper',
                                       'zotsu_lower_to_all_ratio', 'zlinear_fit_slope', 
                                       'zlinear_fit_slope_sigma','zlinear_fit_reduced_chi2',
                                       'Yanderson_darling_normal','Yinter_percentile_range_5',
                                       'Ychi2', 'Ystetson_K', 'Yweighted_mean','Yduration', 
                                       'Yotsu_mean_diff','Yotsu_std_lower', 'Yotsu_std_upper',
                                       'Yotsu_lower_to_all_ratio', 'Ylinear_fit_slope', 
                                       'Ylinear_fit_slope_sigma','Ylinear_fit_reduced_chi2']

            if 'objid' in data.keys():
                self.metadata_names = ['objid', 'redshift', 'type', 'code',
                                       'orig_sample', 'queryable']
            elif 'id' in data.keys():
                self.metadata_names = ['id', 'redshift', 'type', 'code',
                                       'orig_sample', 'queryable']
                
            if 'last_rmag' in data.keys():
                self.metadata_names.append('last_rmag')

            for name in self.telescope_names:
                if 'cost_' + name in data.keys():
                    self.metadata_names = self.metadata_names + ['cost_' + name]

        else:
            raise ValueError('Only "DES" and "LSST" filters are ' + \
                             'implemented at this point!')

        if sample == None:
            self.features = data[self.features_names].values
            self.metadata = data[self.metadata_names]

            if screen:
                print('Loaded ', self.metadata.shape[0], ' samples!')

        elif sample == 'train':
            self.train_features = data[self.features_names].values
            self.train_metadata = data[self.metadata_names]

            if screen:
                print('Loaded ', self.train_metadata.shape[0], ' ' +  \
                      sample + ' samples!')

        elif sample == 'test':
            self.test_features = data[self.features_names].values
            self.test_metadata = data[self.metadata_names]

            if screen:
                print('Loaded ', self.test_metadata.shape[0], ' ' + \
                       sample +  ' samples!')

        elif sample == 'validation':
            self.validation_features = data[self.features_names].values
            self.validation_metadata = data[self.metadata_names]

            if screen:
                print('Loaded ', self.validation_metadata.shape[0], ' ' + \
                       sample +  ' samples!')

        elif sample == 'pool':
            self.pool_features = data[self.features_names].values
            self.pool_metadata = data[self.metadata_names]

            if screen:
                print('Loaded ', self.pool_metadata.shape[0], ' ' + \
                      sample +  ' samples!')

    def load_photometry_features(self, path_to_photometry_file: str,
                                 screen=False, sample=None):
        """Load photometry features from file.

        Gets as input file containing fitted flux in homogeneized cadences.
        Each line of the file is 1 object with concatenated fitted flux values.
        Populate properties: data, features, feature_list, header
        and header_list.

        Parameters
        ----------
        path_to_photometry_file: str
            Complete path to photometry features file.
        screen: bool (optional)
            If True, print on screen number of light curves processed.
            Default is False.
        sample: str (optional)
            If None, sample is given by a column within the given file.
            else, read independent files for 'train' and 'test'.
            Default is None.
        """

        # read matrix with full photometry
        if '.tar.gz' in path_to_photometry_file:
            tar = tarfile.open(path_to_photometry_file, 'r:gz')
            fname = tar.getmembers()[0]
            content = tar.extractfile(fname).read()
            data = pd.read_csv(io.BytesIO(content))
            tar.close()
        else:
            data = pd.read_csv(path_to_photometry_file,
                               index_col=False)
            if ' ' in data.keys()[0]:
                data = pd.read_csv(path_to_photometry_file,
                                   sep=' ', index_col=False)

        # list of features to use
        self.features_names = data.keys()[5:]

        if 'objid' in data.keys():
            id_name = 'objid'
        elif 'id' in data.keys():
            id_name = 'id'

        self.metadata_names = [id_name, 'redshift', 'type',
                               'code', 'orig_sample']

        if sample == None:
            self.features = data[self.features_names]
            self.metadata = data[self.metadata_names]

            if screen:
                print('Loaded ', self.metadata.shape[0], ' samples!')

        elif sample == 'train':
            self.train_features = data[self.features_names].values
            self.train_metadata = data[self.metadata_names]

            if screen:
                print('Loaded ', self.train_metadata.shape[0], ' samples!')

        elif sample == 'test':
            self.test_features = data[self.features_names].values
            self.test_metadata = data[self.metadata_names]

            if screen:
                print('\n Loaded ', self.test_metadata.shape[0],
                      ' samples! \n')

    def load_features(self, path_to_file: str, feature_extractor: str ='bazin',
                      screen=False, survey='DES', sample=None ):
        """Load features according to the chosen feature extraction method.

        Populates properties: data, features, feature_list, header
        and header_list.

        Parameters
        ----------
        path_to_file: str
            Complete path to features file.
        feature_extractor: str (optional)
            Feature extraction method. The current implementation only
            accepts =='bazin', 'bump', 'malanchev', or 'photometry'.
            Default is 'bazin'.
        screen: bool (optional)
            If True, print on screen number of light curves processed.
            Default is False.
        survey: str (optional)
            Survey used to obtain the data. The current implementation
            only accepts survey='DES' or 'LSST'.
            Default is 'DES'.
        sample: str (optional)
            If None, sample is given by a column within the given file.
            else, read independent files for 'train' and 'test'.
            Default is None.
        """
        if feature_extractor == "photometry":
            self.load_photometry_features(path_to_file, screen=screen,
                                          survey=survey, sample=sample)
        elif feature_extractor in FEATURE_EXTRACTOR_MAPPING:
            self.load_features_from_file(
                path_to_file, screen=screen, survey=survey,
                sample=sample, feature_extractor=feature_extractor)
        else:
            raise ValueError('Only bazin, bump, malanchev, or photometry features are implemented!'
                             '\n Feel free to add other options.')

    def load_plasticc_mjd(self, path_to_data_dir):
        """Return all MJDs from 1 file from PLAsTiCC simulations.

        Parameters
        ----------
        path_to_data_dir: str
            Complete path to PLAsTiCC data directory.
        """

        # list of PLAsTiCC photometric files
        flist = ['plasticc_test_lightcurves_' + str(x).zfill(2) + '.csv.gz'
                  for x in range(1, 12)]

        # add training file
        flist = flist + ['plasticc_train_lightcurves.csv.gz']

        # store max and min mjds
        min_mjd = []
        max_mjd = []

        for fname in flist:
        # read photometric points
            if '.tar.gz' in fname:
                tar = tarfile.open(path_to_data_dir + fname, 'r:gz')
                name = tar.getmembers()[0]
                content = tar.extractfile(name).read()
                all_photo = pd.read_csv(io.BytesIO(content))
            else:
                all_photo = pd.read_csv(path_to_data_dir + fname, index_col=False)

                if ' ' in all_photo.keys()[0]:
                    all_photo = pd.read_csv(path_to_data_dir + fname, sep=' ',
                                            index_col=False)

            # get limit mjds
            min_mjd.append(min(all_photo['mjd']))
            max_mjd.append(max(all_photo['mjd']))


        self.plasticc_mjd_lim = [min(min_mjd), max(max_mjd)]

    def identify_keywords(self):
        """Break degenerescency between keywords with equal meaning.

        Returns
        -------

        id_name: str
            String of object identification.
        """

        if 'id' in self.metadata_names:
            id_name = 'id'
        elif 'objid' in self.metadata_names:
            id_name = 'objid'

        return id_name

    def build_orig_samples(self, nclass=2, screen=False, queryable=False,
                           sep_files=False):
        """Construct train and test samples as given in the original data set.

        Populate properties: train_features, train_header, test_features,
        test_header, queryable_ids (if flag available), train_labels and
        test_labels.

        Parameters
        ----------
        nclass: int (optional)
            Number of classes to consider in the classification
            Currently only nclass == 2 is implemented.
        queryable: bool (optional)
            If True use queryable flag from file. Default is False.
        screen: bool (optional)
            If True display the dimensions of training and test samples.
        sep_files: bool (optional)
            If True, consider train and test samples separately read
            from independent files.
        """

        # object if keyword
        id_name = self.identify_keywords()

        if sep_files:
            # get samples labels in a separate object
            if self.train_metadata.shape[0] > 0:
                train_labels = self.train_metadata['type'].values == 'Ia'
                self.train_labels = train_labels.astype(int)

            if self.test_metadata.shape[0] > 0:
                test_labels = self.test_metadata['type'].values == 'Ia'
                self.test_labels = test_labels.astype(int)

            if self.validation_metadata.shape[0] > 0:
                validation_labels = self.validation_metadata['type'].values == 'Ia'
                self.validation_labels = validation_labels.astype(int)

            if self.pool_metadata.shape[0] > 0:
                pool_labels = self.pool_metadata['type'].values == 'Ia'
                self.pool_labels = pool_labels.astype(int)

            # identify asked to consider queryable flag
            if queryable and len(self.pool_metadata) > 0:
                queryable_flag = self.pool_metadata['queryable'].values
                self.queryable_ids = self.pool_metadata[queryable_flag][id_name].values

            elif len(self.pool_metadata) > 0:
                self.queryable_ids = self.pool_metadata[id_name].values

        else:
            train_flag = self.metadata['orig_sample'] == 'train'
            train_data = self.features[train_flag]
            self.train_features = train_data
            self.train_metadata = self.metadata[train_flag]

            test_flag = self.metadata['orig_sample'] == 'test'
            test_data = self.features[test_flag]
            self.test_features = test_data
            self.test_metadata = self.metadata[test_flag]

            if 'validation' in self.metadata['orig_sample'].values:
                val_flag = self.metadata['orig_sample'] == 'validation'
            else:
                val_flag = test_flag

            val_data = self.features[val_flag]
            self.validation_features = val_data
            self.validation_metadata = self.metadata[val_flag]

            if 'pool' in self.metadata['orig_sample'].values:
                pool_flag = self.metadata['orig_sample'] == 'pool'
            else:
                pool_flag = test_flag

            pool_data = self.features[pool_flag]
            self.pool_features = pool_data
            self.pool_metadata = self.metadata[pool_flag]

            if queryable:
                queryable_flag = self.pool_metadata['queryable'].values
                self.queryable_ids = self.pool_metadata[queryable_flag][id_name].values
            else:
                self.queryable_ids = self.pool_metadata[id_name].values

            if nclass == 2:
                train_ia_flag = self.train_metadata['type'].values == 'Ia'
                self.train_labels = train_ia_flag.astype(int)

                test_ia_flag = self.test_metadata['type'].values == 'Ia'
                self.test_labels = test_ia_flag.astype(int)

                val_ia_flag = self.validation_metadata['type'].values == 'Ia'
                self.validation_labels = val_ia_flag.astype(int)

                pool_ia_flag = self.pool_metadata['type'].values == 'Ia'
                self.pool_labels = pool_ia_flag.astype(int)

            else:
                raise ValueError("Only 'Ia x non-Ia' are implemented! "
                                 "\n Feel free to add other options.")

        if screen:
            print('\n')
            print('** Inside build_orig_samples: **')
            print('Training set size: ', len(self.train_metadata))
            print('Test set size: ', len(self.test_metadata))
            print('Validation set size: ', len(self.validation_metadata))
            print('Pool set size: ', len(self.pool_metadata))
            print('   From which queryable: ', len(self.queryable_ids), '\n')

        # check repeated ids between training and pool
        if len(self.train_metadata) > 0 and len(self.pool_metadata) > 0:
            for name in self.train_metadata[id_name].values:
                if name in self.pool_metadata[id_name].values:
                    raise ValueError('Object ', name, 'found in both, training ' +\
                                    'and pool samples!')

        # check if there are repeated ids within each sample
        names = ['train', 'pool', 'validation', 'test']
        pds = [self.train_metadata, self.pool_metadata,
               self.validation_metadata, self.test_metadata]

        repeated_ids_samples = []
        for i in range(4):
            if pds[i].shape[0] > 0:
                delta = len(np.unique(pds[i][id_name].values)) - pds[i].shape[0]
                if abs(delta) > 0:
                    repeated_ids_samples.append([names[i], delta])

        if len(repeated_ids_samples) > 0:
            raise ValueError('There are repeated ids within ' + \
                             str(repeated_ids_samples)  + ' sample!')


    def build_random_training(self, initial_training: int, nclass=2, screen=False,
                              Ia_frac=0.5, queryable=False, sep_files=False):
        """Construct initial random training and corresponding test sample.

        Populate properties: train_features, train_header, test_features,
        test_header, queryable_ids (if flag available), train_labels and
        test_labels.

        Parameters
        ----------
        initial_training : int
            Required number of samples at random
            Default is 10.
        nclass: int (optional)
            Number of classes to consider in the classification
            Currently only nclass == 2 is implemented.
        queryable: bool (optional)
            If True build also queryable sample for time domain analysis.
            Default is False.
        screen: bool (optional)
            If True display the dimensions of training and test samples.
        Ia_frac: float in [0,1] (optional)
            Fraction of Ia required in initial training sample.
            Default is 0.5.
        sep_files: bool (optional)
            If True, consider train and test samples separately read
            from independent files. Default is False.
        """

        # object if keyword
        id_name = self.identify_keywords()

        # identify Ia
        if sep_files:
            data_copy = self.train_metadata.copy()
        else:
            data_copy = self.metadata.copy()

        ia_flag = data_copy['type'] == 'Ia'

        # separate per class
        Ia_data = data_copy[ia_flag]
        nonIa_data = data_copy[~ia_flag]

        # get subsamples for training
        temp_train_ia = Ia_data.sample(n=int(Ia_frac * initial_training))
        temp_train_nonia = nonIa_data.sample(n=int((1-Ia_frac)*initial_training))

        # join classes
        frames_train = [temp_train_ia, temp_train_nonia]
        temp_train = pd.concat(frames_train, ignore_index=True, axis=0)
        train_flag = np.array([data_copy[id_name].values[i] in temp_train[id_name].values
                               for i in range(data_copy.shape[0])])

        self.train_metadata = data_copy[train_flag]

        if sep_files:
            self.train_features = self.train_features[train_flag]
            test_labels = self.test_metadata['type'].values == 'Ia'
            self.test_labels = test_labels.astype(int)
            validation_labels = self.validation_metadata['type'].values == 'Ia'
            self.validation_labels = validation_labels.astype(int)
            pool_labels = self.pool_metadata['type'].values == 'Ia'
            self.pool_labels = pool_labels.astype(int)

        else:
            self.train_features = self.features[train_flag]

            # get test sample
            test_flag = ~train_flag
            self.test_metadata = data_copy[test_flag]
            self.test_features = self.features[test_flag]
            self.pool_metadata = self.test_metadata
            self.pool_features = self.test_features
            self.validation_features = self.test_features
            self.validation_metadata = self.test_metadata
            test_label_flag = data_copy['type'][test_flag].values == 'Ia'
            self.test_labels = test_label_flag.astype(int)
            self.pool_labels = self.test_labels
            self.validation_labels = self.test_labels

        train_label_flag = data_copy['type'][train_flag].values == 'Ia'
        self.train_labels = train_label_flag.astype(int)

        if queryable and not sep_files:
            queryable_flag = data_copy['queryable'].values
            combined_flag = np.logical_and(~train_flag, queryable_flag)
            self.queryable_ids = data_copy[combined_flag][id_name].values
        elif not queryable and not sep_files:
            self.queryable_ids = self.test_metadata[id_name].values
        elif queryable and sep_files:
            queryable_flag = self.pool_metadata['queryable'].values == True
            self.queryable_ids = self.pool_metadata[id_name].values[queryable_flag]
        elif not queryable and sep_files:
            self.queryable_ids = self.pool_metadata[id_name].values

        if screen:
            print('\n')
            print('** Inside build_random_training: **')
            print('Training set size: ', self.train_metadata.shape[0])
            print('Test set size: ', self.test_metadata.shape[0])
            print('Validation set size: ', self.validation_metadata.shape[0])
            print('Pool set size: ', self.pool_metadata.shape[0])
            print('   From which queryable: ', self.queryable_ids.shape[0], '\n')

        # check if there are repeated ids
        for name in self.train_metadata[id_name].values:
            if name in self.pool_metadata[id_name].values:
                raise ValueError('Object ', name, ' present in both, ' + \
                                 'training and pool samples!')

        # check if there are repeated ids within each sample
        names = ['train', 'pool', 'validation', 'test']
        pds = [self.train_metadata, self.pool_metadata,
               self.validation_metadata, self.test_metadata]

        repeated_ids_samples = []
        for i in range(4):

            delta = len(np.unique(pds[i][id_name].values)) - pds[i].shape[0]
            if abs(delta) > 0:
                repeated_ids_samples.append([names[i], delta])

        if len(repeated_ids_samples) > 0:
            raise ValueError('There are repeated ids within ' + \
                             str(repeated_ids_samples)  + ' sample!')

    def build_samples(self, initial_training='original', nclass=2,
                      screen=False, Ia_frac=0.5,
                      queryable=False, save_samples=False, sep_files=False,
                      survey='DES', output_fname=None):
        """Separate train, test and validation samples.

        Populate properties: train_features, train_header, test_features,
        test_header, queryable_ids (if flag available), train_labels and
        test_labels.

        Parameters
        ----------
        initial_training : str or int
            Choice of initial training sample.
            If 'original': begin from the train sample flagged in original file
            elif 'previous': continue from a previously run loop
            elif int: choose the required number of samples at random,
            ensuring that at least half are SN Ia.
        Ia_frac: float in [0,1] (optional)
            Fraction of Ia required in initial training sample.
            Default is 0.5.
        nclass: int (optional)
            Number of classes to consider in the classification
            Currently only nclass == 2 is implemented.
        output_fname: str (optional)
            Complete path to output file where initial training will be stored.
            Only used if save_samples == True.
        queryable: bool (optional)
            If True build also queryable sample for time domain analysis.
            Default is False.
        screen: bool (optional)
            If True display the dimensions of training and test samples.
        save_samples: bool (optional)
            If True, save training and test samples to file.
            Default is False.
        survey: str (optional)
            Survey used to obtain the data. The current implementation
            only accepts survey='DES' or 'LSST'. Default is 'DES'.
        sep_files: bool (optional)
            If True, consider samples separately read
            from independent files. Default is False.
        """

        if initial_training == 'original':
            self.build_orig_samples(nclass=nclass, screen=screen,
                                    queryable=queryable, sep_files=sep_files)

        elif isinstance(initial_training, int):
            self.build_random_training(initial_training=initial_training,
                                       nclass=nclass, screen=screen,
                                       Ia_frac=Ia_frac, queryable=queryable,
                                       sep_files=sep_files)

        if screen:
            print('\n')
            print('** Inside build_samples ** : ')
            print('Training set size: ', self.train_metadata.shape[0])
            print('Test set size: ', self.test_metadata.shape[0])
            print('Validation set size: ', self.validation_metadata.shape[0])
            print('Pool set size: ', self.pool_metadata.shape[0])
            if len(self.pool_metadata) > 0:
                print('   From which queryable: ',
                      self.queryable_ids.shape[0], '\n')

        if isinstance(initial_training, int) and output_fname is not None:

            full_header = self.metadata_names + self.features_names
            wsample = open(output_fname, 'w')
            for item in full_header:
                wsample.write(item + ',')
            wsample.write('\n')

            for j in range(self.train_metadata.shape[0]):
                for name in self.metadata_names:
                    wsample.write(str(self.train_metadata[name].iloc[j]) + ',')
                for k in range(self.train_features.shape[1] - 1):
                    wsample.write(str(self.train_features[j][k]) + ',')
                wsample.write(str(self.train_features[j][-1]) + '\n')
            wsample.close()

    def classify(self, method: str, save_predictions=False, pred_dir=None,
                 loop=None, screen=False, **kwargs):
        """Apply a machine learning classifier.

        Populate properties: predicted_class and class_prob

        Parameters
        ----------
        method: str
            Chosen classifier.
            The current implementation accepts `RandomForest`,
            'GradientBoostedTrees', 'KNN', 'MLP', 'SVM' and 'NB'.
        loop: int (boolean)
            Iteration loop. Only used if save+predictions==True.
            Default is None
        screen: bool (optional)
            If True, print debug statements to screen.
            Default is False.
        save_predictions: bool (optional)
            Save predictions to file. Default is False.
        pred_dir: str (optional)
            Output directory to store class predictions.
            Only used if `save_predictions == True`. Default is None.
        kwargs: extra parameters
            Parameters required by the chosen classifier.
        """

        if screen:
            print('\n Inside classify: ')
            print('   ... train_features: ', self.train_features.shape)
            print('   ... train_labels: ', self.train_labels.shape)
            print('   ... pool_features: ', self.pool_features.shape)

        if method == 'RandomForest':
            self.predicted_class,  self.classprob, self.classifier = \
                   random_forest(self.train_features, self.train_labels,
                                 self.pool_features, **kwargs)

        elif method == 'GradientBoostedTrees':
            self.predicted_class,  self.classprob, self.classifier = \
                gradient_boosted_trees(self.train_features, self.train_labels,
                                       self.pool_features, **kwargs)
        elif method == 'KNN':
            self.predicted_class,  self.classprob, self.classifier = \
                knn(self.train_features, self.train_labels,
                               self.pool_features, **kwargs)
        elif method == 'MLP':
            self.predicted_class,  self.classprob, self.classifier = \
                mlp(self.train_features, self.train_labels,
                               self.pool_features, **kwargs)
        elif method == 'SVM':
            self.predicted_class, self.classprob, self.classifier = \
                svm(self.train_features, self.train_labels,
                               self.pool_features, **kwargs)
        elif method == 'NB':
            self.predicted_class, self.classprob, self.classifier = \
                nbg(self.train_features, self.train_labels,
                          self.pool_features, **kwargs)
        else:
            raise ValueError("The only classifiers implemented are" +
                              "'RandomForest', 'GradientBoostedTrees'," +
                              "'KNN', 'MLP' and NB'." +
                             "\n Feel free to add other options.")

        # estimate classification for validation sample
        self.validation_class = \
            self.classifier.predict(self.validation_features)
        self.validation_prob = \
            self.classifier.predict_proba(self.validation_features)

        if save_predictions:
            id_name = self.identify_keywords()

            if self.alt_label:
                out_fname = 'predict_loop_' + str(loop) + '_alt_label.csv'
            else:
                out_fname = 'predict_loop_' + str(loop) + '.csv'
            op = open(pred_dir + '/' + out_fname, 'w')
            op.write(id_name + ',' + 'prob_nIa, prob_Ia,pred_class\n')
            for i in range(self.validation_metadata.shape[0]):
                op.write(str(self.validation_metadata[id_name].iloc[i]) + ',')
                op.write(str(self.validation_prob[i][0]) + ',')
                op.write(str(self.validation_prob[i][1]) + ',')
                op.write(str(self.validation_class[i]) + '\n')
            op.close()

    def classify_bootstrap(self, method: str, save_predictions=False, pred_dir=None,
                           loop=None, n_ensembles=10, screen=False, **kwargs):
        """Apply a machine learning classifier bootstrapping the classifier.

        Populate properties: predicted_class, class_prob and ensemble_probs.

        Parameters
        ----------
        method: str
            Chosen classifier.
            The current implementation accepts `RandomForest`,
            'GradientBoostedTrees', 'KNN', 'MLP', 'SVM' and 'NB'.
        save_predictions: bool (optional)
            Save predictions to file. Default is False.
        pred_dir: str (optional)
            Output directory to store class predictions.
            Only used if `save_predictions == True`. Default is None.
        loop: int (optional)
            Corresponding loop. Default is None.
        kwargs: extra parameters
            Parameters required by the chosen classifier.
        """
        if screen:
            print('\n Inside classify_bootstrap: ')
            print('   ... train_features: ', self.train_features.shape)
            print('   ... train_labels: ', self.train_labels.shape)
            print('   ... pool_features: ', self.pool_features.shape)

        if method == 'RandomForest':
            self.predicted_class, self.classprob, self.ensemble_probs, self.classifier = \
            bootstrap_clf(random_forest, n_ensembles,
                          self.train_features, self.train_labels,
                          self.pool_features, **kwargs)

        elif method == 'GradientBoostedTrees':
            self.predicted_class, self.classprob, self.ensemble_probs, self.classifier = \
            bootstrap_clf(gradient_boosted_trees, n_ensembles,
                          self.train_features, self.train_labels,
                          self.pool_features, **kwargs)
        elif method == 'KNN':
            self.predicted_class, self.classprob, self.ensemble_probs, self.classifier = \
            bootstrap_clf(knn, n_ensembles,
                          self.train_features, self.train_labels,
                          self.pool_features, **kwargs)
        elif method == 'MLP':
            self.predicted_class, self.classprob, self.ensemble_probs, self.classifier = \
            bootstrap_clf(mlp, n_ensembles,
                          self.train_features, self.train_labels,
                          self.pool_features, **kwargs)
        elif method == 'SVM':
            self.predicted_class, self.classprob, self.ensemble_probs, self.classifier = \
            bootstrap_clf(svm, n_ensembles,
                          self.train_features, self.train_labels,
                          self.pool_features, **kwargs)
        elif method == 'NB':
            self.predicted_class, self.classprob, self.ensemble_probs, self.classifier = \
            bootstrap_clf(nbg, n_ensembles,
                          self.train_features, self.train_labels,
                          self.pool_features, **kwargs)
        else:
            raise ValueError('Classifier not recognized!')

        self.validation_class = \
            self.classifier.predict(self.validation_features)
        self.validation_prob = \
            self.classifier.predict_proba(self.validation_features)

        

        if save_predictions:
            id_name = self.identify_keywords()

            out_fname = 'predict_loop_' + str(loop) + '.csv'
            op = open(pred_dir + '/' + out_fname, 'w')
            op.write(id_name + ',' + 'prob_nIa, prob_Ia,pred_class\n')
            for i in range(self.validation_metadata.shape[0]):
                op.write(str(self.validation_metadata[id_name].iloc[i]) + ',')
                op.write(str(self.validation_prob[i][0]) + ',')
                op.write(str(self.validation_prob[i][1]) + ',')
                op.write(str(self.validation_class[i]) + '\n')
            op.close()

    def calculate_photoIa(self, threshold: float):
        """Get photometrically classified Ia sample.

        Populate the attribute 'photo_Ia_metadata'.

        Parameters
        ----------
        threshold: float
            Probability threshold above which an object is considered Ia.
        """

        # photo Ia flag
        photo_flag = self.validation_prob[:,1] >= threshold

        if 'objid' in self.validation_metadata.keys():
            id_name = 'objid'
        elif 'id' in self.validation_metadata.keys():
            id_name = 'id'

        # get ids
        photo_Ia_metadata = self.validation_metadata[photo_flag]

        self.photo_Ia_metadata = photo_Ia_metadata


    def translate_types(self, metadata_fname: str):
        """Translate types from zenodo to SNANA codes.

        Populates the attribute 'photo_Ia_metadata'.

        Parameters
        ----------
        metadata_fname: str
            Full path to PLAsTiCC zenodo metadata file.
        """

        data = self.photo_Ia_metadata.copy(deep=True)
        data_z = pd.read_csv(metadata_fname)

        data['code_zenodo'] = data.copy()['code'].values

        codes = []
        for i in range(data.shape[0]):
            
            sncode = data.iloc[i]['code']
            if  sncode not in [62, 42, 6]:
                codes.append(self.SNANA_types[sncode])
            else:
                flag = data_z['object_id'].values == data.iloc[i]['id']
                submodel = data_z[flag]['true_submodel'].values[0]
                codes.append(self.SNANA_types[sncode][submodel])

        # convert integers when necessary
        data['id'] = data['id'].values.astype(int)
        data['code'] = codes
        data['code_zenodo'] = data['code_zenodo'].values.astype(int)

        del self.photo_Ia_metadata
        self.photo_Ia_metadata = data

    def output_photo_Ia(self, threshold: float, filename: str, 
                        metadata_fname: str,
                        to_file=True, SNANA_types=False):
        """Returns the metadata for  photometrically classified SN Ia.

        Parameters
        ----------
        metadata_fname: str
            Full path to PLAsTiCC zenodo test metadata file.
        threshold: float
            Probability threshold above which an object is considered Ia.
        SNANA_types: bool (optional)
            if True, translate type to SNANA codes and
            add column with original values. Default is False.
        to_file: bool (optional)
            If true, populate the photo_Ia_list attribute. Otherwise
            write to file. Default is False.
        filename: str (optional)
            Name of output file. Only used if to_file is True.

        Returns
        -------
        pd.DataFrame
            if to_file is False, otherwise write DataFrame to file.
        """

        # identify metadata
        self.calculate_photoIa(threshold=threshold)

        # translate to SNANA types
        if SNANA_types:
            self.translate_types(metadata_fname=metadata_fname)

        if to_file:
            self.photo_Ia_metadata.to_csv(filename, index=False)

    def evaluate_classification(self, metric_label='snpcc', screen=False):
        """Evaluate results from classification.

        Populate properties: metric_list_names and metrics_list_values.

        Parameters
        ----------
        metric_label: str (optional)
            Choice of metric. Currenlty only `snpcc` is accepted.
        screen: bool (optional)
            If True, display debug comments on screen.
            Default is False.
        """

        if metric_label == 'snpcc':
            self.metrics_list_names, self.metrics_list_values = \
                get_snpcc_metric(list(self.validation_class),
                                 list(self.validation_labels))
        else:
            raise ValueError('Only snpcc metric is implemented!'
                             '\n Feel free to add other options.')

        if screen:
            print('\n Metrics names: ', self.metrics_list_names)
            print('Metrics values: ', self.metrics_list_values)


    def make_query_budget(self, budgets, strategy='UncSampling', screen=False) -> list:
        """Identify new object to be added to the training sample.

        Parameters
        ----------
        budgets: tuple of ints
            Budgets for 4m and 8m respectively. 
        strategy: str (optional)
            Strategy used to choose the most informative object.
            Current implementation accepts 'UncSampling' and
            'RandomSampling', 'UncSamplingEntropy',
            'UncSamplingLeastConfident', 'UncSamplingMargin',
            'QBDMI', 'QBDEntropy', . Default is `UncSampling`.
        screen: bool (optional)
            If true, display on screen information about the
            displacement in order and classificaion probability due to
            constraints on queryable sample. Default is False.

        Returns
        -------
        query_indx: list
            List of indexes identifying the objects to be queried within budget.
        """
        if screen:
            print('\n Inside make_query_budget: ')
            print('       ... classprob: ', self.classprob.shape[0])
            print('       ... queryable_ids: ', self.queryable_ids.shape[0])
            print('       ... pool_ids: ', self.pool_metadata.shape[0])
            
        id_name = self.identify_keywords()
        queryable_ids = self.queryable_ids
        pool_metadata = self.pool_metadata
        
        if strategy == 'UncSampling':
            query_indx = batch_queries_uncertainty(class_probs=self.classprob,
                                                   id_name=id_name,
                                                   queryable_ids=queryable_ids,
                                                   pool_metadata=pool_metadata,
                                                   budgets=budgets,
                                                   criteria="uncertainty" )

        elif strategy == 'UncSamplingEntropy':
            query_indx = batch_queries_uncertainty(class_probs=self.classprob,
                                                   id_name=id_name,
                                                   queryable_ids=queryable_ids,
                                                   pool_metadata=pool_metadata,
                                                   budgets=budgets,
                                                   criteria="entropy" )

        elif strategy == 'UncSamplingLeastConfident':
            query_indx = batch_queries_uncertainty(class_probs=self.classprob,
                                                   id_name=id_name,
                                                   queryable_ids=queryable_ids,
                                                   pool_metadata=pool_metadata,
                                                   budgets=budgets,
                                                   criteria="least_confident" )

        elif strategy == 'UncSamplingMargin':
            query_indx = batch_queries_uncertainty(class_probs=self.classprob,
                                                   id_name=id_name,
                                                   queryable_ids=queryable_ids,
                                                   pool_metadata=pool_metadata,
                                                   budgets=budgets,
                                                   criteria="margin" )

        elif strategy == 'QBDMI':
            query_indx = batch_queries_mi_entropy(probs_B_K_C=self.ensemble_probs,
                                                  id_name=id_name,
                                                  queryable_ids=queryable_ids,
                                                  pool_metadata=pool_metadata,
                                                  budgets=budgets,
                                                  criteria="MI" )

        elif strategy =='QBDEntropy':
            query_indx = batch_queries_mi_entropy(probs_B_K_C=self.ensemble_probs,
                                                  id_name=id_name,
                                                  queryable_ids=queryable_ids,
                                                  pool_metadata=pool_metadata,
                                                  budgets=budgets,
                                                  criteria="entropy" )

        elif strategy == 'RandomSampling':
            query_indx = batch_queries_uncertainty(class_probs=self.classprob,
                                                   id_name=id_name,
                                                   queryable_ids=queryable_ids,
                                                   pool_metadata=pool_metadata,
                                                   budgets=budgets,
                                                   criteria="random" )

        else:
            raise ValueError('Invalid strategy.')

        for n in query_indx:
            if self.pool_metadata[id_name].values[n] not in self.queryable_ids:
                raise ValueError('Chosen object is not available for query!')

        return query_indx

    def make_query(self, strategy='UncSampling', batch=1,
                   screen=False, queryable=False, query_thre=1.0) -> list:
        """Identify new object to be added to the training sample.

        Parameters
        ----------
        strategy: str (optional)
            Strategy used to choose the most informative object.
            Current implementation accepts 'UncSampling' and
            'RandomSampling', 'UncSamplingEntropy',
            'UncSamplingLeastConfident', 'UncSamplingMargin',
            'QBDMI', 'QBDEntropy', . Default is `UncSampling`.
        batch: int (optional)
            Number of objects to be chosen in each batch query.
            Default is 1.
        queryable: bool (optional)
            If True, consider only queryable objects.
            Default is False.
        query_thre: float (optional)
            Percentile threshold where a query is considered worth it.
            Default is 1 (no limit).
        screen: bool (optional)
            If true, display on screen information about the
            displacement in order and classificaion probability due to
            constraints on queryable sample. Default is False.

        Returns
        -------
        query_indx: list
            List of indexes identifying the objects to be queried in decreasing
            order of importance.
            If strategy=='RandomSampling' the order is irrelevant.
        """
        if screen:
            print('\n Inside make_query: ')
            print('       ... classprob: ', self.classprob.shape[0])
            print('       ... queryable_ids: ', self.queryable_ids.shape[0])
            print('       ... pool_ids: ', self.pool_metadata.shape[0])

        id_name = self.identify_keywords()

        if strategy == 'UncSampling':
            query_indx = uncertainty_sampling(class_prob=self.classprob,
                                              queryable_ids=self.queryable_ids,
                                              test_ids=self.pool_metadata[id_name].values,
                                              batch=batch, screen=screen,
                                              query_thre=query_thre)


        elif strategy == 'UncSamplingEntropy':
            query_indx = uncertainty_sampling_entropy(class_prob=self.classprob,
                                              queryable_ids=self.queryable_ids,
                                              test_ids=self.pool_metadata[id_name].values,
                                              batch=batch, screen=screen,
                                              query_thre=query_thre)

        elif strategy == 'UncSamplingLeastConfident':
            query_indx = uncertainty_sampling_least_confident(class_prob=self.classprob,
                                              queryable_ids=self.queryable_ids,
                                              test_ids=self.pool_metadata[id_name].values,
                                              batch=batch, screen=screen,
                                              query_thre=query_thre)

        elif strategy == 'UncSamplingMargin':
            query_indx = uncertainty_sampling_margin(class_prob=self.classprob,
                                              queryable_ids=self.queryable_ids,
                                              test_ids=self.pool_metadata[id_name].values,
                                              batch=batch, screen=screen,
                                              query_thre=query_thre)
            return query_indx
        elif strategy == 'QBDMI':
            query_indx = qbd_mi(ensemble_probs=self.ensemble_probs,
                                queryable_ids=self.queryable_ids,
                                test_ids=self.pool_metadata[id_name].values,
                                batch=batch, screen=screen,
                                query_thre=query_thre)

        elif strategy =='QBDEntropy':
            query_indx = qbd_entropy(ensemble_probs=self.ensemble_probs,
                                    queryable_ids=self.queryable_ids,
                                    test_ids=self.pool_metadata[id_name].values,
                                    batch=batch, screen=screen,
                                    query_thre=query_thre)

        elif strategy == 'RandomSampling':
            query_indx = random_sampling(queryable_ids=self.queryable_ids,
                                         test_ids=self.pool_metadata[id_name].values,
                                         queryable=queryable, batch=batch,
                                         query_thre=query_thre, screen=screen)

        else:
            raise ValueError('Invalid strategy.')

        if screen:
            print('       ... queried obj id: ', self.pool_metadata[id_name].values[query_indx[0]])

        # check if there are repeated ids
        for n in query_indx:
            if self.pool_metadata[id_name].values[n] not in self.queryable_ids:
                raise ValueError('Chosen object is not available for query!')

        return query_indx

    def update_samples(self, query_indx: list, epoch=20,
                       queryable=False, screen=False, alternative_label=False):
        """Add the queried obj(s) to training and remove them from test.

        Update properties: train_headers, train_features, train_labels,
        test_labels, test_headers and test_features.

        Parameters
        ----------
        query_indx: list
            List of indexes identifying objects to be moved.
        alternative_label: bool (optional)
            Update the training sample with the opposite label
            from the one optained from the classifier.
            Default is False.
            At this point it only works with batch=1.
        epoch: int (optional)
            Day since beginning of survey. Default is 20.
        queryable: bool (optinal)
            If True, consider queryable flag. Default is False.
        screen: bool (optional)
            If True, display debug comments on screen.
            Default is False.
        """
        id_name = self.identify_keywords()

        ### keep track of number evolution ####
        npool = self.pool_metadata.shape[0]
        ntrain = self.train_metadata.shape[0]
        ntest = self.test_metadata.shape[0]
        nvalidation = self.validation_metadata.shape[0]

        q2 = np.copy(query_indx)
        nquery = len(q2)

        data_copy = self.pool_metadata.copy()
        query_ids = [data_copy['id'].values[item] for item in query_indx]

        while len(query_indx) > 0 and self.pool_metadata.shape[0] > 0:

            if self.pool_metadata.shape[0] != self.pool_labels.shape[0]:
                raise ValueError('Missing data in the pool sample!')

            # identify queried object index
            obj = query_indx[0]

            # add object to the query sample
            query_header0 = self.pool_metadata.values[obj]

            # check if we add normal or reversed label
            if nquery == 1 and alternative_label:
                self.alt_label = True
                new_header = []

                for i in range(len(query_header0)):
                    # add all elements of header, except type
                    if i < 2 or i > 3:
                        new_header.append(query_header0[i])
                    # add reverse label
                    elif i == 2 and query_header0[i] == 'Ia':
                        new_header.append('X')
                        new_header.append(99)
                    elif i == 2 and query_header0[i] != 'Ia':
                        new_header.append('Ia')
                        new_header.append(90)

                query_header = new_header

            elif not alternative_label:
                query_header = query_header0

            elif nquery > 1 and alternative_label:
                raise ValueError('Alternative label only works with batch=1!')

            query_features = self.pool_features[obj]
            line = [epoch]
            for item in query_header:
                line.append(item)
            for item1 in query_features:
                line.append(item1)

            self.queried_sample.append(line)

            # add object to the training sample
            new_header = pd.DataFrame([query_header], columns=self.metadata_names)
            self.train_metadata = pd.concat([self.train_metadata, new_header], axis=0,
                                            ignore_index=True)
            self.train_features = np.append(self.train_features,
                                            np.array([self.pool_features[obj]]),
                                            axis=0)
            self.train_labels = np.append(self.train_labels,
                                          np.array([self.pool_labels[obj]]),
                                          axis=0)

            # remove queried object from pool sample
            query_flag = self.pool_metadata[id_name].values == \
                 self.pool_metadata[id_name].iloc[obj]

            if sum(query_flag) > 1:
                print('Repeated id: ')
                print(self.pool_metadata[query_flag])
                raise ValueError('Found repeated ids in pool sample!')

            pool_metadata_temp = self.pool_metadata.copy()
            self.pool_metadata = pool_metadata_temp[~query_flag]
            self.pool_labels = self.pool_labels[~query_flag]
            self.pool_features = self.pool_features[~query_flag]

            if queryable:
                qids_flag = self.pool_metadata['queryable'].values
                self.queryable_ids = self.pool_metadata[id_name].values[qids_flag]
            else:
                self.queryable_ids = self.pool_metadata[id_name].values

            # check if queried object is also in other samples
            test_ids = self.test_metadata[id_name].values
            if query_header[0] in test_ids:
                qtest_flag = self.test_metadata[id_name].values == \
                    query_header[0]
                test_metadata_temp = self.test_metadata.copy()
                self.test_labels = self.test_labels[~qtest_flag]
                self.test_features = self.test_features[~qtest_flag]
                self.test_metadata = test_metadata_temp[~qtest_flag]

            validation_ids = self.validation_metadata[id_name].values
            if query_header[0] in validation_ids:
                qval_flag = self.validation_metadata[id_name].values == \
                    query_header[0]
                validation_metadata_temp = self.validation_metadata.copy()
                self.validation_labels = self.validation_labels[~qval_flag]
                self.validation_features = self.validation_features[~qval_flag]
                self.validation_metadata = validation_metadata_temp[~qval_flag]

            # update ids order
            query_indx.remove(obj)

            new_query_indx = []

            for item in query_indx:
                if item < obj:
                    new_query_indx.append(item)
                else:
                    new_query_indx.append(item - 1)

            query_indx = new_query_indx

            if screen:
                print('remaining  queries: ', query_indx)

        # test
        npool2 = self.pool_metadata.shape[0]
        ntrain2 = self.train_metadata.shape[0]
        ntest2 = self.test_metadata.shape[0]
        nvalidation2 = self.validation_metadata.shape[0]

        if screen:
            print('query_ids: ', query_ids)
            print('queried sample: ', self.queried_sample[-1][1])
            print('----------------------------------------------')

        if ntest2 > ntest or nvalidation2 > nvalidation:
            raise ValueError('Wrong dimensionality for test/val samples.')

        for name in query_ids:
            if name in self.pool_metadata['id'].values:
                raise ValueError('Queried object ', name, ' is still in pool sample!')

            if name not in self.train_metadata['id'].values:
                raise ValueError('Queried object ', name, ' not in training!')

        # check if there are repeated ids
        for name in self.train_metadata['id'].values:
            if name in self.pool_metadata['id'].values:
                raise ValueError('After update! Object ', name,
                                 ' found in pool and training samples!')

            if name in self.test_metadata['id'].values:
                raise ValueError('After update! Object ', name,
                                 ' found in test and training samples!')

            if name in self.validation_metadata['id'].values:
                raise ValueError('After update! Object ', name,
                                 ' found in validation and training samples!')

        if ntrain2 != ntrain + nquery or npool2 != npool - nquery:
            raise ValueError('Wrong dimensionality for train/pool samples!')

    def save_metrics(self, loop: int, output_metrics_file: str, epoch: int, batch=1):
        """Save current metrics to file.

        If loop == 0 the 'output_metrics_file' will be created or overwritten.
        Otherwise results will be added to an existing 'output_metrics file'.

        Parameters
        ----------
        loop: int
            Number of learning loops finished at this stage.
        output_metrics_file: str
            Full path to file to store metrics results.
        batch: int
            Number of queries in each loop.
        epoch: int
            Days since the beginning of the survey.
        """

        # add header to metrics file
        if not os.path.exists(output_metrics_file) or loop == 0:
            with open(output_metrics_file, 'w') as metrics:
                metrics.write('loop,')
                for name in self.metrics_list_names:
                    metrics.write(name + ',')
                for j in range(batch - 1):
                    metrics.write('query_id' + str(j + 1) + ',')
                metrics.write('query_id' + str(batch) + '\n')

        # write to file)
        queried_sample = np.array(self.queried_sample)
        flag = queried_sample[:,0].astype(int) == epoch

        if sum(flag) > 0:
            with open(output_metrics_file, 'a') as metrics:
                metrics.write(str(epoch) + ',')
                for value in self.metrics_list_values:
                    metrics.write(str(value) + ',')
                for j in range(sum(flag) - 1):
                    metrics.write(str(queried_sample[flag][j][1]) + ',')
                metrics.write(str(queried_sample[flag][sum(flag) - 1][1]) + '\n')


    def save_queried_sample(self, queried_sample_file: str, loop: int,
                            full_sample=False, batch=1, epoch=20):
        """Save queried sample to file.

        Parameters
        ----------
        queried_sample_file: str
            Complete path to output file.
        loop: int
            Number of learning loops finished at this stage.
        full_sample: bool (optional)
            If true, write down a complete queried sample stored in
            property 'queried_sample'. Otherwise append 1 line per loop to
            'queried_sample_file'. Default is False.
        epoch: int  (optional)
            Days since the beginning of the survey. Default is 20.
        """

        if full_sample and len(self.queried_sample) > 0:
            full_header = ['epoch'] + self.metadata_names + self.features_names
            query_sample = pd.DataFrame(self.queried_sample, columns=full_header)
            query_sample.sort_values(by='epoch').to_csv(queried_sample_file, index=False)

        elif isinstance(loop, int):
            queried_sample = np.array(self.queried_sample)
            flag = queried_sample[:,0].astype(int) == epoch
            if sum(flag) > 0:
                if not os.path.exists(queried_sample_file) or loop == 0:
                    # add header to query sample file
                    full_header = self.metadata_names + self.features_names
                    with open(queried_sample_file, 'w') as query:
                        query.write('day,')
                        for item in full_header:
                            query.write(item + ',')
                        query.write('\n')

                # save query sample to file
                with open(queried_sample_file, 'a') as query:
                    for batch in range(batch):
                        for elem in queried_sample[flag][batch]:
                            query.write(str(elem) + ',')
                        query.write('\n')


def main():
    return None


if __name__ == '__main__':
    main()
