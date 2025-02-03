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

import glob
import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import progressbar
from sklearn.neighbors import NearestNeighbors, KernelDensity

from resspect.database import DataBase
from resspect.feature_extractors.light_curve import LightCurve
from resspect.lightcurves_utils import SNPCC_CANONICAL_FEATURES
from resspect.lightcurves_utils import SNPCC_META_HEADER

__all__ = ['Canonical', 'build_snpcc_canonical', 'plot_snpcc_train_canonical']


class Canonical:
    """Canonical sample object.

    Attributes
    ----------
    canonical_ids: list
        List of ids for objects in the canonical sample.
    canonical_sample: list
        Complete data matrix for the canonical sample.
    meta_data: pd.DataFrame
        Metadata on sim peakmag and SNR for all objects in the original data set.
    test_ia_data: pd.DataFrame
        Metadata on sim peakmag and SNR for SN Ias in the test sample.
    test_ia_id: np.array
        Set of ids for all SN Ia in the test sample.
    test_ibc_data: pd.DataFrame
        Metadata on sim peakmag and SNR for SN Ibcs in the test sample.
    test_ibc_id: np.array
        Set of ids for all SN Ibc in the test sample.
    test_ii_data: pd.DataFrame
        Metadata on sim peakmag and SNR for SN IIs in the test sample.
    test_ii_id: np.array
        Set of ids for all SN II in the test sample.
    train_ia_data: pd.DataFrame
        Metadata on sim peakmag and SNR for SN Ias in the train sample.
    train_ia_id: np.array
        Set of ids for all SN Ia in the train sample.
    train_ibc_data: pd.DataFrame
        Metadata on sim peakmag and SNR for SN Ibcs in the train sample.
    train_ibc_id: np.array
        Set of ids for all SN Ibc in the train sample.
    train_ii_data: pd.DataFrame
        Metadata on sim peakmag and SNR for SN IIs in the train sample.
    train_ii_id: np.array
        Set of ids for all SN II in the train sample.

    Methods
    -------
    snpcc_get_canonical_info(path_to_rawdata_dir: str, canonical_output_file: st,
                             compute: bool, save: bool, canonical_input_file: str)
        Load SNPCC metada data required to characterize objects.
    snpcc_identify_samples()
        Identify training and test sample.
    find_neighbors()
        Identify 1 nearest neighbor for each object in training.
    """

    def __init__(self):
        self.canonical_ids = []
        self.canonical_sample = []
        self.meta_data = pd.DataFrame()
        self.test_ia_data = pd.DataFrame()
        self.test_ia_id = np.array([])
        self.test_ibc_data = pd.DataFrame()
        self.test_ibc_id = np.array([])
        self.test_ii_data = pd.DataFrame()
        self.test_ii_id = np.array([])
        self.train_ia_data = pd.DataFrame()
        self.train_ia_id = np.array([])
        self.train_ibc_data = pd.DataFrame()
        self.train_ibc_id = np.array([])
        self.train_ii_data = pd.DataFrame()
        self.train_ii_id = np.array([])
        self.header = None

    def snpcc_get_canonical_info(self, path_to_rawdata: str,
                                 canonical_output_file: str,
                                 compute_meta_data: bool = True,
                                 save: bool = True,
                                 canonical_input_file: str = None):
        """
        Load SNPCC metada data required to characterize objects.

        Populates attribute: data.

        Parameters
        ----------
        path_to_rawdata: str
            Complete path to directory holding raw data files.
        canonical_output_file: str
            Complete path to output canonical sample file.
        canonical_input_file: str (optional)
            Path to input file if required metadata was previously calculated.
            If name is give, 'compute' must be False.
        compute_meta_data: bool (optional)
            Compute required metada from raw data files.
            Default is True.
        save: bool (optional)
            Save metadata to file. Default is True.
        """
        if compute_meta_data:
            snpcc_sim_meta_info = self.get_light_curves_meta_data(
                path_to_rawdata)
            self.meta_data = pd.DataFrame(snpcc_sim_meta_info,
                                          columns=SNPCC_META_HEADER)
        else:
            if not canonical_input_file:
                raise ValueError('File not found! Set "calculate = True" '
                                 'to build canonical info file.')
            self.meta_data = pd.read_csv(canonical_input_file,
                                         index_col=False)
        if save:
            self.meta_data.to_csv(canonical_output_file, index=False)

    def get_light_curves_meta_data(self, path_to_rawdata: str) -> list:
        light_cure_files = _get_files_list(path_to_rawdata)
        snpcc_sim_meta_info = []
        for each_file in light_cure_files:
            snpcc_sim_meta_info.append(
                self.process_light_curve_file(each_file))
        return snpcc_sim_meta_info

    def process_light_curve_file(self, file_path: str) -> list:
        light_curve_data = LightCurve()
        light_curve_data.load_snpcc_lc(file_path)
        meta_info = get_light_curve_meta_info(light_curve_data)
        meta_info.extend(self._process_light_curve_filters(light_curve_data))
        return meta_info

    @staticmethod
    def _process_light_curve_filters(light_curve_data: LightCurve) -> list:
        snr_info = []
        for each_filter in light_curve_data.filters:
            filter_indices = (
                    light_curve_data.photometry['band'].values == each_filter)
            if np.sum(filter_indices) > 0:
                snr_info.append(np.mean(
                    light_curve_data.photometry['SNR'].values[filter_indices]))
            else:
                snr_info.append(0)
        return snr_info

    def snpcc_identify_samples(self):
        """Identify training and test sample.

        Populates attributes: train_ia_data, train_ia_id, train_ibc_data,
        train_ibc_id, train_ii_data, train_ibc_id, test_ia_data, test_ia_id,
        test_ibc_data, test_ibc_id, test_ii_data and test_ii_id.
        """
        train_indices, ia_indices, ibc_indices, ii_indices = (
            self._get_train_and_class_indices())
        self._load_train_samples(train_indices, ia_indices, ibc_indices,
                                 ii_indices)
        self._load_test_samples(train_indices, ia_indices, ibc_indices,
                                ii_indices)

    def _load_train_samples(self, train_indices, ia_indices,
                            ibc_indices, ii_indices):
        # get training sample

        self.train_ia_data = self.meta_data[
            SNPCC_CANONICAL_FEATURES][np.logical_and(train_indices, ia_indices)]
        self.train_ia_id = self.meta_data[
            'snid'].values[np.logical_and(train_indices, ia_indices)]
        self.train_ibc_data = self.meta_data[
            SNPCC_CANONICAL_FEATURES][np.logical_and(train_indices, ibc_indices)]
        self.train_ibc_id = self.meta_data[
            'snid'].values[np.logical_and(train_indices, ibc_indices)]
        self.train_ii_data = self.meta_data[
            SNPCC_CANONICAL_FEATURES][np.logical_and(train_indices, ii_indices)]
        self.train_ii_id = self.meta_data[
            'snid'].values[np.logical_and(train_indices, ii_indices)]

    def _load_test_samples(self, train_indices, ia_indices,
                           ibc_indices, ii_indices):
        # get test sample
        self.test_ia_data = self.meta_data[
            SNPCC_CANONICAL_FEATURES][np.logical_and(~train_indices, ia_indices)]
        self.test_ia_id = self.meta_data[
            'snid'].values[np.logical_and(~train_indices, ia_indices)]
        self.test_ibc_data = self.meta_data[
            SNPCC_CANONICAL_FEATURES][np.logical_and(~train_indices, ibc_indices)]
        self.test_ibc_id = self.meta_data[
            'snid'].values[np.logical_and(~train_indices, ibc_indices)]
        self.test_ii_data = self.meta_data[
            SNPCC_CANONICAL_FEATURES][np.logical_and(~train_indices, ii_indices)]
        self.test_ii_id = self.meta_data[
            'snid'].values[np.logical_and(~train_indices, ii_indices)]

    def _get_train_and_class_indices(self):
        train_flag = self.meta_data['orig_sample'].values == 'train'
        ia_flag = self.meta_data['sntype'].values == 'Ia'
        ibc_flag = self.meta_data['sntype'].values == 'Ibc'
        ii_flag = self.meta_data['sntype'].values == 'II'
        return train_flag, ia_flag, ibc_flag, ii_flag

    @staticmethod
    def _get_nearest_neighbor_indices(
            nearest_neighbor_class: NearestNeighbors,
            current_test_sample: pd.DataFrame,
            current_train_sample: np.ndarray):
        nearest_neighbor_class.fit(current_test_sample)
        neighbor_indices = nearest_neighbor_class.kneighbors(
            current_train_sample)
        return neighbor_indices[1]

    def _update_canonical_ids(self, nearest_neighbor_indices: list):
        test_ids = [self.test_ia_id, self.test_ibc_id, self.test_ii_id]
        sampled_indices = []
        for index, samples in enumerate(nearest_neighbor_indices):
            for each_sample in samples:
                for current_index in each_sample:
                    # only add elements which were not already added
                    # to sampled_indices
                    if current_index not in sampled_indices:
                        current_id = test_ids[index][current_index]
                        self.canonical_ids.append(current_id)
                        sampled_indices.append(current_index)
                        break

    def find_neighbors(self, number_of_neighbors: int = 10,
                       nearest_neighbor_algorithm: str = 'auto'):
        """
        Identify 1 nearest neighbor for each object in training.
        Populates attribute: canonical_ids.

        Parameters
        ----------
        number_of_neighbors: int
            number of neighbors in each sample
        nearest_neighbor_algorithm
            algorithm to estimate nearest neighbors, check scikit
            NearestNeighbors documentation
        """

        # gather samples by type
        train_samples = [self.train_ia_data, self.train_ibc_data,
                         self.train_ii_data]
        test_samples = [self.test_ia_data, self.test_ibc_data,
                        self.test_ii_data]

        nearest_neighbor_indices = []
        for index, each_test_sample in enumerate(
                progressbar.progressbar(test_samples)):
            nearest_neighbor_class = NearestNeighbors(
                n_neighbors=number_of_neighbors,
                algorithm=nearest_neighbor_algorithm)
            current_train_sample = train_samples[index]
            nearest_neighbor_indices.append(
                self._get_nearest_neighbor_indices(nearest_neighbor_class,
                                                   each_test_sample,
                                                   current_train_sample))
        self._update_canonical_ids(nearest_neighbor_indices)


def get_light_curve_meta_info(light_curve_data: LightCurve) -> list:
    """
        Get ligtht curve meta information

    Parameters
    ----------
    light_curve_data
        LightCurve class instance
    """
    meta_info = [light_curve_data.id, light_curve_data.sample,
                 light_curve_data.sntype, light_curve_data.redshift]
    meta_info.extend(light_curve_data.sim_peakmag.tolist())
    return meta_info


def _get_files_list(folder_path: str, file_name_prefix: str = 'DES') -> list:
    """
        Load file names with file name prefix

    Parameters
    ----------
    folder_path
        folder path to load files from
    file_name_prefix
        file names prefix
    """
    folder_path = os.path.join(folder_path, file_name_prefix + '*')
    return glob.glob(folder_path)


def get_meta_data_from_features(path_to_features: str,
                                features_method: str,
                                screen: bool = True) -> DataBase:
    """
        Get metadata from features file
       #TODO: maybe update screen argument after refactoring DataBase()

    Parameters
    ----------
    path_to_features: str
        Complete path to Bazin features files
    features_method: str (optional)
        Method for feature extraction. Only 'bazin' is implemented.
    """
    data = DataBase()
    data.load_features(path_to_file=path_to_features, feature_extractor=features_method,
                       screen=screen)
    return data


def build_snpcc_canonical(path_to_raw_data: str, path_to_features: str,
                          output_canonical_file: str, output_info_file='',
                          compute=True, save=True, input_info_file='',
                          features_method='bazin', screen=False,
                          number_of_neighbors=1):
    """Build canonical sample for SNPCC data.

    Parameters
    ----------
    path_to_raw_data: str
        Complete path to raw data directory.
    path_to_features: str
        Complete path to Bazin features files.
    output_canonical_file: str
        Complete path to output canonical sample file.
    output_info_file: str
        Complete path to output metadata file for canonical sample.
        This includes SNR and simulated peak magnitude for each filter.
    compute: bool (optional)
        If True, compute metadata information on SNR and sim peak mag.
        If False, read info from file.
        Default is True.
    features_method: str (optional)
        Method for feature extraction. Only 'Bazin' is implemented.
    input_info_file: str (optional)
        Complete path to sim metadata file.
        This must be provided if save == False.
    save: bool (optional)
        Save simulation metadata information to file.
        Default is True.
    screen: bool (optional)
        If true, display steps info on screen. Default is False.
    number_of_neighbors: int (optional)
        Number of neighbors in each sample. Default is 1.

    Returns
    -------
    resspect.Canonical: obj
        Updated canonical object with the attribute 'canonical_sample'.
    """

    # initiate canonical object
    sample = Canonical()
    # get necessary info
    sample.snpcc_get_canonical_info(path_to_rawdata=path_to_raw_data,
                                    canonical_output_file=output_info_file,
                                    canonical_input_file=input_info_file,
                                    compute_meta_data=compute, save=save)

    # identify samples
    sample.snpcc_identify_samples()
    # find neighbors
    sample.find_neighbors(number_of_neighbors=number_of_neighbors)

    # get metadata from features file
    features_data = get_meta_data_from_features(
        path_to_features, features_method, screen)
    sample.header = features_data.metadata
    # identify new samples

    flag = features_data.metadata["id"].isin(sample.canonical_ids)
    for i in range(flag.shape[0]):
        features_data.metadata.at[i, 'queryable'] = flag[i]

    # save to file
    features = pd.DataFrame(features_data.features,
                            columns=features_data.features_names)
    features_data.data = pd.concat([features_data.metadata, features], axis=1)
    features_data.data.to_csv(output_canonical_file, index=False)

    # update Canonical object
    sample.canonical_sample = features_data.data
    return sample


def plot_snpcc_train_canonical(sample: Canonical, output_plot_file=False):
    """Plot comparison between training and canonical samples.

    Parameters
    ----------
    sample: resspect.Canonical
        Canonical object holding infor for canonical sample
    output_plot_file: str (optional)
        Complete path to output plot.
        If not provided, plot is displayed on screen only.
    """

    # prepare data
    ztrain = sample.meta_data['z'][sample.meta_data['orig_sample'].values == 'train']
    ztrain_axis = np.linspace(min(ztrain) - 0.25, max(ztrain) + 0.25, 1000)[:, np.newaxis]
    kde_ztrain = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(np.array(ztrain).reshape(-1, 1))
    log_dens_ztrain = kde_ztrain.score_samples(ztrain_axis)

    canonical_flag = sample.canonical_sample['queryable']
    zcanonical = sample.canonical_sample[canonical_flag]['redshift'].values
    zcanonical_axis = np.linspace(min(zcanonical) - 0.25, max(zcanonical) + 0.25, 1000)[:, np.newaxis]
    kde_zcanonical = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(np.array(zcanonical).reshape(-1, 1))
    log_dens_zcanonical = kde_zcanonical.score_samples(zcanonical_axis)

    gtrain = sample.meta_data['g_pkmag'][sample.meta_data['orig_sample'] == 'train'].values
    gtrain_axis = np.linspace(min(gtrain) - 0.25, max(gtrain) + 0.25, 1000)[:, np.newaxis]
    kde_gtrain = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(gtrain).reshape(-1, 1))
    log_dens_gtrain = kde_gtrain.score_samples(gtrain_axis)

    gcanonical = [sample.meta_data['g_pkmag'].values[i] for i in range(sample.meta_data.shape[0])
                  if sample.meta_data['snid'][i] in sample.canonical_ids]
    gcanonical_axis = np.linspace(min(gcanonical) - 0.25, max(gcanonical) + 0.25, 1000)[:, np.newaxis]
    kde_gcanonical = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(gcanonical).reshape(-1, 1))
    log_dens_gcanonical = kde_gcanonical.score_samples(gcanonical_axis)

    rtrain = sample.meta_data['r_pkmag'][sample.meta_data['orig_sample'] == 'train'].values
    rtrain_axis = np.linspace(min(rtrain) - 0.25, max(rtrain) + 0.25, 1000)[:, np.newaxis]
    kde_rtrain = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(rtrain).reshape(-1, 1))
    log_dens_rtrain = kde_rtrain.score_samples(rtrain_axis)

    rcanonical = [sample.meta_data['r_pkmag'].values[i] for i in range(sample.meta_data.shape[0])
                  if sample.meta_data['snid'][i] in sample.canonical_ids]
    rcanonical_axis = np.linspace(min(rcanonical) - 0.25, max(rcanonical) + 0.25, 1000)[:, np.newaxis]
    kde_rcanonical = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(rcanonical).reshape(-1, 1))
    log_dens_rcanonical = kde_rcanonical.score_samples(rcanonical_axis)

    plt.figure(figsize=(20, 7))
    plt.subplot(1, 3, 1)
    plt.plot(ztrain_axis, np.exp(log_dens_ztrain), label='original train')
    plt.plot(zcanonical_axis, np.exp(log_dens_zcanonical), label='canonical')
    plt.xlabel('redshift', fontsize=20)
    plt.ylabel('PDF', fontsize=20)
    plt.legend(loc='upper right', fontsize=15)

    plt.subplot(1, 3, 2)
    plt.plot(gtrain_axis, np.exp(log_dens_gtrain), label='original train')
    plt.plot(gcanonical_axis, np.exp(log_dens_gcanonical), label='canonical')
    plt.xlabel('g_peakmag', fontsize=20)
    plt.ylabel('PDF', fontsize=20)
    plt.xlim(15, 35)
    plt.legend(loc='upper right', fontsize=15)

    plt.subplot(1, 3, 3)
    plt.plot(rtrain_axis, np.exp(log_dens_rtrain), label='original train')
    plt.plot(rcanonical_axis, np.exp(log_dens_rcanonical), label='canonical')
    plt.xlabel('r_peakmag', fontsize=20)
    plt.ylabel('PDF', fontsize=20)
    plt.xlim(15, 35)
    plt.legend(loc='upper right', fontsize=15)

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.99, wspace=0.2)

    if output_plot_file:
        plt.savefig(output_plot_file)
        plt.close('all')
    else:
        plt.show()


def main():
    return None


if __name__ == '__main__':
    main()
