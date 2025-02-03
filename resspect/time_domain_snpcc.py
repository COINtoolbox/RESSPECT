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

import logging
import multiprocessing
import os
from itertools import repeat

from resspect.feature_extractors.bazin import BazinFeatureExtractor
from resspect.feature_extractors.bump import BumpFeatureExtractor
from resspect.feature_extractors.malanchev import MalanchevFeatureExtractor
from resspect.lightcurves_utils import BAZIN_HEADERS
from resspect.lightcurves_utils import MALANCHEV_HEADERS
from resspect.lightcurves_utils import get_files_list
from resspect.lightcurves_utils import get_query_flags
from resspect.lightcurves_utils import maybe_create_directory

logging.basicConfig(level=logging.INFO)

__all__ = ['SNPCCPhotometry']


FEATURE_EXTRACTOR_MAPPING = {
    "bazin": BazinFeatureExtractor,
    "bump": BumpFeatureExtractor,
    "malanchev": MalanchevFeatureExtractor
}


FEATURE_EXTRACTOR_HEADERS_MAPPING = {
    "bazin": BAZIN_HEADERS,
    "malanchev": MALANCHEV_HEADERS
}


class SNPCCPhotometry:
    """
    Handles photometric information for entire SNPCC data.

    This class only works for Bazin and Malanchev feature extraction methods.

    Attributes
    ----------
    max_epoch: float
        Maximum MJD for the entire data set.
    min_epoch: float
        Minimum MJD for the entire data set.
    rmag_lim: float
        Maximum r-band magnitude allowing a query.

    Methods
    -------
    get_lim_mjds(raw_data_dir)
        Get minimum and maximum MJD for complete sample.
    create_daily_file(raw_data_dir: str, day: int, output_dir: str,
                      header: str)
        Creates one file for a given day of the survey.
        Only populates the file with header.
        It will erase existing files!
    build_one_epoch(raw_data_dir: str, day_of_survey: int,
                    time_domain_dir: str, feature_extractor: str,
                    dataset: str)
        Selects objects with observed points until given MJD,
        performs feature extraction and evaluate if query is possible.
        Save results to file.
    """
    def __init__(self,
                 max_epoch: int = 56352,
                 min_epoch: int = 56171,
                 rmag_lim: int = 24):
        self.max_epoch = max_epoch
        self.min_epoch = min_epoch
        self.rmag_lim = rmag_lim
        self._header = None  # type: list
        self._features_file_name = None  # type: str
        self._today = None  # type: int
        self._number_of_telescopes = None  # type: int
        self._kwargs = None  # type: dict

    def create_daily_file(self, output_dir: str,
                          day: int, feature_extractor: str = 'bazin',
                          get_cost: bool = False):
        """
        Create one file for a given day of the survey.

        The file contains only header for the features file.

        Parameters
        ----------
        output_dir: str
            Complete path to raw data directory.
        day: int
            Day passed since the beginning of the survey.
        get_cost: bool (optional)
            If True, calculate cost of taking a spectra in the last
            observed photometric point. Default is False.
        feature_extractor: str
            Feature extraction method, only possibilities are 'Bazin' and 'malanchev'.
        """
        maybe_create_directory(output_dir)
        self._features_file_name = os.path.join(
            output_dir, 'day_' + str(day) + '.csv')
        logging.info('Creating features file')
        with open(self._features_file_name, 'w') as features_file:
            if feature_extractor not in FEATURE_EXTRACTOR_HEADERS_MAPPING:
                raise ValueError('Only Bazin and Malanchev headers are supported')
            self._header = FEATURE_EXTRACTOR_HEADERS_MAPPING[
                feature_extractor]['snpcc_header']
            if get_cost:
                self._header = FEATURE_EXTRACTOR_HEADERS_MAPPING[
                    feature_extractor]['snpcc_header_with_cost']

            features_file.write(','.join(self._header) + '\n')

    def _verify_telescope_names(self, telescope_names: list, get_cost: bool):
        """
        Verifies telescope names.

        Parameters
        ----------
        telescope_names: list
            list with telescope names.
        get_cost: bool
           if cost of taking a spectra is computed
        """

        if (('cost_' + telescope_names[0] not in self._header or
            'cost_' + telescope_names[1] not in self._header)
                and get_cost):

            raise ValueError('Unknown or not supported telescope names')

    def _maybe_create_features_file(self, output_dir: str, day_of_survey: int,
                                    feature_extractor: str, get_cost: bool):
        """
        Creates features output file if not available

        Parameters
        ----------
        output_dir: str
            output directory path to save features file
        day_of_survey: int
            Day since the beginning of survey.
        feature_extractor: str
            Feature extraction method, only possibilities are 'Bazin' and 'malanchev'.
        get_cost: bool
           if True, cost of taking a spectra is computed.
        """
        if not os.path.isfile(self._features_file_name):
            logging.info('Features file doesnt exist')
            self.create_daily_file(
                output_dir, day_of_survey, feature_extractor, get_cost)

    @staticmethod
    def _verify_dataset_and_features_method(dataset_name: str,
                                            feature_extractor: str):
        """
        Verifies if valid dataset name and features method is passed
        Parameters
        ----------
        dataset_name: str
            name of the dataset used
        feature_extractor: str
            Feature extraction method, only possibilities are 'Bazin' and 'malanchev'.
        """
        if dataset_name != 'SNPCC':
            raise ValueError('This class supports only SNPCC dataset!')
        # TODO: Update when bump headers are available
        if feature_extractor != 'bazin' and feature_extractor!='malanchev':
            raise ValueError('Only bazin and malanchev features are implemented!!')

    def _check_queryable(self, light_curve_data,
                         queryable_criteria: int,
                         days_since_last_observation: int) -> bool:
        """
        Applies check_queryable method of LightCurve class

        Parameters
        ----------
        light_curve_data:
            An instance of LightCurve class
        queryable_criteria: [1 or 2]
            Criteria to determine if an obj is queryable.
            1 -> r-band cut on last measured photometric point.
            2 -> last obs was further than a given limit,
                 use Bazin estimate of flux today. Otherwise, use
                 the last observed point.
            Default is 1.
        days_since_last_observation: int
            Day since last observation to consider for spectroscopic
            follow-up without the need to extrapolate light curve.
        """
        return light_curve_data.check_queryable(
            mjd=self._today, filter_lim=self.rmag_lim,
            criteria=queryable_criteria,
            days_since_last_obs=days_since_last_observation)

    def _update_queryable_if_get_cost(
            self, light_curve_data, telescope_names: list,
            telescope_sizes: list, spectroscopic_snr: int,
            kwargs: dict):
        """
        Updates time required tp take a spectra calc_exp_time() method of
        LightCurve class

        Parameters
        ----------
        light_curve_data
            An instance of LightCurve class
        telescope_names
            Names of the telescopes under consideration for spectroscopy.
            Only used if "get_cost == True".
            Default is ["4m", "8m"].
        telescope_sizes
            Primary mirrors diameters of potential spectroscopic telescopes.
            Only used if "get_cost == True".
            Default is [4, 8].
        spectroscopic_snr
            SNR required for spectroscopic follow-up. Default is 10.
        kwargs: dict
            Any input required by ExpTimeCalc.findexptime function.
        """
        local_keys = ['mag', 'SNRin', 'cwl_nm', 'bandpass_nm', 'band', 'airmass', 'skymode',
                      'skymag', 'nread', 'skyADU', 'fwhm']

        # select input for exposure time calculator
        kwargs2 = {}
        for name in kwargs.keys():
            if name in local_keys:
                 kwargs2[name] = kwargs[name]
        
        for index in range(self._number_of_telescopes):
            light_curve_data.calc_exp_time(
                telescope_diam=telescope_sizes[index],
                telescope_name=telescope_names[index],
                SNR=spectroscopic_snr, **kwargs2
            )
        return light_curve_data

    def _process_each_light_curve(
            self, light_curve_data, queryable_criteria: int,
            days_since_last_observation: int, telescope_names: list,
            telescope_sizes: list, spectroscopic_snr: int, kwargs: dict,
            min_available_points: int = 5):
        """
        Processes each light curve files.

        Parameters
        ----------
        light_curve_data:
            An instance of LightCurve class
        queryable_criteria: int
            Criteria to determine if an obj is queryable.
            1 -> r-band cut on last measured photometric point.
            2 -> last obs was further than a given limit,
                 use Bazin estimate of flux today. Otherwise, use
                 the last observed point.
        days_since_last_observation: int
            Day since last observation to consider for spectroscopic
            follow-up without the need to extrapolate light curve.
        telescope_names: list
            Names of the telescopes under consideration for spectroscopy.
            Only used if "get_cost == True".
        telescope_sizes: list
            Primary mirrors diameters of potential spectroscopic telescopes.
            Only used if "get_cost == True".
        spectroscopic_snr: int
            SNR required for spectroscopic follow-up.
        kwargs: dict (optional)
            Any input required by ExpTimeCalc.findexptime function.
        min_available_points: int (optional)
            minimum number of survived points. Default is 5.
        """
        photo_flag = light_curve_data.photometry['mjd'].values <= self._today
        if sum(photo_flag) >= min_available_points:
            light_curve_data.photometry = light_curve_data.photometry[
                photo_flag]
            light_curve_data.fit_all()

            if (len(light_curve_data.features) > 0 and
                    'None' not in light_curve_data.features):
                light_curve_data.queryable = self._check_queryable(
                    light_curve_data, queryable_criteria,
                    days_since_last_observation)
                light_curve_data = self._update_queryable_if_get_cost(
                    light_curve_data, telescope_names, telescope_sizes,
                    spectroscopic_snr, kwargs)
                light_curve_data.queryable = bool(sum(get_query_flags(
                    light_curve_data, telescope_names
                )))
                return light_curve_data
        return None

    def _get_features_to_write(self, light_curve_data,
                               get_cost: bool, telescope_names: list) -> list:
        """
        Returns features list to write.

        Parameters
        ----------
        light_curve_data:
            fitted light curve data
        get_cost: bool
           if cost of taking a spectra is computed
        telescope_names: list
            Names of the telescopes under consideration for spectroscopy.
            Only used if "get_cost == True".
            Default is ["4m", "8m"].
        """
        features_list = [light_curve_data.id, light_curve_data.redshift,
                         light_curve_data.sntype, light_curve_data.sncode,
                         light_curve_data.sample, light_curve_data.queryable,
                         light_curve_data.last_mag]
        if get_cost:
            for index in range(self._number_of_telescopes):
                features_list.append(str(
                    light_curve_data.exp_time[telescope_names[index]]))
        features_list.extend(light_curve_data.features)
        return features_list

    def _get_current_sample_features(
            self, file_name: str, raw_data_dir: str, queryable_criteria: int,
            days_since_observation: int, telescope_names: list,
            telescope_sizes: list, spec_SNR: int, feature_extractor: str):
        """
        Reads a SNPCC file and updates time domain features

        Parameters
        ----------
        file_name
            SNPCC file name
        raw_data_dir: str
            Complete path to raw data directory
        queryable_criteria: int [1 or 2] (optional)
            Criteria to determine if an obj is queryable.
            1 -> r-band cut on last measured photometric point.
            2 -> last obs was further than a given limit,
                 use Bazin estimate of flux today. Otherwise, use
                 the last observed point.
            Default is 1.
        days_since_observation: int (optional)
            Day since last observation to consider for spectroscopic
            follow-up without the need to extrapolate light curve.
            Only used if "queryable_criteria == 2". Default is 2.
        telescope_names: list (optional)
            Names of the telescopes under consideration for spectroscopy.
            Only used if "get_cost == True".
            Default is ["4m", "8m"].
        telescope_sizes: list (optional)
            Primary mirrors diameters of potential spectroscopic telescopes.
            Only used if "get_cost == True".
            Default is [4, 8].
        spec_SNR: float (optional)
            SNR required for spectroscopic follow-up. Default is 10.
        feature_extractor: str
            Feature extraction method
        """
        light_curve_data = FEATURE_EXTRACTOR_MAPPING[feature_extractor]()
        light_curve_data.load_snpcc_lc(
            os.path.join(raw_data_dir, file_name))
        light_curve_data = self._process_each_light_curve(
            light_curve_data, queryable_criteria, days_since_observation,
            telescope_names, telescope_sizes, spec_SNR, self._kwargs)
        return light_curve_data

    # TODO: Too many arguments. Refactor and update docs
    def build_one_epoch(self, raw_data_dir: str, day_of_survey: int,
                        time_domain_dir: str, feature_extractor: str = 'bazin',
                        dataset: str = 'SNPCC', days_since_obs: int = 2,
                        queryable_criteria: int = 1, get_cost: bool = False,
                        tel_sizes: list = [4, 8],
                        tel_names: list = ['4m', '8m'], spec_SNR: int = 10,
                        number_of_processors: int = 1, **kwargs):
        """
        Fit features for all objects with enough points in a given day.

        Generate 1 file containing best-fit Bazin or malanchev parameters for a given
        day of the survey.

        Parameters
        ----------
        raw_data_dir: str
            Complete path to raw data directory
        day_of_survey: int
            Day since the beginning of survey.
        time_domain_dir: str
            Output directory to store time domain files.
        dataset: str (optional)
            Name of the data set.
            Only possibility is 'SNPCC'.
        days_since_obs: int (optional)
            Day since last observation to consider for spectroscopic
            follow-up without the need to extrapolate light curve.
            Only used if "queryable_criteria == 2". Default is 2.
        feature_extractor: str (optional)
            Feature extraction method.
            Only possibilities are 'Bazin' or 'malanchev'.
        get_cost: bool (optional)
            If True, calculate cost of taking a spectra in the last
            observed photometric point. Default is False.
        queryable_criteria: int [1 or 2] (optional)
            Criteria to determine if an obj is queryable.
            1 -> r-band cut on last measured photometric point.
            2 -> last obs was further than a given limit,
                 use Bazin estimate of flux today. Otherwise, use
                 the last observed point.
            Default is 1.
        spec_SNR: float (optional)
            SNR required for spectroscopic follow-up. Default is 10.
        tel_names: list (optional)
            Names of the telescopes under consideration for spectroscopy.
            Only used if "get_cost == True".
            Default is ["4m", "8m"].
        tel_sizes: list (optional)
            Primary mirrors diameters of potential spectroscopic telescopes.
            Only used if "get_cost == True".
            Default is [4, 8].
        number_of_processors: int, default 1
            Number of cpu processes to use
        kwargs: extra parameters
            Any input required by ExpTimeCalc.find exptime function.
        """
        self._kwargs = kwargs
        self._verify_dataset_and_features_method(dataset, feature_extractor)
        self._verify_telescope_names(tel_names, get_cost)
        self._maybe_create_features_file(
            time_domain_dir, day_of_survey, feature_extractor, get_cost)
        files_list = get_files_list(raw_data_dir, 'DES_SN')
        self._today = day_of_survey + self.min_epoch
        self._number_of_telescopes = len(tel_names)

        multi_process = multiprocessing.Pool(number_of_processors)
        
        logging.info("Starting SNPCC time domain features extraction...")
        
        with open(self._features_file_name, 'a') as snpcc_features_file:
            iterator_list = zip(
                files_list, repeat(raw_data_dir), repeat(queryable_criteria),
                repeat(days_since_obs), repeat(tel_names), repeat(tel_sizes),
                repeat(spec_SNR), repeat(feature_extractor))
            for light_curve_data in multi_process.starmap(
                    self._get_current_sample_features, iterator_list):
                if light_curve_data is not None:
                    features_to_write = self._get_features_to_write(
                        light_curve_data, get_cost, tel_names)
                    snpcc_features_file.write(
                        ','.join(str(each_feature) for each_feature
                                 in features_to_write) + '\n')
        logging.info("Features have been saved to: %s", self._features_file_name)


def main():
    return None


if __name__ == '__main__':
    main()
