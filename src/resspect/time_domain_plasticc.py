# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
#
# created on 26 February 2020
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

import json
from itertools import repeat
import logging
import multiprocessing
import os
from copy import deepcopy
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import progressbar

from resspect.feature_extractors.bazin import BazinFeatureExtractor
from resspect.feature_extractors.bump import BumpFeatureExtractor
from resspect.lightcurves_utils import BAZIN_HEADERS
from resspect.lightcurves_utils import get_query_flags
from resspect.lightcurves_utils import maybe_create_directory
from resspect.lightcurves_utils import PLASTICC_TARGET_TYPES
from resspect.lightcurves_utils import read_plasticc_full_photometry_data


FEATURE_EXTRACTOR_HEADERS_MAPPING = {
    "bazin": BAZIN_HEADERS
}

FEATURE_EXTRACTOR_MAPPING = {
    "bazin": BazinFeatureExtractor,
    "bump": BumpFeatureExtractor
}


class PLAsTiCCPhotometry:
    """
    Handles photometric information for the PLAsTiCC data.

    Attributes
    ----------
    max_epoch: float
        Maximum MJD for the entire data set.
    metadata: pd.DataFrame
        Metadata from PLAsTiCC zenodo test sample.
    min_epoch: float
        Minimum MJD for the entire data set.
    rmag_lim: float
        Maximum r-band magnitude allowing a query.

    Methods
    -------
    build()
        Create dictionary with photometric file names.
    create_all_daily_files(raw_data_dir: str)
        Create 1 file per day for all days of the survey.
    create_daily_file(output_dir: str, day: int, vol: int, header: str)
        Create one file for a given day of the survey. Contains only header.
    fit_one_lc(raw_data_dir: str, snid: int, sample: str)
        Fit one light curve throughout the entire survey.
    read_metadata(path_to_data_dir: str, classes: list)
        Read metadata and filter only required classes.
    """
    def __init__(self,
                 max_epoch: int = 60675,
                 min_epoch: int = 59580,
                 metadata: pd.DataFrame = pd.DataFrame(),
                 rmag_lim: int = 24):
        self.metadata = metadata
        self.max_epoch = max_epoch
        self.min_epoch = min_epoch
        self.rmag_lim = rmag_lim
        self._class_mappings = None  # type: dict
        self._last_file_index = None # type: int
        self._features_file_name = None  # type: str
        self._header = None  # type: list
        self._file_list_dict = {}  # type: dict
        self._today = None  # type: int
        self._number_of_telescopes = None  # type: int
        self._previous_day_features = None
        self._previous_day_index_mapping = None
        self._kwargs = None
        self.build()

    def build(self, config='original', photo_file=None,
              sample=None):
        """Create dictionary with photometric file names. 
        
        Parameters
        ----------
        config: str (optional)
            If 'original', read original zenodo files, else
            use user provided names. Default is 'original'.
        photo_file: str or None (optional)
            Path to light curve file. Only  used if 'conf' != 'original'.
            Default is None.        
        sample: str or None (optional)
            Sample to populate with file names. Options are 'train' or 'test'. 
            Only  used if 'conf' != 'original'. Default is None.
        """

        self._last_file_index = 11
        if config == 'original':
            self._file_list_dict['test'] = [
                'plasticc_test_lightcurves_' + str(i).zfill(2) + '.csv.gz'
                for i in range(1, self._last_file_index + 1)]
            self._file_list_dict['train'] = ['plasticc_train_lightcurves.csv.gz']
        else:
            self._file_list_dict[sample] = [photo_file]

    def _set_header(self, get_cost: bool = False, feature_extractor: str = 'bazin'):
        """
        Initializes header

        Parameters
        ----------
        get_cost: bool (optional)
            If True, calculate cost of taking a spectra in the last
            observed photometric point. Default is False.
        feature_extractor: str (optional)
            List of elements to be added to the header.
            Separate by 1 space.
            Default option uses header for Bazin features file.
        """
        if feature_extractor not in FEATURE_EXTRACTOR_HEADERS_MAPPING:
            raise ValueError('Only bazin headers are supported')
        self._header = FEATURE_EXTRACTOR_HEADERS_MAPPING[
                feature_extractor]['plasticc_header']
        if get_cost:
            self._header = FEATURE_EXTRACTOR_HEADERS_MAPPING[
                feature_extractor]['plasticc_header_with_cost']

    def create_daily_file(self, output_dir: str, day: int,
                          feature_extractor: str = 'bazin', get_cost: bool = False):
        """
        Create one file for a given day of the survey.

        The file contains only header for the features file.

        Parameters
        ----------
        output_dir: str
            Complete path to output directory.
        day: int
            Day passed since the beginning of the survey.
        feature_extractor: str (optional)
            List of elements to be added to the header.
            Separate by 1 space.
            Default option uses header for Bazin features file.
        get_cost: bool (optional)
            If True, calculate cost of taking a spectra in the last
            observed photometric point. Default is False.
        """
        maybe_create_directory(output_dir)
        self._features_file_name = os.path.join(
            output_dir, 'day_' + str(day) + '.csv')
        with open(self._features_file_name, 'w') as features_file:
            self._set_header(get_cost, feature_extractor=feature_extractor)
            features_file.write(' '.join(self._header) + '\n')

    def create_all_daily_files(self, output_dir:str,
                               get_cost=False):
        """
        Create 1 file per day for all days of the survey.

        Each file contains only the header.

        Parameters
        ----------
        output_dir: str
            Output directory.
        get_cost: bool (optional)
            If True, calculate cost of taking a spectra in the last
            observed photometric point. Default is False.
        """
        # create daily files by iterating through all days of the survey
        for day_of_survey in range(1, self.max_epoch - self.min_epoch + 1):
            self.create_daily_file(output_dir=output_dir,
                                   day=day_of_survey, get_cost=get_cost)

    def read_metadata(self, path_to_data_dir: str, classes: list,
                      field: str = 'DDF', meta_data_file_name:
                      str = 'plasticc_test_metadata.csv.gz'):
        """
        Read metadata and filter only required classes.
        Populates the metadata attribute.

        Parameters
        ----------
        path_to_data_dir: str
            Directory containing all PlAsTiCC zenodo files.
        classes: list of int
            Codes for classes we wish to keep.
        field: str (optional)
            Telescope cadence.
            'DDF', 'WFD' or 'DDF+WFD'. Default is 'DDF'.
        meta_data_file_name: str (optional)
            Meta data file name. Default is 'plasticc_test_metadata.csv'.
        """

        meta_data_file_name = os.path.join(
            path_to_data_dir, meta_data_file_name)
        meta_data_raw = read_plasticc_full_photometry_data(meta_data_file_name)
        meta_data_raw = meta_data_raw[
            ['object_id', 'true_target', 'true_z', 'true_peakmjd',
             'true_distmod', 'ddf_bool']]
        classes_mask = meta_data_raw['true_target'].isin(classes)
        if field in ['WFD', 'DDF']:
            ddf_mask = meta_data_raw.ddf_bool.astype(bool).values
            if field == 'DDF':
                filter_mask = np.logical_and(ddf_mask, classes_mask)
            else:
                filter_mask = np.logical_and(~ddf_mask, classes_mask)

            self.metadata = meta_data_raw[filter_mask]
        else:
            self.metadata = meta_data_raw[classes_mask]
        self.metadata = self.metadata.drop(columns=['ddf_bool'])

    def _verify_telescope_names(self, telescope_names: list, get_cost: bool):
        """
        Verifies telescope names
        Parameters
        ----------
        telescope_names
            list with telescope names
        get_cost
           if cost of taking a spectra is computed
        """
        if (('cost_' + telescope_names[0] not in self._header or
            'cost_' + telescope_names[1] not in self._header)
                and get_cost):
            raise ValueError('Unknown or not supported telescope names')

    def _load_plasticc_data(self, raw_data_dir: str, volume: int,
                            snid: int, sample='test', feature_extractor: str = 'bazin'):
        """
        Loads PLAsTiCC dataset files to LightCurve class

        Parameters
        ----------
        raw_data_dir
            Complete path to all PLAsTiCC zenodo files.
        snid: int
            Object id for the transient to be fitted.
        vol: int (optional)
            Index of the original PLAsTiCC zenodo light curve
            files where the photometry for this object is stored.
        sample: str (optional)
            Sample to load, 'train' or 'test'. Default is 'test'.
        """
        light_curve_data = FEATURE_EXTRACTOR_MAPPING[feature_extractor]()
        if sample == 'test':
            file_name = os.path.join(
                    raw_data_dir, self._file_list_dict['test'][volume - 1])

        else:
            file_name = os.path.join(
                    raw_data_dir, self._file_list_dict['train'][0])
        light_curve_data.load_plasticc_lc(file_name, snid)

        return light_curve_data

    def _get_fit_days(self, day: Union[int, None],
                      light_curve_data) -> list:
        """
        Parameters
        ----------
        day
            Day since beginning of survey to be considered.
            If None, fit all days. Default is None.
        light_curve_data
            An instance of LightCurve class
        """
        if day is None:
            if not light_curve_data.photometry.empty:
                min_mjd = min(light_curve_data.photometry['mjd'].values)
                max_mjd = max(light_curve_data.photometry['mjd'].values)
                return list(range(int(min_mjd - self.min_epoch),
                                  int(max_mjd - self.min_epoch) + 1))
        return [day]

    @staticmethod
    def _verify_features_method(feature_extractor: str):
        """
        Verifies if valid dataset name and features method is passed
        Parameters
        ----------
        feature_method
            Feature extraction method, only possibility is 'Bazin'.
        """
        if feature_extractor not in FEATURE_EXTRACTOR_MAPPING:
            raise ValueError(
                'Provided feature extractor method has not been implemented!!')

    def _check_queryable(self, light_curve_data,
                         queryable_criteria: int,
                         days_since_last_observation: int) -> bool:
        """
        Applies check_queryable method of LightCurve class

        Parameters
        ----------
        light_curve_data
            An instance of LightCurve class
        queryable_criteria: [1 or 2]
            Criteria to determine if an obj is queryable.
            1 -> r-band cut on last measured photometric point.
            2 -> last obs was further than a given limit,
                 use Bazin estimate of flux today. Otherwise, use
                 the last observed point.
            Default is 1.
        days_since_last_observation
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
            **kwargs):
        """
        Updates time required to take a spectra calc_exp_time() method of
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
        kwargs
            Any input required by ExpTimeCalc.findexptime function.
        """
        for index in range(self._number_of_telescopes):
            light_curve_data.calc_exp_time(
                telescope_diam=telescope_sizes[index],
                telescope_name=telescope_names[index],
                SNR=spectroscopic_snr, **kwargs
            )

        return light_curve_data

    def _process_current_day(
            self, light_curve_data_day, queryable_criteria: int,
            days_since_last_observation: int, telescope_names: list,
            telescope_sizes: list, spectroscopic_snr: int,
            min_available_points: int = 4, **kwargs):
        """
        Processes data for current day

        Parameters
        ----------
        light_curve_data_day
            An instance of LightCurve class
        queryable_criteria
            Criteria to determine if an obj is queryable.
            1 -> r-band cut on last measured photometric point.
            2 -> last obs was further than a given limit,
                 use Bazin estimate of flux today. Otherwise, use
                 the last observed point.
        days_since_last_observation
            Day since last observation to consider for spectroscopic
            follow-up without the need to extrapolate light curve.
        telescope_names
            Names of the telescopes under consideration for spectroscopy.
            Only used if "get_cost == True".
        telescope_sizes
            Primary mirrors diameters of potential spectroscopic telescopes.
            Only used if "get_cost == True".
        spectroscopic_snr
            SNR required for spectroscopic follow-up.
        kwargs
            Any input required by ExpTimeCalc.findexptime function.
        min_available_points
            minimum number of survived points
        """
        photo_flag = (
                light_curve_data_day.photometry['mjd'].values <= self._today)

        if np.sum(photo_flag) > min_available_points:
            light_curve_data_day.photometry = light_curve_data_day.photometry[
                photo_flag]
            light_curve_data_day.fit_all()

            if (len(light_curve_data_day.features) > 0 and
                    'None' not in light_curve_data_day.features):

                light_curve_data_day.queryable = self._check_queryable(
                    light_curve_data_day, queryable_criteria,
                    days_since_last_observation)
                light_curve_data_day = self._update_queryable_if_get_cost(
                    light_curve_data_day, telescope_names, telescope_sizes,
                    spectroscopic_snr, **kwargs)
                light_curve_data_day.queryable = bool(sum(get_query_flags(
                    light_curve_data_day, telescope_names
                )))

                return light_curve_data_day

        return None

    def _get_features_to_write(
            self, light_curve_data_day,
            get_cost: bool, telescope_names: list) -> list:
        """
        Returns features list to write

        Parameters
        ----------
        light_curve_data_day
            fitted light curve data of current snid
        get_cost
           if cost of taking a spectra is computed
        telescope_names
            Names of the telescopes under consideration for spectroscopy.
            Only used if "get_cost == True".
            Default is ["4m", "8m"].
        """
        light_curve_data_day.sntype = PLASTICC_TARGET_TYPES[
            light_curve_data_day.sncode]
        light_curve_data_day.sample = 'pool'
        features_list = [
            light_curve_data_day.id, light_curve_data_day.redshift,
            light_curve_data_day.sntype, light_curve_data_day.sncode,
            light_curve_data_day.sample, light_curve_data_day.queryable,
            light_curve_data_day.last_mag]

        if get_cost:
            for index in range(self._number_of_telescopes):
                features_list.append(str(
                    light_curve_data_day.exp_time[telescope_names[index]]))
        features_list.extend(light_curve_data_day.features)

        return features_list

    def _update_light_curve_meta_data(self, light_curve_data_day,
                                      snid: int):
        """
        Loads light curve data of the given SNID

        Parameters
        ----------
        light_curve_data_day
            An instance of LightCurve class instance of the data
        snid
            Object id for the transient to be fitted.

        Returns
        -------
        light_curve_data_day
            LightCurve class data for the given SNID
        """
        if light_curve_data_day is not None:
            snid_mask = self.metadata['object_id'].values == snid
            if np.sum(snid_mask) > 0:
                light_curve_data_day.redshift = self.metadata['true_z'].values[
                    snid_mask][0]
                light_curve_data_day.sncode = (
                    self.metadata['true_target'].values[snid_mask][0])
                light_curve_data_day.id = snid

                return light_curve_data_day

        return None

    # TODO: Too many arguments. Refactor and update docs
    def fit_one_lc(self, raw_data_dir: str, snid: int, output_dir: str,
                   vol=None, day=None, queryable_criteria=1,
                   days_since_last_obs=2, get_cost=False, tel_sizes=[4, 8],
                   tel_names=['4m', '8m'], feature_extractor='bazin', spec_SNR=10,
                   time_window=[0, 1095], sample='test', bar=False,
                   **kwargs):
        """
        Fit one light curve throughout a portion of the survey. Save it to file.

        Save results to appropriate file, considering 1 day survey
        evolution.

        Parameters
        ----------
        raw_data_dir: str
            Complete path to all PLAsTiCC zenodo files.
        snid: int
            Object id for the transient to be fitted.
        output_dir:
            Directory to store output time domain files.
        vol: int or None (optional)
            Index of the original PLAsTiCC zenodo light curve
            files where the photometry for this object is stored.
            If None, search for id in all light curve files.
            Default is None.
        day: int or None (optional)
            Day since beginning of survey to be considered.
            If None, fit all days. Default is None.
        queryable_criteria: int [1, 2 or 3] (optional)
            Criteria to determine if an obj is queryable.
            1 -> Cut on last measured photometric point.
            2 -> if last obs was further than days_since_last_obs,
                 use Bazin estimate for today. Otherwise, use
                 the last observed point.
            Default is 1.
        days_since_last_obs: int (optional)
            If there is an observation within these days, use the
            measured value, otherwise estimate current mag.
            Only used if "criteria == 2". Default is 2.
        get_cost: bool (optional)
            If True, calculate cost of taking a spectra in the last
            observed photometric point. Default is False.
        tel_names: list (optional)
            Names of the telescopes under consideration for spectroscopy.
            Only used if "get_cost == True".
            Default is ["4m", "8m"].
        tel_sizes: list (optional)
            Primary mirrors diameters of potential spectroscopic telescopes.
            Only used if "get_cost == True".
            Default is [4, 8].
        feature_extractor: str (optional)
            Feature extraction method.
            Only possibility is 'Bazin'.
        spec_SNR: float (optional)
            SNR required for spectroscopic follow-up. Default is 10.
        time_window: list or None (optional)
            Days of the survey to process, in days since the start of the survey.
            Default is the entire survey = [0, 1095].
        sample: str (optional)
            Sample to load, 'train' or 'test'. Default is 'test'.
        bar: bool (optional)
            If True, show progress. Default is False.
        kwargs: extra parameters (optional)
            Any input required by ExpTimeCalc.findexptime function.
        """
        self._verify_telescope_names(tel_names, get_cost)
        self._verify_features_method(feature_extractor)
        self._number_of_telescopes = len(tel_names)

        light_curve_data = self._load_plasticc_data(
            raw_data_dir, vol, snid, sample, feature_extractor)
        fit_days = np.arange(time_window[0], time_window[1])

        if bar:
            group = progressbar.progressbar(fit_days)
        else:
            group = fit_days

        for day_of_survey in fit_days:

            light_curve_data_day = deepcopy(light_curve_data)
            self._today = day_of_survey + self.min_epoch

            # check number of points to today
            photo_flag = (
                light_curve_data_day.photometry['mjd'].values <= self._today)
            ndays_new = sum(photo_flag)

            if day_of_survey == fit_days[0]:
                ndays = sum(photo_flag)

            # only calculate features if there is a new observed point
            if ndays_new > ndays or day_of_survey == fit_days[0]:
                light_curve_data_day = self._process_current_day(
                    light_curve_data_day, queryable_criteria, days_since_last_obs,
                    tel_names, tel_sizes, spec_SNR, **kwargs)

                light_curve_data_day = self._update_light_curve_meta_data(
                        light_curve_data_day, snid)

            else:
                print("Copying")
                light_curve_data_day = lc_old

            if light_curve_data_day is not None:
                features_to_write = self._get_features_to_write(
                        light_curve_data_day, get_cost, tel_names)
                features_file_name = os.path.join(
                        output_dir, 'day_' + str(day_of_survey) + '.csv')
                with open(features_file_name, 'a') as plasticc_features_file:
                    plasticc_features_file.write(
                            ','.join(str(each_feature) for each_feature
                                     in features_to_write) + '\n')

                    if not bar:
                        print('wrote epoch: ', day_of_survey, ' snid: ', snid)

                # store this to compare to next day
                ndays = light_curve_data_day.photometry.shape[0]

            lc_old = deepcopy(light_curve_data_day)

    def _maybe_create_feature_to_file(self, features_file_name: str):
        """
        Creates feature file with header if it doesn't exist
        Parameters
        ----------
        features_file_name
            feature file name
        """
        if not os.path.isfile(features_file_name):
            with open(features_file_name, 'w') as features_file:
                features_file.write(' '.join(self._header) + '\n')

    def _maybe_create_daily_feature_files(
            self, time_window: list, output_dir: str, get_cost: bool, 
            ask_user=True):
        """
        Creates daily feature files for the specified time window
        Parameters
        ----------
        time_window
            Days of the survey to process, in days since the start of the survey.
            Default is the entire survey = [0, 1095].
        output_dir
            Output directory to save feature files
        get_cost
           if cost of taking a spectra is computed
        ask_user: bool (optional)
           If True, ask user if daily file should be created. Default is True.
        """
        if ask_user:
            user_input = input("Are you sure want to create new daily files?(yes/no): ")
            if user_input.lower() == "yes":
                for day_of_survey in range(time_window[0], time_window[1]):
                    self.create_daily_file(output_dir=output_dir,
                                           day=day_of_survey, get_cost=get_cost)
            elif user_input.lower() == "no":
                logging.info("Not creating new daily feature files.")
            else:
                raise ValueError("Unknown input! Please specify yes or no.")
        else:
            for day_of_survey in range(time_window[0], time_window[1]):
                    self.create_daily_file(output_dir=output_dir,
                                           day=day_of_survey, get_cost=get_cost)
            logging.info("Daily feature files created, warning suppressed by user.")
            

    def fit_all_snids_lc(
            self, raw_data_dir: str, snids: np.ndarray, output_dir: str,
            vol: int = None, queryable_criteria: int = 1,
            days_since_last_obs: int = 2, get_cost: bool = False,
            tel_sizes: list = [4, 8], tel_names: list = ['4m', '8m'],
            feature_extractor: str = 'bazin', spec_SNR: int = 10,
            time_window: list = [0, 1095], sample: str = 'test',
            number_of_processors: int = 1, create_daily_files: bool = False,
            ask_user: bool = True, **kwargs):
        """
        Fits light curves for all the available snids for the time period
         provided in time window and saves features to individual day features
         file

        Parameters
        ----------
        raw_data_dir
            Complete path to all PLAsTiCC zenodo files.
        snids
            Object ids for the transient to be fitted.
        output_dir
            Directory to store output time domain files.
        vol
            Index of the original PLAsTiCC zenodo light curve
            files where the photometry for this object is stored.
            If None, search for id in all light curve files.
            Default is None.
        queryable_criteria
            Criteria to determine if an obj is queryable.
            1 -> Cut on last measured photometric point.
            2 -> if last obs was further than days_since_last_obs,
                 use Bazin estimate for today. Otherwise, use
                 the last observed point.
            Default is 1.
        days_since_last_obs
            If there is an observation within these days, use the
            measured value, otherwise estimate current mag.
            Only used if "criteria == 2". Default is 2.
        get_cost
            If True, calculate cost of taking a spectra in the last
            observed photometric point. Default is False.
        tel_sizes
            Primary mirrors diameters of potential spectroscopic telescopes.
            Only used if "get_cost == True".
            Default is [4, 8].
        tel_names
            Names of the telescopes under consideration for spectroscopy.
            Only used if "get_cost == True".
            Default is ["4m", "8m"].
        feature_extractor
            Feature extraction method.
        spec_SNR
            SNR required for spectroscopic follow-up. Default is 10.
        time_window
            Days of the survey to process, in days since the start of the survey.
            Default is the entire survey = [0, 1095].
        sample
            Sample to load, 'train' or 'test'. Default is 'test'.
        number_of_processors
            Number of cpu processes to use.
        create_daily_files
            if feature files for all the days should be created
            before startign the fitting process
        ask_user: bool (optional)
           If True, ask user if daily file should be created. Default is True.
        kwargs
            Any input required by ExpTimeCalc.findexptime function.
        """
        self._set_header(get_cost=get_cost)
        self._verify_telescope_names(tel_names, get_cost)
        self._verify_features_method(feature_extractor)
        self._number_of_telescopes = len(tel_names)
        self._kwargs = kwargs
        if create_daily_files:
            self._maybe_create_daily_feature_files(
                time_window, output_dir, get_cost, ask_user=ask_user)

        for day_of_survey in range(time_window[0], time_window[1]):
            # Load previous day features if available
            self._previous_day_features, self._previous_day_index_mapping = (
                _load_previous_day_features(day_of_survey, output_dir))
            features_file_name = os.path.join(
                output_dir, 'day_' + str(day_of_survey) + '.csv')
            number_of_points_mapping_file_name = os.path.join(
                output_dir, "snid_number_of_points.json")
            # Load snids and number of observed points till previous day fit mapping
            previous_day_number_of_points = _load_snids_to_points_mapping(
                number_of_points_mapping_file_name)
            multi_process = multiprocessing.Pool(number_of_processors)
            logging.info("Generting features for day %s", day_of_survey)
            iterator_list = zip(
                snids, repeat(raw_data_dir), repeat(vol), repeat(day_of_survey),
                repeat(sample), repeat(get_cost), repeat(tel_names),
                repeat(queryable_criteria), repeat(days_since_last_obs),
                repeat(tel_sizes), repeat(spec_SNR),
                repeat(previous_day_number_of_points),
            )
            self._maybe_create_feature_to_file(features_file_name)
            with open(features_file_name, 'a') as plasticc_features_file:
                for features_to_write, number_of_observation_points in multi_process.starmap(
                        self._process_each_snid, iterator_list):
                    previous_day_number_of_points.update(number_of_observation_points)
                    if features_to_write is not None:
                        plasticc_features_file.write(features_to_write)
            with open(number_of_points_mapping_file_name, "w") as json_file:
                json.dump(previous_day_number_of_points, json_file)

    def _process_each_snid(
            self, snid: int, raw_data_dir: str, vol: int, day_of_survey: int,
            sample: str, get_cost, telescope_names: list,
            queryable_criteria: int, days_since_last_obs: int,
            telescope_sizes: list, spectroscopic_snr: int,
            previous_day_number_of_points: dict) -> Tuple[Union[str, None],
        Union[int, None]]:
        """
        Fits snid light curve for the day of survey and returns features

        Parameters
        ----------
        snid
            Object id for the transient to be fitted.
        raw_data_dir
            Complete path to all PLAsTiCC zenodo files.
        vol
            Index of the original PLAsTiCC zenodo light curve
            files where the photometry for this object is stored.
            If None, search for id in all light curve files.
            Default is None.
        day_of_survey
            current day of survey
        sample
            Sample to load, 'train' or 'test'. Default is 'test'.
        get_cost
            If True, calculate cost of taking a spectra in the last
            observed photometric point. Default is False.
        telescope_names
            Names of the telescopes under consideration for spectroscopy.
            Only used if "get_cost == True".
            Default is ["4m", "8m"].
        queryable_criteria
            Criteria to determine if an obj is queryable.
            1 -> Cut on last measured photometric point.
            2 -> if last obs was further than days_since_last_obs,
                 use Bazin estimate for today. Otherwise, use
                 the last observed point.
            Default is 1.
        days_since_last_obs
            If there is an observation within these days, use the
            measured value, otherwise estimate current mag.
            Only used if "criteria == 2". Default is 2.
        telescope_sizes
            Primary mirrors diameters of potential spectroscopic telescopes.
            Only used if "get_cost == True".
            Default is [4, 8].
        spectroscopic_snr
            SNR required for spectroscopic follow-up. Default is 10.
        previous_day_number_of_points
            mapping of snid and total number of observed points till previous day
        Returns
        -------
        features_to_write
            light curve features of given snid
        number_of_observation_points
            number of observed points
        """

        light_curve_data = self._load_plasticc_data(raw_data_dir, vol, snid, sample)
        self._today = day_of_survey + self.min_epoch
        light_curve_data_day = deepcopy(light_curve_data)

        ndays_new = sum((
                light_curve_data_day.photometry['mjd'].values <= self._today))
        number_of_observation_points = {str(snid): int(ndays_new)}
        if (snid in previous_day_number_of_points and
                previous_day_number_of_points[snid] > ndays_new) or (
                snid not in previous_day_number_of_points):
            light_curve_data_day = self._process_current_day(
                light_curve_data_day, queryable_criteria, days_since_last_obs,
                telescope_names, telescope_sizes, spectroscopic_snr, **self._kwargs)

            light_curve_data_day = self._update_light_curve_meta_data(
                light_curve_data_day, snid)
        else:
            snid = str(snid)
            if snid in self._previous_day_index_mapping:
                return (self._previous_day_features[
                            self._previous_day_index_mapping[snid]],
                        number_of_observation_points)
        if light_curve_data_day is not None:
            features_to_write = self._get_features_to_write(
                light_curve_data_day, get_cost, telescope_names)
            return (','.join(str(each_feature) for each_feature
                             in features_to_write) + '\n',
                    number_of_observation_points)
        return None, number_of_observation_points


def _load_previous_day_features(
        day_of_survey: int, output_dir: str,
        file_name_prefix: str = "day_") -> Tuple[Union[None, list], dict]:
    """
    Loads previous day features file to list and generates snid to its index
     in the list mapping, which can be used to copy the features if no new observing
     points are available in the light curve

    Parameters
    ----------
    day_of_survey
        day of survey
    output_dir
        Directory to store output time domain files.
    file_name_prefix
        features file name prefix string
    Returns
    -------
    previous_day_features
        loaded previous day features
    previous_day_index_mapping
        snid to its index in the features list mapping
    """

    previous_day_file_name = file_name_prefix + str(day_of_survey - 1) + '.csv'
    previous_day_file_name = os.path.join(output_dir, previous_day_file_name)
    if (day_of_survey < 2) or (not os.path.isfile(previous_day_file_name)):
        return None, {}
    with open(previous_day_file_name, 'r') as file:
        previous_day_features = file.readlines()
    previous_day_index_mapping = {
        line.split(" ", 1)[0]: index for index, line in enumerate(
            previous_day_features)}

    return previous_day_features, previous_day_index_mapping


def _load_snids_to_points_mapping(file_name: str) -> dict:
    """
    Loads mapping of snid and total number of observed points till
     previous day file
    Parameters
    ----------
    file_name
        mapping file_name "snid_number_of_points.json"
    Returns
    -------
    mapping
        snid to number of observed points mapping
    """
    if os.path.isfile(file_name):
        with open(file_name, "r") as json_file:
            return json.load(json_file)
    return {}
