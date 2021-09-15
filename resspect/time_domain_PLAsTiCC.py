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


import logging
import os
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import progressbar
from resspect import LightCurve
from resspect.lightcurves_utils import BAZIN_HEADERS
from resspect.lightcurves_utils import get_query_flags
from resspect.lightcurves_utils import maybe_create_directory
from resspect.lightcurves_utils import PLASTICC_TARGET_TYPES
from resspect.lightcurves_utils import read_plasticc_full_photometry_data


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
        self._bazin_header = None  # type: list
        self._file_list_dict = {}  # type: dict
        self._today = None  # type: int
        self._number_of_telescopes = None  # type: int
        self.build()

    def build(self):
        self._last_file_index = 11
        self._file_list_dict['test'] = [
            'plasticc_test_lightcurves_' + str(i).zfill(2) + '.csv.gz'
            for i in range(1, self._last_file_index + 1)]
        self._file_list_dict['train'] = ['plasticc_train_lightcurves.csv.gz']

    def create_daily_file(self, output_dir: str, day: int,
                          header: str = 'Bazin', get_cost: bool = False):
        """
        Create one file for a given day of the survey.

        The file contains only header for the features file.

        Parameters
        ----------
        output_dir: str
            Complete path to output directory.
        day: int
            Day passed since the beginning of the survey.
        header: str (optional)
            List of elements to be added to the header.
            Separate by 1 space.
            Default option uses header for Bazin features file.
        get_cost: bool (optional)
            If True, calculate cost of taking a spectra in the last
            observed photometric point. Default is False.
        """
        maybe_create_directory(output_dir)
        self._features_file_name = os.path.join(
            output_dir, 'day_' + str(day) + '.dat')
        logging.info('Creating features file')
        with open(self._features_file_name, 'w') as features_file:
            if header == 'Bazin':
                self._bazin_header = BAZIN_HEADERS['plasticc_header']
                if get_cost:
                    self._bazin_header = BAZIN_HEADERS[
                        'plasticc_header_with_cost']
            else:
                raise ValueError('Only Bazin headers are supported')
            features_file.write(' '.join(self._bazin_header) + '\n')

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
        for day_of_survey in range(1, self.max_epoch - self.min_epoch):
            self.create_daily_file(output_dir=output_dir,
                                   day=day_of_survey, get_cost=get_cost)

    def read_metadata(self, path_to_data_dir: str, classes: list,
                      field: str = 'DDF', meta_data_file_name:
                      str = 'plasticc_test_metadata.csv'):
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
            ddf_mask = meta_data_raw.ddf_bool.astype(np.bool).values
            if field == 'DDF':
                filter_mask = np.logical_and(ddf_mask, classes_mask)
            else:
                filter_mask = np.logical_and(~ddf_mask, classes_mask)
                print(sum(filter_mask))
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
        if (('cost_' + telescope_names[0] not in self._bazin_header or
            'cost_' + telescope_names[1] not in self._bazin_header)
                and get_cost):
            raise ValueError('Unknown or not supported telescope names')

    def _load_plasticc_data(self, raw_data_dir: str, volume: Union[int, None],
                            snid: int) -> LightCurve:
        """
        loads PLAsTiCC dataset files to LightCurve class
        Parameters
        ----------
        raw_data_dir
            Complete path to all PLAsTiCC zenodo files.
        snid: int
            Object id for the transient to be fitted.
        vol: int or None (optional)
            Index of the original PLAsTiCC zenodo light curve
            files where the photometry for this object is stored.
            If None, search for id in all light curve files.
            Default is None.
        """
        light_curve_data = LightCurve()
        if volume is None:
            volume = 1
            while volume < self._last_file_index:
                volume += 1
                file_name = os.path.join(
                    raw_data_dir, self._file_list_dict['test'][volume - 1])
                light_curve_data.load_plasticc_lc(file_name, snid)
        else:
            file_name = os.path.join(
                raw_data_dir, self._file_list_dict['test'][volume - 1])
            light_curve_data.load_plasticc_lc(file_name, snid)
        return light_curve_data

    def _get_fit_days(self, day: Union[int, None],
                      light_curve_data: LightCurve) -> list:
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
    def _verify_features_method(feature_method: str):
        """
        Verifies if valid dataset name and features method is passed
        Parameters
        ----------
        feature_method
            Feature extraction method, only possibility is 'Bazin'.
        """
        if feature_method != 'Bazin':
            raise ValueError('Only Bazin features are implemented!!')

    def _check_queryable(self, light_curve_data: LightCurve,
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
            self, light_curve_data: LightCurve, telescope_names: list,
            telescope_sizes: list, spectroscopic_snr: int,
            kwargs: dict) -> LightCurve:
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
            self, light_curve_data_day: LightCurve, queryable_criteria: int,
            days_since_last_observation: int, telescope_names: list,
            telescope_sizes: list, spectroscopic_snr: int, kwargs: dict,
            min_available_points: int = 4):
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
            light_curve_data_day.fit_bazin_all()
            if (len(light_curve_data_day.bazin_features) > 0 and
                    'None' not in light_curve_data_day.bazin_features):
                light_curve_data_day.queryable = self._check_queryable(
                    light_curve_data_day, queryable_criteria,
                    days_since_last_observation)
                light_curve_data_day = self._update_queryable_if_get_cost(
                    light_curve_data_day, telescope_names, telescope_sizes,
                    spectroscopic_snr, kwargs)
                light_curve_data_day.queryable = bool(sum(get_query_flags(
                    light_curve_data_day, telescope_names
                )))
                return light_curve_data_day
        return None

    def _get_features_to_write(
            self, light_curve_data_day: LightCurve,
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
        features_list.extend(light_curve_data_day.bazin_features)
        return features_list

    def _update_light_curve_meta_data(self, light_curve_data_day: LightCurve,
                                      snid: int) -> Union[LightCurve, None]:
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
                   tel_names=['4m', '8m'], feature_method='Bazin', spec_SNR=10,
                   **kwargs):
        """
        Fit one light curve throughout the entire survey.

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
        feature_method: str (optional)
            Feature extraction method.
            Only possibility is 'Bazin'.
        spec_SNR: float (optional)
            SNR required for spectroscopic follow-up. Default is 10.
        kwargs: extra parameters
            Any input required by ExpTimeCalc.findexptime function.
        """
        self._verify_telescope_names(tel_names, get_cost)
        self._verify_features_method(feature_method)
        self._number_of_telescopes = len(tel_names)
        light_curve_data = self._load_plasticc_data(raw_data_dir, vol, snid)
        fit_days = self._get_fit_days(day, light_curve_data)
        for day_of_survey in progressbar.progressbar(fit_days):
            light_curve_data_day = deepcopy(light_curve_data)
            self._today = day_of_survey + self.min_epoch
            light_curve_data_day = self._process_current_day(
                light_curve_data_day, queryable_criteria, days_since_last_obs,
                tel_names, tel_sizes, spec_SNR, kwargs)
            light_curve_data_day = self._update_light_curve_meta_data(
                light_curve_data_day, snid)
            if light_curve_data_day is not None:
                features_to_write = self._get_features_to_write(
                    light_curve_data_day, get_cost, tel_names)
                features_file_name = os.path.join(
                    output_dir, 'day_' + str(day_of_survey) + '.dat')
                with open(features_file_name, 'w') as plasticc_features_file:
                    plasticc_features_file.write(
                        ' '.join(self._bazin_header) + '\n')
                    plasticc_features_file.write(
                        ' '.join(str(each_feature) for each_feature
                                 in features_to_write) + '\n')
