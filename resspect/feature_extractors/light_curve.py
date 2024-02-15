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
from typing import Tuple
import warnings

import numpy as np
import pandas as pd

from resspect.exposure_time_calculator import ExpTimeCalc
from resspect.lightcurves_utils import read_file
from resspect.lightcurves_utils import load_snpcc_photometry_df
from resspect.lightcurves_utils import get_photometry_with_id_name_and_snid
from resspect.lightcurves_utils import read_plasticc_full_photometry_data
from resspect.lightcurves_utils import load_plasticc_photometry_df
from resspect.lightcurves_utils import get_snpcc_sntype


warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO)

__all__ = ['LightCurve']


class LightCurve:
    """ Light Curve object, holding meta and photometric data.

    Attributes
    ----------
    features_names: list
        List of names of the feature extraction parameters.
    features: list
        List with the 5 best-fit feature extraction parameters in all filters.
        Concatenated from blue to red.
    dataset_name: str
        Name of the survey or data set being analyzed.
    exp_time: dict
        Exposure time required to take a spectra.
        Keywords indicate telescope e.g.['4m', '8m'].
    filters: list
        List of broad band filters.
    full_photometry: pd.DataFrame
        Photometry for a set of light curves read from file.
    id: int
        SN identification number.
    id_name:
        Column name of object identifier.
    last_mag: float
        r-band magnitude of last observed epoch.
    photometry: pd.DataFrame
        Photometry information.
        Minimum keys --> [mjd, band, flux, fluxerr].
    redshift: float
        Redshift
    sample: str
        Original sample to which this light curve is assigned.
    sim_peakmag: np.array
        Simulated peak magnitude in each filter.
    sim_pkmjd: float
        Simulated day of maximum, observer frame.
    sncode: int
        Number identifying the SN model used in the simulation.
    sntype: str
        General classification, possibilities are: Ia, II or Ibc.
    unique_ids: str or array
        List of unique ids available in the photometry file.
        Only used for PLAsTiCC data.

    Methods
    -------
    calc_exp_time(telescope_diam: float, SNR: float, telescope_name: str)
        Calculates time required to take a spectra in the last obs epoch.
    check_queryable(mjd: float, r_lim: float)
        Check if this light can be queried in a given day.
    conv_flux_mag(flux: np.array)
        Convert positive flux into magnitude.
    evaluate_bazin(param: list, time: np.array) -> np.array
        Evaluate the Bazin function given parameter values.
    load_snpcc_lc(path_to_data: str)
        Reads header and photometric information for 1 light curve.
    load_plasticc_lc(photo_file: str, snid: int)
    	Load photometric information for 1 PLAsTiCC light curve.
    load_resspect_lc(photo_file: str, snid: int)
    	Load photometric information for 1 RESSPECT light curve.
    plot_bump_fit(save: bool, show: bool, output_file: srt)
        Plot photometric points and Bump fitted curve.

    """

    def __init__(self):
        self.queryable = None
        self.features = []
        #self.features_names = ['p1', 'p2', 'p3', 'time_shift', 'max_flux']
        self.dataset_name = ' '
        self.exp_time = {}
        self.filters = []
        self.full_photometry = pd.DataFrame()
        self.id = 0
        self.id_name = None
        self.last_mag = None
        self.photometry = pd.DataFrame()
        self.redshift = 0
        self.sample = ' '
        self.sim_peakmag = []
        self.sim_pkmjd = None
        self.sncode = 0
        self.sntype = ' '

    def _get_snpcc_photometry_raw_and_header(
            self, lc_data: np.ndarray,
            sntype_test_value: str = "-9") -> Tuple[np.ndarray, list]:
        """
        Reads SNPCC photometry raw and header data

        Parameters
        ----------
        lc_data
            SNPCC light curve data
        sntype_test_value
            test sample SNTYPE value
        """
        photometry_raw = []
        header = []
        for each_row in lc_data:
            name = each_row[0]
            value = each_row[1]
            if name == 'SNID:':
                self.id = int(value)
                self.id_name = 'SNID'
            elif name == 'SNTYPE:':
                self.sample = 'test' if value == sntype_test_value else 'train'
            elif name == 'SIM_REDSHIFT:':
                self.redshift = float(value)
            elif name == 'SIM_NON1a:':
                self.sncode = value
                self.sntype = get_snpcc_sntype(int(value))
            elif name == 'VARLIST:':
                header = each_row[1:]
            elif name == 'OBS:':
                photometry_raw.append(np.array(each_row[1:]))
            elif name == 'SIM_PEAKMAG:':
                self.sim_peakmag = np.array(each_row[1:5]).astype(float)
            elif name == 'SIM_PEAKMJD:':
                self.sim_pkmjd = float(value)
        return np.array(photometry_raw), header

    def load_snpcc_lc(self, path_to_data: str):
        """Reads one LC from SNPCC data.

        Populates the attributes: dataset_name, id, sample, redshift, sncode,
        sntype, photometry, sim_peakmag and sim_pkmjd.

        Parameters
        ---------
        path_to_data: str
            Path to text file with data from a single SN.
        """
        self.dataset_name = 'SNPCC'
        self.filters = ['g', 'r', 'i', 'z']

        lc_data = np.array(read_file(path_to_data), dtype=object)
        photometry_raw, header = self._get_snpcc_photometry_raw_and_header(
            lc_data)

        if photometry_raw.size > 0:
            self.photometry = load_snpcc_photometry_df(photometry_raw, header)

    def load_plasticc_lc(self, photo_file: str, snid: int):
        """
        Return 1 light curve from PLAsTiCC simulations.

        Parameters
        ----------
        photo_file: str
            Complete path to light curve file.
        snid: int
            Identification number for the desired light curve.
        """
        self.dataset_name = 'PLAsTiCC'
        self.filters = ['u', 'g', 'r', 'i', 'z', 'Y']
        self.id = snid

        if self.full_photometry.empty:
            self.full_photometry = read_plasticc_full_photometry_data(
                photo_file)

        id_names_list = ['object_id', 'SNID', 'snid']

        filtered_photometry, self.id_name = (
            get_photometry_with_id_name_and_snid(
                self.full_photometry, id_names_list, snid))

        filter_mapping_dict = {
            0: 'u', 1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'Y'
        }

        if not filtered_photometry.empty:
            self.photometry = load_plasticc_photometry_df(
                filtered_photometry, filter_mapping_dict)

    @staticmethod
    def conv_flux_mag(flux, zpt: float = 27.5):
        """Convert FLUXCAL to magnitudes.

        Parameters
        ----------
        flux: list or np.array
            Values of flux to be converted into mag.
        zpt: float (optional)
            Zero point. Default is 27.5 (from SNANA).

        Returns
        -------
        mag: list or np.array
            Magnitude values. If flux < 1e-5 returns 9999.
        """

        if not isinstance(flux[0], float):
            flux = np.array([item[0] for item in flux])
        mag = [zpt - 2.5 * np.log10(f) if f > 1e-5 else 9999 for f in flux]
        return np.array(mag)

    def check_queryable(self, mjd: float, filter_lim: float, criteria: int =1,
                        days_since_last_obs=2, feature_method: str = 'Bazin',
                        filter_cut='r'):
        """Check if this object can be queried in a given day.

        Parameters
        ----------
        mjd: float
            MJD where the query will take place.
        filter_lim: float
            Magnitude limit below which query is possible.
        criteria: int [1, 2 or 3] (optional)
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
        feature_method: str (optional)
            Feature extraction method. Only 'Bazin' is implemented.
            Default is 'Bazin'.
        filter_cut: str (optional)
            Band in which cut is applied. Default is 'r'.

        Returns
        -------
        bool
            True if current magnitude lower than "filter_lim".
        """

        # create photo flag
        photo_flag = self.photometry['mjd'].values <= mjd
        rband_flag = self.photometry['band'].values == filter_cut
        surv_flag = np.logical_and(photo_flag, rband_flag)

        if criteria == 1 and sum(surv_flag):
            if 'MAG' in self.photometry.keys():
                # check surviving photometry
                self.last_mag = self.photometry['MAG'].values[surv_flag][-1]

            else:
                surv_flux = self.photometry['flux'].values[surv_flag]
                self.last_mag = self.conv_flux_mag([surv_flux[-1]])[0]

        elif criteria == 2 and sum(surv_flag):
            # check if there is an observation recently
            surv_mjd = self.photometry['mjd'].values[surv_flag]
            gap = mjd - surv_mjd[-1]

            if gap <= days_since_last_obs:
                if 'MAG' in self.photometry.keys():
                    # check surviving photometry
                    self.last_mag = \
                        self.photometry['MAG'].values[surv_flag][-1]

                else:
                    surv_flux = self.photometry['flux'].values[surv_flag]
                    self.last_mag = self.conv_flux_mag([surv_flux[-1]])[0]

            elif feature_method == 'Bazin':
                # get first day of observation in this filter
                mjd_min = min(self.photometry['mjd'].values[surv_flag])

                # estimate flux based on Bazin function
                fitted_flux = self.evaluate([mjd - mjd_min])[filter_cut][0]
                self.last_mag = self.conv_flux_mag([fitted_flux])[0]

            else:
                raise ValueError('Only "Bazin" and "malanchev" features are implemented!')

        elif sum(surv_flag):
            raise ValueError('Criteria needs to be "1" or "2". \n ' + \
                             'See docstring for further info.')
        elif not sum(surv_flag):
            return False

        if self.last_mag <= filter_lim:
            return True
        else:
            return False

    def calc_exp_time(self, telescope_diam: float, SNR: float,
                      telescope_name: str, **kwargs):
        """Calculates time required to take a spectra in the last obs epoch.

        Populates attribute exp_time.

        Parameters
        ----------
        SNR: float
            Required SNR.
        telescope_diam: float
            Diameter of primary mirror for spectroscopic telescope in meters.
        telescope_name: str
            Identification for telescope.
        kwargs: extra parameters
            Any input required by ExpTimeCalc.findexptime function.

        Returns
        -------
        exp_time: float
            Exposure time require for taking a spectra in seconds.
        """

        # check if last magnitude was calculated
        if self.last_mag is None:
            raise ValueError('Magnitude at last epoch not calculated.\n' + \
                             'Run the check_queryable function.')

        etc = ExpTimeCalc()
        etc.diameter = telescope_diam
        exp_time = etc.findexptime(SNRin=SNR, mag=self.last_mag, **kwargs)

        if 60 < exp_time < 7200 and self.last_mag < 30:
            self.exp_time[telescope_name] = exp_time
            return exp_time
        else:
            self.exp_time[telescope_name] = 9999
            return 9999

    def clear_data(self):
        """ Reset to default values """
        self.photometry = []
        self.redshift = None
        self.sntype = None
        self.sncode = None
        self.sample = None
