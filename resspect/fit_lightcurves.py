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

import multiprocessing
import logging
import os
from copy import copy
from itertools import repeat
from typing import IO
from typing import Tuple
from typing import Union
import warnings

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from resspect.bazin import bazin
from resspect.bazin import fit_scipy
from resspect.bump import bump
from resspect.bump import fit_bump
from resspect.exposure_time_calculator import ExpTimeCalc
from resspect.lightcurves_utils import read_file
from resspect.lightcurves_utils import get_resspect_header_data
from resspect.lightcurves_utils import load_snpcc_photometry_df
from resspect.lightcurves_utils import get_photometry_with_id_name_and_snid
from resspect.lightcurves_utils import read_plasticc_full_photometry_data
from resspect.lightcurves_utils import load_plasticc_photometry_df
from resspect.lightcurves_utils import read_resspect_full_photometry_data
from resspect.lightcurves_utils import insert_band_column_to_resspect_df
from resspect.lightcurves_utils import load_resspect_photometry_df
from resspect.lightcurves_utils import get_snpcc_sntype
from resspect.lightcurves_utils import SNPCC_FEATURES_HEADER
from resspect.lightcurves_utils import find_available_key_name_in_header
from resspect.lightcurves_utils import PLASTICC_TARGET_TYPES
from resspect.lightcurves_utils import PLASTICC_RESSPECT_FEATURES_HEADER

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO)

__all__ = ['LightCurve', 'fit_snpcc_bazin', 'fit_resspect_bazin',
           'fit_plasticc_bazin']


class LightCurve:
    """ Light Curve object, holding meta and photometric data.

    Attributes
    ----------
    bazin_features_names: list
        List of names of the Bazin function parameters.
    bazin_features: list
        List with the 5 best-fit Bazin parameters in all filters.
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
    fit_bazin(band: str) -> list
        Calculates best-fit parameters from the Bazin function in 1 filter.
    fit_bazin_all()
        Calculates  best-fit parameters from the Bazin func for all filters.
    plot_bazin_fit(save: bool, show: bool, output_file: srt)
        Plot photometric points and Bazin fitted curve.
    fit_bump(band: str) -> list
        Calculates best-fit parameters from the Bump function in 1 filter.
    fit_bump_all()
        Calculates  best-fit parameters from the Bump func for all filters.
    plot_bump_fit(save: bool, show: bool, output_file: srt)
        Plot photometric points and Bump fitted curve.

    Examples
    --------

    ##### for RESSPECT and PLAsTiCC light curves it is necessary to
    ##### input the object identification for dealing with 1 light curve

    >>> import io
    >>> import pandas as pd
    >>> import tarfile

    >>> from resspect import LightCurve

    # path to header file
    >>> path_to_header = '~/RESSPECT_PERFECT_V2_TRAIN_HEADER.tar.gz'

    # openning '.tar.gz' files requires some juggling ...
    >>> tar = tarfile.open(path_to_header, 'r:gz')
    >>> fname = tar.getmembers()[0]
    >>> content = tar.extractfile(fname).read()
    >>> header = pd.read_csv(io.BytesIO(content))
    >>> tar.close()

    # choose one object
    >>> snid = header['objid'].values[4]

    # once you have the identification you can use this class
    >>> path_to_lightcurves = '~/RESSPECT_PERFECT_V2_TRAIN_LIGHTCURVES.tar.gz'

    >>> lc = LightCurve()                        # create light curve instance
    >>> lc.load_snpcc_lc(path_to_lightcurves)    # read data
    >>> lc.photometry
             mjd band       flux   fluxerr         SNR
    0    53214.0    u   0.165249  0.142422    1.160276
    1    53214.0    g  -0.041531  0.141841   -0.292803
    ..       ...  ...        ...       ...         ...
    472  53370.0    z  68.645930  0.297934  230.406460
    473  53370.0    Y  63.254270  0.288744  219.067050

    >>> lc.fit_bazin_all()               # perform Bazin fit in all filters
    >>> lc.bazin_features                # display Bazin parameters
    [198.63302952843623, -9.38297128588733, 43.99971014717201,
    ... ...
    -1.546372806815066]

    for fitting the entire sample ...

    >>> output_file = 'RESSPECT_PERFECT_TRAIN.DAT'
    >>> fit_resspect_bazin(path_to_lightcurves, path_to_header,
                           output_file, sample='train')
    """

    def __init__(self):
        self.queryable = None
        self.bazin_features = []
        self.bazin_features_names = ['a', 'b', 't0', 'tfall', 'trise']
        self.bump_features = []
        self.bump_features_names = ['p1', 'p2', 'p3', 'time_shift', 'max_flux']
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

    def load_resspect_lc(self, photo_file: str, snid: int):
        """
        Return 1 light curve from RESSPECT simulations.

        Parameters
        ----------
        photo_file: str
            Complete path to light curves file.
        snid: int
            Identification number for the desired light curve.
        """

        self.dataset_name = 'RESSPECT'
        self.filters = ['u', 'g', 'r', 'i', 'z', 'Y']
        self.id = snid

        if self.full_photometry.empty:
            _, self.full_photometry = read_resspect_full_photometry_data(
                photo_file)
        id_names_list = ['SNID', 'snid', 'objid', 'id']
        filtered_photometry, self.id_name = (
            get_photometry_with_id_name_and_snid(
                self.full_photometry, id_names_list, snid))

        if not filtered_photometry.empty:
            filtered_photometry = insert_band_column_to_resspect_df(
                filtered_photometry, self.filters)
            self.photometry = load_resspect_photometry_df(filtered_photometry)

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

    def conv_flux_mag(self, flux, zpt=27.5):
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

    def check_queryable(self, mjd: float, filter_lim: float, criteria=1,
                        days_since_last_obs=2, feature_method='Bazin',
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
                fitted_flux = self.evaluate_bazin([mjd - mjd_min])[filter_cut][0]
                self.last_mag = self.conv_flux_mag([fitted_flux])[0]

            else:
                raise ValueError('Only "Bazin" features are implemented!')

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
        if self.last_mag == None:
            raise ValueError('Magnitude at last epoch not calculated.\n' + \
                             'Run the check_queryable function.')

        etc = ExpTimeCalc()
        etc.diameter = telescope_diam
        exp_time = etc.findexptime(SNRin=SNR, mag=self.last_mag, **kwargs)

        if exp_time > 60 and exp_time < 7200 and self.last_mag < 30:
            self.exp_time[telescope_name] = exp_time
            return exp_time
        else:
            self.exp_time[telescope_name] = 9999
            return 9999

    def fit_bazin(self, band: str) -> np.ndarray:
        """Extract Bazin features for one filter.

        Parameters
        ----------
        band: str
            Choice of broad band filter

        Returns
        -------
        bazin_param: np.ndarray
            Best fit parameters for the Bazin function:
            [a, b, t0, tfall, trise].
        """

        # build filter flag
        band_indices = self.photometry['band'] == band
        if not sum(band_indices) > (len(self.bazin_features_names) - 1):
            return np.array([])

        # get info for this filter
        time = self.photometry['mjd'].values[band_indices]
        flux = self.photometry['flux'].values[band_indices]
        fluxerr = self.photometry['fluxerr'].values[band_indices]

        # fit Bazin function
        bazin_param = fit_scipy(time - time[0], flux, fluxerr)

        return bazin_param

    def evaluate_bazin(self, time: np.array):
        """Evaluate the Bazin function given parameter values.

        Parameters
        ----------
        time: np.array or list
            Time since first light curve observation.

        Returns
        -------
        dict
            Value of the Bazin flux in each required time per filter.
        """
        # store flux values and starting points
        flux = {}

        for k in range(len(self.filters)):
            # store flux values per filter
            flux[self.filters[k]] = []

            # check if Bazin features exist
            if 'None' not in self.bazin_features[k * 5 : (k + 1) * 5]:
                for item in time:
                    flux[self.filters[k]].append(\
                           bazin(item, self.bazin_features[0 + k * 5],
                                 self.bazin_features[1 + k * 5],
                                 self.bazin_features[2 + k * 5],
                                 self.bazin_features[3 + k * 5],
                                 self.bazin_features[4 + k * 5]))
            else:
                flux[self.filters[k]].append(None)

        return flux

    def fit_bazin_all(self):
        """
        Perform Bazin fit for all filters independently and concatenate results.
        Populates the attributes: bazin_features.
        """
        default_bazin_features = ['None'] * len(self.bazin_features_names)

        if self.photometry.shape[0] < 1:
            self.bazin_features = ['None'] * len(self.bazin_features_names) * len(self.filters)

        elif 'None' not in self.bazin_features:
            self.bazin_features = []
            for each_band in self.filters:
                best_fit = self.fit_bazin(band=each_band)
                if (best_fit.size > 0) and (not np.isnan(np.sum(best_fit))):
                    self.bazin_features.extend(best_fit.tolist())
                else:
                    self.bazin_features.extend(default_bazin_features)
        else:
            self.bazin_features.extend(default_bazin_features)
            
    def fit_bump(self, band: str) -> np.ndarray:
        """Extract Bump features for one filter.

        Parameters
        ----------
        band: str
            Choice of broad band filter

        Returns
        -------
        bump_param: np.ndarray
            Best fit parameters for the Bazin function:
            [p1, p2, p3, time_shift, max_flux].
        """

        # build filter flag
        band_indices = self.photometry['band'] == band
        if not sum(band_indices) > (len(self.bump_features_names) - 2):
            return np.array([])

        # get info for this filter
        time = self.photometry['mjd'].values[band_indices]
        flux = self.photometry['flux'].values[band_indices]
        fluxerr = self.photometry['fluxerr'].values[band_indices]

        # fit Bump function
        bump_param = fit_bump(time, flux, fluxerr)

        return bump_param

    def evaluate_bump(self, time: np.array):
        """Evaluate the Bump function given parameter values.

        Parameters
        ----------
        time: np.array or list
            Time since first light curve observation.

        Returns
        -------
        dict
            Value of the Bump flux in each required time per filter.
        """
        # store flux values and starting points
        flux = {}

        for k in range(len(self.filters)):
            # store flux values per filter
            flux[self.filters[k]] = []

            # check if Bump features exist
            if 'None' not in self.bump_features[k * 5 : (k + 1) * 5]:
                for item in time:
                    flux[self.filters[k]].append(\
                           bump(item, self.bump_features[0 + k * 5],
                                 self.bump_features[1 + k * 5],
                                 self.bump_features[2 + k * 5]))
            else:
                flux[self.filters[k]].append(None)

        return flux

    def fit_bump_all(self):
        """
        Perform Bump fit for all filters independently and concatenate results.
        Populates the attributes: bump_features.
        """
        default_bump_features = ['None'] * len(self.bump_features_names)

        if self.photometry.shape[0] < 1:
            self.bump_features = ['None'] * len(self.bump_features_names) * len(self.filters)

        elif 'None' not in self.bump_features:
            self.bump_features = []
            for each_band in self.filters:
                best_fit = self.fit_bump(band=each_band)
                if (len(best_fit) > 0) and (not np.isnan(np.sum(best_fit))):
                    self.bump_features.extend(best_fit)
                else:
                    self.bump_features.extend(default_bump_features)
        else:
            self.bump_features.extend(default_bump_features)

    def clear_data(self):
        """ Reset to default values """
        self.photometry = []
        self.redshift = None
        self.sntype = None
        self.sncode = None
        self.sample = None

    def plot_bazin_fit(self, save=True, show=False, output_file=' ',
                       figscale=1, extrapolate=False,
                       time_flux_pred=None, unit='flux'):
        """
        Plot data and Bazin fitted function.

        Parameters
        ----------
        figscale: float (optional)
            Allow to control the size of the figure.
        extrapolate: bool (optional)
            If True, also plot the estimated flux values.
            Default is False.
        output_file: str (optional)
            Name of file to store the plot.
        save: bool (optional)
             Save figure to file. Default is True.
        show: bool (optinal)
             Display plot in windown. Default is False.
        time_flux_pred: list (optional)
            Time since first observation where flux is to be
            estimated. It is only used if "extrapolate == True".
            Default is None.
        unit: str (optional)
            Unit for plot. Options are 'flux' or 'mag'.
            Use zero point from SNANA for flux-to-mag conversion
            ==> mag = 2.5 * (11 - np.log10(flux)).
            Default is 'flux'.
        """

        # number of columns in the plot
        ncols = len(self.filters) / 2 + len(self.filters) % 2
        fsize = (figscale * 5 * ncols , figscale * 10)

        plt.figure(figsize=fsize)

        for i in range(len(self.filters)):
            plt.subplot(2, int(ncols), i + 1)
            plt.title('Filter: ' + self.filters[i])

            # filter flag
            filter_flag = self.photometry['band'] == self.filters[i]
            x = self.photometry['mjd'][filter_flag].values
            y = self.photometry['flux'][filter_flag].values
            yerr = self.photometry['fluxerr'][filter_flag].values

            # check Bazin fit convergence
            if 'None' in self.bazin_features[i * 5 : (i + 1) * 5]:
                plot_fit = False
            else:
                plot_fit = True

            # shift to avoid large numbers in x-axis
            time = x - min(x)

            if plot_fit:
                xaxis = np.linspace(0, max(time), 500)[:, np.newaxis]
                fitted_flux = self.evaluate_bazin(xaxis)
                if unit == 'flux':
                    plt.plot(xaxis, fitted_flux[self.filters[i]], color='red',
                             lw=1.5, label='Bazin fit')
                elif unit == 'mag':
                    mag = self.conv_flux_mag(fitted_flux[self.filters[i]])
                    mag_flag = mag < 50
                    plt.plot(xaxis[mag_flag], mag[mag_flag], color='red',
                             lw=1.5)
                else:
                    raise ValueError('Unit can only be "flux" or "mag".')

                if extrapolate:
                    xaxis_extrap = list(xaxis.flatten()) + list(time_flux_pred)
                    xaxis_extrap = np.sort(np.array(xaxis_extrap))
                    ext_flux = self.evaluate_bazin(xaxis_extrap)
                    if unit == 'flux':
                        plt.plot(xaxis_extrap, ext_flux[self.filters[i]],
                                 color='red', lw=1.5, ls='--',
                                 label='Bazin extrap')
                    elif unit == 'mag':
                        ext_mag = self.conv_flux_mag(ext_flux[self.filters[i]])
                        ext_mag_flag = ext_mag < 50
                        plt.plot(xaxis_extrap[ext_mag_flag],
                                 ext_mag[ext_mag_flag],
                                 color='red', lw=1.5, ls='--')

            if unit == 'flux':
                plt.errorbar(time, y, yerr=yerr, color='blue', fmt='o',
                             label='obs')
                plt.ylabel('FLUXCAL')
            elif unit == 'mag':
                mag_obs  = self.conv_flux_mag(y)
                mag_obs_flag = mag_obs < 50
                time_mag = time[mag_obs_flag]

                plt.scatter(time_mag, mag_obs[mag_obs_flag], color='blue',
                            label='calc mag', marker='s')

                # if MAG is provided in the table, also plot it
                # this allows checking the flux mag conversion
                if 'MAG' in self.photometry.keys():
                    mag_flag = self.photometry['MAG'].values < 50
                    mag_ff = np.logical_and(filter_flag, mag_flag)
                    mag_table = self.photometry['MAG'][mag_ff].values
                    mjd_table = self.photometry['mjd'][mag_ff].values - min(x)

                    plt.scatter(mjd_table, mag_table, color='black',
                                label='table mag', marker='x')

                ax = plt.gca()
                ax.set_ylim(ax.get_ylim()[::-1])
                plt.ylabel('mag')

            plt.xlabel('days since first observation')
            plt.tight_layout()

        if save:
            plt.savefig(output_file)
        if show:
            plt.show()
                        
    def plot_bump_fit(self, save=True, show=False, output_file=' ',
                       figscale=1, extrapolate=False,
                       time_flux_pred=None, unit='flux'):
        """
        Plot data and Bump fitted function.

        Parameters
        ----------
        figscale: float (optional)
            Allow to control the size of the figure.
        extrapolate: bool (optional)
            If True, also plot the estimated flux values.
            Default is False.
        output_file: str (optional)
            Name of file to store the plot.
        save: bool (optional)
             Save figure to file. Default is True.
        show: bool (optinal)
             Display plot in windown. Default is False.
        time_flux_pred: list (optional)
            Time after last observation where flux is to be
            estimated. It is only used if "extrapolate == True".
            Default is None.
        unit: str (optional)
            Unit for plot. Options are 'flux' or 'mag'.
            Use zero point from SNANA for flux-to-mag conversion
            ==> mag = 2.5 * (11 - np.log10(flux)).
            Default is 'flux'.
        """

        # number of columns in the plot
        ncols = len(self.filters) / 2 + len(self.filters) % 2
        fsize = (figscale * 5 * ncols , figscale * 10)

        plt.figure(figsize=fsize)

        for i in range(len(self.filters)):
            plt.subplot(2, int(ncols), i + 1)
            plt.title('Filter: ' + self.filters[i])

            # filter flag
            filter_flag = self.photometry['band'] == self.filters[i]
            x = self.photometry['mjd'][filter_flag].values
            y = self.photometry['flux'][filter_flag].values
            yerr = self.photometry['fluxerr'][filter_flag].values

            # check Bump fit convergence
            if 'None' in self.bump_features[i * 5 : (i + 1) * 5]:
                plot_fit = False
            else:
                plot_fit = True

            # shift to avoid large numbers in x-axis
            time = x + self.bump_features[i * 5 + 3]

            if plot_fit:
                xaxis = np.linspace(min(time), max(time), 500)[:, np.newaxis]
                fitted_flux = np.array(self.evaluate_bump(xaxis)[self.filters[i]]) * self.bump_features[i * 5 + 4]
                if unit == 'flux':
                    plt.plot(xaxis, fitted_flux, color='red',
                             lw=1.5, label='Bump fit')
                elif unit == 'mag':
                    mag = self.conv_flux_mag(fitted_flux[self.filters[i]])
                    mag_flag = mag < 50
                    plt.plot(xaxis[mag_flag], mag[mag_flag], color='red',
                             lw=1.5)
                else:
                    raise ValueError('Unit can only be "flux" or "mag".')

                if extrapolate:
                    xaxis_extrap = list(xaxis.flatten()) + [time_flux_pred + max(time)]
                    xaxis_extrap = np.sort(np.array(xaxis_extrap))
                    ext_flux = np.array(self.evaluate_bump(xaxis_extrap)[self.filters[i]]) \
                               * self.bump_features[i * 5 + 4]
                    if unit == 'flux':
                        plt.plot(xaxis_extrap, ext_flux,
                                 color='red', lw=1.5, ls='--',
                                 label='Bump extrap')
                    elif unit == 'mag':
                        ext_mag = self.conv_flux_mag(ext_flux[self.filters[i]])
                        ext_mag_flag = ext_mag < 50
                        plt.plot(xaxis_extrap[ext_mag_flag],
                                 ext_mag[ext_mag_flag],
                                 color='red', lw=1.5, ls='--')

            if unit == 'flux':
                plt.errorbar(time, y, yerr=yerr, color='blue', fmt='o',
                             label='obs')
                plt.ylabel('FLUXCAL')
            elif unit == 'mag':
                mag_obs  = self.conv_flux_mag(y)
                mag_obs_flag = mag_obs < 50
                time_mag = time[mag_obs_flag]

                plt.scatter(time_mag, mag_obs[mag_obs_flag], color='blue',
                            label='calc mag', marker='s')

                # if MAG is provided in the table, also plot it
                # this allows checking the flux mag conversion
                if 'MAG' in self.photometry.keys():
                    mag_flag = self.photometry['MAG'].values < 50
                    mag_ff = np.logical_and(filter_flag, mag_flag)
                    mag_table = self.photometry['MAG'][mag_ff].values
                    mjd_table = self.photometry['mjd'][mag_ff].values - min(x)

                    plt.scatter(mjd_table, mag_table, color='black',
                                label='table mag', marker='x')

                ax = plt.gca()
                ax.set_ylim(ax.get_ylim()[::-1])
                plt.ylabel('mag')

            plt.xlabel('days since first observation')
            plt.tight_layout()

        if save:
            plt.savefig(output_file)
        if show:
            plt.show()


def _get_features_to_write(light_curve_data: LightCurve) -> list:
    """
    Returns features list to write

    Parameters
    ----------
    light_curve_data
        fitted light curve data
    """
    features_list = [light_curve_data.id, light_curve_data.redshift,
                     light_curve_data.sntype, light_curve_data.sncode,
                     light_curve_data.sample]
    features_list.extend(light_curve_data.bazin_features)
    return features_list


def write_features_to_output_file(
        light_curve_data: LightCurve, features_file: IO):
    """
    Writes fitted light curve data to output features file

    Parameters
    ----------
    light_curve_data
        fitted ligtht curve data
    features_file
        features output file
    """
    current_features = _get_features_to_write(
        light_curve_data)
    features_file.write(
        ' '.join(str(each_feature) for each_feature
                 in current_features) + '\n')


def _snpcc_sample_fit_bazin(
        file_name: str, path_to_data_dir: str) -> LightCurve:
    """
    Reads SNPCC file and performs bazin fit
    Parameters
    ----------
    file_name
        SNPCC file name
    path_to_data_dir
         Path to directory containing the set of individual files,
         one for each light curve.

    """
    light_curve_data = LightCurve()
    light_curve_data.load_snpcc_lc(
        os.path.join(path_to_data_dir, file_name))
    light_curve_data.fit_bazin_all()
    return light_curve_data


def fit_snpcc_bazin(
        path_to_data_dir: str, features_file: str,
        file_prefix: str = "DES_SN", number_of_processors: int = 1):
    """
    Perform Bazin fit to all objects in the SNPCC data.

     Parameters
     ----------
     path_to_data_dir: str
         Path to directory containing the set of individual files,
         one for each light curve.
     features_file: str
         Path to output file where results should be stored.
     file_prefix: str
        File names prefix
     number_of_processors: int, default 1
        Number of cpu processes to use.
    """
    files_list = os.listdir(path_to_data_dir)
    files_list = [each_file for each_file in files_list
                  if each_file.startswith(file_prefix)]
    multi_process = multiprocessing.Pool(number_of_processors)
    logging.info("Starting SNPCC bazin fit...")
    with open(features_file, 'w') as snpcc_features_file:
        snpcc_features_file.write(' '.join(SNPCC_FEATURES_HEADER) + '\n')
        for light_curve_data in multi_process.starmap(
                _snpcc_sample_fit_bazin, zip(
                    files_list, repeat(path_to_data_dir))):
            if 'None' not in light_curve_data.bazin_features:
                write_features_to_output_file(
                    light_curve_data, snpcc_features_file)
    logging.info("Features have been saved to: %s", features_file)


def _resspect_sample_fit_bazin(
        index: int, snid: int, path_photo_file: str,
        sample: str, light_curve_data: LightCurve, meta_header: pd.DataFrame,
        redshift_name: Union[str, None], sncode_name: Union[str, None],
        sntype_name: Union[str, None]) -> LightCurve:
    """
    Performs bazin fit for PLAsTiCC dataset with snid

    Parameters
    ----------
    index
        index of snid
    snid
        Identification number for the desired light curve.
    path_photo_file: str
        Complete path to light curve file.
    sample: str
        'train' or 'test'. Default is None.
    light_curve_data
        light curve class
    meta_header
        photometry meta header data
    redshift_name
        redshift meta header column name
    sncode_name
        sncode meta header column name
    sntype_name
        sntype meta header column name
    """
    light_curve_data.load_resspect_lc(path_photo_file, snid)
    light_curve_data.fit_bazin_all()
    light_curve_data.redshift = meta_header[redshift_name][index]
    light_curve_data.sncode = meta_header[sncode_name][index]
    light_curve_data.sntype = meta_header[sntype_name][index]
    light_curve_data.sample = sample
    light_curve_data_copy = copy(light_curve_data)
    light_curve_data.clear_data()
    return light_curve_data_copy


def fit_resspect_bazin(path_photo_file: str, path_header_file: str,
                       output_file: str, sample=None,
                       number_of_processors: int = 1):
    """Perform Bazin fit to all objects in a given RESSPECT data file.

    Parameters
    ----------
    path_photo_file: str
        Complete path to light curve file.
    path_header_file: str
        Complete path to header file.
    output_file: str
        Output file where the features will be stored.
    sample: str
        'train' or 'test'. Default is None.
    number_of_processors: int, default 1.
        Number of cpu processes to use
    """
    meta_header = get_resspect_header_data(path_header_file, path_photo_file)

    meta_header_keys = meta_header.keys().tolist()
    id_name = find_available_key_name_in_header(
        meta_header_keys, ['SNID', 'snid', 'objid'])
    z_name = find_available_key_name_in_header(
        meta_header_keys, ['redshift', 'REDSHIFT_FINAL'])
    type_name = find_available_key_name_in_header(
        meta_header_keys, ['type', 'SIM_TYPE_NAME', 'TYPE'])
    subtype_name = find_available_key_name_in_header(
        meta_header_keys, ['code', 'SIM_TYPE_INDEX', 'SNTYPE_SUBCLASS'])

    light_curve_data = LightCurve()
    snid_values = meta_header[id_name]
    snid_values = np.array(list(snid_values.items()))
    multi_process = multiprocessing.Pool(number_of_processors)
    logging.info("Starting RESSPECT bazin fit...")
    with open(output_file, 'w') as ressepect_features_file:
        ressepect_features_file.write(
            ' '.join(PLASTICC_RESSPECT_FEATURES_HEADER) + '\n')
        iterator_list = zip(
            snid_values[:, 0].tolist(), snid_values[:, 1].tolist(),
            repeat(path_photo_file), repeat(sample), repeat(light_curve_data),
            repeat(meta_header), repeat(z_name), repeat(subtype_name),
            repeat(type_name))
        for light_curve_data in multi_process.starmap(
                _resspect_sample_fit_bazin, iterator_list):
            if 'None' not in light_curve_data.bazin_features:
                write_features_to_output_file(
                    light_curve_data, ressepect_features_file)
            light_curve_data.clear_data()
    logging.info("Features have been saved to: %s", output_file)


def _plasticc_sample_fit_bazin(
        index: int, snid: int, path_photo_file: str,
        sample: str, light_curve_data: LightCurve,
        meta_header: pd.DataFrame) -> LightCurve:
    """
    Performs bazin fit for PLAsTiCC dataset with snid

    Parameters
    ----------
    index
        index of snid
    snid
        Identification number for the desired light curve.
    path_photo_file: str
        Complete path to light curve file.
    sample: str
        'train' or 'test'. Default is None.
    light_curve_data
        light curve class
    meta_header
        photometry meta header data
    """
    light_curve_data.load_plasticc_lc(path_photo_file, snid)
    light_curve_data.fit_bazin_all()
    light_curve_data.redshift = meta_header['true_z'][index]
    light_curve_data.sncode = meta_header['true_target'][index]
    light_curve_data.sntype = PLASTICC_TARGET_TYPES[
        light_curve_data.sncode]
    light_curve_data.sample = sample
    light_curve_data_copy = copy(light_curve_data)
    light_curve_data.clear_data()
    return light_curve_data_copy


def fit_plasticc_bazin(path_photo_file: str, path_header_file: str,
                       output_file: str, sample='train',
                       number_of_processors: int = 1):
    """
    Perform Bazin fit to all objects in a given PLAsTiCC data file.
    Parameters
    ----------
    path_photo_file: str
        Complete path to light curve file.
    path_header_file: str
        Complete path to header file.
    output_file: str
        Output file where the features will be stored.
    sample: str
        'train' or 'test'. Default is 'train'.
    number_of_processors: int, default 1
        Number of cpu processes to use.
    """

    name_list = ['SNID', 'snid', 'objid', 'object_id']
    meta_header = read_plasticc_full_photometry_data(path_header_file)
    meta_header_keys = meta_header.keys().tolist()
    id_name = find_available_key_name_in_header(
        meta_header_keys, name_list)
    light_curve_data = LightCurve()

    if sample == 'train':
        snid_values = meta_header[id_name]
    elif sample == 'test':
        light_curve_data.full_photometry = read_plasticc_full_photometry_data(
            path_photo_file)
        snid_values = pd.DataFrame(np.unique(
            light_curve_data.full_photometry[id_name].values),
            columns=[id_name])[id_name]
    snid_values = np.array(list(snid_values.items()))
    multi_process = multiprocessing.Pool(number_of_processors)
    logging.info("Starting PLAsTiCC bazin fit...")
    with open(output_file, 'w') as plasticc_features_file:
        plasticc_features_file.write(
            ' '.join(PLASTICC_RESSPECT_FEATURES_HEADER) + '\n')
        iterator_list = zip(
            snid_values[:, 0].tolist(), snid_values[:, 1].tolist(), repeat(path_photo_file),
            repeat(sample), repeat(light_curve_data), repeat(meta_header))
        for light_curve_data in multi_process.starmap(
                _plasticc_sample_fit_bazin, iterator_list):
            if 'None' not in light_curve_data.bazin_features:
                write_features_to_output_file(
                    light_curve_data, plasticc_features_file)
    logging.info("Features have been saved to: %s", output_file)


def main():
    return None


if __name__ == '__main__':
    main()
