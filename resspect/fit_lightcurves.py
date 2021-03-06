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

from resspect.bazin import bazin, fit_scipy
from resspect.exposure_time_calculator import ExpTimeCalc
from resspect.snana_fits_to_pd import read_fits

import io
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import tarfile

__all__ = ['LightCurve', 'fit_snpcc_bazin', 'fit_resspect_bazin',
           'fit_plasticc_bazin']


class LightCurve(object):
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
        self.bazin_features = []
        self.bazin_features_names = ['a', 'b', 't0', 'tfall', 'trsise']
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

    def load_snpcc_lc(self, path_to_data: str):
        """Reads one LC from SNPCC data.

        Populates the attributes: dataset_name, id, sample, redshift, sncode,
        sntype, photometry, sim_peakmag and sim_pkmjd.

        Parameters
        ---------
        path_to_data: str
            Path to text file with data from a single SN.
        """

        # set the designation of the data set
        self.dataset_name = 'SNPCC'

        # set filters
        self.filters = ['g', 'r', 'i', 'z']

        # set SN types
        snii = ['2', '3', '4', '12', '15', '17', '19', '20', '21', '24', '25',
                '26', '27', '30', '31', '32', '33', '34', '35', '36', '37',
                '38', '39', '40', '41', '42', '43', '44']

        snibc = ['1', '5', '6', '7', '8', '9', '10', '11', '13', '14', '16',
                 '18', '22', '23', '29', '45', '28']

        # read light curve data
        op = open(path_to_data, 'r')
        lin = op.readlines()
        op.close()

        # separate elements
        data_all = np.array([elem.split() for elem in lin], dtype=object)

        # flag useful lines
        flag_lines = np.array([True if len(line) > 1 else False \
                               for line in data_all])

        # get only informative lines
        data = data_all[flag_lines]

        photometry_raw = []               # store photometry
        header = []                      # store parameter header

        # get header information
        for line in data:
            if line[0] == 'SNID:':
                self.id = int(line[1])
                self.id_name = 'SNID'
            elif line[0] == 'SNTYPE:':
                if line[1] == '-9':
                    self.sample = 'test'
                else:
                    self.sample = 'train'
            elif line[0] == 'SIM_REDSHIFT:':
                self.redshift = float(line[1])
            elif line[0] == 'SIM_NON1a:':
                self.sncode = line[1]
                if line[1] in snibc:
                    self.sntype = 'Ibc'
                elif line[1] in snii:
                    self.sntype = 'II'
                elif line[1] == '0':
                    self.sntype = 'Ia'
                else:
                    raise ValueError('Unknown supernova type!')
            elif line[0] == 'VARLIST:':
                header: list = line[1:]
            elif line[0] == 'OBS:':
                photometry_raw.append(np.array(line[1:]))
            elif line[0] == 'SIM_PEAKMAG:':
                self.sim_peakmag = np.array([float(item) \
                                             for item in line[1:5]])
            elif line[0] == 'SIM_PEAKMJD:':
                self.sim_pkmjd = float(line[1])

        # transform photometry into array
        photometry_raw = np.array(photometry_raw)

        # put photometry into data frame
        self.photometry['mjd'] = np.array([float(item) \
                         for item in photometry_raw[:, header.index('MJD')]])
        self.photometry['band'] = \
                         np.array(photometry_raw[:, header.index('FLT')])
        self.photometry['flux'] = np.array([float(item) \
                     for item in photometry_raw[:, header.index('FLUXCAL')]])
        self.photometry['fluxerr'] = np.array([float(item) \
                  for item in photometry_raw[:, header.index('FLUXCALERR')]])
        self.photometry['SNR'] = np.array([float(item) \
                         for item in photometry_raw[:, header.index('SNR')]])
        self.photometry['MAG'] = np.array([float(item) \
                         for item in photometry_raw[:, header.index('MAG')]])
        self.photometry['MAGERR'] = np.array([float(item) \
                      for item in photometry_raw[:, header.index('MAGERR')]])

    def load_resspect_lc(self, photo_file, snid):
        """
        Return 1 light curve from RESSPECT simulations.
    
        Parameters
        ----------
        photo_file: str
            Complete path to light curves file.
        snid: int
            Identification number for the desired light curve.
        """

        if self.full_photometry.shape[0] == 0:
            if '.tar.gz' in photo_file:
                tar = tarfile.open(photo_file, 'r:gz')
                fname = tar.getmembers()[0]
                content = tar.extractfile(fname).read()
                self.full_photometry = pd.read_csv(io.BytesIO(content))
                tar.close()
            elif '.FITS' in photo_file:
                df_header, self.full_photometry = \
                            read_fits(photo_file, drop_separators=True)
            else:    
                self.full_photometry = pd.read_csv(photo_file, 
                                                   index_col=False)

        if 'SNID' in self.full_photometry.keys():
            flag = self.full_photometry['SNID'] == snid
            self.id_name = 'SNID'
        elif 'snid' in self.full_photometry.keys():
            flag = self.full_photometry['snid'] == snid
            self.id_name = 'snid'
        elif 'objid' in self.full_photometry.keys():
            flag = self.full_photometry['objid'] == snid
            self.id_name = 'objid'
        elif 'id' in self.full_photometry.keys():
            flag = self.full_photometry['id'] == snid
            self.id_name = 'id'

        photo = self.full_photometry[flag]

        self.dataset_name = 'RESSPECT'                      # name of data set
        self.filters = ['u', 'g', 'r', 'i', 'z', 'Y']       # list of filters  
        
        # check filter name
        if 'b' in str(photo['FLT'].values[0]):
            band = []
            for i in range(photo.shape[0]):
                for f in self.filters:
                    if "b'" + f + " '" == str(photo['FLT'].values[i]) or \
                    "b'" + f + "'" == str(photo['FLT'].values[i]) or \
                    "b'" + f + "' " == str(photo['FLT'].values[i]):
                        band.append(f)
            photo.insert(1, 'band', band, True)

        else:
            photo.insert(1, 'band', photo['FLT'].values, True)
                        
        self.id = snid 
        self.photometry = {}
        self.photometry['mjd'] = photo['MJD'].values
        self.photometry['band'] = photo['band'].values
        self.photometry['flux'] = photo['FLUXCAL'].values
        self.photometry['fluxerr'] = photo['FLUXCALERR'].values

        if 'SNR' in photo.keys():
            self.photometry['SNR'] = photo['SNR'].values
        else:
            signal = self.photometry['flux']
            noise = self.photometry['fluxerr']
            self.photometry['SNR'] = \
                np.array([signal[i]/noise[i] for i in range(signal.shape[0])])
            
        self.photometry = pd.DataFrame(self.photometry)
        
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

        # read from file if full photometry not available
        if self.full_photometry.shape[0] == 0:
            if '.tar.gz' in photo_file:
                tar = tarfile.open(photo_file, 'r:gz')
                fname = tar.getmembers()[0]
                content = tar.extractfile(fname).read()
                self.full_photometry = pd.read_csv(io.BytesIO(contente))
            else:
                self.full_photometry = pd.read_csv(photo_file, 
                                                   index_col=False)

                if ' ' in self.full_photometry.keys()[0]:
                    self.full_photometry = pd.read_csv(photo_file, sep=' ',
                                                       index_col=False)

        if 'object_id' in self.full_photometry.keys():
            flag = self.full_photometry['object_id'] == snid
            self.id_name = 'object_id'
        elif 'SNID' in self.full_photometry.keys():
            flag = self.full_photometry['SNID'] == snid
            self.id_name = 'SNID'
        elif 'snid' in self.full_photometry.keys():
            flag = self.full_photometry['snid'] == snid
            self.id_name = 'snid'

        photo = self.full_photometry[flag]

        filter_dict = {0:'u', 1:'g', 2:'r', 3:'i', 4:'z', 5:'Y'}
           
        self.dataset_name = 'PLAsTiCC'              # name of data set
        self.filters = ['u', 'g', 'r', 'i', 'z', 'Y']       # list of filters
        self.id = snid
        self.photometry = {}
        self.photometry['mjd'] = photo['mjd'].values
        self.photometry['band'] = [filter_dict[photo['passband'].values[k]] 
                                   for k in range(photo['passband'].shape[0])]
        self.photometry['flux'] = photo['flux'].values
        self.photometry['fluxerr'] = photo['flux_err'].values
        self.photometry['detected_bool'] = photo['detected_bool'].values
        self.photometry = pd.DataFrame(self.photometry)

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

        if criteria == 1:
            if 'MAG' in self.photometry.keys():
                # check surviving photometry
                self.last_mag = self.photometry['MAG'].values[surv_flag][-1]

            else:
                surv_flux = self.photometry['flux'].values[surv_flag]
                self.last_mag = self.conv_flux_mag([surv_flux[-1]])[0]

        elif criteria == 2:
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

        else:
            raise ValueError('Criteria needs to be "1" or "2". \n ' + \
                             'See docstring for further info.')

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

    def fit_bazin(self, band: str):
        """Extract Bazin features for one filter.

        Parameters
        ----------
        band: str
            Choice of broad band filter

        Returns
        -------
        bazin_param: list
            Best fit parameters for the Bazin function: 
            [a, b, t0, tfall, trise].
        """

        # build filter flag
        filter_flag = self.photometry['band'] == band

        # get info for this filter
        time = self.photometry['mjd'].values[filter_flag]
        flux = self.photometry['flux'].values[filter_flag]

        # fit Bazin function
        bazin_param = fit_scipy(time - time[0], flux)

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
        """Perform Bazin fit for all filters independently and 
           concatenate results.

        Populates the attributes: bazin_features.
        """
        # remove previous fit attempts
        self.bazin_features = []

        for band in self.filters:
            # build filter flag
            filter_flag = self.photometry['band'] == band

            if sum(filter_flag) > 4:
                best_fit = self.fit_bazin(band)

                if sum([str(item) == 'nan' for item in best_fit]) == 0:
                    for fit in best_fit:
                        self.bazin_features.append(fit)
                else:
                    for i in range(5):
                        self.bazin_features.append('None')
            else:
                for i in range(5):
                    self.bazin_features.append('None')

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
            plt.subplot(2, ncols, i + 1)
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
                    xaxis_extrap = list(xaxis) + list(time_flux_pred)
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


def fit_snpcc_bazin(path_to_data_dir: str, features_file: str):
    """Perform Bazin fit to all objects in the SNPCC data.

    Parameters
    ----------
    path_to_data_dir: str
        Path to directory containing the set of individual files,
        one for each light curve.
    features_file: str
        Path to output file where results should be stored.
    """

    # read file names
    file_list_all = os.listdir(path_to_data_dir)
    lc_list = [elem for elem in file_list_all if 'DES_SN' in elem]

    # count survivers
    count_surv = 0

    # add headers to files
    with open(features_file, 'w') as param_file:
        param_file.write('id redshift type code orig_sample gA gB ' + \
                         'gt0 gtfall gtrise rA rB rt0 rtfall rtrise' + \
                         ' iA iB it0 itfall itrise zA zB zt0 ztfall' + \
                         ' ztrise\n')

    for file in lc_list:

        # fit individual light curves
        lc = LightCurve()
        lc.load_snpcc_lc(path_to_data_dir + file)
        lc.fit_bazin_all()

        print(lc_list.index(file), ' - id:', lc.id)

        # append results to the correct matrix
        if 'None' not in lc.bazin_features:
            count_surv = count_surv + 1
            print('Survived: ', count_surv)

            # save features to file
            with open(features_file, 'a') as param_file:
                param_file.write(str(lc.id) + ' ' + str(lc.redshift) + ' ' + \
                                 str(lc.sntype) + ' ')
                param_file.write(str(lc.sncode) + ' ' + str(lc.sample) + ' ')
                for item in lc.bazin_features:
                    param_file.write(str(item) + ' ')
                param_file.write('\n')

    param_file.close()

def fit_resspect_bazin(path_photo_file: str, path_header_file:str,
                       output_file: str, sample=None):
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
    """
    # count survivers
    count_surv = 0

    # read header information
    if '.tar.gz' in path_header_file:
        tar = tarfile.open(path_header_file, 'r:gz')
        fname = tar.getmembers()[0]
        content = tar.extractfile(fname).read()
        header = pd.read_csv(io.BytesIO(content))
        tar.close()
    elif 'FITS' in path_header_file:
        header, photo = read_fits(path_photo_file, drop_separators=True)    
    else:   
        header = pd.read_csv(path_header_file, index_col=False)
    
    # add headers to files
    with open(output_file, 'w') as param_file:
        param_file.write('id redshift type code orig_sample uA uB ut0 utfall ' +
                         'utrise gA gB gt0 gtfall gtrise rA rB rt0 rtfall ' +
                         'rtrise iA iB it0 itfall itrise zA zB zt0 ztfall ' + 
                         'ztrise YA YB Yt0 Ytfall Ytrise\n')

    # check id flag
    if 'SNID' in header.keys():
        id_name = 'SNID'
    elif 'snid' in header.keys():
        id_name = 'snid'
    elif 'objid' in header.keys():
        id_name = 'objid'

    # check redshift flag
    if 'redshift' in header.keys():
        z_name = 'redshift'
    elif 'REDSHIFT_FINAL' in header.keys():
        z_name = 'REDSHIFT_FINAL'

    # check type flag
    if 'type' in header.keys():
        type_name = 'type'
    elif 'SIM_TYPE_NAME' in header.keys():
        type_name = 'SIM_TYPE_NAME'
    elif 'TYPE' in header.keys():
        type_name = 'TYPE'

    # check subtype flag
    if 'code' in header.keys():
        subtype_name = 'code'
    elif 'SIM_TYPE_INDEX' in header.keys():
        subtype_name = 'SIM_TYPE_NAME'
    elif 'SNTYPE_SUBCLASS' in header.keys():
        subtype_name = 'SNTYPE_SUBCLASS'

    lc = LightCurve()
    
    for snid in header[id_name].values:      

        # load individual light curves                       
        lc.load_resspect_lc(path_photo_file, snid)

        # fit all bands                
        lc.fit_bazin_all()

        # get model name 
        lc.redshift = header[z_name][header[lc.id_name] == snid].values[0]
        lc.sntype = header[type_name][header[lc.id_name] == snid].values[0]
        lc.sncode = header[subtype_name][header[lc.id_name] == snid].values[0]
        lc.sample = sample

        # append results to the correct matrix
        if 'None' not in lc.bazin_features:
            count_surv = count_surv + 1
            print('Survived: ', count_surv)

            # save features to file
            with open(output_file, 'a') as param_file:
                param_file.write(str(lc.id) + ' ' + str(lc.redshift) + ' ' + \
                                 str(lc.sntype) + ' ')
                param_file.write(str(lc.sncode) + ' ' + str(lc.sample) + ' ')
                for item in lc.bazin_features:
                    param_file.write(str(item) + ' ')
                param_file.write('\n')

        lc.redshift = None
        lc.sntype = None
        lc.sncode = None
        lc.sample = None
        lc.photometry = []

    param_file.close()


def fit_plasticc_bazin(path_photo_file: str, path_header_file:str,
                       output_file: str, sample=None):
    """Perform Bazin fit to all objects in a given PLAsTiCC data file.

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
    """
    types = {90: 'Ia', 67: '91bg', 52:'Iax', 42:'II', 62:'Ibc', 
             95: 'SLSN', 15:'TDE', 64:'KN', 88:'AGN', 92:'RRL', 65:'M-dwarf',
             16:'EB',53:'Mira', 6:'MicroL', 991:'MicroLB', 992:'ILOT', 
             993:'CART', 994:'PISN',995:'MLString'}

    # count survivers
    count_surv = 0

    # read header information
    if '.tar.gz' in path_header_file:
        tar = tarfile.open(path_header_file, 'r:gz')
        fname = tar.getmembers()[0]
        content = tar.extracfile(fname).read()
        header = pd.read_csv(io.BytesIO(content))
        tar.close()
    else:
        header = pd.read_csv(path_header_file, index_col=False)

        if ' ' in header.keys()[0]:
            header = pd.read_csv(path_header_file, sep=' ', index_col=False)

    # add headers to files
    with open(output_file, 'w') as param_file:
        param_file.write('id redshift type code orig_sample uA uB ut0 utfall ' +
                         'utrise gA gB gt0 gtfall gtrise rA rB rt0 rtfall ' +
                         'rtrise iA iB it0 itfall itrise zA zB zt0 ztfall ' + 
                         'ztrise YA YB Yt0 Ytfall Ytrise\n')

    # check id flag
    if 'SNID' in header.keys():
        id_name = 'SNID'
    elif 'snid' in header.keys():
        id_name = 'snid'
    elif 'objid' in header.keys():
        id_name = 'objid'
    elif 'object_id' in header.keys():
        id_name = 'object_id'

    lc = LightCurve()
    
    for snid in header[id_name].values:      

        # load individual light curves                      
        lc.load_plasticc_lc(path_photo_file, snid) 
        lc.fit_bazin_all()

        # get model name 
        lc.redshift = header['true_z'][header[lc.id_name] == snid].values[0]
        lc.sntype = \
            types[header['true_target'][header[lc.id_name] == snid].values[0]] 
        lc.sncode = header['true_target'][header[lc.id_name] == snid].values[0]
        lc.sample = sample

        # append results to the correct matrix
        if 'None' not in lc.bazin_features:
            count_surv = count_surv + 1
            print('Survived: ', count_surv)

            # save features to file
            with open(output_file, 'a') as param_file:
                param_file.write(str(lc.id) + ' ' + str(lc.redshift) + ' ' + \
                                 str(lc.sntype) + ' ')
                param_file.write(str(lc.sncode) + ' ' + str(lc.sample) + ' ')
                for item in lc.bazin_features:
                    param_file.write(str(item) + ' ')
                param_file.write('\n')

        lc.photometry = []
        lc.redshift = None
        lc.sntype = None
        lc.sncode = None
        lc.sample = None

    param_file.close()


def main():
    return None


if __name__ == '__main__':
    main()
