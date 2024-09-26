"""
    Bump feature extractor
"""

import matplotlib.pylab as plt
import numpy as np

from resspect import bump
from resspect.bump import fit_bump
from resspect.feature_extractors.light_curve import LightCurve


class BumpFeatureExtractor(LightCurve):
    def __init__(self):
        super().__init__()
        self.features_names = ['p1', 'p2', 'p3', 'time_shift', 'max_flux']

    def evaluate(self, time: np.array) -> dict:
        """
        Evaluate the Bump function given parameter values.

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
            if 'None' not in self.features[k * 5 : (k + 1) * 5]:
                for item in time:
                    flux[self.filters[k]].append(
                        bump(item, self.features[0 + k * 5],
                             self.features[1 + k * 5],
                             self.features[2 + k * 5]))
            else:
                flux[self.filters[k]].append(None)
        return flux

    def fit(self, band: str) -> np.ndarray:
        """
        Extract Bump features for one filter.

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
        if not sum(band_indices) > (len(self.features_names) - 2):
            return np.array([])

        # get info for this filter
        time = self.photometry['mjd'].values[band_indices]
        flux = self.photometry['flux'].values[band_indices]
        flux_error = self.photometry['fluxerr'].values[band_indices]

        # fit Bump function
        bump_param = fit_bump(time, flux, flux_error)
        return bump_param

    def fit_all(self):
        """
        Perform Bump fit for all filters independently and concatenate results.
        Populates the attributes: bump_features.
        """
        default_bump_features = ['None'] * len(self.features_names)

        if self.photometry.shape[0] < 1:
            self.features = ['None'] * len(self.features_names) * len(self.filters)

        elif 'None' not in self.features:
            self.features = []
            for each_band in self.filters:
                best_fit = self.fit(band=each_band)
                if (len(best_fit) > 0) and (not np.isnan(np.sum(best_fit))):
                    self.features.extend(best_fit)
                else:
                    self.features.extend(default_bump_features)
        else:
            self.features.extend(default_bump_features)

    def plot_fit(self, save=True, show=False, output_file=' ',
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
            if 'None' in self.features[i * 5 : (i + 1) * 5]:
                plot_fit = False
            else:
                plot_fit = True

            # shift to avoid large numbers in x-axis
            time = x + self.features[i * 5 + 3]

            if plot_fit:
                xaxis = np.linspace(min(time), max(time), 500)[:, np.newaxis]
                fitted_flux = np.array(self.evaluate(xaxis)[self.filters[i]]) * self.features[i * 5 + 4]
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
                    ext_flux = np.array(self.evaluate(xaxis_extrap)[self.filters[i]]) \
                               * self.features[i * 5 + 4]
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
