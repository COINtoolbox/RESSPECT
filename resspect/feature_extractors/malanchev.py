# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
#
# created on 9 April 2023
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

## This modules uses features from https://github.com/light-curve/light-curve

import numpy as np
import light_curve as licu
from resspect.feature_extractors.light_curve import LightCurve

__all__ = ['MalanchevFeatureExtractor']

class MalanchevFeatureExtractor(LightCurve):
    def __init__(self):
        super().__init__()
        self.features_names = ['anderson_darling_normal',
                               'inter_percentile_range_5',
                               'chi2',
                               'stetson_K',
                               'weighted_mean',
                               'duration',
                               'otsu_mean_diff',
                               'otsu_std_lower',
                               'otsu_std_upper',
                               'otsu_lower_to_all_ratio',
                               'linear_fit_slope',
                               'linear_fit_slope_sigma',
                               'linear_fit_reduced_chi2']

        
    def fit(self, band: str) -> np.ndarray:
        """
        Extracts malanchev-light-curve features for one filter.

        Parameters
        ----------
        band: str
            Choice of broad band filter

        Returns
        -------
        mlc_param: np.ndarray
            Features from malanchev-light-curve:
            ['anderson_darling_normal',
                               'inter_percentile_range_5',
                               'chi2',
                               'stetson_K',
                               'weighted_mean',
                               'duration',
                               'otsu_mean_diff',
                               'otsu_std_lower',
                               'otsu_std_upper',
                               'otsu_lower_to_all_ratio',
                               'linear_fit_slope',
                               'linear_fit_slope_sigma',
                               'linear_fit_reduced_chi2'].
        """

        # build filter flag
        band_indices = self.photometry['band'] == band

        extractor = licu.Extractor(licu.AndersonDarlingNormal(),
                                   licu.InterPercentileRange(0.05),
                                   licu.ReducedChi2(),
                                   licu.StetsonK(),
                                   licu.WeightedMean(),
                                   licu.Duration(),
                                   licu.OtsuSplit(),
                                   licu.LinearFit())

        # get info for this filter
        time = self.photometry['mjd'].values[band_indices]
        idx = np.argsort(time)
        flux = self.photometry['flux'].values[band_indices]
        flux_error = self.photometry['fluxerr'].values[band_indices]

        time = time[idx].astype(float)
        flux = flux[idx].astype(float)
        flux_error = flux_error[idx].astype(float)

        return extractor(time, flux, flux_error,
                                  fill_value=-999,
                                  sorted=True,
                                  check=False)



    def fit_all_points(self):
        """
        Extracts Malanchev's light_curve features for all data points in all filters together.
        
        Populates self.photometry with features:
            AndersonDarlingNormal, InterPercentileRange(0.05),
            ReducedChi2, StetsonK, WeightedMean, Duration, OtsuSplit,
            LinearFit.

        Returns
        -------
        mlc_param: np.ndarray
            Features from malanchev-light-curve:
            ['anderson_darling_normal',
                               'inter_percentile_range_5',
                               'chi2',
                               'stetson_K',
                               'weighted_mean',
                               'duration',
                               'otsu_mean_diff',
                               'otsu_std_lower',
                               'otsu_std_upper',
                               'otsu_lower_to_all_ratio',
                               'linear_fit_slope',
                               'linear_fit_slope_sigma',
                               'linear_fit_reduced_chi2'].
        """
        
        extractor = licu.Extractor(licu.AndersonDarlingNormal(),
                                   licu.InterPercentileRange(0.05), 
                                   licu.ReducedChi2(),
                                   licu.StetsonK(), 
                                   licu.WeightedMean(), 
                                   licu.Duration(), 
                                   licu.OtsuSplit(), 
                                   licu.LinearFit())

        # get info for this filter
        time = self.photometry['mjd'].values
        idx = np.argsort(time)
        
        time = time[idx].astype(float)
        flux = self.photometry['flux'].values[idx].astype(float)
        flux_error = self.photometry['fluxerr'].values[idx].astype(float)

        
        return extractor(time, flux, flux_error,
                             fill_value = -999, 
                             sorted = True, 
                             check = False)

    def fit_all(self):
        """
        Performs malanchev-light-curve feature extraction for all filters independently and concatenate results.
        Populates the attributes: mlcfeatures.
        """
        default_mlcfeatures = ['None'] * len(self.features_names)

        if self.photometry.shape[0] < 1:
            self.features = ['None'] * len(
                self.features_names) * len(self.filters)

        elif 'None' not in self.features:
            self.features = []
            for each_band in self.filters:
                best_fit = self.fit(band=each_band)
                if (best_fit.size > 0) and (not np.isnan(np.sum(best_fit))):
                    self.features.extend(best_fit.tolist())
                else:
                    self.features.extend(default_mlcfeatures)
        else:
            self.features.extend(default_mlcfeatures)