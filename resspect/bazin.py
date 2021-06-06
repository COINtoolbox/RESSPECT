"""
# Author: Alexandre Boucaud and Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 25 January 2018
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
"""

import numpy as np
from scipy.optimize import least_squares

__all__ = ['bazin', 'errfunc', 'fit_scipy']


def bazin(time, a, b, t0, tfall, r):
    """
    Parametric light curve function proposed by Bazin et al., 2009.

    Parameters
    ----------
    time : np.array
        exploratory variable (time of observation)
    a: float
        Normalization parameter
    b: float
        Shift parameter
    t0: float
        Time of maximum
    tfall: float
        Characteristic decline time
    r: float
        Ratio of the characteristic decline time (tfall) and the characteristic raise time (trise). This ratio is enforced to be >1.


    Returns
    -------
    array_like
        response variable (flux)

    """
    with np.errstate(over='ignore', invalid='ignore'):
        e = np.exp(-(time - t0) / tfall)
        X = e / (1  + e**r)
        return a * X + b

def errfunc(params, time, flux, fluxerr):
    """
    Absolute difference between theoretical and measured flux.

    Parameters
    ----------
    params : list of float
        light curve parameters: (a, b, t0, tfall, r)
    time : array_like
        exploratory variable (time of observation)
    flux : array_like
        response variable (measured flux)
    fluxerr : array_like
        error in response variable (flux)

    Returns
    -------
    diff : float
        absolute difference between theoretical and observed flux

    """

    return abs(flux - bazin(time, *params)) / fluxerr


def fit_scipy(time, flux, fluxerr):
    """
    Find best-fit parameters using scipy.least_squares.

    Parameters
    ----------
    time : array_like
        exploratory variable (time of observation)
    flux : array_like
        response variable (measured flux)
    fluxerr : array_like
        error in response variable (flux)

    Returns
    -------
    output : np.ndarray of floats
        best fit parameter values

    """
    flux = np.asarray(flux)
    imax = flux.argmax()
    t0 = time[imax]
    try:
        scale = time[imax-2:imax+2].std()/2
        assert(not np.isnan(scale))
    except:
        try:
            scale = time[imax-1:imax+1].std()/2
            assert(not np.isnan(scale))
        except:
            scale=50
    if scale<1:
        scale=50
    A = 2*flux.max()
    #guess = [A, 0, t0, scale, scale/2]
    guess = [A, 0, t0, scale, 2.]
    result = least_squares(errfunc, guess, args=(time, flux, fluxerr), method='trf', loss='linear',\
                                  bounds=([1.e-3, -np.inf, 0, 1.e-3, 1], np.inf))

    return result.x

def main():
    return None


if __name__ == '__main__':
    main()
