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


def bazin(time, a, b, t0, tfall, trise):
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
    trise: float
        Characteristic raise time

    Returns
    -------
    array_like
        response variable (flux)

    """
    with np.errstate(over='ignore', invalid='ignore'):
        X = np.exp(-(time - t0) / tfall) / (1 + np.exp(-(time - t0) / trise))
        return a * X + b

def bazinr(time, a, b, t0, tfall, r):
    """
    A wrapper function for bazin() which replaces trise by r = tfall/trise.

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
        Ratio of tfall to trise. This ratio is enforced to be >1.

    Returns
    -------
    array_like
        response variable (flux)

    """

    trise = tfall/r    
    res = bazin(time, a, b, t0, tfall, trise)
    
    if max(res) < 10e10:
        return res
    else:
        return np.array([item if item < 10e10 else 10e10 for item in res])

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

    return abs(flux - bazinr(time, *params)) / fluxerr


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
    flux_max = flux[imax]
    
    # Parameter bounds
    a_bounds = [1.e-3, 10e10]
    b_bounds = [-10e10, 10e10]
    t0_bounds = [-0.5*time.max(), 1.5*time.max()]
    tfall_bounds = [1.e-3, 10e10]
    r_bounds = [1, 10e10]

    # Parameter guess
    a_guess = 2*flux_max
    b_guess = 0
    t0_guess = time[imax]
    
    tfall_guess = time[imax-2:imax+2].std()/2
    if np.isnan(tfall_guess):
        tfall_guess = time[imax-1:imax+1].std()/2
        if np.isnan(tfall_guess):
            tfall_guess=50
    if tfall_guess<1:
        tfall_guess=50

    r_guess = 2

    # Clip guesses to stay in bound
    a_guess = np.clip(a=a_guess,a_min=a_bounds[0],a_max=a_bounds[1])
    b_guess = np.clip(a=b_guess,a_min=b_bounds[0],a_max=b_bounds[1])
    t0_guess = np.clip(a=t0_guess,a_min=t0_bounds[0],a_max=t0_bounds[1])
    tfall_guess = np.clip(a=tfall_guess,a_min=tfall_bounds[0],a_max=tfall_bounds[1])
    r_guess = np.clip(a=r_guess,a_min=r_bounds[0],a_max=r_bounds[1])


    guess = [a_guess,b_guess,t0_guess,tfall_guess,r_guess]


    bounds = [[a_bounds[0], b_bounds[0], t0_bounds[0], tfall_bounds[0], r_bounds[0]],
              [a_bounds[1], b_bounds[1], t0_bounds[1], tfall_bounds[1], r_bounds[1]]]
    
    result = least_squares(errfunc, guess, args=(time, flux, fluxerr), method='trf', loss='linear',bounds=bounds)
    
    a_fit,b_fit,t0_fit,tfall_fit,r_fit = result.x
    trise_fit = tfall_fit/r_fit
    final_result = np.array([a_fit,b_fit,t0_fit,tfall_fit,trise_fit])
    
    return final_result

def main():
    return None


if __name__ == '__main__':
    main()
