"""
# Author: Etienne Russeil and Emille E. O. Ishida
#    
# created on 2 July 2022
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
from iminuit import Minuit
from iminuit.cost import LeastSquares

__all__ = ['protected_exponent', 'protected_sig', 'bump', 'fit_bump']

def protected_exponent(x):
    """
    Exponential function : cannot exceed e**10
    """
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 10, np.exp(x), np.exp(10))


def protected_sig(x):
    """
    Sigmoid function using the protected exponential function
    """
    return 1/(1+protected_exponent(-x))


def bump(x, p1, p2, p3):
    """ Parametric function, fit transient behavior
        Need to fit normalised light curves (divided by maximum flux)
    
    Parameters
    ----------
    x : np.array 
        Array of mjd translated to 0
    p1,p2,p3 : floats
        Parameters of the function
        
    Returns
    -------
    np.array
        Fitted flux array
    """
    
    # The function is by construction meant to fit light curve centered on 40
    x = x + 40
    
    return protected_sig(p1*x + p2 - protected_exponent(p3*x))



def fit_bump(time, flux, fluxerr):
    
    """
    Find best-fit parameters using iminuit least squares.
    
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
        Array is [p1, p2, p3, time_shift, max_flux]
        
         p1, p2, p3 are best fit parameter values
         time_shift is time at maximum flux
         max_flux is the maximum flux
         
    """
    
    # Center the maxflux at 0
    time_shift = -time[np.argmax(flux)] 
    time = time + time_shift
    
    # The function is by construction meant to fit light curve with flux normalised
    max_flux = np.max(flux)
    flux = flux / max_flux
    fluxerr = fluxerr / max_flux
    
    # Initial guess of the fit
    parameters_dict = {'p1':0.225, 'p2':-2.5, 'p3':0.038}

    least_squares = LeastSquares(time, flux, fluxerr, bump)
    fit = Minuit(least_squares, **parameters_dict)

    fit.migrad()

    # Create output array
    parameters = []
    for fit_values in range(len(fit.values)):
        parameters.append(fit.values[fit_values])
        
    parameters.append(time_shift)
    parameters.append(max_flux)
    
    return parameters


def main():
    return None


if __name__ == '__main__':
    main()
