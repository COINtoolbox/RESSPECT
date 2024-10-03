# Copyright 2020 resspect software
# Author: Kara Ponder
#
# created on 23 March 2020
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

import numpy as np

from astropy.cosmology import w0waCDM
from astropy import constants as const


__all__ = ['assign_cosmo', 'fish_deriv_m', 'fisher_results',
           'column_deriv_m', 'update_matrix', 'find_most_useful',
           'compare_two_fishers']


def assign_cosmo(cosmo, model=[70, 0.3, 0.7, -0.9, 0.0]):
    """Define a new cosmology model.

    Parameters
    ----------
    cosmo: astropy.cosmology Cosmology Object
        Assumes original cosmology was astropy.cosmology.w0waCDM.
    model: list (optional)
        Cosmology parameters: [H0, Om, Ode, w0, wa].
        Default is [70, 0.3, 0.7, -0.9, 0.0].
        Hard code Ob0 (Omega baryons) = 0.022

    Returns
    -------
    newcosmo: astropy.cosmology Cosmology Object
        New Cosmology Object with updated cosmology parameters
    """

    ob0=0.022
    om0=model[1]
    ode0 =model[2]

    newcosmo = cosmo.clone(name='temp cosmo', H0=model[0], Ob0=ob0,
                           Om0=om0, Ode0=ode0, w0=model[3], wa=model[4])

    return newcosmo


def fish_deriv_m(redshift, model, step, screen=False):
    """Calculates the derivatives and the base function at given redshifts.

    Parameters
    ----------
    redshift: float or list
        Redshift where derivatives will be calculated.
    model: list
        List of cosmological model parameters.
        Order is [H0, Om, Ode, w0, wa].
    step: list
        List of steps the cosmological model parameter will take when determining
         the derivative.
        If a given entry is zero, that parameter will be kept constant.
        Length must match the number of parameters in "model" variable.
    screen: bool (optional)
        Print debug options to screen. Default is False.

    Returns
    -------
    m: list
        List of theoretical distance modulus (mu) at a given redshift from
         the base cosmology.
    m_deriv: list [len(redshift), len(model)]
        List of parameter derivatives of the likelihood function
         at given redshifts.
    """
    
    Ob0 = 0.022
    Om0 = model[1]
    Ode0 = model[2]
    cosmo = w0waCDM(H0=model[0], Ob0=Ob0, Om0=Om0, Ode0=Ode0, w0=model[3], wa=model[4], name="w0waCDM")

    cosmo=assign_cosmo(cosmo, model)

    m = []
    m_deriv = []
    c = const.c.to('km/s')
    base_theory = cosmo.distmod(redshift)
    m = base_theory.value
    step_inds = np.where(step)[0] # look for non-zero step indices
    deriv = np.zeros((len(base_theory), len(model)))

    if (step_inds.size==0):
        print('No steps taken, abort')
        exit

    else:
        if screen:
            print('\n')
            print('Computing Fisher derivatives...')

        for i, stepp in enumerate(step_inds):
            if screen:
                print('we are stepping in :', model[stepp], 
                      ' with step size', step[stepp])
            cosmo = assign_cosmo(cosmo, model)

            theory = np.zeros((len(base_theory),2))
            for count,j  in enumerate([-1,1]):

                tempmodel = list(model)
                tempmodel[stepp] = model[stepp] + j*step[stepp]
                c = const.c.to('km/s')
                cosmo = assign_cosmo(cosmo, tempmodel)
                tmp = cosmo.distmod(redshift)
                theory[:,count] = tmp

            deriv[:,stepp] = (theory[:,1] - theory[:,0])/(2.*step[stepp])

    m_deriv = deriv

    return m, m_deriv


def fisher_results(redshift, mu_err):
    """Computes the Fisher Matrix. Assumes we only care about Om and w0.

    TBD: make stepvec an input.
    TBD: Priors as inputs?

    Parameters
    ----------
    redshift: list [float]
        Redshift.
    mu_err: list [float]
        Error in distance modulus.

    Returns
    -------
    sigma: list
        Error/Standard deviation of Om and w0, respectively.
    covmat: np.array [2, 2]
        Covariance matrix of Om and w0. Om is first row, w0 is second.
    """
    if any(np.array(redshift) < 0):
        raise ValueError('Redshift must be greater than zero! Galaxies must be moving away.')

    stepvec = np.array([0, 0.001, 0.00, 0.1, 0., 0.0, 0.0, 0.0])

    model = [70., 0.3, 0.7, -1.0, 0.]
    names = ['hubble', 'omega_m', 'omega_de', 'w0', 'wa']

    step_inds = np.where(stepvec)[0]
    fishermu, deriv = fish_deriv_m(redshift, model, stepvec)

    cov = np.diag(mu_err**2)
    inv_cov = np.diag(1./mu_err**2.)

    # Initialising the Fisher Matrix
    FM = np.zeros((len(step_inds), len(step_inds), len(mu_err) ))


    # Compute the Fisher matrix
    for i in range(len(step_inds)):
        # loop over variables
        for j in range(len(step_inds)):
            # loop over variables
            for k in range(len(redshift)):
                # loop over redshifts
                invcov = inv_cov[k,k]
                FM[i,j,k] = np.dot(np.dot(deriv[k, step_inds[i]], invcov), deriv[k, step_inds[j]])

    # sum over the redshift direction
    fishmat = np.sum(FM,axis=2)

    # Compute the prior matrix
    prior_vec = np.array([0.1, 0.02, 0.0006, 0.2, 0.2])
    priormat = np.diag(1./prior_vec[step_inds]**2.)

    final_FM = fishmat + priormat
    covmat = np.linalg.inv(final_FM)
    sigma = np.sqrt(covmat.diagonal())

    return sigma, covmat


def column_deriv_m(redshift, mu_err, model, step):
    """Calculate a column derivative of your model.
    
       Define a matrix P such that P_ik = 1/sigma_i del(M(i, params)/del(param_k)
       and M=model. This column matrix holds k constant.

    Parameters
    ----------
    redshift: float or list
        Redshift.
    mu_err: float or list
        Error in distance modulus.
    model: list
        List of cosmological model parameters.
        Order is [H0, om, ol, w0, wa].
    step: list
        List of steps the cosmological model paramter will take when determining
        the derivative.
        If a given entry is zero, that parameter will be kept constant.
        Length must match the number of parameters in "model" variable.

    Returns
    -------
    m: list
        List of theoretical distance modulus (mu) at a given redshift from
         the base cosmology.
    m_deriv: list
        List of derivatives for one parameter of the likelihood function
         at given redshifts.
    """

    Ob0=0.022
    Om0=model[1]
    Ode0 =model[2]
    cosmo = w0waCDM(model[0], Ob0, Om0, Ode0, model[3],model[4])

    cosmo=assign_cosmo(cosmo, model)
    m = []
    m_deriv = []
    c = const.c.to('km/s')
    base_theory = cosmo.distmod(redshift)
    m = base_theory.value
    step_inds = np.where(step)[0] # look for non-zero step indices
    deriv = np.zeros(len(model))

    if (step_inds.size==0):
        print('No steps taken, abort')
        exit

    else:

        for i, stepp in enumerate(step_inds):
            cosmo = assign_cosmo(cosmo, model)

            theory = np.zeros((1,2))
            for count,j  in enumerate([-1,1]):

                tempmodel = list(model)
                tempmodel[stepp] = model[stepp] + j*step[stepp]
                c = const.c.to('km/s')
                cosmo = assign_cosmo(cosmo, tempmodel)
                tmp = cosmo.distmod(redshift)
                theory[:,count] = tmp.value

            deriv[stepp] = (theory[:,1] - theory[:,0])/(2.*step[stepp])

    m_deriv = 1.0/mu_err * deriv

    return m, m_deriv


def update_matrix(redshift, mu_err, covmat):
    """Update matrix calculated from Hees et al (2019).
        https://ui.adsabs.harvard.edu/abs/2019ApJ...880...87H/abstract
       How much the Fisher Matrix changed given a new set of observations.

    TBD: make stepvec an input.

    Parameters
    ----------
    redshift: float or list
        Redshift.
    mu_err: float or list
        Error in distance modulus.
    covmat: np.array
        Covariance matrix from running the full Fisher Matrix analysis.

    Returns
    -------
    u: np.array (2, 2)
        Update to the Fisher Matrix covariance matrix given new observations.
    """
    stepvec = np.array([0, 0.001, 0.00, 0.1, 0.])
    step_inds = np.where(stepvec)[0]
    model = [70., 0.3, 0.7, -1.0, 0.]
    names = ['hubble', 'omega_m', 'omega_de', 'w0', 'wa']

    partialmu, partial_deriv = column_deriv_m(np.array(redshift),
                                              np.array(mu_err),
                                              model, stepvec)
    p_deriv = partial_deriv[step_inds]

    top = np.matmul(np.matmul(covmat, p_deriv[np.newaxis].T), 
                   (np.matmul(covmat, p_deriv[np.newaxis].T)).transpose())

    bottom = 1.0 + np.matmul(np.dot(np.transpose(p_deriv[np.newaxis].T), covmat),
                             p_deriv[np.newaxis].T)

    u = top/bottom

    return u


def find_most_useful(ID, redshift, error, covmat, N=None, tot=False):
    """Find which objects improve w0 the most given a list of observations.
       Uses update_matrix to find best improvements in w0.

    Parameters
    ----------
    ID: int list
        Identification number
    redshift: list
        Redshifts of new observations.
    error: list
        Uncertainties of distance modulus of new observations.
    covmat: np.array (2, 2)
        Covariance matrix from original Fisher Matrix.
    N: int (optional)
        If defined, return only the top N objects that most improve w0.
        Default is None.
    tot: bool (optional)
        Prints the total change in w0 for every object in list.
        Default is False.

    Returns
    -------
    u_stack: sorted list (len(objects), 2)
        Sorted list of the amount w0 changes for the input observations.
        Returns concatenated list of the object IDs and amount w0 changes.
    """

    u = np.zeros(len(redshift))
    for i, (red, err) in enumerate(zip(redshift, error)):
        u[i] = update_matrix(red, err, covmat)[1][1]

    u_stack = np.vstack([ID, u]).T
    u_stack = sorted(u_stack, reverse=True, key=lambda upm: upm[1])

    if tot:
         print('Change in w0:', u.sum())

    if N is None:
        return u_stack
    else:
        return u_stack[0:N]


def compare_two_fishers(data1, data2, screen=False):
    """Compare different in w0 precision given 2 different data sets.

    Parameters
    ----------
    data1: np.array 
        List of redshift, mu, error on distance modulus for data set 1.
        Order is [z, mu, mu_err]. Number of lines == minimum 3,
        number of columns == number of objects.
    data2: list
        List of redshift, mu, error on distance modulus for data set 2.
        Order is [z, mu, mu_err]. Number of lines == minimum 3,
        number of columns == number of objects.
    screen: bool (optional)
        Print debug options to screen. Default is False.

    Returns
    -------
    delta_sig: float
        Difference in precision on w0 between data set 1 and data set 2.
        If positive, data set 2 has tighter constraints than data set 1.
        If negative, data set 1 has tighter constraints than data set 2.
    """

    sig_fr1, cov1 = fisher_results(data1[0], data1[2])
    sig_fr2, cov2 = fisher_results(data2[0], data2[2])

    if screen:
        print('error for Om for 1:', sig_fr1[0])
        print('error for w0 for 1:', sig_fr1[1])
        print('error for Om for 2:', sig_fr2[0])
        print('error for w0 for 2:', sig_fr2[1])

    delta_sigw = sig_fr1[1] - sig_fr2[1]

    return delta_sigw


def main():
    return None


if __name__ == '__main__':
    main()
