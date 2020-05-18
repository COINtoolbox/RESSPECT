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


import astropy as ap
import numpy as np
import pandas as pd

from astropy.cosmology import Planck15, Flatw0waCDM
from astropy.cosmology import w0waCDM
from astropy import constants as const

__all__ = ['assign_cosmo', 'fish_deriv_m', 'fisher_results',
           'column_deriv_m', 'update_matrix', 'find_most_useful',
           'full_check', 'compare_two_fishers']


# original wa was 0.2
# We start by having a model that will change the cosmology within the Fisher matrix
def assign_cosmo(cosmo, model=[70, 0.3,0.7, -0.9, 0.0]):
    """Do something.

    Parameters
    ----------
    cosmo:  XXX
        XXXX
    model: list (optional)
        Cosmology parameters: [H0, om, ol, w0, wa].
        Default is [70, 0.3,0.7, -0.9, 0.0].

    Returns
    -------
    newcosmo: XX
        XXX
    """

    ob0=0.022
    om0=model[1]
    ode0 =model[2]

    newcosmo = cosmo.clone(name='temp cosmo', H0=model[0], Ob0=ob0,
                           Om0=om0, Ode0=ode0, w0=model[3], wa=model[4])

    return newcosmo


# Define code that returns the mu and the Fisher matrix
def fish_deriv_m(redshift, model, step, screen=False):
    """Calculates the derivatives and the base function at given redshifts.

    Parameters
    ----------
    redshift: float
        Redshift where derivatives will be calculated.
    model: list
        List of cosmological model parameters.
        Order is [H0, om, ol, w0, wa].
    step: XXX
        XXXXX
    screen: bool (optional)
        Print debug options to screen.
        Default is False.

    Returns
    -------
    m: XX
        XXXX
    m_deriv: XXX
        XXXX
    """
    
    Ob0=0.022
    Om0=model[1]
    Ode0 =model[2]
    cosmo = w0waCDM(model[0], Ob0, Om0, Ode0, model[3],model[4])

    cosmo=assign_cosmo(cosmo, model)

    #print cosmo.Ok0
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
                #print tempmodel
                c = const.c.to('km/s')
                cosmo = assign_cosmo(cosmo, tempmodel)
                tmp = cosmo.distmod(redshift)
                theory[:,count] = tmp

            deriv[:,stepp] = (theory[:,1] - theory[:,0])/(2.*step[stepp])

    m_deriv = deriv

    return m, m_deriv


def fisher_results(redshift, mu_err):
    """Do something.


    ** original doc string, there are some missing inputs in the function definition? **
    Inputs are redshift, probability, mu (distance modulus), mu_err in arrays.
    TBD: make stepvec an input.

    Parameters
    ----------
    redshift: float
        Redshift.
    mu_err: float
        Error in distance modulus.

    Returns
    -------
    sigma: XXX
        XXXX
    covmat: XXX
        XXXX
    """
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


def column_deriv_m(redshift, sigma, model, step):
    """Do something.

    ** original docstring: the description seems exactly the same as was given for fish_deriv_m **
    takes the model vector - for now [h0,om,ok,w0,wa], step vector (0 if not step) \
    data vector and gives back the derivs and the base function value at those \
    redshifts

    Parameters
    ----------
    redshift: float
        Redshift.
    sigma: float
        XXXX
    model: list
         List of cosmological model parameters.
        Order is [H0, om, ol, w0, wa].
    step: XX
        XXXX

    Returns
    -------
    m: XXX
        XXXXXX
    m_deriv: XXX
        XXXXX    
    """

    Ob0=0.022
    Om0=model[1]
    Ode0 =model[2]
    cosmo = w0waCDM(model[0], Ob0, Om0, Ode0, model[3],model[4])

    cosmo=assign_cosmo(cosmo, model)
    #print cosmo.Ok0
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
                #print tempmodel
                c = const.c.to('km/s')
                cosmo = assign_cosmo(cosmo, tempmodel)
                tmp = cosmo.distmod(redshift)
                theory[:,count] = tmp.value

            deriv[stepp] = (theory[:,1] - theory[:,0])/(2.*step[stepp])

    m_deriv = 1.0/sigma * deriv

    return m, m_deriv


def update_matrix(redshift, sigma, covmat):
    """Do something.

    ** original docstring **
    TBD: generalize this back to take in step and model.

    Parameters
    ----------
    redshift: float
        Redshift.
    sigma: float
        XXXX
    covmat: XXXX
        XXXXXX

    Returns
    -------
    XXXXX
    """
    stepvec = np.array([0, 0.001, 0.00, 0.1, 0.])
    step_inds = np.where(stepvec)[0]
    model = [70., 0.3, 0.7, -1.0, 0.]
    names = ['hubble', 'omega_m', 'omega_de', 'w0', 'wa']

    partialmu, partial_deriv = column_deriv_m(np.array(redshift),
                                              np.array(sigma),
                                              model, stepvec)
    p_deriv = partial_deriv[step_inds]

    top = np.matmul(np.matmul(covmat, p_deriv[np.newaxis].T), 
                   (np.matmul(covmat, p_deriv[np.newaxis].T)).transpose())

    bottom = 1.0 + np.matmul(np.dot(np.transpose(p_deriv[np.newaxis].T), covmat),
                             p_deriv[np.newaxis].T)

    return top/bottom


def find_most_useful(ID, redshift, error, covmat, N, tot=False):
    """Do something.

    Parameters
    ----------
    ID: int 
        Identification number
    redshift: XXXX 
        Redshifts XXX
    error: XXXX
        Uncertainties of distance modulus
    covmat: XXXX
        Covariance matrix from original Fisher Matrix
    N: int
        Top number to search
    tot: bool (optional)
        XXXX. Default is False.

    Returns
    -------
    u_stack: XXX
        XXX
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

def full_check(redshift, error, covmat, N):
    """Do something.

    Parameters
    ----------
    redshift: XXX
        XXXXX
    error: XXX
        XXXX
    covmat: XXXX
        XXX
    N: int
        Top number to search

    Returns
    -------
    """
    ndim = 5

    full_check = np.zeros((ndim, ndim, len(test)))#all_red) ))
    for i, om in enumerate(np.linspace(0.2, 0.4, ndim)):
        for j, w0 in enumerate(np.linspace(-0.9, -1.1, ndim)):
            print(om, w0)
            model = [70., om, 0.7, w0, 0.]
            for k, (red, err) in enumerate(zip(redshift, error)):
                full_check[i, j, k] = update_matrix(red, err, covmat)[1][1]

    arg_sort_full_check = np.argsort(full_check)

    largest = []
    number = []
    for l in np.arange(0,5):
        for n in np.arange(0,5):
            arg_sort_l = np.argsort(full_check[l][n])
            arg_sort_short = arg_sort_l[-10:]
            largest.append(arg_sort_short)
            number.append(full_check[l][n][arg_sort_short])

    numberf = np.array(number).flatten()
    arg_sort_numberf = np.argsort(numberf)

    return arg_sort_numberf[-N:]


def compare_two_fishers(data1, data2, screen=False):
    """Do something. 
    
    ** original docstring ***
        data should be of the form redshift, mu, and error on distance modulus
        
        if positive, data set 2 has tighter constraints than data set 1
        if negative, data set 1 has tighter constraints than data set 2

    Parameters
    ----------
    data1: XXX
        XXXX
    data2: XXX
        XXXXX
    screen: bool (optional)
        Print debug options to screen.
        Default is False.

    Returns
    -------
    delta_sig: XXX
        XXXX
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
