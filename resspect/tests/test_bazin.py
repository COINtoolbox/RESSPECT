"""
Tests related to bazin.py module.
"""

import numpy as np
import os
import pytest

from pandas import read_csv

from resspect import testing


def test_bazin():
    """
    Test the Bazin function evaluation.
    """

    from resspect import bazin
    
    time = 3
    a = 1
    b = 1
    t0 = 10
    tfall = 3
    trise = 4
    r = tfall/trise

    res = bazin(time, a, b, t0, tfall, r)
    
    assert not np.isnan(res).any()


def test_errfunc():
    """
    Test the error between calculates and observed error.
    """

    from resspect import bazin, errfunc
    
    # input for bazin
    time = np.arange(0, 50, 3.5)
    a = 1
    b = 1
    t0 = 10
    tfall = 3
    trise = 4
    r = tfall/trise

    # calculate fiducial flux values
    flux_fid = bazin(time, a, b, t0, tfall, r)
    
    # add noise
    flux = [np.random.normal(loc=item, scale=0.01) for item in flux_fid]
    
    # construct parameters vector
    params = [a, b, t0, tfall, r]

    # HAVE TO ADD FLUXERR BELOW AS ERFFUNC WAS CHANGED
    res = errfunc(params, time, flux)
    
    assert not np.isnan(res).any()
    assert np.all(res > 0)
    
    
def test_fit_scipy():
    """
    Test the scipy fit to Bazin parametrization.
    """
    from resspect import fit_scipy
    
    fname = testing.download_data('tests/lc_mjd_flux.csv')
    data = read_csv(fname)
    
    time = data['mjd'].values
    flux = data['flux'].values
    fluxerr = data['fluxerr'].values
    
    res = fit_scipy(time, flux,fluxerr)
    
    assert not np.isnan(res).any()


if __name__ == '__main__':
    pytest.main()
