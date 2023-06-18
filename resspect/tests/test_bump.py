"""
Tests related to bump.py module.
"""

import numpy as np
import os
import pytest

from pandas import read_csv

from resspect import testing


def test_bump():
    """
    Test the Bump function evaluation.
    """

    from resspect.utils.bump_utils import bump
    
    time = np.array([0])
    p1 = 0.225
    p2 = -2.5
    p3 = 0.038
    
    res = bump(p1, p2, p3)
    
    assert not np.isnan(res).any()
   
    
def test_fit_bump():
    """
    Test fit to Bump parametrization.
    """
    from resspect import fit_bump
    
    fname = testing.download_data('tests/lc_mjd_flux.csv')
    data = read_csv(fname)
    
    time = data['mjd'].values
    flux = data['flux'].values
    fluxerr = np.array([1])
    
    res = fit_scipy(time, flux,fluxerr)
    
    assert not np.isnan(res).any()


if __name__ == '__main__':
    pytest.main()
