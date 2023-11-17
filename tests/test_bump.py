"""
Tests related to bump.py module.
"""

import numpy as np
import os
import pytest

from pandas import read_csv

def test_bump():
    """
    Test the Bump function evaluation.
    """

    from resspect.utils.bump_utils import bump
    
    time = np.array([0])
    p1 = 0.225
    p2 = -2.5
    p3 = 0.038
    
    res = bump(time, p1, p2, p3)
    
    assert not np.isnan(res).any()
   
    
def test_fit_bump(test_data_path):
    """
    Test fit to Bump parametrization.
    """
    fname = test_data_path / 'lc_mjd_flux.csv'
    data = read_csv(fname)
    
    time = data['mjd'].values
    flux = data['flux'].values
    fluxerr = np.array([1])
    
    res = fit_bump(time, flux, fluxerr)
    
    assert not np.isnan(res).any()


if __name__ == '__main__':
    pytest.main()
