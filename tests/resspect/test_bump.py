"""
Tests related to bump.py module.
"""

import numpy as np
import pandas as pd

from resspect.bump import bump, fit_bump, protected_exponent, protected_sig


def test_protected_exponent():
    """Test the protected_exponent() function."""
    values = np.arange(0, 20)
    results = protected_exponent(values)

    # Input values 0-10 should all return exp(x). Anything above that should
    # return exp(10.0).
    np.testing.assert_allclose(results[:11], np.exp(np.arange(0, 11)))
    np.testing.assert_allclose(results[10:], [np.exp(10.0)] * 10)


def test_protected_sig():
    """Test the protected_sig() function."""
    values = np.arange(-20, 10)
    results = protected_sig(values)

    expected_upper = 1.0 / (1.0 + np.exp(np.arange(10, -10, -1)))
    expected_lower = 1.0 / (1.0 + np.exp(np.full(10, 10.0)))

    # Input values [-20, -10] should return 1.0 / (1.0 + exp(10.0))
    # and input values [-10, 10] should return 1.0 / (1.0 + exp(-x))
    np.testing.assert_allclose(results[:10], expected_lower)
    np.testing.assert_allclose(results[10:], expected_upper)


def test_bump():
    """Test the Bump function evaluation."""
    time = np.arange(-1, 5, 1)
    p1 = 0.225
    p2 = -2.5
    p3 = 0.038

    res = bump(time, p1, p2, p3)
    assert not np.isnan(res).any()

    # These were manually computed using the function and so this currently
    # only will detect future changes in behavior (breakages).
    expected = [0.86683499, 0.87300292, 0.87822063, 0.88254351, 0.88601519, 0.88866704]
    np.testing.assert_allclose(res, expected)

    
def test_fit_bump(test_data_path):
    """
    Test fit to Bump parametrization.
    """
    fname = test_data_path / 'lc_mjd_flux.csv'
    data = pd.read_csv(fname)
    
    time = data['mjd'].values
    flux = data['flux'].values
    fluxerr = np.array([1])
    
    res = fit_bump(time, flux, fluxerr)
    
    assert not np.isnan(res).any()


if __name__ == '__main__':
    pytest.main()
