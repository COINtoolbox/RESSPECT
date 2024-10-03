#!/usr/bin/env python
"""
    Test cosmo_metric_utils

    PyTest for cosmo_metric_utils.

    """

## TBD: simulate data to use to test the metric.

from resspect import cosmo_metric_utils as cmu

import numpy as np

from astropy.cosmology import w0waCDM


# multiple unit tests?
def test_assign_cosmo():
    """
        Test to make sure it reassigns the cosmology
    """
    cosmo = w0waCDM(70, 0.3, 0.7, -0.9, 0.0, name='w0waCDM')
    updated_cosmo = cmu.assign_cosmo(cosmo, model=[72, 0.29, 0.71, -1, 0.0])

    assert int(updated_cosmo.H0.value) == int(72)
    assert hasattr(updated_cosmo, 'distmod')
    assert isinstance(updated_cosmo, w0waCDM)


def test_fish_deriv_m():
    redshift = np.arange(0.2, 1.5, 0.1)
    step = np.array([0, 0.001, 0.00, 0.1, 0., 0.0, 0.0, 0.0])
    m, m_deriv = cmu.fish_deriv_m(redshift, [72, 0.29, 0.71, -1, 0.0], step)

    assert np.shape(m) == np.shape(redshift)
    assert len(m_deriv) > 0

    assert not np.isnan(m).any()
    assert not np.isnan(m_deriv).any()
