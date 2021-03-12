#!/usr/bin/env python
"""
    Test cosmo_metric_utils

    PyTest for cosmo_metric_utils.

    """
import pytest

## TBD: simulate data to use to test the metric.

from resspect import cosmo_metric_utils

import astropy as ap
import numpy as np
import pandas as pd

from astropy.cosmology import w0waCDM
from astropy import constants as const

def make_fake_data():
    return


# multiple unit tests?
def test_assign_cosmo():
    """
        Test to make sure it reassigns the cosmology
    """
    from resspect import cosmo_metric_utils as cmu

    cosmo = w0waCDM(70, 0.3, 0.7, -0.9, 0.0)
    updated_cosmo = cmu.assign_cosmo(cosmo, model=[72, 0.29, 0.71, -1, 0.0])

    assert int(updated_cosmo.H0.value) == int(72)
    assert hasattr(updated_cosmo, 'distmod')
    assert isinstance(updated_cosmo, w0waCDM)

