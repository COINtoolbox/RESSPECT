"""
Tests related to fit_lightcurves.py module.
"""

import os
import pytest


def test_can_read_light_curve(setup_test):
    """
    Test that we can read a file containing a light curve.

    Parameters
    ----------
    setup_test : fixture
        Custom fixture that unpacks the input data in the test directory.
    """
    from resspect.fit_lightcurves import LightCurve

    path_to_lc = os.path.join(setup_test, "SIMGEN_PUBLIC_DES/DES_SN848233.DAT")
    lc = LightCurve()
    lc.load_snpcc_lc(path_to_lc)


if __name__ == '__main__':
    pytest.main()
