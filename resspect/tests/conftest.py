"""
PyTest Configuration file with functions that can be used by any test within
this folder.
"""

import os
import shutil
import pytest
import tarfile


@pytest.fixture(scope='session')
def path_to_test_data(env_var="RESSPECT_TEST"):
    """
    PyTest fixture that creates a folder inside $RESSPECT_TEST, copies the tar
    file inside it, decompress the tar file and returns the path to where the
    input files for a given test module live.

    Parameters
    ----------
    env_var : str
        Environment variable that contains the root path to the input data
        directory.

    Returns
    -------
    str:
        Path to the input data directory.
    """
    data_dir = os.getenv(env_var)

    if data_dir is None:
        pytest.skip('Environment variable not set: $RESSPECT_TEST')

    return os.path.expanduser(data_dir.strip())
