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
    path = os.getenv(env_var)

    if path is None:
        pytest.skip('Environment variable not set: $RESSPECT_TEST')

    path = os.path.expanduser(path.strip())

    if not os.path.exists(path):
        raise FileNotFoundError(
            " Could not find path to input data:\n    {:s}".format(path))

    if not os.access(path, os.R_OK):
        pytest.fail('\n  Path to input test data exists but is not accessible: '
                    '\n    {:s}'.format(path))

    print(f"Using the following path to the inputs:\n  {path}\n")
    return path
