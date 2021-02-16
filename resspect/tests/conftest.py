"""
PyTest Configuration file with functions that can be used by any test within
this folder.
"""

import os
import shutil
import pytest
import tarfile


@pytest.fixture(scope='session')
def setup_test(env_var="RESSPECT_TEST"):
    """
    PyTest fixture that creates a folder inside $RESSPECT_TEST, copies the tar
    file inside it, decompress the tar file and returns the path to where the
    input files for a given test module live.

    Parameters
    ----------
    env_var : str
        Environment variable that contains the root path to the input data.

    Returns
    -------
    str:
        Path to the input files.
    """
    path_to_test_data = os.getenv(env_var)

    if path_to_test_data is None:
        pytest.skip('Environment variable not set: $RESSPECT_TEST')

    path_to_test_data = os.path.expanduser(path_to_test_data).strip()

    # Clean up test folder to start fresh every time
    if os.path.exists(path_to_test_data):
        shutil.rmtree(path_to_test_data)

    # Create sub-folders
    os.makedirs(path_to_test_data, exist_ok=True)
    os.makedirs(os.path.join(path_to_test_data, "plots"), exist_ok=True)
    os.makedirs(os.path.join(path_to_test_data, "results"), exist_ok=True)

    # Decompress data
    cwd = os.getcwd()
    os.chdir(path_to_test_data)

    tar = tarfile.open(
        os.path.join(cwd, "data/SIMGEN_PUBLIC_DES.tar.gz"), "r:gz")

    tar.extractall()
    tar.close()

    return path_to_test_data
