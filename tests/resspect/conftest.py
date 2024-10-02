"""
PyTest Configuration file with functions that can be used by any test within
this folder.
"""
import os
import pytest
from pathlib import Path


@pytest.fixture
def test_data_path():
    return Path(__file__).parent.parent.parent / "data" / "tests"


@pytest.fixture
def test_des_data_path(test_data_path):
    return test_data_path / "DES_data"


@pytest.fixture
def test_des_lc_file(test_des_data_path):
    return test_des_data_path / "DES_SN848233.DAT"


@pytest.fixture(scope="session")
def base_temp(tmp_path_factory):
    """
    Created a place to store the tests outputs. Can be set using the command
    line --basetemp (WARNING: WILL DELETE ALL OF ITS CURRENT CONTENT)

    Parameters
    ----------
    tmp_path_factory : fixture
        PyTest's build-in fixture.

    Returns
    -------
    str : Path for the tests results for the current session
    """
    return tmp_path_factory.mktemp("resspect-tests-")


@pytest.fixture(scope='module')
def path_to_output_data(request, base_temp):
    """
    PyTest fixture that creates a temporary folder to save tests outputs. You
    can set the base directory by passing the ``--basetemp=mydir/`` argument to
    the PyTest call (See [Pytest - Temporary Directories and Files][1]).

    [1]: https://docs.pytest.org/en/stable/tmpdir.html#temporary-directories-and-files

    Returns
    -------
    str
        Path to the output data.

    Raises
    ------
    IOError
        If output path does not exits.
    """
    module_path = request.module.__name__.split('.')
    module_path = [item for item in module_path if item not in "tests"]
    path = os.path.join(base_temp, *module_path)
    os.makedirs(path, exist_ok=True)

    return path


# noinspection PyUnusedLocal
def pytest_report_header(config):
    """ Adds the test folder to the Pytest Header """
    return f"RESSPECT_TEST directory: {os.getenv('RESSPECT_TEST_TEST')}"