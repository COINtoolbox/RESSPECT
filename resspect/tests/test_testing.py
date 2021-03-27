
import pytest
from resspect import testing

test_cases = ("SIMGEN_PUBLIC_DES.tar.gz", "tests/DES_SN848233.DAT")


@pytest.mark.parametrize("filename", test_cases)
def test_download_data(filename):
    path = testing.download_data(filename)

    filename = filename.split()[-1]

    assert isinstance(path, str)
    assert filename in path
