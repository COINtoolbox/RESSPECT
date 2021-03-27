
from resspect import testing


def test_download_data():
    path = testing.download_data("SIMGEN_PUBLIC_DES.tar.gz")

    assert isinstance(path, str)
    assert "SIMGEN_PUBLIC_DES.tar.gz" in path
