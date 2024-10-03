import pytest

from resspect.lightcurves_utils import (
    get_files_list,
    get_snpcc_sntype,
    read_file,
    SNPCC_LC_MAPPINGS,
)


def test_get_snpcc_sntype():
    """Test that the get_snpcc_sntype() returns the mappings defined in SNPCC_LC_MAPPINGS."""
    assert get_snpcc_sntype(0) == "Ia"
    for value in SNPCC_LC_MAPPINGS["snibc"]:
        assert get_snpcc_sntype(value) == "Ibc"
    for value in SNPCC_LC_MAPPINGS["snii"]:
        assert get_snpcc_sntype(value) == "II"
    
    with pytest.raises(ValueError):
        _ = get_snpcc_sntype(1000)


def test_read_file(test_des_data_path, test_des_lc_file):
    """Test that we can read in a DES lightcurve file."""
    lines = read_file(test_des_lc_file)
    assert len(lines) > 0

    # Check that the first line is 'SURVEY: DES'.
    assert lines[0][0] == "SURVEY:"
    assert lines[0][1] == "DES"

    with pytest.raises(FileNotFoundError):
        _ = read_file(test_des_data_path / "no_such_file.txt")


def test_get_files_list(test_data_path):
    """Test that we can list files in a directory."""
    assert len(get_files_list(test_data_path, "")) > 0
    assert len(get_files_list(test_data_path, "plasticc_lightcurves.csv")) == 1
    assert len(get_files_list(test_data_path, "no_such_file")) == 0


if __name__ == '__main__':  
    pytest.main()
