import os
import pytest
import tempfile

from resspect import TimeDomainConfiguration, LoopConfiguration

def test_create_time_domain_configuration():
    """test file path checks and other validation steps."""
    with tempfile.TemporaryDirectory() as dir_name:
        real_file_path = os.path.join(dir_name, "test.txt")
        f = open(real_file_path, "w")
        f.write("out of all the files that exist, this is one of them.")
        f.close()
        ini_files = {
            "train": real_file_path
        }

        TimeDomainConfiguration(
            days=[1],
            strategy="RandomSampling",
            path_to_features_dir=dir_name,
            output_metrics_file=os.path.join(dir_name, "blah.txt"),
            output_queried_file=os.path.join(dir_name, "blah2.txt"),
            fname_pattern=["blah_", ".csv"],
            path_to_ini_files=ini_files,
        )

        # bad path_to_features
        with pytest.raises(ValueError):
            TimeDomainConfiguration(
                days=[1],
                strategy="RandomSampling",
                path_to_features_dir="/not/real/dir",
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
                fname_pattern=["blah_", ".csv"],
                path_to_ini_files=ini_files,
            )

        # bad path_to_ini_files
        bad_ini_files = {
            "train": "/not/real/dir/bad_file.csv"
        }
        with pytest.raises(ValueError):
            TimeDomainConfiguration(
                days=[1],
                strategy="RandomSampling",
                path_to_features_dir=dir_name,
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
                fname_pattern=["blah_", ".csv"],
                path_to_ini_files=bad_ini_files,
            )

        # bad strategies
        with pytest.raises(ValueError):
            TimeDomainConfiguration(
                days=[1],
                strategy="idk whatever",
                path_to_features_dir=dir_name,
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
                fname_pattern=["blah_", ".csv"],
                path_to_ini_files=ini_files,
            )
        TimeDomainConfiguration(
                days=[1],
                strategy="QBDMI",
                path_to_features_dir=dir_name,
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
                fname_pattern=["blah_", ".csv"],
                path_to_ini_files=ini_files,
                clf_bootstrap=True
            )
        with pytest.raises(ValueError):
            TimeDomainConfiguration(
                days=[1],
                strategy="QBDMI",
                path_to_features_dir=dir_name,
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
                fname_pattern=["blah_", ".csv"],
                path_to_ini_files=ini_files,
                clf_bootstrap=False
            )

def test_to_and_from_dict():
    """dump `LoopConfiguration` to a dict and recreate."""
    with tempfile.TemporaryDirectory() as dir_name:
        real_file_path = os.path.join(dir_name, "test.txt")
        f = open(real_file_path, "w")
        f.write("out of all the files that exist, this is one of them.")
        f.close()
        ini_files = {
            "train": real_file_path
        }

        tdc1 = TimeDomainConfiguration(
            days=[1],
            strategy="RandomSampling",
            path_to_features_dir=dir_name,
            output_metrics_file=os.path.join(dir_name, "blah.txt"),
            output_queried_file=os.path.join(dir_name, "blah2.txt"),
            fname_pattern=["blah_", ".csv"],
            path_to_ini_files=ini_files,
        )
        d = tdc1.to_dict()

        tdc2 = TimeDomainConfiguration.from_dict(d)

        assert tdc1 == tdc2

def test_to_and_from_json():
    """dump `LoopConfiguration` to a dict and recreate."""
    with tempfile.TemporaryDirectory() as dir_name:
        real_file_path = os.path.join(dir_name, "test.txt")
        f = open(real_file_path, "w")
        f.write("out of all the files that exist, this is one of them.")
        f.close()
        ini_files = {
            "train": real_file_path
        }

        tdc1 = TimeDomainConfiguration(
            days=[1],
            strategy="RandomSampling",
            path_to_features_dir=dir_name,
            output_metrics_file=os.path.join(dir_name, "blah.txt"),
            output_queried_file=os.path.join(dir_name, "blah2.txt"),
            fname_pattern=["blah_", ".csv"],
            path_to_ini_files=ini_files,
        )
        json_path = os.path.join(dir_name, "test.json")
        tdc1.to_json(json_path)

        tdc2 = TimeDomainConfiguration.from_json(json_path)

        assert tdc1 == tdc2

def test_incompatibility():
    """Ensure that we fail gracefully if we try to generate another BaseConfiguration
    class from a TimeDomainConfiguration json."""
    with tempfile.TemporaryDirectory() as dir_name:
        real_file_path = os.path.join(dir_name, "test.txt")
        f = open(real_file_path, "w")
        f.write("out of all the files that exist, this is one of them.")
        f.close()
        ini_files = {
            "train": real_file_path
        }

        tdc = TimeDomainConfiguration(
            days=[1],
            strategy="RandomSampling",
            path_to_features_dir=dir_name,
            output_metrics_file=os.path.join(dir_name, "blah.txt"),
            output_queried_file=os.path.join(dir_name, "blah2.txt"),
            fname_pattern=["blah_", ".csv"],
            path_to_ini_files=ini_files,
        )
        json_path = os.path.join(dir_name, "test.json")
        tdc.to_json(json_path)

        with pytest.raises(TypeError):
            LoopConfiguration.from_json(json_path)
