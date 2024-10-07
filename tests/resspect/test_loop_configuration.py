import os
import pytest
import tempfile

from resspect import LoopConfiguration

def test_create_loop_configuration():
    """test file path checks and other validation steps."""
    with tempfile.TemporaryDirectory() as dir_name:
        real_file_path = os.path.join(dir_name, "test.txt")
        f = open(real_file_path, "w")
        f.write("out of all the files that exist, this is one of them.")
        f.close()

        LoopConfiguration(
            nloops=1,
            strategy="RandomSampling",
            path_to_features=real_file_path,
            output_metrics_file=os.path.join(dir_name, "blah.txt"),
            output_queried_file=os.path.join(dir_name, "blah2.txt"),
        )

        ptf_dict = {
            "train": real_file_path,
            "test": real_file_path,
        }
        LoopConfiguration(
            nloops=1,
            strategy="RandomSampling",
            path_to_features=ptf_dict,
            output_metrics_file=os.path.join(dir_name, "blah.txt"),
            output_queried_file=os.path.join(dir_name, "blah2.txt"),
        )

        # bad path_to_features
        with pytest.raises(ValueError):
            LoopConfiguration(
                nloops=1,
                strategy="RandomSampling",
                path_to_features="fake.txt",
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
            )
        bad_ptf_dict = {
            "train": "fake.txt",
            "test": "fake2.txt",
        }
        with pytest.raises(ValueError):
            LoopConfiguration(
                nloops=1,
                strategy="RandomSampling",
                path_to_features=bad_ptf_dict,
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
            )
        with pytest.raises(ValueError):
            LoopConfiguration(
                nloops=1,
                strategy="RandomSampling",
                path_to_features=0,
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
            )

        # SNANA_types
        LoopConfiguration(
            nloops=1,
            strategy="RandomSampling",
            path_to_features=real_file_path,
            output_metrics_file=os.path.join(dir_name, "blah.txt"),
            output_queried_file=os.path.join(dir_name, "blah2.txt"),
            SNANA_types=True,
            metadata_fname=real_file_path,
        )
        with pytest.raises(ValueError):
            LoopConfiguration(
                nloops=1,
                strategy="RandomSampling",
                path_to_features=real_file_path,
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
                SNANA_types=True,
            )
        with pytest.raises(ValueError):
            LoopConfiguration(
                nloops=1,
                strategy="RandomSampling",
                path_to_features=real_file_path,
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
                SNANA_types=True,
                metadata_fname="fake.txt"
            )

        # save_predictions
        LoopConfiguration(
            nloops=1,
            strategy="RandomSampling",
            path_to_features=real_file_path,
            output_metrics_file=os.path.join(dir_name, "blah.txt"),
            output_queried_file=os.path.join(dir_name, "blah2.txt"),
            save_predictions=True,
            pred_dir=dir_name,
        )
        with pytest.raises(ValueError):
            LoopConfiguration(
                nloops=1,
                strategy="RandomSampling",
                path_to_features=real_file_path,
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
                save_predictions=True,
            )
        with pytest.raises(ValueError):
            LoopConfiguration(
                nloops=1,
                strategy="RandomSampling",
                path_to_features=real_file_path,
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
                save_predictions=True,
                pred_dir="~/thisIsNotARealDir/",
            )

        # bad strategies
        with pytest.raises(ValueError):
            LoopConfiguration(
                nloops=1,
                strategy="idk whatever",
                path_to_features=real_file_path,
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
            )
        LoopConfiguration(
                nloops=1,
                strategy="QBDMI",
                path_to_features=real_file_path,
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
                classifier_bootstrap=True
            )
        with pytest.raises(ValueError):
            LoopConfiguration(
                nloops=1,
                strategy="QBDMI",
                path_to_features=real_file_path,
                output_metrics_file=os.path.join(dir_name, "blah.txt"),
                output_queried_file=os.path.join(dir_name, "blah2.txt"),
            )

def test_to_and_from_dict():
    """dump `LoopConfiguration` to a dict and recreate."""
    with tempfile.TemporaryDirectory() as dir_name:
        real_file_path = os.path.join(dir_name, "test.txt")
        f = open(real_file_path, "w")
        f.write("out of all the files that exist, this is one of them.")
        f.close()

        lc1 = LoopConfiguration(
            nloops=1,
            strategy="RandomSampling",
            path_to_features=real_file_path,
            output_metrics_file=os.path.join(dir_name, "blah.txt"),
            output_queried_file=os.path.join(dir_name, "blah2.txt"),
        )
        d = lc1.to_dict()

        lc2 = LoopConfiguration.from_dict(d)

        assert lc1 == lc2

def test_to_and_from_json():
    """dump `LoopConfiguration` to a json file and recreate."""
    with tempfile.TemporaryDirectory() as dir_name:
        real_file_path = os.path.join(dir_name, "test.txt")
        f = open(real_file_path, "w")
        f.write("out of all the files that exist, this is one of them.")
        f.close()

        lc1 = LoopConfiguration(
            nloops=1,
            strategy="RandomSampling",
            path_to_features=real_file_path,
            output_metrics_file=os.path.join(dir_name, "blah.txt"),
            output_queried_file=os.path.join(dir_name, "blah2.txt"),
        )

        json_path = os.path.join(dir_name, "test.json")
        lc1.to_json(json_path)

        lc2 = LoopConfiguration.from_json(json_path)

        assert lc1 == lc2