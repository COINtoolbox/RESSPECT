import os
import pytest
import tempfile

from resspect.time_domain_loop import time_domain_loop
from resspect.time_domain_configuration import TimeDomainConfiguration

def test_can_run_time_domain_loop(test_data_path):
    """Test that learn_loop can load data and run."""
    with tempfile.TemporaryDirectory() as dir_name:
        ini_file = {
            "train": os.path.join(test_data_path, "general_hot_features_day_2.csv")
        }

        td_config = TimeDomainConfiguration(
            days = [2],
            strategy="RandomSampling",
            path_to_features_dir=test_data_path,
            output_metrics_file=os.path.join(dir_name,"just_a_name.csv"),
            output_queried_file=os.path.join(dir_name,"just_other_name.csv"),
            fname_pattern=["general_hot_features_day_", ".csv"],
            path_to_ini_files=ini_file,
            sep_files=False,
            feature_extraction_method="Malanchev"
        )

        time_domain_loop(td_config)

if __name__ == '__main__':
    pytest.main()