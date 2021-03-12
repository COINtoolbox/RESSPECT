
import pytest


def test_can_run_learn_loop(extract_feature):
    """Just a sanity test"""

    from resspect.learn_loop import learn_loop

    learn_loop(nloops=1,
               features_method="Bazin",
               strategy="RandomSampling",
               path_to_features=extract_feature,
               output_metrics_file="just_a_name.csv",
               output_queried_file="just_other_name.csv")


@pytest.fixture(scope="function")
def extract_feature(setup_test):
    from resspect import fit_snpcc_bazin

    path_to_data_dir = setup_test
    output_file = 'output_file.dat'

    fit_snpcc_bazin(path_to_data_dir=path_to_data_dir, features_file=output_file)

    return output_file


if __name__ == '__main__':
    pytest.main()
