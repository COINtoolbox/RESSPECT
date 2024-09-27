
import pytest
import numpy as np


def test_entropy_from_probs_b_M_C():

    from resspect import batch_functions

    np.random.seed(42)
    num_data_points = 3
    committee_size = 10
    num_classes = 5
    x = np.random.rand(num_data_points, committee_size, num_classes)

    foo = batch_functions.entropy_from_probs_b_M_C(x)


if __name__ == '__main__':
    pytest.main()
