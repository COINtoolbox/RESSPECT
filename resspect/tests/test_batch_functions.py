
import pytest
import numpy as np


@pytest.mark.skip(reason='Test failing with numpy.AxisError - Check inputs')
def test_entropy_from_probs_b_M_C():

    from resspect import batch_functions

    np.random.seed(42)
    x = np.random.rand(100)

    foo = batch_functions.entropy_from_probs_b_M_C(x)


if __name__ == '__main__':
    pytest.main()
