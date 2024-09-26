#!/usr/bin/env python
"""
Test Example

Contains some PyTest examples that can be applied to RESSPECT.

"""

import pytest
import numpy as np


def test_one_is_an_integer():
    """
    Checks that '1' is an integer.
    """
    assert isinstance(1, int)


def test_random_smaller_than_one():
    """
    Check that np.random.rand returns a value smaller than 1.
    """
    my_array = np.random.rand(10)

    assert len(my_array) == 10
    np.testing.assert_array_less(my_array, 1)


if __name__ == '__main__':
    pytest.main()
