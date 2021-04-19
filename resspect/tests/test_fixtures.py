"""
Fixtures are custom PyTest functions that help us setting up and/or tearing
down tests. This file tests some of our existing fixtures.
"""

import os
import pytest


def test_change_working_dir(change_working_dir):
    """
    Test the change_working_dir fixture.

    Parameters
    ----------
    change_working_dir : fixture
        Custom DRAGONS fixture.
    """
    assert "resspect/test_fixtures/outputs" not in os.getcwd()

    with change_working_dir():
        assert "resspect/test_fixtures/outputs" in os.getcwd()

    assert "resspect/test_fixtures/outputs" not in os.getcwd()

    with change_working_dir("my_sub_dir"):
        assert "resspect/test_fixtures/outputs/my_sub_dir" in os.getcwd()

    assert "resspect/test_fixtures/outputs" not in os.getcwd()

    dragons_basetemp = os.getenv("$RESSPECT_TEST_OUT")
    if dragons_basetemp:
        assert dragons_basetemp in os.getcwd()


def test_path_to_inputs(path_to_input_data):
    assert isinstance(path_to_input_data, str)
    assert "test_fixtures/inputs" in path_to_input_data


if __name__ == '__main__':
    pytest.main()
