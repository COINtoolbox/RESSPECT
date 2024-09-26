import pytest


def test_run_loop_arg_check():
    """Test that the logic that parses CLI inputs works as expected. Specifically
    that the training sample is correctly parsed as either an integer or the string 'original'.
    """

    from resspect.scripts.run_loop import _parse_training

    assert _parse_training('original') == 'original'
    assert _parse_training('OrigINAL') == 'original'
    assert _parse_training('10') == 10
    with pytest.raises(ValueError):
        _parse_training('not_a_number')
    with pytest.raises(ValueError):
        _parse_training('1.0')
    with pytest.raises(ValueError):
        _parse_training('')
    assert _parse_training('010') == 10
