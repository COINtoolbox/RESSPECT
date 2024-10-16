"""Tests for query_strategies.py."""

import itertools
import numpy as np
import pytest

from resspect.query_strategies import (
    qbd_entropy,
    qbd_mi,
    random_sampling,
    uncertainty_sampling,
    uncertainty_sampling_entropy,
    uncertainty_sampling_least_confident,
    uncertainty_sampling_margin,
)


@pytest.mark.parametrize("batch_size,queryable",list(itertools.product([1, 5, 10], [True, False])))
def test_random_sampling(batch_size, queryable):
    """Test the random sampling functionality."""
    test_ids = np.arange(0, 100)
    queryable_ids = test_ids[test_ids % 3 == 0]

    # Test that we generate the correct number of samples.
    sample = random_sampling(test_ids, queryable_ids, batch=batch_size, queryable=queryable)
    assert len(sample) == batch_size
    assert len(np.unique(sample)) == batch_size

    if queryable:
        assert np.all(np.array(sample) % 3 == 0)


def test_uncertainty_sampling():
    """Test the uncertainity sampling functionality."""
    test_ids = np.arange(0, 10)
    queryable_ids = np.array([0, 1, 2, 3, 4, 7, 8, 9])  # No 5 or 6
    class1_probs = np.array([
        0.01,  # 0 - very low
        0.50,  # 1 - very high
        0.10,  # 2 - low
        0.20,  # 3 - low
        0.65,  # 4 - medium high
        0.45,  # 5 - very high
        0.25,  # 6 - medium
        0.80,  # 7 - low
        0.40,  # 8 - high
        0.02,  # 9 - very low
    ])
    class_probs = np.array([class1_probs, 1.0 - class1_probs]).T

    # Test that we generate the correct number of samples.
    sample = uncertainty_sampling(class_probs, test_ids, queryable_ids, batch=3)
    assert len(sample) == 3
    assert np.array_equal(sample, [1, 8, 4])


@pytest.mark.parametrize("batch_size",[1, 5, 10, 20])
def test_uncertainty_sampling_entropy_random(batch_size):
    """Test the entropy-based uncertainty sampling functionality with random data."""
    num_samples = 100
    num_classes = 5
    test_ids = np.arange(0, 100)
    queryable_ids = test_ids[test_ids % 3 == 0]

    # Generate class probabilities.
    np.random.seed(100)
    class_prob = np.random.random((num_samples, num_classes))
    normalized_probs =  class_prob / np.tile(np.sum(class_prob, axis=1), (num_classes, 1)).T

    # Test that we generate the correct number of samples.
    sample = uncertainty_sampling_entropy(
        normalized_probs,
        test_ids,
        queryable_ids,
        batch=batch_size
    )
    assert len(sample) == batch_size
    assert len(np.unique(sample)) == batch_size
    assert np.all(np.array(sample) % 3 == 0)


def test_uncertainty_sampling_entropy_known():
    """Test the entropy-based uncertainty sampling functionality with known entropies."""
    test_ids = np.arange(0, 8)
    queryable_ids = np.arange(0, 8)
    class_prob = np.array(
        [
            [1.0, 0.0, 0.0],              # 0.0
            [0.5, 0.5, 0.0],              # 0.693
            [1.0/3.0, 1.0/3.0, 1.0/3.0],  # 1.098
            [0.5, 0.0, 0.5],              # 0.693
            [0.05, 0.9, 0.05],            # 0.394
            [0.2, 0.4, 0.4],              # 1.055
            [0.1, 0.5, 0.4],              # 0.943
            [0.1, 0.7, 0.2],              # 0.802
        ]
    )

    sample = uncertainty_sampling_entropy(class_prob, test_ids, queryable_ids, batch=3)
    assert np.array_equal(sample, [2, 5, 6])


def test_uncertainty_sampling_least_confident():
    """Test the least confident based uncertainty sampling."""
    test_ids = np.arange(0, 8)
    queryable_ids = np.arange(0, 8)
    class_prob = np.array(
        [
            [1.0, 0.0, 0.0],              # most confident (1.0)
            [0.45, 0.49, 0.06],           # middle (0.49)
            [1.0/3.0, 1.0/3.0, 1.0/3.0],  # very low (1/3)
            [0.5, 0.0, 0.5],              # middle (0.5)
            [0.05, 0.9, 0.05],            # high (0.9)
            [0.2, 0.4, 0.4],              # low (0.4)
            [0.1, 0.55, 0.35],            # middle (0.55)
            [0.1, 0.7, 0.2],              # high (0.7)
        ]
    )
    sample = uncertainty_sampling_least_confident(class_prob, test_ids, queryable_ids, batch=3)
    assert np.array_equal(sample, [2, 5, 1])

    # If we don't allow 5, we get 3 instead.
    queryable_ids = np.array([0, 1, 2, 3, 4, 6, 7])
    sample = uncertainty_sampling_least_confident(class_prob, test_ids, queryable_ids, batch=3)
    assert np.array_equal(sample, [2, 1, 3])


def test_uncertainty_sampling_margin():
    """Test the margin-based uncertainty sampling."""
    test_ids = np.arange(0, 8)
    queryable_ids = np.arange(0, 8)
    class_prob = np.array(
        [
            [1.0, 0.0, 0.0],              # margin = 1.0
            [0.45, 0.49, 0.06],           # margin = 0.04
            [0.3, 0.3, 0.4],              # margin = 0.1
            [0.5, 0.0, 0.5],              # margin = 0.0
            [0.05, 0.9, 0.05],            # margin = 0.85
            [0.2, 0.39, 0.41],            # margin = 0.02
            [0.1, 0.55, 0.35],            # margin = 0.2
            [0.1, 0.7, 0.2],              # margin = 0.5
        ]
    )
    sample = uncertainty_sampling_margin(class_prob, test_ids, queryable_ids, batch=3)
    assert np.array_equal(sample, [3, 5, 1])

    # If we don't allow 5, we get 3 instead.
    queryable_ids = np.array([0, 1, 2, 3, 4, 6, 7])
    sample = uncertainty_sampling_margin(class_prob, test_ids, queryable_ids, batch=3)
    assert np.array_equal(sample, [3, 1, 2])


def test_qbd_entropy():
    """Test the ensemble average prediction entropy sampling."""
    test_ids = np.arange(0, 5)
    queryable_ids = np.arange(0, 5)

    # Probabilities coming out of the ensembles are 3-d matrices with dimensions:
    # number of points (5), the number of ensemble members (3), and the number of events (2).
    ensemble_probs = np.array(
        [
            [[1.00, 0.00], [0.95, 0.05], [0.99, 0.01]],  # very low entropy (high agreement)
            [[0.80, 0.20], [0.60, 0.40], [0.20, 0.80]],  # high entropy (low agreement)
            [[0.10, 0.90], [0.10, 0.90], [0.05, 0.95]],  # low entropy (high agreement)
            [[0.75, 0.25], [0.80, 0.20], [0.78, 0.22]],  # medium entropy (high agreement)
            [[0.50, 0.50], [0.50, 0.50], [0.49, 0.51]],  # high entropy (high agreement)
        ]
    )
    sample = qbd_entropy(ensemble_probs, test_ids, queryable_ids, batch=5)
    assert np.array_equal(sample, [4, 1, 3, 2, 0])


def test_qbd_mi():
    """Test the ensemble qbd_mi sampling."""
    test_ids = np.arange(0, 5)
    queryable_ids = np.arange(0, 5)

    # Probabilities coming out of the ensembles are 3-d matrices with dimensions:
    # number of points (5), the number of ensemble members (3), and the number of events (2).
    ensemble_probs = np.array(
        [
            [[1.00, 0.00], [0.95, 0.05], [0.80, 0.20]],
            [[0.80, 0.20], [0.60, 0.40], [0.20, 0.80]],
            [[0.10, 0.90], [0.10, 0.90], [0.05, 0.95]],
            [[0.75, 0.25], [0.80, 0.20], [0.78, 0.22]],
            [[0.50, 0.50], [0.50, 0.50], [0.49, 0.51]],
        ]
    )
    sample = qbd_mi(ensemble_probs, test_ids, queryable_ids, batch=5)
    assert np.array_equal(sample, [1, 0, 2, 3, 4])


if __name__ == '__main__':
    pytest.main()
