"""Tests for sample_utils.py."""

import numpy as np
import pytest

from resspect.samples_utils import sep_samples


def test_sep_samples():
    """Test that we can generate separate samples."""
    all_ids = np.arange(0, 100)
    samples = sep_samples(all_ids, n_test_val=10, n_train=50)
    assert len(samples) == 4

    # Check that each partition is the correct size and disjoint.
    assert len(samples["train"]) == 50
    assert len(np.unique(samples["train"])) == 50
    all_seen = np.copy(samples["train"])

    assert len(samples["val"]) == 10
    assert len(np.unique(samples["val"])) == 10
    all_seen = np.union1d(all_seen, samples["val"])
    assert len(all_seen) == 60

    assert len(samples["test"]) == 10
    assert len(np.unique(samples["test"])) == 10
    all_seen = np.union1d(all_seen, samples["test"])
    assert len(all_seen) == 70

    assert len(samples["query"]) == 30
    assert len(np.unique(samples["query"])) == 30
    all_seen = np.union1d(all_seen, samples["query"])
    assert len(all_seen) == 100


def test_sep_samples_too_many():
    """Test that we fail if we try to generate more samples than IDs."""
    all_ids = np.arange(0, 100)
    with pytest.raises(ValueError):
        samples = sep_samples(all_ids, n_test_val=50, n_train=80)
    with pytest.raises(ValueError):
        samples = sep_samples(all_ids, n_test_val=15, n_train=80)

    # We are okay with exactly the same number of samples and IDs.
    # But the 'query' bucket is empty
    samples = sep_samples(all_ids, n_test_val=10, n_train=80)
    assert len(samples) == 4
    assert len(samples["query"]) == 0


if __name__ == '__main__':
    pytest.main()
