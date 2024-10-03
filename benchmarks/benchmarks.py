"""Benchmarks to compute runtime and memory usage of core functions.

To manually run the benchmarks use: asv run

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

from pathlib import Path
import tempfile

from resspect import fit_snpcc
from resspect.learn_loop import  learn_loop


_TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "tests"


def time_feature_creation():
    """Benchmark how long it takes to read the example files and generate features."""
    input_file_path = _TEST_DATA_DIR / "DES_data"
    with tempfile.TemporaryDirectory() as dir_name:
        features_file = Path(dir_name) / "Malanchev.csv"
        fit_snpcc(
            path_to_data_dir=input_file_path,
            features_file=str(features_file),
            feature_extractor="malanchev"
        )

def time_learn_loop(ml_model, strategy):
    """Benchmark how long it takes for a run of the learning loop."""
    # Use the precomputed features so we don't time their creation.
    features_file = str(_TEST_DATA_DIR / "test_features.csv")
    with tempfile.TemporaryDirectory() as dir_name:
        metrics_file = str(Path(dir_name) / "metrics.csv")
        output_queried_file = str(Path(dir_name) / "queried.csv")
        learn_loop(
            nloops=25,
            features_method="malanchev",
            classifier=ml_model,
            strategy=strategy,
            path_to_features=features_file,
            output_metrics_file=metrics_file,
            output_queried_file=output_queried_file,
            training="original",
            batch=1,
        )  


# Parameterize the ML models and strategies we benchmark.
time_learn_loop.params = [
    ["RandomForest", "KNN"],            # The different ML methods
    ["RandomSampling", "UncSampling"],  # The different strategies.
]
