from dataclasses import dataclass, asdict
import json
from os import path

from resspect import VALID_STRATEGIES, BaseConfiguration

@dataclass
class TimeDomainConfiguration(BaseConfiguration):
    """
    Atributes
    ----------
    days: list
        List of 2 elements. First and last day of observations since the
        beginning of the survey.
    output_metrics_file: str
        Full path to output file to store metrics for each loop.
    output_queried_file: str
        Full path to output file to store the queried sample.
    path_to_features_dir: str
        Complete path to directory holding features files for all days.
    strategy: str
        Query strategy. Options are (all can be run with budget):
        "UncSampling",
        "UncSamplingEntropy",
        "UncSamplingLeastConfident",
        "UncSamplingMargin",
        "QBDMI",
        "QBDEntropy",
        "RandomSampling",
    fname_pattern: str
        List of strings. Set the pattern for filename, except day of
        survey. If file name is 'day_1_vx.csv' -> ['day_', '_vx.csv'].
    path_to_ini_files: dict (optional)
        Path to initial full light curve files.
        Possible keywords are: "train", "test" and "validation".
        At least "train" is mandatory.
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    canonical: bool (optional)
        If True, restrict the search to the canonical sample.
    classifier: str (optional)
        Machine Learning algorithm.
        Currently 'RandomForest', 'GradientBoostedTrees',
        'KNN', 'MLP', 'SVM' and 'NB' are implemented.
        Default is 'RandomForest'.
    clf_bootstrap: bool (default: False)
        If true build a boostrap ensemble of the classifier.
    budgets: tuple of floats (default: None)
        Budgets for each of the telescopes
    ia_frac: float in [0,1] (optional)
        Fraction of Ia required in initial training sample.
        Default is 0.5.
    nclass
        Number of classes to consider in the classification
        Currently only nclass == 2 is implemented.
    path_to_canonical: str (optional)
        Path to canonical sample features files.
        It is only used if "strategy==canonical".
    queryable: bool (optional)
        If True, allow queries only on objects flagged as queryable.
        Default is True.
    query_thre: float (optional)
        Percentile threshold for query. Default is 1.
    save_samples: bool (optional)
        If True, save training and test samples to file.
        Default is False.
    save_full_query: bool (optional)
        If True, save complete queried sample to file.
        Otherwise, save only first element. Default is False.
    sep_files: bool (optional)
        If True, consider samples separately read
        from independent files. Default is False.
    survey: str (optional)
        Name of survey to be analyzed. Accepts 'DES' or 'LSST'.
        Default is LSST.
    initial_training: str or int (optional)
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        eilf 'previous': read training and queried from previous run.
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia
        Default is 'original'.
    feature_extraction_method: str (optional)
        Feature extraction method. The current implementation only
        accepts method=='Bazin' or 'photometry'.
        Default is 'Bazin'.
    """
    days: list
    output_metrics_file: str
    output_queried_file: str
    path_to_features_dir: str
    strategy: str
    fname_pattern: list
    path_to_ini_files: dict
    batch: int = 1
    canonical: bool = False
    classifier: str = "RandomForest"
    clf_bootstrap: bool = False
    budgets: tuple = None
    nclass: int = 2
    ia_frac: float = 0.5
    path_to_canonical: str = ""
    queryable: bool = True
    query_thre: float = 1.0
    save_samples: bool = False
    sep_files: bool = False
    survey: str = "LSST"
    initial_training: str = "original"
    feature_extraction_method: str = "Bazin"
    save_full_query: bool = False

    def __post_init__(self):
        # file checking
        if not path.isdir(self.path_to_features_dir):
            raise ValueError("`path_to_features` must be an existing directory.")

        # check strategy
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(f"{self.strategy} is not a valid strategy.")
        if "QBD" in self.strategy and not self.clf_bootstrap:
            raise ValueError("Bootstrap must be true when using disagreement strategy")

        for key in self.path_to_ini_files.keys():
            if not path.isfile(self.path_to_ini_files[key]):
                raise ValueError(f"{key} does not point to existing file.")