from dataclasses import dataclass
from os import path

from resspect import VALID_STRATEGIES, BaseConfiguration

@dataclass
class LoopConfiguration(BaseConfiguration):
    """Configuration for the `resspect.learn_loop` function.

    Attributes
    ----------
    nloops: int
        Number of active learning loops to run.
    strategy: str
        Query strategy. Options are 'UncSampling', 'RandomSampling',
        'UncSamplingEntropy', 'UncSamplingLeastConfident', 'UncSamplingMargin',
        'QBDMI' and 'QBDEntropy'.
    path_to_features: str or dict
        Complete path to input features file.
        if dict, keywords should be 'train' and 'test',
        and values must contain the path for separate train
        and test sample files.
    output_metrics_file: str
        Full path to output file to store metric values of each loop.
    output_queried_file: str
        Full path to output file to store the queried sample.
    features_method: str (optional)
        Feature extraction method. Currently only 'Bazin', 'Bump', and 'Malanchev' are implemented.
    classifier: str (optional)
        Machine Learning algorithm.
        Currently implemented options are 'RandomForest', 'GradientBoostedTrees',
        'K-NNclassifier','MLPclassifier','SVMclassifier' and 'NBclassifier'.
        Default is 'RandomForest'.
    training: str or int (optional)
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia
        Default is 'original'.
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    survey: str (optional)
        'DES' or 'LSST'. Default is 'DES'.
        Name of the survey which characterizes filter set.
    nclass: int (optional)
        Number of classes to consider in the classification
        Currently only nclass == 2 is implemented.
    photo_class_thr: float (optional)
        Threshold for photometric classification. Default is 0.5.
        Only used if photo_ids is True.
    photo_ids_to_file: bool (optional)
        If True, save photometric ids to file. Default is False.
    photo_ids_froot: str (optional)
        Output root of file name to store photo ids.
        Only used if photo_ids is True.
    classifier_bootstrap: bool (optional)
        Flag for bootstrapping on the classifier
        Must be true if using disagreement based strategy.
    save_predictions: bool (optional)
        If True, save classification predictions to file in each loop.
        Default is False.
    sep_files: bool (optional)
        If True, consider train and test samples separately read
        from independent files. Default is False.
    pred_dir: str (optional)
        Output diretory to store prediction file for each loop.
        Only used if `save_predictions==True`.
    queryable: bool (optional)
        If True, check if randomly chosen object is queryable.
        Default is False.
    metric_label: str (optional)
        Choice of metric.
        Currently only "snpcc", "cosmo" or "snpcc_cosmo" are accepted.
        Default is "snpcc".
    save_alt_class: bool (optional)
        If True, train the model and save classifications for alternative
        query label (this is necessary to calculate impact on cosmology).
        Default is False.
        If True, translate zenodo types to SNANA codes.
    SNANA_types: bool (optional)
        Default is False.
    metadata_fname: str (optional)
        Complete path to PLAsTiCC zenodo test metadata. Only used it
        SNANA_types == True. Default is None.
        Default is False.
    bar: bool (optional)
        If True, display progress bar.
    initial_training_samples_file
        File name to save initial training samples.
        File will be saved if "training"!="original".
    pretrained_model_path: str (optional)
        Filepath to a pretrained model. If provided, the model will be loaded
        and used to predict the queried samples.
    """
    nloops: int
    strategy: str
    path_to_features: str
    output_metrics_file: str
    output_queried_file: str
    features_method: str = 'Bazin'
    classifier: str = 'RandomForest'
    training: str = 'original'
    batch: int =1
    survey: str = 'DES'
    nclass: int = 2
    photo_class_thr: float = 0.5
    photo_ids_to_file: bool = False
    photo_ids_froot: str =' '
    classifier_bootstrap: bool = False
    save_predictions: bool = False
    sep_files: bool = False
    pred_dir: str = None
    queryable: bool = False
    metric_label: str = 'snpcc'
    save_alt_class: bool = False
    SNANA_types: bool = False
    metadata_fname: str = None
    bar: bool = True
    initial_training_samples_file: str = None
    pretrained_model_path: str = None

    def __post_init__(self):
        # file checking
        if isinstance(self.path_to_features, str):
            if not path.isfile(self.path_to_features):
                raise ValueError("`path_to_features` must be an existing file.")
        elif isinstance(self.path_to_features, dict):
            for key in self.path_to_features.keys():
                if not path.isfile(self.path_to_features[key]):
                    raise ValueError(f"path for '{key}' does not exist.")
        else:
            raise ValueError("`path_to_features` must be a str or dict.")

        if isinstance(self.pretrained_model_path, str):
            if not path.isfile(self.pretrained_model_path):
                raise ValueError("`pretrained_model_path` must be an existing file.")

        if self.SNANA_types:
            if self.metadata_fname is None:
                raise ValueError("`SNANA_types` is enabled and metadata file was not provided.")
            if not path.isfile(self.metadata_fname):
                raise ValueError("provided `metadata_fname` does not exist.")
            
        if self.save_predictions:
            if self.pred_dir is None:
                raise ValueError("cannot save predictions, no `pred_dir` was provided.")
            if not path.isdir(self.pred_dir):
                raise ValueError("provided `pred_dir` does not exist/is not a directory.")

        # check strategy
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(f"{self.strategy} is not a valid strategy.")
        if "QBD" in self.strategy and not self.classifier_bootstrap:
            raise ValueError("Bootstrap must be true when using disagreement strategy")