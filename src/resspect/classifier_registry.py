from resspect.plugin_utils import get_or_load_class
from resspect.classifiers import ResspectClassifer

__all__ = ["CLASSIFIER_REGISTRY", "register_builtin_classifiers", "fetch_classifier_class"]

CLASSIFIER_REGISTRY = {}

def register_builtin_classifiers():
    """Add all built-in classifiers to the registry."""
    subclasses = ResspectClassifer.__subclasses__()
    for subclass in subclasses:
        CLASSIFIER_REGISTRY[subclass.__name__] = subclass


def fetch_classifier_class(classifier_name: str) -> type:
    """Fetch the classifier class from the registry.

    Parameters
    ----------
    classifier_name : str
        The name of the classifier class to retrieve. This should either be the
    name of the class or the import specification for the class.

    Returns
    -------
    type
        The classifier class.

    Raises
    ------
    ValueError
        If a built-in classifier was requested, but not found in the registry.
    ValueError
        If no classifier was specified in the runtime configuration.
    """

    clf_class = None

    try:
        clf_class = get_or_load_class(classifier_name, CLASSIFIER_REGISTRY)
    except ValueError as exc:
        raise ValueError(f"Error fetching class: {classifier_name}") from exc

    return clf_class
