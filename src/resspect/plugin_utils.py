import importlib
from resspect.classifiers import CLASSIFIER_REGISTRY, ResspectClassifier
from resspect.query_strategies import QUERY_STRATEGY_REGISTRY, QueryStrategy

def get_or_load_class(class_name: str, registry: dict) -> type:
    """Given the name of a class and a registry dictionary, attempt to return
    the requested class either from the registry or by dynamically importing it.

    Parameters
    ----------
    class_name : str
        The name of the class to be returned.
    registry : dict
        The registry dictionary of <class name> : <class type> pairs.

    Returns
    -------
    type
        The returned class to be instantiated

    Raises
    ------
    ValueError
        User failed to specify a class to load in the runtime configuration. No
        `name` key was found in the config.
    """

    returned_class = None

    try:
        if class_name in registry:
            returned_class = registry[class_name]
        else:
            returned_class = import_module_from_string(class_name)
    except ValueError as exc:
        raise ValueError(f"Error fetching class: {class_name}") from exc

    return returned_class


def import_module_from_string(module_path: str) -> type:
    """Dynamically import a module from a string.

    Parameters
    ----------
    module_path : str
        The import specification for the class. Should be of the form:
        "module.submodule.class_name"

    Returns
    -------
    returned_cls : type
        The class to be instantiated.

    Raises
    ------
    AttributeError
        If the class is not found in the module that is loaded.
    ModuleNotFoundError
        If the module is not found using the provided import specification.
    """

    module_name, class_name = module_path.rsplit(".", 1)
    returned_cls = None

    try:
        # Attempt to find the module spec, i.e. `module.submodule.`.
        # Will raise exception if `submodule`, 'subsubmodule', etc. is not found.
        importlib.util.find_spec(module_name)

        # `importlib.util.find_spec()` will return None if `module_name` is not found.
        if (importlib.util.find_spec(module_name)) is not None:
            # Import the requested module
            imported_module = importlib.import_module(module_name)

            # Check if the requested class is in the imported module
            if hasattr(imported_module, class_name):
                returned_cls = getattr(imported_module, class_name)
            else:
                raise AttributeError(f"The class {class_name} was not found in module {module_name}")

        # Raise an exception if the base module of the spec is not found
        else:
            raise ModuleNotFoundError(f"Module {module_name} not found")

    # Exception raised when a submodule of the spec is not found
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"Module {module_name} not found") from exc

    return returned_cls


def fetch_classifier_class(classifier_name: str) -> ResspectClassifier:
    """Fetch the classifier class from the registry.

    Parameters
    ----------
    classifier_name : str
        The name of the classifier class to retrieve. This should either be the
    name of the class or the import specification for the class.

    Returns
    -------
    ResspectClassifier
        The subclass of ResspectClassifier.

    Raises
    ------
    ValueError
        If a built-in classifier was requested, but not found in the registry.
    ValueError
        If no classifier was specified in the runtime configuration.
    """

    return get_or_load_class(classifier_name, CLASSIFIER_REGISTRY)


def fetch_query_strategy_class(query_strategy_name: str) -> QueryStrategy:
    """Fetch the query strategy class from the registry.

    Parameters
    ----------
    query_strategy_name : str
        The name of the query strategy class to retrieve. This should either be the
    name of the class or the import specification for the class.

    Returns
    -------
    QueryStrategy
        The subclass of QueryStrategy.

    Raises
    ------
    ValueError
        If a built-in query strategy was requested, but not found in the registry.
    ValueError
        If no query strategy was specified in the runtime configuration.
    """

    return get_or_load_class(query_strategy_name, QUERY_STRATEGY_REGISTRY)
