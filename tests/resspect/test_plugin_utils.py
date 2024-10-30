import pytest
from resspect.plugin_utils import import_module_from_string, fetch_classifier_class, fetch_query_strategy_class


def test_import_module_from_string():
    """Test the import_module_from_string function."""
    module_path = "builtins.BaseException"

    returned_cls = import_module_from_string(module_path)

    assert returned_cls.__name__ == "BaseException"


def test_import_module_from_string_no_base_module():
    """Test that the import_module_from_string function raises an error when
    the base module is not found."""

    module_path = "nonexistent.BaseException"

    with pytest.raises(ModuleNotFoundError) as excinfo:
        import_module_from_string(module_path)

    assert "Module nonexistent not found" in str(excinfo.value)


def test_import_module_from_string_no_submodule():
    """Test that the import_module_from_string function raises an error when
    a submodule is not found."""

    module_path = "builtins.nonexistent.BaseException"

    with pytest.raises(ModuleNotFoundError) as excinfo:
        import_module_from_string(module_path)

    assert "Module builtins.nonexistent not found" in str(excinfo.value)


def test_import_module_from_string_no_class():
    """Test that the import_module_from_string function raises an error when
    a class is not found."""

    module_path = "builtins.Nonexistent"

    with pytest.raises(AttributeError) as excinfo:
        import_module_from_string(module_path)

    assert "The class Nonexistent was not found" in str(excinfo.value)


def test_fetch_classifier_class():
    """Test the fetch_classifier_class function."""
    requested_class = "builtins.BaseException"

    returned_cls = fetch_classifier_class(requested_class)

    assert returned_cls.__name__ == "BaseException"


def test_fetch_classifier_class_not_in_registry():
    """Test that an exception is raised when a model is requested that is not in the registry."""

    requested_class = "Nonexistent"

    with pytest.raises(ValueError) as excinfo:
        fetch_classifier_class(requested_class)

    assert "Error fetching class: Nonexistent" in str(excinfo.value)


def test_fetch_query_strategy_class():
    """Test the fetch_query_strategy_class function."""
    requested_class = "builtins.BaseException"

    returned_cls = fetch_query_strategy_class(requested_class)

    assert returned_cls.__name__ == "BaseException"


def test_fetch_query_strategy_class_not_in_registry():
    """Test that an exception is raised when a model is requested that is not in the registry."""

    requested_class = "Nonexistent"

    with pytest.raises(ValueError) as excinfo:
        fetch_query_strategy_class(requested_class)

    assert "Error fetching class: Nonexistent" in str(excinfo.value)
