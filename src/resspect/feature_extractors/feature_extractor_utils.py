import itertools
from typing import List

def make_column_headers(
        filters: List[str],
        features: List[str],
        **kwargs
    ) -> List[str]:
    """
    This function returns the full header list for given filters and features.

    Parameters
    ----------
    filters : list
        Filter values. e.g. ['g', 'r', 'i', 'z']
    features : list
        Feature values. e.g. ['A', 'B']
    kwargs
        Can include the following flags:
        - override_primary_columns: List[str] of primary columns to override the default ones
        - with_queryable: flag for adding "queryable" column
        - with_last_rmag: flag for adding "last_rmag" column
        - with_cost: flag for adding "cost_4m" and "cost_8m" columns

    Returns
    -------
    List[str]
        The complete metadata and feature header list
    """

    header = []

    # Create metadata columns
    metadata_columns = make_metadata_column_names(**kwargs)
    header += metadata_columns

    # Create all pairs of filter + feature strings and append to the header
    filter_features = make_filter_feature_names(filters, features)
    header += filter_features

    return header

def make_metadata_column_names(**kwargs) -> List[str]:
    """The default header names are: ['id', 'redshift', 'type', 'code', 'orig_sample'].
    Using the keys in kwargs, we can add additional columns to the header.

    Parameters
    ----------
    kwargs
        Can include the following flags:
        - override_primary_columns: List[str] of primary columns to override the default ones
        - with_queryable: flag for adding "queryable" column
        - with_last_rmag: flag for adding "last_rmag" column
        - with_cost: flag for adding "cost_4m" and "cost_8m" columns

    Returns
    -------
    List[str]
        metadata header list
    """

    metadata_columns = []
    metadata_columns.extend(['id', 'redshift', 'type', 'code', 'orig_sample'])

    # There are rare instances where we need to override the primary columns
    if kwargs.get('override_primary_columns', False):
        metadata_columns = kwargs.get('override_primary_columns')

    if kwargs.get('with_queryable', False):
        metadata_columns.append('queryable')

    if kwargs.get('with_last_rmag', False):
        metadata_columns.append('last_rmag')

    if len(kwargs.get('with_cost', [])):
        metadata_columns.extend(kwargs.get('with_cost'))

    return metadata_columns

def make_filter_feature_names(filters: List[str], features: List[str]) -> List[str]:
    """This function returns the list of concatenated filters and features. e.g.
    filter = ['g', 'r'], features = ['A', 'B'] => ['gA', 'gB', 'rA', 'rB']

    Parameters
    ----------
    filters : List[str]
        Filter name list
    features : List[str]
        Feature name list

    Returns
    -------
    List[str]
        List of filter-feature pairs.
    """

    return [''.join(pair) for pair in itertools.product(filters, features)]
