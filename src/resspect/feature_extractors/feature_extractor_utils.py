import itertools
from typing import List

def make_features_header(
        filters: List[str],
        features: List[str],
        **kwargs
    ) -> list:
    """
    This function returns header list for given filters and features. The default
    header names are: ['id', 'redshift', 'type', 'code', 'orig_sample'].

    Parameters
    ----------
    filters : list
        Filter values. e.g. ['g', 'r', 'i', 'z']
    features : list
        Feature values. e.g. ['A', 'B']
    with_cost : bool
        Flag for adding cost values. Default is False
    kwargs
        Can include the following flags:
        - override_primary_columns: List[str] of primary columns to override the default ones
        - with_queryable: flag for adding "queryable" column
        - with_last_rmag: flag for adding "last_rmag" column
        - with_cost: flag for adding "cost_4m" and "cost_8m" columns

    Returns
    -------
    header
        header list
    """

    header = []
    header.extend(['id', 'redshift', 'type', 'code', 'orig_sample'])

    # There are rare instances where we need to override the primary columns
    if kwargs.get('override_primary_columns', False):
        header = kwargs.get('override_primary_columns')

    if kwargs.get('with_queryable', False):
        header.append('queryable')
    if kwargs.get('with_last_rmag', False):
        header.append('last_rmag')

    #TODO: find where the 'with_cost' flag is used to make sure we apply there
    if kwargs.get('with_cost', False):
        header.extend(['cost_4m', 'cost_8m'])

    # Create all pairs of filter + feature strings and append to the header
    filter_features = create_filter_feature_names(filters, features)
    header += filter_features

    return header

def create_filter_feature_names(filters: List[str], features: List[str]) -> List[str]:
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
