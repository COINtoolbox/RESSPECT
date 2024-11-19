from resspect.feature_extractors.feature_extractor_utils import make_filter_feature_names, make_column_headers

def test_create_filter_feature_names():
    filters = ['g', 'r']
    features = ['A', 'B']
    assert make_filter_feature_names(filters, features) == ['gA', 'gB', 'rA', 'rB']

def test_create_filter_feature_names_empty():
    filters = []
    features = ['A', 'B']
    assert make_filter_feature_names(filters, features) == []

def test_make_features_header():
    filters = ['g', 'r', 'i', 'z']
    features = ['A', 'B']
    assert make_column_headers(filters, features) == ['id', 'redshift', 'type', 'code', 'orig_sample', 'gA', 'gB', 'rA', 'rB', 'iA', 'iB', 'zA', 'zB']

def test_make_features_header_with_cost():
    filters = ['g', 'r', 'i', 'z']
    features = ['A', 'B']
    assert make_column_headers(filters, features, with_cost=['cost_4m', 'cost_8m']) == ['id', 'redshift', 'type', 'code', 'orig_sample', 'cost_4m', 'cost_8m', 'gA', 'gB', 'rA', 'rB', 'iA', 'iB', 'zA', 'zB']

def test_make_features_header_with_queryable():
    filters = ['g', 'r', 'i', 'z']
    features = ['A', 'B']
    assert make_column_headers(filters, features, with_queryable=True) == ['id', 'redshift', 'type', 'code', 'orig_sample', 'queryable', 'gA', 'gB', 'rA', 'rB', 'iA', 'iB', 'zA', 'zB']

def test_make_features_header_with_last_rmag():
    filters = ['g', 'r', 'i', 'z']
    features = ['A', 'B']
    assert make_column_headers(filters, features, with_last_rmag=True) == ['id', 'redshift', 'type', 'code', 'orig_sample', 'last_rmag', 'gA', 'gB', 'rA', 'rB', 'iA', 'iB', 'zA', 'zB']

def test_make_features_header_with_override_primary_columns():
    filters = ['g', 'r', 'i', 'z']
    features = ['A', 'B']
    assert make_column_headers(filters, features, override_primary_columns=['new_id', 'new_redshift', 'new_type', 'new_code', 'new_orig_sample']) == ['new_id', 'new_redshift', 'new_type', 'new_code', 'new_orig_sample', 'gA', 'gB', 'rA', 'rB', 'iA', 'iB', 'zA', 'zB']

def test_make_features_header_with_all_flags():
    filters = ['g', 'r', 'i', 'z']
    features = ['A', 'B']

    expected = [
        'new_id', 'new_redshift', 'new_type', 'new_code', 'new_orig_sample',
        'queryable', 'last_rmag',
        'cost_4m', 'cost_8m',
        'gA', 'gB', 'rA', 'rB', 'iA', 'iB', 'zA', 'zB'
    ]

    result = make_column_headers(
        filters,
        features,
        with_cost=['cost_4m', 'cost_8m'],
        with_queryable=True,
        with_last_rmag=True,
        override_primary_columns=['new_id', 'new_redshift', 'new_type', 'new_code', 'new_orig_sample']
    )

    assert result == expected
