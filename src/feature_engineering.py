"""
feature_engineering.py — derived variables and column elimination decisions
made after the correlation and mutual information analyses.

All transformations here are deterministic (no fitted state), so the same
function applies identically to train, validation, and test data.
"""

COLUMNS_TO_DROP = ['census_msa', 'hhs_geo_region', 'opinion_sick_from_vacc_combined', 'stratify_col']


def add_household_features(df):
    """Adds household_children_binary (0/1) and household_size
    (household_adults + household_children). Assumes both source columns
    are already free of missing values (post Strategy B Group 1)."""
    df = df.copy()
    df['household_children_binary'] = df['household_children'].apply(lambda x: 1 if x > 0 else 0)
    df['household_size'] = df['household_children'] + df['household_adults']
    return df


def drop_eliminated_columns(df, columns_to_drop=COLUMNS_TO_DROP):
    """Drops the columns removed during feature engineering. Uses
    errors='ignore' since the test set never had stratify_col, and may
    already lack other columns depending on the calling context."""
    return df.drop(columns=columns_to_drop, errors='ignore')


def apply_feature_engineering(df):
    """Applies the full feature engineering step: adds derived household
    variables, then drops the columns eliminated after the correlation /
    mutual information analyses (census_msa, hhs_geo_region,
    opinion_sick_from_vacc_combined, stratify_col).

    Note: opinion_h1n1_sick_from_vacc / opinion_seas_sick_from_vacc are
    NOT merged here — that merge was reverted after the mutual information
    analysis showed both originals carry more signal than the combined
    variable. They are kept as-is, already present in the input dataframe.
    """
    df = add_household_features(df)
    df = drop_eliminated_columns(df)
    return df
