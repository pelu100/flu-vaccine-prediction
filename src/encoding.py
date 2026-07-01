"""
encoding.py — variable type classification and model-family-specific
feature matrix builders.

These functions are pure (no fitted state): they only inspect dtypes and
reshape data, so the same functions apply identically to train,
validation, and test data.
"""

import pandas as pd

DEFAULT_EXCLUDE_COLS = ['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine', 'stratify_col']


def classify_features(df, feature_list):
    """Splits an explicit feature list into ordinal / nominal / numeric,
    based on dtype."""
    ordinal_cols, nominal_cols, numeric_cols = [], [], []
    for col in feature_list:
        dtype = df[col].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            (ordinal_cols if dtype.ordered else nominal_cols).append(col)
        else:
            numeric_cols.append(col)
    return ordinal_cols, nominal_cols, numeric_cols


def get_full_feature_list(df, exclude_cols=DEFAULT_EXCLUDE_COLS):
    """The 'full' feature set: every predictor surviving feature
    engineering, prior to Lasso/RF-based selection."""
    return [c for c in df.columns if c not in exclude_cols]


def build_linear_rf_matrix(df, feature_list):
    """Numeric matrix for Logistic Regression / Random Forest:
    ordinal -> integer codes, nominal -> one-hot (k-1), numeric -> as-is."""
    ordinal_cols, nominal_cols, numeric_cols = classify_features(df, feature_list)

    X = pd.DataFrame(index=df.index)
    for col in ordinal_cols:
        X[col] = df[col].cat.codes.astype(float)
    if nominal_cols:
        dummies = pd.get_dummies(df[nominal_cols], drop_first=True)
        X = pd.concat([X, dummies], axis=1)
    for col in numeric_cols:
        X[col] = df[col].astype(float)
    return X


def build_gbm_matrix(df, feature_list, nominal_categories=None):
    """Matrix for XGBoost / LightGBM: ordinal -> integer codes (treated as
    numeric, since native categorical support does not preserve order),
    nominal -> kept as pandas 'category' dtype for native categorical
    handling, numeric -> as-is.

    `nominal_categories`, if provided, is a dict {column: [categories...]}
    used to force each nominal column onto the exact category set seen
    during training (e.g. config['nominal_categories']). This matters for
    new data such as the test set: a category absent from training would
    otherwise silently become a different (and inconsistent) set of codes,
    or — if forced via this dtype — become NaN, which is the correct and
    visible behavior to catch unseen categories rather than mis-encode them.
    """
    ordinal_cols, nominal_cols, numeric_cols = classify_features(df, feature_list)

    X = pd.DataFrame(index=df.index)
    for col in ordinal_cols:
        X[col] = df[col].cat.codes.astype(int)
    for col in nominal_cols:
        series = df[col]
        if nominal_categories is not None and col in nominal_categories:
            series = series.astype(pd.CategoricalDtype(categories=nominal_categories[col]))
        X[col] = series
    for col in numeric_cols:
        X[col] = df[col].astype(float)
    return X
