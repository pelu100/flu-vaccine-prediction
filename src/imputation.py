"""
imputation.py — Strategy B (mechanism-informed) missing-value imputation.

This module separates the FIT step (learning imputers/classifiers from a
training set) from the TRANSFORM step (applying already-fitted objects to
new data — validation or test — without re-fitting anything).

Design principle: every transform function receives its fitted objects as
explicit arguments, rather than loading them internally. This keeps the
module testable and makes data flow explicit, consistent with the rest of
the project's approach.
"""

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONSTANTS — deterministic mappings that do not depend on fitted state
# =============================================================================

GROUP1_MODE_COLS = [
    'behavioral_antiviral_meds', 'behavioral_avoidance', 'behavioral_face_mask',
    'behavioral_wash_hands', 'behavioral_large_gatherings',
    'behavioral_outside_home', 'behavioral_touch_face',
    'h1n1_concern', 'h1n1_knowledge'
]
GROUP1_MEDIAN_COLS = ['household_adults', 'household_children']

GROUP2_ORDINAL_COLS = [
    'education', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
    'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
    'opinion_seas_risk', 'opinion_seas_sick_from_vacc'
]
GROUP2_BINARY_STR_COLS = ['marital_status', 'rent_or_own']
GROUP2_BINARY_NUM_COLS = ['health_worker', 'chronic_med_condition', 'child_under_6_months']

GROUP3_COLS = ['doctor_recc_h1n1', 'doctor_recc_seasonal']
GROUP5_COLS = ['employment_industry', 'employment_occupation']

BINARY_TO_LABEL = {0.0: 'No', 1.0: 'Yes'}


# =============================================================================
# DTYPE PREPARATION — mirrors the original notebook's initial type setup
# (string -> category, ordered dtype for ordinal variables)
# =============================================================================

def prepare_raw_dataframe(df, config):
    """
    Applies the same initial dtype conversions used when df_train/df_val
    were first prepared: object/string columns -> category, and ordered
    CategoricalDtype for the three ordinal variables. Must be called on any
    new raw dataframe (e.g. the competition test set) before imputation.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype('category')

    age_group_dtype = CategoricalDtype(categories=config['age_group_order'], ordered=True)
    education_dtype = CategoricalDtype(categories=config['education_order'], ordered=True)
    income_poverty_dtype = CategoricalDtype(categories=config['income_poverty_order'], ordered=True)

    if 'age_group' in df.columns:
        df['age_group'] = df['age_group'].astype(age_group_dtype)
    if 'education' in df.columns:
        df['education'] = df['education'].astype(education_dtype)
    if 'income_poverty' in df.columns:
        df['income_poverty'] = df['income_poverty'].astype(income_poverty_dtype)

    return df


# =============================================================================
# ORDINAL <-> CODE HELPERS
# =============================================================================

def ordinal_to_codes(series, order):
    mapping = {label: i for i, label in enumerate(order)}
    return series.map(mapping)


def codes_to_ordinal_label(codes, order):
    clipped = codes.clip(lower=0, upper=len(order) - 1)
    rounded = clipped.round().astype(int)
    return rounded.map(lambda i: order[i])


def get_binary_str_reverse_maps(binary_str_maps):
    return {col: {v: k for k, v in m.items()} for col, m in binary_str_maps.items()}


# =============================================================================
# GROUP 2 MATRIX BUILDERS (shared by fit and transform)
# =============================================================================

def build_group2_matrix(df, config):
    """Numeric matrix for the Group 2 IterativeImputer — ordinal + binary
    survey-engagement block, plus age_group/sex as auxiliary predictors."""
    age_group_order = config['age_group_order']
    education_order = config['education_order']
    binary_str_maps = config['binary_str_maps']

    parts = {}
    parts['education_code'] = ordinal_to_codes(df['education'], education_order)
    for col in ['opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
                'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
                'opinion_seas_risk', 'opinion_seas_sick_from_vacc']:
        parts[col] = df[col]

    for col in GROUP2_BINARY_STR_COLS:
        parts[f'{col}_num'] = df[col].map(binary_str_maps[col])

    for col in GROUP2_BINARY_NUM_COLS:
        parts[col] = df[col]

    parts['age_group_code'] = ordinal_to_codes(df['age_group'], age_group_order)
    parts['sex_num'] = df['sex'].map(binary_str_maps['sex'])

    return pd.DataFrame(parts, index=df.index)


def apply_group2_results(df, result, config):
    """Writes IterativeImputer output back onto the original-style columns."""
    education_order = config['education_order']
    binary_str_maps = config['binary_str_maps']
    binary_str_reverse_maps = get_binary_str_reverse_maps(binary_str_maps)

    df = df.copy()
    df['education'] = codes_to_ordinal_label(result['education_code'], education_order)
    for col in ['opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
                'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
                'opinion_seas_risk', 'opinion_seas_sick_from_vacc']:
        df[col] = result[col].clip(1, 5).round()

    for col in GROUP2_BINARY_STR_COLS:
        rounded = result[f'{col}_num'].clip(0, 1).round().astype(int)
        df[col] = rounded.map(binary_str_reverse_maps[col])

    for col in GROUP2_BINARY_NUM_COLS:
        df[col] = result[col].clip(0, 1).round()

    return df


# =============================================================================
# EMPLOYMENT_STATUS FEATURE BUILDER (shared by fit and transform)
# =============================================================================

def build_employment_features(df, config):
    """Numeric features for the employment_status classifier — reuses the
    already-imputed (no-missing) Group 2 columns plus age_group/sex."""
    age_group_order = config['age_group_order']
    education_order = config['education_order']
    binary_str_maps = config['binary_str_maps']

    parts = {}
    parts['education_code'] = ordinal_to_codes(df['education'], education_order)
    for col in ['opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
                'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
                'opinion_seas_risk', 'opinion_seas_sick_from_vacc']:
        parts[col] = df[col]
    for col in GROUP2_BINARY_STR_COLS:
        parts[f'{col}_num'] = df[col].map(binary_str_maps[col])
    for col in GROUP2_BINARY_NUM_COLS:
        parts[col] = df[col]
    parts['age_group_code'] = ordinal_to_codes(df['age_group'], age_group_order)
    parts['sex_num'] = df['sex'].map(binary_str_maps['sex'])
    return pd.DataFrame(parts, index=df.index)


def sample_employment_status(features, missing_mask, classifier, scaler, rng):
    """Predicts class probabilities for missing rows and samples one
    category per row from that distribution (preserves natural variance
    instead of always picking the most likely class)."""
    if missing_mask.sum() == 0:
        return pd.Series(dtype='object', index=features.index)

    X_missing = features.loc[missing_mask]
    X_missing_scaled = scaler.transform(X_missing)
    proba = classifier.predict_proba(X_missing_scaled)
    classes = classifier.classes_

    sampled = [rng.choice(classes, p=proba[i]) for i in range(proba.shape[0])]
    return pd.Series(sampled, index=X_missing.index)


# =============================================================================
# INCOME_POVERTY MATRIX BUILDER (shared by fit and transform)
# =============================================================================

def build_income_matrix(df_imp_b_so_far, df_original, config):
    income_poverty_order = config['income_poverty_order']
    education_order = config['education_order']
    age_group_order = config['age_group_order']
    binary_str_maps = config['binary_str_maps']
    employment_status_categories = config['employment_status_categories']

    parts = {}
    parts['income_poverty_code'] = ordinal_to_codes(
        df_original['income_poverty'], income_poverty_order
    )
    parts['education_code'] = ordinal_to_codes(
        df_imp_b_so_far['education'], education_order
    )
    parts['age_group_code'] = ordinal_to_codes(
        df_imp_b_so_far['age_group'], age_group_order
    )
    parts['rent_or_own_num'] = df_imp_b_so_far['rent_or_own'].map(binary_str_maps['rent_or_own'])

    dummies = pd.get_dummies(df_imp_b_so_far['employment_status'], prefix='employment_status', dtype=float)
    for cat in employment_status_categories:
        dummy_col = f'employment_status_{cat}'
        if dummy_col not in dummies.columns:
            dummies[dummy_col] = 0.0
    # Keep only the columns the imputer was originally fitted on, in the
    # same order, in case the new data introduces unexpected extra dummies.
    expected_dummy_cols = [f'employment_status_{cat}' for cat in employment_status_categories]
    parts.update({c: dummies[c] for c in expected_dummy_cols})

    return pd.DataFrame(parts, index=df_imp_b_so_far.index)


# =============================================================================
# DOCTOR_RECC / HEALTH_INSURANCE — deterministic "Unknown" encoding
# =============================================================================

def apply_unknown_category_encoding(df, cols):
    """Maps 0/1 -> 'No'/'Yes' and fills missing with an explicit 'Unknown'
    category. Fully deterministic — no fitted state involved."""
    df = df.copy()
    for col in cols:
        df[col] = df[col].map(BINARY_TO_LABEL)
        df[col] = df[col].fillna('Unknown').astype('category')
    return df


def apply_not_employed_encoding(df, cols):
    """Fills missing employment_industry/employment_occupation with an
    explicit 'Not_Employed' category. Fully deterministic."""
    df = df.copy()
    for col in cols:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype('category')
        if 'Not_Employed' not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories(['Not_Employed'])
        df[col] = df[col].fillna('Not_Employed')
    return df


def recast_categorical_dtypes(df, config):
    """Re-applies ordered/unordered category dtypes lost during imputation
    (codes_to_ordinal_label and string-mapping steps return plain
    object/string Series, not category). Mirrors the original notebook's
    dedicated re-casting cell, run once after all imputation groups."""
    df = df.copy()

    education_dtype = pd.CategoricalDtype(categories=config['education_order'], ordered=True)
    income_poverty_dtype = pd.CategoricalDtype(categories=config['income_poverty_order'], ordered=True)
    marital_status_dtype = pd.CategoricalDtype(categories=['Married', 'Not Married'], ordered=False)
    rent_or_own_dtype = pd.CategoricalDtype(categories=['Own', 'Rent'], ordered=False)

    df['education'] = df['education'].astype(education_dtype)
    df['income_poverty'] = df['income_poverty'].astype(income_poverty_dtype)
    df['marital_status'] = df['marital_status'].astype(marital_status_dtype)
    df['rent_or_own'] = df['rent_or_own'].astype(rent_or_own_dtype)

    return df

# =============================================================================
# FIT — learns all Strategy B imputers/classifiers from a training set
# =============================================================================

def fit_strategy_b(df_train_raw, random_state=42):
    """
    Fits the full Strategy B pipeline on a raw training dataframe (i.e. one
    that has already passed through prepare_raw_dataframe, but has not yet
    been imputed). Returns the imputed dataframe, a dict of fitted objects,
    and a dict of config values needed later to transform new data.

    Use this to (re)fit Strategy B — for example, when retraining on
    train+validation combined ahead of generating competition predictions.
    Do NOT use this on the test set: test data must only go through
    transform_strategy_b with the already-fitted objects.
    """
    rng = np.random.default_rng(random_state)
    df = df_train_raw.copy()

    age_group_order = ['18 - 34 Years', '35 - 44 Years', '45 - 54 Years',
                        '55 - 64 Years', '65+ Years']
    education_order = ['< 12 Years', '12 Years', 'Some College', 'College Graduate']
    income_poverty_order = ['Below Poverty', '<= $75,000, Above Poverty', '> $75,000']

    binary_str_maps = {
        'marital_status': {'Married': 1, 'Not Married': 0},
        'rent_or_own': {'Own': 1, 'Rent': 0},
        'sex': {'Female': 1, 'Male': 0},
    }

    employment_status_categories = (
        df['employment_status'].cat.categories.tolist()
        if hasattr(df['employment_status'], 'cat')
        else sorted(df['employment_status'].dropna().unique())
    )

    config = {
        'age_group_order': age_group_order,
        'education_order': education_order,
        'income_poverty_order': income_poverty_order,
        'binary_str_maps': binary_str_maps,
        'employment_status_categories': employment_status_categories,
    }

    # --- Group 1 ---
    group1_mode_values = {col: df[col].mode(dropna=True)[0] for col in GROUP1_MODE_COLS}
    group1_median_values = {col: df[col].median() for col in GROUP1_MEDIAN_COLS}
    for col in GROUP1_MODE_COLS:
        df[col] = df[col].fillna(group1_mode_values[col])
    for col in GROUP1_MEDIAN_COLS:
        df[col] = df[col].fillna(group1_median_values[col])

    # --- Group 2 ---
    group2_matrix = build_group2_matrix(df, config)
    group2_imputer = IterativeImputer(
        estimator=BayesianRidge(), random_state=random_state,
        max_iter=15, sample_posterior=True
    )
    group2_result = pd.DataFrame(
        group2_imputer.fit_transform(group2_matrix),
        columns=group2_matrix.columns, index=group2_matrix.index
    )
    df = apply_group2_results(df, group2_result, config)

    # --- Group 2B: employment_status ---
    employment_features = build_employment_features(df, config)
    observed_mask = df_train_raw['employment_status'].notna()
    missing_mask = df_train_raw['employment_status'].isna()

    X_fit = employment_features.loc[observed_mask]
    y_fit = df_train_raw.loc[observed_mask, 'employment_status']

    employment_scaler = StandardScaler()
    X_fit_scaled = employment_scaler.fit_transform(X_fit)

    employment_classifier = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=random_state)
    employment_classifier.fit(X_fit_scaled, y_fit)

    sampled = sample_employment_status(
        employment_features, missing_mask, employment_classifier, employment_scaler, rng
    )
    employment_status_dtype = pd.CategoricalDtype(
        categories=employment_status_categories, ordered=False
    )
    new_employment_status = df_train_raw['employment_status'].astype(object).copy()
    new_employment_status.loc[sampled.index] = sampled
    df['employment_status'] = new_employment_status.astype(employment_status_dtype)

    # --- Group 3: doctor_recc pair ---
    df = apply_unknown_category_encoding(df, GROUP3_COLS)

    # --- Group 4a: health_insurance ---
    df = apply_unknown_category_encoding(df, ['health_insurance'])

    # --- Group 4b: income_poverty ---
    df['income_poverty_was_missing'] = df_train_raw['income_poverty'].isna().astype(int)
    income_matrix = build_income_matrix(df, df_train_raw, config)
    income_imputer = IterativeImputer(
        estimator=BayesianRidge(), random_state=random_state,
        max_iter=15, sample_posterior=True
    )
    income_result = pd.DataFrame(
        income_imputer.fit_transform(income_matrix),
        columns=income_matrix.columns, index=income_matrix.index
    )
    df['income_poverty'] = codes_to_ordinal_label(
        income_result['income_poverty_code'], income_poverty_order
    )

    # --- Group 5: employment_industry / employment_occupation ---
    df = apply_not_employed_encoding(df, GROUP5_COLS)

    df = recast_categorical_dtypes(df, config)

    fitted_objects = {
        'group2_imputer': group2_imputer,
        'employment_classifier': employment_classifier,
        'employment_scaler': employment_scaler,
        'income_imputer': income_imputer,
        'group1_mode_values': group1_mode_values,
        'group1_median_values': group1_median_values,
    }

    config['nominal_categories'] = {
        col: df[col].cat.categories.tolist()
        for col in df.columns
        if isinstance(df[col].dtype, pd.CategoricalDtype)
    }

    return df, fitted_objects, config


# =============================================================================
# TRANSFORM — applies already-fitted Strategy B objects to new data
# =============================================================================

def transform_strategy_b(df_new_raw, fitted_objects, config, random_state=42):
    """
    Applies the already-fitted Strategy B pipeline to new data (validation
    or test) without re-fitting anything. `fitted_objects` and `config`
    should come from fit_strategy_b (or be loaded from the persisted
    artifacts saved during the original training run).

    `random_state` controls only the employment_status sampling step
    (Group 2B), so repeated calls with the same value are reproducible.
    """
    rng = np.random.default_rng(random_state)
    df = df_new_raw.copy()

    group1_mode_values = fitted_objects['group1_mode_values']
    group1_median_values = fitted_objects['group1_median_values']
    group2_imputer = fitted_objects['group2_imputer']
    employment_classifier = fitted_objects['employment_classifier']
    employment_scaler = fitted_objects['employment_scaler']
    income_imputer = fitted_objects['income_imputer']

    income_poverty_order = config['income_poverty_order']

    # --- Group 1 ---
    for col in GROUP1_MODE_COLS:
        df[col] = df[col].fillna(group1_mode_values[col])
    for col in GROUP1_MEDIAN_COLS:
        df[col] = df[col].fillna(group1_median_values[col])

    # --- Group 2 ---
    group2_matrix = build_group2_matrix(df, config)
    group2_result = pd.DataFrame(
        group2_imputer.transform(group2_matrix),
        columns=group2_matrix.columns, index=group2_matrix.index
    )
    df = apply_group2_results(df, group2_result, config)

    # --- Group 2B: employment_status ---
    employment_features = build_employment_features(df, config)
    missing_mask = df_new_raw['employment_status'].isna()

    sampled = sample_employment_status(
        employment_features, missing_mask, employment_classifier, employment_scaler, rng
    )
    employment_status_dtype = pd.CategoricalDtype(
        categories=config['employment_status_categories'], ordered=False
    )
    new_employment_status = df_new_raw['employment_status'].astype(object).copy()
    new_employment_status.loc[sampled.index] = sampled
    df['employment_status'] = new_employment_status.astype(employment_status_dtype)

    # --- Group 3: doctor_recc pair ---
    df = apply_unknown_category_encoding(df, GROUP3_COLS)

    # --- Group 4a: health_insurance ---
    df = apply_unknown_category_encoding(df, ['health_insurance'])

    # --- Group 4b: income_poverty ---
    df['income_poverty_was_missing'] = df_new_raw['income_poverty'].isna().astype(int)
    income_matrix = build_income_matrix(df, df_new_raw, config)
    income_result = pd.DataFrame(
        income_imputer.transform(income_matrix),
        columns=income_matrix.columns, index=income_matrix.index
    )
    df['income_poverty'] = codes_to_ordinal_label(
        income_result['income_poverty_code'], income_poverty_order
    )

    # --- Group 5: employment_industry / employment_occupation ---
    df = apply_not_employed_encoding(df, GROUP5_COLS)

    df = recast_categorical_dtypes(df, config)

    return df