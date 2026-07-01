"""
inference.py — applies the full fitted pipeline (Strategy B imputation +
feature engineering + encoding) to new raw data and generates predictions
with the final trained LightGBM models.

This module is for INFERENCE ONLY: it loads already-fitted artifacts and
never re-fits anything. It is intended for the competition test set, using
the artifacts produced during training (see the persistence cell run at
the end of the training notebook).
"""

import os

import joblib
import numpy as np
import pandas as pd

from . import encoding
from . import feature_engineering
from . import imputation

EMPLOYMENT_STATUS_RANDOM_STATE = 42

# Every raw predictor column from the original survey (excludes
# respondent_id and the two targets), in the order the raw CSVs use them.
ALL_RAW_COLUMNS = [
    'h1n1_concern', 'h1n1_knowledge', 'behavioral_antiviral_meds',
    'behavioral_avoidance', 'behavioral_face_mask', 'behavioral_wash_hands',
    'behavioral_large_gatherings', 'behavioral_outside_home', 'behavioral_touch_face',
    'doctor_recc_h1n1', 'doctor_recc_seasonal', 'chronic_med_condition',
    'child_under_6_months', 'health_worker', 'health_insurance',
    'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
    'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc',
    'age_group', 'education', 'race', 'sex', 'income_poverty',
    'marital_status', 'rent_or_own', 'employment_status',
    'hhs_geo_region', 'census_msa', 'household_adults', 'household_children',
    'employment_industry', 'employment_occupation',
]

# Columns that are free-text/category in the raw CSV (as opposed to
# numerically-coded survey responses). A brand-new, all-NaN row defaults to
# float64 for every column, which would silently misrepresent these as
# numeric instead of categorical. Forcing them to 'object' here makes a
# freshly built row behave exactly like a row read from a real CSV, so the
# rest of the pipeline (prepare_raw_dataframe's object/string detection)
# needs no special-casing for this single-row scenario.
TEXT_COLUMNS = [
    'age_group', 'education', 'race', 'sex', 'income_poverty',
    'marital_status', 'rent_or_own', 'employment_status',
    'hhs_geo_region', 'census_msa', 'employment_industry', 'employment_occupation',
]


def build_input_row(form_answers, respondent_id=0):
    """
    Builds a single-row raw dataframe (same shape/dtypes as one row of
    training_set_features.csv) from a partial dictionary of form answers.

    `form_answers` should map a subset of ALL_RAW_COLUMNS to already-encoded
    raw values — e.g. {'age_group': '55 - 64 Years', 'doctor_recc_h1n1': 1.0,
    'opinion_h1n1_risk': 3}. Any column not present in form_answers is left
    as NaN, and is handled downstream by Strategy B's missing-value
    treatment exactly as it would be for a real survey respondent who
    skipped that question — no special-casing is needed here for "unknown"
    answers beyond the mapping already applied by the caller (e.g. the
    Streamlit form maps "No sé" to NaN for Yes/No questions, and to 3 for
    opinion scales, before calling this function).

    Returns a one-row DataFrame with columns ['respondent_id'] + ALL_RAW_COLUMNS,
    ready to pass into predict() or preprocess_new_data().
    """
    row = {}
    for col in ALL_RAW_COLUMNS:
        if col in TEXT_COLUMNS:
            row[col] = pd.array([None], dtype='object')
        else:
            row[col] = [np.nan]

    df_row = pd.DataFrame(row, index=[0])

    for col, value in form_answers.items():
        if col not in ALL_RAW_COLUMNS:
            raise ValueError(f"Unknown column in form_answers: {col!r}")
        df_row.at[0, col] = value

    df_row.insert(0, 'respondent_id', [respondent_id])

    return df_row


def load_artifacts(artifacts_dir='artifacts'):
    """Loads every persisted artifact needed for inference. Returns a dict
    with keys: fitted_objects, config, final_models."""
    fitted_objects = {
        'group2_imputer': joblib.load(os.path.join(artifacts_dir, 'group2_imputer.joblib')),
        'employment_classifier': joblib.load(os.path.join(artifacts_dir, 'employment_classifier.joblib')),
        'employment_scaler': joblib.load(os.path.join(artifacts_dir, 'employment_scaler.joblib')),
        'income_imputer': joblib.load(os.path.join(artifacts_dir, 'income_imputer.joblib')),
    }

    config = joblib.load(os.path.join(artifacts_dir, 'config.joblib'))
    # group1_mode_values / group1_median_values were patched into config.joblib
    # directly (see the persistence-patch step) rather than into a separate
    # fitted-objects file. Mirror them into fitted_objects so
    # transform_strategy_b receives them the same way regardless of where
    # they were originally saved.
    fitted_objects['group1_mode_values'] = config['group1_mode_values']
    fitted_objects['group1_median_values'] = config['group1_median_values']

    final_models = joblib.load(os.path.join(artifacts_dir, 'final_models.joblib'))

    return {
        'fitted_objects': fitted_objects,
        'config': config,
        'final_models': final_models,
    }


def preprocess_new_data(df_raw, artifacts, random_state=EMPLOYMENT_STATUS_RANDOM_STATE):
    """
    Runs the full preprocessing chain on new raw data (e.g. the competition
    test set): initial dtype setup -> Strategy B transform (using already-
    fitted objects) -> feature engineering. Returns the fully processed
    dataframe, ready for encoding.
    """
    config = artifacts['config']
    fitted_objects = artifacts['fitted_objects']

    df = imputation.prepare_raw_dataframe(df_raw, config)
    df = imputation.transform_strategy_b(df, fitted_objects, config, random_state=random_state)
    df = feature_engineering.apply_feature_engineering(df)

    return df


def predict(df_raw_test, artifacts_dir='artifacts', random_state=EMPLOYMENT_STATUS_RANDOM_STATE):
    """
    End-to-end inference: takes a raw test dataframe (same columns as
    training_set_features.csv, including respondent_id, WITHOUT target
    columns), and returns a submission-ready dataframe with columns
    respondent_id, h1n1_vaccine, seasonal_vaccine (float probabilities).
    """
    artifacts = load_artifacts(artifacts_dir)
    config = artifacts['config']
    final_models = artifacts['final_models']

    df_processed = preprocess_new_data(df_raw_test, artifacts, random_state=random_state)

    full_features = config['full_features']
    nominal_categories = config['nominal_categories']

    X = encoding.build_gbm_matrix(df_processed, full_features, nominal_categories=nominal_categories)

    # Flag any row where an unseen category produced NaN in a nominal
    # column, so this does not pass silently into the model.
    nan_counts = X.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print("Warning — unseen categories produced NaN in:")
        print(nan_cols.to_string())

    h1n1_proba = final_models['h1n1_vaccine'].predict_proba(X)[:, 1]
    seasonal_proba = final_models['seasonal_vaccine'].predict_proba(X)[:, 1]

    submission = pd.DataFrame({
        'respondent_id': df_raw_test['respondent_id'].values,
        'h1n1_vaccine': h1n1_proba,
        'seasonal_vaccine': seasonal_proba,
    })

    return submission
