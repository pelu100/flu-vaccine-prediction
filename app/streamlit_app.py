"""
streamlit_app.py — Interactive flu vaccination probability predictor.

Reuses the same production pipeline (src/imputation.py, feature_engineering.py,
encoding.py, inference.py) used for the competition test set, so the
behavior here is guaranteed consistent with the documented, validated
model — no separate prediction logic is reimplemented for the app.

The questionnaire covers every raw variable that a real respondent could
plausibly answer (31 of 35). Four variables are excluded: employment_industry
and employment_occupation are anonymized codes with no answerable meaning;
hhs_geo_region and census_msa, while answerable, are dropped during feature
engineering and have zero effect on the prediction, so asking them would be
wasted effort.

Run locally with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Make the project's src/ package importable regardless of the working
# directory the app is launched from (Streamlit Cloud included).
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src import encoding, inference

ARTIFACTS_DIR = str(Path(__file__).resolve().parent.parent / "artifacts")

st.set_page_config(page_title="Predictor de vacunación gripal", page_icon="💉", layout="centered")


# =============================================================================
# CACHED LOADING — artifacts and SHAP explainers are loaded once per session,
# not re-loaded on every form interaction.
# =============================================================================

@st.cache_resource
def load_pipeline():
    artifacts = inference.load_artifacts(ARTIFACTS_DIR)
    explainers = {
        target: shap.TreeExplainer(model)
        for target, model in artifacts["final_models"].items()
    }
    return artifacts, explainers


artifacts, explainers = load_pipeline()
config = artifacts["config"]


# =============================================================================
# FORM ANSWER MAPPINGS — UI labels (Spanish) -> raw encoded values expected
# by the pipeline. "No sé" maps to NaN wherever Strategy B already has a
# mechanism to handle missingness for that variable (which is every
# variable here except age_group and sex — see the notes below). For the
# six 1-5 opinion scales, "No sé" maps to value 3, the survey's own
# official "Don't know" code, rather than to NaN.
# =============================================================================

AGE_GROUP_OPTIONS = config["age_group_order"]

# 'age_group' and 'sex' are used internally as auxiliary predictors by the
# Strategy B imputers, which assume they are always present — in the real
# survey data they are essentially never missing, and no imputation group
# exists for either of them. Unlike every other question, they have no
# "No sé" option: leaving either NaN would break internal imputation steps
# that were never designed to tolerate a missing value here.
SEX_OPTIONS = {"Mujer": "Female", "Hombre": "Male"}

YES_NO_UNKNOWN = {"Sí": 1.0, "No": 0.0, "No sé": np.nan}

OPINION_SCALE_RISK = {
    "Muy bajo": 1, "Algo bajo": 2, "No sé": 3, "Algo alto": 4, "Muy alto": 5,
}
OPINION_SCALE_EFFECTIVE = {
    "Nada efectiva": 1, "Poco efectiva": 2, "No sé": 3, "Bastante efectiva": 4, "Muy efectiva": 5,
}
OPINION_SCALE_WORRY = {
    "Nada preocupado/a": 1, "Poco preocupado/a": 2, "No sé": 3, "Algo preocupado/a": 4, "Muy preocupado/a": 5,
}

# h1n1_concern and h1n1_knowledge use their own 0-3 / 0-2 scales, with no
# official "Don't know" code built in (unlike the 1-5 opinion scales
# above) — so here "No sé" maps to NaN, handled by Strategy B's Group 1
# mode imputation, which was specifically designed for these two columns.
H1N1_CONCERN_OPTIONS = {
    "Nada preocupado/a": 0, "Poco preocupado/a": 1, "Algo preocupado/a": 2, "Muy preocupado/a": 3, "No sé": np.nan,
}
H1N1_KNOWLEDGE_OPTIONS = {
    "Ningún conocimiento": 0, "Algo de conocimiento": 1, "Mucho conocimiento": 2, "No sé": np.nan,
}

EDUCATION_OPTIONS = {
    "Menos de 12 años de estudios": "< 12 Years",
    "12 años de estudios (secundaria)": "12 Years",
    "Estudios universitarios sin finalizar": "Some College",
    "Título universitario": "College Graduate",
    "No sé / prefiero no decirlo": np.nan,
}
RACE_OPTIONS = {
    "Blanca": "White", "Negra": "Black", "Hispana": "Hispanic", "Otra o múltiple": "Other or Multiple",
    "No sé / prefiero no decirlo": np.nan,
}
INCOME_POVERTY_OPTIONS = {
    "Por debajo del umbral de pobreza": "Below Poverty",
    "Hasta $75,000 (por encima del umbral de pobreza)": "<= $75,000, Above Poverty",
    "Más de $75,000": "> $75,000",
    "No sé / prefiero no decirlo": np.nan,
}
MARITAL_STATUS_OPTIONS = {"Casado/a": "Married", "No casado/a": "Not Married", "No sé": np.nan}
RENT_OR_OWN_OPTIONS = {"En propiedad": "Own", "En alquiler": "Rent", "No sé": np.nan}
EMPLOYMENT_STATUS_OPTIONS = {
    "Empleado/a": "Employed", "Fuera de la fuerza laboral": "Not in Labor Force",
    "Desempleado/a": "Unemployed", "No sé": np.nan,
}
HOUSEHOLD_COUNT_OPTIONS = {"0": 0.0, "1": 1.0, "2": 2.0, "3 o más": 3.0, "No sé": np.nan}


# =============================================================================
# TRANSLATION LAYER — friendly Spanish names/values for the explanation
# section. Reverse maps are built from the option dicts above, so question
# wording and explanation wording always stay in sync automatically.
# =============================================================================

def make_reverse_map(options_dict):
    """Builds a raw_value -> label reverse map, skipping the 'No sé' (NaN)
    entry since that case is handled separately by translate_value."""
    return {v: k for k, v in options_dict.items() if not (isinstance(v, float) and pd.isna(v))}


EDUCATION_VALUE_LABELS = make_reverse_map(EDUCATION_OPTIONS)
RACE_VALUE_LABELS = make_reverse_map(RACE_OPTIONS)
INCOME_POVERTY_VALUE_LABELS = make_reverse_map(INCOME_POVERTY_OPTIONS)
MARITAL_STATUS_VALUE_LABELS = make_reverse_map(MARITAL_STATUS_OPTIONS)
RENT_OR_OWN_VALUE_LABELS = make_reverse_map(RENT_OR_OWN_OPTIONS)
EMPLOYMENT_STATUS_VALUE_LABELS = make_reverse_map(EMPLOYMENT_STATUS_OPTIONS)
H1N1_CONCERN_VALUE_LABELS = make_reverse_map(H1N1_CONCERN_OPTIONS)
H1N1_KNOWLEDGE_VALUE_LABELS = make_reverse_map(H1N1_KNOWLEDGE_OPTIONS)

AGE_GROUP_ES = {
    '18 - 34 Years': '18-34 años', '35 - 44 Years': '35-44 años', '45 - 54 Years': '45-54 años',
    '55 - 64 Years': '55-64 años', '65+ Years': '65+ años',
}
SEX_ES = {'Male': 'Hombre', 'Female': 'Mujer'}
YESNO_ES = {'Yes': 'Sí', 'No': 'No', 'Unknown': 'No sé / no indicado'}

RISK_VALUE_LABELS = {v: k for k, v in OPINION_SCALE_RISK.items()}
EFFECTIVE_VALUE_LABELS = {v: k for k, v in OPINION_SCALE_EFFECTIVE.items()}
WORRY_VALUE_LABELS = {v: k for k, v in OPINION_SCALE_WORRY.items()}

COLUMN_VALUE_SCALE = {
    'opinion_h1n1_risk': RISK_VALUE_LABELS,
    'opinion_seas_risk': RISK_VALUE_LABELS,
    'opinion_h1n1_vacc_effective': EFFECTIVE_VALUE_LABELS,
    'opinion_seas_vacc_effective': EFFECTIVE_VALUE_LABELS,
    'opinion_seas_sick_from_vacc': WORRY_VALUE_LABELS,
    'opinion_h1n1_sick_from_vacc': WORRY_VALUE_LABELS,
}

BINARY_NUMERIC_COLS_PREFIXES = ('behavioral_',)
BINARY_NUMERIC_COLS = {'chronic_med_condition', 'health_worker', 'child_under_6_months'}

FEATURE_LABELS = {
    'age_group': 'tu grupo de edad',
    'sex': 'tu sexo',
    'doctor_recc_h1n1': 'la recomendación de tu médico sobre la vacuna H1N1',
    'doctor_recc_seasonal': 'la recomendación de tu médico sobre la vacuna estacional',
    'health_insurance': 'tener seguro médico',
    'opinion_h1n1_risk': 'tu percepción del riesgo de H1N1',
    'opinion_h1n1_vacc_effective': 'tu opinión sobre la efectividad de la vacuna H1N1',
    'opinion_h1n1_sick_from_vacc': 'tu preocupación por enfermar a causa de la vacuna H1N1',
    'opinion_seas_risk': 'tu percepción del riesgo de la gripe estacional',
    'opinion_seas_vacc_effective': 'tu opinión sobre la efectividad de la vacuna estacional',
    'opinion_seas_sick_from_vacc': 'tu preocupación por enfermar a causa de la vacuna estacional',
    'h1n1_concern': 'tu nivel de preocupación por el H1N1',
    'h1n1_knowledge': 'tu nivel de conocimiento sobre el H1N1',
    'behavioral_antiviral_meds': 'haber tomado medicamentos antivirales',
    'behavioral_avoidance': 'evitar el contacto cercano con personas con síntomas',
    'behavioral_face_mask': 'haber comprado mascarillas',
    'behavioral_wash_hands': 'lavarte las manos con frecuencia',
    'behavioral_large_gatherings': 'reducir tu asistencia a grandes reuniones',
    'behavioral_outside_home': 'reducir el contacto con personas fuera de tu hogar',
    'behavioral_touch_face': 'evitar tocarte los ojos, la nariz o la boca',
    'chronic_med_condition': 'tener una condición médica crónica',
    'child_under_6_months': 'tener contacto regular con un bebé menor de 6 meses',
    'health_worker': 'trabajar en el sector sanitario',
    'education': 'tu nivel educativo',
    'race': 'tu raza/etnia declarada',
    'income_poverty': 'tu nivel de ingresos',
    'income_poverty_was_missing': 'que no indicaras tu nivel de ingresos',
    'marital_status': 'tu estado civil',
    'rent_or_own': 'tener vivienda en propiedad o en alquiler',
    'employment_status': 'tu situación laboral',
    'household_adults': 'el número de adultos en tu hogar',
    'household_children': 'el número de menores en tu hogar',
}


def friendly_label(col):
    """Returns a natural-language Spanish phrase for a feature column,
    falling back to a prettified version of the raw column name."""
    return FEATURE_LABELS.get(col, col.replace('_', ' '))


def translate_value(col, raw_value):
    """Returns a human-readable Spanish string for a feature's underlying
    (pre-encoding) value, as it appears in df_processed — not the encoded
    matrix X. Falls back to the raw value's string form when no specific
    translation is defined."""
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return "no indicado"

    if col == 'age_group':
        return AGE_GROUP_ES.get(raw_value, str(raw_value))
    if col == 'sex':
        return SEX_ES.get(raw_value, str(raw_value))
    if col in ('doctor_recc_h1n1', 'doctor_recc_seasonal', 'health_insurance'):
        return YESNO_ES.get(raw_value, str(raw_value))
    if col in COLUMN_VALUE_SCALE:
        try:
            return COLUMN_VALUE_SCALE[col][int(round(float(raw_value)))]
        except (ValueError, TypeError, KeyError):
            return str(raw_value)
    if col == 'h1n1_concern':
        return H1N1_CONCERN_VALUE_LABELS.get(int(round(float(raw_value))), str(raw_value))
    if col == 'h1n1_knowledge':
        return H1N1_KNOWLEDGE_VALUE_LABELS.get(int(round(float(raw_value))), str(raw_value))
    if col == 'education':
        return EDUCATION_VALUE_LABELS.get(raw_value, str(raw_value))
    if col == 'race':
        return RACE_VALUE_LABELS.get(raw_value, str(raw_value))
    if col == 'income_poverty':
        return INCOME_POVERTY_VALUE_LABELS.get(raw_value, str(raw_value))
    if col == 'marital_status':
        return MARITAL_STATUS_VALUE_LABELS.get(raw_value, str(raw_value))
    if col == 'rent_or_own':
        return RENT_OR_OWN_VALUE_LABELS.get(raw_value, str(raw_value))
    if col == 'employment_status':
        return EMPLOYMENT_STATUS_VALUE_LABELS.get(raw_value, str(raw_value))
    if col in ('household_adults', 'household_children'):
        try:
            n = int(float(raw_value))
            return "3 o más" if n >= 3 else str(n)
        except (ValueError, TypeError):
            return str(raw_value)
    if col == 'income_poverty_was_missing':
        try:
            return "sí" if float(raw_value) == 1 else "no"
        except (ValueError, TypeError):
            return str(raw_value)
    if col in BINARY_NUMERIC_COLS or col.startswith(BINARY_NUMERIC_COLS_PREFIXES):
        try:
            return "Sí" if float(raw_value) == 1 else "No"
        except (ValueError, TypeError):
            return str(raw_value)
    return str(raw_value)


# =============================================================================
# UI
# =============================================================================

st.title("💉 Predictor de vacunación gripal")
st.write(
    "Responde estas preguntas para estimar la probabilidad de haberte vacunado "
    "frente al H1N1 y frente a la gripe estacional, según un modelo entrenado con "
    "datos de la Encuesta Nacional de Gripe H1N1 de 2009 (EE.UU.). "
    "Puedes dejar cualquier pregunta en \"No sé\" si no aplica o prefieres no responder, "
    "salvo el grupo de edad y el sexo, que el modelo necesita siempre."
)

with st.expander("ℹ️ Acerca de este modelo"):
    st.write(
        "Modelo LightGBM, uno entrenado por separado para cada vacuna, con AUC-ROC "
        "de validación de 0.867 (H1N1) y 0.863 (estacional), y un AUC oficial de "
        "0.860 en el conjunto de test de la competición DrivenData. "
        "Este formulario cubre prácticamente todas las variables del modelo que "
        "una persona real puede responder; las pocas que se excluyen corresponden "
        "a códigos anonimizados sin significado preguntable, o a variables que ya "
        "se eliminaron del modelo final por no aportar señal predictiva."
    )

with st.form("vaccination_form"):
    st.subheader("Datos generales")
    age_group = st.selectbox("¿En qué grupo de edad te encuentras?", AGE_GROUP_OPTIONS)
    sex_label = st.selectbox("¿Cuál es tu sexo?", list(SEX_OPTIONS.keys()))
    race_label = st.selectbox("¿Con qué raza/etnia te identificas?", list(RACE_OPTIONS.keys()))
    education_label = st.selectbox("¿Cuál es tu nivel educativo?", list(EDUCATION_OPTIONS.keys()))
    marital_status_label = st.radio("¿Cuál es tu estado civil?", list(MARITAL_STATUS_OPTIONS.keys()), horizontal=True)
    employment_status_label = st.selectbox("¿Cuál es tu situación laboral?", list(EMPLOYMENT_STATUS_OPTIONS.keys()))
    income_poverty_label = st.selectbox("¿Cuál es tu nivel de ingresos del hogar?", list(INCOME_POVERTY_OPTIONS.keys()))
    rent_or_own_label = st.radio("¿Tu vivienda es...?", list(RENT_OR_OWN_OPTIONS.keys()), horizontal=True)
    household_adults_label = st.selectbox("¿Cuántos otros adultos viven en tu hogar?", list(HOUSEHOLD_COUNT_OPTIONS.keys()))
    household_children_label = st.selectbox("¿Cuántos menores viven en tu hogar?", list(HOUSEHOLD_COUNT_OPTIONS.keys()))

    st.subheader("Salud y entorno")
    chronic_med_condition_label = st.radio(
        "¿Tienes alguna condición médica crónica (asma, diabetes, enfermedad cardíaca...)?",
        list(YES_NO_UNKNOWN.keys()), horizontal=True
    )
    child_under_6_months_label = st.radio(
        "¿Tienes contacto regular con un bebé menor de 6 meses?", list(YES_NO_UNKNOWN.keys()), horizontal=True
    )
    health_worker_label = st.radio(
        "¿Trabajas en el sector sanitario?", list(YES_NO_UNKNOWN.keys()), horizontal=True
    )
    health_insurance_label = st.radio(
        "¿Tienes seguro médico?", list(YES_NO_UNKNOWN.keys()), horizontal=True
    )

    st.subheader("Recomendación médica")
    doctor_recc_h1n1_label = st.radio(
        "¿Tu médico te recomendó la vacuna contra el H1N1?", list(YES_NO_UNKNOWN.keys()), horizontal=True
    )
    doctor_recc_seasonal_label = st.radio(
        "¿Tu médico te recomendó la vacuna estacional contra la gripe?", list(YES_NO_UNKNOWN.keys()), horizontal=True
    )

    st.subheader("Conocimiento y preocupación sobre el H1N1")
    h1n1_concern_label = st.radio(
        "¿Cuánto te preocupaba el H1N1?", list(H1N1_CONCERN_OPTIONS.keys()), horizontal=True
    )
    h1n1_knowledge_label = st.radio(
        "¿Cuánto conocimiento tenías sobre el H1N1?", list(H1N1_KNOWLEDGE_OPTIONS.keys()), horizontal=True
    )

    st.subheader("Tu opinión sobre el H1N1")
    opinion_h1n1_risk_label = st.radio(
        "¿Qué riesgo crees que tienes de enfermar de H1N1 si no te vacunas?",
        list(OPINION_SCALE_RISK.keys()), horizontal=True
    )
    opinion_h1n1_vacc_effective_label = st.radio(
        "¿Qué tan efectiva crees que es la vacuna del H1N1?",
        list(OPINION_SCALE_EFFECTIVE.keys()), horizontal=True
    )
    opinion_h1n1_sick_from_vacc_label = st.radio(
        "¿Cuánto te preocupa enfermar a causa de la propia vacuna del H1N1?",
        list(OPINION_SCALE_WORRY.keys()), horizontal=True
    )

    st.subheader("Tu opinión sobre la gripe estacional")
    opinion_seas_risk_label = st.radio(
        "¿Qué riesgo crees que tienes de enfermar de gripe estacional si no te vacunas?",
        list(OPINION_SCALE_RISK.keys()), horizontal=True
    )
    opinion_seas_vacc_effective_label = st.radio(
        "¿Qué tan efectiva crees que es la vacuna estacional?",
        list(OPINION_SCALE_EFFECTIVE.keys()), horizontal=True
    )
    opinion_seas_sick_from_vacc_label = st.radio(
        "¿Cuánto te preocupa enfermar a causa de la propia vacuna estacional?",
        list(OPINION_SCALE_WORRY.keys()), horizontal=True
    )

    st.subheader("Conductas preventivas frente al H1N1")
    behavioral_antiviral_meds_label = st.radio(
        "¿Has tomado medicamentos antivirales?", list(YES_NO_UNKNOWN.keys()), horizontal=True
    )
    behavioral_avoidance_label = st.radio(
        "¿Has evitado el contacto cercano con personas con síntomas gripales?", list(YES_NO_UNKNOWN.keys()), horizontal=True
    )
    behavioral_face_mask_label = st.radio(
        "¿Has comprado una mascarilla?", list(YES_NO_UNKNOWN.keys()), horizontal=True
    )
    behavioral_wash_hands_label = st.radio(
        "¿Te lavas las manos o usas gel desinfectante con frecuencia?", list(YES_NO_UNKNOWN.keys()), horizontal=True
    )
    behavioral_large_gatherings_label = st.radio(
        "¿Has reducido tu asistencia a grandes reuniones?", list(YES_NO_UNKNOWN.keys()), horizontal=True
    )
    behavioral_outside_home_label = st.radio(
        "¿Has reducido el contacto con personas fuera de tu hogar?", list(YES_NO_UNKNOWN.keys()), horizontal=True
    )
    behavioral_touch_face_label = st.radio(
        "¿Has evitado tocarte los ojos, la nariz o la boca?", list(YES_NO_UNKNOWN.keys()), horizontal=True
    )

    submitted = st.form_submit_button("Calcular probabilidad")


# =============================================================================
# PREDICTION + SHAP EXPLANATION
# =============================================================================

def get_individual_explanation(target, X):
    """Returns a shap.Explanation object for a single-row matrix X, robust
    to the LightGBM/SHAP version difference where TreeExplainer output can
    be either a single 2D array or a [negative_class, positive_class] list."""
    explainer = explainers[target]
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(np.atleast_1d(expected_value)) > 1 else expected_value[0]

    return shap.Explanation(
        values=shap_values[0],
        base_values=expected_value,
        data=X.iloc[0].values,
        feature_names=X.columns.tolist(),
    )


def build_plain_language_summary(explanation, df_processed_row, top_n=4):
    """Returns two lists of markdown bullet strings: the top_n features
    that most increased the predicted probability, and the top_n that most
    decreased it — independently ranked, so neither direction gets crowded
    out if most contributions point the same way."""
    contributions = list(zip(explanation.feature_names, explanation.values))

    positives = sorted([c for c in contributions if c[1] > 0], key=lambda t: t[1], reverse=True)[:top_n]
    negatives = sorted([c for c in contributions if c[1] < 0], key=lambda t: t[1])[:top_n]

    def render(items):
        lines = []
        for col, _ in items:
            raw_value = df_processed_row.get(col)
            label = friendly_label(col)
            value_text = translate_value(col, raw_value)
            lines.append(f"- **{label}** ({value_text})")
        return lines

    return render(positives), render(negatives)


def build_friendly_chart_explanation(explanation, df_processed_row):
    """Returns a copy of a shap.Explanation with translated feature names
    and display values, used only for the optional detailed waterfall
    chart — the underlying numeric values/ordering are untouched."""
    friendly_names = [friendly_label(c) for c in explanation.feature_names]
    friendly_values = [translate_value(c, df_processed_row.get(c)) for c in explanation.feature_names]

    return shap.Explanation(
        values=explanation.values,
        base_values=explanation.base_values,
        data=explanation.data,
        display_data=np.array(friendly_values, dtype=object),
        feature_names=friendly_names,
    )


if submitted:
    form_answers = {
        "age_group": age_group,
        "sex": SEX_OPTIONS[sex_label],
        "race": RACE_OPTIONS[race_label],
        "education": EDUCATION_OPTIONS[education_label],
        "marital_status": MARITAL_STATUS_OPTIONS[marital_status_label],
        "employment_status": EMPLOYMENT_STATUS_OPTIONS[employment_status_label],
        "income_poverty": INCOME_POVERTY_OPTIONS[income_poverty_label],
        "rent_or_own": RENT_OR_OWN_OPTIONS[rent_or_own_label],
        "household_adults": HOUSEHOLD_COUNT_OPTIONS[household_adults_label],
        "household_children": HOUSEHOLD_COUNT_OPTIONS[household_children_label],
        "chronic_med_condition": YES_NO_UNKNOWN[chronic_med_condition_label],
        "child_under_6_months": YES_NO_UNKNOWN[child_under_6_months_label],
        "health_worker": YES_NO_UNKNOWN[health_worker_label],
        "health_insurance": YES_NO_UNKNOWN[health_insurance_label],
        "doctor_recc_h1n1": YES_NO_UNKNOWN[doctor_recc_h1n1_label],
        "doctor_recc_seasonal": YES_NO_UNKNOWN[doctor_recc_seasonal_label],
        "h1n1_concern": H1N1_CONCERN_OPTIONS[h1n1_concern_label],
        "h1n1_knowledge": H1N1_KNOWLEDGE_OPTIONS[h1n1_knowledge_label],
        "opinion_h1n1_risk": OPINION_SCALE_RISK[opinion_h1n1_risk_label],
        "opinion_h1n1_vacc_effective": OPINION_SCALE_EFFECTIVE[opinion_h1n1_vacc_effective_label],
        "opinion_h1n1_sick_from_vacc": OPINION_SCALE_WORRY[opinion_h1n1_sick_from_vacc_label],
        "opinion_seas_risk": OPINION_SCALE_RISK[opinion_seas_risk_label],
        "opinion_seas_vacc_effective": OPINION_SCALE_EFFECTIVE[opinion_seas_vacc_effective_label],
        "opinion_seas_sick_from_vacc": OPINION_SCALE_WORRY[opinion_seas_sick_from_vacc_label],
        "behavioral_antiviral_meds": YES_NO_UNKNOWN[behavioral_antiviral_meds_label],
        "behavioral_avoidance": YES_NO_UNKNOWN[behavioral_avoidance_label],
        "behavioral_face_mask": YES_NO_UNKNOWN[behavioral_face_mask_label],
        "behavioral_wash_hands": YES_NO_UNKNOWN[behavioral_wash_hands_label],
        "behavioral_large_gatherings": YES_NO_UNKNOWN[behavioral_large_gatherings_label],
        "behavioral_outside_home": YES_NO_UNKNOWN[behavioral_outside_home_label],
        "behavioral_touch_face": YES_NO_UNKNOWN[behavioral_touch_face_label],
    }

    df_row = inference.build_input_row(form_answers)
    df_processed = inference.preprocess_new_data(df_row, artifacts, random_state=42)

    full_features = config["full_features"]
    nominal_categories = config["nominal_categories"]
    X = encoding.build_gbm_matrix(df_processed, full_features, nominal_categories=nominal_categories)

    h1n1_proba = artifacts["final_models"]["h1n1_vaccine"].predict_proba(X)[0, 1]
    seasonal_proba = artifacts["final_models"]["seasonal_vaccine"].predict_proba(X)[0, 1]

    st.subheader("Resultado")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilidad — vacuna H1N1", f"{h1n1_proba:.1%}")
        st.progress(min(int(h1n1_proba * 100), 100))
    with col2:
        st.metric("Probabilidad — vacuna estacional", f"{seasonal_proba:.1%}")
        st.progress(min(int(seasonal_proba * 100), 100))

    st.caption(
        "Estas probabilidades son una estimación con fines demostrativos, "
        "basada en patrones de la encuesta de 2009, y no constituyen "
        "asesoramiento médico."
    )

    st.subheader("¿Qué influyó en esta predicción?")
    st.write(
        "Estas son las respuestas que más influyeron en tu resultado, ordenadas "
        "de mayor a menor peso."
    )

    df_processed_row = df_processed.iloc[0].to_dict()
    tab_h1n1, tab_seasonal = st.tabs(["H1N1", "Estacional"])

    with tab_h1n1:
        explanation_h1n1 = get_individual_explanation("h1n1_vaccine", X)
        positives, negatives = build_plain_language_summary(explanation_h1n1, df_processed_row)

        if positives:
            st.markdown("**⬆️ Esto aumentó tu probabilidad de vacunación contra el H1N1:**")
            st.markdown("\n".join(positives))
        if negatives:
            st.markdown("**⬇️ Esto redujo tu probabilidad de vacunación contra el H1N1:**")
            st.markdown("\n".join(negatives))

        with st.expander("Ver detalle técnico (gráfico)"):
            friendly_exp_h1n1 = build_friendly_chart_explanation(explanation_h1n1, df_processed_row)
            fig = plt.figure()
            shap.plots.waterfall(friendly_exp_h1n1, show=False, max_display=6)
            st.pyplot(fig)
            plt.close(fig)

    with tab_seasonal:
        explanation_seasonal = get_individual_explanation("seasonal_vaccine", X)
        positives, negatives = build_plain_language_summary(explanation_seasonal, df_processed_row)

        if positives:
            st.markdown("**⬆️ Esto aumentó tu probabilidad de vacunación estacional:**")
            st.markdown("\n".join(positives))
        if negatives:
            st.markdown("**⬇️ Esto redujo tu probabilidad de vacunación estacional:**")
            st.markdown("\n".join(negatives))

        with st.expander("Ver detalle técnico (gráfico)"):
            friendly_exp_seasonal = build_friendly_chart_explanation(explanation_seasonal, df_processed_row)
            fig = plt.figure()
            shap.plots.waterfall(friendly_exp_seasonal, show=False, max_display=6)
            st.pyplot(fig)
            plt.close(fig)