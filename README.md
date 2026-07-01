# Flu Shot Learning — Predicting H1N1 and Seasonal Flu Vaccination

End-to-end machine learning project for the [DrivenData "Flu Shot
Learning"](https://www.drivendata.org/competitions/66/flu-shot-learning/)
competition, predicting the probability that a respondent of the 2009
National H1N1 Flu Survey received the H1N1 vaccine and/or the seasonal flu
vaccine.

**🔗 Live interactive demo:** [Streamlit app](#) *(https://diego-flu-vaccine-prediction.streamlit.app/)*
**📓 Full analysis notebook:** [`EDA.ipynb`](./EDA.ipynb)

---

## Results

| Metric | Value |
|---|---|
| Official competition AUC | **0.8599** |
| Leaderboard rank | 650 / 2381 (top ~27%) |
| Best AUC in the competition (reference) | 0.8658 |
| Project's minimum target | 0.855 ✅ |
| Public benchmark (simple logistic regression) | ~0.83–0.84 |

Two independent LightGBM models (one per target), tuned with Optuna,
trained on data imputed via a mechanism-informed missing-value strategy
designed specifically for this dataset.

## Project objectives

1. Beat the official benchmark (target: AUC > 0.855).
2. Build a fully documented end-to-end pipeline: EDA → missing-value
   handling → feature engineering → model selection → evaluation →
   deployment, with every decision explained.
3. Rigorously compare at least two missing-value imputation strategies and
   measure their real impact on AUC.
4. Compare at least three model families (Logistic Regression, Random
   Forest, XGBoost/LightGBM) with proper cross-validation.
5. Apply SHAP for interpretability, with clinical context behind the
   findings.
6. Deploy an interactive prediction app in production.

See the introductory markdown cells at the top of [`EDA.ipynb`](./EDA.ipynb)
for the full problem description and objectives.

## Methodology summary

- **EDA & missingness analysis**: identified five distinct missing-data
  mechanisms across the dataset's variables.
- **Imputation**: two strategies compared — a naive hybrid baseline
  (Strategy A) and a mechanism-informed approach (Strategy B) using
  `IterativeImputer`, an explicit "Unknown" category for doctor-
  recommendation variables, and a dedicated classifier for
  `employment_status`. Strategy B selected based on correlation-structure
  recovery, mutual information against the targets, and a real,
  non-noise AUC improvement in cross-validation.
- **Feature engineering**: decisions revisited and reversed where mutual
  information evidence contradicted earlier choices (e.g. reinstating
  `employment_industry`/`employment_occupation`, keeping both original
  vaccine-side-effect opinion variables instead of merging them).
- **Modeling**: Logistic Regression, Random Forest, XGBoost, and LightGBM
  compared via 5-fold cross-validation; Random Forest and LightGBM tuned
  with Optuna (50 trials each); LightGBM selected as the final model for
  both targets.
- **Interpretability**: SHAP `TreeExplainer` analysis surfaced clinically
  interpretable findings — an age/immunity interaction specific to the
  2009 pandemic context, a threshold effect in vaccine-effectiveness
  beliefs, and a differential meaning of missingness between
  `health_insurance` and `doctor_recc_*`.
- **Deployment**: the full pipeline (imputation, feature engineering,
  encoding) was modularized into a `src/` package, verified for
  consistency against the original notebook results, and reused directly
  by an interactive Streamlit app.

Full decision-by-decision documentation, including all intermediate
results and the reasoning behind every reversal, is written directly into
the markdown cells of [`EDA.ipynb`](./EDA.ipynb), interleaved with the
corresponding code and outputs.

## Repository structure

```
.
├── EDA.ipynb              # Full analysis notebook (EDA → SHAP), with outputs
│                             and markdown documentation for every decision
├── src/                   # Production pipeline (imputation, feature
│                             engineering, encoding, inference), reused by
│                             both the notebook and the Streamlit app
├── app/
│   ├── streamlit_app.py   # Interactive prediction app
│   └── requirements.txt   # App-specific dependencies (for Streamlit Cloud)
├── artifacts/              # Fitted imputers, encoders, and final models (joblib)
├── requirements.txt        # Full notebook environment dependencies
└── data/                   # Not included — see "Data" section below
```

## Data

This project uses the official DrivenData competition dataset, which is
**not included in this repository** per the competition's terms of use.
To reproduce the notebook, download `training_set_features.csv`,
`training_set_labels.csv`, and `test_set_features.csv` from the
[competition's data page](https://www.drivendata.org/competitions/66/flu-shot-learning/data/)
and place them in a local `data/` folder.

## Running locally

**Notebook:**
```bash
pip install -r requirements.txt
jupyter notebook EDA.ipynb
```

**Streamlit app:**
```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

The app loads pre-fitted artifacts from `artifacts/`, so it runs without
needing the raw competition data or re-running the notebook.

## Tech stack

Python, pandas, scikit-learn, LightGBM, XGBoost, Optuna, SHAP, Streamlit.

## License

MIT — see [LICENSE](./LICENSE).
