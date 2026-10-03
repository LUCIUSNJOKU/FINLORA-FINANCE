# Finlora Transaction Risk Review Queue — Streamlit Dashboard

Prototype analyst dashboard for the Finlora transaction risk scoring project.

## Model in production

Per the notebook's "Model Comparison & Threshold Tuning" section, this
dashboard scores transactions with a **Logistic Regression model at its
F1-optimal threshold (0.935)** — it matched or beat six other techniques
(tuned Random Forest, gradient boosting, SMOTE, calibration, ensembling) on
F1 while staying the simplest and most explainable option. A **Random
Forest** is kept alongside it purely to drive the "why flagged"
feature-importance view — it does not score transactions itself.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Files

- `app.py` — the dashboard
- `finlora_models.joblib` — the production Logistic Regression pipeline (with
  its tuned threshold), the Random Forest explainability pipeline, and the
  full model-comparison results, all bundled so the app doesn't need to
  retrain on launch
- `finlora_clean_transactions.csv` — cleaned, analysis-ready dataset used as
  the default data source

## Features

- **Model toggle** — Production (Logistic Regression, recommended) vs.
  Random Forest (explainability view)
- **Threshold control** — the model's own tuned F1 threshold, a target
  recall (70/80/90%, backtested live against known labels), or a manual cutoff
- **Filters** — account type, channel, merchant category
- **Prioritized queue** — flagged transactions ranked by risk score, each
  with a plain-language "why flagged" explanation
- **Transaction drill-down** — inspect the underlying feature values for any
  flagged transaction
- **Upload your own CSV** — score a different transaction file
- **CSV export** — download the current flagged queue

## Notes

This is a prototype per the project's scope: it is not connected to
Finlora's live payment systems, does not perform automated blocking, and has
no analyst feedback/retraining loop. Risk scores are decision-support
signals, not automated fraud determinations.
