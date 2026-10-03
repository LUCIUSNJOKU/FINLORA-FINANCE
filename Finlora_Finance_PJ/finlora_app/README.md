# Finlora Fraud Detection Workbench

A Streamlit app that walks stakeholders through the whole project: load, clean, explore, engineer features,
train and evaluate models, tune the alert threshold, and work an explainable review queue.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Put `finlora_transactions.csv` and `finlora_accounts.csv` in `data/` (or set `FINLORA_DATA_DIR`, or use the in-app setup screen).
Optionally add `cleaned_finlora_data.csv` and `featured_engineered_dataset.csv`: the app then verifies that its live
pipeline reproduces them exactly.

## How it works
- Cleaning and feature engineering are re-run live from the raw files (logic mirrors the notebooks).
- Seven models are trained on first launch (about 2-4 minutes) and cached in `artifacts/`; delete that folder to retrain.
- Production model: Logistic Regression at its F1-optimal threshold (0.935). Tuned RF parameters are those recorded in Model_2.ipynb.

## Files
`app.py` navigation | `data.py` loading and cleaning | `models.py` training and explainability | `theme.py` styling |
`views_a-d.py` pages | `ctx.py` shared setup
