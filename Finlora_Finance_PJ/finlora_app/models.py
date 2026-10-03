"""Model layer: trains the notebook's model family, persists a light bundle, and provides explainability helpers."""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CATEGORICAL_FEATS, NUMERIC_FEATS, TARGET, Datasets, artifact_dir

RS = 42
PROD = "Logistic Regression"
TUNED_RF_PARAMS = {"max_depth": 14, "min_samples_leaf": 5}  # RandomizedSearchCV result recorded in Model_2.ipynb
THRESH_GRID = np.linspace(0.05, 0.95, 181)

MODEL_INFO = {
    "Logistic Regression": ("Linear baseline, class-balanced weights, standardised inputs. Fast, transparent, fully explainable.", "class_weight='balanced'"),
    "Random Forest": ("300 trees, depth 10. Captures non-linear interactions; used for the importance narrative.", "class_weight='balanced_subsample'"),
    "Random Forest (tuned)": ("Random Forest with the depth / leaf size found by RandomizedSearchCV (scored on average precision).", "balanced_subsample + tuning"),
    "Gradient Boosting": ("HistGradientBoosting, 300 iterations, depth 8, learning rate 0.05.", "class_weight='balanced'"),
    "Random Forest + SMOTE": ("Synthetic minority oversampling to a 0.3 fraud:legit ratio, then a Random Forest.", "SMOTE (0.3)"),
    "Ensemble (LR + tuned RF)": ("Soft vote: average of the Logistic Regression and tuned Random Forest probabilities.", "probability averaging"),
    "Random Forest (calibrated)": ("Tuned Random Forest wrapped in isotonic probability calibration (3-fold).", "isotonic calibration"),
}


def _pre() -> ColumnTransformer:
    return ColumnTransformer([("num", StandardScaler(), NUMERIC_FEATS),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATS)])


def make_lr(C: float = 1.0, balanced: bool = True) -> Pipeline:
    return Pipeline([("p", _pre()), ("c", LogisticRegression(C=C, class_weight="balanced" if balanced else None, max_iter=2000, random_state=RS))])


def make_rf(n=300, depth=10, leaf=1) -> Pipeline:
    return Pipeline([("p", _pre()), ("c", RandomForestClassifier(n_estimators=n, max_depth=depth, min_samples_leaf=leaf,
                                                                class_weight="balanced_subsample", random_state=RS, n_jobs=-1))])


def make_hgb(iters=300, depth=8, lr=0.05) -> Pipeline:
    return Pipeline([("p", _pre()), ("c", HistGradientBoostingClassifier(max_iter=iters, max_depth=depth, learning_rate=lr,
                                                                        class_weight="balanced", random_state=RS))])


# ----------------------------------------------------------------------------- metrics
def best_f1_threshold(y, p) -> tuple[float, float]:
    f = np.array([f1_score(y, p >= t, zero_division=0) for t in THRESH_GRID])
    i = int(np.argmax(f))
    return float(THRESH_GRID[i]), float(f[i])


def metrics_at(y, p, t: float) -> dict:
    y = np.asarray(y)
    pred = np.asarray(p) >= t
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(t), "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "accuracy": float((pred == y).mean()), "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn), "alerts": int(pred.sum())}


def precision_at_recall(y, p, targets=(0.7, 0.8, 0.9)) -> pd.DataFrame:
    prec, rec, thr = precision_recall_curve(y, p)
    rows = []
    for t in targets:
        idx = np.where(rec[:-1] >= t)[0]
        if len(idx):
            i = idx[-1]
            rows.append({"Recall target": f"{int(t * 100)}%", "Score threshold": round(float(thr[i]), 4),
                         "Precision": round(float(prec[i]), 4), "Recall achieved": round(float(rec[i]), 4)})
    return pd.DataFrame(rows)


def sweep(y, p) -> pd.DataFrame:
    ts = np.linspace(0.02, 0.99, 98)
    rows = [metrics_at(y, p, t) for t in ts]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- training
def _split(ds: Datasets):
    return train_test_split(ds.X, ds.y, test_size=0.25, stratify=ds.y, random_state=RS)


def train_all(ds: Datasets, progress=None) -> dict:
    Xtr, Xte, ytr, yte = _split(ds)
    probas, meta, fitted = {}, {}, {}
    steps = ["Logistic Regression", "Random Forest", "Random Forest (tuned)", "Gradient Boosting", "Random Forest + SMOTE",
             "Random Forest (calibrated)", "Honest threshold check (5-fold)"]

    def tick(i, label):
        if progress:
            progress(i / len(steps), f"Training {label} ({i + 1}/{len(steps)})")

    def fit(name, model, i):
        tick(i, name)
        t0 = time.time()
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        probas[name] = p
        meta[name] = {"seconds": round(time.time() - t0, 1), "auc": float(roc_auc_score(yte, p)), "ap": float(average_precision_score(yte, p))}
        fitted[name] = model

    fit("Logistic Regression", make_lr(), 0)
    fit("Random Forest", make_rf(), 1)
    fit("Random Forest (tuned)", make_rf(depth=TUNED_RF_PARAMS["max_depth"], leaf=TUNED_RF_PARAMS["min_samples_leaf"]), 2)
    fit("Gradient Boosting", make_hgb(), 3)
    smote = ImbPipeline([("p", _pre()), ("s", SMOTE(sampling_strategy=0.3, random_state=RS)),
                         ("c", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RS, n_jobs=-1))])
    fit("Random Forest + SMOTE", smote, 4)
    cal = Pipeline([("p", _pre()), ("c", CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=RS, n_jobs=-1, **TUNED_RF_PARAMS),
        method="isotonic", cv=3))])
    fit("Random Forest (calibrated)", cal, 5)

    ens = (probas["Logistic Regression"] + probas["Random Forest (tuned)"]) / 2
    probas["Ensemble (LR + tuned RF)"] = ens
    meta["Ensemble (LR + tuned RF)"] = {"seconds": 0.0, "auc": float(roc_auc_score(yte, ens)), "ap": float(average_precision_score(yte, ens))}

    # Honest threshold check: choose the threshold WITHOUT looking at the test set (out-of-fold on train), then apply to test.
    tick(6, steps[6])
    oof = cross_val_predict(clone(make_lr()), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=RS), method="predict_proba")[:, 1]
    oof_t, oof_f1 = best_f1_threshold(ytr, oof)
    honest = metrics_at(yte, probas["Logistic Regression"], oof_t)
    honest["oof_f1"] = oof_f1

    rf_pre = fitted["Random Forest"].named_steps["p"]
    names = [n.split("__", 1)[1] for n in rf_pre.get_feature_names_out()]
    rf_imp = pd.Series(fitted["Random Forest"].named_steps["c"].feature_importances_, index=names)

    results = {}
    for n, p in probas.items():
        t_best, f_best = best_f1_threshold(yte, p)
        results[n] = {"default": metrics_at(yte, p, 0.5), "tuned": metrics_at(yte, p, t_best), "auc": meta[n]["auc"], "ap": meta[n]["ap"]}

    bundle = {"signature": ds.signature, "lr": fitted["Logistic Regression"], "probas": probas, "meta": meta, "results": results,
              "y_test": yte, "test_index": yte.index.to_numpy(), "train_size": len(Xtr), "test_size": len(Xte),
              "train_fraud": float(ytr.mean()), "test_fraud": float(yte.mean()), "rf_importance": rf_imp,
              "prod_threshold": results[PROD]["tuned"]["threshold"], "honest": honest, "feature_names": names}
    if progress:
        progress(1.0, "Done")
    return bundle


@st.cache_resource(show_spinner=False)
def _read_bundle(path: str, mtime: float):
    return joblib.load(path)


def bundle_path():
    return artifact_dir() / "finlora_bundle.joblib"


def bundle_current(ds: Datasets) -> bool:
    p = bundle_path()
    if not p.exists():
        return False
    try:
        return _read_bundle(str(p), p.stat().st_mtime)["signature"] == ds.signature
    except Exception:
        return False


def get_bundle(ds: Datasets, progress=None, force: bool = False) -> dict:
    p = bundle_path()
    if not force and bundle_current(ds):
        return _read_bundle(str(p), p.stat().st_mtime)
    b = train_all(ds, progress)
    joblib.dump(b, p, compress=3)
    _read_bundle.clear()
    return _read_bundle(str(p), p.stat().st_mtime)


# ----------------------------------------------------------------------------- explainability
def _group_of(feature_name: str) -> str:
    if feature_name.startswith("num__"):
        return feature_name[5:]
    f = feature_name[5:]
    for c in CATEGORICAL_FEATS:
        if f.startswith(c + "_"):
            return c
    return f


def score_and_explain(bundle: dict, X: pd.DataFrame):
    """Return (probabilities, per-feature-group logit contributions, intercept) for the production Logistic Regression."""
    lr = bundle["lr"]
    pre, clf = lr.named_steps["p"], lr.named_steps["c"]
    M = pre.transform(X)
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    contrib = M * clf.coef_[0]
    groups = [_group_of(n) for n in pre.get_feature_names_out()]
    cdf = pd.DataFrame(contrib, index=X.index, columns=groups).T.groupby(level=0).sum().T
    logit = contrib.sum(1) + clf.intercept_[0]
    return 1 / (1 + np.exp(-logit)), cdf, float(clf.intercept_[0])


def reason_text(feature: str, row: pd.Series) -> str:
    v = row[feature]
    if feature == "amount_to_avg_ratio":
        return f"Amount is {float(v):,.1f}x the account's 30-day average"
    if feature == "transaction_velocity_1h":
        return f"{int(v)} transaction(s) in the previous hour"
    if feature == "is_new_device":
        return "Unrecognised device" if int(v) == 1 else "Known / no device"
    if feature == "is_cross_border":
        return "Cross-border transaction" if int(v) == 1 else "Domestic transaction"
    if feature == "account_age_days":
        return f"Account is {int(v):,} days old"
    if feature == "hour_of_day":
        return f"Transaction at {int(v):02d}:00"
    if feature == "is_refund":
        return "Refund-type transaction" if int(v) == 1 else "Not a refund"
    labels = {"merchant_category": "Merchant category", "channel": "Channel", "kyc_tier": "KYC level"}
    if feature in labels:
        return f"{labels[feature]}: {v}"
    if feature == "account_type":
        return f"{v} account"
    return feature


BINARY_FEATURES = {"is_new_device", "is_cross_border", "is_refund"}


def top_reasons(contrib_row: pd.Series, raw_row: pd.Series, k: int = 3) -> str:
    """Plain-language list of the factors that pushed this transaction's risk up."""
    pos = contrib_row[contrib_row > 0.1].sort_values(ascending=False)
    pos = [f for f in pos.index if not (f in BINARY_FEATURES and int(raw_row[f]) == 0)][:k]
    return " | ".join(reason_text(f, raw_row) for f in pos) or "No strong risk drivers"


def tiers(score, thr: float) -> pd.Series:
    """Priority tiers: Critical (>=0.99), High (>= production threshold), Watch (>= 0.5), Low."""
    s = np.asarray(score)
    crit = max(thr, 0.99)
    return pd.Series(np.select([s >= crit, s >= thr, s >= 0.5], ["Critical", "High", "Watch"], "Low"))
