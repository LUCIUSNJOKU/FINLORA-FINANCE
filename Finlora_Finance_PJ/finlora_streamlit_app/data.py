"""Data layer: locate files, load raw data, re-run the cleaning pipeline, rebuild engineered features."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
NUMERIC_FEATS = ["amount_to_avg_ratio", "transaction_velocity_1h", "is_new_device", "is_cross_border",
                 "account_age_days", "hour_of_day", "is_refund"]
CATEGORICAL_FEATS = ["account_type", "kyc_tier", "merchant_category", "channel"]
TARGET = "is_fraud"
RAW_FILES = {"transactions": "finlora_transactions.csv", "accounts": "finlora_accounts.csv"}
OPTIONAL_FILES = {"cleaned": "cleaned_finlora_data.csv", "featured": "featured_engineered_dataset.csv"}
CAT_COLS_TO_STRIP = ["account_type", "kyc_tier", "merchant_category", "channel", "status", "currency", "day_of_week"]


# ----------------------------------------------------------------------------- locating files
def candidate_dirs() -> list[Path]:
    out: list[Path] = []
    ov = st.session_state.get("data_dir_override")
    if ov:
        out.append(Path(ov).expanduser())
    if os.environ.get("FINLORA_DATA_DIR"):
        out.append(Path(os.environ["FINLORA_DATA_DIR"]).expanduser())
    out += [APP_DIR / "data", APP_DIR, Path.cwd() / "data", Path.cwd()]
    return out


def find_data_dir() -> Path | None:
    for d in candidate_dirs():
        if all((d / f).exists() for f in RAW_FILES.values()):
            return d
    return None


def artifact_dir() -> Path:
    for base in (APP_DIR, Path(os.environ.get("TMPDIR", "/tmp"))):
        try:
            p = base / "artifacts"
            p.mkdir(exist_ok=True)
            return p
        except OSError:
            continue
    return Path(".")


# ----------------------------------------------------------------------------- dataclass
@dataclass
class Datasets:
    data_dir: Path
    tx_raw: pd.DataFrame
    acc_raw: pd.DataFrame
    cleaned: pd.DataFrame
    log: list[dict]
    missing_before: pd.Series
    missing_after: pd.Series
    ratio_examples: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    dummies: pd.DataFrame
    clean_check: dict | None = None
    feat_check: dict | None = None
    signature: str = ""
    notes: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- cleaning (mirrors Data_Cleaning.ipynb)
def run_cleaning(tx_raw: pd.DataFrame, acc_raw: pd.DataFrame):
    tx, acc = tx_raw.copy(), acc_raw.copy()
    log: list[dict] = []

    def add(step, issue, n, treatment, why):
        log.append({"Step": step, "Issue found": issue, "Rows affected": int(n), "Treatment": treatment, "Why it matters": why})

    # 1 duplicates
    dup_tx, dup_acc = int(tx["transaction_id"].duplicated().sum()), int(acc["account_id"].duplicated().sum())
    tx = tx.drop_duplicates(subset="transaction_id")
    acc = acc.drop_duplicates(subset="account_id")
    add("Duplicates", "Repeated transaction_id / account_id", dup_tx + dup_acc,
        "Dropped on the unique identifiers (none were found)", "Duplicates would double-count fraud and leak across the train/test split")

    # 2 missing merchant
    n = tx["merchant_name"].isnull().sum()
    tx["merchant_name"] = tx["merchant_name"].fillna("N/A (transfer)")
    add("Missing merchant", "merchant_name empty on Payroll / P2P / Wire style transfers", n,
        "Filled with 'N/A (transfer)'", "Transfers legitimately have no merchant - the gap is structural, not an error")

    # 3 missing device
    n = tx["device_id"].isnull().sum()
    tx["device_id"] = tx["device_id"].fillna("UNKNOWN")
    tx["is_new_device"] = tx["is_new_device"].fillna(0).astype(int)
    add("Missing device", "device_id and is_new_device empty (channels with no device fingerprint)", n,
        "device_id = 'UNKNOWN', is_new_device = 0", "Keeps rows in the model without inventing a 'new device' signal")

    # 4 refunds
    n = (tx["amount"] < 0).sum()
    tx["is_refund"] = (tx["amount"] < 0).astype(int)
    add("Negative amounts", "amount < 0 (all Retail - consistent with refunds)", n,
        "Kept, flagged with new is_refund feature", "Negative values are genuine refunds, so they are information rather than noise")

    # 5 invalid baseline
    bad = tx["avg_transaction_amount_30d"] <= 0
    n_bad = int(bad.sum())
    acct_med = tx.loc[~bad].groupby("account_id")["avg_transaction_amount_30d"].median()
    glob_med = tx.loc[~bad, "avg_transaction_amount_30d"].median()
    tx.loc[bad, "avg_transaction_amount_30d"] = tx.loc[bad, "account_id"].map(acct_med).fillna(glob_med)
    add("Invalid 30-day baseline", "avg_transaction_amount_30d <= 0", n_bad,
        "Replaced with the account's median positive baseline (global median as fallback)",
        "A zero/negative baseline makes every ratio infinite or meaningless")

    # 6 ratio recompute (measure mismatch on raw first, like the notebook)
    chk = tx_raw["amount"].abs() / tx_raw["avg_transaction_amount_30d"]
    mismatch = (chk - tx_raw["amount_to_avg_ratio"]).abs() > 0.5
    examples = tx_raw.loc[mismatch, ["transaction_id", "amount", "avg_transaction_amount_30d", "amount_to_avg_ratio"]].head(200).copy()
    examples["recomputed_ratio"] = chk[mismatch].head(200).values
    tx["amount_to_avg_ratio"] = (tx["amount"].abs() / tx["avg_transaction_amount_30d"]).round(4)
    add("Inconsistent ratio", "amount_to_avg_ratio disagrees with |amount| / avg_30d by > 0.5", mismatch.sum(),
        "Recomputed as |amount| / avg_transaction_amount_30d (4 d.p.)",
        "This ratio is the single strongest fraud signal, so it must be exactly right")

    # 7 whitespace
    changed = 0
    for c in CAT_COLS_TO_STRIP:
        s = tx[c].astype(str)
        changed += int((s != s.str.strip()).sum())
        tx[c] = tx[c].astype(str).str.strip()
    add("Category hygiene", "Leading / trailing whitespace in categorical labels", changed,
        "Stripped whitespace on 7 categorical columns", "Prevents 'Retail ' vs 'Retail' splitting into two categories on encoding")

    # 8 timestamp
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    add("Timestamps", "timestamp stored as text", len(tx), "Parsed to datetime (all values valid)", "Enables time-of-day, weekday and trend analysis")

    # 9 merge
    acc["account_created_date"] = pd.to_datetime(acc["account_created_date"])
    out = tx.merge(acc[["account_id", "account_holder_name", "account_created_date", "personal_spend_baseline_usd"]],
                   on="account_id", how="left")
    unmatched = int(out["account_created_date"].isnull().sum())
    add("Join accounts", "Transactions without a matching account", unmatched,
        f"Left-joined account profile -> {len(out):,} rows x {out.shape[1]} columns", "Adds holder, creation date and personal spend baseline")
    return out, log, examples


def _pct_missing(df: pd.DataFrame) -> pd.Series:
    s = df.isnull().sum()
    return s[s > 0].sort_values(ascending=False)


def _validate_cleaned(derived: pd.DataFrame, saved: pd.DataFrame) -> dict:
    same_shape = derived.shape == saved.shape
    ids = same_shape and bool((derived["transaction_id"].values == saved["transaction_id"].values).all())
    return {
        "Same shape as saved file": same_shape,
        "Same transaction_id order": ids,
        "Identical fraud labels": ids and bool((derived["is_fraud"].values == saved["is_fraud"].values).all()),
        "amount_to_avg_ratio matches (tol 1e-3)": ids and bool(np.allclose(derived["amount_to_avg_ratio"], saved["amount_to_avg_ratio"], atol=1e-3)),
        "is_refund flag matches": ids and bool((derived["is_refund"].values == saved["is_refund"].values).all()),
        "is_new_device matches": ids and bool((derived["is_new_device"].values == saved["is_new_device"].values).all()),
    }


def _validate_featured(dummies: pd.DataFrame, y: pd.Series, saved: pd.DataFrame) -> dict:
    cols_ok = list(dummies.columns) == [c for c in saved.columns if c != TARGET]
    vals_ok = cols_ok and dummies.shape[0] == saved.shape[0] and bool(
        np.allclose(dummies.values.astype(float), saved.drop(columns=TARGET).values.astype(float)))
    return {"Same 33 feature columns, same order": cols_ok, "Identical feature values (all rows)": vals_ok,
            "Identical target": bool((y.values == saved[TARGET].values).all()) if dummies.shape[0] == saved.shape[0] else False}


@st.cache_resource(show_spinner=False)
def _load(dir_str: str, sig: tuple) -> Datasets:
    d = Path(dir_str)
    tx_raw = pd.read_csv(d / RAW_FILES["transactions"])
    acc_raw = pd.read_csv(d / RAW_FILES["accounts"])
    cleaned, log, examples = run_cleaning(tx_raw, acc_raw)
    X = cleaned[NUMERIC_FEATS + CATEGORICAL_FEATS].copy()
    y = cleaned[TARGET].copy()
    dummies = pd.get_dummies(X, columns=CATEGORICAL_FEATS)
    ds = Datasets(d, tx_raw, acc_raw, cleaned, log, _pct_missing(tx_raw), _pct_missing(cleaned), examples, X, y, dummies)
    pc, pf = d / OPTIONAL_FILES["cleaned"], d / OPTIONAL_FILES["featured"]
    if pc.exists():
        ds.clean_check = _validate_cleaned(cleaned, pd.read_csv(pc))
    if pf.exists():
        ds.feat_check = _validate_featured(dummies, y, pd.read_csv(pf))
    ds.signature = f"v2-{len(cleaned)}-{int(y.sum())}-{cleaned['amount_to_avg_ratio'].sum():.2f}"
    return ds


def get_datasets() -> Datasets | None:
    d = find_data_dir()
    if d is None:
        return None
    files = [d / f for f in list(RAW_FILES.values()) + list(OPTIONAL_FILES.values())]
    sig = tuple((f.name, f.stat().st_mtime_ns if f.exists() else 0) for f in files)
    return _load(str(d), sig)
