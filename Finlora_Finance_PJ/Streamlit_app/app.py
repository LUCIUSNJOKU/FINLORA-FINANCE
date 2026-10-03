"""
Finlora — Transaction Risk Scoring Dashboard
==============================================
Prioritized, explainable fraud review queue.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Expects the following files in the same folder:
    - finlora_models.joblib          (production + explainability pipelines, scaler, feature schema)
    - finlora_clean_transactions.csv (cleaned, analysis-ready transaction data)
Both are provided alongside this script. To score a different transaction file,
use the "Upload transactions" option in the sidebar — it must contain the same
columns as finlora_clean_transactions.csv (or at minimum the feature columns
listed in FEATURES below).

Model selection: the notebook benchmarked seven techniques (see its
"Model Comparison & Threshold Tuning" section) and selected a Logistic
Regression, at its F1-optimal threshold, as the production scoring model —
it matched or beat every more complex alternative on F1 while staying the
simplest, fastest, most explainable option. A Random Forest is kept
alongside it purely to drive the "why flagged" feature-importance view
below, since tree-based importances read more naturally than logistic
coefficients — it never scores transactions itself.
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Finlora Risk Review Queue", page_icon="🛡️", layout="wide")

# ----------------------------------------------------------------------------
# Load model bundle + data
# ----------------------------------------------------------------------------
@st.cache_resource
MODEL_PATH = Path(__file__).resolve().parent / "finlora_models.joblib"

def load_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data(path="C:\Users\User\Documents\AMDARI_IMS\Finlora_Finance_PJ\Streamlit_app\finlora_clean_transactions.csv"):
    df = pd.read_csv(path)
    df["is_new_device"] = df["is_new_device"].fillna(0).astype(int)
    df["is_cross_border"] = df["is_cross_border"].astype(int)
    return df

bundle = load_bundle()
NUM_FEATS = bundle["num_feats"]
CAT_FEATS = bundle["cat_feats"]

# New-style bundle (production/explainability pipelines). Fall back to the
# older rf_model/lr_model + manual dummy-encoding bundle if this dashboard is
# ever pointed at a joblib file saved before the comparison step was added.
HAS_PIPELINE_BUNDLE = "production_model" in bundle

if HAS_PIPELINE_BUNDLE:
    PRODUCTION_MODEL = bundle["production_model"]
    PRODUCTION_MODEL_NAME = bundle["production_model_name"]
    PRODUCTION_THRESHOLD = bundle["production_threshold"]
    EXPLAINABILITY_MODEL = bundle["explainability_model"]
else:
    RF = bundle["rf_model"]
    LR = bundle["lr_model"]
    SCALER = bundle["scaler"]
    FEATURE_COLUMNS = bundle["feature_columns"]
    PRODUCTION_THRESHOLD = 0.5

df_default = load_data()

# ----------------------------------------------------------------------------
# Sidebar — data source, model choice, threshold
# ----------------------------------------------------------------------------
st.sidebar.title("🛡️ Finlora Risk Scoring")
st.sidebar.caption("Transaction monitoring — prioritized review queue")

uploaded = st.sidebar.file_uploader("Upload transactions CSV (optional)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    df["is_new_device"] = df["is_new_device"].fillna(0).astype(int)
    df["is_cross_border"] = df["is_cross_border"].astype(int)
    st.sidebar.success(f"Loaded {len(df):,} uploaded transactions")
else:
    df = df_default
    st.sidebar.caption(f"Using bundled dataset ({len(df):,} transactions)")

if HAS_PIPELINE_BUNDLE:
    model_choice = st.sidebar.radio(
        "Model",
        [f"Production — {PRODUCTION_MODEL_NAME.split('. ', 1)[-1]} (recommended)", "Random Forest (explainability view)"],
    )
    using_production = model_choice.startswith("Production")
else:
    model_choice = st.sidebar.radio("Model", ["Random Forest (primary)", "Logistic Regression (baseline)"])
    using_production = model_choice.startswith("Logistic")  # legacy bundle has no real "production" pick

st.sidebar.markdown("---")
st.sidebar.subheader("Review threshold")
threshold_mode = st.sidebar.radio("Set threshold by", ["Recall target", "Manual score", "Model's tuned F1 threshold"] if HAS_PIPELINE_BUNDLE else ["Recall target", "Manual score"])

# ----------------------------------------------------------------------------
# Score transactions
# ----------------------------------------------------------------------------
if HAS_PIPELINE_BUNDLE:
    X_raw = df[NUM_FEATS + CAT_FEATS]
    if using_production:
        scores = PRODUCTION_MODEL.predict_proba(X_raw)[:, 1]
    else:
        scores = EXPLAINABILITY_MODEL.predict_proba(X_raw)[:, 1]

    # feature importances always come from the Random Forest (explainability model),
    # regardless of which model is scoring — tree importances drive the "why flagged" text
    rf_step = EXPLAINABILITY_MODEL.named_steps["classifier"]
    rf_feature_names = EXPLAINABILITY_MODEL.named_steps["preprocess"].get_feature_names_out()
    importances = pd.Series(rf_step.feature_importances_, index=rf_feature_names)
else:
    def build_feature_matrix(data):
        Xd = pd.get_dummies(data[NUM_FEATS + CAT_FEATS], columns=CAT_FEATS)
        return Xd.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    X = build_feature_matrix(df)
    if model_choice.startswith("Random Forest"):
        scores = RF.predict_proba(X)[:, 1]
        importances = pd.Series(RF.feature_importances_, index=FEATURE_COLUMNS)
    else:
        scores = LR.predict_proba(SCALER.transform(X))[:, 1]
        importances = pd.Series(np.abs(LR.coef_[0]), index=FEATURE_COLUMNS)

df = df.copy()
df["risk_score"] = scores
TOP_FEATURES = importances.sort_values(ascending=False).head(6).index.tolist()

has_labels = "is_fraud" in df.columns

# threshold selection
if threshold_mode == "Model's tuned F1 threshold":
    threshold = float(PRODUCTION_THRESHOLD) if using_production else 0.5
    st.sidebar.caption(f"→ using the notebook's F1-optimal threshold for this model: **{threshold:.3f}**")
elif threshold_mode == "Recall target" and has_labels:
    target_recall = st.sidebar.select_slider("Target recall", options=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95], value=0.8)
    order = np.argsort(-df["risk_score"].values)
    sorted_labels = df["is_fraud"].values[order]
    sorted_scores = df["risk_score"].values[order]
    total_fraud = sorted_labels.sum()
    cum_fraud = np.cumsum(sorted_labels)
    recall_at_k = cum_fraud / max(total_fraud, 1)
    idx = np.searchsorted(recall_at_k, target_recall)
    idx = min(idx, len(sorted_scores) - 1)
    threshold = float(sorted_scores[idx])
    st.sidebar.caption(f"→ score threshold **{threshold:.3f}** achieves ~{target_recall:.0%} recall on this labeled set")
else:
    default_manual = float(PRODUCTION_THRESHOLD) if (HAS_PIPELINE_BUNDLE and using_production) else 0.5
    threshold = st.sidebar.slider("Minimum risk score to flag", 0.0, 1.0, default_manual, 0.01)

df["flagged"] = df["risk_score"] >= threshold

# ----------------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
acct_filter = st.sidebar.multiselect("Account type", sorted(df["account_type"].unique()))
channel_filter = st.sidebar.multiselect("Channel", sorted(df["channel"].unique()))
merchant_filter = st.sidebar.multiselect("Merchant category", sorted(df["merchant_category"].unique()))

view = df[df["flagged"]].copy()
if acct_filter:
    view = view[view["account_type"].isin(acct_filter)]
if channel_filter:
    view = view[view["channel"].isin(channel_filter)]
if merchant_filter:
    view = view[view["merchant_category"].isin(merchant_filter)]
view = view.sort_values("risk_score", ascending=False)

# ----------------------------------------------------------------------------
# Header KPIs
# ----------------------------------------------------------------------------
st.title("Transaction Risk Review Queue")
st.caption(f"Model: **{model_choice}** &nbsp;|&nbsp; Threshold: **{threshold:.3f}**")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transactions scored", f"{len(df):,}")
c2.metric("Flagged for review", f"{df['flagged'].sum():,}", f"{df['flagged'].mean():.1%} of volume")

if has_labels:
    tp = ((df["flagged"]) & (df["is_fraud"] == 1)).sum()
    fp = ((df["flagged"]) & (df["is_fraud"] == 0)).sum()
    fn = ((~df["flagged"]) & (df["is_fraud"] == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    c3.metric("Precision @ threshold", f"{precision:.1%}")
    c4.metric("Recall @ threshold", f"{recall:.1%}")
    c5.metric("True fraud caught", f"{tp:,} / {tp+fn:,}")
else:
    c3.metric("Avg risk score (flagged)", f"{view['risk_score'].mean():.2f}" if len(view) else "—")
    c4.metric("Max risk score", f"{df['risk_score'].max():.2f}")
    c5.metric("Labels available", "No — upload includes is_fraud to backtest")

st.markdown("---")

# ----------------------------------------------------------------------------
# Score distribution chart
# ----------------------------------------------------------------------------
left, right = st.columns([2, 1])
with left:
    st.subheader("Risk score distribution")
    hist_df = df[["risk_score"]].copy()
    if has_labels:
        hist_df["label"] = df["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
        chart = alt.Chart(hist_df).mark_bar(opacity=0.7).encode(
            x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=40), title="Risk score"),
            y=alt.Y("count()", title="Transactions", stack=None),
            color=alt.Color("label:N", scale=alt.Scale(domain=["Legitimate", "Fraud"], range=["#4C72B0", "#C44E52"])),
        ).properties(height=300)
    else:
        chart = alt.Chart(hist_df).mark_bar(color="#C44E52").encode(
            x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=40), title="Risk score"),
            y=alt.Y("count()", title="Transactions"),
        ).properties(height=300)
    rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(color="black", strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(chart + rule, use_container_width=True)

with right:
    st.subheader("Flagged by segment")
    if len(view):
        seg = view["merchant_category"].value_counts().reset_index()
        seg.columns = ["merchant_category", "count"]
        seg_chart = alt.Chart(seg.head(8)).mark_bar(color="#C44E52").encode(
            x=alt.X("count:Q", title="Flagged"),
            y=alt.Y("merchant_category:N", sort="-x", title=""),
        ).properties(height=300)
        st.altair_chart(seg_chart, use_container_width=True)
    else:
        st.caption("No flagged transactions at this threshold/filter combination.")

st.markdown("---")

# ----------------------------------------------------------------------------
# Explainability helper — human-readable "why flagged"
# ----------------------------------------------------------------------------
def explain_row(row):
    reasons = []
    if row.get("amount_to_avg_ratio", 0) >= 5:
        reasons.append(f"amount is {row['amount_to_avg_ratio']:.1f}x the account's normal spend")
    if row.get("transaction_velocity_1h", 0) >= 2:
        reasons.append(f"{int(row['transaction_velocity_1h'])} transactions in the trailing hour")
    if row.get("is_new_device", 0) == 1:
        reasons.append("new/unrecognized device")
    if row.get("is_cross_border", 0) == 1:
        reasons.append("cross-border transaction")
    if row.get("merchant_category", "") in ("Wire Transfer", "Payroll Transfer", "Crypto Exchange"):
        reasons.append(f"high-risk category ({row['merchant_category']})")
    if not reasons:
        reasons.append("elevated combination of lower-magnitude signals")
    return "; ".join(reasons)

view["why_flagged"] = view.apply(explain_row, axis=1)

# ----------------------------------------------------------------------------
# Review queue table
# ----------------------------------------------------------------------------
st.subheader(f"Prioritized review queue — {len(view):,} flagged transactions")

display_cols = ["transaction_id", "account_id", "account_type", "amount", "currency",
                 "merchant_category", "channel", "risk_score", "why_flagged"]
if has_labels:
    display_cols.insert(-1, "is_fraud")

st.dataframe(
    view[display_cols].head(500).style.format({"risk_score": "{:.3f}", "amount": "{:,.2f}"}),
    use_container_width=True,
    height=420,
)
if len(view) > 500:
    st.caption(f"Showing top 500 of {len(view):,} flagged transactions, ranked by risk score.")

st.download_button(
    "Download full flagged queue (CSV)",
    view[display_cols].to_csv(index=False).encode("utf-8"),
    file_name="finlora_flagged_queue.csv",
    mime="text/csv",
)

# ----------------------------------------------------------------------------
# Transaction drill-down
# ----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Transaction detail")
if len(view):
    selected_id = st.selectbox("Select a transaction to inspect", view["transaction_id"].head(200))
    row = view[view["transaction_id"] == selected_id].iloc[0]

    d1, d2 = st.columns([1, 1])
    with d1:
        st.markdown(f"**Transaction ID:** {row['transaction_id']}")
        st.markdown(f"**Account:** {row['account_id']} ({row['account_type']}, {row['kyc_tier']})")
        st.markdown(f"**Amount:** {row['amount']:,.2f} {row['currency']}")
        st.markdown(f"**Channel:** {row['channel']} &nbsp;|&nbsp; **Category:** {row['merchant_category']}")
        st.markdown(f"**Risk score:** `{row['risk_score']:.3f}`")
        st.markdown(f"**Why flagged:** {row['why_flagged']}")
    with d2:
        feat_vals = pd.DataFrame({
            "feature": ["amount_to_avg_ratio", "transaction_velocity_1h", "is_new_device",
                        "is_cross_border", "account_age_days"],
            "value": [row["amount_to_avg_ratio"], row["transaction_velocity_1h"], row["is_new_device"],
                      row["is_cross_border"], row["account_age_days"]],
        })
        st.bar_chart(feat_vals.set_index("feature"))
else:
    st.caption("No flagged transactions to inspect at the current threshold/filters.")

st.markdown("---")
st.caption(
    "Prototype dashboard — scores are model outputs, not automated decisions. "
    "Not connected to Finlora's live payment systems. See project brief for scope."
)
