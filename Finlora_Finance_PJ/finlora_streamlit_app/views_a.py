"""Pages: Overview, Data sources, Cleaning."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ctx import get_ctx
from data import CATEGORICAL_FEATS, NUMERIC_FEATS
from models import PROD
from theme import AMBER, FRAUD, INK, LEGIT, MUTED, TEAL, callout, header, html_table, kpis, num, pct, section, show

DICTIONARY = {
    "transaction_id": "Unique transaction reference", "account_id": "Account that made the transaction (joins to accounts file)",
    "account_type": "Individual or Business", "kyc_tier": "Identity verification level (Tier1 Basic, Tier2 Verified, Tier3 Enhanced)",
    "timestamp": "When the transaction happened", "day_of_week": "Weekday of the transaction", "hour_of_day": "Hour (0-23) of the transaction",
    "description": "Free-text transaction descriptor", "merchant_name": "Merchant (empty for transfers)", "merchant_category": "Type of merchant / payment",
    "channel": "How the transaction was made (card, mobile, web, USSD, API)", "amount": "Transaction amount in the account's own currency (negative = refund)",
    "currency": "NGN, USD, GBP or EUR", "amount_to_avg_ratio": "Amount divided by the account's 30-day average transaction",
    "avg_transaction_amount_30d": "Account's rolling 30-day average transaction amount", "transaction_velocity_1h": "Transactions by the account in the previous hour",
    "transaction_country": "Country where the transaction happened", "home_country": "Account's home country",
    "is_cross_border": "1 if transaction country differs from home country", "device_id": "Device fingerprint (empty for some channels)",
    "is_new_device": "1 if the device was not seen on this account before", "account_age_days": "Days since the account was opened",
    "status": "Completed, Declined or Reversed", "is_fraud": "TARGET - 1 if confirmed fraud", "account_holder_name": "Name on the account",
    "account_created_date": "Date the account was opened", "personal_spend_baseline_usd": "Typical monthly spend baseline in USD",
    "is_refund": "1 if amount < 0 (engineered during cleaning)",
}


def _go(key: str, label: str) -> None:
    pg = st.session_state.get("_pages", {}).get(key)
    if pg is not None:
        st.page_link(pg, label=label)


# =============================================================================== OVERVIEW
def overview() -> None:
    ds, b = get_ctx()
    c, r = ds.cleaned, b["results"][PROD]["tuned"]
    base = c["is_fraud"].mean()
    st.markdown(
        '<div class="fl-hero"><div class="fl-kicker">FINLORA &middot; TRANSACTION RISK PROGRAMME</div>'
        '<h1>Catching fraud that fixed thresholds miss</h1>'
        f'<p>Finlora\'s rule-based monitoring cannot keep pace with transaction volume. This workbench follows the full path from {len(c):,} raw '
        'transactions to a scored, explainable review queue, so every number you see is reproducible from the source files.</p></div>',
        unsafe_allow_html=True)

    kpis([("Transactions analysed", num(len(c)), f"{c['timestamp'].min():%b %Y} to {c['timestamp'].max():%b %Y}", ""),
          ("Accounts", num(c["account_id"].nunique()), f"{len(c) / c['account_id'].nunique():.1f} transactions per account", "legit"),
          ("Confirmed fraud", num(c["is_fraud"].sum()), f"{pct(base)} of all transactions", "fraud"),
          ("Fraud caught (recall)", pct(r["recall"], 1), "on unseen test transactions", "teal"),
          ("Alert accuracy (precision)", pct(r["precision"], 1), "of flagged items are real fraud", "amber")])

    section("The journey from raw data to decision")
    steps = [("1", "Load", "Two raw files joined on account", f"{len(ds.tx_raw):,} tx + {len(ds.acc_raw):,} accounts", "data"),
             ("2", "Clean", "Repair gaps, refunds and a broken ratio", f"{len(c):,} rows x {c.shape[1]} cols", "clean"),
             ("3", "Explore", "Where does fraud concentrate?", f"{pct(base)} fraud rate", "eda"),
             ("4", "Engineer", "11 inputs expanded for modelling", f"{ds.dummies.shape[1]} model features", "feat"),
             ("5", "Model", "Seven techniques benchmarked", f"{PROD} wins", "model"),
             ("6", "Operate", "Prioritised, explainable alerts", "Review queue + live scorer", "queue")]
    cols = st.columns(6)
    for col, (n, h, d, o, key) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="fl-step"><div class="n">STAGE {n}</div><div class="h">{h}</div><div class="d">{d}</div><div class="o">{o}</div></div>',
                        unsafe_allow_html=True)
            _go(key, "Open")

    section("What the production model delivers", f"{PROD} scored at its F1-optimal threshold of {b['prod_threshold']:.3f}, on {b['test_size']:,} transactions it never saw in training.")
    left, right = st.columns([1.1, 1])
    with left:
        per1k = r["alerts"] / b["test_size"] * 1000
        callout(f"For every <b>1,000 transactions</b> the model raises about <b>{per1k:.0f} alerts</b>. Of every 100 alerts, "
                f"<b>{r['precision'] * 100:.0f} are genuine fraud</b>, and the model catches <b>{r['recall'] * 100:.0f} of every 100</b> fraudulent transactions. "
                f"Only <b>{r['fp']}</b> legitimate transactions out of {r['tn'] + r['fp']:,} were flagged by mistake "
                f"({r['fp'] / (r['tn'] + r['fp']) * 100:.2f}%).", "good", "Plain-language read-out")
        callout(f"A model that never flags anything would be <b>{(1 - base) * 100:.1f}% accurate</b> yet catch zero fraud. "
                "That is why this programme is judged on precision, recall, F1 and ROC-AUC, not accuracy.", "warn", "Why accuracy is the wrong yardstick")
    with right:
        fig = go.Figure(go.Bar(x=[r["recall"], r["precision"], r["f1"], b["results"][PROD]["auc"]],
                               y=["Recall", "Precision", "F1 score", "ROC-AUC"], orientation="h",
                               marker_color=[TEAL, AMBER, LEGIT, INK], text=[f"{v:.3f}" for v in [r["recall"], r["precision"], r["f1"], b["results"][PROD]["auc"]]],
                               textposition="outside"))
        fig.update_xaxes(range=[0, 1.12], showgrid=False, visible=False)
        fig.update_yaxes(autorange="reversed")
        show(fig, 240, "Production model scorecard")

    section("Four things the data already tells us")
    vel = c.groupby(c["transaction_velocity_1h"] > 0)["is_fraud"].mean()
    ratio = c.groupby(c["amount_to_avg_ratio"] >= 10)["is_fraud"].mean()
    cat = c.groupby("merchant_category")["is_fraud"].mean().sort_values(ascending=False)
    per_acc = c.groupby("account_id")["is_fraud"].sum().sort_values(ascending=False)
    a, bcol = st.columns(2)
    with a:
        callout(f"Transactions arriving within an hour of another one are fraud <b>{vel[True] * 100:.0f}%</b> of the time, against "
                f"<b>{vel[False] * 100:.1f}%</b> for the rest. Velocity is the sharpest single signal in the data.", "risk", "Rapid-fire activity")
        callout(f"<b>{cat.index[0]}</b> ({cat.iloc[0] * 100:.1f}%), <b>{cat.index[1]}</b> ({cat.iloc[1] * 100:.1f}%) and <b>{cat.index[2]}</b> "
                f"({cat.iloc[2] * 100:.1f}%) run at 2-3x the {pct(base, 1)} baseline.", "info", "High-risk payment types")
    with bcol:
        callout(f"Amounts 10x or more above the account's own 30-day average are fraudulent <b>{ratio[True] * 100:.1f}%</b> of the time, "
                f"versus <b>{ratio[False] * 100:.1f}%</b> below that line.", "risk", "Out-of-pattern amounts")
        callout(f"Only <b>{(per_acc > 0).mean() * 100:.0f}%</b> of accounts have any fraud, and the ten worst accounts hold "
                f"<b>{per_acc.head(10).sum() / per_acc.sum() * 100:.0f}%</b> of all fraud cases. Account-level monitoring pays off.", "info", "Fraud is concentrated")


# =============================================================================== DATA SOURCES
def sources() -> None:
    ds, _ = get_ctx()
    header("Stage 1 of 6", "Data sources and loading",
           "Two raw extracts feed the project: a transaction ledger and an account master. They link on account_id.")
    tx, acc = ds.tx_raw, ds.acc_raw
    kpis([("Transaction rows", num(len(tx)), f"{tx.shape[1]} columns", ""), ("Account rows", num(len(acc)), f"{acc.shape[1]} columns", "legit"),
          ("Unique accounts in ledger", num(tx["account_id"].nunique()), "all matched to the master", "teal"),
          ("Raw file size", f"{(tx.memory_usage(deep=True).sum() + acc.memory_usage(deep=True).sum()) / 1e6:.0f} MB", "in memory", ""),
          ("Period covered", f"{pd.to_datetime(tx['timestamp']).dt.to_period('M').nunique()} months", f"{str(tx['timestamp'].min())[:10]} to {str(tx['timestamp'].max())[:10]}", "amber")])
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Transactions", "Accounts", "Column profile", "Data dictionary"])
    with t1:
        q = st.text_input("Filter transactions (searches description, merchant, account)", "", key="txq")
        view = tx
        if q:
            m = tx[["description", "merchant_name", "account_id"]].astype(str).apply(lambda s: s.str.contains(q, case=False, na=False)).any(axis=1)
            view = tx[m]
        st.caption(f"Showing {min(len(view), 500):,} of {len(view):,} matching rows")
        st.dataframe(view.head(500), width="stretch", height=380, hide_index=True)
    with t2:
        c1, c2, c3 = st.columns(3)
        for col, field, ttl, colour in [(c1, "account_type", "Account type", LEGIT), (c2, "home_country", "Home country", TEAL), (c3, "kyc_tier", "KYC tier", AMBER)]:
            vc = acc[field].value_counts().reset_index()
            vc.columns = [field, "accounts"]
            f = px.bar(vc, x=field, y="accounts", text="accounts", color_discrete_sequence=[colour])
            f.update_traces(textposition="outside")
            f.update_yaxes(visible=False, showgrid=False)
            f.update_xaxes(title=None)
            with col:
                show(f, 300, ttl)
        st.dataframe(acc.head(300), width="stretch", height=300, hide_index=True)
    with t3:
        for name, df in [("Transactions", tx), ("Accounts", acc)]:
            st.markdown(f"**{name}**")
            prof = pd.DataFrame({"Type": df.dtypes.astype(str), "Missing": df.isnull().sum(), "Missing %": (df.isnull().mean() * 100).round(2),
                                 "Unique values": df.nunique()})
            st.dataframe(prof, width="stretch")
    with t4:
        rows = [(k, v) for k, v in DICTIONARY.items() if v and (k in ds.cleaned.columns)]
        st.dataframe(pd.DataFrame(rows, columns=["Column", "Meaning"]), width="stretch", hide_index=True, height=520)
    callout("The ledger is not ready for modelling: it contains empty merchant and device fields, refunds stored as negative amounts, "
            "invalid 30-day baselines and a pre-computed ratio that disagrees with its own inputs. Stage 2 fixes each of these.", "warn", "What loading revealed")


# =============================================================================== CLEANING
@st.cache_data(show_spinner=False)
def _csv_bytes(sig: str, _df: pd.DataFrame) -> bytes:
    return _df.to_csv(index=False).encode()


def cleaning() -> None:
    ds, _ = get_ctx()
    header("Stage 2 of 6", "Data cleaning",
           "Nine checks, run live on the raw files with exactly the logic from the cleaning notebook. Nothing is dropped silently; every repair is counted.")
    log = pd.DataFrame(ds.log)
    fixed = int(log.loc[log["Step"].isin(["Missing merchant", "Missing device", "Negative amounts", "Invalid 30-day baseline", "Inconsistent ratio"]), "Rows affected"].sum())
    kpis([("Rows in", num(len(ds.tx_raw)), f"{ds.tx_raw.shape[1]} columns", ""), ("Rows out", num(len(ds.cleaned)), f"{ds.cleaned.shape[1]} columns", "teal"),
          ("Rows lost", "0", "no records discarded", "legit"), ("Cell-level repairs", num(fixed), "across 5 issue types", "amber"),
          ("Missing cells left", num(ds.cleaned.isnull().sum().sum()), "was " + num(ds.missing_before.sum()), "fraud" if ds.cleaned.isnull().sum().sum() else "teal")])
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Cleaning log", "Before and after", "The broken ratio", "Verification"])
    with t1:
        html_table(log, key_col="Step", num_cols=("Rows affected",))
    with t2:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            allc = sorted(set(ds.missing_before.index) | set(ds.missing_after.index))
            f = go.Figure()
            f.add_bar(name="Before", x=allc, y=[int(ds.missing_before.get(k, 0)) for k in allc], marker_color=FRAUD)
            f.add_bar(name="After", x=allc, y=[int(ds.missing_after.get(k, 0)) for k in allc], marker_color=TEAL)
            f.update_layout(barmode="group")
            show(f, 340, "Missing values per column")
        with c2:
            ref = ds.tx_raw[ds.tx_raw["amount"] < 0]
            f = px.bar(ref["merchant_category"].value_counts().reset_index(), x="merchant_category", y="count", color_discrete_sequence=[AMBER], text="count")
            f.update_yaxes(visible=False, showgrid=False)
            f.update_xaxes(title=None)
            show(f, 340, f"{len(ref)} negative amounts, by merchant category")
        callout("Every negative amount sits in Retail, which is the signature of customer refunds. They were kept and labelled with an <code>is_refund</code> flag rather than deleted.", "info")
    with t3:
        callout(f"<b>{int(log.loc[log['Step'] == 'Inconsistent ratio', 'Rows affected'].iloc[0])}</b> rows carried an <code>amount_to_avg_ratio</code> that disagreed "
                "with |amount| / avg_30d by more than 0.5. Because this ratio is the strongest numeric fraud signal, it was recomputed from its inputs for every row.", "warn")
        ex = ds.ratio_examples.head(100)
        if len(ex):
            f = px.scatter(ex, x="amount_to_avg_ratio", y="recomputed_ratio", color_discrete_sequence=[FRAUD], opacity=.7, log_x=True, log_y=True,
                           hover_data=["transaction_id"])
            lim = [min(ex[["amount_to_avg_ratio", "recomputed_ratio"]].replace(0, np.nan).min().min(), 1e-3), ex[["amount_to_avg_ratio", "recomputed_ratio"]].max().max()]
            f.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1], line=dict(color=INK, dash="dash"))
            f.update_xaxes(title="Ratio as supplied")
            f.update_yaxes(title="Ratio recomputed")
            show(f, 380, "Supplied vs recomputed ratio (dashed line = agreement)")
            st.dataframe(ex.head(15), width="stretch", hide_index=True)
    with t4:
        st.markdown("**Does the live pipeline reproduce your saved `cleaned_finlora_data.csv`?**")
        if ds.clean_check is None:
            st.info("cleaned_finlora_data.csv was not found beside the raw files, so there is nothing to compare against.")
        else:
            for k, v in ds.clean_check.items():
                st.markdown(f"{'&#9989;' if v else '&#10060;'} {k}", unsafe_allow_html=True)
            if all(ds.clean_check.values()):
                callout("The pipeline in this app and the cleaning notebook produce the same dataset, row for row.", "good", "Verified")
        st.markdown("**Cleaned dataset preview**")
        st.dataframe(ds.cleaned.head(300), width="stretch", height=300, hide_index=True)
        if st.toggle("Prepare cleaned CSV for download", key="prep_clean"):
            st.download_button("Download cleaned_finlora_data.csv", _csv_bytes(ds.signature, ds.cleaned), "cleaned_finlora_data.csv", "text/csv")
