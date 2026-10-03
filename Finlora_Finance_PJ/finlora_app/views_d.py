"""Pages: Review queue, Live scorer, Handover."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import models as M
from ctx import get_ctx
from data import CATEGORICAL_FEATS, NUMERIC_FEATS
from theme import AMBER, FRAUD, INK, LEGIT, MUTED, TEAL, callout, header, kpis, num, pct, section, show

TIER_COLOURS = {"Critical": FRAUD, "High": AMBER, "Watch": LEGIT, "Low": "#B7C2CE"}


def _queue_threshold(b: dict, widget_key: str, label: str) -> float:
    """Slider whose value survives page changes (Streamlit drops widget state when a widget is not rendered)."""
    if widget_key not in st.session_state:
        st.session_state[widget_key] = float(np.clip(st.session_state.get("_queue_thr", b["prod_threshold"]), 0.30, 0.99))
    v = st.slider(label, 0.30, 0.99, step=0.005, key=widget_key, format="%.3f")
    st.session_state["_queue_thr"] = float(v)
    return float(v)


def _contrib_chart(contrib: pd.Series, raw: pd.Series, title: str, height: int = 420) -> go.Figure:
    c = contrib.sort_values()
    labels = [f"{M.reason_text(k, raw)}  ({v:+.2f})" for k, v in c.items()]
    f = go.Figure(go.Bar(x=c.values, y=labels, orientation="h", marker_color=[FRAUD if v > 0 else LEGIT for v in c.values]))
    f.update_xaxes(title="Pushes risk up (red) or down (blue), in log-odds", zeroline=True, zerolinecolor=INK)
    f.update_yaxes(automargin=True)
    return f


# =============================================================================== REVIEW QUEUE
def review_queue() -> None:
    ds, b = get_ctx()
    header("Stage 6 of 6", "Fraud review queue",
           "Every transaction scored, ranked and explained in plain language, so an analyst can start at the top and know why each item is there.")
    c1, c2 = st.columns([1, 1.4])
    scope = c1.radio("Transactions to score", ["Held-out test set", "All transactions"], horizontal=True,
                     help="The test set is out-of-sample and gives honest performance. 'All' includes rows the model was trained on.")
    with c2:
        thr = _queue_threshold(b, "q_thr_w", "Alert threshold (production default = F1-optimal)")
    idx = b["test_index"] if scope.startswith("Held") else ds.cleaned.index.to_numpy()
    X = ds.X.loc[idx]
    score, cdf, icpt = M.score_and_explain(b, X)
    q = ds.cleaned.loc[idx, ["transaction_id", "account_id", "account_holder_name", "timestamp", "merchant_category", "channel", "transaction_country", "amount", "currency",
                             "account_type", "is_fraud"]].copy()
    q["score"], q["tier"] = score, M.tiers(score, thr).values
    q["logit"] = cdf.sum(axis=1).reindex(q.index).values + icpt
    q = q.sort_values("logit", ascending=False)  # log-odds break ties among saturated 1.000 scores

    tt = q.groupby("tier")["is_fraud"].agg(Items="count", Confirmed="sum", Rate="mean").reindex(["Critical", "High", "Watch", "Low"]).fillna(0).reset_index()
    flagged = q[q["score"] >= thr]
    cols = st.columns(5)
    for col, (_, r) in zip(cols[:4], tt.iterrows()):
        sub = f"{r['Rate'] * 100:.0f}% confirmed fraud" if r["Items"] else "none"
        col.markdown(f'<div class="fl-kpi" style="border-left-color:{TIER_COLOURS[r["tier"]]}"><div class="l">{r["tier"]}</div><div class="v">{int(r["Items"]):,}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)
    kpi_flag = f'<div class="fl-kpi teal"><div class="l">Alerts at threshold</div><div class="v">{len(flagged):,}</div><div class="s">{flagged["is_fraud"].mean() * 100 if len(flagged) else 0:.0f}% are real fraud</div></div>'
    cols[4].markdown(kpi_flag, unsafe_allow_html=True)
    st.write("")
    callout("<b>Critical</b> = score of 0.99 or more. <b>High</b> = at or above the alert threshold. <b>Watch</b> = 0.5 up to the threshold. Scores are rankings for prioritisation, "
            "not calibrated probabilities (class-balancing inflates them), so read 'confirmed fraud' above as the true hit-rate of each tier.", "info", "How to read the tiers")

    f1, f2, f3, f4 = st.columns(4)
    tiers_sel = f1.multiselect("Tier", ["Critical", "High", "Watch", "Low"], default=["Critical", "High"])
    cats = f2.multiselect("Merchant category", sorted(q["merchant_category"].unique()))
    chans = f3.multiselect("Channel", sorted(q["channel"].unique()))
    acct = f4.text_input("Account ID contains", "")
    v = q[q["tier"].isin(tiers_sel)]
    if cats:
        v = v[v["merchant_category"].isin(cats)]
    if chans:
        v = v[v["channel"].isin(chans)]
    if acct:
        v = v[v["account_id"].str.contains(acct, case=False)]
    st.caption(f"{len(v):,} transactions match. Reasons are generated for the top 2,000.")
    top = v.head(2000).copy()
    cb = cdf.loc[top.index]
    top["Why flagged"] = [M.top_reasons(cb.loc[i], ds.X.loc[i]) for i in top.index]
    top["Known outcome"] = top["is_fraud"].map({1: "Fraud", 0: "Legitimate"})
    top.insert(0, "Rank", np.arange(1, len(top) + 1))
    show_cols = ["Rank", "transaction_id", "score", "tier", "Why flagged", "account_holder_name", "merchant_category", "channel", "amount", "currency", "timestamp", "account_id", "Known outcome"]
    st.dataframe(top[show_cols], width="stretch", hide_index=True, height=420,
                 column_config={"score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.3f"), "amount": st.column_config.NumberColumn(format="%.2f"),
                                "Why flagged": st.column_config.TextColumn(width="large"), "tier": st.column_config.TextColumn("Tier", width="small"), "timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")})
    st.download_button("Download queue (CSV)", top[show_cols].to_csv(index=False).encode(), "finlora_review_queue.csv", "text/csv")

    if len(top):
        section("Inspect a transaction")
        pick = st.selectbox("Choose from the filtered queue", top["transaction_id"].head(300).tolist(), format_func=lambda t: f"{t}  |  score {top.loc[top['transaction_id'] == t, 'score'].iloc[0]:.3f}")
        i = top.index[top["transaction_id"] == pick][0]
        row = top.loc[i]
        a, c_ = st.columns([1.2, 1])
        with a:
            show(_contrib_chart(cdf.loc[i], ds.X.loc[i], ""), 430, f"Why {pick} scored {row['score']:.3f}")
        with c_:
            tone = {"Critical": "fraud", "High": "warn", "Watch": "ok", "Low": "ok"}[row["tier"]]
            st.markdown(f'<span class="fl-pill {tone}">{row["tier"].upper()}</span> &nbsp; Known outcome: **{row["Known outcome"]}**', unsafe_allow_html=True)
            st.markdown(f"**{row['account_holder_name']}** &middot; {row['account_id']} &middot; {row['account_type']}")
            st.markdown(f"{row['amount']:,.2f} {row['currency']} &middot; {row['merchant_category']} via {row['channel']} &middot; {row['transaction_country']}")
            hist = ds.cleaned[ds.cleaned["account_id"] == row["account_id"]].sort_values("timestamp", ascending=False)
            st.caption(f"This account has {len(hist):,} transactions, {int(hist['is_fraud'].sum())} confirmed fraud.")
            st.dataframe(hist[["timestamp", "merchant_category", "amount", "amount_to_avg_ratio", "is_fraud"]].head(8), width="stretch", hide_index=True, height=min(38 * (min(len(hist), 8) + 1) + 4, 340))


# =============================================================================== LIVE SCORER
PRESETS = {
    "Everyday grocery purchase": dict(sc_ratio=0.9, sc_vel=0, sc_dev=False, sc_xb=False, sc_age=800, sc_hour=13, sc_ref=False, sc_type="Individual", sc_kyc="Tier3_Enhanced", sc_cat="Groceries", sc_chan="Mobile App"),
    "Unusual wire (borderline)": dict(sc_ratio=14.0, sc_vel=0, sc_dev=True, sc_xb=True, sc_age=20, sc_hour=3, sc_ref=False, sc_type="Business", sc_kyc="Tier1_Basic", sc_cat="Wire Transfer", sc_chan="API/Integration"),
    "Suspicious large wire": dict(sc_ratio=40.0, sc_vel=1, sc_dev=True, sc_xb=True, sc_age=20, sc_hour=3, sc_ref=False, sc_type="Business", sc_kyc="Tier1_Basic", sc_cat="Wire Transfer", sc_chan="API/Integration"),
    "Rapid-fire card testing": dict(sc_ratio=1.5, sc_vel=3, sc_dev=True, sc_xb=False, sc_age=60, sc_hour=2, sc_ref=False, sc_type="Individual", sc_kyc="Tier2_Verified", sc_cat="Electronics", sc_chan="Card Not Present"),
}


def _apply_preset(name: str) -> None:
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


def live_scorer() -> None:
    ds, b = get_ctx()
    header("Stage 6 of 6", "Live transaction scorer",
           "Describe a transaction and see the production model's risk score, the decision, and exactly which factors drove it.")
    for k, v in PRESETS["Everyday grocery purchase"].items():
        st.session_state.setdefault(k, v)
    thr = st.session_state.get("_queue_thr", float(b["prod_threshold"]))
    st.markdown("**Start from a scenario**")
    pc = st.columns(len(PRESETS))
    for col, name in zip(pc, PRESETS):
        col.button(name, on_click=_apply_preset, args=(name,), width="stretch")
    left, right = st.columns([1, 1.15])
    with left:
        with st.container(border=True):
            a, c_ = st.columns(2)
            ratio = a.number_input("Amount vs 30-day average (x)", 0.0, 1000.0, step=0.5, key="sc_ratio", help="1.0 = typical for this account. 10 = ten times the usual amount.")
            vel = c_.number_input("Transactions in previous hour", 0, 20, key="sc_vel")
            age = a.number_input("Account age (days)", 0, 6000, key="sc_age")
            hour = c_.slider("Hour of day", 0, 23, key="sc_hour")
            t1, t2, t3 = st.columns(3)
            dev, xb, ref = t1.toggle("New device", key="sc_dev"), t2.toggle("Cross-border", key="sc_xb"), t3.toggle("Refund", key="sc_ref")
            a, c_ = st.columns(2)
            atype = a.selectbox("Account type", sorted(ds.X["account_type"].unique()), key="sc_type")
            kyc = c_.selectbox("KYC tier", sorted(ds.X["kyc_tier"].unique()), key="sc_kyc")
            cat = a.selectbox("Merchant category", sorted(ds.X["merchant_category"].unique()), key="sc_cat")
            chan = c_.selectbox("Channel", sorted(ds.X["channel"].unique()), key="sc_chan")
    row = pd.DataFrame([{"amount_to_avg_ratio": ratio, "transaction_velocity_1h": vel, "is_new_device": int(dev), "is_cross_border": int(xb), "account_age_days": age,
                         "hour_of_day": hour, "is_refund": int(ref), "account_type": atype, "kyc_tier": kyc, "merchant_category": cat, "channel": chan}])
    score, cdf, icpt = M.score_and_explain(b, row)
    s = float(score[0])
    tier = M.tiers([s], thr).iloc[0]
    with right:
        g = go.Figure(go.Indicator(mode="gauge+number", value=s, number=dict(valueformat=".3f", font=dict(family="IBM Plex Mono", size=44)),
                                   gauge=dict(axis=dict(range=[0, 1]), bar=dict(color=INK, thickness=.28),
                                              steps=[dict(range=[0, .5], color="#E4EBF4"), dict(range=[.5, thr], color="#CFE0F7"), dict(range=[thr, .99], color="#FBE3A8"), dict(range=[.99, 1], color="#F7B9BF")],
                                              threshold=dict(line=dict(color=FRAUD, width=4), thickness=.85, value=thr))))
        show(g, 260, "Risk score")
        verdict = ("FLAG FOR REVIEW", "fraud") if s >= thr else ("WATCH", "warn") if s >= .5 else ("ALLOW", "ok")
        st.markdown(f'<span class="fl-pill {verdict[1]}" style="font-size:1rem">{verdict[0]}</span> &nbsp; Priority tier: <b>{tier}</b> &nbsp;&middot;&nbsp; alert threshold {thr:.3f}', unsafe_allow_html=True)
    show(_contrib_chart(cdf.iloc[0], row.iloc[0], ""), 430, "What drove this score")
    st.caption(f"Score = 1 / (1 + e^-(baseline {icpt:+.2f} + sum of contributions)). Contributions are measured against an average transaction, so a bar can only push the score up or down relative to typical behaviour.")


# =============================================================================== HANDOVER
def handover() -> None:
    ds, b = get_ctx()
    r = b["results"][M.PROD]["tuned"]
    h = b["honest"]
    header("Wrap-up", "Handover summary",
           "What was built, what we recommend, and what the team should know before relying on it.")
    kpis([("Production model", "Logistic Regression", "simple, fast, explainable", "legit"), ("Alert threshold", f"{b['prod_threshold']:.3f}", "F1-optimal", "amber"),
          ("Precision / recall", f"{r['precision']:.2f} / {r['recall']:.2f}", "held-out test set", "teal"), ("F1 / ROC-AUC", f"{r['f1']:.2f} / {b['results'][M.PROD]['auc']:.2f}", "", ""),
          ("Honest-threshold F1", f"{h['f1']:.3f}", "threshold chosen without test data", "")])
    st.write("")
    section("Project framework: status")
    rows = [("Problem definition and data inventory", "Done", "Overview, Stage 1"), ("Clean, analysis-ready dataset", "Done", "Stage 2: reproduced exactly"),
            ("EDA summary report", "Done", "Stage 3"), ("Feature importance / behavioural drivers", "Done", "Drivers"),
            ("Two trained models (LR baseline, RF primary)", "Done", "Stage 5, plus 5 extra techniques"),
            ("Evaluation: precision, recall, F1, ROC-AUC, confusion matrix, thresholds", "Done", "Evaluation, Threshold studio"),
            ("Prioritised, explainable review queue", "Done", "Stage 6"), ("Documentation and handover", "Done", "This page")]
    st.dataframe(pd.DataFrame(rows, columns=["Deliverable", "Status", "Where to find it"]), width="stretch", hide_index=True)

    section("Recommendations")
    st.markdown(f"""
- **Deploy Logistic Regression at threshold {b['prod_threshold']:.2f}** as the scoring engine. It matches the best F1 of far heavier models, is trivial to run and can be explained to regulators and analysts line by line.
- **Run a tiered workflow.** Critical alerts go to an immediate-action lane; High to standard review; Watch to batch review or automated step-up verification.
- **Pair transaction scoring with account controls.** A handful of accounts hold most of the fraud, so limits and watch-lists on repeat offenders are cheap and effective.
- **Plan for seasonality.** Volume and fraud both surge in November and December; staff the review desk accordingly.
- **Tune the threshold to capacity, not habit.** Use the Threshold studio with real loss and review-cost figures to pick the operating point.
""")
    section("What to watch before go-live")
    callout("The <b>velocity</b> feature separates fraud almost perfectly (fraud rate above 90% whenever another transaction happened in the previous hour). "
            "Confirm with the data owners that it is computed from information available <i>at decision time</i> and not derived after the fact. If it is not, performance will fall in production.", "risk", "Verify the strongest signal")
    callout("The split is random, not chronological. Fraud patterns drift, so validate on a later time window (train on earlier months, test on later) before trusting these figures for next year.", "warn", "Out-of-time validation")
    callout("Class weighting inflates raw scores, so treat them as rankings. If a true probability is needed (for example for expected-loss pricing), use the calibrated Random Forest or recalibrate the Logistic Regression.", "warn", "Scores are not probabilities")
    callout("Amounts are stored in four currencies (NGN, USD, GBP, EUR). The model avoids the issue by using the ratio to each account's own baseline, but any monetary reporting must convert to a common currency first.", "info", "Mixed currencies")
    callout("Maintain a monthly review of precision, recall and alert volume; retrain when either drifts, and keep the pipeline in this app as the executable specification.", "info", "Monitoring")

    with st.expander("Analyst notes on the source notebooks (for the project team)"):
        st.markdown(f"""
- **`Featured_Engineering.ipynb` was empty.** The stage is reconstructed from the modelling notebook (7 numeric + 4 categorical inputs, one-hot encoded) and verified identical to `featured_engineered_dataset.csv`.
- **The verdict text in `Model_2.ipynb` says the best F1 is "~0.93".** The notebook's own output shows ~0.711 (that is Logistic Regression at threshold 0.935). The 0.93 figure is the ROC-AUC. This app reports the computed values.
- `Model_2.ipynb` loads `artifacts/feature_config.json`, which was not supplied; the app derives the same configuration directly.
- `Data_Cleaning.ipynb` uses `px` without importing plotly, and the EDA helper `fraud_rate_bar` refers to an undefined `df`. Neither affects results; both are worth tidying.
- All notebooks read from Windows-specific absolute paths. This app locates the files by folder instead.
- The tuned Random Forest uses the parameters recorded by the notebook's RandomizedSearchCV (`max_depth=14`, `min_samples_leaf=5`) rather than re-running the search.
""")
    with st.expander("How to run this app"):
        st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
        st.caption("Place finlora_transactions.csv and finlora_accounts.csv in a data/ folder next to app.py (or set FINLORA_DATA_DIR). The first launch trains the models and caches them in artifacts/.")
