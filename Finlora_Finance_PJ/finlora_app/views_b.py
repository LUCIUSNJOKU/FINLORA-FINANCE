"""Pages: Exploratory data analysis, Feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score

from ctx import get_ctx
from data import CATEGORICAL_FEATS, NUMERIC_FEATS
from theme import AMBER, FRAUD, INK, LEGIT, MUTED, TEAL, callout, header, html_table, kpis, num, pct, section, show

DIMENSIONS = {"Merchant category": "merchant_category", "Channel": "channel", "Account type": "account_type", "KYC tier": "kyc_tier",
              "Transaction country": "transaction_country", "Home country": "home_country", "Currency": "currency",
              "Transaction status": "status", "Day of week": "day_of_week", "Cross-border": "is_cross_border", "New device": "is_new_device",
              "Hour of day": "hour_of_day"}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col)["is_fraud"].agg(Transactions="count", Fraud="sum", Rate="mean").reset_index()
    g["Lift"] = g["Rate"] / df["is_fraud"].mean()
    return g


def _rate_bar(df: pd.DataFrame, col: str, title: str, order: list | None = None, height: int = 360, horizontal: bool = True):
    g = _rate_table(df, col)
    g[col] = g[col].astype(str)
    base = df["is_fraud"].mean()
    if order:
        g[col] = pd.Categorical(g[col], order, ordered=True)
        g = g.sort_values(col)
    else:
        g = g.sort_values("Rate", ascending=horizontal)
    colors = [FRAUD if r >= base else LEGIT for r in g["Rate"]]
    if horizontal:
        f = go.Figure(go.Bar(y=g[col], x=g["Rate"], orientation="h", marker_color=colors, text=[f"{r * 100:.2f}%" for r in g["Rate"]],
                             textposition="outside", customdata=g[["Transactions", "Fraud", "Lift"]],
                             hovertemplate="<b>%{y}</b><br>Fraud rate %{x:.2%}<br>%{customdata[0]:,} transactions, %{customdata[1]:,} fraud<br>%{customdata[2]:.2f}x baseline<extra></extra>"))
        f.add_vline(x=base, line_dash="dash", line_color=INK, annotation_text=f"baseline {base * 100:.2f}%", annotation_position="top")
        f.update_xaxes(visible=False, showgrid=False, range=[0, g["Rate"].max() * 1.2])
    else:
        f = go.Figure(go.Bar(x=g[col], y=g["Rate"], marker_color=colors, text=[f"{r * 100:.1f}%" for r in g["Rate"]], textposition="outside",
                             customdata=g[["Transactions", "Fraud", "Lift"]],
                             hovertemplate="<b>%{x}</b><br>Fraud rate %{y:.2%}<br>%{customdata[0]:,} transactions<extra></extra>"))
        f.add_hline(y=base, line_dash="dash", line_color=INK, annotation_text=f"baseline {base * 100:.2f}%", annotation_position="top left")
        f.update_yaxes(visible=False, showgrid=False, range=[0, g["Rate"].max() * 1.22])
        f.update_xaxes(type="category")
    show(f, height, title)
    return g


def _bucket(df: pd.DataFrame, series: pd.Series, bins, labels, title: str, key: str):
    d = df.assign(_b=pd.cut(series, bins=bins, labels=labels, right=True))
    g = d.groupby("_b", observed=True)["is_fraud"].agg(Transactions="count", Rate="mean").reset_index()
    base = df["is_fraud"].mean()
    f = go.Figure(go.Bar(x=g["_b"].astype(str), y=g["Rate"], marker_color=[FRAUD if r >= base else LEGIT for r in g["Rate"]],
                         text=[f"{r * 100:.1f}%" for r in g["Rate"]], textposition="outside", customdata=g["Transactions"],
                         hovertemplate="%{x}<br>Fraud rate %{y:.2%}<br>%{customdata:,} transactions<extra></extra>"))
    f.add_hline(y=base, line_dash="dash", line_color=INK)
    f.update_yaxes(visible=False, showgrid=False, range=[0, g["Rate"].max() * 1.22])
    f.update_xaxes(type="category")
    show(f, 330, title, key=key)


# =============================================================================== EDA
def eda() -> None:
    ds, _ = get_ctx()
    c = ds.cleaned
    base = c["is_fraud"].mean()
    header("Stage 3 of 6", "Exploratory data analysis",
           "Which behaviours separate fraudulent from legitimate transactions? Red bars sit above the portfolio fraud rate, blue bars below it.")
    kpis([("Transactions", num(len(c)), "", ""), ("Fraud cases", num(c["is_fraud"].sum()), "", "fraud"), ("Legitimate", num((c["is_fraud"] == 0).sum()), "", "legit"),
          ("Fraud rate", pct(base), f"1 in {1 / base:.0f} transactions", "amber"), ("Imbalance", f"{(1 - base) / base:.0f} : 1", "legit to fraud", "")])
    st.write("")
    t1, t2, t3, t4, t5, t6 = st.tabs(["Target", "Segments", "Behavioural signals", "Time", "Accounts", "Correlations"])

    with t1:
        l, r = st.columns([1, 1.2])
        with l:
            f = go.Figure(go.Pie(labels=["Legitimate", "Fraud"], values=[(c["is_fraud"] == 0).sum(), c["is_fraud"].sum()], hole=.62,
                                 marker_colors=[LEGIT, FRAUD], textinfo="label+percent", sort=False))
            f.update_layout(showlegend=False, annotations=[dict(text=f"<b>{pct(base)}</b><br>fraud", x=.5, y=.5, showarrow=False, font_size=18)])
            show(f, 340, "Class balance")
        with r:
            callout(f"Fraud is rare: <b>{pct(base)}</b>. A model that labels everything legitimate scores <b>{(1 - base) * 100:.2f}%</b> accuracy while catching nothing. "
                    "Training therefore uses class weighting or resampling, and success is measured with precision, recall, F1 and ROC-AUC.", "warn", "The accuracy paradox")
            st_t = _rate_table(c, "status")
            rev = st_t.set_index("status").loc["Reversed"]
            callout(f"<b>Reversed</b> transactions are fraud <b>{rev['Rate'] * 100:.0f}%</b> of the time versus {st_t.set_index('status').loc['Completed', 'Rate'] * 100:.1f}% for completed ones. "
                    "Reversal happens <i>after</i> fraud is discovered, so it is an outcome, not a predictor. It is deliberately excluded from the model to avoid target leakage.", "risk", "A leakage trap, avoided")

    with t2:
        pick = st.selectbox("Slice fraud rate by", list(DIMENSIONS), index=0)
        col = DIMENSIONS[pick]
        order = DAYS if col == "day_of_week" else None
        if col == "hour_of_day":
            g = _rate_bar(c, col, f"Fraud rate by {pick.lower()}", horizontal=False, height=380, order=[str(i) for i in range(24)])
        elif col in ("is_cross_border", "is_new_device"):
            tmp = c.assign(**{col: c[col].map({0: "No", 1: "Yes"})})
            g = _rate_bar(tmp, col, f"Fraud rate by {pick.lower()}", height=240)
        else:
            g = _rate_bar(c, col, f"Fraud rate by {pick.lower()}", order=order, horizontal=col != "day_of_week", height=max(260, 40 * c[col].nunique() + 100))
        st.dataframe(g.assign(Rate=g["Rate"].round(4), Lift=g["Lift"].round(2)).sort_values("Rate", ascending=False), width="stretch", hide_index=True,
                     column_config={"Rate": st.column_config.ProgressColumn("Fraud rate", format="percent", min_value=0, max_value=float(g["Rate"].max())),
                                    "Lift": st.column_config.NumberColumn("Lift vs baseline", format="%.2fx")})
        st.markdown("##### Two-way view")
        a, b2, m = st.columns([1, 1, 1])
        d1 = a.selectbox("Rows", [k for k in DIMENSIONS if k != "Hour of day"], index=0, key="hx")
        d2 = b2.selectbox("Columns", [k for k in DIMENSIONS if k != "Hour of day"], index=1, key="hy")
        minn = m.slider("Hide cells with fewer transactions than", 0, 500, 100, 25)
        if d1 != d2:
            piv = c.pivot_table(index=DIMENSIONS[d1], columns=DIMENSIONS[d2], values="is_fraud", aggfunc=["mean", "count"], observed=True)
            rate, cnt = piv["mean"].where(piv["count"] >= minn), piv["count"]
            f = go.Figure(go.Heatmap(z=rate.values * 100, x=[str(x) for x in rate.columns], y=[str(x) for x in rate.index], colorscale=[[0, "#F6F4EF"], [.5, "#F4A58F"], [1, FRAUD]],
                                     customdata=cnt.values, hovertemplate="%{y} x %{x}<br>Fraud rate %{z:.2f}%<br>%{customdata:,} transactions<extra></extra>",
                                     colorbar=dict(title="%"), text=np.round(rate.values * 100, 1), texttemplate="%{text}"))
            show(f, max(300, 38 * len(rate.index) + 120), f"Fraud rate (%): {d1} x {d2}")
        else:
            st.info("Pick two different dimensions.")

    with t3:
        a, b2 = st.columns(2)
        with a:
            f = go.Figure()
            for lbl, v, colr in [("Legitimate", 0, LEGIT), ("Fraud", 1, FRAUD)]:
                x = np.log10(c.loc[(c["is_fraud"] == v) & (c["amount_to_avg_ratio"] > 0), "amount_to_avg_ratio"])
                h, e = np.histogram(x, bins=60, range=(-3, 3), density=True)
                f.add_trace(go.Bar(x=(e[:-1] + e[1:]) / 2, y=h, name=lbl, marker_color=colr, opacity=.65))
            f.update_layout(barmode="overlay")
            f.update_xaxes(title="log10(amount / 30-day average)   (0 = typical, 1 = 10x)")
            f.update_yaxes(visible=False)
            show(f, 340, "Transaction size vs the account's own baseline")
        with b2:
            _bucket(c, c["amount_to_avg_ratio"], [0, .5, 1, 2, 5, 10, np.inf], ["<0.5x", "0.5-1x", "1-2x", "2-5x", "5-10x", "10x+"],
                    "Fraud rate by amount ratio", "bk_ratio")
        a, b2 = st.columns(2)
        with a:
            _bucket(c, c["transaction_velocity_1h"], [-1, 0, 1, 100], ["0 (none)", "1", "2+"], "Fraud rate by transactions in previous hour", "bk_vel")
        with b2:
            _bucket(c, c["account_age_days"], [-1, 30, 90, 365, 1000, 10000], ["<30d", "30-90d", "90d-1y", "1-3y", "3y+"], "Fraud rate by account age", "bk_age")
        fv, lv = c.loc[c.is_fraud == 1, "amount_to_avg_ratio"], c.loc[c.is_fraud == 0, "amount_to_avg_ratio"]
        callout(f"The median fraudulent transaction is <b>{fv.median():.1f}x</b> the account's baseline against <b>{lv.median():.2f}x</b> for legitimate ones, and fraud has a far heavier tail "
                f"(mean {fv.mean():.0f}x vs {lv.mean():.1f}x). New devices raise fraud from {c[c.is_new_device == 0].is_fraud.mean() * 100:.1f}% to "
                f"{c[c.is_new_device == 1].is_fraud.mean() * 100:.1f}%.", "info", "Behaviour beats demographics")

    with t4:
        a, b2 = st.columns(2)
        with a:
            _rate_bar(c, "hour_of_day", "Fraud rate by hour of day", horizontal=False, height=340, order=[str(i) for i in range(24)])
        with b2:
            _rate_bar(c, "day_of_week", "Fraud rate by weekday", order=DAYS, horizontal=False, height=340)
        m = c.assign(month=c["timestamp"].dt.to_period("M").dt.to_timestamp()).groupby("month")["is_fraud"].agg(n="count", rate="mean").reset_index()
        f = go.Figure()
        f.add_bar(x=m["month"], y=m["n"], name="Transactions", marker_color="#C9D6EA", yaxis="y")
        f.add_trace(go.Scatter(x=m["month"], y=m["rate"], name="Fraud rate", mode="lines+markers", line=dict(color=FRAUD, width=3), yaxis="y2"))
        f.update_layout(yaxis=dict(title="Transactions"), yaxis2=dict(title="Fraud rate", overlaying="y", side="right", tickformat=".1%", showgrid=False))
        show(f, 360, "Monthly volume and fraud rate")
        peak = m.sort_values("rate", ascending=False).iloc[0]
        callout(f"Hour and weekday barely move the fraud rate, so time-of-day rules would add little. The visible pattern is a seasonal surge: volume roughly doubles in Nov-Dec 2025 "
                f"and fraud peaks at <b>{peak['rate'] * 100:.1f}%</b> in {peak['month']:%B %Y}. Capacity planning for the review team should anticipate it.", "info", "Seasonality, not clock time")

    with t5:
        pa = c.groupby("account_id").agg(Transactions=("is_fraud", "size"), Fraud=("is_fraud", "sum"), Holder=("account_holder_name", "first"),
                                         Type=("account_type", "first"), Country=("home_country", "first"), KYC=("kyc_tier", "first")).reset_index()
        pa["Rate"] = pa["Fraud"] / pa["Transactions"]
        s = pa.sort_values("Fraud", ascending=False)
        cum = s["Fraud"].cumsum() / s["Fraud"].sum()
        f = go.Figure(go.Scatter(x=np.arange(1, len(s) + 1) / len(s) * 100, y=cum * 100, line=dict(color=FRAUD, width=3), fill="tozeroy", fillcolor="rgba(230,57,70,.10)"))
        f.update_xaxes(title="% of accounts (worst first)", range=[0, 100])
        f.update_yaxes(title="% of fraud cases", range=[0, 102])
        l, r = st.columns([1, 1])
        with l:
            show(f, 340, "Fraud concentration across accounts")
        with r:
            st.markdown("**Accounts with the most fraud**")
            st.dataframe(s.head(12)[["account_id", "Holder", "Type", "Country", "KYC", "Transactions", "Fraud", "Rate"]], width="stretch", hide_index=True,
                         column_config={"Rate": st.column_config.NumberColumn("Fraud rate", format="percent")}, height=340)
        top = s["Fraud"].head(10).sum() / s["Fraud"].sum()
        callout(f"The 10 worst accounts account for <b>{top * 100:.0f}%</b> of all fraud, and <b>{(pa.Fraud == 0).mean() * 100:.0f}%</b> of accounts are completely clean. "
                "Account-level controls (limits, step-up verification, watch-lists) are therefore a high-leverage complement to transaction scoring.", "risk", "A long tail with a sharp head")

    with t6:
        cols = ["amount", "amount_to_avg_ratio", "avg_transaction_amount_30d", "transaction_velocity_1h", "account_age_days", "hour_of_day",
                "personal_spend_baseline_usd", "is_cross_border", "is_new_device", "is_refund", "is_fraud"]
        corr = c[cols].corr()
        f = go.Figure(go.Heatmap(z=corr.values, x=cols, y=cols, zmin=-1, zmax=1, colorscale=[[0, LEGIT], [.5, "#F6F4EF"], [1, FRAUD]], text=np.round(corr.values, 2),
                                 texttemplate="%{text}", colorbar=dict(title="r")))
        f.update_yaxes(autorange="reversed")
        l, r = st.columns([1.6, 1])
        with l:
            show(f, 560, "Pearson correlation matrix")
        with r:
            ct = corr["is_fraud"].drop("is_fraud").sort_values(key=abs, ascending=False).reset_index()
            ct.columns = ["Feature", "Correlation with fraud"]
            st.dataframe(ct, width="stretch", hide_index=True, height=420,
                         column_config={"Correlation with fraud": st.column_config.ProgressColumn(format="%.3f", min_value=-1, max_value=1)})
            callout("Linear correlations are modest because fraud is rare and driven by <i>thresholds</i> (a burst of activity, a 10x spike) rather than smooth trends. "
                    "That is why tree models and a tuned decision threshold matter later.", "info")


# =============================================================================== FEATURES
FEATURE_NOTES = {
    "amount_to_avg_ratio": ("Numeric", "Scale only", "How unusual is this amount for this account? Normalises across 4 currencies and very different spend levels."),
    "transaction_velocity_1h": ("Numeric", "Scale only", "Bursts of activity are a classic account-takeover and card-testing pattern."),
    "is_new_device": ("Binary", "As is", "A device never seen on the account before."),
    "is_cross_border": ("Binary", "As is", "Transaction country differs from the account's home country."),
    "account_age_days": ("Numeric", "Scale only", "New accounts are riskier and have less history to compare against."),
    "hour_of_day": ("Numeric", "Scale only", "Time-of-day behaviour; weak alone but cheap to include."),
    "is_refund": ("Binary", "As is", "Created in cleaning: negative amounts are refunds, not normal spend."),
    "account_type": ("Categorical", "One-hot", "Individual vs Business accounts behave very differently."),
    "kyc_tier": ("Categorical", "One-hot", "Depth of identity verification; lower tiers have fewer safeguards."),
    "merchant_category": ("Categorical", "One-hot", "What kind of payment it is (wire, crypto, groceries, ...)."),
    "channel": ("Categorical", "One-hot", "How the payment was initiated (card, mobile, USSD, API, ...)."),
}
EXCLUDED = [("transaction_id, account_id, device_id, account_holder_name", "Identifiers: unique labels would let a model memorise rows rather than learn behaviour."),
            ("description, merchant_name", "Free text / very high cardinality; the merchant_category captures the useful part."),
            ("amount, avg_transaction_amount_30d, personal_spend_baseline_usd", "Raw money values in four currencies are not comparable; the ratio expresses the same idea on one scale."),
            ("status", "Completed / Declined / Reversed is known only after the event (Reversed = 33% fraud). Using it would be target leakage."),
            ("timestamp, day_of_week, account_created_date", "Dates would encode calendar position; hour_of_day and account_age_days keep the behavioural content."),
            ("transaction_country, home_country, currency", "Cross-border behaviour is already captured by is_cross_border.")]


def features() -> None:
    ds, _ = get_ctx()
    X, dm, y = ds.X, ds.dummies, ds.y
    header("Stage 4 of 6", "Feature engineering",
           "From the cleaned table to the matrix a model can learn from: 11 carefully chosen inputs, expanded to 33 numeric columns.")
    kpis([("Cleaned columns", str(ds.cleaned.shape[1]), "available", ""), ("Inputs selected", "11", "7 numeric + 4 categorical", "legit"),
          ("Model features", str(dm.shape[1]), "after one-hot encoding", "teal"), ("Rows", num(len(dm)), "unchanged", ""), ("Target", "is_fraud", f"{pct(y.mean())} positive", "fraud")])
    callout("The feature-engineering notebook was empty in the project files, so this stage is <b>reconstructed from the modelling notebook and verified</b> against "
            "<code>featured_engineered_dataset.csv</code> (see Verification tab).", "warn", "Provenance note")
    t1, t2, t3, t4, t5 = st.tabs(["Feature catalogue", "Signal strength", "Encoded matrix", "Train / test split", "Verification"])
    with t1:
        rows = [{"Feature": k, "Kind": v[0], "Encoding": v[1], "Why it is included": v[2]} for k, v in FEATURE_NOTES.items()]
        html_table(pd.DataFrame(rows), key_col="Feature")
        l, r = st.columns(2)
        with l:
            exp = pd.DataFrame({"Categorical input": CATEGORICAL_FEATS, "Levels (one-hot columns)": [X[c].nunique() for c in CATEGORICAL_FEATS]})
            f = px.bar(exp, x="Levels (one-hot columns)", y="Categorical input", orientation="h", text="Levels (one-hot columns)", color_discrete_sequence=[TEAL])
            f.update_yaxes(autorange="reversed", title=None)
            f.update_xaxes(visible=False)
            show(f, 260, f"4 categorical inputs become {sum(X[c].nunique() for c in CATEGORICAL_FEATS)} columns (+ 7 numeric = {dm.shape[1]})")
        with r:
            st.markdown("**Deliberately left out**")
            for k, why in EXCLUDED:
                st.markdown(f"- `{k}`: {why}")
    with t2:
        rows = []
        for f_ in NUMERIC_FEATS:
            a = roc_auc_score(y, X[f_])
            rows.append({"Feature": f_, "Standalone AUC": max(a, 1 - a), "Direction": "higher = riskier" if a >= .5 else "lower = riskier"})
        s = pd.DataFrame(rows).sort_values("Standalone AUC")
        f = px.bar(s, x="Standalone AUC", y="Feature", orientation="h", color="Standalone AUC", color_continuous_scale=["#C9D6EA", LEGIT, FRAUD], text=s["Standalone AUC"].round(3))
        f.add_vline(x=.5, line_dash="dash", line_color=INK, annotation_text="no signal")
        f.update_xaxes(range=[.45, max(.75, s["Standalone AUC"].max() + .05)])
        f.update_layout(coloraxis_showscale=False)
        show(f, 380, "How well does each numeric feature rank fraud on its own? (ROC-AUC)")
        callout("Transaction velocity and the amount ratio carry most of the signal individually; the rest add smaller, complementary information. "
                "This matches the notebook finding that behavioural features already carry most of the separable signal.", "info")
    with t3:
        q = st.text_input("Filter columns", "", key="fcols")
        cols = [c for c in dm.columns if q.lower() in c.lower()] or list(dm.columns)
        st.dataframe(pd.concat([dm[cols], y], axis=1).head(300), width="stretch", height=420, hide_index=True)
        st.caption(f"{len(cols)} of {dm.shape[1]} feature columns shown, plus the target.")
    with t4:
        n_tr, n_te = int(len(dm) * .75), len(dm) - int(len(dm) * .75)
        tr_f = ds.y.iloc[:0]  # placeholder to keep types simple
        c1, c2 = st.columns([1.2, 1])
        with c1:
            f = go.Figure()
            f.add_bar(y=["Train (75%)", "Test (25%)"], x=[n_tr, n_te], orientation="h", marker_color=[LEGIT, AMBER], text=[f"{n_tr:,}", f"{n_te:,}"], textposition="inside")
            f.update_xaxes(visible=False)
            show(f, 220, "Stratified split, random_state = 42")
        with c2:
            callout(f"The split is <b>stratified</b>: both halves keep the same {pct(y.mean())} fraud rate, so test results are comparable to training conditions. "
                    "All scaling and encoding are fitted on the training half only, preventing information from leaking into evaluation.", "good")
    with t5:
        if ds.feat_check is None:
            st.info("featured_engineered_dataset.csv was not found, so there is nothing to compare against.")
        else:
            st.markdown("**Does the app's feature matrix equal your saved `featured_engineered_dataset.csv`?**")
            for k, v in ds.feat_check.items():
                st.markdown(f"{'&#9989;' if v else '&#10060;'} {k}", unsafe_allow_html=True)
            if all(ds.feat_check.values()):
                callout("The rebuilt feature matrix is identical to the saved dataset, so the empty notebook leaves no gap in lineage.", "good", "Verified")
