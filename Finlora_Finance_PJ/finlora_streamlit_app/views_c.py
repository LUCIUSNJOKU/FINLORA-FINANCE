"""Pages: Model development, Evaluation, Threshold studio, Drivers."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score, roc_curve

import models as M
from ctx import get_ctx
from data import CATEGORICAL_FEATS
from theme import AMBER, FRAUD, INK, LEGIT, MUTED, TEAL, callout, header, kpis, num, pct, section, show

PALETTE = {"Logistic Regression": LEGIT, "Random Forest": FRAUD, "Random Forest (tuned)": "#B4232F", "Gradient Boosting": AMBER,
           "Random Forest + SMOTE": "#7A5CFA", "Ensemble (LR + tuned RF)": TEAL, "Random Forest (calibrated)": "#8A99A8"}


def _leaderboard(b: dict) -> pd.DataFrame:
    rows = []
    for n, r in b["results"].items():
        d, t = r["default"], r["tuned"]
        rows.append({"Model": n, "ROC-AUC": r["auc"], "Avg precision": r["ap"], "Precision @0.5": d["precision"], "Recall @0.5": d["recall"], "F1 @0.5": d["f1"],
                     "Best threshold": t["threshold"], "Precision @best": t["precision"], "Recall @best": t["recall"], "F1 @best": t["f1"]})
    return pd.DataFrame(rows).sort_values("F1 @best", ascending=False).reset_index(drop=True)


# =============================================================================== MODELS
def modelling() -> None:
    ds, b = get_ctx()
    header("Stage 5 of 6", "Model development",
           "Two required models (Logistic Regression and Random Forest) plus five techniques tried to improve on them, all on one identical train/test split.")
    kpis([("Training set", num(b["train_size"]), f"{pct(b['train_fraud'])} fraud", "legit"), ("Test set", num(b["test_size"]), f"{pct(b['test_fraud'])} fraud", "amber"),
          ("Model features", str(ds.dummies.shape[1]), "7 scaled + 26 one-hot", "teal"), ("Models compared", str(len(b["results"])), "same split, same seed", ""),
          ("In production", "Logistic Regression", f"threshold {b['prod_threshold']:.3f}", "fraud")])
    st.write("")
    t1, t2, t3 = st.tabs(["Benchmark", "Model cards", "Experiment sandbox"])
    with t1:
        lb = _leaderboard(b)
        f = go.Figure()
        f.add_bar(name="F1 at default 0.5 threshold", x=lb["Model"], y=lb["F1 @0.5"], marker_color="#C9D6EA")
        f.add_bar(name="F1 at tuned threshold", x=lb["Model"], y=lb["F1 @best"], marker_color=LEGIT, text=lb["F1 @best"].round(3), textposition="outside")
        f.update_layout(barmode="group", xaxis_tickangle=-20)
        f.update_yaxes(range=[0, .85], title="F1")
        show(f, 400, "Tuning the decision threshold moves F1 far more than changing algorithm")
        gain = lb["F1 @best"].max() - lb["F1 @0.5"].max()
        spread = lb["ROC-AUC"].max() - lb["ROC-AUC"].min()
        callout(f"Across all seven approaches the ROC-AUC spread is only <b>{spread:.3f}</b> ({lb['ROC-AUC'].min():.3f} to {lb['ROC-AUC'].max():.3f}), so the signal is in the engineered features, not the algorithm. "
                f"Moving off the default 0.5 cut-off lifts the best F1 by <b>{gain:.2f}</b>, which is worth more than any model swap. "
                f"The simple, fast, explainable <b>{M.PROD}</b> matches or beats every alternative on F1, so it goes to production.", "good", "The verdict")
        view = lb[["Model", "ROC-AUC", "Avg precision", "F1 @0.5", "Best threshold", "Precision @best", "Recall @best", "F1 @best"]]
        st.dataframe(view, width="stretch", hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in view.columns if c != "Model"})
        st.download_button("Download benchmark (CSV)", lb.to_csv(index=False).encode(), "finlora_model_benchmark.csv", "text/csv")
    with t2:
        cols = st.columns(2)
        for i, (n, (desc, tech)) in enumerate(M.MODEL_INFO.items()):
            m = b["meta"][n]
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"**{n}**" + ("  &nbsp; :red-badge[PRODUCTION]" if n == M.PROD else ""))
                    st.caption(desc)
                    a, c_, d = st.columns(3)
                    a.metric("ROC-AUC", f"{m['auc']:.3f}")
                    c_.metric("Avg precision", f"{m['ap']:.3f}")
                    d.metric("Train time", f"{m['seconds']:.0f}s" if m["seconds"] else "n/a")
                    st.caption(f"Imbalance handling: {tech}")
        if st.button("Retrain all models from scratch", help="Re-runs the whole benchmark (about 2-4 minutes)."):
            with st.status("Retraining", expanded=True) as s:
                bar = st.progress(0.0)
                M.get_bundle(ds, lambda f_, l: bar.progress(min(f_, 1.0), text=l), force=True)
                s.update(label="Done", state="complete")
            st.rerun()
    with t3:
        st.markdown("Train your own variant on the exact same split and compare it with production, live.")
        a, c_ = st.columns([1, 2])
        algo = a.selectbox("Algorithm", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
        with c_:
            if algo == "Logistic Regression":
                C = st.select_slider("Regularisation strength C (higher = weaker regularisation)", [.001, .01, .1, 1, 10, 100], value=1)
                bal = st.checkbox("Balance class weights", True)
                make = lambda: M.make_lr(C, bal)
            elif algo == "Random Forest":
                x1, x2, x3 = st.columns(3)
                n_, d_, l_ = x1.slider("Trees", 20, 150, 60, 10), x2.slider("Max depth", 3, 20, 10), x3.slider("Min samples / leaf", 1, 30, 5)
                make = lambda: M.make_rf(n_, d_, l_)
            else:
                x1, x2, x3 = st.columns(3)
                it, d_, lr_ = x1.slider("Iterations", 50, 300, 100, 50), x2.slider("Max depth", 2, 12, 6), x3.select_slider("Learning rate", [.01, .03, .05, .1, .2], value=.05)
                make = lambda: M.make_hgb(it, d_, lr_)
        if st.button("Train and evaluate", type="primary"):
            Xtr, Xte, ytr, yte = M._split(ds)
            with st.spinner("Training..."):
                t0 = time.time()
                mdl = make().fit(Xtr, ytr)
                p = mdl.predict_proba(Xte)[:, 1]
            t_best, _ = M.best_f1_threshold(yte, p)
            st.session_state["sandbox"] = {"algo": algo, "m": M.metrics_at(yte, p, t_best), "auc": float(roc_auc_score(yte, p)),
                                           "sec": time.time() - t0}
        sb = st.session_state.get("sandbox")
        if sb:
            prod = b["results"][M.PROD]["tuned"]
            k = [("Your model", sb["algo"], f"trained in {sb['sec']:.0f}s", "legit"),
                 ("F1 (tuned threshold)", f"{sb['m']['f1']:.3f}", f"production {prod['f1']:.3f}", "teal" if sb["m"]["f1"] >= prod["f1"] else "fraud"),
                 ("Precision", f"{sb['m']['precision']:.3f}", f"production {prod['precision']:.3f}", ""),
                 ("Recall", f"{sb['m']['recall']:.3f}", f"production {prod['recall']:.3f}", ""),
                 ("ROC-AUC", f"{sb['auc']:.3f}", f"production {b['results'][M.PROD]['auc']:.3f}", "")]
            kpis(k)


# =============================================================================== EVALUATION
def evaluation() -> None:
    ds, b = get_ctx()
    y, P = b["y_test"].values, b["probas"]
    header("Stage 5 of 6", "Model evaluation",
           f"Everything here is measured on {b['test_size']:,} held-out transactions ({int(y.sum())} fraud) that no model saw during training.")
    names = list(P)
    c1, c2 = st.columns([2, 1])
    sel = c1.multiselect("Models to compare", names, default=[M.PROD, "Random Forest"])
    mode = c2.radio("Operating point", ["Default (0.5)", "Tuned for best F1"], horizontal=True)
    if not sel:
        st.info("Select at least one model.")
        return
    key = "default" if mode.startswith("Default") else "tuned"
    rows = []
    for n in sel:
        m = b["results"][n][key]
        rows.append({"Model": n, "Threshold": m["threshold"], "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"], "Accuracy": m["accuracy"],
                     "ROC-AUC": b["results"][n]["auc"], "Fraud caught": m["tp"], "Fraud missed": m["fn"], "False alarms": m["fp"]})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Threshold", "Precision", "Recall", "F1", "Accuracy", "ROC-AUC"]})
    t1, t2, t3, t4 = st.tabs(["ROC and precision-recall", "Confusion matrices", "Gains and score separation", "Classification report"])
    with t1:
        a, c_ = st.columns(2)
        fr, fp_ = go.Figure(), go.Figure()
        for n in sel:
            fpr, tpr, _ = roc_curve(y, P[n])
            fr.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{n} (AUC {b['results'][n]['auc']:.3f})", line=dict(color=PALETTE[n], width=3)))
            pr, rc, _ = precision_recall_curve(y, P[n])
            fp_.add_trace(go.Scatter(x=rc, y=pr, name=n, line=dict(color=PALETTE[n], width=3)))
        fr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=MUTED, dash="dash"), name="Random guess"))
        fp_.add_hline(y=y.mean(), line_dash="dash", line_color=MUTED, annotation_text="no-skill baseline")
        fr.update_xaxes(title="False positive rate")
        fr.update_yaxes(title="True positive rate")
        fp_.update_xaxes(title="Recall")
        fp_.update_yaxes(title="Precision")
        with a:
            show(fr, 420, "ROC curve")
        with c_:
            show(fp_, 420, "Precision-recall curve")
        callout("The ROC curves of all models almost overlap, which is why ROC-AUC barely discriminates between them. The precision-recall view, which is more honest for rare events, "
                "shows where the real trade-offs are.", "info")
    with t2:
        cols = st.columns(min(len(sel), 3))
        for i, n in enumerate(sel):
            m = b["results"][n][key]
            z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
            f = go.Figure(go.Heatmap(z=[[0, 1], [1, 0]], x=["Predicted legit", "Predicted fraud"], y=["Actual legit", "Actual fraud"], showscale=False,
                                     colorscale=[[0, "#E8EEF8"], [1, "#FBE0E3"]], text=[[f"{v:,}" for v in r] for r in z], texttemplate="<b>%{text}</b>", textfont_size=20,
                                     hoverinfo="skip"))
            f.update_yaxes(autorange="reversed")
            with cols[i % len(cols)]:
                show(f, 300, f"{n} @ {m['threshold']:.3f}", key=f"cm_{n}_{key}")
                st.caption(f"Catches {m['tp']:,} of {m['tp'] + m['fn']:,} fraud with {m['fp']:,} false alarms.")
    with t3:
        a, c_ = st.columns(2)
        order = np.argsort(-P[M.PROD])
        cum = np.cumsum(y[order]) / y.sum()
        x = np.arange(1, len(y) + 1) / len(y)
        f = go.Figure()
        for n in sel:
            o = np.argsort(-P[n])
            f.add_trace(go.Scatter(x=x[::60], y=(np.cumsum(y[o]) / y.sum())[::60], name=n, line=dict(color=PALETTE[n], width=3)))
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random review", line=dict(color=MUTED, dash="dash")))
        f.update_xaxes(title="Share of transactions reviewed (highest score first)", tickformat=".0%")
        f.update_yaxes(title="Share of fraud caught", tickformat=".0%")
        with a:
            show(f, 400, "Cumulative gains")
        with c_:
            pick = st.selectbox("Score distribution for", sel, key="sd")
            f = go.Figure()
            for lbl, v, colr in [("Legitimate", 0, LEGIT), ("Fraud", 1, FRAUD)]:
                h, e = np.histogram(P[pick][y == v], bins=40, range=(0, 1))
                f.add_trace(go.Bar(x=(e[:-1] + e[1:]) / 2, y=h, name=lbl, marker_color=colr, opacity=.75))
            f.update_layout(barmode="overlay")
            f.update_yaxes(type="log", title="Transactions (log scale)")
            f.add_vline(x=b["results"][pick][key]["threshold"], line_dash="dash", line_color=INK)
            f.update_xaxes(title="Risk score")
            show(f, 400, f"{pick}: how well do scores separate the classes?")
        top5 = int(len(y) * .05)
        o = np.argsort(-P[M.PROD])[:top5]
        callout(f"Reviewing only the top <b>5%</b> of transactions by {M.PROD} score ({top5:,} items) surfaces <b>{y[o].sum() / y.sum() * 100:.0f}%</b> of all fraud. "
                "That is the lever for sizing the investigation team.", "good", "Gains in plain terms")
    with t4:
        n = st.selectbox("Model", sel, key="cr")
        m = b["results"][n][key]
        rep = classification_report(y, P[n] >= m["threshold"], target_names=["Legitimate", "Fraud"], output_dict=True, digits=3)
        st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")


# =============================================================================== THRESHOLD STUDIO
def threshold_studio() -> None:
    ds, b = get_ctx()
    y, P = b["y_test"].values, b["probas"]
    header("Stage 5 of 6", "Threshold studio",
           "A model produces a risk score; the threshold decides which scores become alerts. That is a business choice about review capacity and risk appetite, not a technical constant.")
    names = list(P)
    if "thr_w" not in st.session_state:  # widget state is dropped when leaving the page, so re-seed from a plain key
        st.session_state["thr_w"] = float(st.session_state.get("_thr", b["prod_threshold"]))
    l, r = st.columns([1, 2])
    model = l.selectbox("Model", names, index=names.index(M.PROD), key="thr_model")
    best_t = b["results"][model]["tuned"]["threshold"]
    l.button(f"Jump to F1-optimal ({best_t:.3f})", on_click=lambda: st.session_state.__setitem__("thr_w", float(best_t)))
    thr = r.slider("Alert threshold (flag when risk score is at or above)", 0.02, 0.99, step=0.005, key="thr_w", format="%.3f")
    st.session_state["_thr"] = float(thr)
    m = M.metrics_at(y, P[model], thr)
    n = len(y)
    kpis([("Precision", pct(m["precision"], 1), "alerts that are real fraud", "amber"), ("Recall", pct(m["recall"], 1), "fraud that gets caught", "teal"), ("F1 score", f"{m['f1']:.3f}", "balance of both", "legit"),
          ("Alerts per 1,000 txns", f"{m['alerts'] / n * 1000:.1f}", f"{m['alerts']:,} in the test set", ""), ("Fraud missed", num(m["fn"]), f"of {m['tp'] + m['fn']:,}", "fraud")])
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Trade-off curve", "Recall targets", "Cost view", "Is the threshold trustworthy?"])
    sw = M.sweep(y, P[model])
    with t1:
        f = go.Figure()
        for col, nm, colr in [("precision", "Precision", AMBER), ("recall", "Recall", TEAL), ("f1", "F1", LEGIT)]:
            f.add_trace(go.Scatter(x=sw["threshold"], y=sw[col], name=nm, line=dict(color=colr, width=3)))
        f.add_vline(x=thr, line_color=FRAUD, line_dash="dash", annotation_text=f"selected {thr:.3f}")
        f.update_xaxes(title="Threshold")
        f.update_yaxes(tickformat=".0%")
        a, c_ = st.columns([1.7, 1])
        with a:
            show(f, 400, f"{model}: precision, recall and F1 by threshold")
        with c_:
            z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
            fh = go.Figure(go.Heatmap(z=[[0, 1], [1, 0]], x=["Pred. legit", "Pred. fraud"], y=["Actual legit", "Actual fraud"], showscale=False, colorscale=[[0, "#E8EEF8"], [1, "#FBE0E3"]],
                                      text=[[f"{v:,}" for v in rr] for rr in z], texttemplate="<b>%{text}</b>", textfont_size=18, hoverinfo="skip"))
            fh.update_yaxes(autorange="reversed")
            show(fh, 400, "Outcome at this threshold")
        if st.button("Use this threshold for the review queue and live scorer", type="primary"):
            st.session_state["_queue_thr"] = float(thr)
            st.success(f"Review queue and live scorer will now flag scores of {thr:.3f} or higher.")
    with t2:
        tbl = M.precision_at_recall(y, P[model])
        st.dataframe(tbl, width="stretch", hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in tbl.columns if c != "Recall target"})
        if len(tbl):
            r70 = tbl.iloc[0]
            callout(f"To catch <b>70%</b> of fraud the model's alerts are about <b>{r70['Precision'] * 100:.0f}%</b> accurate. Pushing recall to <b>90%</b> catches nearly everything but precision falls to "
                    f"<b>{tbl.iloc[-1]['Precision'] * 100:.0f}%</b>, i.e. roughly {1 / tbl.iloc[-1]['Precision']:.0f} alerts reviewed per fraud found. The right point depends on how many alerts the team can review.", "info")
    with t3:
        a, c_, d = st.columns(3)
        loss = a.number_input("Average loss per missed fraud", 10, 100000, 500, 50)
        rev = c_.number_input("Cost to review one alert", 0.5, 1000.0, 8.0, 0.5)
        d.caption("Illustrative inputs. Replace with Finlora's real chargeback loss and analyst cost.")
        sw["cost"] = sw["fn"] * loss + sw["alerts"] * rev
        oi = int(sw["cost"].idxmin())
        none_cost, cur = (m["tp"] + m["fn"]) * loss, m["fn"] * loss + m["alerts"] * rev
        f = go.Figure(go.Scatter(x=sw["threshold"], y=sw["cost"], line=dict(color=INK, width=3), fill="tozeroy", fillcolor="rgba(11,27,43,.06)"))
        f.add_vline(x=thr, line_color=FRAUD, line_dash="dash", annotation_text="selected")
        f.add_trace(go.Scatter(x=[sw.loc[oi, "threshold"]], y=[sw.loc[oi, "cost"]], mode="markers", marker=dict(size=14, color=TEAL), name="Cost-optimal"))
        f.update_xaxes(title="Threshold")
        f.update_yaxes(title="Expected total cost on the test set")
        show(f, 380, "Total cost = missed fraud losses + review effort")
        kpis([("Cost-optimal threshold", f"{sw.loc[oi, 'threshold']:.2f}", f"total {sw.loc[oi, 'cost']:,.0f}", "teal"), ("Your selected threshold", f"{thr:.3f}", f"total {cur:,.0f}", "fraud"),
              ("With no model at all", f"{none_cost:,.0f}", "every fraud goes unseen", ""), ("Saving vs no model", pct(1 - cur / none_cost, 0), "at the selected threshold", "amber")])
    with t4:
        h = b["honest"]
        pt = b["results"][M.PROD]["tuned"]
        callout("The F1-optimal threshold above was found by scanning the <i>test</i> set, which makes results slightly optimistic. As a check, the threshold was chosen again using only "
                "<i>out-of-fold predictions on the training data</i>, then applied once to the test set.", "warn", "Honest threshold check (Logistic Regression)")
        kpis([("Threshold from test scan", f"{pt['threshold']:.3f}", f"test F1 {pt['f1']:.3f}", "legit"), ("Threshold from training only", f"{h['threshold']:.3f}", f"test F1 {h['f1']:.3f}", "teal"),
              ("Precision / recall", f"{h['precision']:.2f} / {h['recall']:.2f}", "with the training-only threshold", ""), ("Performance given up", f"{(pt['f1'] - h['f1']) * 100:.1f} pts", "F1, in exchange for honesty", "amber")])
        callout(f"Choosing the threshold without peeking costs only <b>{(pt['f1'] - h['f1']) * 100:.1f} F1 points</b>. The production threshold is not an artefact of tuning on test data.", "good")


# =============================================================================== DRIVERS
def _group(name: str) -> str:
    for c in CATEGORICAL_FEATS:
        if name.startswith(c + "_") or name.startswith("cat__" + c + "_"):
            return c
    return name.replace("num__", "")


def drivers() -> None:
    ds, b = get_ctx()
    header("Stage 5 of 6", "What drives fraud scores",
           "Two complementary views: Logistic Regression coefficients show direction (does it raise or lower risk), Random Forest importances show overall weight.")
    lr = b["lr"]
    names = [n.split("__", 1)[1] for n in lr.named_steps["p"].get_feature_names_out()]
    coef = pd.Series(lr.named_steps["c"].coef_[0], index=names)
    rf = b["rf_importance"]
    pretty = lambda s: s.replace("_", " ")
    t1, t2, t3 = st.tabs(["By feature group", "Logistic Regression coefficients", "Random Forest importance"])
    with t1:
        X = ds.X.loc[b["test_index"]]
        _, cdf, _ = M.score_and_explain(b, X)
        lr_g = cdf.abs().mean().sort_values(ascending=False)
        rf_g = rf.groupby([_group(n) for n in rf.index]).sum().sort_values(ascending=False)
        f = go.Figure()
        order = list(lr_g.index)
        f.add_bar(name="Logistic Regression (share of avg |contribution|)", x=[lr_g[k] / lr_g.sum() for k in order], y=[pretty(k) for k in order], orientation="h", marker_color=LEGIT)
        f.add_bar(name="Random Forest (share of importance)", x=[rf_g.get(k, 0) / rf_g.sum() for k in order], y=[pretty(k) for k in order], orientation="h", marker_color=FRAUD)
        f.update_layout(barmode="group")
        f.update_yaxes(autorange="reversed")
        f.update_xaxes(tickformat=".0%")
        show(f, 520, "Weight of each original input (categoricals combined)")
        callout("Both algorithms agree on the ranking at the top: <b>transaction velocity</b> and the <b>amount-to-baseline ratio</b> dominate, with merchant category next. "
                "Agreement between a linear and a tree model is strong evidence that these are real behavioural drivers rather than artefacts of one algorithm.", "good", "Consistent story across models")
    with t2:
        top = coef.reindex(coef.abs().sort_values(ascending=False).head(15).index).iloc[::-1]
        f = go.Figure(go.Bar(x=top.values, y=[pretty(i) for i in top.index], orientation="h", marker_color=[FRAUD if v > 0 else LEGIT for v in top.values],
                             text=[f"{v:+.2f}" for v in top.values], textposition="outside"))
        f.update_xaxes(title="Coefficient on standardised inputs (red raises fraud odds, blue lowers them)", zeroline=True)
        show(f, 520, "Top 15 Logistic Regression coefficients")
    with t3:
        top = rf.sort_values(ascending=False).head(15).iloc[::-1]
        f = go.Figure(go.Bar(x=top.values, y=[pretty(i) for i in top.index], orientation="h", marker_color=FRAUD, text=[f"{v:.3f}" for v in top.values], textposition="outside"))
        f.update_xaxes(title="Mean decrease in impurity")
        show(f, 520, "Top 15 Random Forest importances")
        st.caption("Random Forest importances show weight but not direction; use the coefficient view to see which way a feature pushes risk.")
