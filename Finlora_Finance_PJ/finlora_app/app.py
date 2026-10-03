"""Finlora Fraud Detection - stakeholder workbench.   Run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Finlora Fraud Detection", page_icon=":material/shield:", layout="wide", initial_sidebar_state="expanded")

import data  # noqa: E402
import theme  # noqa: E402
import views_a, views_b, views_c, views_d  # noqa: E402,E401

theme.inject()

PAGES = {
    "Brief": [("home", views_a.overview, "Overview", ":material/dashboard:")],
    "Data pipeline": [("data", views_a.sources, "1 - Load data", ":material/database:"),
                      ("clean", views_a.cleaning, "2 - Clean", ":material/cleaning_services:"),
                      ("eda", views_b.eda, "3 - Explore", ":material/query_stats:"),
                      ("feat", views_b.features, "4 - Engineer features", ":material/tune:")],
    "Modelling": [("model", views_c.modelling, "5a - Models", ":material/model_training:"),
                  ("eval", views_c.evaluation, "5b - Evaluation", ":material/fact_check:"),
                  ("thr", views_c.threshold_studio, "5c - Threshold studio", ":material/ads_click:"),
                  ("drivers", views_c.drivers, "5d - Drivers", ":material/insights:")],
    "Operations": [("queue", views_d.review_queue, "6a - Review queue", ":material/list_alt:"),
                   ("scorer", views_d.live_scorer, "6b - Live scorer", ":material/speed:")],
    "Handover": [("wrap", views_d.handover, "Summary and next steps", ":material/flag:")],
}

registry, nav_spec = {}, {}
for section_name, items in PAGES.items():
    nav_spec[section_name] = []
    for key, fn, title, icon in items:
        pg = st.Page(fn, title=title, icon=icon, url_path=key, default=(key == "home"))
        registry[key] = pg
        nav_spec[section_name].append(pg)
st.session_state["_pages"] = registry

page = st.navigation(nav_spec, position="hidden")  # custom nav below, so the brand can sit on top

with st.sidebar:
    st.markdown('<div class="fl-brand">Finlora</div><div class="fl-brand-sub">Fraud detection and scoring workbench</div>', unsafe_allow_html=True)
    for section_name, pgs in nav_spec.items():
        st.caption(section_name)
        for pg in pgs:
            st.page_link(pg)
    st.divider()
    d = data.find_data_dir()
    if d:
        st.caption(f"Data folder: `{d}`")
        with st.expander("Change data folder"):
            p = st.text_input("Folder path", value=str(d), label_visibility="collapsed")
            if st.button("Use this folder") and p != str(d):
                st.session_state["data_dir_override"] = p
                st.rerun()
    else:
        st.caption("No data connected yet.")

page.run()
