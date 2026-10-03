"""Shared page context: locate data, load datasets, make sure models are trained."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import data
import models
from theme import callout, header


def setup_screen() -> None:
    header("Setup", "Connect the Finlora data",
           "The app needs the two raw files. Everything else (cleaned data, engineered features, models) is rebuilt live from them.")
    st.markdown("**Option 1 - point to a folder** that contains `finlora_transactions.csv` and `finlora_accounts.csv`.")
    p = st.text_input("Data folder", value=st.session_state.get("data_dir_override", ""), placeholder="e.g. C:/Users/you/Finlora/Dataset/Raw_data")
    if p:
        st.session_state["data_dir_override"] = p
        if data.find_data_dir():
            st.rerun()
        st.warning("Those two files were not found in that folder.")
    st.markdown("**Option 2 - upload the files** (saved into the app's `data/` folder).")
    c1, c2 = st.columns(2)
    up_t = c1.file_uploader("finlora_transactions.csv", type="csv")
    up_a = c2.file_uploader("finlora_accounts.csv", type="csv")
    if up_t and up_a:
        dest = data.APP_DIR / "data"
        dest.mkdir(exist_ok=True)
        (dest / data.RAW_FILES["transactions"]).write_bytes(up_t.getvalue())
        (dest / data.RAW_FILES["accounts"]).write_bytes(up_a.getvalue())
        st.rerun()
    callout("Optional: also drop <code>cleaned_finlora_data.csv</code> and <code>featured_engineered_dataset.csv</code> in the same folder "
            "and the app will verify that its live pipeline reproduces them exactly.", "info")


def get_ctx():
    """Return (datasets, model bundle). Shows setup screen / first-run training UI when needed."""
    ds = data.get_datasets()
    if ds is None:
        setup_screen()
        st.stop()
    if not models.bundle_current(ds):
        with st.status("First run: training the 7 models (about 2-4 minutes, one time only)", expanded=True) as status:
            bar = st.progress(0.0, text="Starting")
            bundle = models.get_bundle(ds, lambda f, label: bar.progress(min(f, 1.0), text=label))
            status.update(label="Models trained and saved", state="complete", expanded=False)
    else:
        bundle = models.get_bundle(ds)
    return ds, bundle
