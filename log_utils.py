# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: dp-app
#     language: python
#     name: dp-app
# ---

import pandas as pd
from datetime import datetime
import streamlit as st

# +
LOG_FILE = "GCDapp_usage_log.csv"

# --- Cached logs store ---
@st.cache_data(ttl=None)
def get_logs():
    # start with empty list
    return []

def log_usage(user: str, file_name: str, status: str):
    logs = get_logs()
    logs.append([datetime.now().isoformat(), user, file_name, status])
    # Update cache
    st.cache_data.clear()  # clear to allow refresh
    @st.cache_data(ttl=None)
    def get_logs():
        return logs
    return logs
