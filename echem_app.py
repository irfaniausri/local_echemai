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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import streamlit as st
import pandas as pd
import os, sys

# Get base dir of current notebook
this_file = __file__ if "__file__" in globals() else os.path.abspath("")
notebook_dir = os.path.dirname(this_file)

project_root = os.path.abspath(os.path.join(notebook_dir, ".."))
# print(f"project_root: {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Codes.local_echemai import GCD_functions as functions

# +
st.title("🧪 GCD Processor")

file = st.file_uploader("Upload CSV", type="csv")

if file:
    df = pd.read_csv(file, header=0, skiprows=[1])
    st.write("📥 Input Data", df)

    # Run your processing function
    cap_df = functions.analyze_gcd_df(df, current=20, has_metadata_row=True)
    
    fig1 = functions.plot_capacitance_vs_cycle(cap_df, y_min=400, y_max=2000)
    st.pyplot(fig1, clear_figure=True)

    fig2 = functions.plot_retention_vs_cycle(cap_df)
    st.pyplot(fig2, clear_figure=True)
    
    # Allow download
    csv = cap_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Result", csv, "processed.csv", "text/csv")
