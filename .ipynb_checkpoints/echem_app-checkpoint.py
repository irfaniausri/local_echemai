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

# +
import streamlit as st
import pandas as pd
import os, sys, io, zipfile

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

# file = st.file_uploader("Upload CSV", type="csv")
uploaded_files = st.file_uploader("Upload GCD files", type=["csv", "txt"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        base_name = uploaded_file.name.split(".")[0]

        results = functions.analyze_gcd(uploaded_file, current=0.01,  output_path=None, base_name=base_name, 
                              mode="app", plot_base_data=True, plot_debug=False)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            # Add capacitance CSV
            cap_df = results.get("capacitance", None)
            if cap_df is not None and not cap_df.empty:
                csv_buffer = io.StringIO()
                cap_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                zf.writestr(f"{base_name}_capacitance.csv", csv_buffer.getvalue())

            # Add warnings TXT
            warnings = results.get("warnings", [])
            if warnings:
                zf.writestr(f"{base_name}_warnings.txt", "\n".join(warnings))

            # Add figures JPG
            figs = results.get("figs", {})
            for key, fig in figs.items():
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format="jpg", dpi=300)
                img_buffer.seek(0)
                zf.writestr(f"{base_name}_{key}.jpg", img_buffer.read())

        zip_buffer.seek(0)
        st.download_button(
            label=f"Download All Results ({base_name})",
            data=zip_buffer,
            file_name=f"{base_name}_results.zip",
            mime="application/zip"
        )

# if file:
#     df = pd.read_csv(file, header=0, skiprows=[1])
#     st.write("📥 Input Data", df)

#     # Run your processing function
#     # cap_df = functions.analyze_gcd_df(df, current=20, has_metadata_row=True)
#     cap_df = functions.analyze_gcd(input_file, current, output_path, base_name=None, 
#                               mode="app", plot_base_data=True, plot_debug=False)
    
#     # fig1 = functions.plot_capacitance_vs_cycle(cap_df, y_min=400, y_max=2000)
#     # st.pyplot(fig1, clear_figure=True)

#     # fig2 = functions.plot_retention_vs_cycle(cap_df)
#     # st.pyplot(fig2, clear_figure=True)
    
#     # Allow download
#     # csv = cap_df.to_csv(index=False).encode("utf-8")
#     # st.download_button("Download Result", csv, "processed.csv", "text/csv")
