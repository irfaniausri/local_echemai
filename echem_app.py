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
from datetime import datetime

# Get base dir of current notebook
this_file = __file__ if "__file__" in globals() else os.path.abspath("")
notebook_dir = os.path.dirname(this_file)

project_root = os.path.abspath(os.path.join(notebook_dir, ".."))
# print(f"project_root: {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from local_echemai import GCD_functions as functions


# +
# --- Shared log storage ---
@st.cache_data(ttl=None)
def get_logs():
    """Return the shared usage logs (list of lists)."""
    return []

def log_usage(user, file_name, status):
    """Append a new log entry to the shared logs."""
    logs = get_logs()
    logs.append([datetime.now().isoformat(), user, file_name, status])
    update_logs(logs)

def update_logs(new_logs):
    """Overwrite the shared logs in cache with a new list."""
    get_logs.clear()         # clear only this cached function
    get_logs.set(new_logs)   # save updated logs


# +
st.title("🧪 GCD Processor")

# --- Step 1: User input ---
user_name = st.text_input("Enter your name (required):")

# --- Step 2: File upload ---
uploaded_files = st.file_uploader("Upload GCD files", type=["csv", "txt"], accept_multiple_files=True)

if  user_name and uploaded_files:
    for uploaded_file in uploaded_files:
        base_name = uploaded_file.name.split(".")[0]
        try:
            results = functions.analyze_gcd(uploaded_file, current=0.01,  output_path=None, base_name=base_name, 
                                            mode="app", plot_base_data=True, plot_debug=False)

            # ✅ Log success
            log_usage(user_name, uploaded_file.name, "success")

            # --- Package results into ZIP ---
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
            
        except Exception as e:
            st.error(f"❌ Failed {uploaded_file.name}: {e}")
            log_usage(user_name, uploaded_file.name, f"failed: {e}")

# --- Step 3: Hidden Admin Section ---
query_params = st.experimental_get_query_params()
if query_params.get("admin") == ["1"]:  # Only visible with ?admin=1 in URL
    st.subheader("Admin Section")
    if st.checkbox("📥 Show usage logs"):
        logs = get_logs()
        if logs:
            df = pd.DataFrame(logs, columns=["timestamp", "user", "file", "status"])
            st.dataframe(df)
            st.download_button(
                "Download logs.csv",
                df.to_csv(index=False),
                "usage_log.csv",
                mime="text/csv"
            )
        else:
            st.write("No logs yet.")
