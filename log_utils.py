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

import csv
from datetime import datetime
import os

# +
LOG_FILE = "GCDapp_usage_log.csv"

def log_usage(user_name: str, filename: str, status: str):
    """
    Append a log entry to usage_log.csv
    
    Args:
        user_name (str): Name entered by the user
        filename (str): Name of file processed
        status (str): "success" or error message
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True) if os.path.dirname(LOG_FILE) else None

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            user_name,
            filename,
            status
        ])
