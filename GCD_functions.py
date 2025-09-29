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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from IPython.display import display

from scipy.signal import savgol_filter, find_peaks, correlate
from scipy.stats import linregress
from datetime import datetime

import sys
print(sys.executable)


# +
# 1. Load CSV file
def load_voltage_data(filepath):
    """
    Smart loader for GCD data (.txt or .csv) that:
    - Skips metadata blocks (in .txt)
    - Finds the line with 'Time/sec'
    - Cleans and averages duplicate timestamps
    """
    ext = os.path.splitext(filepath)[-1].lower()
    
    # Step 1: Find header line index
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    header_line = None
    for idx, line in enumerate(lines):
        if "time" in line.lower() and "potential" in line.lower():
            header_line = idx
            break

    if header_line is None:
        raise ValueError("🛑 Could not find a valid data header (e.g. 'Time/sec, Potential/V')")

    # Step 2: Read from header line
    if ext == ".csv":
        # Already proper columns
        df = pd.read_csv(filepath, skiprows=header_line, engine="python")
    else:  # .txt
        # Data is comma-separated but not split into columns
        df = pd.read_csv(
            filepath,
            skiprows=header_line,
            sep=",",  # force split at commas
            engine="python",
            names=["Time/sec", "Potential/V"]
        )

    # Step 3: Clean duplicate timestamps
    time, voltage = clean_gcd_data(file_path=None, df=df, time_col="Time/sec", voltage_col="Potential/V", mode="preserve")
    cleaned_df = pd.DataFrame({"Time/sec": time, "Potential/V": voltage})
    
    return time, voltage

def clean_gcd_data(file_path=None, df=None, time_col="Time/sec", voltage_col="Potential/V", mode="preserve"):
    """
    Cleans GCD data by:
    - Reading from file or using given DataFrame
    - Removing non-numeric entries
    - Handling duplicate timestamps based on `mode`

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input dataframe with GCD data.
    file_path : str, optional
        Path to file if df is not provided.
    time_col : str
        Column name for time.
    voltage_col : str
        Column name for voltage.
    mode : str, default="preserve"
        How to handle duplicate timestamps:
        - "preserve": keep all duplicates (no averaging) ✅ recommended for GCD
        - "average": average duplicates
        - "median": take median of duplicates
        - "first": take first occurrence
        - "last": take last occurrence
    """
    # Step 1: Load
    if df is None and file_path:
        df = pd.read_csv(
            file_path,
            sep=r"[\s,]+", engine="python", comment="#"
        )
    if df is None:
        raise ValueError("Either file_path or df must be provided.")

    # Step 2: Normalize column names (strip spaces, lowercase for safety)
    df.columns = [c.strip() for c in df.columns]
    df = df[[time_col, voltage_col]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # Step 3: Group by duplicate timestamps and average
    if mode == "preserve":
        cleaned = df  # keep all duplicates
    elif mode == "average":
        cleaned = df.groupby(time_col, as_index=False).mean()
    elif mode == "median":
        cleaned = df.groupby(time_col, as_index=False).median()
    elif mode == "first":
        cleaned = df.groupby(time_col, as_index=False).first()
    elif mode == "last":
        cleaned = df.groupby(time_col, as_index=False).last()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Step 4: Return cleaned arrays
    time_array = cleaned[time_col].values
    voltage_array = cleaned[voltage_col].values

    return time_array, voltage_array

def detect_cycle_peaks_and_valleys(voltage, time, prominence=0.01, default_distance=30):
    """
    Detect both peaks and valleys in the GCD voltage signal.

    Parameters:
        voltage (array-like): Voltage values
        time (array-like): Time values 
        prominence (float): Minimum required prominence of peaks/valleys
        default_distance (int): Fallback minimum spacing between extrema (in data points)
        plot (bool): If True, generate a plot
        return_fig (bool): If True, return the matplotlib figure object

    Returns:
        peaks (list[int]): Indices of detected peaks
        valleys (list[int]): Indices of detected valleys
        fig (matplotlib.figure.Figure | None): Figure if return_fig=True and plot=True
    """
    # Estimate average spacing between cycles using autocorrelation
    voltage_zero_mean = voltage - np.mean(voltage)
    corr = correlate(voltage_zero_mean, voltage_zero_mean, mode='full')
    corr = corr[len(corr)//2:]  # keep second half only

    # Heuristic to skip the initial peak at lag = 0
    corr_peak = np.argmax(corr[default_distance:]) + default_distance
    estimated_distance = max(corr_peak, default_distance)

    # Detect peaks and valleys using scipy's find_peaks
    peaks, _ = find_peaks(voltage, prominence=prominence, distance=estimated_distance)
    valleys, _ = find_peaks(-voltage, prominence=prominence, distance=estimated_distance)
    
    print(f"✅ Detected {len(peaks)} peaks and {len(valleys)} valleys")

    # Optional: estimate how many full cycles (peak → valley)
    estimated_pairs = sum(1 for p in peaks if any(v > p for v in valleys))
    print(f"📊 Estimated usable peak–valley pairs (cycles): {estimated_pairs}")
    
    return peaks, valleys

def match_peak_valley_pairs(peaks, valleys):
    """
    Match each peak to the *next valley* that comes after it in time.

    Returns:
        List of (peak_idx, valley_idx) tuples.
    """
    matched = []
    v_pointer = 0

    for p in peaks:
        # Advance valley pointer until we find one that comes after the peak
        while v_pointer < len(valleys) and valleys[v_pointer] <= p:
            v_pointer += 1
        if v_pointer < len(valleys):
            matched.append((p, valleys[v_pointer]))
            v_pointer += 1  # Move on to next valley
        else:
            break

    return matched

def plot_discharge_fit(t_seg, v_seg, pred, C, r2, cycle_id):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_seg, v_seg, label="Discharge Segment")
    ax.plot(t_seg, pred, '--', label=f"Fit: C={C:.2f}F, R²={r2:.2f}")
    ax.set_title(f"Cycle {cycle_id}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig

def compute_capacitance_from_peak_valley_pairs(
    time, voltage, peak_valley_pairs, current,
    plot_debug=False
):
    capacitance_values = []
    r2_values = []
    cycle_ids = []
    debug_figs = {}
    warnings = []

    for i, (p_idx, v_idx) in enumerate(peak_valley_pairs):
        t_seg = time[p_idx:v_idx+1]
        v_seg = voltage[p_idx:v_idx+1]

        # Skip if too short or time not increasing
        if len(t_seg) < 3 or not np.all(np.diff(t_seg) > 0):
            msg = f"⚠️ Cycle {i+1}: segment too short or time not increasing"
            warnings.append(msg)
            continue

        # Fit slope using polyfit
        slope, intercept = np.polyfit(t_seg, v_seg, 1)

        # Calculate R² manually (for info only, not filtering)
        pred = intercept + slope * t_seg
        ss_res = np.sum((v_seg - pred) ** 2)
        ss_tot = np.sum((v_seg - np.mean(v_seg)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

        # Only accept if slope is negative (discharge)
        if slope >= 0:
            msg = f"⚠️ Cycle {i+1}: slope not negative (slope={slope:.4f})"
            warnings.append(msg)
            continue

        # Calculate capacitance
        C = current / abs(slope)

        capacitance_values.append(C)
        r2_values.append(r2)
        cycle_ids.append(i + 1)

        if plot_debug:
            debug_figs[f"cycle_{i+1}"] = plot_discharge_fit(t_seg, v_seg, pred, C, r2, i+1)

    print(f"warnings: {warnings}")
    
    cap_df = pd.DataFrame({
        "cycle": cycle_ids,
        "capacitance_F": capacitance_values,
        "r2": r2_values
    })

    return (cap_df, debug_figs, warnings) if plot_debug else (cap_df, {}, warnings)

# Plot graphs
def plot_raw_voltage(time, voltage):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, voltage, label="Raw Voltage")
    ax.set_title("Raw GCD Data")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_peaks_valleys(time, voltage, peaks, valleys):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, voltage, label="Voltage")
    ax.plot(time[peaks], voltage[peaks], "ro", label="Peaks")
    ax.plot(time[valleys], voltage[valleys], "go", label="Valleys")
    ax.set_title("Cycle Detection: Peaks & Valleys")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_capacitance_vs_cycle(cap_df):
    x = np.ravel(cap_df["cycle"].values)
    y = np.ravel(cap_df["capacitance_F"].values)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, marker='o', linestyle='-')
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Capacitance (F)")
    ax.set_title("Capacitance per Cycle")
    
    # Define y-axis limits if given
    # if y_min is not None or y_max is not None:
    #     ax.set_ylim(y_min, y_max)
    
    # Grid: horizontal only, no vertical lines
    ax.grid(axis='y')
    fig.tight_layout()
    # plt.show()
    return fig
    
# Plot retention vs cycle
def plot_retention_vs_cycle(cap_df):
    x = np.ravel(cap_df["cycle"].values)
    y = np.ravel(cap_df["retention_pct"].values)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, marker='o', linestyle='-')
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Capacitance Retention (%)")
    ax.set_title("Capacitance Retention")
    # ax.set_ylim(0, 120)
    ax.grid(axis='y')
    fig.tight_layout()
    # plt.show()
    return fig

### Wrapper Function ###
def analyze_gcd_core(df, current, plot_base_data=True, plot_debug=False):
    """
    Full GCD analysis pipeline with robust cycle detection and capacitance calculation.

    Parameters:
        df (pd.DataFrame): DataFrame with [time, voltage] columns
        current (float): Applied current (A)
        plot_base_data : Plots the raw and peak valley data
        plot_debug : Plots successful capacitance fitting
    """

    results = {}
    figs = {}

    # 1. Handle metadata row
    if isinstance(df, tuple):
        time, voltage = df  # unpack directly
    else:
        # DataFrame case
        if len(df) > 1 and not pd.api.types.is_numeric_dtype(df.iloc[1, 0]):
            df = df.drop(df.index[1]).reset_index(drop=True)
        time = df.iloc[:, 0].to_numpy()
        voltage = df.iloc[:, 1].to_numpy()

    if plot_base_data:
        figs["raw"] = plot_raw_voltage(time, voltage)
           
    # 2. Detect peaks + valleys as cycle boundaries
    peaks, valleys = detect_cycle_peaks_and_valleys(voltage, time)
    peak_valley_pairs = match_peak_valley_pairs(peaks, valleys)
    results["peak_valley_pairs"] = peak_valley_pairs

    if plot_base_data:
        figs["peaks_valleys"] = plot_peaks_valleys(time, voltage, peaks, valleys)
        
    # 3. Compute capacitance
    cap_df, debug_figs, warnings = compute_capacitance_from_peak_valley_pairs(
        time, voltage, peak_valley_pairs, current, plot_debug=plot_debug
    )
    
    results["capacitance"] = cap_df
    results["warnings"] = warnings
    
    if plot_debug:
        figs.update(debug_figs)

    # 4. Add retention % to the results
    if not cap_df.empty:
        cap_df["retention_pct"] = (
            100 * cap_df["capacitance_F"] / cap_df["capacitance_F"].iloc[0]
        )
    
        # 6. Add summary plots
        figs["cap_vs_cycle"] = plot_capacitance_vs_cycle(cap_df)
        figs["retention_vs_cycle"] = plot_retention_vs_cycle(cap_df)
    else:
        msg = "❌ No valid capacitance data calculated — check input or cycle detection."
        results["warnings"].append(msg)
        print(msg)  # for Jupyter

    results["figs"] = figs
    return results

# --- Wrappers for different environments ---
def analyze_gcd(input_data, current, output_path=None, base_name=None, 
                mode="notebook", plot_base_data=True, plot_debug=False):
    """
    Unified wrapper for GCD analysis.
    
    Parameters:
        input_data (str | pd.DataFrame): Filepath to data OR preloaded DataFrame.
        current (float): Applied current (A).
        output_path (str | None): Where to save plots if mode='app'.
        base_name (str | None): Base name for saving plots. If None and input is a file, it's inferred.
        mode (str): 'notebook' (show plots), 'app' (streamlit + save).
    """
    # Handle UploadedFile
    if hasattr(input_data, "read"):  # Streamlit UploadedFile
        ext = os.path.splitext(input_data.name)[-1].lower()

        # Write UploadedFile to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(input_data.read())
            tmp_path = tmp.name

        df = load_voltage_data(tmp_path)   # ✅ reuse your notebook function
        if base_name is None:
            base_name = os.path.splitext(input_data.name)[0]

    # Handle string filepath
    elif isinstance(input_data, str):
        df = load_voltage_data(input_data)
        if base_name is None:
            base_name = os.path.splitext(os.path.basename(input_data))[0]

    # Handle DataFrame
    else:
        df = input_data

    # Run the core pipeline
    results = analyze_gcd_core(df, current, plot_base_data=plot_base_data, plot_debug=plot_debug)

    figs = results.get("figs", {})
    cap_df = results.get("capacitance", pd.DataFrame())
    warnings = results.get("warnings", [])

    # --- Setup timestamp for filenames ---
    timestamp = datetime.now().strftime("%Y%m%d")

    # --- Display/save figures in fixed order ---
    plot_order = ["raw", "peaks_valleys", "cap_vs_cycle", "retention_vs_cycle"]
    for key in plot_order:
        if key in figs:
            fig = figs[key]
            if mode == "notebook":
                plt.show(fig)
                if output_path:
                    fig.savefig(
                        os.path.join(output_path, f"{key}_{timestamp}.jpg"),
                        format="jpg", dpi=300
                    )
            elif mode == "app":
                st.pyplot(fig)
                # if output_path and base_name:
                #     fig.savefig(os.path.join(output_path, f"{base_name}_{key}_{timestamp}.jpg"),
                #                 format="jpg", dpi=300)
                #     plt.close(fig)

    # --- Handle warnings ---
    if mode == "notebook" and warnings and output_path and base_name:
        with open(os.path.join(output_path, f"{base_name}_warnings.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(warnings))

    # --- Save capacitance results ---
    if mode == "notebook" and output_path and base_name:
        cap_file = f"{base_name}_capacitance_{timestamp}.csv"
        cap_path = os.path.join(output_path, cap_file)

        # Always save as CSV
        cap_df.to_csv(cap_path, index=False, encoding="utf-8-sig")

    return results
