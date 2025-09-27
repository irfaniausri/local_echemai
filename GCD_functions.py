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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
    # if file_path:
    #     df = pd.read_csv(
    #         file_path,
    #         sep=r"\s+", engine="python", comment="#",
    #         names=[time_col, voltage_col], header=0
    #     )
    
    # if df is None:
    #     raise ValueError("Either file_path or df must be provided.")

    # Step 2: Coerce to numeric
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

# 2. Smooth voltage using Savitzky-Golay filter
def smooth_voltage(time, voltage, window_length=11, polyorder=3, plot=False, title="Voltage Smoothing Preview"):
    smoothed = savgol_filter(voltage, window_length=window_length, polyorder=polyorder)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(time, voltage, label="Raw Voltage", color="gray", alpha=0.6)
        plt.plot(time, smoothed, label=f"Smoothed (window={window_length}, poly={polyorder})", color="blue")
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return smoothed

def detect_cycle_peaks_and_valleys(smoothed_voltage, time, prominence=0.01, default_distance=30, plot=False):
    """
    Detect both peaks and valleys from a smoothed GCD voltage signal.

    Parameters:
        smoothed_voltage (array-like): Smoothed voltage data (1D array)
        time (array-like): Time values corresponding to the voltage data
        prominence (float): Minimum required prominence of peaks/valleys
        default_distance (int): Fallback minimum spacing between extrema (in data points)

    Returns:
        peaks (np.ndarray): Indices of peak points (charging ends)
        valleys (np.ndarray): Indices of valley points (discharging ends)
    """
    # Estimate average spacing between cycles using autocorrelation
    voltage_zero_mean = smoothed_voltage - np.mean(smoothed_voltage)
    corr = correlate(voltage_zero_mean, voltage_zero_mean, mode='full')
    corr = corr[len(corr)//2:]  # keep second half only

    # Heuristic to skip the initial peak at lag = 0
    corr_peak = np.argmax(corr[default_distance:]) + default_distance
    estimated_distance = max(corr_peak, default_distance)

    # Detect peaks and valleys using scipy's find_peaks
    peaks, _ = find_peaks(smoothed_voltage, prominence=prominence, distance=estimated_distance)
    valleys, _ = find_peaks(-smoothed_voltage, prominence=prominence, distance=estimated_distance)
    
    print(f"✅ Detected {len(peaks)} peaks and {len(valleys)} valleys")

    # Optional: estimate how many full cycles (peak → valley)
    estimated_pairs = sum(1 for p in peaks if any(v > p for v in valleys))
    print(f"📊 Estimated usable peak–valley pairs (cycles): {estimated_pairs}")
    
    # Optional plot
    if plot:
        peak_valley_indices = np.sort(np.concatenate([peaks, valleys]))
        plot_detected_boundaries(time, smoothed_voltage, peak_valley_indices, title="Detected Peaks and Valleys")
    
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

def plot_detected_boundaries(time, voltage, peak_indices, title="Detected Cycle Transitions"):
    """
    Plot the smoothed voltage and overlay the detected cycle boundaries (peaks + valleys).
    
    Parameters:
        time (array): Time values
        voltage (array): Smoothed voltage values
        peak_indices (array): Detected peak + valley indices
        title (str): Plot title
    """
    plt.figure(figsize=(10, 4))
    plt.plot(time, voltage, label="Smoothed Voltage", linewidth=1.2)
    plt.plot(time[peak_indices], voltage[peak_indices], 'ro', label="Detected Peaks/Valleys", markersize=5)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_discharge_debug(t_seg, v_seg, t_fit, slope, intercept, cycle_id, issue_type, r2=None):
    """
    Plots a debug view of a discharge segment with a bad fit.
    
    Args:
        t_seg: Full discharge time segment (peak to valley)
        v_seg: Full voltage segment
        t_fit: Portion used for linear fitting
        slope: Fitted slope
        intercept: Fitted intercept
        cycle_id: Cycle number (1-based)
        issue_type: "invalid_slope" or "poor_fit"
        r2: Optional R² value for annotation
    """
    plt.figure(figsize=(6, 3))
    plt.plot(t_seg, v_seg, label="Discharge Segment")

    if t_fit is not None and slope is not None:
        plt.plot(t_fit, intercept + slope * t_fit, '--', label="Linear Fit")

    title = f"⚠️ Cycle {cycle_id} — "
    if issue_type == "invalid_slope":
        title += "Invalid Slope (≥ 0)"
    elif issue_type == "poor_fit":
        title += f"Poor Fit (R²={r2:.2f})" if r2 is not None else "Poor Fit"

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_capacitance_from_peak_valley_pairs(
    time, voltage, peak_valley_pairs, current,
    plot_good=False
):
    capacitance_values = []
    r2_values = []
    cycle_ids = []

    for i, (p_idx, v_idx) in enumerate(peak_valley_pairs):
        t_seg = time[p_idx:v_idx+1]
        v_seg = voltage[p_idx:v_idx+1]

        # Skip if too short or time not increasing
        if len(t_seg) < 3 or not np.all(np.diff(t_seg) > 0):
            print(f"⚠️ Cycle {i+1}: segment too short or time not increasing")
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
            print(f"⚠️ Cycle {i+1}: slope not negative (slope={slope:.4f})")
            continue

        # Calculate capacitance
        C = current / abs(slope)

        capacitance_values.append(C)
        r2_values.append(r2)
        cycle_ids.append(i + 1)

        # Optional plot
        if plot_good:
            plt.figure(figsize=(6, 3))
            plt.plot(t_seg, v_seg, label="Discharge Segment")
            plt.plot(t_seg, pred, '--', label=f"Fit: C={C:.2f}F, R²={r2:.2f}")
            plt.title(f"Cycle {i+1}")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame({
        "cycle": cycle_ids,
        "capacitance_F": capacitance_values,
        "r2": r2_values
    })
    
# def compute_capacitance_from_peak_valley_pairs(
#     time, voltage, peak_valley_pairs, current,
#     r2_threshold=0.5, window_size=5, plot_good=False,
#     debug_plot=False
# ):
#     capacitance_values = []
#     cycle_ids = []

#     for i, (p_idx, v_idx) in enumerate(peak_valley_pairs):
#         t_seg = time[p_idx:v_idx+1]
#         v_seg = voltage[p_idx:v_idx+1]

#         if len(t_seg) < 3 or not np.all(np.diff(t_seg) > 0):
#             print(f"⚠️ Cycle {i+1}: segment too short or time not increasing")
#             continue

#         t_fit = t_seg
#         v_fit = v_seg

#         # Linear fit
#         slope, intercept, r_value, _, _ = linregress(t_fit, v_fit)

#         if slope >= 0:
#             print(f"⚠️ Cycle {i+1}: slope not negative (slope={slope:.4f})")
#             if debug_plot:
#                 plot_discharge_debug(t_seg, v_seg, t_fit, slope, intercept, i + 1, issue_type="invalid_slope")
#             continue

#         if r_value**2 < r2_threshold:
#             print(f"⚠️ Cycle {i+1}: poor fit (R²={r_value**2:.3f})")
#             if debug_plot:
#                 plot_discharge_debug(t_seg, v_seg, t_fit, slope, intercept, i + 1, issue_type="poor_fit", r2=r_value**2)
#             continue

#         C = current / abs(slope)
#         capacitance_values.append(C)
#         cycle_ids.append(i + 1)

#         if plot_good:
#             plt.figure(figsize=(6, 3))
#             plt.plot(t_seg, v_seg, label="Discharge Segment")
#             plt.plot(t_fit, intercept + slope * t_fit, '--', label=f"Fit: C={C:.2f}F, R²={r_value**2:.2f}")
#             plt.title(f"Cycle {i+1}")
#             plt.xlabel("Time (s)")
#             plt.ylabel("Voltage (V)")
#             plt.legend()
#             plt.grid(True)
#             plt.tight_layout()
#             plt.show()

#     return pd.DataFrame({
#         "cycle": cycle_ids,
#         "capacitance_F": capacitance_values
#     })

# 5. Optional: plot capacitance vs cycle
def plot_capacitance_vs_cycle(cap_df, y_min=None, y_max=None):
    x = np.ravel(cap_df["cycle"].values)
    y = np.ravel(cap_df["capacitance_F"].values)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o', linestyle='-')

    # Labels
    plt.xlabel("Cycle Number")
    plt.ylabel("Capacitance (F)")
    plt.title("Capacitance per Cycle")
    
    # Define y-axis limits if given
    if y_min is not None or y_max is not None:
        plt.ylim(y_min, y_max)
    
    # Grid: horizontal only, no vertical lines
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# 6. Optional: plot retention vs cycle
def plot_retention_vs_cycle(cap_df):
    x = np.ravel(cap_df["cycle"].values)
    y = np.ravel(cap_df["retention_pct"].values)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o', linestyle='-')
    
    plt.xlabel("Cycle Number")
    plt.ylabel("Capacitance Retention (%)")
    plt.title("Capacitance Retention")
    plt.ylim(0, 120)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

### Wrapper Function ###
def analyze_gcd(filepath, output_path, current, y_min=400, y_max=800, plot_raw=True, plot_good=False, debug_plot=False,
                smooth_window=11, poly_order=3):
    """
    Full GCD analysis pipeline with robust cycle detection and capacitance calculation.

    Parameters:
        filepath (str): Path to raw GCD data file
        output_path (str): Where to save the results
        current (float): Applied current (A)
        expected_cycles (int): Approximate expected number of cycles
        y_min, y_max (float): Plotting limits
        smooth_window (int): Window length for Savitzky-Golay smoothing
        poly_order (int): Polynomial order for smoothing
        debug_plot (bool): If True, show per-cycle debug plots
    """
    # 1. Load and smooth
    time, voltage = load_voltage_data(filepath)
    
    if plot_raw:
        plt.figure(figsize=(8, 4))
        plt.plot(time, voltage, label="Raw Voltage")
        plt.title("Raw GCD Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
#     smoothed_voltage = smooth_voltage(time, voltage, window_length=11, polyorder=3, plot=False)
    
    # 2. Detect peaks + valleys as cycle boundaries
    peaks, valleys = detect_cycle_peaks_and_valleys(voltage, time, plot=True)
    peak_valley_pairs = match_peak_valley_pairs(peaks, valleys)

    # 3. Compute capacitance
    cap_df = compute_capacitance_from_peak_valley_pairs(
        time, voltage, peak_valley_pairs, current, plot_good=plot_good
    )
        
    # 4. Add retention % to the results
    cap_df["retention_pct"] = 100 * cap_df["capacitance_F"] / cap_df["capacitance_F"].iloc[0]
    
    # 5. Build output filename based on input name
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{base_name}_capacitance_{timestamp}.csv"
    output_file_path = os.path.join(output_path, filename)
    
    # 6. Save results
    cap_df.to_csv(output_file_path, index=False)
    print(f"✅ Saved results to {output_file_path}")
    
    # 7. Plot results
    plot_capacitance_vs_cycle(cap_df, y_min, y_max)
    plot_retention_vs_cycle(cap_df)
    
def analyze_gcd_df(df, current, has_metadata_row=True):
    # If the second row is metadata, drop it here instead of skiprows during read
    if has_metadata_row and len(df) > 1:
        df = df.drop(df.index[1]).reset_index(drop=True)

    time = df.iloc[:, 0].to_numpy()
    voltage = df.iloc[:, 1].to_numpy()

    smoothed_voltage = smooth_voltage(voltage)
    peaks = detect_cycle_peaks(smoothed_voltage)
    cap_df = compute_capacitance_per_cycle(time, smoothed_voltage, peaks, current)
    cap_df["retention_pct"] = 100 * cap_df["capacitance_F"] / cap_df["capacitance_F"].iloc[0]

    return cap_df

