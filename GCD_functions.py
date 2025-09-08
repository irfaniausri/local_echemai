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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, correlate
from scipy.stats import linregress
from datetime import datetime


# +
# 1. Load CSV file
def load_voltage_data(filepath):
    """
    Smart loader for GCD data (.txt or .csv) that:
    - Skips metadata blocks (in .txt)
    - Finds the line with 'Time/sec'
    - Cleans and averages duplicate timestamps
    """
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
    df = pd.read_csv(
        filepath,
        skiprows=header_line,
        sep=r"[\s,]+",  # support space, tab, or comma
        engine="python",
        names=["Time/sec", "Potential/V"]
    )

    # Step 3: Clean duplicate timestamps
    time, voltage = clean_gcd_data(df=df)
    return time, voltage

def clean_gcd_data(file_path=None, df=None, time_col="Time/sec", voltage_col="Potential/V"):
    """
    Cleans GCD data by:
    - Reading from file or using given DataFrame
    - Removing non-numeric entries
    - Averaging voltage values for duplicate timestamps
    - Returning clean time and voltage arrays
    """
    # Step 1: Load
    if file_path:
        df = pd.read_csv(
            file_path,
            sep=r"\s+", engine="python", comment="#",
            names=[time_col, voltage_col], header=0
        )
    
    if df is None:
        raise ValueError("Either file_path or df must be provided.")

    # Step 2: Coerce to numeric
    df = df[[time_col, voltage_col]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # Step 3: Group by duplicate timestamps and average
    grouped = df.groupby(time_col, as_index=False).mean()

    # Step 4: Return cleaned arrays
    time_array = grouped[time_col].values
    voltage_array = grouped[voltage_col].values

    return time_array, voltage_array

# 2. Smooth voltage using Savitzky-Golay filter
def plot_voltage_smoothing(time, voltage, window_length=11, polyorder=3, plot=False, title="Voltage Smoothing Preview"):
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

# 3. Detect peak indices (where slope changes from + to -)
def detect_cycle_boundaries(smoothed_voltage, time, prominence=0.01, default_distance=30):
    """
    Automatically detect GCD cycle boundaries (peaks and valleys) without needing expected_cycles.
    
    Parameters:
        smoothed_voltage (array): Smoothed voltage array
        time (array): Time array (same length)
        prominence (float): Peak prominence filter (how strong a peak needs to be)
        default_distance (int): Fallback minimum spacing between peaks (in data points)

    Returns:
        np.ndarray: Sorted list of global peak/valley indices
    """
    # Estimate average distance between true cycles
    dt = time[-1] - time[0]              # Total duration
    dt_per_point = np.mean(np.diff(time))  # Sampling rate
    total_points = len(time)

    # Heuristic: use autocorrelation to guess peak spacing
    from scipy.signal import correlate
    corr = correlate(smoothed_voltage - np.mean(smoothed_voltage), smoothed_voltage - np.mean(smoothed_voltage), mode='full')
    corr = corr[len(corr)//2:]
    corr_peak = np.argmax(corr[default_distance:]) + default_distance  # skip first region
    estimated_distance = max(corr_peak, default_distance)

    # Find peaks
    peaks, _ = find_peaks(smoothed_voltage, prominence=prominence, distance=estimated_distance)
    valleys, _ = find_peaks(-smoothed_voltage, prominence=prominence, distance=estimated_distance)

    return np.sort(np.concatenate([peaks, valleys]))

def detect_cycle_peaks_and_valleys(smoothed_voltage, time, prominence=0.01, default_distance=30):
    """
    Detect both peaks and valleys from a smoothed GCD voltage signal.

    Parameters:
        smoothed_voltage (array-like): Smoothed voltage data (1D array)
        time (array-like): Time values corresponding to the voltage data
        prominence (float): Minimum required prominence of peaks/valleys
        default_distance (int): Fallback minimum spacing between extrema (in data points)

    Returns:
        np.ndarray: Sorted array of extrema indices (including both peaks and valleys)
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

    # Combine and sort
    extrema_indices = np.sort(np.concatenate([peaks, valleys]))

    return extrema_indices

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


# 4. Compute capacitance per cycle
def compute_capacitance_from_peak_valley_pairs(
    time, voltage, peak_valley_pairs, current,
    r2_threshold=0.5, window_fit=False, window_size=5,
    debug_plot=False
):
    capacitance_values = []
    cycle_ids = []

    for i, (p_idx, v_idx) in enumerate(peak_valley_pairs):
        t_seg = time[p_idx:v_idx+1]
        v_seg = voltage[p_idx:v_idx+1]

        if len(t_seg) < 3 or not np.all(np.diff(t_seg) > 0):
            print(f"⚠️ Cycle {i+1}: segment too short or time not increasing")
            continue

        if window_fit:
            # Fit around steepest part
            dv = np.diff(v_seg)
            dt = np.diff(t_seg)
            slope_seg = np.divide(dv, dt, out=np.zeros_like(dv), where=dt != 0)
            steepest = np.argmin(slope_seg)
            start_idx = max(steepest - window_size // 2, 0)
            end_idx = min(steepest + window_size // 2 + 1, len(t_seg))
            t_fit = t_seg[start_idx:end_idx]
            v_fit = v_seg[start_idx:end_idx]
        else:
            t_fit = t_seg
            v_fit = v_seg

        # Linear fit
        slope, intercept, r_value, _, _ = linregress(t_fit, v_fit)

        if slope >= 0:
            print(f"⚠️ Cycle {i+1}: slope not negative (slope={slope:.4f})")
            continue
        if r_value**2 < r2_threshold:
            print(f"⚠️ Cycle {i+1}: poor fit (R²={r_value**2:.3f})")
            continue

        C = current / abs(slope)
        capacitance_values.append(C)
        cycle_ids.append(i + 1)

        if debug_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 3))
            plt.plot(t_seg, v_seg, label="Discharge Segment")
            plt.plot(t_fit, intercept + slope * t_fit, '--', label=f"Fit: C={C:.2f}F, R²={r_value**2:.2f}")
            plt.title(f"Cycle {i+1}")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame({
        "cycle": cycle_ids,
        "capacitance_F": capacitance_values
    })

def compute_capacitance_per_cycle(time, voltage, peak_indices, current, r2_threshold=0.90, debug_plot=False):
    """
    Computes capacitance per cycle using linear regression on the discharge segment.
    
    Parameters:
        time (array): Time values (1D)
        voltage (array): Smoothed voltage values (1D)
        peak_indices (list or array): List of peak indices defining cycle boundaries
        current (float): Applied current in amperes
        r2_threshold (float): Minimum R² value for accepting a linear fit
        debug_plot (bool): If True, show per-cycle debug plots
        
    Returns:
        pd.DataFrame with columns: cycle, capacitance_F
    """
    
    capacitance_values = []
    cycle_ids = []

    for i in range(len(peak_indices) - 1):
        start = peak_indices[i]
        end = peak_indices[i + 1]
        
        t_seg = np.asarray(time[start:end])
        v_seg = np.asarray(voltage[start:end])
        
        # Sanity check
        if len(t_seg) < 3 or not np.all(np.diff(t_seg) > 0):
            print(f"⚠️ Skipping cycle {i + 1}: segment too short or non-monotonic time")
            continue
            
        # Compute per-segment slope via linear regression
        slope, intercept, r_value, p_value, std_err = linregress(t_seg, v_seg)

        if slope >= 0:
#             print(f"⚠️ Skipping cycle {i + 1}: positive slope (not discharge)")
            continue

        if r_value**2 < r2_threshold:
            print(f"⚠️ Skipping cycle {i + 1}: poor linear fit (R²={r_value**2:.3f})")
            continue

        C = current / abs(slope)
        capacitance_values.append(C)
        cycle_ids.append(i + 1)
        
        if debug_plot:
            plt.figure(figsize=(6, 3))
            plt.plot(t_seg, v_seg, label="Discharge segment", marker='o', linewidth=1.5)
            plt.plot(t_seg, intercept + slope * t_seg, label=f"Fit (R²={r_value**2:.3f})", linestyle='--')
            plt.title(f"Cycle {i + 1} — Capacitance: {C:.2f} F")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    return pd.DataFrame({
        "cycle": cycle_ids,
        "capacitance_F": capacitance_values
    })

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
def analyze_gcd(filepath, output_path, current, y_min=400, y_max=800, smooth_window=11, poly_order=3,
                debug_plot=False):
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
    smoothed_voltage = plot_voltage_smoothing(time, voltage, window_length=11, polyorder=3, plot=False)
    
    # 2. Detect peaks + valleys as cycle boundaries
#     cycle_boundaries = detect_cycle_boundaries(
#         smoothed_voltage, time,
#         prominence=0.01
#     )
#     plot_detected_boundaries(time, smoothed_voltage, cycle_boundaries)
    peaks, valleys = detect_cycle_peaks_and_valleys(smoothed_voltage, time)
    peak_valley_pairs = match_peak_valley_pairs(peaks, valleys)

    # 3. Compute capacitance per cycle
#     cap_df = compute_capacitance_per_cycle(
#         time, smoothed_voltage, cycle_boundaries, current,
#         debug_plot=debug_plot
#     )
    cap_df = compute_capacitance_from_peak_valley_pairs(
        time, smoothed_voltage, peak_valley_pairs, current,
        window_fit=True, debug_plot=True
    )
    
#     cap_df = compute_capacitance_per_cycle(time, smoothed_voltage, peaks, current, debug_plot=True)
    
    # Add retention % to the results
    cap_df["retention_pct"] = 100 * cap_df["capacitance_F"] / cap_df["capacitance_F"].iloc[0]
    
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"capacitance_results_{timestamp}.csv"
    output_file_path = os.path.join(output_path, filename)
    cap_df.to_csv(output_file_path, index=False)

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

