ECHEM APP MVP

A lightweight analysis app for galvanostatic charge–discharge (GCD) experiments. It provides a complete pipeline for data cleaning, cycle detection, capacitance calculation, and retention plotting. The app can be run in Jupyter notebooks or deployed as a Streamlit app.

Features
- Load and clean .csv or .txt GCD data.
- Automatically detect charge–discharge cycles.
- Compute capacitance per cycle with regression fitting.
- Plot raw data, cycle detection, capacitance vs. cycle, and retention plots.
- Export results to CSV for further analysis.
- Run in Jupyter notebooks or interactively via Streamlit.

Function Overview
The core pipeline is driven by analyze_gcd_core and wrapped by analyze_gcd for notebook/Streamlit use. These functions call several helpers that handle data loading, cleaning, analysis, and visualization:

Data Handling
- load_voltage_data(filepath) – Smart loader for .csv and .txt GCD data. Finds headers, skips metadata, and cleans duplicate timestamps. Returns clean arrays for time and voltage.
- clean_gcd_data(file_path=None, df=None, ...) – Cleans raw GCD data: removes non-numeric values, handles duplicate timestamps, and supports modes like preserve, average, median, first, or last.

Cycle Detection
- detect_cycle_peaks_and_valleys(voltage, time, ...) – Detects local maxima (peaks) and minima (valleys) in the voltage signal to identify charge–discharge cycles. Uses autocorrelation to estimate cycle spacing.
- match_peak_valley_pairs(peaks, valleys) – Pairs each detected peak with the subsequent valley, defining usable charge–discharge segments.

Capacitance Calculation
- compute_capacitance_from_peak_valley_pairs(time, voltage, pairs, current, ...) – Fits a linear regression to each discharge segment, ensuring a negative slope, then computes capacitance per cycle as C=I/∣dV/dt∣. Returns results as a DataFrame with cycle numbers, capacitance values, and R² goodness-of-fit scores.
- plot_discharge_fit(...) – Debug utility to visualize a single discharge fit, showing raw data, regression line, and calculated capacitance.

Plotting
- plot_raw_voltage(time, voltage) – Plots raw GCD voltage vs. time.
- plot_peaks_valleys(time, voltage, peaks, valleys) – Plots detected peaks and valleys on top of the raw GCD curve.
- plot_capacitance_vs_cycle(cap_df) – Plots capacitance value vs. cycle number.
- plot_retention_vs_cycle(cap_df) – Plots retention percentage relative to the first cycle.

Streamlit support for interactive use and file uploads.

Installation
git clone https://github.com/yourusername/echem-app.git
cd echem-app
pip install -r requirements.txt

Usage
Run in Notebook
from local_echemai import GCD_functions

functions.analyze_gcd(input_file, current, output_path, base_name=None, mode="notebook", plot_base_data=True, plot_debug=False))

Run in Streamlit
streamlit run echem_app.py

Once deployed, the app will be available at a public URL from Streamlit Cloud (e.g. https://echem-app-posbpw.streamlit.app).

Streamlit App Workflow
When run via Streamlit, the app provides an interactive interface for GCD processing:
1. Title & Setup
- Displays app title: “🧪 GCD Processor”.
- Prompts the user to enter their name.
2. File Upload
- Users upload one or more .csv or .txt GCD files.
- For each uploaded file, the app runs the full analyze_gcd pipeline with the specified current (default: 0.01 A).
3. Processing & Results
- The app:
  - Computes capacitance values per cycle.
  - Logs warnings (e.g., invalid cycles, bad fits).
  - Generates figures: raw data, cycle detection, capacitance vs. cycle, retention plots.
- All results (CSV, TXT, JPGs) are packaged into a ZIP file for download with a single click.
4. Logging
- Each run is logged (usage_log.csv) with:
  - Timestamp
  - User name
  - Uploaded file name
  - Status (success or failure)
5. Admin Section (Hidden)
- Accessible via ?admin=1 query parameter in the app URL.
- Allows admins to:
  - View usage logs as a table.
  - Download logs as a .csv file.

Project Structure
- echem_app.py: Main entry point for Streamlit.
- Codes/local_echemai/GCD_functions.py: Core GCD analysis functions.
- requirements.txt: Python dependencies.
- data/: Example input files.
- results/: Output plots and CSVs.

Example Outputs
- Raw GCD curve with detected peaks and valleys.
- Capacitance vs. cycle plot.
- Retention percentage vs. cycle number.
- Capacitance, r2, and retention values in csv