import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def load_and_validate_data(csv_path):
    """
    Load Phase 1 dataset and validate row count, sampling interval, and monotonicity.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    row_count = len(df)
    missing_count = df.isna().sum().sum()
    dt_series = df['timestamp'].diff().dropna()
    dt_median = round(dt_series.median(), 3)
    is_monotonic = df['timestamp'].is_monotonic_increasing
    
    validation_summary = {
        'row_count': row_count,
        'missing_count': missing_count,
        'dt_median': dt_median,
        'is_monotonic': is_monotonic,
        'sampling_rate_hz': round(1.0 / dt_median, 1) if dt_median > 0 else 0
    }
    
    print("=== DATA VALIDATION ===")
    print(f"File: {csv_path}")
    print(f"Rows: {row_count:,}")
    print(f"Sampling Step (dt): {dt_median} s ({validation_summary['sampling_rate_hz']} Hz)")
    print(f"Missing Values: {missing_count}")
    print(f"Timestamp Monotonicity: {'PASS ✅' if is_monotonic else 'FAIL ❌'}")
    
    return df, validation_summary


def lowpass_filter(data, cutoff_hz=3.0, fs_hz=10.0, order=2):
    """
    Applies a 2nd order Butterworth low-pass filter to smooth sensor noise.
    Cutoff = 3.0 Hz preserves true vehicle acceleration/braking dynamics.
    """
    nyq = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def clean_accelerometer_with_3axis_gravity(acc_x, acc_y, acc_z, grav_x, grav_y, grav_z, fs_hz=10.0):
    """
    PART 3 — Accelerometer Preprocessing with Full 3-Axis Gravity Subtraction.
    Subtracts exact 3-axis gravity components (g_x, g_y, g_z) to eliminate phone mount tilt bias.
    """
    # 3.1 Sanity Check & Spike Detection
    ax = np.nan_to_num(acc_x, nan=0.0, posinf=0.0, neginf=0.0)
    ay = np.nan_to_num(acc_y, nan=0.0, posinf=0.0, neginf=0.0)
    az = np.nan_to_num(acc_z, nan=0.0, posinf=0.0, neginf=0.0)
    
    gx = np.nan_to_num(grav_x, nan=0.0, posinf=0.0, neginf=0.0)
    gy = np.nan_to_num(grav_y, nan=0.0, posinf=0.0, neginf=0.0)
    gz = np.nan_to_num(grav_z, nan=9.80665, posinf=9.80665, neginf=9.80665)
    
    ax = np.clip(ax, -30.0, 30.0)
    ay = np.clip(ay, -30.0, 30.0)
    az = np.clip(az, -30.0, 30.0)
    
    # 3.2 Full 3-Axis Component-Wise Gravity Subtraction
    ax_dyn = ax - gx
    ay_dyn = ay - gy
    az_dyn = az - gz
    
    # 3.3 Low-pass Filtering (3 Hz cutoff)
    ax_clean = lowpass_filter(ax_dyn, cutoff_hz=3.0, fs_hz=fs_hz)
    ay_clean = lowpass_filter(ay_dyn, cutoff_hz=3.0, fs_hz=fs_hz)
    az_clean = lowpass_filter(az_dyn, cutoff_hz=3.0, fs_hz=fs_hz)
    
    return ax_clean, ay_clean, az_clean


def clean_gyroscope(gyro_x, gyro_y, gyro_z, fs_hz=10.0):
    """
    PART 4 — Gyroscope Preprocessing.
    - Sanity check & spike thresholding
    - Static bias estimation & removal (derived from initial 20 samples)
    - Low-pass filtering (3.0 Hz cutoff)
    """
    gx = np.nan_to_num(gyro_x, nan=0.0, posinf=0.0, neginf=0.0)
    gy = np.nan_to_num(gyro_y, nan=0.0, posinf=0.0, neginf=0.0)
    gz = np.nan_to_num(gyro_z, nan=0.0, posinf=0.0, neginf=0.0)
    
    gx = np.clip(gx, -5.0, 5.0)
    gy = np.clip(gy, -5.0, 5.0)
    gz = np.clip(gz, -5.0, 5.0)
    
    init_steps = min(20, len(gx))
    bias_x = np.mean(gx[:init_steps])
    bias_y = np.mean(gy[:init_steps])
    bias_z = np.mean(gz[:init_steps])
    
    gx_unbiased = gx - bias_x
    gy_unbiased = gy - bias_y
    gz_unbiased = gz - bias_z
    
    gx_clean = lowpass_filter(gx_unbiased, cutoff_hz=3.0, fs_hz=fs_hz)
    gy_clean = lowpass_filter(gy_unbiased, cutoff_hz=3.0, fs_hz=fs_hz)
    gz_clean = lowpass_filter(gz_unbiased, cutoff_hz=3.0, fs_hz=fs_hz)
    
    return gx_clean, gy_clean, gz_clean


def process_phase2_data(df_phase1, raw_s_path, fs_hz=10.0):
    """
    Main Phase 2 processing pipeline incorporating raw 3-axis gravity channels.
    """
    # Load raw smartphone dataset to access 3-axis GRAVITY X/Y/Z channels
    df_raw_s = pd.read_csv(raw_s_path, encoding='latin1')
    df_raw_s.columns = df_raw_s.columns.str.strip()
    
    min_len = min(len(df_phase1), len(df_raw_s))
    df_phase1 = df_phase1.iloc[:min_len].reset_index(drop=True)
    df_raw_s = df_raw_s.iloc[:min_len].reset_index(drop=True)
    
    acc_x = df_phase1['acc_x'].values
    acc_y = df_phase1['acc_y'].values
    acc_z = df_phase1['acc_z'].values
    
    gyro_x = df_phase1['gyro_x'].values
    gyro_y = df_phase1['gyro_y'].values
    gyro_z = df_phase1['gyro_z'].values
    
    # Raw 3-Axis Gravity channels
    grav_x = df_raw_s['GRAVITY X (m/s²)'].values
    grav_y = df_raw_s['GRAVITY Y (m/s²)'].values
    grav_z = df_raw_s['GRAVITY Z (m/s²)'].values
    
    # Clean signals with 3-axis gravity subtraction
    acc_x_clean, acc_y_clean, acc_z_clean = clean_accelerometer_with_3axis_gravity(
        acc_x, acc_y, acc_z, grav_x, grav_y, grav_z, fs_hz=fs_hz
    )
    gyro_x_clean, gyro_y_clean, gyro_z_clean = clean_gyroscope(gyro_x, gyro_y, gyro_z, fs_hz=fs_hz)
    
    # Phone -> Vehicle Frame Alignment
    longitudinal_acc = acc_y_clean
    lateral_acc = acc_x_clean
    vertical_acc = acc_z_clean
    yaw_rate = gyro_z_clean
    
    # Derived Features
    acc_magnitude = np.sqrt(acc_x_clean**2 + acc_y_clean**2 + acc_z_clean**2)
    gyro_magnitude = np.sqrt(gyro_x_clean**2 + gyro_y_clean**2 + gyro_z_clean**2)
    
    # Rolling Features (5-step trailing window = 0.5s)
    window = 5
    acc_series = pd.Series(acc_magnitude)
    gyro_series = pd.Series(gyro_magnitude)
    
    acc_mean = acc_series.rolling(window, min_periods=1).mean().values
    acc_std = acc_series.rolling(window, min_periods=1).std().fillna(0.0).values
    gyro_mean = gyro_series.rolling(window, min_periods=1).mean().values
    gyro_std = gyro_series.rolling(window, min_periods=1).std().fillna(0.0).values
    
    df_phase2 = pd.DataFrame({
        'timestamp': df_phase1['timestamp'],
        # RAW SIGNALS
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z,
        # PROCESSED SIGNALS (3-axis gravity removed)
        'acc_x_clean': acc_x_clean,
        'acc_y_clean': acc_y_clean,
        'acc_z_clean': acc_z_clean,
        'gyro_x_clean': gyro_x_clean,
        'gyro_y_clean': gyro_y_clean,
        'gyro_z_clean': gyro_z_clean,
        # DERIVED FEATURES
        'acc_magnitude': acc_magnitude,
        'gyro_magnitude': gyro_magnitude,
        'longitudinal_acc': longitudinal_acc,
        'lateral_acc': lateral_acc,
        'vertical_acc': vertical_acc,
        'yaw_rate': yaw_rate,
        'acc_mean': acc_mean,
        'acc_std': acc_std,
        'gyro_mean': gyro_mean,
        'gyro_std': gyro_std,
        # GROUND TRUTH
        'ground_truth_lat': df_phase1['ground_truth_lat'],
        'ground_truth_lon': df_phase1['ground_truth_lon'],
        'ground_truth_speed': df_phase1['ground_truth_speed'],
        'ground_truth_heading': df_phase1['ground_truth_heading'],
        # SYSTEM STATUS
        'gnss_status': df_phase1['gnss_status']
    })
    
    return df_phase2


if __name__ == "__main__":
    ghosttrack_dir = "/Users/hrithika/Desktop/GhostTrack"
    input_csv = os.path.join(ghosttrack_dir, "data", "processed", "ghosttrack_primary_Vfa01.csv")
    raw_s_path = "/Users/hrithika/Desktop/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-Vfa01.csv"
    output_csv = os.path.join(ghosttrack_dir, "data", "processed", "ghosttrack_phase2_primary.csv")
    
    df_raw, summary = load_and_validate_data(input_csv)
    df_p2 = process_phase2_data(df_raw, raw_s_path)
    
    df_p2.to_csv(output_csv, index=False)
    print(f"\n✅ Updated Phase 2 Processed Dataset Saved (3-Axis Gravity Subtracted): {output_csv}")
    print(f"Shape: {df_p2.shape}")
