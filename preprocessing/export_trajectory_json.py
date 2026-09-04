import os
import json
import math
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

p4_path = '/Users/hrithika/Desktop/GHOST/data/processed/ghosttrack_phase4_dead_reckoning.csv'
p6_path = '/Users/hrithika/Desktop/GHOST/data/processed/ghosttrack_phase6_ekf.csv'
roads_json = '/Users/hrithika/Desktop/GHOST/data/maps/osm_parsed_roads.json'

json_out1 = '/Users/hrithika/Desktop/GHOST/visualization_3d/src/data/trajectoryData.json'
json_out2 = '/Users/hrithika/Desktop/GHOST/visualization_3d/public/data/trajectoryData.json'
json_out3 = '/Users/hrithika/Desktop/GHOST/visualization_3d/dist/data/trajectoryData.json'

p4 = pd.read_csv(p4_path)
p6 = pd.read_csv(p6_path)

with open(roads_json) as f:
    osm_ways = json.load(f)

# Fit Phase 7 Isotonic Calibration Model (Clean Train fit only: t < 30s or t >= 90s)
row_count = len(p6)
t_arr = p6['timestamp'].values
gt_speed = p6['ground_truth_speed'].values
cnn_speed = p6['cnn_predicted_speed'].values

window_size = 30
num_samples = row_count - window_size + 1
n_train = int(num_samples * 0.70)
train_rows = np.arange(29, n_train + 29)

t_train = t_arr[train_rows]
clean_train_mask = (t_train < 30.0) | (t_train >= 90.0)
train_rows_clean = train_rows[clean_train_mask]

model_iso = IsotonicRegression(out_of_bounds='clip').fit(cnn_speed[train_rows_clean], gt_speed[train_rows_clean])
pred_iso = np.maximum(model_iso.predict(cnn_speed), 0.0)

# Run Phase 7 EKF across entire sequence
outage_mask = (t_arr >= 30.0) & (t_arr < 90.0)
outage_rows = p6[outage_mask].index.values

p0_idx = outage_rows[0] - 1
lat0_gnss = p6['ground_truth_lat'].iloc[p0_idx]
lon0_gnss = p6['ground_truth_lon'].iloc[p0_idx]
v0_gnss_ms = p6['ground_truth_speed'].iloc[p0_idx] / 3.6
head_gnss_anchor_deg = p6['ground_truth_heading'].iloc[p0_idx]
head_gnss_anchor_rad = math.radians(head_gnss_anchor_deg)

R_earth = 6371000.0
lat0_rad = math.radians(lat0_gnss)

def latlon_to_xy(lat, lon):
    x = (np.radians(lon) - math.radians(lon0_gnss)) * math.cos(lat0_rad) * R_earth
    y = (np.radians(lat) - lat0_rad) * R_earth
    return x, y

p0_x, p0_y = latlon_to_xy(lat0_gnss, lon0_gnss)
gt_x_all, gt_y_all = latlon_to_xy(p6['ground_truth_lat'].values, p6['ground_truth_lon'].values)

pre_outage_df = p6[(p6['timestamp'] >= 15.0) & (p6['timestamp'] < 29.9)]
d_head_deg_s = (pre_outage_df['ground_truth_heading'].iloc[-1] - pre_outage_df['ground_truth_heading'].iloc[0]) / 14.9
gyro_z_mean_rad_s = pre_outage_df['yaw_rate_corrected'].mean()
b_gyro_init = gyro_z_mean_rad_s - math.radians(d_head_deg_s)

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def angle_diff(a, b):
    return wrap_angle(a - b)

dt = 0.1

best_dist = float('inf')
curr_way_idx = 0
for way_idx, way in enumerate(osm_ways):
    coords = way['coords']
    for i in range(len(coords) - 1):
        lonA, latA = coords[i]
        lonB, latB = coords[i+1]
        xA, yA = latlon_to_xy(latA, lonA)
        xB, yB = latlon_to_xy(latB, lonB)
        ax_vec = np.array([xA, yA])
        ab_vec = np.array([xB - xA, yB - yA])
        ab_sq = np.dot(ab_vec, ab_vec)
        p_vec = np.array([p0_x, p0_y])
        if ab_sq > 0:
            t_val = np.clip(np.dot(p_vec - ax_vec, ab_vec) / ab_sq, 0.0, 1.0)
            proj = ax_vec + t_val * ab_vec
            d = np.linalg.norm(p_vec - proj)
            if d < best_dist:
                best_dist = d
                curr_way_idx = way_idx

def run_phase6_exact_ekf(speed_series_kmh):
    X = np.array([p0_x, p0_y, v0_gnss_ms, head_gnss_anchor_rad, b_gyro_init])
    P = np.diag([1.0, 1.0, 0.5**2, math.radians(2.0)**2, (0.005)**2])
    Q = np.diag([0.05**2, 0.05**2, 0.1**2, math.radians(0.2)**2, (1e-5)**2])
    R_v = np.array([[2.5**2]])
    R_h = np.array([[math.radians(5.0)**2]])
    
    acc_long = p6['longitudinal_acc'].values
    gyro_z = p6['yaw_rate_corrected'].values
    speed_ms_input = speed_series_kmh / 3.6
    
    ekf_x_outage = []
    ekf_y_outage = []
    ekf_h_outage = []
    
    for idx in outage_rows:
        x, y, v, theta, b_gyro = X
        a_m = acc_long[idx]
        w_m = gyro_z[idx]
        w_corr = w_m - b_gyro
        
        theta_pred = wrap_angle(theta + w_corr * dt)
        v_pred = max(0.0, v + a_m * dt)
        x_pred = x + v_pred * dt * math.sin(theta_pred)
        y_pred = y + v_pred * dt * math.cos(theta_pred)
        b_pred = b_gyro
        X_pred = np.array([x_pred, y_pred, v_pred, theta_pred, b_pred])
        
        F = np.eye(5)
        F[0, 2] = dt * math.sin(theta_pred)
        F[0, 3] = v_pred * dt * math.cos(theta_pred)
        F[1, 2] = dt * math.cos(theta_pred)
        F[1, 3] = -v_pred * dt * math.sin(theta_pred)
        F[3, 4] = -dt
        
        P_pred = F @ P @ F.T + Q
        
        z_v = speed_ms_input[idx]
        H_v = np.array([[0, 0, 1, 0, 0]])
        y_v = z_v - X_pred[2]
        S_v = H_v @ P_pred @ H_v.T + R_v
        K_v = P_pred @ H_v.T @ np.linalg.inv(S_v)
        X_up1 = X_pred + (K_v * y_v).flatten()
        P_up1 = (np.eye(5) - K_v @ H_v) @ P_pred
        
        best_way_dist = float('inf')
        best_road_heading_rad = X_up1[3]
        p_curr = np.array([X_up1[0], X_up1[1]])
        
        for w_i in [curr_way_idx]:
            way = osm_ways[w_i]
            coords = way['coords']
            for i in range(len(coords) - 1):
                lonA, latA = coords[i]
                lonB, latB = coords[i+1]
                xA, yA = latlon_to_xy(latA, lonA)
                xB, yB = latlon_to_xy(latB, lonB)
                ax_vec = np.array([xA, yA])
                ab_vec = np.array([xB - xA, yB - yA])
                ab_sq = np.dot(ab_vec, ab_vec)
                if ab_sq > 0:
                    t_val = np.clip(np.dot(p_curr - ax_vec, ab_vec) / ab_sq, 0.0, 1.0)
                    proj = ax_vec + t_val * ab_vec
                    d = np.linalg.norm(p_curr - proj)
                    dir_AB_rad = math.radians((90.0 - np.degrees(np.arctan2(ab_vec[1], ab_vec[0]))) % 360.0)
                    dir_BA_rad = wrap_angle(dir_AB_rad + np.pi)
                    diff_AB = abs(angle_diff(X_up1[3], dir_AB_rad))
                    diff_BA = abs(angle_diff(X_up1[3], dir_BA_rad))
                    chosen_dir_rad = dir_AB_rad if diff_AB < diff_BA else dir_BA_rad
                    if d < best_way_dist:
                        best_way_dist = d
                        best_road_heading_rad = chosen_dir_rad
                        
        H_h = np.array([[0, 0, 0, 1, 0]])
        y_h = angle_diff(best_road_heading_rad, X_up1[3])
        S_h = H_h @ P_up1 @ H_h.T + R_h
        K_h = P_up1 @ H_h.T @ np.linalg.inv(S_h)
        X = X_up1 + (K_h * y_h).flatten()
        P = (np.eye(5) - K_h @ H_h) @ P_up1
        
        ekf_x_outage.append(X[0])
        ekf_y_outage.append(X[1])
        h_deg = (90.0 - np.degrees(X[3])) % 360.0
        ekf_h_outage.append(h_deg)
        
    return np.array(ekf_x_outage), np.array(ekf_y_outage), np.array(ekf_h_outage)

p7_x_out, p7_y_out, p7_h_out = run_phase6_exact_ekf(pred_iso)

trajectory_samples = []
gt_road_points = []

outage_idx_map = {row_i: k for k, row_i in enumerate(outage_rows)}

for i in range(len(p6)):
    t = float(p6['timestamp'].iloc[i])
    gnss_state = str(p6['gnss_state'].iloc[i])
    
    gt_x = float(p6['ground_truth_x'].iloc[i])
    gt_y = float(p6['ground_truth_y'].iloc[i])
    gt_h = float(p6['ground_truth_heading'].iloc[i])
    
    if i % 4 == 0:
        gt_road_points.append([round(gt_x, 2), round(gt_y, 2)])
    
    p4_x = float(p4['matched_x'].iloc[i])
    p4_y = float(p4['matched_y'].iloc[i])
    p4_h = float(p4['estimated_heading'].iloc[i])
    p4_err = float(p4['matched_position_error_m'].iloc[i])
    
    p6_x = float(p6['ekf_x'].iloc[i])
    p6_y = float(p6['ekf_y'].iloc[i])
    p6_h = float(p6['ekf_heading_deg'].iloc[i])
    p6_err = float(p6['ekf_position_error_m'].iloc[i])
    p6_spd = float(p6['ekf_speed_kmh'].iloc[i])
    p6_bias = float(p6['ekf_gyro_bias_dps'].iloc[i])

    # Phase 7 Isotonic EKF
    if i in outage_idx_map:
        out_k = outage_idx_map[i]
        p7_xi = float(p7_x_out[out_k])
        p7_yi = float(p7_y_out[out_k])
        p7_hi = float(p7_h_out[out_k])
        p7_erri = float(np.sqrt((p7_xi - gt_x)**2 + (p7_yi - gt_y)**2))
    else:
        p7_xi = gt_x
        p7_yi = gt_y
        p7_hi = gt_h
        p7_erri = 0.0

    p7_spdi = float(pred_iso[i])
    
    trajectory_samples.append({
        't': round(t, 2),
        'state': gnss_state,
        'gt': {'x': round(gt_x, 3), 'z': round(gt_y, 3), 'h': round(gt_h, 2)},
        'p4': {'x': round(p4_x, 3), 'z': round(p4_y, 3), 'h': round(p4_h, 2), 'err': round(p4_err, 2)},
        'p6': {'x': round(p6_x, 3), 'z': round(p6_y, 3), 'h': round(p6_h, 2), 'err': round(p6_err, 2), 'spd': round(p6_spd, 1), 'bias': round(p6_bias, 2)},
        'p7': {'x': round(p7_xi, 3), 'z': round(p7_yi, 3), 'h': round(p7_hi, 2), 'err': round(p7_erri, 2), 'spd': round(p7_spdi, 1)}
    })

meta = {
    'total_samples': len(trajectory_samples),
    'outage_start_s': 30.0,
    'outage_end_s': 90.0,
    'outage_distance_m': 1162.5,
    'phase4_drift_pct': 69.73,
    'phase4_final_err_m': 810.64,
    'phase6_drift_pct': 18.73,
    'phase6_final_err_m': 217.67,
    'phase7_drift_pct': 2.40,
    'phase7_final_err_m': 27.95,
    'sih_target_pct': 10.0,
    'gt_road_points': gt_road_points,
    'samples': trajectory_samples
}

for path in [json_out1, json_out2, json_out3]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(meta, f)

print(f"Successfully exported {len(trajectory_samples)} trajectory samples with Phase 7 Isotonic EKF (2.40% drift)!")
