import os
import subprocess
import pandas as pd
import numpy as np
import urllib.parse

base_dir = "/Users/hrithika/Desktop/IO-VNBD"
ghosttrack_dir = "/Users/hrithika/Desktop/GhostTrack"

def ensure_lfs_downloaded(rel_path, local_path):
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 500:
        url_rel = urllib.parse.quote(rel_path.replace("\\", "/"))
        url = f"https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/{url_rel}"
        print(f"Downloading LFS file: {rel_path}...")
        subprocess.run(["curl", "-k", "-s", "-L", url, "-o", local_path], check=False)

print("=== PART 1 & 2: INSPECTING KEY IO-VNBD SEQUENCES ===")

uncat_s_base = os.path.join(base_dir, "Synchronised V abd S datasets", "Uncategorised IOVNB Dataset", "S-Dataset")
uncat_v_base = os.path.join(base_dir, "Synchronised V abd S datasets", "Uncategorised IOVNB Dataset", "V-Dataset")

# Key sequences to inspect
target_seq_ids = ["Vfa01", "Vfa02", "Vta1", "Vta2", "Vtb1", "Vw1", "Vw2", "S1", "S2", "S3a"]

sequences_summary = []

for seq_id in target_seq_ids:
    s_fname = f"S-{seq_id}.csv" if not seq_id.startswith("S") else f"S-{seq_id}.csv"
    v_fname = f"V-{seq_id}.csv"
    
    # Handle naming variations in dataset
    s_full = os.path.join(uncat_s_base, s_fname)
    v_full = os.path.join(uncat_v_base, v_fname)
    
    if not os.path.exists(s_full):
        # Check lower case variations
        s_fname = f"S-{seq_id.lower()}.csv"
        v_fname = f"V-{seq_id.lower()}.csv"
        s_full = os.path.join(uncat_s_base, s_fname)
        v_full = os.path.join(uncat_v_base, v_fname)

    if os.path.exists(s_full) and os.path.exists(v_full):
        s_rel = os.path.relpath(s_full, base_dir)
        v_rel = os.path.relpath(v_full, base_dir)
        
        ensure_lfs_downloaded(s_rel, s_full)
        ensure_lfs_downloaded(v_rel, v_full)
        
        try:
            df_s = pd.read_csv(s_full, encoding='latin1')
            df_v = pd.read_csv(v_full)
            df_s.columns = df_s.columns.str.strip()
            df_v.columns = df_v.columns.str.strip()
            
            s_rows = len(df_s)
            v_rows = len(df_v)
            
            # Data quality metrics
            nulls_s = df_s[['ACCELEROMETER X (m/s²)', 'ACCELEROMETER Y (m/s²)', 'ACCELEROMETER Z (m/s²)']].isna().sum().sum()
            nulls_v = df_v[['Latitude (degrees)', 'Longitude (degrees)', 'Velocity (km/hr)']].isna().sum().sum()
            
            lat_min, lat_max = df_v['Latitude (degrees)'].min(), df_v['Latitude (degrees)'].max()
            lon_min, lon_max = df_v['Longitude (degrees)'].min(), df_v['Longitude (degrees)'].max()
            speed_max = df_v['Velocity (km/hr)'].max()
            
            duration_min = round(min(s_rows, v_rows) * 0.1 / 60, 2)
            
            # Check route variation (has turns and speed range)
            heading_std = df_v['Heading (degrees)'].std()
            is_suitable = (s_rows > 1000) and (nulls_s == 0) and (nulls_v == 0) and (heading_std > 20)
            
            sequences_summary.append({
                'seq_id': seq_id,
                's_file': s_fname,
                'v_file': v_fname,
                'rows': min(s_rows, v_rows),
                'duration_min': duration_min,
                'sampling_rate': '10 Hz',
                'speed_max_kmh': round(speed_max, 1),
                'heading_std': round(heading_std, 1),
                'nulls': nulls_s + nulls_v,
                'suitable': 'YES' if is_suitable else 'NO'
            })
        except Exception as e:
            print(f"Error reading sequence {seq_id}: {e}")

df_res = pd.DataFrame(sequences_summary)
print("\n--- GHOSTTRACK DAY 1: SEQUENCE EVALUATION TABLE ---")
print(df_res.to_string(index=False))

# Save output table
df_res.to_csv(os.path.join(ghosttrack_dir, "results", "day1", "sequences_summary.csv"), index=False)
