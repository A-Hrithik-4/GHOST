import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ghosttrack_dir = "/Users/hrithika/Desktop/GhostTrack"
p2_csv = os.path.join(ghosttrack_dir, "data", "processed", "ghosttrack_phase2_primary.csv")
out_dir = os.path.join(ghosttrack_dir, "results", "day2")
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(p2_csv)
t = df['timestamp'] / 60.0  # minutes

print("=== PART 8: GENERATING SIGNAL QUALITY PLOTS ===")

# Plot 1: Raw vs Processed Accelerometer
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t, df['acc_x'], color='lightgray', label='Raw Acc X', alpha=0.8)
axes[0].plot(t, df['acc_x_clean'], color='crimson', label='Clean Acc X (3Hz LP)', lw=1.2)
axes[0].set_ylabel('Acc X (m/s²)')
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].plot(t, df['acc_y'], color='lightgray', label='Raw Acc Y', alpha=0.8)
axes[1].plot(t, df['acc_y_clean'], color='forestgreen', label='Clean Acc Y (3Hz LP)', lw=1.2)
axes[1].set_ylabel('Acc Y (m/s²)')
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.5)

axes[2].plot(t, df['acc_z'] - 9.80665, color='lightgray', label='Raw Acc Z (Grav Removed)', alpha=0.8)
axes[2].plot(t, df['acc_z_clean'], color='royalblue', label='Clean Acc Z (3Hz LP)', lw=1.2)
axes[2].set_ylabel('Acc Z Dyn (m/s²)')
axes[2].set_xlabel('Time (minutes)')
axes[2].legend(loc='upper right')
axes[2].grid(True, linestyle='--', alpha=0.5)

fig.suptitle('Plot 1: Raw vs Filtered Accelerometer Signals (3Hz Butterworth LP Filter)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot1_acc_raw_vs_clean.png"), dpi=200)
plt.close()

# Plot 2: Raw vs Processed Gyroscope
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t, df['gyro_x'], color='lightgray', label='Raw Gyro X', alpha=0.8)
axes[0].plot(t, df['gyro_x_clean'], color='crimson', label='Clean Gyro X (Unbiased & LP)', lw=1.2)
axes[0].set_ylabel('Gyro X (rad/s)')
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].plot(t, df['gyro_y'], color='lightgray', label='Raw Gyro Y', alpha=0.8)
axes[1].plot(t, df['gyro_y_clean'], color='forestgreen', label='Clean Gyro Y (Unbiased & LP)', lw=1.2)
axes[1].set_ylabel('Gyro Y (rad/s)')
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.5)

axes[2].plot(t, df['gyro_z'], color='lightgray', label='Raw Gyro Z', alpha=0.8)
axes[2].plot(t, df['gyro_z_clean'], color='royalblue', label='Clean Gyro Z (Unbiased & LP)', lw=1.2)
axes[2].set_ylabel('Gyro Z (rad/s)')
axes[2].set_xlabel('Time (minutes)')
axes[2].legend(loc='upper right')
axes[2].grid(True, linestyle='--', alpha=0.5)

fig.suptitle('Plot 2: Raw vs Filtered Gyroscope Signals (3Hz Butterworth LP Filter)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot2_gyro_raw_vs_clean.png"), dpi=200)
plt.close()

# Plot 3: Acceleration Magnitude
plt.figure(figsize=(10, 4))
plt.plot(t, df['acc_magnitude'], color='darkmagenta', lw=1.2, label='Filtered Acceleration Magnitude')
plt.title('Plot 3: Total Dynamic Acceleration Magnitude vs Time')
plt.xlabel('Time (minutes)')
plt.ylabel('Magnitude (m/s²)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot3_acc_magnitude.png"), dpi=200)
plt.close()

# Plot 4: Longitudinal vs Lateral Acceleration
plt.figure(figsize=(10, 4))
plt.plot(t, df['longitudinal_acc'], color='darkorange', lw=1.2, label='Longitudinal Acc (Propulsion / Braking)')
plt.plot(t, df['lateral_acc'], color='teal', lw=1.2, label='Lateral Acc (Turning / Centripetal)')
plt.title('Plot 4: Vehicle Frame Acceleration Breakdown (Longitudinal vs Lateral)')
plt.xlabel('Time (minutes)')
plt.ylabel('Acceleration (m/s²)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot4_long_vs_lat_acc.png"), dpi=200)
plt.close()

# Plot 5: Yaw Rate
plt.figure(figsize=(10, 4))
plt.plot(t, df['yaw_rate'] * (180.0 / np.pi), color='firebrick', lw=1.2, label='Vehicle Yaw Rate (deg/s)')
plt.title('Plot 5: Vehicle Yaw Rate (Heading Change Rate) vs Time')
plt.xlabel('Time (minutes)')
plt.ylabel('Yaw Rate (deg/sec)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot5_yaw_rate.png"), dpi=200)
plt.close()

print("✅ Saved all 5 signal comparison plots to GhostTrack/results/day2/")
