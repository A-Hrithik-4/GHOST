import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="GHOST — Intelligent GNSS-Free Fallback Navigation",
    page_icon="🚘",
    layout="wide"
)

st.title("🚘 GHOST — GNSS-Free Hybrid Onboard Sensor Tracker")
st.caption("Smart India Hackathon (SIH PS 26168) — ISRO | Phase 6 EKF Sensor Fusion Active Prototype")

@st.cache_data
def load_data():
    p6_path = "/Users/hrithika/Desktop/GhostTrack/data/processed/ghosttrack_phase6_ekf.csv"
    p5_path = "/Users/hrithika/Desktop/GhostTrack/data/processed/ghosttrack_phase5_refined.csv"
    p4_path = "/Users/hrithika/Desktop/GhostTrack/data/processed/ghosttrack_phase4_dead_reckoning.csv"
    
    if os.path.exists(p6_path):
        df = pd.read_csv(p6_path)
    elif os.path.exists(p5_path):
        df = pd.read_csv(p5_path)
    elif os.path.exists(p4_path):
        df = pd.read_csv(p4_path)
    else:
        df = pd.read_csv("/Users/hrithika/Desktop/GhostTrack/data/processed/ghosttrack_phase2_primary.csv")
    return df

df = load_data()

st.sidebar.header("🕹️ Simulation & Model Controls")
step_index = st.sidebar.slider(
    "Timeline Position (Time Step)",
    min_value=0,
    max_value=len(df) - 1,
    value=500,  # t = 50s (Outage Active)
    step=10,
    format="Step %d"
)

active_traj = st.sidebar.selectbox(
    "Active Vehicle Tracker Mode",
    ["Phase 6 EKF Sensor Fusion (18.7% Drift)", "Phase 5.2 Continuous Road-Constrained (53.8% Drift)", "Phase 4 Baseline DR (69.7% Drift)"]
)

row = df.iloc[step_index]
curr_time = row['timestamp']
gnss_state = row.get('gnss_state', 'AVAILABLE')

st.markdown("---")
c_status, c_time, c_drift, c_bias = st.columns([2, 1, 1, 1])

with c_status:
    if gnss_state == 'AVAILABLE':
        st.success("🛰️ **GNSS STATUS: AVAILABLE** (Signal Locked — 10 Hz Fix)")
    elif gnss_state == 'OUTAGE':
        st.error("🚨 **GNSS STATUS: OUTAGE DETECTED** (GHOST Active — 5-State EKF + AI Speed + OSM Constraint)")
    else:
        st.info("🔄 **GNSS STATUS: RESTORED** (Signal Re-aligned)")

with c_time:
    st.metric("Timestamp", f"{curr_time:.1f} s", delta=f"{curr_time/60:.2f} min")

with c_drift:
    if 'ekf_position_error_m' in df.columns and gnss_state == 'OUTAGE':
        err_m = row['ekf_position_error_m']
        st.metric("EKF Position Error", f"{err_m:.1f} m")
    elif 'matched_position_error_m' in df.columns and gnss_state == 'OUTAGE':
        err_m = row['matched_position_error_m']
        st.metric("Position Error", f"{err_m:.1f} m")
    else:
        st.metric("Position Error", "0.0 m")

with c_bias:
    if 'ekf_gyro_bias_dps' in df.columns:
        b_dps = row['ekf_gyro_bias_dps']
        st.metric("EKF Gyro Bias", f"{b_dps:.2f} °/s")
    else:
        st.metric("Gyro Bias", "0.0 °/s")

map_col, panel_col = st.columns([2, 1])

with map_col:
    st.subheader("🗺️ Real-Time Navigation Map (OpenStreetMap Highway Geometry)")
    
    lats = df['ground_truth_lat'].values
    lons = df['ground_truth_lon'].values
    
    fig = go.Figure()
    
    # 1. Ground Truth Route
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='lines',
        line=dict(width=3, color='rgb(41, 128, 185)'),
        name='Ground Truth Route'
    ))
    
    outage_mask = (df['gnss_state'] == 'OUTAGE')
    
    # 2. Phase 4 Baseline
    if 'matched_lat' in df.columns:
        fig.add_trace(go.Scattermapbox(
            lat=df['matched_lat'][outage_mask],
            lon=df['matched_lon'][outage_mask],
            mode='lines',
            line=dict(width=3, color='rgb(231, 76, 60)', dash='dash'),
            name='Phase 4 Baseline (69.7% Drift)'
        ))
        
    # 3. Phase 6 EKF Trajectory
    if 'ekf_lat' in df.columns:
        fig.add_trace(go.Scattermapbox(
            lat=df['ekf_lat'][outage_mask],
            lon=df['ekf_lon'][outage_mask],
            mode='lines',
            line=dict(width=5, color='rgb(46, 204, 113)'),
            name='Phase 6 EKF Fusion (18.7% Drift)'
        ))
        
    # Active Vehicle Marker
    if "Phase 6" in active_traj and 'ekf_lat' in row:
        curr_lat, curr_lon = row['ekf_lat'], row['ekf_lon']
    elif 'matched_lat' in row:
        curr_lat, curr_lon = row['matched_lat'], row['matched_lon']
    else:
        curr_lat, curr_lon = row['ground_truth_lat'], row['ground_truth_lon']
        
    veh_color = 'rgb(46, 204, 113)' if gnss_state == 'AVAILABLE' else 'rgb(231, 76, 60)'
    
    fig.add_trace(go.Scattermapbox(
        lat=[curr_lat],
        lon=[curr_lon],
        mode='markers',
        marker=dict(size=18, color=veh_color),
        name='Active Vehicle Position'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=curr_lat, lon=curr_lon),
            zoom=14
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=520,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

with panel_col:
    st.subheader("📊 5-State EKF & Sensor Telemetry")
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Longitudinal Accel", f"{row['longitudinal_acc']:.2f} m/s²")
        st.metric("Yaw Rate (Corrected)", f"{row['yaw_rate_corrected']*(180/np.pi):.1f} °/s" if 'yaw_rate_corrected' in row else f"{row['yaw_rate']*(180/np.pi):.1f} °/s")
    with m2:
        st.metric("Lateral Accel", f"{row['lateral_acc']:.2f} m/s²")
        st.metric("EKF Heading State", f"{row['ekf_heading_deg']:.1f} °" if 'ekf_heading_deg' in row else f"{row['ground_truth_heading']:.1f} °")
        
    st.markdown("---")
    st.subheader("🤖 Speed Fusion & AI Estimation")
    k1, k2 = st.columns(2)
    with k1:
        ekf_spd = row.get('ekf_speed_kmh', row['ground_truth_speed'])
        st.metric("EKF Fused Speed", f"{ekf_spd:.1f} km/h")
    with k2:
        cnn_spd = row.get('cnn_predicted_speed', row['ground_truth_speed'])
        st.metric("CNN AI Speed", f"{cnn_spd:.1f} km/h")
        
    st.info("💡 **GHOST Active:** 5-State Extended Kalman Filter fusing IMU accelerations, online gyro bias tracking, 1D-CNN speed updates, and circular OpenStreetMap road heading constraints.")

st.markdown("---")
st.caption("GHOST — GNSS-Free Hybrid Onboard Sensor Tracker | Developed for SIH ISRO PS 26168")
