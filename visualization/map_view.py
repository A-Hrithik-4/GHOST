import plotly.graph_objects as go
import matplotlib.pyplot as plt

def create_interactive_map(df, current_step=0, gnss_status=1):
    """
    Renders an interactive Plotly map displaying:
    - Complete route polyline
    - Start (Green) & End (Red) markers
    - Active Vehicle position marker (Blue car / arrow)
    - GNSS Blackout highlight segment
    """
    lats = df['ground_truth_lat'].values
    lons = df['ground_truth_lon'].values
    status_list = df['gnss_status'].values
    
    # Cap current_step
    current_step = max(0, min(current_step, len(lats) - 1))
    
    curr_lat = lats[current_step]
    curr_lon = lons[current_step]
    curr_status = status_list[current_step]
    
    fig = go.Figure()
    
    # 1. Complete Route Trace
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='lines',
        line=dict(width=4, color='rgb(41, 128, 185)'),
        name='Ground Truth Route',
        hoverinfo='skip'
    ))
    
    # 2. Highlight Outage Segments
    outage_mask = (status_list == 0)
    if outage_mask.any():
        outage_lats = np.where(outage_mask, lats, None)
        outage_lons = np.where(outage_mask, lons, None)
        fig.add_trace(go.Scattermapbox(
            lat=outage_lats,
            lon=outage_lons,
            mode='lines',
            line=dict(width=6, color='rgb(231, 76, 60)'),
            name='GNSS Outage Segment',
            hoverinfo='skip'
        ))
    
    # 3. Start & End Markers
    fig.add_trace(go.Scattermapbox(
        lat=[lats[0]],
        lon=[lons[0]],
        mode='markers+text',
        marker=dict(size=12, color='green'),
        text=['START'],
        textposition='top center',
        name='Start'
    ))
    
    fig.add_trace(go.Scattermapbox(
        lat=[lats[-1]],
        lon=[lons[-1]],
        mode='markers+text',
        marker=dict(size=12, color='red'),
        text=['END'],
        textposition='top center',
        name='End'
    ))
    
    # 4. Current Vehicle Marker
    marker_color = 'rgb(46, 204, 113)' if curr_status == 1 else 'rgb(231, 76, 60)'
    marker_label = '🚘 Active Vehicle (GNSS ON)' if curr_status == 1 else '🚨 Active Vehicle (GNSS OUTAGE)'
    
    fig.add_trace(go.Scattermapbox(
        lat=[curr_lat],
        lon=[curr_lon],
        mode='markers',
        marker=dict(size=18, color=marker_color),
        name=marker_label
    ))
    
    # Map Layout Configuration
    center_lat = curr_lat
    center_lon = curr_lon
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=14
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        showlegend=True
    )
    
    return fig


def plot_static_map(df, current_step=0, save_path=None):
    """
    Renders a Matplotlib fallback map figure.
    """
    lats = df['ground_truth_lat'].values
    lons = df['ground_truth_lon'].values
    status_list = df['gnss_status'].values
    
    curr_lat = lats[current_step]
    curr_lon = lons[current_step]
    curr_status = status_list[current_step]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(lons, lats, color='royalblue', lw=2.5, label='Route')
    
    # Plot Outage
    outage_mask = (status_list == 0)
    if outage_mask.any():
        ax.plot(lons[outage_mask], lats[outage_mask], color='crimson', lw=4, label='GNSS Outage Window')
        
    ax.scatter(lons[0], lats[0], color='green', s=100, label='Start')
    ax.scatter(lons[-1], lats[-1], color='red', s=100, label='End')
    
    veh_color = 'limegreen' if curr_status == 1 else 'orange'
    ax.scatter(curr_lon, curr_lat, color=veh_color, edgecolor='black', s=180, zorder=10, label='Active Vehicle')
    
    ax.set_title(f'GhostTrack Vehicle Location (Step {current_step} - GNSS {"AVAILABLE" if curr_status == 1 else "OUTAGE"})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    return fig
