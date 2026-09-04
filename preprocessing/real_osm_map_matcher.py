import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ghosttrack_dir = "/Users/hrithika/Desktop/GhostTrack"
roads_json = os.path.join(ghosttrack_dir, "data", "maps", "osm_parsed_roads.json")
p2_csv = os.path.join(ghosttrack_dir, "data", "processed", "ghosttrack_phase2_primary.csv")
results_dir = os.path.join(ghosttrack_dir, "results", "day1")

with open(roads_json) as f:
    osm_ways = json.load(f)

df = pd.read_csv(p2_csv)

# Filter segment of trajectory inside bounding box area
lats = df['ground_truth_lat'].values[::5]   # 2 Hz sampling
lons = df['ground_truth_lon'].values[::5]
headings = df['ground_truth_heading'].values[::5]

# Filter trajectory inside bounding box [52.558, -1.485, 52.570, -1.475]
mask = (lats >= 52.558) & (lats <= 52.570) & (lons >= -1.485) & (lons <= -1.475)
sub_lats = lats[mask]
sub_lons = lons[mask]
sub_headings = headings[mask]

# Add artificial GPS noise + drift (15-30m spatial jitter)
np.random.seed(101)
noisy_lats = sub_lats + np.random.normal(0, 0.00025, len(sub_lats))
noisy_lons = sub_lons + np.random.normal(0, 0.00025, len(sub_lons))

# Disambiguated Candidate Road Matcher
matched_lats = []
matched_lons = []
matched_road_names = []

for p_lon, p_lat, p_head in zip(noisy_lons, noisy_lats, sub_headings):
    best_score = float('inf')
    best_proj = (p_lon, p_lat)
    best_road_name = "Unknown"
    
    p = np.array([p_lon, p_lat])
    
    for way in osm_ways:
        coords = np.array(way['coords'])
        road_name = way['name']
        
        for i in range(len(coords) - 1):
            a = coords[i]
            b = coords[i+1]
            ab = b - a
            ab_sq = np.dot(ab, ab)
            if ab_sq == 0:
                proj = a
                road_heading = p_head
            else:
                t_val = np.clip(np.dot(p - a, ab) / ab_sq, 0.0, 1.0)
                proj = a + t_val * ab
                # Compute road segment angle in degrees
                road_heading = (np.degrees(np.arctan2(ab[0], ab[1])) + 360.0) % 360.0
            
            dist = np.linalg.norm(p - proj)
            
            # Heading delta cost (penalize wrong direction parallel roads)
            head_diff = abs(p_head - road_heading) % 360.0
            head_diff = min(head_diff, 360.0 - head_diff)
            
            # Composite map-matching score: spatial distance + heading penalty
            score = dist + 0.00005 * (head_diff / 180.0)
            
            if score < best_score:
                best_score = score
                best_proj = proj
                best_road_name = road_name
                
    matched_lons.append(best_proj[0])
    matched_lats.append(best_proj[1])
    matched_road_names.append(best_road_name)

# Plot Real OSM Map Matching Feasibility Spike Result
plt.figure(figsize=(10, 8))

# 1. Plot all background candidate OSM roads
road_labeled = False
for way in osm_ways:
    c = np.array(way['coords'])
    label = 'Real OSM Road Network' if not road_labeled else ""
    plt.plot(c[:, 0], c[:, 1], color='gray', lw=1.5, alpha=0.6, label=label)
    road_labeled = True

# 2. Plot Noisy Drifting Trajectory
plt.plot(noisy_lons, noisy_lats, 'r.--', lw=1.2, alpha=0.7, label='Noisy Drifting Trajectory (Raw DR / Outage)')

# 3. Plot Disambiguated Map-Matched Trajectory
plt.plot(matched_lons, matched_lats, 'b.-', lw=2.0, alpha=0.9, label='Matched Trajectory (Snapped to Best OSM Candidate)')

plt.title('REAL OSM MAP MATCHING FEASIBILITY SPIKE (Candidate Disambiguation)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()

out_fig = os.path.join(results_dir, "plot6_map_matching_feasibility.png")
plt.savefig(out_fig, dpi=200)
plt.close()

print(f"✅ Real OSM Map-Matching Feasibility Spike Complete! Plot saved to: {out_fig}")
print(f"Sample Matched OSM Roads: {set(matched_road_names[:10])}")
