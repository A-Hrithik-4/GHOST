import json
import requests

bbox = [52.558, -1.485, 52.585, -1.470]
url = "https://overpass.kumi.systems/api/interpreter"
headers = {'User-Agent': 'GhostTrack-SIH26168/1.0'}

query = f"""
[out:json][timeout:15];
(
  way["highway"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
);
out body;
>;
out skel qt;
"""

print(f"Fetching from Kumi mirror for {bbox}...")
res = requests.post(url, data={'data': query}, headers=headers, timeout=15)
res.raise_for_status()
data = res.json()

out_file = "/Users/hrithika/Desktop/GhostTrack/data/maps/osm_roads_vfa01.json"
with open(out_file, "w") as f:
    json.dump(data, f)

elements = data.get('elements', [])
ways = [e for e in elements if e.get('type') == 'way']
nodes = {e['id']: (e['lon'], e['lat']) for e in elements if e.get('type') == 'node'}

print(f"✅ SUCCESS! Fetched {len(ways)} real OSM highway road segments and {len(nodes)} nodes.")
print(f"Saved OSM map file: {out_file}")
