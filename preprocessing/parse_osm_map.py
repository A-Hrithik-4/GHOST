import xml.etree.ElementTree as ET
import json
import os
import numpy as np

xml_file = "/Users/hrithika/Desktop/GhostTrack/data/maps/osm_raw_map.xml"
tree = ET.parse(xml_file)
root = tree.getroot()

nodes = {}
for node in root.findall('node'):
    node_id = int(node.attrib['id'])
    lat = float(node.attrib['lat'])
    lon = float(node.attrib['lon'])
    nodes[node_id] = (lon, lat)

ways = []
for way in root.findall('way'):
    is_highway = False
    name = "Unnamed Road"
    highway_type = ""
    
    for tag in way.findall('tag'):
        k = tag.attrib.get('k')
        v = tag.attrib.get('v')
        if k == 'highway':
            is_highway = True
            highway_type = v
        if k == 'name':
            name = v
            
    if is_highway and highway_type not in ['footway', 'steps', 'path', 'cycleway', 'pedestrian']:
        nd_refs = [int(nd.attrib['ref']) for nd in way.findall('nd') if int(nd.attrib['ref']) in nodes]
        if len(nd_refs) >= 2:
            coords = [nodes[ref] for ref in nd_refs]
            ways.append({
                'id': int(way.attrib['id']),
                'name': name,
                'type': highway_type,
                'coords': coords
            })

print(f"✅ Successfully parsed OSM XML map! Extracted {len(ways)} real road ways and {len(nodes)} node coordinates.")
out_json = "/Users/hrithika/Desktop/GhostTrack/data/maps/osm_parsed_roads.json"
with open(out_json, "w") as f:
    json.dump(ways, f)

print(f"Saved parsed road graph to: {out_json}")
