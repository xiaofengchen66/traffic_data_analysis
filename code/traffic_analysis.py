#!/usr/bin/env python3
"""
Telematics Data Analysis Script
Data: 2022-09-28_00.csv (Buffalo, NY area)

Analysis:
  1. Driving Behavior — Hard braking & hard acceleration distribution
  2. Trip Patterns    — Departure time, duration, distance
  3. Geo Heatmap      — Event density by area
  4. Safety           — Seat belt usage, speeding
"""

import pandas as pd
import numpy as np
import json
import csv as csv_mod
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, cos, sin, asin, sqrt
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================
# 0. Data Loading & Preprocessing
# ============================================================
print("=" * 60)
print("Step 0: Data Loading & Preprocessing")
print("=" * 60)

CSV_PATH = "../data/2022-09-28_00.csv"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = []
with open(CSV_PATH, 'r') as f:
    reader = csv_mod.reader(f, escapechar='\\')
    header = next(reader)
    for row in reader:
        if len(row) == len(header):
            rows.append(row)
df = pd.DataFrame(rows, columns=header)
print(f"Raw data: {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# --- Parse nested JSON columns ---
def safe_json_parse(val):
    if pd.isna(val):
        return {}
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return {}

print("\nParsing nested JSON columns (location, metrics, status, event)...")

loc_parsed = df['location'].apply(safe_json_parse)
df['latitude'] = loc_parsed.apply(lambda x: x.get('latitude'))
df['longitude'] = loc_parsed.apply(lambda x: x.get('longitude'))
df['postalCode'] = loc_parsed.apply(lambda x: x.get('postalCode'))
df['regionCode'] = loc_parsed.apply(lambda x: x.get('regionCode'))

met_parsed = df['metrics'].apply(safe_json_parse)
df['speed'] = met_parsed.apply(lambda x: x.get('speed'))
df['heading'] = met_parsed.apply(lambda x: x.get('heading'))
df['odometer'] = met_parsed.apply(lambda x: x.get('odometer'))
df['accelerationX'] = met_parsed.apply(lambda x: x.get('accelerationX'))
df['accelerationY'] = met_parsed.apply(lambda x: x.get('accelerationY'))
df['fuelConsumption'] = met_parsed.apply(lambda x: x.get('fuelConsumption'))
df['exteriorTemp'] = met_parsed.apply(
    lambda x: x.get('environment', {}).get('exteriorTemperature'))

stat_parsed = df['status'].apply(safe_json_parse)
df['ignitionState'] = stat_parsed.apply(lambda x: x.get('ignitionState'))

evt_parsed = df['event'].apply(safe_json_parse)
df['eventType'] = evt_parsed.apply(lambda x: x.get('eventType'))
df['eventMetadata'] = evt_parsed.apply(lambda x: x.get('eventMetadata', {}))

df['capturedDateTime'] = pd.to_datetime(df['capturedDateTime'])
df['hour'] = df['capturedDateTime'].dt.hour
df['minute'] = df['capturedDateTime'].dt.minute

for col in ['latitude', 'longitude', 'speed', 'heading', 'odometer',
            'accelerationX', 'accelerationY', 'fuelConsumption', 'exteriorTemp']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("Parsing complete!")
print(f"\nEvent Type Distribution:")
print(df['eventType'].value_counts().to_string())
print(f"\nSpeed field non-null count: {df['speed'].notna().sum()}")
print(f"Odometer field non-null count: {df['odometer'].notna().sum()}")


# ============================================================
# 1. Driving Behavior — Hard Brake & Hard Acceleration
# ============================================================
print("\n" + "=" * 60)
print("Step 1: Driving Behavior — Hard Brake & Hard Acceleration")
print("=" * 60)

accel_events = df[df['eventType'] == 'ACCELERATION_CHANGE'].copy()
accel_events['accelerationType'] = accel_events['eventMetadata'].apply(
    lambda x: x.get('accelerationType') if isinstance(x, dict) else None)

print(f"\nTotal acceleration events: {len(accel_events)}")
print(f"Acceleration type distribution:")
print(accel_events['accelerationType'].value_counts().to_string())

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Driving Behavior — Hard Brake & Hard Acceleration', fontsize=16, fontweight='bold')

# 1a. Pie chart
ax = axes[0, 0]
type_counts = accel_events['accelerationType'].value_counts()
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
       colors=colors[:len(type_counts)], startangle=90)
ax.set_title('Hard Accel / Hard Brake Proportion')

# 1b. Hourly distribution
ax = axes[0, 1]
for atype in accel_events['accelerationType'].unique():
    subset = accel_events[accel_events['accelerationType'] == atype]
    hour_counts = subset.groupby('hour').size()
    ax.bar(hour_counts.index, hour_counts.values, alpha=0.6, label=atype)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Event Count')
ax.set_title('Hard Brake / Hard Accel by Hour')
ax.legend()
ax.set_xticks(range(0, 24))

# 1c. Hard brake scatter
ax = axes[1, 0]
hard_brake = accel_events[accel_events['accelerationType'] == 'HARD_BRAKE']
ax.scatter(hard_brake['longitude'], hard_brake['latitude'],
           c='red', alpha=0.4, s=10, edgecolors='none')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(f'Hard Brake Locations (n={len(hard_brake)})')

# 1d. Hard accel scatter
ax = axes[1, 1]
hard_accel = accel_events[accel_events['accelerationType'] == 'HARD_ACCELERATION']
ax.scatter(hard_accel['longitude'], hard_accel['latitude'],
           c='orange', alpha=0.4, s=10, edgecolors='none')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(f'Hard Acceleration Locations (n={len(hard_accel)})')

plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, '1_driving_behavior.png')
plt.savefig(path1)
plt.close()
print(f"\nChart saved: {path1}")

print("\nHard Brake TOP 10 Postal Code Hotspots:")
print(hard_brake['postalCode'].value_counts().head(10).to_string())
print("\nHard Accel TOP 10 Postal Code Hotspots:")
print(hard_accel['postalCode'].value_counts().head(10).to_string())


# ============================================================
# 2. Trip Patterns
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Trip Pattern Analysis")
print("=" * 60)

journey_events = df[df['eventType'] == 'JOURNEY'].copy()
journey_events['journeyEventType'] = journey_events['eventMetadata'].apply(
    lambda x: x.get('journeyEventType') if isinstance(x, dict) else None)

starts = journey_events[journey_events['journeyEventType'] == 'START'].copy()
ends = journey_events[journey_events['journeyEventType'] == 'END'].copy()
print(f"Journey START events: {len(starts)}")
print(f"Journey END events: {len(ends)}")

starts_agg = starts.groupby('journeyId').agg(
    start_time=('capturedDateTime', 'first'),
    start_lat=('latitude', 'first'),
    start_lon=('longitude', 'first'),
    start_odo=('odometer', 'first')
).reset_index()

ends_agg = ends.groupby('journeyId').agg(
    end_time=('capturedDateTime', 'first'),
    end_lat=('latitude', 'first'),
    end_lon=('longitude', 'first'),
    end_odo=('odometer', 'first')
).reset_index()

trips = starts_agg.merge(ends_agg, on='journeyId', how='inner')
trips['duration_min'] = (trips['end_time'] - trips['start_time']).dt.total_seconds() / 60
trips['odo_distance_km'] = trips['end_odo'] - trips['start_odo']
trips['start_hour'] = trips['start_time'].dt.hour

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))

trips['straight_dist_km'] = trips.apply(
    lambda r: haversine(r['start_lon'], r['start_lat'], r['end_lon'], r['end_lat'])
    if pd.notna(r['start_lon']) else np.nan, axis=1)

valid_trips = trips[
    (trips['duration_min'] > 1) &
    (trips['duration_min'] < 300) &
    (trips['straight_dist_km'] > 0.1)
].copy()

print(f"\nMatched trips: {len(trips)}")
print(f"Valid trips (filtered): {len(valid_trips)}")
print(f"\nTrip Duration Stats (minutes):")
print(valid_trips['duration_min'].describe().to_string())
print(f"\nStraight-line Distance Stats (km):")
print(valid_trips['straight_dist_km'].describe().to_string())

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Trip Pattern Analysis', fontsize=16, fontweight='bold')

ax = axes[0, 0]
starts['hour'].value_counts().sort_index().plot(kind='bar', ax=ax, color='#3498db', alpha=0.8)
ax.set_xlabel('Departure Hour')
ax.set_ylabel('Trip Count')
ax.set_title('Trip Departure Time Distribution')
ax.set_xticks(range(0, 24))

ax = axes[0, 1]
ax.hist(valid_trips['duration_min'], bins=50, color='#2ecc71', alpha=0.8, edgecolor='white')
ax.set_xlabel('Duration (min)')
ax.set_ylabel('Trip Count')
ax.set_title(f'Trip Duration Distribution (median: {valid_trips["duration_min"].median():.1f} min)')
ax.axvline(valid_trips['duration_min'].median(), color='red', linestyle='--', label='Median')
ax.legend()

ax = axes[1, 0]
dist_clip = valid_trips['straight_dist_km'].clip(upper=50)
ax.hist(dist_clip, bins=50, color='#e67e22', alpha=0.8, edgecolor='white')
ax.set_xlabel('Straight-line Distance (km)')
ax.set_ylabel('Trip Count')
ax.set_title(f'Trip Distance Distribution (median: {valid_trips["straight_dist_km"].median():.1f} km)')
ax.axvline(valid_trips['straight_dist_km'].median(), color='red', linestyle='--', label='Median')
ax.legend()

ax = axes[1, 1]
ax.scatter(valid_trips['straight_dist_km'], valid_trips['duration_min'],
           alpha=0.3, s=8, c='#9b59b6')
ax.set_xlabel('Straight-line Distance (km)')
ax.set_ylabel('Duration (min)')
ax.set_title('Trip Duration vs Distance')
ax.set_xlim(0, 50)
ax.set_ylim(0, 120)

plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, '2_trip_patterns.png')
plt.savefig(path2)
plt.close()
print(f"\nChart saved: {path2}")


# ============================================================
# 3. Geo Heatmap
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Geographic Heatmap — Event Density")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Event Geographic Density', fontsize=16, fontweight='bold')

valid_geo = df.dropna(subset=['latitude', 'longitude'])
geo_focus = valid_geo[
    (valid_geo['latitude'].between(42.0, 43.3)) &
    (valid_geo['longitude'].between(-79.5, -78.3))
]
print(f"Valid geo data points: {len(valid_geo)}")
print(f"Focus area (Buffalo): {len(geo_focus)}")

ax = axes[0]
hb = ax.hexbin(geo_focus['longitude'], geo_focus['latitude'],
               gridsize=60, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, ax=ax, label='Event Count')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('All Events Heatmap (Hexbin)')

ax = axes[1]
event_colors = {
    'ACCELERATION_CHANGE': ('red', 'Hard Brake/Accel'),
    'JOURNEY': ('blue', 'Journey'),
    'SIGNAL_STATE_CHANGE': ('green', 'Turn Signal'),
    'LIGHT_STATE_CHANGE': ('orange', 'Lights'),
    'SEAT_BELT_CHANGE': ('purple', 'Seat Belt'),
}
for etype, (color, label) in event_colors.items():
    subset = geo_focus[geo_focus['eventType'] == etype]
    ax.scatter(subset['longitude'], subset['latitude'],
               c=color, alpha=0.15, s=3, label=f'{label} ({len(subset)})', edgecolors='none')
ax.legend(markerscale=5, fontsize=8)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Events by Type — Geographic Distribution')

plt.tight_layout()
path3 = os.path.join(OUTPUT_DIR, '3_geo_heatmap.png')
plt.savefig(path3)
plt.close()
print(f"Chart saved: {path3}")

try:
    import folium
    from folium.plugins import HeatMap

    center_lat = geo_focus['latitude'].median()
    center_lon = geo_focus['longitude'].median()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')
    sample = geo_focus.sample(min(10000, len(geo_focus)), random_state=42)
    heat_data = sample[['latitude', 'longitude']].values.tolist()
    HeatMap(heat_data, radius=8, blur=10, max_zoom=13).add_to(m)
    path3_html = os.path.join(OUTPUT_DIR, '3_interactive_heatmap.html')
    m.save(path3_html)
    print(f"Interactive heatmap saved: {path3_html}")
except ImportError:
    print("folium not installed, skipping interactive heatmap")

print("\nTop 15 Postal Codes by Event Density:")
print(geo_focus['postalCode'].value_counts().head(15).to_string())


# ============================================================
# 4. Safety — Seat Belt & Speeding
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Safety Analysis — Seat Belt & Speeding")
print("=" * 60)

belt_events = df[df['eventType'].isin(['SEAT_BELT_CHANGE', 'SEAT_BELT_WARNING'])].copy()
belt_events['seatBeltStatus'] = belt_events['eventMetadata'].apply(
    lambda x: x.get('seatBeltStatus') if isinstance(x, dict) else None)
belt_events['seatIdentifier'] = belt_events['eventMetadata'].apply(
    lambda x: x.get('seatIdentifier') if isinstance(x, dict) else None)

print(f"Seat belt events total: {len(belt_events)}")
print(f"\nSeat belt status distribution:")
print(belt_events['seatBeltStatus'].value_counts().to_string())
print(f"\nSeat identifier distribution:")
print(belt_events['seatIdentifier'].value_counts().to_string())

speed_data = df[df['speed'].notna()].copy()
print(f"\nData points with speed: {len(speed_data)}")
print(f"Speed stats (km/h):")
print(speed_data['speed'].describe().to_string())

speed_data['speed_category'] = pd.cut(
    speed_data['speed'],
    bins=[0, 50, 80, 110, speed_data['speed'].max() + 1],
    labels=['< 50', '50-80', '80-110', '> 110'],
    right=False
)
print(f"\nSpeed range distribution:")
print(speed_data['speed_category'].value_counts().sort_index().to_string())

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Safety Analysis — Seat Belt & Speeding', fontsize=16, fontweight='bold')

ax = axes[0, 0]
belt_status = belt_events['seatBeltStatus'].value_counts()
colors_belt = ['#e74c3c' if s == 'UNLATCHED' else '#2ecc71' for s in belt_status.index]
belt_status.plot(kind='bar', ax=ax, color=colors_belt)
ax.set_title('Seat Belt Status Events')
ax.set_ylabel('Event Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

ax = axes[0, 1]
belt_driver = belt_events[belt_events['seatIdentifier'] == 'DRIVER']
for status in ['LATCHED', 'UNLATCHED']:
    sub = belt_driver[belt_driver['seatBeltStatus'] == status]
    hourly = sub.groupby('hour').size()
    ax.plot(hourly.index, hourly.values, marker='o', markersize=4,
            label=status, color='#2ecc71' if status == 'LATCHED' else '#e74c3c')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Event Count')
ax.set_title('Driver Seat Belt Status by Hour')
ax.legend()
ax.set_xticks(range(0, 24))

ax = axes[1, 0]
ax.hist(speed_data['speed'], bins=80, color='#3498db', alpha=0.8, edgecolor='white')
ax.axvline(50, color='green', linestyle='--', linewidth=1.5, label='50 km/h (urban)')
ax.axvline(80, color='orange', linestyle='--', linewidth=1.5, label='80 km/h')
ax.axvline(110, color='red', linestyle='--', linewidth=1.5, label='110 km/h (highway)')
ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Data Point Count')
ax.set_title(f'Speed Distribution (median: {speed_data["speed"].median():.1f} km/h)')
ax.legend()

ax = axes[1, 1]
high_speed = speed_data[speed_data['speed'] > 80]
normal_speed = speed_data[speed_data['speed'] <= 80].sample(min(2000, len(speed_data)))
ax.scatter(normal_speed['longitude'], normal_speed['latitude'],
           c='#3498db', alpha=0.2, s=5, label='<= 80 km/h', edgecolors='none')
ax.scatter(high_speed['longitude'], high_speed['latitude'],
           c='red', alpha=0.5, s=15, label=f'> 80 km/h (n={len(high_speed)})', edgecolors='none')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('High Speed Locations')
ax.legend()

plt.tight_layout()
path4 = os.path.join(OUTPUT_DIR, '4_safety_analysis.png')
plt.savefig(path4)
plt.close()
print(f"\nChart saved: {path4}")


# ============================================================
# 5. Summary
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)
print(f"""
Data Overview:
  - Total records: {len(df):,}
  - Time range: {df['capturedDateTime'].min()} ~ {df['capturedDateTime'].max()}
  - Region: New York State (primarily Buffalo area)
  - Unique journeys: {df['journeyId'].nunique():,}

1. Driving Behavior:
  - Hard brake events: {len(hard_brake):,}
  - Hard acceleration events: {len(hard_accel):,}
  - Top hard-brake postal code: {hard_brake['postalCode'].value_counts().index[0] if len(hard_brake) > 0 else 'N/A'}

2. Trip Patterns:
  - Valid trips: {len(valid_trips):,}
  - Avg trip duration: {valid_trips['duration_min'].mean():.1f} min
  - Median trip distance: {valid_trips['straight_dist_km'].median():.1f} km (straight-line)

3. Geographic Density:
  - Densest postal code: {geo_focus['postalCode'].value_counts().index[0]}
  - Interactive heatmap: 3_interactive_heatmap.html

4. Safety:
  - Seat belt unlatched events: {len(belt_events[belt_events['seatBeltStatus'] == 'UNLATCHED']):,}
  - Seat belt latched events: {len(belt_events[belt_events['seatBeltStatus'] == 'LATCHED']):,}
  - Records > 80 km/h: {len(speed_data[speed_data['speed'] > 80]):,} ({len(speed_data[speed_data['speed'] > 80])/len(speed_data)*100:.1f}%)
  - Max speed: {speed_data['speed'].max():.1f} km/h
""")

print("Output files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith(('1_', '2_', '3_', '4_')):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")

print("\nAnalysis complete!")
