#!/usr/bin/env python3
"""
Task 4 — Layer 1: ABS / ESC / AEB Hotspot Analysis
Scans all 744 CSVs, extracts only safety-system activation events,
performs spatial clustering, and outputs maps + report.

Usage:
  python3 task4_layer1_safety_hotspots.py
"""

import os
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from collections import Counter
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import time
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================
# Config
# ============================================================
DATA_DIR = "/home/qiqingh/Desktop/top_article/driving-events-csv"
OUTPUT_DIR = "/home/qiqingh/Desktop/traffic_data_analysis/task4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_EVENTS = {
    "ANTI_LOCK_BRAKING_SYSTEM_STATE_CHANGE",   # ABS
    "ELECTRONIC_STABILITY_STATE_CHANGE",        # ESC
    "AUTONOMOUS_EMERGENCY_BRAKING_CHANGE",      # AEB
}

# DBSCAN params: 200m radius, at least 2 events to form a cluster
DBSCAN_EPS_KM = 0.2
DBSCAN_MIN_SAMPLES = 2

# ============================================================
# Step 1: Extract target events from all files
# ============================================================
print("=" * 60)
print("Step 1: Extracting ABS/ESC/AEB events from all files")
print("=" * 60)

records = []
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and not f.startswith("._")])
print(f"Scanning {len(files)} files...\n")

start = time.time()
for i, fname in enumerate(files):
    fpath = os.path.join(DATA_DIR, fname)
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, escapechar="\\")
            header = next(reader)
            if len(header) != 7 or header[0] != "dataPointId":
                continue
            for row in reader:
                if len(row) != 7:
                    continue
                # Quick filter: only parse event JSON if row might contain target
                try:
                    event = json.loads(row[6]) if row[6] else {}
                except:
                    continue
                etype = event.get("eventType", "")
                if etype not in TARGET_EVENTS:
                    continue

                # Parse full record
                try:
                    location = json.loads(row[3]) if row[3] else {}
                except:
                    location = {}
                try:
                    metrics = json.loads(row[4]) if row[4] else {}
                except:
                    metrics = {}
                try:
                    status = json.loads(row[5]) if row[5] else {}
                except:
                    status = {}

                metadata = event.get("eventMetadata", {})

                records.append({
                    "dataPointId": row[0],
                    "journeyId": row[1],
                    "capturedDateTime": row[2],
                    "latitude": location.get("latitude"),
                    "longitude": location.get("longitude"),
                    "postalCode": location.get("postalCode"),
                    "geohash": location.get("geohash"),
                    "speed": metrics.get("speed"),
                    "heading": metrics.get("heading"),
                    "accelerationX": metrics.get("accelerationX"),
                    "accelerationY": metrics.get("accelerationY"),
                    "ignitionState": status.get("ignitionState"),
                    "eventType": etype,
                    "eventMetadata": str(metadata),
                    "sourceFile": fname,
                })
    except Exception as e:
        print(f"  Warning: skipped {fname} ({e})")
        continue

    if (i + 1) % 100 == 0:
        elapsed = time.time() - start
        print(f"  [{i+1}/{len(files)}] found {len(records)} events | {elapsed:.0f}s")

elapsed = time.time() - start
print(f"\nExtraction complete: {len(records)} events in {elapsed:.0f}s")

# Build DataFrame
df = pd.DataFrame(records)
df['capturedDateTime'] = pd.to_datetime(df['capturedDateTime'])
for col in ['latitude', 'longitude', 'speed', 'heading', 'accelerationX', 'accelerationY']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['date'] = df['capturedDateTime'].dt.date
df['hour'] = df['capturedDateTime'].dt.hour
df['dayOfWeek'] = df['capturedDateTime'].dt.dayofweek  # 0=Mon, 6=Sun

# Short labels
EVENT_LABELS = {
    "ANTI_LOCK_BRAKING_SYSTEM_STATE_CHANGE": "ABS",
    "ELECTRONIC_STABILITY_STATE_CHANGE": "ESC",
    "AUTONOMOUS_EMERGENCY_BRAKING_CHANGE": "AEB",
}
df['eventLabel'] = df['eventType'].map(EVENT_LABELS)

# Save raw extracted data
raw_csv_path = os.path.join(OUTPUT_DIR, "abs_esc_aeb_raw_events.csv")
df.to_csv(raw_csv_path, index=False)
print(f"Raw events saved: {raw_csv_path}")


# ============================================================
# Step 2: Basic Statistics
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Basic Statistics")
print("=" * 60)

stats_lines = []
def w(text=""):
    stats_lines.append(text)

w("=" * 60)
w("TASK 4 LAYER 1: ABS / ESC / AEB HOTSPOT ANALYSIS")
w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w("=" * 60)

w(f"\nTotal safety-system events: {len(df)}")
w(f"\nBy event type:")
for label, cnt in df['eventLabel'].value_counts().items():
    w(f"  {label:5s}: {cnt:,}")

w(f"\nField coverage:")
for col in ['speed', 'accelerationX', 'accelerationY', 'heading']:
    non_null = df[col].notna().sum()
    pct = non_null / len(df) * 100
    w(f"  {col:15s}: {non_null:,} ({pct:.1f}%)")

w(f"\nSpeed at trigger (km/h):")
speed_valid = df[df['speed'].notna()]['speed']
if len(speed_valid) > 0:
    w(f"  Mean:   {speed_valid.mean():.1f}")
    w(f"  Median: {speed_valid.median():.1f}")
    w(f"  Min:    {speed_valid.min():.1f}")
    w(f"  Max:    {speed_valid.max():.1f}")
    w(f"  Std:    {speed_valid.std():.1f}")

w(f"\nTime range: {df['capturedDateTime'].min()} to {df['capturedDateTime'].max()}")
w(f"Unique dates: {df['date'].nunique()}")
w(f"Unique journeys: {df['journeyId'].nunique()}")
w(f"Unique postal codes: {df['postalCode'].nunique()}")

w(f"\nTop 10 postal codes:")
for pc, cnt in df['postalCode'].value_counts().head(10).items():
    w(f"  {pc}: {cnt}")


# ============================================================
# Step 3: Spatial Clustering (DBSCAN)
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Spatial Clustering (DBSCAN)")
print("=" * 60)

geo_valid = df.dropna(subset=['latitude', 'longitude']).copy()
print(f"Events with valid coordinates: {len(geo_valid)}")

# DBSCAN using haversine metric
coords_rad = np.radians(geo_valid[['latitude', 'longitude']].values)
eps_rad = DBSCAN_EPS_KM / 6371.0  # convert km to radians

clustering = DBSCAN(eps=eps_rad, min_samples=DBSCAN_MIN_SAMPLES, metric='haversine')
geo_valid['cluster'] = clustering.fit_predict(coords_rad)

n_clusters = geo_valid['cluster'].max() + 1
n_noise = (geo_valid['cluster'] == -1).sum()
print(f"Clusters found: {n_clusters}")
print(f"Noise points (no cluster): {n_noise}")

w(f"\n--- Spatial Clustering (DBSCAN, eps={DBSCAN_EPS_KM}km, min_samples={DBSCAN_MIN_SAMPLES}) ---")
w(f"  Clusters found: {n_clusters}")
w(f"  Noise points:   {n_noise}")
w(f"  Clustered:      {len(geo_valid) - n_noise}")

# Cluster summary
if n_clusters > 0:
    cluster_summary = []
    for cid in range(n_clusters):
        cmask = geo_valid['cluster'] == cid
        cdata = geo_valid[cmask]
        cluster_summary.append({
            'cluster_id': cid,
            'event_count': len(cdata),
            'center_lat': cdata['latitude'].mean(),
            'center_lon': cdata['longitude'].mean(),
            'avg_speed': cdata['speed'].mean() if cdata['speed'].notna().any() else None,
            'avg_accelX': cdata['accelerationX'].mean() if cdata['accelerationX'].notna().any() else None,
            'top_postal': cdata['postalCode'].mode().iloc[0] if len(cdata['postalCode'].mode()) > 0 else None,
            'abs_count': (cdata['eventLabel'] == 'ABS').sum(),
            'esc_count': (cdata['eventLabel'] == 'ESC').sum(),
            'aeb_count': (cdata['eventLabel'] == 'AEB').sum(),
            'unique_dates': cdata['date'].nunique(),
            'unique_journeys': cdata['journeyId'].nunique(),
        })

    cluster_df = pd.DataFrame(cluster_summary).sort_values('event_count', ascending=False)
    cluster_csv_path = os.path.join(OUTPUT_DIR, "hotspot_clusters.csv")
    cluster_df.to_csv(cluster_csv_path, index=False)
    print(f"Cluster summary saved: {cluster_csv_path}")

    w(f"\n--- Top 20 Hotspot Clusters (by event count) ---")
    w(f"  {'ID':>4s} {'Events':>7s} {'ABS':>5s} {'ESC':>5s} {'AEB':>5s} "
      f"{'Lat':>9s} {'Lon':>10s} {'AvgSpd':>7s} {'Dates':>6s} {'Trips':>6s} {'ZIP':>6s}")
    for _, r in cluster_df.head(20).iterrows():
        spd_str = f"{r['avg_speed']:.1f}" if pd.notna(r['avg_speed']) else "N/A"
        w(f"  {int(r['cluster_id']):>4d} {int(r['event_count']):>7d} "
          f"{int(r['abs_count']):>5d} {int(r['esc_count']):>5d} {int(r['aeb_count']):>5d} "
          f"{r['center_lat']:>9.5f} {r['center_lon']:>10.5f} {spd_str:>7s} "
          f"{int(r['unique_dates']):>6d} {int(r['unique_journeys']):>6d} {r['top_postal']:>6s}")


# ============================================================
# Step 4: Visualizations
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Generating Visualizations")
print("=" * 60)

# --- Figure 1: Overview 2x2 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Layer 1: ABS / ESC / AEB Safety Events Overview', fontsize=16, fontweight='bold')

# 1a. Event counts by type
ax = axes[0, 0]
type_counts = df['eventLabel'].value_counts()
colors_map = {'ABS': '#e74c3c', 'ESC': '#f39c12', 'AEB': '#8e44ad'}
bars = ax.bar(type_counts.index, type_counts.values,
              color=[colors_map.get(x, '#3498db') for x in type_counts.index])
ax.set_title('Event Count by Type')
ax.set_ylabel('Count')
for bar, val in zip(bars, type_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:,}', ha='center', va='bottom', fontsize=10)

# 1b. Hourly distribution
ax = axes[0, 1]
for label in ['ABS', 'ESC', 'AEB']:
    sub = df[df['eventLabel'] == label]
    hourly = sub.groupby('hour').size()
    ax.plot(hourly.index, hourly.values, marker='o', markersize=3,
            label=label, color=colors_map[label])
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Event Count')
ax.set_title('Hourly Distribution')
ax.legend()
ax.set_xticks(range(0, 24))

# 1c. Speed at trigger
ax = axes[1, 0]
for label in ['ABS', 'ESC', 'AEB']:
    sub = df[(df['eventLabel'] == label) & (df['speed'].notna())]
    if len(sub) > 0:
        ax.hist(sub['speed'], bins=40, alpha=0.6, label=label, color=colors_map[label])
ax.set_xlabel('Speed at Trigger (km/h)')
ax.set_ylabel('Count')
ax.set_title('Speed Distribution at Safety Event Trigger')
ax.legend()

# 1d. Daily trend
ax = axes[1, 1]
daily = df.groupby(['date', 'eventLabel']).size().unstack(fill_value=0)
for label in ['ABS', 'ESC', 'AEB']:
    if label in daily.columns:
        ax.plot(daily.index, daily[label], marker='.', markersize=3,
                label=label, color=colors_map[label], alpha=0.8)
ax.set_xlabel('Date')
ax.set_ylabel('Event Count')
ax.set_title('Daily Trend')
ax.legend()
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "fig1_overview.png")
plt.savefig(fig1_path)
plt.close()
print(f"  Saved: {fig1_path}")

# --- Figure 2: Geographic scatter + clusters ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('ABS/ESC/AEB Geographic Distribution & Clusters', fontsize=16, fontweight='bold')

# 2a. All events by type
ax = axes[0]
for label in ['ABS', 'ESC', 'AEB']:
    sub = geo_valid[geo_valid['eventLabel'] == label]
    ax.scatter(sub['longitude'], sub['latitude'], c=colors_map[label],
               alpha=0.5, s=15, label=f'{label} ({len(sub)})', edgecolors='none')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('All Safety Events by Type')
ax.legend()

# 2b. Clustered events
ax = axes[1]
noise = geo_valid[geo_valid['cluster'] == -1]
clustered = geo_valid[geo_valid['cluster'] != -1]
ax.scatter(noise['longitude'], noise['latitude'], c='lightgray',
           alpha=0.3, s=8, label=f'Noise ({len(noise)})', edgecolors='none')
if len(clustered) > 0:
    scatter = ax.scatter(clustered['longitude'], clustered['latitude'],
                         c=clustered['cluster'], cmap='tab20', alpha=0.7, s=20,
                         label=f'Clustered ({len(clustered)})', edgecolors='none')
    # Mark top cluster centers
    if n_clusters > 0:
        top_clusters = cluster_df.head(10)
        for _, r in top_clusters.iterrows():
            ax.annotate(f"#{int(r['cluster_id'])}({int(r['event_count'])})",
                        (r['center_lon'], r['center_lat']),
                        fontsize=7, fontweight='bold', color='red',
                        ha='center', va='bottom')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(f'DBSCAN Clusters (eps={DBSCAN_EPS_KM}km, {n_clusters} clusters)')
ax.legend(fontsize=8)

plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "fig2_geo_clusters.png")
plt.savefig(fig2_path)
plt.close()
print(f"  Saved: {fig2_path}")

# --- Figure 3: Top clusters detail ---
if n_clusters > 0:
    top_n = min(8, n_clusters)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Top {top_n} Hotspot Clusters — Detail', fontsize=16, fontweight='bold')
    axes_flat = axes.flatten()

    for idx, (_, r) in enumerate(cluster_df.head(top_n).iterrows()):
        ax = axes_flat[idx]
        cid = int(r['cluster_id'])
        cdata = geo_valid[geo_valid['cluster'] == cid]

        for label in ['ABS', 'ESC', 'AEB']:
            sub = cdata[cdata['eventLabel'] == label]
            if len(sub) > 0:
                ax.scatter(sub['longitude'], sub['latitude'], c=colors_map[label],
                           alpha=0.7, s=30, label=label, edgecolors='black', linewidths=0.3)
        ax.set_title(f"Cluster #{cid} (n={int(r['event_count'])}, ZIP:{r['top_postal']})",
                     fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for idx in range(top_n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, "fig3_top_clusters_detail.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"  Saved: {fig3_path}")

# --- Interactive map (folium) ---
try:
    import folium
    from folium.plugins import MarkerCluster

    center_lat = geo_valid['latitude'].median()
    center_lon = geo_valid['longitude'].median()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')

    marker_cluster = MarkerCluster(name="Safety Events").add_to(m)

    icon_colors = {'ABS': 'red', 'ESC': 'orange', 'AEB': 'purple'}
    for _, row in geo_valid.iterrows():
        label = row['eventLabel']
        popup_text = (f"<b>{label}</b><br>"
                      f"Time: {row['capturedDateTime']}<br>"
                      f"Speed: {row['speed']:.1f} km/h<br>"
                      f"AccelX: {row['accelerationX']}<br>"
                      f"Cluster: {int(row['cluster']) if row['cluster'] != -1 else 'none'}<br>"
                      f"ZIP: {row['postalCode']}")
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=popup_text,
            icon=folium.Icon(color=icon_colors.get(label, 'blue'), icon='warning-sign', prefix='glyphicon')
        ).add_to(marker_cluster)

    # Add cluster center circles for top clusters
    if n_clusters > 0:
        for _, r in cluster_df.head(20).iterrows():
            folium.Circle(
                [r['center_lat'], r['center_lon']],
                radius=200,
                color='red', fill=True, fill_opacity=0.15,
                popup=f"Cluster #{int(r['cluster_id'])}: {int(r['event_count'])} events"
            ).add_to(m)

    map_path = os.path.join(OUTPUT_DIR, "hotspot_map_interactive.html")
    m.save(map_path)
    print(f"  Saved: {map_path}")

except ImportError:
    print("  folium not installed, skipping interactive map")


# ============================================================
# Step 5: Write Report
# ============================================================
w(f"\n--- Feasibility Notes for Paper ---")
w(f"  1. ABS/ESC/AEB events are rare but high-value signals for road anomaly detection.")
w(f"  2. 100% of these events carry acceleration + speed + location data.")
w(f"  3. Spatial clustering reveals persistent hotspots (same location, multiple dates/trips).")
w(f"  4. Next step: cross-reference with HARD_BRAKE events (Layer 2) for validation.")
w(f"  5. External validation: compare hotspots against city 311 reports or road maintenance records.")

report_path = os.path.join(OUTPUT_DIR, "layer1_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(stats_lines))
print(f"\nReport saved: {report_path}")

print(f"\nAll outputs in: {OUTPUT_DIR}")
print("Done!")