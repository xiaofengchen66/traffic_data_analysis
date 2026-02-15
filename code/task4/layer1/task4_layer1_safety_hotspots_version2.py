#!/usr/bin/env python3
"""
Task 4 — Layer 1: ABS / ESC / AEB Hotspot Analysis (v3 - tightest params)

Version history:
  v1: eps=200m, min_samples=2
  v2: eps=100m, min_samples=3
  v3: eps=50m,  min_samples=4  ← current

Output: task4_layer1_v3/

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
OUTPUT_DIR = "/home/qiqingh/Desktop/traffic_data_analysis/task4_outputs/layer1/version_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_EVENTS = {
    "ANTI_LOCK_BRAKING_SYSTEM_STATE_CHANGE",
    "ELECTRONIC_STABILITY_STATE_CHANGE",
    "AUTONOMOUS_EMERGENCY_BRAKING_CHANGE",
}

# --- v3: tightest params ---
DBSCAN_EPS_KM = 0.05      # 50m radius
DBSCAN_MIN_SAMPLES = 4    # at least 4 events

# ============================================================
# Step 1: Load events (reuse previous raw CSV if available)
# ============================================================
print("=" * 60)
print("Step 1: Loading ABS/ESC/AEB events")
print("=" * 60)

# Try to find previously extracted raw data
RAW_CANDIDATES = [
    "/home/qiqingh/Desktop/traffic_data_analysis/task4_outputs/layer1/version_1/abs_esc_aeb_raw_events.csv",
]
loaded = False
for raw_path in RAW_CANDIDATES:
    if os.path.exists(raw_path):
        print(f"Reusing cached raw data: {raw_path}")
        df = pd.read_csv(raw_path)
        loaded = True
        print(f"Loaded {len(df)} events (skipped re-scanning)")
        break

if not loaded:
    records = []
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and not f.startswith("._")])
    print(f"No cache found, scanning {len(files)} files...\n")
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
                    try:
                        event = json.loads(row[6]) if row[6] else {}
                    except:
                        continue
                    etype = event.get("eventType", "")
                    if etype not in TARGET_EVENTS:
                        continue
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
                    records.append({
                        "dataPointId": row[0], "journeyId": row[1],
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
                        "eventMetadata": str(event.get("eventMetadata", {})),
                        "sourceFile": fname,
                    })
        except:
            continue
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(files)}] found {len(records)} events | {elapsed:.0f}s")
    df = pd.DataFrame(records)
    print(f"Extraction complete: {len(df)} events")

# Ensure types
df['capturedDateTime'] = pd.to_datetime(df['capturedDateTime'], errors='coerce')
for col in ['latitude', 'longitude', 'speed', 'heading', 'accelerationX', 'accelerationY']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['date'] = df['capturedDateTime'].dt.date
df['hour'] = df['capturedDateTime'].dt.hour
df['dayOfWeek'] = df['capturedDateTime'].dt.dayofweek

EVENT_LABELS = {
    "ANTI_LOCK_BRAKING_SYSTEM_STATE_CHANGE": "ABS",
    "ELECTRONIC_STABILITY_STATE_CHANGE": "ESC",
    "AUTONOMOUS_EMERGENCY_BRAKING_CHANGE": "AEB",
}
df['eventLabel'] = df['eventType'].map(EVENT_LABELS)

# Save raw copy
df.to_csv(os.path.join(OUTPUT_DIR, "abs_esc_aeb_raw_events.csv"), index=False)


# ============================================================
# Step 2: Basic Statistics
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Basic Statistics")
print("=" * 60)

L = []
def w(text=""):
    L.append(text)

w("=" * 60)
w("TASK 4 LAYER 1 v3: ABS / ESC / AEB HOTSPOT ANALYSIS")
w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w(f"DBSCAN: eps={DBSCAN_EPS_KM*1000:.0f}m (50m), min_samples={DBSCAN_MIN_SAMPLES}")
w("=" * 60)

w(f"\nTotal safety-system events: {len(df)}")
w(f"\nBy event type:")
for label, cnt in df['eventLabel'].value_counts().items():
    w(f"  {label:5s}: {cnt:,}")

w(f"\nField coverage:")
for col in ['speed', 'accelerationX', 'accelerationY', 'heading']:
    nn = df[col].notna().sum()
    w(f"  {col:15s}: {nn:,} ({nn/len(df)*100:.1f}%)")

sv = df.loc[df['speed'].notna(), 'speed']
if len(sv) > 0:
    w(f"\nSpeed at trigger (km/h):")
    w(f"  Mean: {sv.mean():.1f}  Median: {sv.median():.1f}  "
      f"Min: {sv.min():.1f}  Max: {sv.max():.1f}  Std: {sv.std():.1f}")

w(f"\nTime range: {df['capturedDateTime'].min()} to {df['capturedDateTime'].max()}")
w(f"Unique dates: {df['date'].nunique()}")
w(f"Unique journeys: {df['journeyId'].nunique()}")
w(f"Unique postal codes: {df['postalCode'].nunique()}")

w(f"\nTop 10 postal codes:")
for pc, cnt in df['postalCode'].value_counts().head(10).items():
    w(f"  {pc}: {cnt}")


# ============================================================
# Step 3: Spatial Clustering (DBSCAN - tightest)
# ============================================================
print("\n" + "=" * 60)
print(f"Step 3: DBSCAN Clustering (eps={DBSCAN_EPS_KM*1000:.0f}m, min={DBSCAN_MIN_SAMPLES})")
print("=" * 60)

geo = df.dropna(subset=['latitude', 'longitude']).copy()
print(f"Events with valid coordinates: {len(geo)}")

coords_rad = np.radians(geo[['latitude', 'longitude']].values)
eps_rad = DBSCAN_EPS_KM / 6371.0

clustering = DBSCAN(eps=eps_rad, min_samples=DBSCAN_MIN_SAMPLES, metric='haversine')
geo['cluster'] = clustering.fit_predict(coords_rad)

n_clusters = geo['cluster'].max() + 1
n_noise = (geo['cluster'] == -1).sum()
n_clustered = len(geo) - n_noise
print(f"Clusters: {n_clusters} | Clustered: {n_clustered} | Noise: {n_noise}")

w(f"\n--- Spatial Clustering ---")
w(f"  Radius:     {DBSCAN_EPS_KM*1000:.0f}m")
w(f"  Min events: {DBSCAN_MIN_SAMPLES}")
w(f"  Clusters:   {n_clusters}")
w(f"  Clustered:  {n_clustered} ({n_clustered/len(geo)*100:.1f}%)")
w(f"  Noise:      {n_noise} ({n_noise/len(geo)*100:.1f}%)")

# Build cluster summary
cdf = None
if n_clusters > 0:
    rows = []
    for cid in range(n_clusters):
        c = geo[geo['cluster'] == cid]
        rows.append({
            'cluster_id': cid,
            'event_count': len(c),
            'center_lat': c['latitude'].mean(),
            'center_lon': c['longitude'].mean(),
            'avg_speed': c['speed'].mean() if c['speed'].notna().any() else None,
            'avg_accelX': c['accelerationX'].mean() if c['accelerationX'].notna().any() else None,
            'avg_accelY': c['accelerationY'].mean() if c['accelerationY'].notna().any() else None,
            'top_postal': c['postalCode'].mode().iloc[0] if len(c['postalCode'].mode()) > 0 else None,
            'abs_count': (c['eventLabel'] == 'ABS').sum(),
            'esc_count': (c['eventLabel'] == 'ESC').sum(),
            'aeb_count': (c['eventLabel'] == 'AEB').sum(),
            'unique_dates': c['date'].nunique(),
            'unique_journeys': c['journeyId'].nunique(),
        })
    cdf = pd.DataFrame(rows).sort_values('event_count', ascending=False).reset_index(drop=True)
    cdf.to_csv(os.path.join(OUTPUT_DIR, "hotspot_clusters.csv"), index=False)

    w(f"\n--- All {n_clusters} Clusters (sorted by event count) ---")
    w(f"  {'ID':>4s} {'Evt':>5s} {'ABS':>5s} {'ESC':>4s} {'AEB':>4s} "
      f"{'Lat':>9s} {'Lon':>10s} {'Spd':>6s} {'Days':>5s} {'Trip':>5s} {'ZIP':>6s}")
    for _, r in cdf.iterrows():
        spd = f"{r['avg_speed']:.0f}" if pd.notna(r['avg_speed']) else "—"
        zp = str(r['top_postal']) if pd.notna(r['top_postal']) else "—"
        w(f"  {int(r['cluster_id']):>4d} {int(r['event_count']):>5d} "
          f"{int(r['abs_count']):>5d} {int(r['esc_count']):>4d} {int(r['aeb_count']):>4d} "
          f"{r['center_lat']:>9.5f} {r['center_lon']:>10.5f} {spd:>6s} "
          f"{int(r['unique_dates']):>5d} {int(r['unique_journeys']):>5d} {zp:>6s}")

    # Persistent hotspots
    persistent = cdf[cdf['unique_dates'] >= 3]
    w(f"\n--- Persistent Hotspots (events on 3+ different dates) ---")
    w(f"  Count: {len(persistent)}")
    if len(persistent) > 0:
        for _, r in persistent.iterrows():
            spd = f"{r['avg_speed']:.0f}" if pd.notna(r['avg_speed']) else "—"
            zp = str(r['top_postal']) if pd.notna(r['top_postal']) else "—"
            w(f"  Cluster #{int(r['cluster_id'])}: {int(r['event_count'])} events, "
              f"{int(r['unique_dates'])} dates, {int(r['unique_journeys'])} trips, "
              f"avg speed {spd} km/h, ZIP {zp}, "
              f"({r['center_lat']:.5f}, {r['center_lon']:.5f})")


# ============================================================
# Step 4: Visualizations
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Generating Visualizations")
print("=" * 60)

CM = {'ABS': '#e74c3c', 'ESC': '#f39c12', 'AEB': '#8e44ad'}

# --- Fig 1: Overview 2x2 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Layer 1 v3: ABS/ESC/AEB Overview (50m, min 4)', fontsize=16, fontweight='bold')

ax = axes[0, 0]
tc = df['eventLabel'].value_counts()
bars = ax.bar(tc.index, tc.values, color=[CM.get(x, '#3498db') for x in tc.index])
ax.set_title('Event Count by Type'); ax.set_ylabel('Count')
for b, v in zip(bars, tc.values):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+20, f'{v:,}', ha='center', fontsize=10)

ax = axes[0, 1]
for lb in ['ABS', 'ESC', 'AEB']:
    h = df[df['eventLabel']==lb].groupby('hour').size()
    ax.plot(h.index, h.values, marker='o', markersize=3, label=lb, color=CM[lb])
ax.set_xlabel('Hour'); ax.set_ylabel('Count'); ax.set_title('Hourly Distribution')
ax.legend(); ax.set_xticks(range(0, 24))

ax = axes[1, 0]
for lb in ['ABS', 'ESC', 'AEB']:
    s = df[(df['eventLabel']==lb) & df['speed'].notna()]
    if len(s) > 0: ax.hist(s['speed'], bins=40, alpha=0.6, label=lb, color=CM[lb])
ax.set_xlabel('Speed (km/h)'); ax.set_ylabel('Count'); ax.set_title('Speed at Trigger'); ax.legend()

ax = axes[1, 1]
daily = df.groupby(['date', 'eventLabel']).size().unstack(fill_value=0)
for lb in ['ABS', 'ESC', 'AEB']:
    if lb in daily.columns:
        ax.plot(daily.index, daily[lb], marker='.', markersize=3, label=lb, color=CM[lb], alpha=0.8)
ax.set_xlabel('Date'); ax.set_ylabel('Count'); ax.set_title('Daily Trend')
ax.legend(); ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_overview.png")); plt.close()
print(f"  Saved: fig1_overview.png")

# --- Fig 2: Geo scatter + clusters ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f'ABS/ESC/AEB Clusters (50m, min 4)', fontsize=16, fontweight='bold')

ax = axes[0]
for lb in ['ABS', 'ESC', 'AEB']:
    s = geo[geo['eventLabel']==lb]
    ax.scatter(s['longitude'], s['latitude'], c=CM[lb], alpha=0.5, s=15,
               label=f'{lb} ({len(s)})', edgecolors='none')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.set_title('All Events'); ax.legend()

ax = axes[1]
noise_pts = geo[geo['cluster']==-1]
clust_pts = geo[geo['cluster']!=-1]
ax.scatter(noise_pts['longitude'], noise_pts['latitude'], c='lightgray', alpha=0.3, s=8,
           label=f'Noise ({len(noise_pts)})', edgecolors='none')
if len(clust_pts) > 0:
    ax.scatter(clust_pts['longitude'], clust_pts['latitude'], c=clust_pts['cluster'],
               cmap='tab20', alpha=0.7, s=25, label=f'Clustered ({len(clust_pts)})', edgecolors='none')
    if cdf is not None:
        for _, r in cdf.head(10).iterrows():
            ax.annotate(f"#{int(r['cluster_id'])}({int(r['event_count'])})",
                        (r['center_lon'], r['center_lat']), fontsize=7, fontweight='bold',
                        color='red', ha='center', va='bottom')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title(f'{n_clusters} clusters, {n_clustered} events'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_geo_clusters.png")); plt.close()
print(f"  Saved: fig2_geo_clusters.png")

# --- Fig 3: Top cluster detail ---
if cdf is not None and n_clusters > 0:
    top_n = min(8, n_clusters)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Top {top_n} Clusters Detail (50m)', fontsize=16, fontweight='bold')
    af = axes.flatten()
    for idx, (_, r) in enumerate(cdf.head(top_n).iterrows()):
        ax = af[idx]
        cid = int(r['cluster_id'])
        cd = geo[geo['cluster']==cid]
        for lb in ['ABS', 'ESC', 'AEB']:
            s = cd[cd['eventLabel']==lb]
            if len(s) > 0:
                ax.scatter(s['longitude'], s['latitude'], c=CM[lb], alpha=0.7, s=30,
                           label=lb, edgecolors='black', linewidths=0.3)
        spd = f"spd={r['avg_speed']:.0f}" if pd.notna(r['avg_speed']) else ""
        ax.set_title(f"#{cid} (n={int(r['event_count'])}, {int(r['unique_dates'])}d, "
                     f"ZIP:{r['top_postal']})\n{spd}", fontsize=9)
        ax.legend(fontsize=6); ax.tick_params(labelsize=6)
    for idx in range(top_n, len(af)): af[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_top_clusters.png")); plt.close()
    print(f"  Saved: fig3_top_clusters.png")

# --- Fig 4: Persistent hotspots ---
if cdf is not None:
    persistent = cdf[cdf['unique_dates'] >= 3]
    if len(persistent) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(geo['longitude'], geo['latitude'], c='lightgray', alpha=0.2, s=5, edgecolors='none')
        for _, r in persistent.iterrows():
            cd = geo[geo['cluster']==int(r['cluster_id'])]
            ax.scatter(cd['longitude'], cd['latitude'], s=40, alpha=0.8,
                       edgecolors='red', linewidths=0.5, zorder=5)
            ax.annotate(f"#{int(r['cluster_id'])} ({int(r['event_count'])}ev, {int(r['unique_dates'])}d)",
                        (r['center_lon'], r['center_lat']), fontsize=7, fontweight='bold',
                        color='darkred', ha='center', va='bottom')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_title(f'Persistent Hotspots (3+ dates): {len(persistent)} locations', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "fig4_persistent.png")); plt.close()
        print(f"  Saved: fig4_persistent.png")

# --- Interactive map ---
try:
    import folium
    from folium.plugins import MarkerCluster
    center = [geo['latitude'].median(), geo['longitude'].median()]
    m = folium.Map(location=center, zoom_start=11, tiles='OpenStreetMap')
    mc = MarkerCluster(name="Safety Events").add_to(m)
    ic = {'ABS': 'red', 'ESC': 'orange', 'AEB': 'purple'}
    for _, row in geo.iterrows():
        lb = row['eventLabel']
        popup = (f"<b>{lb}</b><br>Time: {row['capturedDateTime']}<br>"
                 f"Speed: {row['speed']:.1f}<br>Cluster: {int(row['cluster']) if row['cluster']!=-1 else 'none'}")
        folium.Marker([row['latitude'], row['longitude']], popup=popup,
                      icon=folium.Icon(color=ic.get(lb,'blue'), icon='warning-sign', prefix='glyphicon')
                      ).add_to(mc)
    if cdf is not None:
        for _, r in cdf.iterrows():
            color = 'darkred' if r['unique_dates'] >= 3 else 'red'
            folium.Circle([r['center_lat'], r['center_lon']], radius=50,
                          color=color, fill=True, fill_opacity=0.2,
                          popup=f"#{int(r['cluster_id'])}: {int(r['event_count'])}ev, {int(r['unique_dates'])}d"
                          ).add_to(m)
    m.save(os.path.join(OUTPUT_DIR, "hotspot_map.html"))
    print(f"  Saved: hotspot_map.html")
except ImportError:
    print("  folium not installed, skipping map")


# ============================================================
# Step 5: Write Report
# ============================================================
w(f"\n--- Interpretation Guide ---")
w(f"  - High confidence: clusters with 3+ dates, 3+ different journeys")
w(f"    → Fixed road defect (pothole, surface damage, poor drainage)")
w(f"  - Medium confidence: clusters with 2 dates or same-day repeats")
w(f"    → Possible defect, may also be weather or traffic-related")
w(f"  - Noise (unclustered): isolated events, likely driver behavior")
w(f"")
w(f"--- Next Steps ---")
w(f"  1. Layer 2: overlay HARD_BRAKE events to validate hotspots")
w(f"  2. Cross-reference wiper/temp data for weather correction")
w(f"  3. External validation: Buffalo 311 reports / road maintenance")
w(f"  4. Compute Road Quality Index per geohash grid cell")

report_path = os.path.join(OUTPUT_DIR, "layer1_v3_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(L))
print(f"\nReport: {report_path}")
print(f"All outputs: {OUTPUT_DIR}")
print("Done!")