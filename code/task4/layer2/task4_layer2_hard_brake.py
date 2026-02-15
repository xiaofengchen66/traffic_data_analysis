#!/usr/bin/env python3
"""
Task 4 — Layer 2: Hard Brake Cross-Validation & Independent Heatmap

Two tasks:
  A) For each Layer 1 hotspot (ABS/ESC/AEB cluster), count how many
     ACCELERATION_CHANGE (hard brake) events occur within 200m.
  B) Aggregate all hard brake events by geohash7 (~150m grid) to produce
     an independent heatmap and discover new problem areas.

Scans all 744 CSVs, memory-friendly (processes file by file).

Usage:
  python3 task4_layer2_hard_brake.py
"""

import os
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
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
OUTPUT_DIR = "/home/qiqingh/Desktop/traffic_data_analysis/task4_outputs/layer2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Layer 1 cluster centers (v2 results)
CLUSTER_CSV = "/home/qiqingh/Desktop/traffic_data_analysis/task4_outputs/layer1/version_2/hotspot_clusters.csv"

# Matching radius for Task A
MATCH_RADIUS_KM = 0.2  # 200m

# We only care about hard braking, not hard acceleration
# eventMetadata typically contains direction info; we'll collect both
# and filter later if needed
TARGET_EVENT = "ACCELERATION_CHANGE"


# ============================================================
# Step 0: Load Layer 1 cluster centers
# ============================================================
print("=" * 60)
print("Step 0: Loading Layer 1 cluster centers")
print("=" * 60)

clusters = pd.read_csv(CLUSTER_CSV)
print(f"Loaded {len(clusters)} clusters")

# Precompute cluster center coordinates in radians for vectorized haversine
cluster_lats_rad = np.radians(clusters['center_lat'].values)
cluster_lons_rad = np.radians(clusters['center_lon'].values)
n_clusters = len(clusters)

# Task A accumulators: for each cluster, count matching hard brake events
cluster_match_count = np.zeros(n_clusters, dtype=int)
cluster_match_dates = [set() for _ in range(n_clusters)]
cluster_match_journeys = [set() for _ in range(n_clusters)]
cluster_match_speeds = [[] for _ in range(n_clusters)]

# Task B accumulator: geohash7 -> stats
geohash_stats = defaultdict(lambda: {
    "count": 0,
    "brake_count": 0,  # hard brake specifically
    "accel_count": 0,  # hard acceleration
    "speed_sum": 0.0,
    "speed_n": 0,
    "lat_sum": 0.0,
    "lon_sum": 0.0,
    "dates": set(),
    "journeys": set(),
})


# ============================================================
# Vectorized haversine: event coords vs all cluster centers
# ============================================================
EARTH_R = 6371.0  # km

def haversine_vec(lat_rad, lon_rad, cluster_lats_rad, cluster_lons_rad):
    """
    Compute haversine distance from a single point to all cluster centers.
    Returns array of distances in km.
    """
    dlat = cluster_lats_rad - lat_rad
    dlon = cluster_lons_rad - lon_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad) * np.cos(cluster_lats_rad) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


# ============================================================
# Step 1: Scan all files
# ============================================================
print("\n" + "=" * 60)
print("Step 1: Scanning all files for ACCELERATION_CHANGE events")
print("=" * 60)

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and not f.startswith("._")])
print(f"Files to scan: {len(files)}\n")

total_accel_events = 0
total_brake = 0
total_accel = 0
total_rows = 0
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
                total_rows += 1

                # Quick filter on event type
                try:
                    event = json.loads(row[6]) if row[6] else {}
                except:
                    continue
                if event.get("eventType") != TARGET_EVENT:
                    continue

                total_accel_events += 1

                # Parse fields
                try:
                    location = json.loads(row[3]) if row[3] else {}
                except:
                    location = {}
                try:
                    metrics = json.loads(row[4]) if row[4] else {}
                except:
                    metrics = {}

                lat = location.get("latitude")
                lon = location.get("longitude")
                geohash = location.get("geohash", "")
                speed = metrics.get("speed")
                journey_id = row[1]

                # Determine brake vs acceleration from metadata
                metadata = event.get("eventMetadata", {})
                # Common patterns: {"value": "HARD_BRAKE"} or {"value": "HARD_ACCELERATION"}
                # or {"accelerationType": "BRAKE"} etc. — check both
                meta_str = str(metadata).upper()
                is_brake = "BRAKE" in meta_str or "DECEL" in meta_str
                is_accel_event = "ACCEL" in meta_str and not is_brake

                if is_brake:
                    total_brake += 1
                elif is_accel_event:
                    total_accel += 1

                # Extract date from capturedDateTime
                try:
                    dt = row[2][:10]  # "2022-09-28" from "2022-09-28T..."
                except:
                    dt = ""

                # Task B: aggregate by geohash7
                gh7 = geohash[:7] if geohash else ""
                if gh7 and lat is not None and lon is not None:
                    g = geohash_stats[gh7]
                    g["count"] += 1
                    if is_brake:
                        g["brake_count"] += 1
                    elif is_accel_event:
                        g["accel_count"] += 1
                    if speed is not None:
                        g["speed_sum"] += speed
                        g["speed_n"] += 1
                    g["lat_sum"] += lat
                    g["lon_sum"] += lon
                    g["dates"].add(dt)
                    g["journeys"].add(journey_id)

                # Task A: check distance to cluster centers (only for brakes with valid coords)
                if is_brake and lat is not None and lon is not None:
                    lat_r = np.radians(lat)
                    lon_r = np.radians(lon)
                    dists = haversine_vec(lat_r, lon_r, cluster_lats_rad, cluster_lons_rad)
                    matches = np.where(dists <= MATCH_RADIUS_KM)[0]
                    for idx in matches:
                        cluster_match_count[idx] += 1
                        cluster_match_dates[idx].add(dt)
                        cluster_match_journeys[idx].add(journey_id)
                        if speed is not None:
                            cluster_match_speeds[idx].append(speed)

    except Exception as e:
        continue

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        rate = (i + 1) / elapsed
        eta = (len(files) - i - 1) / rate
        print(f"  [{i+1}/{len(files)}] {total_accel_events:,} ACCEL events "
              f"({total_brake:,} brake) | {elapsed:.0f}s | ETA {eta:.0f}s")

elapsed = time.time() - start
print(f"\nScan complete: {total_accel_events:,} ACCELERATION_CHANGE events in {elapsed:.0f}s")
print(f"  Hard brake: {total_brake:,}")
print(f"  Hard acceleration: {total_accel:,}")
print(f"  Unclassified: {total_accel_events - total_brake - total_accel:,}")


# ============================================================
# Step 2: Task A — Enrich Layer 1 clusters with brake counts
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Enriching Layer 1 clusters with hard brake data")
print("=" * 60)

clusters['hard_brake_nearby'] = cluster_match_count
clusters['brake_unique_dates'] = [len(s) for s in cluster_match_dates]
clusters['brake_unique_journeys'] = [len(s) for s in cluster_match_journeys]
clusters['brake_avg_speed'] = [
    np.mean(s) if len(s) > 0 else None for s in cluster_match_speeds
]

# Composite confidence score
# Normalize each component to 0-1 range, then combine
def normalize(arr):
    arr = np.array(arr, dtype=float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

clusters['score_safety_events'] = normalize(clusters['event_count'])
clusters['score_brake_nearby'] = normalize(clusters['hard_brake_nearby'])
clusters['score_persistence'] = normalize(clusters['unique_dates'])
clusters['confidence_score'] = (
    0.4 * clusters['score_safety_events'] +
    0.35 * clusters['score_brake_nearby'] +
    0.25 * clusters['score_persistence']
)

clusters_ranked = clusters.sort_values('confidence_score', ascending=False)
clusters_ranked.to_csv(os.path.join(OUTPUT_DIR, "clusters_with_brake_validation.csv"), index=False)
print("Saved: clusters_with_brake_validation.csv")


# ============================================================
# Step 3: Task B — Geohash heatmap
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Building geohash heatmap")
print("=" * 60)

geo_rows = []
for gh, s in geohash_stats.items():
    if s["count"] < 3:  # skip very sparse cells
        continue
    geo_rows.append({
        "geohash7": gh,
        "total_events": s["count"],
        "brake_count": s["brake_count"],
        "accel_count": s["accel_count"],
        "avg_speed": s["speed_sum"] / s["speed_n"] if s["speed_n"] > 0 else None,
        "avg_lat": s["lat_sum"] / s["count"],
        "avg_lon": s["lon_sum"] / s["count"],
        "unique_dates": len(s["dates"]),
        "unique_journeys": len(s["journeys"]),
    })

geo_df = pd.DataFrame(geo_rows).sort_values("brake_count", ascending=False)
geo_df.to_csv(os.path.join(OUTPUT_DIR, "geohash_brake_heatmap.csv"), index=False)
print(f"Geohash cells with 3+ events: {len(geo_df)}")
print("Saved: geohash_brake_heatmap.csv")


# ============================================================
# Step 4: Visualizations
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Generating Visualizations")
print("=" * 60)

# --- Fig 1: Layer 1 hotspots colored by brake validation ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Layer 2: Hard Brake Cross-Validation of Layer 1 Hotspots",
             fontsize=16, fontweight='bold')

# Left: safety events vs brake count scatter
ax = axes[0]
valid = clusters_ranked[clusters_ranked['hard_brake_nearby'] > 0]
noise = clusters_ranked[clusters_ranked['hard_brake_nearby'] == 0]
ax.scatter(noise['event_count'], [0]*len(noise), c='lightgray', s=20, alpha=0.5,
           label=f'No brake support ({len(noise)})')
if len(valid) > 0:
    sc = ax.scatter(valid['event_count'], valid['hard_brake_nearby'],
                    c=valid['confidence_score'], cmap='YlOrRd', s=40, alpha=0.7,
                    edgecolors='black', linewidths=0.3)
    plt.colorbar(sc, ax=ax, label='Confidence Score')
    # Label top 5
    for _, r in valid.head(5).iterrows():
        ax.annotate(f"#{int(r['cluster_id'])}",
                    (r['event_count'], r['hard_brake_nearby']),
                    fontsize=7, fontweight='bold')
ax.set_xlabel('Safety Events (ABS/ESC/AEB)')
ax.set_ylabel('Hard Brake Events within 200m')
ax.set_title('Safety Events vs Brake Corroboration')
ax.legend(fontsize=8)

# Right: geographic map colored by confidence
ax = axes[1]
sc = ax.scatter(clusters_ranked['center_lon'], clusters_ranked['center_lat'],
                c=clusters_ranked['confidence_score'], cmap='YlOrRd',
                s=clusters_ranked['event_count'] * 0.8 + 5,
                alpha=0.7, edgecolors='black', linewidths=0.2)
plt.colorbar(sc, ax=ax, label='Confidence Score')
for _, r in clusters_ranked.head(10).iterrows():
    ax.annotate(f"#{int(r['cluster_id'])}",
                (r['center_lon'], r['center_lat']),
                fontsize=6, fontweight='bold', color='darkred')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Hotspots by Confidence Score (size=events)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_brake_validation.png"))
plt.close()
print("  Saved: fig1_brake_validation.png")

# --- Fig 2: Independent brake heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Independent Hard Brake Heatmap (geohash7 ~150m grid)",
             fontsize=16, fontweight='bold')

top_geo = geo_df.head(500)  # top 500 cells for visibility

ax = axes[0]
sc = ax.scatter(top_geo['avg_lon'], top_geo['avg_lat'],
                c=top_geo['brake_count'], cmap='hot_r',
                s=np.clip(top_geo['brake_count'] * 0.05, 3, 50),
                alpha=0.6, edgecolors='none')
plt.colorbar(sc, ax=ax, label='Hard Brake Count')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Top 500 Hard Brake Cells')

ax = axes[1]
persistent_geo = geo_df[geo_df['unique_dates'] >= 10].head(200)
if len(persistent_geo) > 0:
    sc = ax.scatter(persistent_geo['avg_lon'], persistent_geo['avg_lat'],
                    c=persistent_geo['brake_count'], cmap='hot_r',
                    s=np.clip(persistent_geo['brake_count'] * 0.1, 5, 60),
                    alpha=0.7, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Hard Brake Count')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Persistent Brake Cells (10+ dates)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_brake_heatmap.png"))
plt.close()
print("  Saved: fig2_brake_heatmap.png")

# --- Fig 3: Top 20 ranked hotspots detail ---
fig, ax = plt.subplots(figsize=(14, 8))
top20 = clusters_ranked.head(20).copy()
top20['label'] = top20.apply(
    lambda r: f"#{int(r['cluster_id'])} (ZIP {r['top_postal']})" 
    if pd.notna(r['top_postal']) else f"#{int(r['cluster_id'])}", axis=1)

y_pos = range(len(top20))
bars1 = ax.barh(y_pos, top20['event_count'], height=0.4, align='edge',
                color='#e74c3c', label='Safety Events (ABS/ESC/AEB)')
bars2 = ax.barh([y + 0.4 for y in y_pos], top20['hard_brake_nearby'], height=0.4,
                align='edge', color='#3498db', label='Hard Brakes within 200m')
ax.set_yticks([y + 0.4 for y in y_pos])
ax.set_yticklabels(top20['label'], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Event Count')
ax.set_title('Top 20 Hotspots: Safety Events + Hard Brake Corroboration',
             fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_top20_comparison.png"))
plt.close()
print("  Saved: fig3_top20_comparison.png")


# ============================================================
# Step 5: Report
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Writing Report")
print("=" * 60)

L = []
def w(t=""):
    L.append(t)

w("=" * 60)
w("TASK 4 LAYER 2: HARD BRAKE CROSS-VALIDATION REPORT")
w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w("=" * 60)

w(f"\n--- Scan Summary ---")
w(f"  Total rows scanned: {total_rows:,}")
w(f"  ACCELERATION_CHANGE events: {total_accel_events:,}")
w(f"    Hard brake: {total_brake:,}")
w(f"    Hard acceleration: {total_accel:,}")
w(f"    Unclassified: {total_accel_events - total_brake - total_accel:,}")
w(f"  Scan time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

w(f"\n--- Task A: Layer 1 Hotspot Validation ---")
w(f"  Layer 1 clusters: {n_clusters}")
w(f"  Match radius: {MATCH_RADIUS_KM * 1000:.0f}m")
validated = (clusters['hard_brake_nearby'] > 0).sum()
w(f"  Clusters with brake corroboration: {validated} ({validated/n_clusters*100:.1f}%)")
w(f"  Clusters without brake support: {n_clusters - validated}")

w(f"\n--- Top 20 Hotspots by Confidence Score ---")
w(f"  {'Rank':>4s} {'ID':>4s} {'SafetyEv':>8s} {'Brakes':>7s} {'BrkDays':>7s} "
  f"{'BrkTrips':>8s} {'Score':>6s} {'ZIP':>8s} {'Lat':>9s} {'Lon':>10s}")
for rank, (_, r) in enumerate(clusters_ranked.head(20).iterrows(), 1):
    zp = str(r['top_postal']) if pd.notna(r['top_postal']) else "—"
    w(f"  {rank:>4d} {int(r['cluster_id']):>4d} {int(r['event_count']):>8d} "
      f"{int(r['hard_brake_nearby']):>7d} {int(r['brake_unique_dates']):>7d} "
      f"{int(r['brake_unique_journeys']):>8d} {r['confidence_score']:>6.3f} "
      f"{zp:>8s} {r['center_lat']:>9.5f} {r['center_lon']:>10.5f}")

w(f"\n--- Task B: Independent Brake Heatmap ---")
w(f"  Geohash7 cells with 3+ events: {len(geo_df)}")
w(f"  Top 10 brake hotspot cells:")
w(f"  {'GH7':>9s} {'Brakes':>7s} {'Total':>6s} {'AvgSpd':>7s} {'Days':>5s} {'Trips':>6s} "
  f"{'Lat':>9s} {'Lon':>10s}")
for _, r in geo_df.head(10).iterrows():
    spd = f"{r['avg_speed']:.0f}" if pd.notna(r['avg_speed']) else "—"
    w(f"  {r['geohash7']:>9s} {int(r['brake_count']):>7d} {int(r['total_events']):>6d} "
      f"{spd:>7s} {int(r['unique_dates']):>5d} {int(r['unique_journeys']):>6d} "
      f"{r['avg_lat']:>9.5f} {r['avg_lon']:>10.5f}")

# Find new hotspots: high brake count cells that are NOT near any Layer 1 cluster
w(f"\n--- New Problem Areas (high brakes, no ABS/ESC/AEB cluster nearby) ---")
new_hotspots = []
for _, r in geo_df.head(100).iterrows():
    lat_r = np.radians(r['avg_lat'])
    lon_r = np.radians(r['avg_lon'])
    dists = haversine_vec(lat_r, lon_r, cluster_lats_rad, cluster_lons_rad)
    if dists.min() > MATCH_RADIUS_KM:
        new_hotspots.append(r)
    if len(new_hotspots) >= 20:
        break

w(f"  Found: {len(new_hotspots)} cells in top-100 brake cells with no nearby ABS/ESC cluster")
if new_hotspots:
    w(f"  {'GH7':>9s} {'Brakes':>7s} {'Days':>5s} {'AvgSpd':>7s} {'Lat':>9s} {'Lon':>10s}")
    for r in new_hotspots[:10]:
        spd = f"{r['avg_speed']:.0f}" if pd.notna(r['avg_speed']) else "—"
        w(f"  {r['geohash7']:>9s} {int(r['brake_count']):>7d} {int(r['unique_dates']):>5d} "
          f"{spd:>7s} {r['avg_lat']:>9.5f} {r['avg_lon']:>10.5f}")

w(f"\n--- Confidence Score Formula ---")
w(f"  confidence = 0.40 * norm(safety_events) + 0.35 * norm(hard_brakes_nearby) + 0.25 * norm(persistence_days)")
w(f"  All components normalized to [0, 1] range before combining.")

report_path = os.path.join(OUTPUT_DIR, "layer2_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(L))
print(f"Report: {report_path}")
print(f"\nAll outputs: {OUTPUT_DIR}")
print("Done!")