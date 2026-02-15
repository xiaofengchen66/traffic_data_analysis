#!/usr/bin/env python3
"""
Task 4 — Layer 3: Weather Correction

Determines whether Layer 1 hotspots are caused by fixed road defects or
weather conditions (rain, low temperature) by:

1. Building hourly weather profiles from WIPER_STATE_CHANGE + exteriorTemp
2. Labeling each Layer 1 ABS/ESC/AEB event with weather context
3. Computing a "weather independence score" per cluster

Usage:
  python3 task4_layer3_weather.py
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
OUTPUT_DIR = "/home/qiqingh/Desktop/traffic_data_analysis/task4_outputs/layer3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Layer 1 raw events (from v1 extraction, reused by v2)
RAW_EVENTS_CSV = "/home/qiqingh/Desktop/traffic_data_analysis/task4_outputs/layer1/version_1/abs_esc_aeb_raw_events.csv"

# Layer 2 enriched clusters
CLUSTERS_CSV = "/home/qiqingh/Desktop/traffic_data_analysis/task4_outputs/layer2/clusters_with_brake_validation.csv"

# Wiper ON/OFF keywords — these are exact field values, not substrings
# Actual metadata format: {'wiperStateChangeType': 'ACTIVATED'/'DEACTIVATED', 'wiperIdentifier': 'FRONT'/'REAR', 'wiperInterval': N}

# Temperature thresholds
TEMP_COLD_C = 5.0     # below this = cold (frost risk)
TEMP_FREEZE_C = 0.0   # below this = freezing


# ============================================================
# Step 1: Build hourly weather profiles
# ============================================================
print("=" * 60)
print("Step 1: Building hourly weather profiles")
print("=" * 60)

# Accumulate per hour: wiper ON count, wiper OFF count, temp sum, temp count
# Key: "YYYY-MM-DD HH" (hour string)
hourly_weather = defaultdict(lambda: {
    "wiper_on": 0,
    "wiper_off": 0,
    "wiper_total": 0,
    "temp_sum": 0.0,
    "temp_count": 0,
    "temp_min": 999.0,
    "temp_max": -999.0,
})

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and not f.startswith("._")])
print(f"Scanning {len(files)} files for WIPER + temperature data...\n")

total_wiper = 0
total_temp = 0
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

                # Parse event type
                try:
                    event = json.loads(row[6]) if row[6] else {}
                except:
                    continue

                etype = event.get("eventType", "")

                # Extract hour key from timestamp
                try:
                    dt_str = row[2][:13]  # "2022-09-28T14" -> hour precision
                    hour_key = dt_str[:10] + " " + dt_str[11:13]  # "2022-09-28 14"
                except:
                    continue

                # Wiper events
                if etype == "WIPER_STATE_CHANGE":
                    total_wiper += 1
                    meta = event.get("eventMetadata", {})

                    h = hourly_weather[hour_key]
                    h["wiper_total"] += 1

                    # Use exact field value
                    wiper_state = meta.get("wiperStateChangeType", "").upper()
                    wiper_id = meta.get("wiperIdentifier", "").upper()

                    # Only count front wipers for rain detection (rear is manual)
                    if wiper_id == "REAR":
                        pass  # skip rear wipers
                    elif wiper_state == "ACTIVATED":
                        h["wiper_on"] += 1
                    elif wiper_state == "DEACTIVATED":
                        h["wiper_off"] += 1
                    else:
                        h["wiper_off"] += 1  # unknown = assume off

                # Temperature from ANY event that has it
                try:
                    metrics = json.loads(row[4]) if row[4] else {}
                except:
                    metrics = {}

                env = metrics.get("environment", {})
                if isinstance(env, dict):
                    temp = env.get("exteriorTemperature")
                    if temp is not None:
                        total_temp += 1
                        h = hourly_weather[hour_key]
                        h["temp_sum"] += temp
                        h["temp_count"] += 1
                        if temp < h["temp_min"]:
                            h["temp_min"] = temp
                        if temp > h["temp_max"]:
                            h["temp_max"] = temp

    except Exception as e:
        continue

    if (i + 1) % 100 == 0:
        elapsed = time.time() - start
        rate = (i + 1) / elapsed
        eta = (len(files) - i - 1) / rate
        print(f"  [{i+1}/{len(files)}] wiper: {total_wiper:,} | temp: {total_temp:,} "
              f"| {elapsed:.0f}s | ETA {eta:.0f}s")

elapsed_scan = time.time() - start
print(f"\nScan complete in {elapsed_scan:.0f}s")
print(f"  Wiper events: {total_wiper:,}")
print(f"  Temperature readings: {total_temp:,}")
print(f"  Unique hours with weather data: {len(hourly_weather)}")

# Build weather DataFrame
weather_rows = []
for hour_key, w in hourly_weather.items():
    rain_ratio = w["wiper_on"] / w["wiper_total"] if w["wiper_total"] > 0 else None
    avg_temp = w["temp_sum"] / w["temp_count"] if w["temp_count"] > 0 else None

    weather_rows.append({
        "hour_key": hour_key,
        "wiper_on": w["wiper_on"],
        "wiper_off": w["wiper_off"],
        "wiper_total": w["wiper_total"],
        "rain_ratio": rain_ratio,
        "is_rainy": rain_ratio is not None and rain_ratio > 0.3,  # >30% wiper-on = rainy
        "avg_temp": avg_temp,
        "min_temp": w["temp_min"] if w["temp_min"] < 900 else None,
        "max_temp": w["temp_max"] if w["temp_max"] > -900 else None,
        "is_cold": avg_temp is not None and avg_temp < TEMP_COLD_C,
        "is_freezing": avg_temp is not None and avg_temp < TEMP_FREEZE_C,
    })

weather_df = pd.DataFrame(weather_rows).sort_values("hour_key")
weather_df.to_csv(os.path.join(OUTPUT_DIR, "hourly_weather_profiles.csv"), index=False)
print(f"Saved: hourly_weather_profiles.csv ({len(weather_df)} hours)")

# Summary
n_rainy_hours = weather_df['is_rainy'].sum()
n_cold_hours = weather_df['is_cold'].sum()
n_freezing_hours = weather_df['is_freezing'].sum()
print(f"  Rainy hours (>30% wiper-on): {n_rainy_hours}")
print(f"  Cold hours (<{TEMP_COLD_C}°C): {n_cold_hours}")
print(f"  Freezing hours (<{TEMP_FREEZE_C}°C): {n_freezing_hours}")


# ============================================================
# Step 2: Label Layer 1 events with weather context
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Labeling safety events with weather context")
print("=" * 60)

events = pd.read_csv(RAW_EVENTS_CSV)
events['capturedDateTime'] = pd.to_datetime(events['capturedDateTime'], errors='coerce')
print(f"Loaded {len(events)} ABS/ESC/AEB events")

# Build hour_key for each event
events['hour_key'] = events['capturedDateTime'].dt.strftime('%Y-%m-%d %H')

# Merge with weather
weather_lookup = weather_df.set_index('hour_key')[['rain_ratio', 'is_rainy', 'avg_temp', 'is_cold', 'is_freezing']]
events = events.merge(weather_lookup, left_on='hour_key', right_index=True, how='left')

# Label weather conditions
events['weather_label'] = 'unknown'
events.loc[events['is_rainy'] == True, 'weather_label'] = 'rainy'
events.loc[(events['is_rainy'] == False) & (events['is_cold'] == False), 'weather_label'] = 'dry_warm'
events.loc[(events['is_rainy'] == False) & (events['is_cold'] == True), 'weather_label'] = 'dry_cold'
events.loc[(events['is_rainy'] == True) & (events['is_cold'] == True), 'weather_label'] = 'rainy_cold'

events.to_csv(os.path.join(OUTPUT_DIR, "safety_events_with_weather.csv"), index=False)
print(f"Saved: safety_events_with_weather.csv")

# Overall weather distribution of safety events
print(f"\nWeather distribution of safety events:")
for label, cnt in events['weather_label'].value_counts().items():
    print(f"  {label}: {cnt} ({cnt/len(events)*100:.1f}%)")


# ============================================================
# Step 3: Compute weather independence per cluster
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Computing weather independence scores")
print("=" * 60)

clusters = pd.read_csv(CLUSTERS_CSV)
print(f"Loaded {len(clusters)} clusters")

# Event labels from Layer 1
EVENT_LABELS = {
    "ANTI_LOCK_BRAKING_SYSTEM_STATE_CHANGE": "ABS",
    "ELECTRONIC_STABILITY_STATE_CHANGE": "ESC",
    "AUTONOMOUS_EMERGENCY_BRAKING_CHANGE": "AEB",
}
events['eventLabel'] = events['eventType'].map(EVENT_LABELS)

# Need cluster assignment — re-run DBSCAN or use proximity matching
# Since clusters are from DBSCAN, let's match events to nearest cluster
from sklearn.cluster import DBSCAN

events_geo = events.dropna(subset=['latitude', 'longitude']).copy()
for col in ['latitude', 'longitude']:
    events_geo[col] = pd.to_numeric(events_geo[col], errors='coerce')
events_geo = events_geo.dropna(subset=['latitude', 'longitude'])

# Match each event to nearest cluster center (within 50m)
cluster_lats = clusters['center_lat'].values
cluster_lons = clusters['center_lon'].values

def find_cluster(lat, lon, threshold_km=0.05):
    dlat = np.radians(cluster_lats - lat)
    dlon = np.radians(cluster_lons - lon)
    lat_r = np.radians(lat)
    clat_r = np.radians(cluster_lats)
    a = np.sin(dlat/2)**2 + np.cos(lat_r) * np.cos(clat_r) * np.sin(dlon/2)**2
    d = 2 * 6371 * np.arcsin(np.sqrt(a))
    idx = np.argmin(d)
    if d[idx] <= threshold_km:
        return clusters.iloc[idx]['cluster_id']
    return -1

events_geo['cluster_id'] = events_geo.apply(
    lambda r: find_cluster(r['latitude'], r['longitude']), axis=1
)

# Compute per-cluster weather stats
cluster_weather = []
for cid in clusters['cluster_id'].values:
    cev = events_geo[events_geo['cluster_id'] == cid]
    n_total = len(cev)
    if n_total == 0:
        cluster_weather.append({
            'cluster_id': cid,
            'n_events_matched': 0,
            'n_dry_warm': 0, 'n_dry_cold': 0,
            'n_rainy': 0, 'n_rainy_cold': 0, 'n_unknown': 0,
            'pct_dry': None, 'pct_rainy': None,
            'avg_temp_at_event': None,
            'weather_independence': None,
        })
        continue

    wl = cev['weather_label'].value_counts()
    n_dry_warm = wl.get('dry_warm', 0)
    n_dry_cold = wl.get('dry_cold', 0)
    n_rainy = wl.get('rainy', 0)
    n_rainy_cold = wl.get('rainy_cold', 0)
    n_unknown = wl.get('unknown', 0)

    n_dry = n_dry_warm + n_dry_cold
    n_wet = n_rainy + n_rainy_cold
    n_known = n_dry + n_wet

    pct_dry = n_dry / n_known if n_known > 0 else None
    pct_rainy = n_wet / n_known if n_known > 0 else None

    avg_temp = cev['avg_temp'].mean() if cev['avg_temp'].notna().any() else None

    # Weather independence score:
    # If events are evenly split between dry and rainy → high independence (fixed defect)
    # If events are concentrated in rainy → low independence (weather-related)
    # Score = min(pct_dry, pct_rainy) / 0.5, capped at 1.0
    # Perfect balance (50/50) → 1.0; all rainy → 0.0; all dry → 1.0 (still independent)
    if pct_dry is not None and n_known >= 3:
        # A location that triggers only in dry weather is ALSO weather-independent
        # (it's clearly a fixed defect). So score is high if pct_dry is high OR balanced.
        # Only score low if pct_rainy is very high (>70%) → weather-dependent
        weather_independence = max(0, 1.0 - max(0, pct_rainy - 0.5) * 2)
        # This gives: pct_rainy=0 → 1.0; pct_rainy=0.5 → 1.0; pct_rainy=0.75 → 0.5; pct_rainy=1.0 → 0.0
    else:
        weather_independence = None

    cluster_weather.append({
        'cluster_id': cid,
        'n_events_matched': n_total,
        'n_dry_warm': n_dry_warm,
        'n_dry_cold': n_dry_cold,
        'n_rainy': n_rainy,
        'n_rainy_cold': n_rainy_cold,
        'n_unknown': n_unknown,
        'pct_dry': pct_dry,
        'pct_rainy': pct_rainy,
        'avg_temp_at_event': avg_temp,
        'weather_independence': weather_independence,
    })

cw_df = pd.DataFrame(cluster_weather)

# Merge back to clusters
clusters = clusters.merge(cw_df, on='cluster_id', how='left')

# Update confidence score: add weather factor
# Final score = original confidence * weather_independence (or keep original if no weather data)
clusters['final_confidence'] = clusters.apply(
    lambda r: r['confidence_score'] * r['weather_independence']
    if pd.notna(r['weather_independence']) else r['confidence_score'] * 0.8,  # penalize unknowns slightly
    axis=1
)

clusters_final = clusters.sort_values('final_confidence', ascending=False)
clusters_final.to_csv(os.path.join(OUTPUT_DIR, "clusters_weather_corrected.csv"), index=False)
print("Saved: clusters_weather_corrected.csv")


# ============================================================
# Step 4: Visualizations
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Generating Visualizations")
print("=" * 60)

# --- Fig 1: Weather timeline ---
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.suptitle("Hourly Weather Conditions (Sep 27 – Oct 28, 2022)", fontsize=16, fontweight='bold')

weather_plot = weather_df.copy()
weather_plot['datetime'] = pd.to_datetime(weather_plot['hour_key'], format='%Y-%m-%d %H')
weather_plot = weather_plot.sort_values('datetime')

ax = axes[0]
ax.fill_between(weather_plot['datetime'], weather_plot['rain_ratio'].fillna(0),
                alpha=0.4, color='steelblue', label='Rain ratio (wiper-on %)')
ax.axhline(0.3, color='red', linestyle='--', linewidth=0.8, label='Rainy threshold (30%)')
ax.set_ylabel('Wiper-On Ratio')
ax.set_title('Rainfall Proxy (Wiper Activity)')
ax.legend(fontsize=8)
ax.set_ylim(0, 1)

ax = axes[1]
temp_valid = weather_plot[weather_plot['avg_temp'].notna()]
ax.plot(temp_valid['datetime'], temp_valid['avg_temp'], color='orange', linewidth=0.5, alpha=0.7)
ax.axhline(TEMP_COLD_C, color='blue', linestyle='--', linewidth=0.8, label=f'Cold ({TEMP_COLD_C}°C)')
ax.axhline(TEMP_FREEZE_C, color='darkblue', linestyle='--', linewidth=0.8, label=f'Freezing ({TEMP_FREEZE_C}°C)')
ax.set_ylabel('Temperature (°C)')
ax.set_xlabel('Date')
ax.set_title('Exterior Temperature')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_weather_timeline.png"))
plt.close()
print("  Saved: fig1_weather_timeline.png")

# --- Fig 2: Safety events by weather condition ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Safety Events by Weather Condition", fontsize=16, fontweight='bold')

weather_colors = {
    'dry_warm': '#f1c40f', 'dry_cold': '#3498db',
    'rainy': '#2ecc71', 'rainy_cold': '#9b59b6', 'unknown': '#bdc3c7'
}

# 2a. Overall pie
ax = axes[0]
wl_counts = events['weather_label'].value_counts()
colors = [weather_colors.get(l, '#bdc3c7') for l in wl_counts.index]
ax.pie(wl_counts.values, labels=wl_counts.index, colors=colors, autopct='%1.1f%%', startangle=90)
ax.set_title('All Safety Events')

# 2b. By event type
ax = axes[1]
ct = pd.crosstab(events['eventLabel'], events['weather_label'])
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
existing_cols = [c for c in weather_colors.keys() if c in ct_pct.columns]
if existing_cols:
    ct_pct[existing_cols].plot(
        kind='bar', stacked=True, ax=ax,
        color=[weather_colors[c] for c in existing_cols])
ax.set_title('Weather % by Event Type')
ax.set_ylabel('Percentage')
ax.set_xlabel('')
ax.legend(fontsize=7)

# 2c. Temperature histogram at event time
ax = axes[2]
temp_at_event = events['avg_temp'].dropna()
if len(temp_at_event) > 0:
    ax.hist(temp_at_event, bins=30, color='orange', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(TEMP_COLD_C, color='blue', linestyle='--', label=f'Cold ({TEMP_COLD_C}°C)')
    ax.axvline(TEMP_FREEZE_C, color='darkblue', linestyle='--', label=f'Freezing ({TEMP_FREEZE_C}°C)')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Event Count')
    ax.set_title('Temperature at Safety Event')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_events_by_weather.png"))
plt.close()
print("  Saved: fig2_events_by_weather.png")

# --- Fig 3: Weather independence vs confidence ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Weather-Corrected Hotspot Confidence", fontsize=16, fontweight='bold')

valid_wi = clusters_final[clusters_final['weather_independence'].notna()]

ax = axes[0]
sc = ax.scatter(valid_wi['confidence_score'], valid_wi['weather_independence'],
                c=valid_wi['final_confidence'], cmap='YlOrRd',
                s=valid_wi['event_count'] * 0.8 + 5, alpha=0.7,
                edgecolors='black', linewidths=0.2)
plt.colorbar(sc, ax=ax, label='Final Confidence')
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Layer 2 Confidence Score')
ax.set_ylabel('Weather Independence (1.0 = fixed defect)')
ax.set_title('Confidence vs Weather Independence')
for _, r in clusters_final.head(5).iterrows():
    if pd.notna(r['weather_independence']):
        ax.annotate(f"#{int(r['cluster_id'])}",
                    (r['confidence_score'], r['weather_independence']),
                    fontsize=7, fontweight='bold')

# Geographic map colored by final confidence
ax = axes[1]
sc = ax.scatter(clusters_final['center_lon'], clusters_final['center_lat'],
                c=clusters_final['final_confidence'], cmap='YlOrRd',
                s=clusters_final['event_count'] * 0.8 + 5,
                alpha=0.7, edgecolors='black', linewidths=0.2)
plt.colorbar(sc, ax=ax, label='Final Confidence')
for _, r in clusters_final.head(10).iterrows():
    ax.annotate(f"#{int(r['cluster_id'])}",
                (r['center_lon'], r['center_lat']),
                fontsize=6, fontweight='bold', color='darkred')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Weather-Corrected Hotspot Map')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_weather_corrected.png"))
plt.close()
print("  Saved: fig3_weather_corrected.png")

# --- Fig 4: Top clusters weather breakdown ---
fig, ax = plt.subplots(figsize=(14, 8))
top20 = clusters_final.head(20).copy()
top20['label'] = top20.apply(
    lambda r: f"#{int(r['cluster_id'])} (ZIP {r['top_postal']})"
    if pd.notna(r['top_postal']) else f"#{int(r['cluster_id'])}", axis=1)

y_pos = range(len(top20))
left = np.zeros(len(top20))
for wl, color in weather_colors.items():
    col = f'n_{wl}'
    if col in top20.columns:
        vals = top20[col].fillna(0).values.astype(float)
        ax.barh(y_pos, vals, left=left, color=color, label=wl, height=0.6)
        left += vals

ax.set_yticks(y_pos)
ax.set_yticklabels(top20['label'], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Event Count')
ax.set_title('Top 20 Hotspots: Weather Condition Breakdown',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_top20_weather.png"))
plt.close()
print("  Saved: fig4_top20_weather.png")


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
w("TASK 4 LAYER 3: WEATHER CORRECTION REPORT")
w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w("=" * 60)

w(f"\n--- Weather Data Summary ---")
w(f"  Wiper events processed: {total_wiper:,}")
w(f"  Temperature readings processed: {total_temp:,}")
w(f"  Hours with weather data: {len(weather_df)}")
w(f"  Rainy hours (>30% wiper-on): {n_rainy_hours} ({n_rainy_hours/len(weather_df)*100:.1f}%)")
w(f"  Cold hours (<{TEMP_COLD_C}°C): {n_cold_hours} ({n_cold_hours/len(weather_df)*100:.1f}%)")
w(f"  Freezing hours (<{TEMP_FREEZE_C}°C): {n_freezing_hours} ({n_freezing_hours/len(weather_df)*100:.1f}%)")

if len(weather_df) > 0 and weather_df['avg_temp'].notna().any():
    temps = weather_df['avg_temp'].dropna()
    w(f"  Temperature range: {temps.min():.1f}°C to {temps.max():.1f}°C (mean {temps.mean():.1f}°C)")

w(f"\n--- Safety Events Weather Distribution ---")
for label, cnt in events['weather_label'].value_counts().items():
    w(f"  {label:15s}: {cnt:,} ({cnt/len(events)*100:.1f}%)")

w(f"\n--- Weather Independence Scoring ---")
w(f"  Formula: independence = max(0, 1 - max(0, pct_rainy - 0.5) * 2)")
w(f"  Interpretation:")
w(f"    1.0 = events occur in all conditions → FIXED DEFECT")
w(f"    0.5 = 75% rainy → LIKELY WEATHER-RELATED")
w(f"    0.0 = 100% rainy → WEATHER-DEPENDENT")

wi_valid = clusters_final[clusters_final['weather_independence'].notna()]
if len(wi_valid) > 0:
    w(f"  Clusters with score: {len(wi_valid)}")
    w(f"  Mean independence: {wi_valid['weather_independence'].mean():.3f}")
    w(f"  High independence (>0.8): {(wi_valid['weather_independence'] > 0.8).sum()}")
    w(f"  Low independence (<0.5): {(wi_valid['weather_independence'] < 0.5).sum()}")

w(f"\n--- Final Confidence = Layer2 Score × Weather Independence ---")

w(f"\n--- Top 20 Weather-Corrected Hotspots ---")
w(f"  {'Rk':>3s} {'ID':>4s} {'Evt':>4s} {'Brake':>6s} {'Dry':>4s} {'Rain':>5s} "
  f"{'WI':>5s} {'Final':>6s} {'ZIP':>8s} {'Lat':>9s} {'Lon':>10s}")
for rank, (_, r) in enumerate(clusters_final.head(20).iterrows(), 1):
    zp = str(r['top_postal']) if pd.notna(r['top_postal']) else "—"
    wi_str = f"{r['weather_independence']:.2f}" if pd.notna(r['weather_independence']) else "—"
    n_dry = int(r.get('n_dry_warm', 0) or 0) + int(r.get('n_dry_cold', 0) or 0)
    n_rain = int(r.get('n_rainy', 0) or 0) + int(r.get('n_rainy_cold', 0) or 0)
    w(f"  {rank:>3d} {int(r['cluster_id']):>4d} {int(r['event_count']):>4d} "
      f"{int(r['hard_brake_nearby']):>6d} {n_dry:>4d} {n_rain:>5d} "
      f"{wi_str:>5s} {r['final_confidence']:>6.3f} "
      f"{zp:>8s} {r['center_lat']:>9.5f} {r['center_lon']:>10.5f}")

# Identify weather-dependent clusters
w(f"\n--- Weather-Dependent Hotspots (independence < 0.5) ---")
weather_dep = clusters_final[
    (clusters_final['weather_independence'].notna()) &
    (clusters_final['weather_independence'] < 0.5) &
    (clusters_final['event_count'] >= 6)
]
w(f"  Count: {len(weather_dep)} clusters with 6+ events and WI < 0.5")
if len(weather_dep) > 0:
    for _, r in weather_dep.head(10).iterrows():
        zp = str(r['top_postal']) if pd.notna(r['top_postal']) else "—"
        n_dry = int(r.get('n_dry_warm', 0) or 0) + int(r.get('n_dry_cold', 0) or 0)
        n_rain = int(r.get('n_rainy', 0) or 0) + int(r.get('n_rainy_cold', 0) or 0)
        w(f"  #{int(r['cluster_id'])}: {int(r['event_count'])} events, "
          f"dry={n_dry}, rain={n_rain}, WI={r['weather_independence']:.2f}, ZIP={zp}")

# Weather-independent (confirmed fixed defects)
w(f"\n--- Confirmed Fixed Defects (independence >= 0.8, 6+ events) ---")
confirmed = clusters_final[
    (clusters_final['weather_independence'].notna()) &
    (clusters_final['weather_independence'] >= 0.8) &
    (clusters_final['event_count'] >= 6)
].sort_values('final_confidence', ascending=False)
w(f"  Count: {len(confirmed)} clusters")
if len(confirmed) > 0:
    for _, r in confirmed.head(20).iterrows():
        zp = str(r['top_postal']) if pd.notna(r['top_postal']) else "—"
        n_dry = int(r.get('n_dry_warm', 0) or 0) + int(r.get('n_dry_cold', 0) or 0)
        n_rain = int(r.get('n_rainy', 0) or 0) + int(r.get('n_rainy_cold', 0) or 0)
        w(f"  #{int(r['cluster_id'])}: {int(r['event_count'])} events, "
          f"dry={n_dry}, rain={n_rain}, WI={r['weather_independence']:.2f}, "
          f"final={r['final_confidence']:.3f}, ZIP={zp}")

report_path = os.path.join(OUTPUT_DIR, "layer3_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(L))
print(f"\nReport: {report_path}")
print(f"All outputs: {OUTPUT_DIR}")
print("Done!")