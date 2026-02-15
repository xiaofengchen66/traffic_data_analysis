#!/usr/bin/env python3
"""
Task 4 — Layer 3 v2: Weather Correction

Key fix from v1: switched from hourly wiper ratio (broken due to
activation/deactivation asymmetry) to DAILY wiper activity count
with data-driven quantile threshold.

Approach:
1. Count front-wiper ACTIVATED events per DAY (not ratio, not hourly)
2. Use distribution to classify days as dry/rainy (quantile-based)
3. Use hourly temperature for cold/freezing labels
4. Compute weather independence per cluster

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

RAW_EVENTS_CSV = "/home/qiqingh/Desktop/traffic_data_analysis/task4_outputs/layer1/version_1/abs_esc_aeb_raw_events.csv"
CLUSTERS_CSV = "/home/qiqingh/Desktop/traffic_data_analysis/task4_outputs/layer2/clusters_with_brake_validation.csv"

TEMP_COLD_C = 5.0
TEMP_FREEZE_C = 0.0

# ============================================================
# Step 1: Scan — collect daily wiper counts + hourly temps
# ============================================================
print("=" * 60)
print("Step 1: Scanning for wiper + temperature data")
print("=" * 60)

# Daily wiper counts (front only, activated only)
daily_wiper_on = defaultdict(int)      # date_str -> count
daily_wiper_off = defaultdict(int)
daily_total_events = defaultdict(int)  # to compute wiper rate

# Hourly temperature
hourly_temp = defaultdict(lambda: {"sum": 0.0, "count": 0, "min": 999.0, "max": -999.0})

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and not f.startswith("._")])
print(f"Scanning {len(files)} files...\n")

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

                # Extract date
                try:
                    date_str = row[2][:10]  # "2022-09-28"
                    hour_key = row[2][:10] + " " + row[2][11:13]  # "2022-09-28 14"
                except:
                    continue

                daily_total_events[date_str] += 1

                try:
                    event = json.loads(row[6]) if row[6] else {}
                except:
                    event = {}

                etype = event.get("eventType", "")

                # Wiper: count front activations per day
                if etype == "WIPER_STATE_CHANGE":
                    total_wiper += 1
                    meta = event.get("eventMetadata", {})
                    wiper_state = meta.get("wiperStateChangeType", "").upper()
                    wiper_id = meta.get("wiperIdentifier", "").upper()

                    if wiper_id == "FRONT":
                        if wiper_state == "ACTIVATED":
                            daily_wiper_on[date_str] += 1
                        elif wiper_state == "DEACTIVATED":
                            daily_wiper_off[date_str] += 1

                # Temperature from any event
                try:
                    metrics = json.loads(row[4]) if row[4] else {}
                except:
                    metrics = {}
                env = metrics.get("environment", {})
                if isinstance(env, dict):
                    temp = env.get("exteriorTemperature")
                    if temp is not None:
                        total_temp += 1
                        h = hourly_temp[hour_key]
                        h["sum"] += temp
                        h["count"] += 1
                        if temp < h["min"]: h["min"] = temp
                        if temp > h["max"]: h["max"] = temp

    except:
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


# ============================================================
# Step 2: Classify days as dry/rainy using data-driven threshold
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Classifying daily weather")
print("=" * 60)

# Build daily weather table
all_dates = sorted(set(list(daily_wiper_on.keys()) + list(daily_total_events.keys())))
daily_rows = []
for d in all_dates:
    w_on = daily_wiper_on.get(d, 0)
    w_off = daily_wiper_off.get(d, 0)
    total = daily_total_events.get(d, 0)

    # Daily average temperature
    day_temps = []
    for h in range(24):
        hk = f"{d} {h:02d}"
        if hk in hourly_temp and hourly_temp[hk]["count"] > 0:
            day_temps.append(hourly_temp[hk]["sum"] / hourly_temp[hk]["count"])
    avg_temp = np.mean(day_temps) if day_temps else None
    min_temp = min(day_temps) if day_temps else None

    daily_rows.append({
        "date": d,
        "front_wiper_on": w_on,
        "front_wiper_off": w_off,
        "front_wiper_total": w_on + w_off,
        "total_events": total,
        "wiper_rate": (w_on + w_off) / total if total > 0 else 0,
        "avg_temp": avg_temp,
        "min_temp": min_temp,
    })

daily_df = pd.DataFrame(daily_rows)

# Data-driven threshold: use distribution of front_wiper_on
# Days with very few activations = dry; days with many = rainy
wiper_counts = daily_df['front_wiper_on'].values
median_wiper = np.median(wiper_counts)
q25 = np.percentile(wiper_counts, 25)
q75 = np.percentile(wiper_counts, 75)

# Use median as threshold: below median = dry, above = rainy
# This gives roughly 50/50 split which is reasonable for Buffalo in fall
RAIN_THRESHOLD = median_wiper

daily_df['is_rainy'] = daily_df['front_wiper_on'] > RAIN_THRESHOLD
daily_df['is_cold'] = daily_df['avg_temp'].apply(lambda x: x < TEMP_COLD_C if pd.notna(x) else False)
daily_df['is_freezing'] = daily_df['avg_temp'].apply(lambda x: x < TEMP_FREEZE_C if pd.notna(x) else False)

# Weather label
def day_weather_label(row):
    rainy = row['is_rainy']
    cold = row['is_cold']
    if rainy and cold:
        return 'rainy_cold'
    elif rainy:
        return 'rainy'
    elif cold:
        return 'dry_cold'
    else:
        return 'dry_warm'

daily_df['weather_label'] = daily_df.apply(day_weather_label, axis=1)

daily_df.to_csv(os.path.join(OUTPUT_DIR, "daily_weather_profiles.csv"), index=False)

n_rainy = daily_df['is_rainy'].sum()
n_dry = (~daily_df['is_rainy']).sum()
n_cold = daily_df['is_cold'].sum()

print(f"  Days total: {len(daily_df)}")
print(f"  Wiper activation distribution: min={wiper_counts.min()}, "
      f"Q25={q25:.0f}, median={median_wiper:.0f}, Q75={q75:.0f}, max={wiper_counts.max()}")
print(f"  Rain threshold (median): {RAIN_THRESHOLD:.0f} front wiper activations/day")
print(f"  Rainy days: {n_rainy} ({n_rainy/len(daily_df)*100:.1f}%)")
print(f"  Dry days:   {n_dry} ({n_dry/len(daily_df)*100:.1f}%)")
print(f"  Cold days:  {n_cold}")
print(f"\n  Daily weather breakdown:")
for label, cnt in daily_df['weather_label'].value_counts().items():
    print(f"    {label}: {cnt}")


# ============================================================
# Step 3: Label safety events with daily weather
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Labeling safety events with weather")
print("=" * 60)

events = pd.read_csv(RAW_EVENTS_CSV)
events['capturedDateTime'] = pd.to_datetime(events['capturedDateTime'], errors='coerce')
events['date'] = events['capturedDateTime'].dt.strftime('%Y-%m-%d')
print(f"Loaded {len(events)} ABS/ESC/AEB events")

# Also get hourly temp for each event
events['hour_key'] = events['capturedDateTime'].dt.strftime('%Y-%m-%d %H')
events['event_temp'] = events['hour_key'].apply(
    lambda hk: hourly_temp[hk]["sum"] / hourly_temp[hk]["count"]
    if hk in hourly_temp and hourly_temp[hk]["count"] > 0 else None
)

# Merge daily weather
daily_lookup = daily_df.set_index('date')[['is_rainy', 'is_cold', 'weather_label', 'avg_temp']]
daily_lookup.columns = ['day_is_rainy', 'day_is_cold', 'day_weather_label', 'day_avg_temp']
events = events.merge(daily_lookup, left_on='date', right_index=True, how='left')

EVENT_LABELS = {
    "ANTI_LOCK_BRAKING_SYSTEM_STATE_CHANGE": "ABS",
    "ELECTRONIC_STABILITY_STATE_CHANGE": "ESC",
    "AUTONOMOUS_EMERGENCY_BRAKING_CHANGE": "AEB",
}
events['eventLabel'] = events['eventType'].map(EVENT_LABELS)

events.to_csv(os.path.join(OUTPUT_DIR, "safety_events_with_weather.csv"), index=False)

print(f"\nWeather distribution of safety events (by day classification):")
for label, cnt in events['day_weather_label'].value_counts().items():
    print(f"  {label}: {cnt} ({cnt/len(events)*100:.1f}%)")


# ============================================================
# Step 4: Weather independence per cluster
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Computing weather independence scores")
print("=" * 60)

clusters = pd.read_csv(CLUSTERS_CSV)
print(f"Loaded {len(clusters)} clusters")

# Match events to clusters (nearest within 50m)
for col in ['latitude', 'longitude']:
    events[col] = pd.to_numeric(events[col], errors='coerce')
events_geo = events.dropna(subset=['latitude', 'longitude']).copy()

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
    return clusters.iloc[idx]['cluster_id'] if d[idx] <= threshold_km else -1

events_geo['cluster_id'] = events_geo.apply(
    lambda r: find_cluster(r['latitude'], r['longitude']), axis=1
)

# Per-cluster weather stats
cluster_weather = []
for cid in clusters['cluster_id'].values:
    cev = events_geo[events_geo['cluster_id'] == cid]
    n_total = len(cev)

    if n_total == 0:
        cluster_weather.append({
            'cluster_id': cid, 'n_events_matched': 0,
            'n_dry_warm': 0, 'n_dry_cold': 0, 'n_rainy': 0, 'n_rainy_cold': 0,
            'pct_dry': None, 'pct_rainy': None,
            'avg_temp_at_event': None, 'weather_independence': None,
        })
        continue

    wl = cev['day_weather_label'].value_counts()
    n_dry_warm = wl.get('dry_warm', 0)
    n_dry_cold = wl.get('dry_cold', 0)
    n_rainy = wl.get('rainy', 0)
    n_rainy_cold = wl.get('rainy_cold', 0)

    n_dry = n_dry_warm + n_dry_cold
    n_wet = n_rainy + n_rainy_cold
    n_known = n_dry + n_wet

    pct_dry = n_dry / n_known if n_known > 0 else None
    pct_rainy = n_wet / n_known if n_known > 0 else None

    avg_temp = cev['event_temp'].mean() if cev['event_temp'].notna().any() else None

    # Weather independence score
    # 1.0 = occurs equally in all conditions (fixed defect)
    # 0.0 = 100% rainy (weather-dependent)
    # Penalize only if rainy fraction is disproportionately high
    # Background rainy rate = n_rainy_days / total_days
    if pct_rainy is not None and n_known >= 3:
        bg_rain_rate = n_rainy / len(daily_df)
        if bg_rain_rate > 0 and bg_rain_rate < 1:
            # Ratio of observed rainy fraction to background rainy rate
            # If ratio ≈ 1 → events follow background weather → weather independent
            # If ratio >> 1 → events concentrated in rain → weather dependent
            rain_enrichment = pct_rainy / bg_rain_rate
            # Score: 1.0 when enrichment <= 1; drops toward 0 when enrichment >= 2
            weather_independence = max(0.0, min(1.0, 2.0 - rain_enrichment))
        else:
            weather_independence = 0.5  # can't determine
    else:
        weather_independence = None

    cluster_weather.append({
        'cluster_id': cid, 'n_events_matched': n_total,
        'n_dry_warm': n_dry_warm, 'n_dry_cold': n_dry_cold,
        'n_rainy': n_rainy, 'n_rainy_cold': n_rainy_cold,
        'pct_dry': pct_dry, 'pct_rainy': pct_rainy,
        'avg_temp_at_event': avg_temp,
        'weather_independence': weather_independence,
    })

cw_df = pd.DataFrame(cluster_weather)
clusters = clusters.merge(cw_df, on='cluster_id', how='left')

# Final confidence = Layer2 score × weather independence
clusters['final_confidence'] = clusters.apply(
    lambda r: r['confidence_score'] * r['weather_independence']
    if pd.notna(r['weather_independence']) else r['confidence_score'] * 0.8,
    axis=1
)

clusters_final = clusters.sort_values('final_confidence', ascending=False)
clusters_final.to_csv(os.path.join(OUTPUT_DIR, "clusters_weather_corrected.csv"), index=False)
print("Saved: clusters_weather_corrected.csv")


# ============================================================
# Step 5: Visualizations
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Generating Visualizations")
print("=" * 60)

weather_colors = {
    'dry_warm': '#f1c40f', 'dry_cold': '#3498db',
    'rainy': '#2ecc71', 'rainy_cold': '#9b59b6',
}

# --- Fig 1: Daily weather overview ---
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle("Daily Weather Conditions (Sep 27 – Oct 28, 2022)", fontsize=16, fontweight='bold')

daily_plot = daily_df.copy()
daily_plot['date_dt'] = pd.to_datetime(daily_plot['date'])

ax = axes[0]
colors_bar = ['#2ecc71' if r else '#f1c40f' for r in daily_plot['is_rainy']]
ax.bar(daily_plot['date_dt'], daily_plot['front_wiper_on'], color=colors_bar, alpha=0.8)
ax.axhline(RAIN_THRESHOLD, color='red', linestyle='--', linewidth=1,
           label=f'Rain threshold ({RAIN_THRESHOLD:.0f})')
ax.set_ylabel('Front Wiper Activations')
ax.set_title('Daily Wiper Activity (green=rainy, yellow=dry)')
ax.legend()

ax = axes[1]
temp_valid = daily_plot[daily_plot['avg_temp'].notna()]
ax.bar(temp_valid['date_dt'], temp_valid['avg_temp'], color='orange', alpha=0.7)
ax.axhline(TEMP_COLD_C, color='blue', linestyle='--', linewidth=1, label=f'Cold ({TEMP_COLD_C}°C)')
ax.axhline(TEMP_FREEZE_C, color='darkblue', linestyle='--', linewidth=1, label=f'Freezing ({TEMP_FREEZE_C}°C)')
ax.set_ylabel('Avg Temperature (°C)')
ax.set_title('Daily Average Temperature')
ax.legend()

ax = axes[2]
label_colors = [weather_colors.get(l, 'gray') for l in daily_plot['weather_label']]
ax.bar(daily_plot['date_dt'], [1]*len(daily_plot), color=label_colors)
ax.set_yticks([])
ax.set_title('Daily Weather Classification')
# Legend
from matplotlib.patches import Patch
legend_patches = [Patch(color=c, label=l) for l, c in weather_colors.items()]
ax.legend(handles=legend_patches, fontsize=8)
ax.set_xlabel('Date')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_daily_weather.png"))
plt.close()
print("  Saved: fig1_daily_weather.png")

# --- Fig 2: Safety events by weather ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Safety Events by Weather Condition", fontsize=16, fontweight='bold')

ax = axes[0]
wl_counts = events['day_weather_label'].value_counts()
existing_labels = [l for l in weather_colors if l in wl_counts.index]
colors_pie = [weather_colors[l] for l in existing_labels]
ax.pie([wl_counts[l] for l in existing_labels], labels=existing_labels,
       colors=colors_pie, autopct='%1.1f%%', startangle=90)
ax.set_title('All Safety Events')

ax = axes[1]
ct = pd.crosstab(events['eventLabel'], events['day_weather_label'])
existing_cols = [c for c in weather_colors if c in ct.columns]
if existing_cols:
    ct_pct = ct[existing_cols].div(ct[existing_cols].sum(axis=1), axis=0) * 100
    ct_pct.plot(kind='bar', stacked=True, ax=ax,
                color=[weather_colors[c] for c in existing_cols])
ax.set_title('Weather % by Event Type')
ax.set_ylabel('Percentage')
ax.set_xlabel('')
ax.legend(fontsize=7)

ax = axes[2]
temp_at_event = events['event_temp'].dropna()
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
if len(valid_wi) > 0:
    sc = ax.scatter(valid_wi['confidence_score'], valid_wi['weather_independence'],
                    c=valid_wi['final_confidence'], cmap='YlOrRd',
                    s=valid_wi['event_count'] * 0.8 + 5, alpha=0.7,
                    edgecolors='black', linewidths=0.2)
    plt.colorbar(sc, ax=ax, label='Final Confidence')
    for _, r in clusters_final.head(5).iterrows():
        if pd.notna(r['weather_independence']):
            ax.annotate(f"#{int(r['cluster_id'])}",
                        (r['confidence_score'], r['weather_independence']),
                        fontsize=7, fontweight='bold')
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Layer 2 Confidence Score')
ax.set_ylabel('Weather Independence (1.0=fixed, 0.0=weather)')
ax.set_title('Confidence vs Weather Independence')

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

# --- Fig 4: Top 20 weather breakdown ---
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
ax.set_title('Top 20 Hotspots: Weather Condition Breakdown', fontsize=14, fontweight='bold')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_top20_weather.png"))
plt.close()
print("  Saved: fig4_top20_weather.png")


# ============================================================
# Step 6: Report
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Writing Report")
print("=" * 60)

L = []
def w(t=""):
    L.append(t)

w("=" * 60)
w("TASK 4 LAYER 3 v2: WEATHER CORRECTION REPORT")
w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w(f"Method: Daily wiper activation count + quantile threshold")
w("=" * 60)

w(f"\n--- Weather Data Summary ---")
w(f"  Wiper events scanned: {total_wiper:,}")
w(f"  Temperature readings: {total_temp:,}")
w(f"  Days with data: {len(daily_df)}")
w(f"  Scan time: {elapsed_scan:.0f}s ({elapsed_scan/60:.1f} min)")

w(f"\n--- Wiper Activation Distribution (per day) ---")
w(f"  Min:    {wiper_counts.min()}")
w(f"  Q25:    {q25:.0f}")
w(f"  Median: {median_wiper:.0f}")
w(f"  Q75:    {q75:.0f}")
w(f"  Max:    {wiper_counts.max()}")
w(f"  Threshold (median): {RAIN_THRESHOLD:.0f}")

w(f"\n--- Daily Weather Classification ---")
for label, cnt in daily_df['weather_label'].value_counts().items():
    w(f"  {label:15s}: {cnt} days ({cnt/len(daily_df)*100:.1f}%)")

if daily_df['avg_temp'].notna().any():
    temps = daily_df['avg_temp'].dropna()
    w(f"  Temperature range: {temps.min():.1f}°C to {temps.max():.1f}°C (mean {temps.mean():.1f}°C)")

w(f"\n--- Safety Events Weather Distribution ---")
for label, cnt in events['day_weather_label'].value_counts().items():
    w(f"  {label:15s}: {cnt:,} ({cnt/len(events)*100:.1f}%)")

w(f"\n--- Weather Independence Scoring ---")
w(f"  Method: Rain enrichment ratio")
w(f"  Background rainy rate: {n_rainy/len(daily_df)*100:.1f}% of days")
w(f"  rain_enrichment = pct_rainy_events / background_rainy_rate")
w(f"  independence = max(0, min(1, 2 - rain_enrichment))")
w(f"  Interpretation:")
w(f"    enrichment ≈ 1.0 → events follow weather → independence = 1.0 (FIXED DEFECT)")
w(f"    enrichment ≈ 2.0 → 2x over-represented in rain → independence = 0.0 (WEATHER)")
w(f"    enrichment ≈ 1.5 → moderately rain-biased → independence = 0.5")

wi_valid = clusters_final[clusters_final['weather_independence'].notna()]
if len(wi_valid) > 0:
    w(f"\n  Clusters with score: {len(wi_valid)}")
    w(f"  Mean independence: {wi_valid['weather_independence'].mean():.3f}")
    w(f"  High independence (≥0.8): {(wi_valid['weather_independence'] >= 0.8).sum()}")
    w(f"  Medium (0.5-0.8): {((wi_valid['weather_independence'] >= 0.5) & (wi_valid['weather_independence'] < 0.8)).sum()}")
    w(f"  Low (<0.5): {(wi_valid['weather_independence'] < 0.5).sum()}")

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

# Confirmed fixed defects
w(f"\n--- Confirmed Fixed Defects (WI ≥ 0.8, 6+ events) ---")
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

# Weather-dependent
w(f"\n--- Weather-Dependent Hotspots (WI < 0.5, 6+ events) ---")
weather_dep = clusters_final[
    (clusters_final['weather_independence'].notna()) &
    (clusters_final['weather_independence'] < 0.5) &
    (clusters_final['event_count'] >= 6)
]
w(f"  Count: {len(weather_dep)} clusters")
if len(weather_dep) > 0:
    for _, r in weather_dep.head(10).iterrows():
        zp = str(r['top_postal']) if pd.notna(r['top_postal']) else "—"
        n_dry = int(r.get('n_dry_warm', 0) or 0) + int(r.get('n_dry_cold', 0) or 0)
        n_rain = int(r.get('n_rainy', 0) or 0) + int(r.get('n_rainy_cold', 0) or 0)
        w(f"  #{int(r['cluster_id'])}: {int(r['event_count'])} events, "
          f"dry={n_dry}, rain={n_rain}, WI={r['weather_independence']:.2f}, ZIP={zp}")

report_path = os.path.join(OUTPUT_DIR, "layer3_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(L))
print(f"\nReport: {report_path}")
print(f"All outputs: {OUTPUT_DIR}")
print("Done!")