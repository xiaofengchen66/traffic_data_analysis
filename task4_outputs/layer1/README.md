# Task 4 — Layer 1: ABS / ESC / AEB Safety-System Hotspot Analysis

> Last updated: 2026-02-15  
> Status: Complete (v1 loose params + v2 tight params)

---

## Overview

This layer identifies **road surface anomaly candidates** by spatially clustering rare safety-system activation events (ABS, ESC, AEB) from one month of vehicle telematics data in the Buffalo, NY metropolitan area. The premise is simple: if multiple vehicles trigger safety systems at the same 50-meter spot across different days and trips, the cause is almost certainly a fixed road defect rather than driver behavior.

---

## Data Source

- **Input**: 744 hourly CSV files, ~101.6 million event records  
- **Time span**: 2022-09-27 to 2022-10-28 (~31 days)  
- **Geographic scope**: Buffalo, NY metro area (114 unique postal codes)  
- **Target events extracted**: 5,660 records  
  - ABS (Anti-Lock Braking System): 5,103 (90.2%)  
  - ESC (Electronic Stability Control): 446 (7.9%)  
  - AEB (Autonomous Emergency Braking): 111 (1.9%)  
- **Field coverage on target events**: 100% for speed, accelerationX, accelerationY, heading, and location

---

## Method

### Extraction
Scanned all 744 CSVs using Python `csv.reader` with `escapechar='\\'` to handle non-standard JSON quoting. Filtered for three event types: `ANTI_LOCK_BRAKING_SYSTEM_STATE_CHANGE`, `ELECTRONIC_STABILITY_STATE_CHANGE`, `AUTONOMOUS_EMERGENCY_BRAKING_CHANGE`.

### Spatial Clustering
Applied DBSCAN with haversine distance metric on (latitude, longitude) pairs.

| Version | Radius (eps) | Min samples | Clusters | Clustered events | Noise |
|---------|-------------|-------------|----------|-----------------|-------|
| v1      | 200m        | 2           | 1,270    | 5,660 (100%)    | 0 (0%) |
| **v2**  | **50m**     | **4**       | **431**  | **3,048 (53.9%)** | **2,612 (46.1%)** |

v1 was too loose (zero noise = everything clustered). v2 with tighter parameters produces meaningful separation between persistent hotspots and isolated/random events.

### Persistence Filter
Clusters appearing on **3+ different calendar dates** are classified as **persistent hotspots** — strong candidates for fixed road defects. Single-date or two-date clusters may be weather-related or coincidental.

---

## Key Findings (v2 — 50m, min 4)

### Summary Statistics
- **431 spatial clusters** identified from 5,660 safety-system events
- **181 persistent hotspots** (events on 3+ different dates)
- Events span 32 unique dates, 2,565 unique journeys, 114 postal codes
- Speed at trigger: mean 30.4 km/h, median 25.9 km/h, max 159.0 km/h

### Top 10 Hotspot Clusters

| Rank | Cluster | Events | ABS | ESC | AEB | Days | Trips | Avg Speed | ZIP   | Coordinates |
|------|---------|--------|-----|-----|-----|------|-------|-----------|-------|-------------|
| 1    | #0      | 202    | 68  | 134 | 0   | 17   | 31    | 28 km/h   | 14052 | 42.767, -78.645 |
| 2    | #26     | 72     | 72  | 0   | 0   | 20   | 36    | 28 km/h   | 14225 | 42.946, -78.725 |
| 3    | #254    | 66     | 66  | 0   | 0   | 15   | 33    | 33 km/h   | 14086 | 42.924, -78.623 |
| 4    | #86     | 36     | 36  | 0   | 0   | 15   | 18    | 36 km/h   | 14204 | 42.884, -78.868 |
| 5    | #60     | 32     | 32  | 0   | 0   | 10   | 16    | 36 km/h   | 14261 | 42.991, -78.790 |
| 6    | #85     | 26     | 26  | 0   | 0   | 10   | 13    | 43 km/h   | 14214 | 42.943, -78.844 |
| 7    | #116    | 26     | 26  | 0   | 0   | 11   | 13    | 43 km/h   | 14227 | 42.884, -78.766 |
| 8    | #255    | 24     | 24  | 0   | 0   | 9    | 12    | 29 km/h   | 14086 | 42.921, -78.638 |
| 9    | #271    | 24     | 24  | 0   | 0   | 10   | 12    | 32 km/h   | 14086 | 42.921, -78.642 |
| 10   | #53     | 22     | 22  | 0   | 0   | 6    | 11    | 37 km/h   | 14059 | 42.822, -78.637 |

### Notable Findings

**Cluster #0 is an extreme outlier.** It contains 202 events (including 134 ESC activations) within a 50m radius, triggered on 17 different dates by 31 different trips. Since the entire dataset only has 446 ESC events total, **30% of all ESC activations in the Buffalo area occur at this single location** (ZIP 14052, near East Aurora). This is an almost certain indicator of a road surface or geometry defect (possibly a curve with damaged pavement or poor drainage).

**High-speed hotspots are safety-critical.** Several clusters show safety-system triggers at elevated speeds:
- Cluster #74: avg 96 km/h (ZIP 14207)
- Cluster #306: avg 83 km/h (ZIP 14224)
- Cluster #92: avg 84 km/h (ZIP 14206)
- Cluster #49: avg 75 km/h (ZIP 14207)

These likely correspond to highway segments with surface issues and represent the highest-risk locations.

**ZIP code 14086 (Lancaster) has three nearby clusters** (#254, #255, #271) totaling 114 events — suggesting a corridor-level problem rather than a single point defect.

---

## Confidence Framework

| Confidence | Criteria | Count | Interpretation |
|------------|----------|-------|----------------|
| **High**   | 3+ dates, 3+ journeys, 6+ events | ~60 clusters | Fixed road defect (pothole, surface damage, drainage) |
| **Medium** | 2 dates or 4-5 events | ~180 clusters | Possible defect; may also be weather or traffic-related |
| **Low / Noise** | <4 events or single occurrence | 2,612 points | Isolated events, likely driver behavior or random |

---

## Output Files

### v1 (200m, min_samples=2) — `task4_outputs/layer1/version_1/`
| File | Description |
|------|-------------|
| `abs_esc_aeb_raw_events.csv` | All 5,660 extracted events with parsed fields (reused by v2) |
| `hotspot_clusters.csv` | Cluster summary table (1,270 clusters) |
| `layer1_report.txt` | Full text report |
| `fig1_overview.png` | 2x2 overview: counts, hourly, speed, daily trend |
| `fig2_geo_clusters.png` | Geographic scatter + cluster visualization |
| `fig3_top_clusters_detail.png` | Top 8 clusters zoomed in |
| `hotspot_map_interactive.html` | Folium interactive map with all events + cluster circles |

### v2 (50m, min_samples=4) — `task4_outputs/layer1/version_2/`
Same file structure as v1, with tighter clustering parameters. Additionally includes:
| File | Description |
|------|-------------|
| `fig4_persistent.png` | Map highlighting only persistent hotspots (3+ dates) |
| `layer1_v3_report.txt` | Full text report with persistence analysis |

---

## Code

| File | Location | Description |
|------|----------|-------------|
| `task4_layer1_safety_hotspots_version1.py` | `code/task4/` | v1: DBSCAN 200m, min 2 |
| `task4_layer1_safety_hotspots_version2.py` | `code/task4/` | v2: DBSCAN 50m, min 4; reuses v1 raw CSV cache |

### Dependencies
```bash
pip3 install --upgrade pandas matplotlib seaborn scikit-learn folium scipy
```

### Usage
```bash
# v2 automatically reuses v1's extracted raw events CSV
python3 code/task4/task4_layer1_safety_hotspots_version2.py
```

---

## Limitations & Caveats

1. **No ground truth.** Hotspots are candidates, not confirmed defects. External validation (e.g., Buffalo 311 reports, road maintenance records, Google Street View) is needed.
2. **Event-driven, not continuous.** Acceleration values are point-in-time snapshots at trigger, not continuous time series. Cannot perform signal processing (FFT, wavelet) for fine-grained defect classification.
3. **No vertical (z-axis) acceleration.** Only X (longitudinal) and Y (lateral) are available. Traditional pothole detection relies on vertical spikes.
4. **Fleet composition unknown.** Different vehicle types have different ABS/ESC trigger thresholds. Cannot normalize across vehicles.
5. **Weather not yet controlled for.** Some hotspots (especially ESC) may correlate with wet/icy conditions rather than road surface. Layer 2 will cross-reference wiper and temperature data.

---

