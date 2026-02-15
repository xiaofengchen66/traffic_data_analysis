# Task 4 — Layer 2: Hard Brake Cross-Validation & Independent Heatmap

> Last updated: 2026-02-15  
> Status: Complete  
> Depends on: Layer 1 v2 results (`hotspot_clusters.csv`)

---

## Overview

This layer serves two purposes:

1. **Cross-validate Layer 1 hotspots** by counting how many hard-brake events occur within 200m of each ABS/ESC/AEB cluster. If a location triggers both safety systems and frequent hard braking across multiple days, confidence in a fixed road defect increases substantially.

2. **Build an independent brake heatmap** by aggregating all hard-brake events on a geohash7 grid (~150m cells) to discover problem areas that Layer 1 may have missed — locations with anomalous braking but no safety-system activation.

---

## Data Source

- **Input**: Same 744 hourly CSV files (~101.6 million rows)
- **Target events**: ACCELERATION_CHANGE (15,710,807 total)
  - Hard brake: 7,136,698 (45.4%)
  - Hard acceleration: 8,574,109 (54.6%)
- **Layer 1 reference**: 431 cluster centers from v2 (50m DBSCAN)
- **Scan time**: 565 seconds (~9.4 minutes)

---

## Method

### Task A: Hotspot Validation

For each of the 7.1 million hard-brake events with valid coordinates:
1. Compute haversine distance to all 431 Layer 1 cluster centers (vectorized with NumPy)
2. If within 200m of any cluster, increment that cluster's brake corroboration count
3. Track unique dates, unique journeys, and speeds of matching brake events

Then compute a **composite confidence score** per cluster:

```
confidence = 0.40 × norm(safety_event_count)
           + 0.35 × norm(hard_brakes_within_200m)
           + 0.25 × norm(unique_dates)
```

All components normalized to [0, 1] before combining.

### Task B: Independent Brake Heatmap

For every ACCELERATION_CHANGE event (both brake and acceleration):
1. Truncate geohash to 7 characters (~150m × 150m grid cell)
2. Accumulate per-cell: total count, brake count, acceleration count, speed sum, unique dates, unique journeys
3. Filter cells with fewer than 3 events
4. Identify "new problem areas": top-100 brake cells that are **not** within 200m of any Layer 1 cluster

---

## Key Findings

### Validation Result

**99.8% of Layer 1 hotspots (430 out of 431) received hard-brake corroboration.** This confirms that Layer 1's spatial clustering of safety-system events is highly reliable — almost every ABS/ESC/AEB hotspot is independently supported by hard-brake evidence.

### Top 10 Hotspots by Confidence Score

| Rank | Cluster | Safety Events | Hard Brakes | Brake Days | Brake Trips | Score | ZIP   | Coordinates |
|------|---------|--------------|-------------|------------|-------------|-------|-------|-------------|
| 1    | #0      | 202          | 221         | 24         | 83          | 0.616 | 14052 | 42.767, -78.645 |
| 2    | #60     | 32           | 13,584      | 32         | 6,279       | 0.501 | 14261 | 42.991, -78.790 |
| 3    | #116    | 26           | 12,071      | 33         | 5,928       | 0.466 | 14227 | 42.884, -78.766 |
| 4    | #254    | 66           | 5,366       | 33         | 2,613       | 0.438 | 14086 | 42.924, -78.623 |
| 5    | #26     | 72           | 1,987       | 32         | 1,146       | 0.435 | 14225 | 42.946, -78.725 |
| 6    | #410    | 9            | 14,576      | 32         | 6,530       | 0.373 | 14221 | 42.991, -78.789 |
| 7    | #95     | 6            | 13,847      | 34         | 4,647       | 0.363 | 14051 | 43.053, -78.728 |
| 8    | #34     | 22           | 8,744       | 33         | 4,407       | 0.352 | 14225 | 42.927, -78.753 |
| 9    | #86     | 36           | 4,057       | 32         | 2,102       | 0.346 | 14204 | 42.884, -78.868 |
| 10   | #184    | 16           | 10,023      | 33         | 4,399       | 0.344 | 14221 | 42.978, -78.725 |

### Diagnostic Interpretation Framework

The relationship between safety-event count and hard-brake count reveals the **nature** of each hotspot:

| Pattern | Safety Events | Hard Brakes | Likely Cause | Priority |
|---------|--------------|-------------|--------------|----------|
| **Road surface defect** | HIGH | LOW | Vehicles lose traction/stability without heavy braking (e.g., Cluster #0: 202 safety / 221 brakes) | **Highest** — direct safety hazard |
| **Dangerous road segment** | HIGH | HIGH | Compound problem: road surface + traffic design (e.g., #60: 32 safety / 13,584 brakes) | High — needs further investigation |
| **Traffic/design issue** | LOW | HIGH | Signal timing, speed transitions, visibility (e.g., new problem areas with 20K+ brakes but zero ABS/ESC) | Medium — may not be road surface |

**Cluster #0 remains the strongest road-defect candidate.** Its signature — extremely high safety events (202, including 134 ESC) but only 221 hard brakes — is uniquely diagnostic. This location does not simply cause braking; it causes **loss of vehicle stability**, which is the hallmark of a road surface or geometry defect.

### Independent Brake Heatmap

- **11,367 geohash7 cells** with 3+ events provide city-wide coverage
- Top cells have 19,000–24,000 hard brakes each across 32–35 days
- **20 "new problem areas"** identified in the top-100 brake cells with no nearby ABS/ESC cluster

Top 5 new problem areas:

| Geohash7 | Hard Brakes | Days | Avg Speed | Coordinates |
|----------|-------------|------|-----------|-------------|
| dr85cn   | 23,568      | 34   | 25 km/h   | 42.881, -78.697 |
| dpxuw2   | 23,014      | 35   | 24 km/h   | 42.981, -78.822 |
| dr8h91   | 22,442      | 34   | 27 km/h   | 42.988, -78.698 |
| dpxgzy   | 20,489      | 35   | 25 km/h   | 42.883, -78.755 |
| dpxgjw   | 19,224      | 35   | 28 km/h   | 42.750, -78.854 |

These locations warrant further investigation. High brake counts without ABS/ESC triggers may indicate traffic design issues (signal timing, speed transitions) or lower-severity surface problems that cause braking but not instability.

---

## Output Files

### `task4_outputs/layer2/`

| File | Description |
|------|-------------|
| `clusters_with_brake_validation.csv` | All 431 Layer 1 clusters enriched with brake counts + confidence scores |
| `geohash_brake_heatmap.csv` | 11,367 geohash7 cells with brake/acceleration counts and stats |
| `layer2_report.txt` | Full text report |
| `fig1_brake_validation.png` | Scatter plot (safety events vs brakes) + geographic confidence map |
| `fig2_brake_heatmap.png` | Independent brake heatmap: top 500 cells + persistent cells (10+ days) |
| `fig3_top20_comparison.png` | Horizontal bar chart comparing safety events and brake corroboration for top 20 |

---

## Code

| File | Location | Description |
|------|----------|-------------|
| `task4_layer2_hard_brake.py` | `code/task4/` | Full Layer 2 pipeline |

### Dependencies
Same as Layer 1 (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`). No additional packages required.

### Usage
```bash
python3 code/task4/task4_layer2_hard_brake.py
```

Requires Layer 1 v2 output (`hotspot_clusters.csv`) as input. Scans all 744 CSVs (~10 min on 31GB RAM machine).

---

## Limitations

1. **200m match radius is generous.** A hard-brake event 200m from a cluster center may not be related to the same road defect. Tighter radii (50m, 100m) could be tested for sensitivity analysis.
2. **No intersection filtering.** High-brake-count locations near traffic signals are expected and do not necessarily indicate road defects. Future work should cross-reference with intersection/signal locations.
3. **Brake vs. acceleration classification relies on metadata strings.** Parsing `eventMetadata` for "BRAKE" / "DECEL" keywords is robust for this dataset (0 unclassified events), but may not generalize.
4. **Confidence score weights (0.4 / 0.35 / 0.25) are heuristic.** No ground truth is available for optimization. The ranking should be treated as a prioritization guide, not a definitive assessment.
5. **Geohash7 cells are fixed grids** that may split a single problem location across cell boundaries. DBSCAN on raw brake events would be more precise but computationally expensive on 7M points.
