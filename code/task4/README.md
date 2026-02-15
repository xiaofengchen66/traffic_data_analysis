# Task 4: Road Surface Condition Monitoring via Vehicle Telematics

> Last updated: 2026-02-15  
> Status: Layers 1–3 complete  
> Location: Buffalo, NY metropolitan area  
> Data: ~101.6 million event records, 744 files, 31 days (Sep 27 – Oct 28, 2022)

---

## Objective

Identify candidate road surface defects (potholes, damaged pavement, drainage issues) by analyzing spatial patterns in vehicle safety-system activations. The core premise: if multiple vehicles trigger ABS, ESC, or AEB at the same location across different days, the cause is almost certainly a fixed road defect rather than driver behavior or chance.

---

## Three-Layer Analysis

### Layer 1 — Safety-System Hotspot Detection

**Goal**: Extract rare safety-system events and cluster them spatially to find persistent problem locations.

**Method**: Extracted 5,660 ABS/ESC/AEB events from the full dataset. Applied DBSCAN (50m radius, min 4 events) to identify spatial clusters. Filtered for persistence (events on 3+ different calendar dates).

**Result**: **431 spatial clusters** identified, of which **181 are persistent hotspots**. The top hotspot (Cluster #0, ZIP 14052) contains 202 events — including 30% of all ESC activations in the entire dataset — concentrated within a 50m radius across 17 different days. This is an extremely strong road-defect signal.

---

### Layer 2 — Hard Brake Cross-Validation

**Goal**: Validate Layer 1 hotspots using an independent data source — hard braking events.

**Method**: Scanned 7.1 million hard-brake events. For each Layer 1 cluster, counted how many hard brakes occurred within 200m. Computed a composite confidence score combining safety-event count (40%), hard-brake corroboration (35%), and temporal persistence (25%). Also built an independent brake heatmap on a ~150m grid to discover new problem areas.

**Result**: **99.8% of Layer 1 hotspots received hard-brake corroboration** (430/431), confirming the spatial clustering is reliable. The analysis also revealed 20 new problem areas with high brake activity but no ABS/ESC triggers — likely traffic-design issues (signal timing, speed transitions) rather than road-surface defects.

---

### Layer 3 — Weather Correction

**Goal**: Determine whether hotspots are caused by fixed road defects or weather conditions (rain, cold).

**Method**: Built daily weather profiles from 694K wiper events and 31.6M temperature readings. Classified each day as dry or rainy using a data-driven threshold (median daily wiper activation count). Computed a "weather independence" score per cluster by comparing each hotspot's rainy-event fraction to the global baseline (91.4% of all safety events occur on rainy days).

**Result**: **All 431 hotspots are weather-independent (WI ≥ 0.8).** Rain increases ABS/ESC activation by ~10.6× globally, but no hotspot is disproportionately rain-dependent compared to the average. The spatial distribution of hotspots is entirely attributable to fixed road characteristics. Notably, the top hotspot (Cluster #0) has a lower-than-average rainy fraction (76% vs 91.4%), meaning it triggers even more in dry conditions — the strongest possible evidence of a fixed defect.

---

## Summary of Top 5 Hotspots

| Rank | Cluster | ZIP | Events | Hard Brakes | WI | Final Score | Likely Cause |
|------|---------|-----|--------|-------------|-----|-------------|-------------|
| 1 | #0 | 14052 | 202 (68 ABS, 134 ESC) | 221 | 1.00 | 0.616 | Road surface/geometry defect — extreme ESC concentration |
| 2 | #60 | 14261 | 32 | 13,584 | 0.97 | 0.488 | High-traffic dangerous segment |
| 3 | #116 | 14227 | 26 | 12,071 | 0.99 | 0.461 | High-traffic dangerous segment |
| 4 | #254 | 14086 | 66 | 5,366 | 1.00 | 0.438 | Road surface defect (Lancaster corridor) |
| 5 | #26 | 14225 | 72 | 1,987 | 0.97 | 0.421 | Road surface defect |

---

## Limitations

- **No ground truth.** Hotspots are candidates requiring external validation (311 reports, road surveys, Street View).
- **No vertical acceleration.** Only X/Y axes available; traditional pothole detection relies on Z-axis.
- **Single season.** Fall data only — winter ice/frost effects cannot be assessed.
- **Fleet composition unknown.** Different vehicle types have different ABS/ESC trigger thresholds.

---

## Next Steps

- Compile results into a report for supervisor review
- External validation of top hotspots (Google Maps, Buffalo 311 data)
- Extend to winter dataset if available for cold-weather analysis
- Integrate into a city-wide Road Quality Index