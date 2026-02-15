#!/usr/bin/env python3
"""
全量数据探查：检查 Task 4 相关字段的覆盖率
逐文件处理，内存友好
结果输出到 diagnosis_outputs/

Usage:
  python3 /home/qiqingh/Desktop/traffic_data_analysis/diagnosis/explore_full.py
"""

import os
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
import time

DATA_DIR = "/home/qiqingh/Desktop/top_article/driving-events-csv"
OUTPUT_DIR = "/home/qiqingh/Desktop/traffic_data_analysis/diagnosis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"explore_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# 计数器
total_rows = 0
total_files = 0
skipped_files = []
field_counts = Counter()
event_counts = Counter()
accel_by_event = defaultdict(lambda: {"total": 0, "has_accelX": 0, "has_accelY": 0})

SAFETY_EVENTS = {
    "ACCELERATION_CHANGE", "ANTI_LOCK_BRAKING_SYSTEM_STATE_CHANGE",
    "ELECTRONIC_STABILITY_STATE_CHANGE", "AUTONOMOUS_EMERGENCY_BRAKING_CHANGE",
    "SPEED_THRESHOLD_CHANGE"
}
safety_event_count = 0
safety_with_accel = 0
safety_with_speed = 0

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
print(f"Found {len(files)} CSV files")
print(f"Output will be saved to: {OUTPUT_FILE}")
print("Scanning...\n")

start = time.time()

for i, fname in enumerate(files):
    fpath = os.path.join(DATA_DIR, fname)
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, escapechar="\\")
            try:
                header = next(reader)
            except StopIteration:
                skipped_files.append((fname, "empty file"))
                continue

            # 验证表头是否符合预期
            if len(header) != 7 or header[0] != "dataPointId":
                skipped_files.append((fname, f"unexpected header: {header[:3]}..."))
                continue

            for row in reader:
                if len(row) != len(header):
                    continue
                total_rows += 1

                try:
                    metrics = json.loads(row[4]) if row[4] else {}
                except:
                    metrics = {}
                try:
                    event = json.loads(row[6]) if row[6] else {}
                except:
                    event = {}

                speed = metrics.get("speed")
                accelX = metrics.get("accelerationX")
                accelY = metrics.get("accelerationY")
                fuel = metrics.get("fuelConsumption")
                env = metrics.get("environment", {})
                temp = env.get("exteriorTemperature") if isinstance(env, dict) else None

                if speed is not None: field_counts["speed"] += 1
                if accelX is not None: field_counts["accelerationX"] += 1
                if accelY is not None: field_counts["accelerationY"] += 1
                if fuel is not None: field_counts["fuelConsumption"] += 1
                if temp is not None: field_counts["exteriorTemp"] += 1

                etype = event.get("eventType", "UNKNOWN")
                event_counts[etype] += 1

                accel_by_event[etype]["total"] += 1
                if accelX is not None: accel_by_event[etype]["has_accelX"] += 1
                if accelY is not None: accel_by_event[etype]["has_accelY"] += 1

                if etype in SAFETY_EVENTS:
                    safety_event_count += 1
                    if accelX is not None or accelY is not None:
                        safety_with_accel += 1
                    if speed is not None:
                        safety_with_speed += 1

    except Exception as e:
        skipped_files.append((fname, str(e)))
        continue

    total_files += 1
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        rate = (i + 1) / elapsed
        eta = (len(files) - i - 1) / rate
        print(f"  [{i+1}/{len(files)}] {total_rows:,} rows so far "
              f"| {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

elapsed = time.time() - start

# ========== 写入报告 ==========
lines = []
def w(text=""):
    lines.append(text)

w("=" * 70)
w("FULL DATASET EXPLORATION REPORT")
w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w(f"Data dir:  {DATA_DIR}")
w("=" * 70)

w(f"\nFiles processed: {total_files}")
w(f"Files skipped:   {len(skipped_files)}")
w(f"Total rows: {total_rows:,}")
w(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

if skipped_files:
    w(f"\n--- Skipped Files ---")
    for fname, reason in skipped_files:
        w(f"  {fname}: {reason}")

w(f"\n--- Field Coverage (Task 4 relevant) ---")
for field in ["speed", "accelerationX", "accelerationY", "fuelConsumption", "exteriorTemp"]:
    cnt = field_counts[field]
    pct = cnt / total_rows * 100 if total_rows else 0
    w(f"  {field:20s}: {cnt:>12,} ({pct:5.1f}%)")

w(f"\n--- Event Type Distribution ---")
for etype, cnt in event_counts.most_common():
    pct = cnt / total_rows * 100
    w(f"  {etype:50s}: {cnt:>10,} ({pct:5.1f}%)")

w(f"\n--- Acceleration Coverage BY Event Type ---")
w(f"  {'Event Type':50s} {'Total':>10s} {'has_X':>10s} {'%X':>7s} {'has_Y':>10s} {'%Y':>7s}")
for etype, stats in sorted(accel_by_event.items(), key=lambda x: -x[1]["total"]):
    t = stats["total"]
    hx = stats["has_accelX"]
    hy = stats["has_accelY"]
    px = hx / t * 100 if t else 0
    py = hy / t * 100 if t else 0
    w(f"  {etype:50s} {t:>10,} {hx:>10,} {px:>6.1f}% {hy:>10,} {py:>6.1f}%")

w(f"\n--- Task 4 Safety Events Summary ---")
w(f"  Safety events total:       {safety_event_count:,}")
if safety_event_count:
    w(f"  With acceleration (X or Y): {safety_with_accel:,} "
      f"({safety_with_accel/safety_event_count*100:.1f}%)")
    w(f"  With speed:                 {safety_with_speed:,} "
      f"({safety_with_speed/safety_event_count*100:.1f}%)")

w(f"\n--- Quick Feasibility Assessment ---")
accel_pct = field_counts["accelerationX"] / total_rows * 100 if total_rows else 0
if accel_pct > 10:
    w(f"  Acceleration coverage {accel_pct:.1f}% — SUFFICIENT for grid-based analysis")
elif accel_pct > 2:
    w(f"  Acceleration coverage {accel_pct:.1f}% — MARGINAL, need to focus on event-based approach")
else:
    w(f"  Acceleration coverage {accel_pct:.1f}% — LOW, must rely purely on event counts + speed")

# 写入文件
report_text = "\n".join(lines)
with open(OUTPUT_FILE, "w") as f:
    f.write(report_text)

print(f"\nDone! Report saved to:\n  {OUTPUT_FILE}")
if skipped_files:
    print(f"  ({len(skipped_files)} files skipped, see report for details)")