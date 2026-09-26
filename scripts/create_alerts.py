from pathlib import Path
import pandas as pd


# ==========================================
# 1. PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "impact_footprint.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "extreme_weather_alerts.csv"
)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("📡 Loading impact footprint data...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["valid_time"]
)

print(f"✅ Loaded {len(df)} records")


# ==========================================
# 3. PROTOTYPE ALERT LOGIC
# ==========================================

def generate_alert(row):

    peak = row["max_peak_mm_3h"]
    support = row["member_support_pct"]
    confidence = row["confidence_score"]

    # --------------------------------------
    # RED / SEVERE
    # --------------------------------------

    if (
        peak >= 50
        and support >= 80
        and confidence >= 0.80
    ):
        return "SEVERE"

    # --------------------------------------
    # ORANGE / HIGH
    # --------------------------------------

    elif (
        peak >= 25
        and support >= 60
        and confidence >= 0.50
    ):
        return "HIGH"

    # --------------------------------------
    # YELLOW / WATCH
    # --------------------------------------

    elif (
        peak >= 10
        or support >= 40
        or confidence >= 0.50
    ):
        return "WATCH"

    # --------------------------------------
    # NORMAL
    # --------------------------------------

    return "LOW"


df["alert_level"] = df.apply(
    generate_alert,
    axis=1
)


# ==========================================
# 4. ALERT SCORE
# ==========================================
#
# This is a prototype decision score,
# NOT a calibrated probability.
# ==========================================

peak_component = (
    df["max_peak_mm_3h"]
    .clip(upper=50)
    / 50
)

support_component = (
    df["member_support_pct"]
    / 100
)

confidence_component = (
    df["confidence_score"]
)

df["alert_score"] = (
    0.40 * peak_component
    +
    0.35 * support_component
    +
    0.25 * confidence_component
)

df["alert_score"] = (
    df["alert_score"]
    .clip(0, 1)
)


# ==========================================
# 5. SORT IMPORTANT ALERTS FIRST
# ==========================================

alert_order = {
    "SEVERE": 0,
    "HIGH": 1,
    "WATCH": 2,
    "LOW": 3
}

df["alert_order"] = (
    df["alert_level"]
    .map(alert_order)
)

df = df.sort_values(
    [
        "valid_time",
        "alert_order",
        "alert_score"
    ],
    ascending=[
        True,
        True,
        False
    ]
)


# ==========================================
# 6. SELECT USEFUL COLUMNS
# ==========================================

output_columns = [
    "valid_time",
    "latitude",
    "longitude",
    "consensus_track_id",
    "member_count",
    "member_support_pct",
    "mean_peak_mm_3h",
    "max_peak_mm_3h",
    "confidence_score",
    "confidence_level",
    "footprint_radius_km",
    "impact_level",
    "alert_score",
    "alert_level"
]

alerts = df[output_columns]


# ==========================================
# 7. SAVE
# ==========================================

alerts.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# 8. SUMMARY
# ==========================================

print(
    "\n========== ALERT SUMMARY =========="
)

print(
    alerts["alert_level"]
    .value_counts()
)

print(
    "\nHighest alert score:"
)

print(
    f"{alerts['alert_score'].max():.3f}"
)


print(
    "\nHighest intensity:"
)

print(
    f"{alerts['max_peak_mm_3h'].max():.2f} mm/3h"
)


print(
    "\n💾 Saved:"
)

print(OUTPUT_FILE)


print(
    "\n✅ Alert engine completed!"
)