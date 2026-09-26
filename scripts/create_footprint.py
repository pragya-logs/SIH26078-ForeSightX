from pathlib import Path

import pandas as pd
import numpy as np


# ==========================================
# 1. PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "consensus_confidence.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "impact_footprint.csv"
)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("📡 Loading consensus confidence data...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["valid_time"]
)

print(f"✅ Loaded {len(df)} records")


# ==========================================
# 3. FOOTPRINT RADIUS
# ==========================================

def footprint_radius(row):

    support = row["member_support_pct"]
    peak = row["max_peak_mm_3h"]

    # Base radius
    radius = 75.0

    # More ensemble support → wider reliable footprint
    radius += support * 0.75

    # Stronger precipitation → wider impact zone
    if peak >= 50:
        radius += 100

    elif peak >= 25:
        radius += 50

    return radius


df["footprint_radius_km"] = (
    df.apply(
        footprint_radius,
        axis=1
    )
)


# ==========================================
# 4. APPROXIMATE BOUNDING BOX
# ==========================================

# 1 degree latitude ≈ 111 km

df["lat_delta"] = (
    df["footprint_radius_km"] / 111.0
)

# Longitude conversion depends on latitude
df["lon_delta"] = (
    df["footprint_radius_km"]
    /
    (
        111.0
        * np.cos(
            np.radians(
                df["latitude"]
            )
        )
    )
)

# ==========================================
# 5. FOOTPRINT BOUNDARIES
# ==========================================

df["min_lat"] = (
    df["latitude"]
    - df["lat_delta"]
)

df["max_lat"] = (
    df["latitude"]
    + df["lat_delta"]
)

df["min_lon"] = (
    df["longitude"]
    - df["lon_delta"]
)

df["max_lon"] = (
    df["longitude"]
    + df["lon_delta"]
)


# ==========================================
# 6. IMPACT LEVEL
# ==========================================

def impact_level(row):

    if (
        row["member_support_pct"] >= 80
        and row["max_peak_mm_3h"] >= 50
    ):
        return "HIGH"

    elif (
        row["member_support_pct"] >= 60
        or row["max_peak_mm_3h"] >= 25
    ):
        return "MEDIUM"

    else:
        return "LOW"


df["impact_level"] = (
    df.apply(
        impact_level,
        axis=1
    )
)


# ==========================================
# 7. SAVE
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# 8. SUMMARY
# ==========================================

print(
    "\n========== IMPACT FOOTPRINT SUMMARY =========="
)

print(
    df["impact_level"]
    .value_counts()
)

print(
    f"\nAverage footprint radius: "
    f"{df['footprint_radius_km'].mean():.1f} km"
)

print(
    f"Maximum footprint radius: "
    f"{df['footprint_radius_km'].max():.1f} km"
)

print(
    "\n💾 Saved:"
)

print(OUTPUT_FILE)

print(
    "\n✅ Geographic impact footprint completed!"
)