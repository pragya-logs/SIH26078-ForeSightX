from pathlib import Path
import xarray as xr
import numpy as np
from scipy.ndimage import label, find_objects

# ==========================================
# 1. Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gefs_india_2019010700_3h.nc"
)

# ==========================================
# 2. Settings
# ==========================================

THRESHOLD = 10.0

# Minimum grid cells required for an object
MIN_CELLS = 4

# ==========================================
# 3. Load precipitation
# ==========================================

print("📡 Loading precipitation...")

ds = xr.open_dataset(INPUT_FILE)
rain = ds["precip_3h"]

print(f"✅ Shape: {rain.shape}")

# ==========================================
# 4. Detect objects
# ==========================================

print("\n🌧️ Detecting extreme precipitation objects...")

for member in rain.member.values:

    print(f"\n========== MEMBER: {member} ==========")

    member_data = rain.sel(member=member)

    total_objects = 0

    # Process each forecast time
    for time in member_data.valid_time.values:

        field = member_data.sel(valid_time=time).values

        # Extreme threshold mask
        mask = field >= THRESHOLD

        # 8-neighbour connectivity
        structure = np.ones((3, 3), dtype=int)

        labeled, num_objects = label(
            mask,
            structure=structure
        )

        objects = []

        for object_id, slc in enumerate(
            find_objects(labeled),
            start=1
        ):

            if slc is None:
                continue

            object_mask = labeled[slc] == object_id
            cell_count = int(object_mask.sum())

            if cell_count < MIN_CELLS:
                continue

            # Grid indices
            lat_indices = np.arange(
                slc[0].start,
                slc[0].stop
            )

            lon_indices = np.arange(
                slc[1].start,
                slc[1].stop
            )

            # Extract actual coordinates
            lat_values = member_data.latitude.values[
                lat_indices
            ]

            lon_values = member_data.longitude.values[
                lon_indices
            ]

            # Object centroid
            centroid_lat = float(
                lat_values.mean()
            )

            centroid_lon = float(
                lon_values.mean()
            )

            # Peak precipitation
            object_values = field[slc][object_mask]

            peak = float(
                np.max(object_values)
            )

            objects.append({
                "object_id": object_id,
                "cells": cell_count,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "peak_mm_3h": peak
            })

        if objects:

            total_objects += len(objects)

            print(
                f"{str(time)[:16]} | "
                f"Objects: {len(objects)} | "
                f"Strongest: "
                f"{max(o['peak_mm_3h'] for o in objects):.2f} mm/3h"
            )

    print(
        f"\n✅ {member}: "
        f"{total_objects} objects detected"
    )

print("\n✅ Extreme object detection completed!")