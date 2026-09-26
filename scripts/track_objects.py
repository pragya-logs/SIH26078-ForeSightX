from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd
from scipy.ndimage import label, find_objects


# ==========================================
# 1. SETTINGS
# ==========================================

THRESHOLD = 10.0          # mm / 3h
MIN_CELLS = 4             # minimum grid cells for an object
MAX_DISTANCE_KM = 350.0   # maximum movement between consecutive times


# ==========================================
# 2. PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gefs_india_2019010700_3h.nc"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACK_FILE = (
    OUTPUT_DIR
    / "extreme_objects_tracked.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "track_lifecycle_summary.csv"
)


# ==========================================
# 3. LOAD DATA
# ==========================================

print("📡 Loading precipitation...")

ds = xr.open_dataset(INPUT_FILE)

rain = ds["precip_3h"]

print(f"✅ Shape: {rain.shape}")


# ==========================================
# 4. HAVERSINE DISTANCE
# ==========================================

def haversine_km(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2
        * R
        * np.arcsin(np.sqrt(a))
    )


# ==========================================
# 5. DETECT EXTREME OBJECTS
# ==========================================

def detect_objects(
    field,
    latitudes,
    longitudes
):

    # Threshold mask
    mask = field >= THRESHOLD

    # 8-neighbour connectivity
    structure = np.ones(
        (3, 3),
        dtype=int
    )

    labeled, _ = label(
        mask,
        structure=structure
    )

    objects = []

    slices = find_objects(labeled)

    for object_id, slc in enumerate(
        slices,
        start=1
    ):

        if slc is None:
            continue

        object_mask = (
            labeled[slc] == object_id
        )

        cell_count = int(
            object_mask.sum()
        )

        # Ignore tiny objects
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

        # Actual coordinates
        lat_values = latitudes[
            lat_indices
        ]

        lon_values = longitudes[
            lon_indices
        ]

        # Centroid
        centroid_lat = float(
            lat_values.mean()
        )

        centroid_lon = float(
            lon_values.mean()
        )

        # Object precipitation values
        values = field[slc][
            object_mask
        ]

        peak = float(
            np.max(values)
        )

        mean_value = float(
            np.mean(values)
        )

        objects.append(
            {
                "local_id": object_id,
                "latitude": centroid_lat,
                "longitude": centroid_lon,
                "cells": cell_count,
                "peak_mm_3h": peak,
                "mean_mm_3h": mean_value,
            }
        )

    return objects


# ==========================================
# 6. TRACK OBJECTS
# ==========================================

all_records = []
all_summaries = []


for member in rain.member.values:

    print(
        f"\n========== MEMBER: {member} =========="
    )

    member_data = rain.sel(
        member=member
    )

    next_track_id = 1

    # Objects ONLY from the immediately
    # previous forecast time
    previous_objects = []

    # Track information
    tracks = {}

    # ======================================
    # LOOP THROUGH FORECAST TIMES
    # ======================================

    for time in member_data.valid_time.values:

        field = (
            member_data
            .sel(valid_time=time)
            .values
        )

        current_objects = detect_objects(
            field,
            member_data.latitude.values,
            member_data.longitude.values
        )

        # ==================================
        # If no objects at this time,
        # previous tracks die here.
        # ==================================

        if not current_objects:

            previous_objects = []

            print(
                f"{str(time)[:16]} | "
                f"Objects: 0 | "
                f"Active tracks: 0"
            )

            continue

        # ==================================
        # MATCH CURRENT OBJECTS TO
        # PREVIOUS TIME OBJECTS
        # ==================================

        candidate_matches = []

        for current_index, current in enumerate(
            current_objects
        ):

            for previous_index, previous in enumerate(
                previous_objects
            ):

                distance = haversine_km(
                    previous["latitude"],
                    previous["longitude"],
                    current["latitude"],
                    current["longitude"]
                )

                if distance <= MAX_DISTANCE_KM:

                    candidate_matches.append(
                        (
                            distance,
                            current_index,
                            previous_index
                        )
                    )

        # Closest matches first
        candidate_matches.sort(
            key=lambda x: x[0]
        )

        matched_current = set()
        matched_previous = set()

        current_to_previous = {}

        for (
            distance,
            current_index,
            previous_index
        ) in candidate_matches:

            if current_index in matched_current:
                continue

            if previous_index in matched_previous:
                continue

            current_to_previous[
                current_index
            ] = previous_index

            matched_current.add(
                current_index
            )

            matched_previous.add(
                previous_index
            )

        # ==================================
        # ASSIGN TRACK IDs
        # ==================================

        current_track_ids = []

        for current_index, current in enumerate(
            current_objects
        ):

            # ------------------------------
            # Existing track
            # ------------------------------

            if current_index in current_to_previous:

                previous_index = (
                    current_to_previous[
                        current_index
                    ]
                )

                track_id = (
                    previous_objects[
                        previous_index
                    ]["track_id"]
                )

                event_type = "continue"

            # ------------------------------
            # New track
            # ------------------------------

            else:

                track_id = next_track_id

                next_track_id += 1

                event_type = "birth"

                tracks[track_id] = {
                    "member": str(member),
                    "track_id": track_id,
                    "start_time": pd.Timestamp(time),
                    "end_time": pd.Timestamp(time),
                    "duration_steps": 1,
                    "peak_mm_3h": current[
                        "peak_mm_3h"
                    ],
                }

            # ==================================
            # UPDATE TRACK INFORMATION
            # ==================================

            if track_id not in tracks:

                tracks[track_id] = {
                    "member": str(member),
                    "track_id": track_id,
                    "start_time": pd.Timestamp(time),
                    "end_time": pd.Timestamp(time),
                    "duration_steps": 1,
                    "peak_mm_3h": current[
                        "peak_mm_3h"
                    ],
                }

            else:

                if event_type == "continue":

                    tracks[
                        track_id
                    ]["end_time"] = pd.Timestamp(
                        time
                    )

                    tracks[
                        track_id
                    ]["duration_steps"] += 1

                    tracks[
                        track_id
                    ]["peak_mm_3h"] = max(
                        tracks[
                            track_id
                        ]["peak_mm_3h"],
                        current[
                            "peak_mm_3h"
                        ]
                    )

            current_track_ids.append(
                track_id
            )

            # ==================================
            # SAVE OBJECT RECORD
            # ==================================

            all_records.append(
                {
                    "member": str(member),
                    "track_id": track_id,
                    "valid_time": pd.Timestamp(time),
                    "latitude": current[
                        "latitude"
                    ],
                    "longitude": current[
                        "longitude"
                    ],
                    "cells": current[
                        "cells"
                    ],
                    "peak_mm_3h": current[
                        "peak_mm_3h"
                    ],
                    "mean_mm_3h": current[
                        "mean_mm_3h"
                    ],
                    "event_type": event_type,
                }
            )

        # ==================================
        # CURRENT OBJECTS BECOME
        # PREVIOUS OBJECTS
        #
        # IMPORTANT:
        # Only next forecast time can match.
        # ==================================

        previous_objects = []

        for current_index, current in enumerate(
            current_objects
        ):

            previous_objects.append(
                {
                    "track_id": current_track_ids[
                        current_index
                    ],
                    "latitude": current[
                        "latitude"
                    ],
                    "longitude": current[
                        "longitude"
                    ],
                }
            )

        print(
            f"{str(time)[:16]} | "
            f"Objects: {len(current_objects)} | "
            f"Active tracks: "
            f"{len(previous_objects)}"
        )

    # ======================================
    # LIFECYCLE SUMMARY
    # ======================================

    for track in tracks.values():

        track[
            "duration_hours"
        ] = (
            track["duration_steps"] - 1
        ) * 3

        all_summaries.append(
            track
        )

    print(
        f"✅ {member}: "
        f"{len(tracks)} lifecycle tracks created"
    )


# ==========================================
# 7. CREATE DATAFRAMES
# ==========================================

tracks_df = pd.DataFrame(
    all_records
)

summary_df = pd.DataFrame(
    all_summaries
)


# ==========================================
# 8. SORT
# ==========================================

tracks_df = tracks_df.sort_values(
    [
        "member",
        "track_id",
        "valid_time"
    ]
)

summary_df = summary_df.sort_values(
    [
        "member",
        "track_id"
    ]
)


# ==========================================
# 9. SAVE TRACK DATA
# ==========================================

tracks_df.to_csv(
    TRACK_FILE,
    index=False
)


# ==========================================
# 10. SAVE LIFECYCLE SUMMARY
# ==========================================

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print(
    "\n========== TRACK SUMMARY =========="
)

print(
    f"Total tracked records: "
    f"{len(tracks_df)}"
)

print(
    f"Total unique tracks: "
    f"{len(summary_df)}"
)

print(
    f"Maximum peak precipitation: "
    f"{tracks_df['peak_mm_3h'].max():.2f} mm/3h"
)


print("\nTracks per member:")

print(
    summary_df
    .groupby("member")["track_id"]
    .count()
)


print(
    "\n========== LIFECYCLE EVENTS =========="
)

print(
    tracks_df[
        "event_type"
    ].value_counts()
)


print(
    "\n💾 Object tracks:"
)

print(TRACK_FILE)


print(
    "\n💾 Lifecycle summary:"
)

print(SUMMARY_FILE)


print(
    "\n✅ Lifecycle-aware object tracking completed!"
)