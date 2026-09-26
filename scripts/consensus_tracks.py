from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# SETTINGS
# ==========================================

TOTAL_MEMBERS = 5

# Maximum distance to consider detections
# part of the same consensus object
CONSENSUS_RADIUS_KM = 300.0

# Maximum movement of a consensus object
# between consecutive 3-hour times
TRACK_RADIUS_KM = 400.0


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "extreme_objects_tracked.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

CONSENSUS_FILE = (
    OUTPUT_DIR
    / "consensus_objects.csv"
)

LIFECYCLE_FILE = (
    OUTPUT_DIR
    / "consensus_lifecycle_summary.csv"
)


# ==========================================
# HAVERSINE DISTANCE
# ==========================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

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
# LOAD TRACKED OBJECTS
# ==========================================

print("📡 Loading tracked objects...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["valid_time"]
)

print(
    f"✅ Loaded {len(df)} tracked records"
)

print(
    f"✅ Members: "
    f"{sorted(df['member'].unique())}"
)


# ==========================================
# CONSENSUS OBJECT CREATION
# ==========================================

def create_consensus_objects(
    time_df
):

    rows = time_df.sort_values(
        "peak_mm_3h",
        ascending=False
    ).to_dict("records")

    used = set()

    consensus = []

    # ======================================
    # Start clusters from strongest objects
    # ======================================

    for i, seed in enumerate(rows):

        if i in used:
            continue

        cluster = [seed]

        used.add(i)

        current_lat = seed["latitude"]
        current_lon = seed["longitude"]

        members_in_cluster = {
            seed["member"]
        }

        # ----------------------------------
        # Add nearby objects from other
        # ensemble members
        # ----------------------------------

        changed = True

        while changed:

            changed = False

            for j, candidate in enumerate(rows):

                if j in used:
                    continue

                # One detection per member
                if candidate["member"] in members_in_cluster:
                    continue

                distance = haversine_km(
                    current_lat,
                    current_lon,
                    candidate["latitude"],
                    candidate["longitude"]
                )

                if distance <= CONSENSUS_RADIUS_KM:

                    cluster.append(
                        candidate
                    )

                    used.add(j)

                    members_in_cluster.add(
                        candidate["member"]
                    )

                    # Recalculate centroid
                    current_lat = np.mean(
                        [
                            x["latitude"]
                            for x in cluster
                        ]
                    )

                    current_lon = np.mean(
                        [
                            x["longitude"]
                            for x in cluster
                        ]
                    )

                    changed = True

        # ----------------------------------
        # Consensus statistics
        # ----------------------------------

        member_count = len(
            members_in_cluster
        )

        support_pct = (
            member_count
            / TOTAL_MEMBERS
            * 100
        )

        peaks = [
            x["peak_mm_3h"]
            for x in cluster
        ]

        consensus_peak = float(
            np.mean(peaks)
        )

        max_peak = float(
            np.max(peaks)
        )

        consensus_lat = float(
            np.mean(
                [
                    x["latitude"]
                    for x in cluster
                ]
            )
        )

        consensus_lon = float(
            np.mean(
                [
                    x["longitude"]
                    for x in cluster
                ]
            )
        )

        members_text = ",".join(
            sorted(members_in_cluster)
        )

        consensus.append(
            {
                "valid_time":
                    time_df["valid_time"].iloc[0],

                "latitude":
                    consensus_lat,

                "longitude":
                    consensus_lon,

                "member_count":
                    member_count,

                "member_support_pct":
                    support_pct,

                "members":
                    members_text,

                "mean_peak_mm_3h":
                    consensus_peak,

                "max_peak_mm_3h":
                    max_peak,

                "object_count":
                    len(cluster)
            }
        )

    return consensus


# ==========================================
# CREATE ALL CONSENSUS OBJECTS
# ==========================================

print(
    "\n🌐 Creating ensemble consensus objects..."
)

all_consensus = []

for time, time_df in df.groupby(
    "valid_time"
):

    objects = create_consensus_objects(
        time_df
    )

    all_consensus.extend(
        objects
    )

    if objects:

        max_support = max(
            x["member_support_pct"]
            for x in objects
        )

        print(
            f"{time} | "
            f"Consensus objects: "
            f"{len(objects)} | "
            f"Max member support: "
            f"{max_support:.0f}%"
        )


consensus_df = pd.DataFrame(
    all_consensus
)


# ==========================================
# CONSENSUS TRACKING
# ==========================================

print(
    "\n🚀 Tracking consensus objects..."
)

consensus_df = consensus_df.sort_values(
    "valid_time"
).reset_index(drop=True)

next_track_id = 1

previous_objects = []

track_records = []

track_summaries = []

active_tracks = {}


for time, time_df in consensus_df.groupby(
    "valid_time"
):

    current_objects = (
        time_df
        .to_dict("records")
    )

    # ======================================
    # Check time gap
    # ======================================

    if previous_objects:

        previous_time = pd.Timestamp(
            previous_objects[0][
                "valid_time"
            ]
        )

        current_time = pd.Timestamp(
            time
        )

        hours_gap = (
            current_time
            - previous_time
        ).total_seconds() / 3600

        # Only connect consecutive 3-hour steps
        if hours_gap > 3.1:

            previous_objects = []

    # ======================================
    # Candidate matches
    # ======================================

    candidates = []

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

            if distance <= TRACK_RADIUS_KM:

                candidates.append(
                    (
                        distance,
                        current_index,
                        previous_index
                    )
                )

    candidates.sort(
        key=lambda x: x[0]
    )

    matched_current = set()
    matched_previous = set()

    current_to_previous = {}

    for (
        distance,
        current_index,
        previous_index
    ) in candidates:

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

    # ======================================
    # Assign consensus track IDs
    # ======================================

    new_previous_objects = []

    for current_index, current in enumerate(
        current_objects
    ):

        if current_index in current_to_previous:

            previous_index = (
                current_to_previous[
                    current_index
                ]
            )

            track_id = (
                previous_objects[
                    previous_index
                ]["consensus_track_id"]
            )

            event_type = "continue"

        else:

            track_id = next_track_id

            next_track_id += 1

            event_type = "birth"

            active_tracks[
                track_id
            ] = {
                "start_time":
                    pd.Timestamp(time),

                "end_time":
                    pd.Timestamp(time),

                "duration_steps":
                    1,

                "peak_mm_3h":
                    current[
                        "max_peak_mm_3h"
                    ],

                "max_support_pct":
                    current[
                        "member_support_pct"
                    ]
            }

        # ==================================
        # Update summary
        # ==================================

        if event_type == "continue":

            active_tracks[
                track_id
            ]["end_time"] = pd.Timestamp(
                time
            )

            active_tracks[
                track_id
            ]["duration_steps"] += 1

            active_tracks[
                track_id
            ]["peak_mm_3h"] = max(
                active_tracks[
                    track_id
                ]["peak_mm_3h"],

                current[
                    "max_peak_mm_3h"
                ]
            )

            active_tracks[
                track_id
            ]["max_support_pct"] = max(
                active_tracks[
                    track_id
                ]["max_support_pct"],

                current[
                    "member_support_pct"
                ]
            )

        # ==================================
        # Save record
        # ==================================

        record = current.copy()

        record[
            "consensus_track_id"
        ] = track_id

        record[
            "event_type"
        ] = event_type

        track_records.append(
            record
        )

        new_previous_objects.append(
            {
                "consensus_track_id":
                    track_id,

                "valid_time":
                    time,

                "latitude":
                    current["latitude"],

                "longitude":
                    current["longitude"]
            }
        )

    previous_objects = (
        new_previous_objects
    )


# ==========================================
# BUILD LIFECYCLE SUMMARY
# ==========================================

for track_id, track in active_tracks.items():

    track_summaries.append(
        {
            "consensus_track_id":
                track_id,

            "start_time":
                track["start_time"],

            "end_time":
                track["end_time"],

            "duration_steps":
                track["duration_steps"],

            "duration_hours":
                (
                    track["duration_steps"]
                    - 1
                ) * 3,

            "peak_mm_3h":
                track["peak_mm_3h"],

            "max_member_support_pct":
                track[
                    "max_support_pct"
                ]
        }
    )


# ==========================================
# SAVE CONSENSUS OBJECTS
# ==========================================

track_df = pd.DataFrame(
    track_records
)

track_df = track_df.sort_values(
    [
        "consensus_track_id",
        "valid_time"
    ]
)

track_df.to_csv(
    CONSENSUS_FILE,
    index=False
)


# ==========================================
# SAVE LIFECYCLE
# ==========================================

lifecycle_df = pd.DataFrame(
    track_summaries
)

lifecycle_df = lifecycle_df.sort_values(
    "consensus_track_id"
)

lifecycle_df.to_csv(
    LIFECYCLE_FILE,
    index=False
)


# ==========================================
# FINAL SUMMARY
# ==========================================

print(
    "\n========== CONSENSUS SUMMARY =========="
)

print(
    f"Total consensus records: "
    f"{len(track_df)}"
)

print(
    f"Total consensus tracks: "
    f"{len(lifecycle_df)}"
)

print(
    f"Maximum member support: "
    f"{track_df['member_support_pct'].max():.0f}%"
)

print(
    f"Maximum consensus peak: "
    f"{track_df['max_peak_mm_3h'].max():.2f} mm/3h"
)

print(
    "\nMember support distribution:"
)

print(
    track_df[
        "member_support_pct"
    ].value_counts()
    .sort_index()
)

print(
    "\n💾 Consensus objects:"
)

print(CONSENSUS_FILE)

print(
    "\n💾 Consensus lifecycle:"
)

print(LIFECYCLE_FILE)

print(
    "\n✅ Ensemble consensus tracking completed!"
)