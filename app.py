from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ExtremeTrack",
    page_icon="🌩️",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED = PROJECT_ROOT / "data" / "processed"

ALERT_FILE = PROCESSED / "extreme_weather_alerts.csv"
MEMBER_FILE = PROCESSED / "extreme_objects_tracked.csv"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_alerts():
    return pd.read_csv(
        ALERT_FILE,
        parse_dates=["valid_time"]
    )


@st.cache_data
def load_members():
    return pd.read_csv(
        MEMBER_FILE,
        parse_dates=["valid_time"]
    )


alerts = load_alerts()
members = load_members()


# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <h1 style="margin-bottom:0;">
        🌩️ ExtremeTrack
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "### AI-Driven Extreme Weather Intelligence"
)

st.caption(
    "5-member GEFS ensemble • Object-based tracking • "
    "Consensus • Confidence • Impact footprint • Alerts"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🎛️ Forecast Controls")

alert_levels = [
    "WATCH",
    "HIGH",
    "SEVERE"
]

selected_levels = st.sidebar.multiselect(
    "Alert levels",
    alert_levels,
    default=alert_levels
)

min_support = st.sidebar.select_slider(
    "Minimum ensemble support",
    options=[20, 40, 60, 80, 100],
    value=20
)


times = sorted(
    pd.to_datetime(
        alerts["valid_time"].dropna().unique()
    )
)

# Select strongest consensus event automatically
severity_rank = {
    "WATCH": 1,
    "HIGH": 2,
    "SEVERE": 3
}

best_event = (
    alerts
    .assign(
        severity_rank=alerts["alert_level"].map(
            severity_rank
        )
    )
    .sort_values(
        [
            "member_support_pct",
            "severity_rank",
            "alert_score",
            "max_peak_mm_3h"
        ],
        ascending=False
    )
    .iloc[0]
)

default_time = pd.Timestamp(
    best_event["valid_time"]
)

default_index = next(
    i
    for i, t in enumerate(times)
    if pd.Timestamp(t) == default_time
)

selected_time = st.sidebar.selectbox(
    "Forecast time",
    times,
    index=default_index,
    format_func=lambda x: pd.Timestamp(x).strftime(
        "%d %b %Y  |  %H:%M UTC"
    ),
    key="forecast_time_v3"
)


view_mode = st.sidebar.radio(
    "Map view",
    [
        "Consensus",
        "Member Spread"
    ]
)


st.sidebar.divider()

st.sidebar.caption(
    "Ensemble support = fraction of the 5 prototype "
    "members associated with the spatial consensus cluster."
)


# =========================================================
# FILTER CURRENT TIME
# =========================================================

selected_timestamp = pd.Timestamp(selected_time)

current_alerts = alerts[
    (alerts["valid_time"] == selected_timestamp)
    &
    (alerts["alert_level"].isin(selected_levels))
    &
    (alerts["member_support_pct"] >= min_support)
].copy()


# =========================================================
# KPI VALUES
# =========================================================

if current_alerts.empty:

    max_peak = 0
    max_support = 0
    severe_count = 0
    object_count = 0
    max_confidence = 0

else:

    max_peak = current_alerts["max_peak_mm_3h"].max()
    max_support = current_alerts["member_support_pct"].max()

    severe_count = (
        current_alerts["alert_level"] == "SEVERE"
    ).sum()

    object_count = len(current_alerts)

    max_confidence = (
        current_alerts["confidence_score"].max()
    )


# =========================================================
# KPI ROW
# =========================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "Ensemble Members",
        "5"
    )

with c2:
    st.metric(
        "Active Objects",
        object_count
    )

with c3:
    st.metric(
        "Peak Rainfall",
        f"{max_peak:.1f} mm/3h"
    )

with c4:
    st.metric(
        "Max Support",
        f"{max_support:.0f}%"
    )

with c5:
    st.metric(
        "Severe Alerts",
        severe_count
    )


st.divider()


# =========================================================
# MAP HELPERS
# =========================================================

def circle_coords(
    lat,
    lon,
    radius_km,
    points=70
):

    angles = np.linspace(
        0,
        2 * np.pi,
        points
    )

    lat_delta = radius_km / 111.0

    cos_lat = np.cos(
        np.radians(lat)
    )

    if abs(cos_lat) < 0.1:
        cos_lat = 0.1

    lon_delta = (
        radius_km
        / (111.0 * cos_lat)
    )

    lats = (
        lat
        + lat_delta * np.sin(angles)
    )

    lons = (
        lon
        + lon_delta * np.cos(angles)
    )

    return lats, lons


# =========================================================
# CONSENSUS MAP
# =========================================================

def build_consensus_map():

    fig = go.Figure()

    # -----------------------------------------------------
    # Background geographic map
    # -----------------------------------------------------

    # Full tracked paths for current filtered alerts
    selected_track_ids = (
        current_alerts["consensus_track_id"]
        .dropna()
        .unique()
        if not current_alerts.empty
        else []
    )

    # -----------------------------------------------------
    # Track history
    # -----------------------------------------------------

    for track_id in selected_track_ids:

        history = alerts[
            alerts["consensus_track_id"] == track_id
        ].sort_values("valid_time")

        if history.empty:
            continue

        fig.add_trace(
            go.Scattergeo(
                lat=history["latitude"],
                lon=history["longitude"],
                mode="lines+markers",
                line=dict(
                    color="#2563EB",
                    width=3
                ),
                marker=dict(
                    size=5,
                    color="#2563EB"
                ),
                name=f"Track {int(track_id)}",
                showlegend=False,
                hoverinfo="skip"
            )
        )

    # -----------------------------------------------------
    # Footprints
    # -----------------------------------------------------

    for _, row in current_alerts.iterrows():

        lats, lons = circle_coords(
            row["latitude"],
            row["longitude"],
            row["footprint_radius_km"]
        )

        color = (
            "rgba(220,38,38,0.75)"
            if row["alert_level"] == "SEVERE"
            else "rgba(245,158,11,0.75)"
        )

        fig.add_trace(
            go.Scattergeo(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(
                    color=color,
                    width=1.5,
                    dash="dot"
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # -----------------------------------------------------
    # Current alert markers
    # -----------------------------------------------------

    marker_config = [
        ("WATCH", "#64748B", "circle", 8),
        ("HIGH", "#F59E0B", "circle", 12),
        ("SEVERE", "#DC2626", "diamond", 16)
    ]

    for level, color, symbol, size in marker_config:

        subset = current_alerts[
            current_alerts["alert_level"] == level
        ]

        if subset.empty:
            continue

        custom = np.stack(
            [
                subset["max_peak_mm_3h"],
                subset["member_support_pct"],
                subset["confidence_score"],
                subset["footprint_radius_km"],
                subset["consensus_track_id"]
            ],
            axis=-1
        )

        fig.add_trace(
            go.Scattergeo(
                lat=subset["latitude"],
                lon=subset["longitude"],
                mode="markers",
                marker=dict(
                    size=size,
                    color=color,
                    symbol=symbol,
                    line=dict(
                        width=1.5,
                        color="white"
                    )
                ),
                name=f"{level} Alert",
                customdata=custom,
                hovertemplate=(
                    f"<b>{level} ALERT</b><br>"
                    "Peak: %{customdata[0]:.1f} mm/3h<br>"
                    "Ensemble support: %{customdata[1]:.0f}%<br>"
                    "Confidence: %{customdata[2]:.3f}<br>"
                    "Footprint: %{customdata[3]:.0f} km<br>"
                    "Track: %{customdata[4]}"
                    "<extra></extra>"
                )
            )
        )

    # -----------------------------------------------------
    # Map layout
    # -----------------------------------------------------

    fig.update_layout(

        geo=dict(
            scope="asia",

            projection=dict(
                type="equirectangular"
            ),

            center=dict(
                lat=20,
                lon=82
            ),

            showland=True,
            landcolor="white",

            showocean=True,
            oceancolor="#EEF6FC",

            showlakes=True,
            lakecolor="#EEF6FC",

            showcountries=True,
            countrycolor="#9CA3AF",

            showcoastlines=True,
            coastlinecolor="#374151",

            showframe=False,

            lataxis=dict(
                range=[5, 37],
                showgrid=True
            ),

            lonaxis=dict(
                range=[65, 100],
                showgrid=True
            )
        ),

        height=680,

        margin=dict(
            l=0,
            r=0,
            t=20,
            b=20
        ),

        legend=dict(
            orientation="h",
            y=0.01,
            x=0.5,
            xanchor="center"
        )
    )

    return fig


# =========================================================
# MEMBER SPREAD MAP
# =========================================================

def build_member_map():

    fig = go.Figure()

    member_colors = {
        "c00": "#2563EB",
        "p01": "#F59E0B",
        "p02": "#10B981",
        "p03": "#8B5CF6",
        "p04": "#EF4444"
    }

    current_members = members[
        members["valid_time"] == selected_timestamp
    ].copy()

    for member in [
        "c00",
        "p01",
        "p02",
        "p03",
        "p04"
    ]:

        subset = current_members[
            current_members["member"] == member
        ]

        if subset.empty:
            continue

        fig.add_trace(
            go.Scattergeo(
                lat=subset["latitude"],
                lon=subset["longitude"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=member_colors[member],
                    symbol="circle",
                    line=dict(
                        width=1.5,
                        color="white"
                    )
                ),
                name=member,

                customdata=np.stack(
                    [
                        subset["peak_mm_3h"],
                        subset["track_id"]
                    ],
                    axis=-1
                ),

                hovertemplate=(
                    f"<b>{member}</b><br>"
                    "Peak: %{customdata[0]:.1f} mm/3h<br>"
                    "Track: %{customdata[1]}"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(

        geo=dict(
            scope="asia",

            projection=dict(
                type="equirectangular"
            ),

            center=dict(
                lat=20,
                lon=82
            ),

            showland=True,
            landcolor="white",

            showocean=True,
            oceancolor="#EEF6FC",

            showcountries=True,
            countrycolor="#9CA3AF",

            showcoastlines=True,
            coastlinecolor="#374151",

            lataxis=dict(
                range=[5, 37],
                showgrid=True
            ),

            lonaxis=dict(
                range=[65, 100],
                showgrid=True
            )
        ),

        height=680,

        margin=dict(
            l=0,
            r=0,
            t=20,
            b=20
        )
    )

    return fig


# =========================================================
# MAP SECTION
# =========================================================

if view_mode == "Consensus":

    st.subheader(
        "🗺️ Ensemble Consensus & Impact Footprint"
    )

    if current_alerts.empty:

        st.info(
            "No consensus alerts match the selected filters."
        )

    else:

        st.plotly_chart(
            build_consensus_map(),
            use_container_width=True
        )

else:

    st.subheader(
        "🧩 Member-wise Extreme Object Spread"
    )

    st.caption(
        "Each point represents an extreme object detected "
        "independently in one GEFS ensemble member."
    )

    st.plotly_chart(
        build_member_map(),
        use_container_width=True
    )


# =========================================================
# DETAILS
# =========================================================

left, right = st.columns(2)


# =========================================================
# CURRENT ALERT DETAILS
# =========================================================

with left:

    st.subheader("🚨 Current Alert Details")

    if current_alerts.empty:

        st.info("No active alert for this selection.")

    else:

        detail_columns = [
            "valid_time",
            "consensus_track_id",
            "latitude",
            "longitude",
            "max_peak_mm_3h",
            "member_support_pct",
            "confidence_score",
            "impact_level",
            "alert_level"
        ]

        details = current_alerts[
            detail_columns
        ].copy()

        details["valid_time"] = (
            details["valid_time"]
            .dt.strftime("%d %b %H:%M")
        )

        st.dataframe(
            details,
            hide_index=True,
            use_container_width=True
        )


# =========================================================
# CONFIDENCE SUMMARY
# =========================================================

with right:

    st.subheader("🎯 Confidence Summary")

    if current_alerts.empty:

        st.info("No confidence data.")

    else:

        confidence = (
            current_alerts["confidence_level"]
            .value_counts()
            .reindex(
                ["LOW", "MEDIUM", "HIGH"],
                fill_value=0
            )
        )

        st.dataframe(
            confidence.rename("Objects")
            .reset_index()
            .rename(
                columns={
                    "index": "Confidence Level"
                }
            ),
            hide_index=True,
            use_container_width=True
        )

        st.metric(
            "Maximum Confidence",
            f"{max_confidence:.3f}"
        )


# =========================================================
# FORECAST TIMELINE
# =========================================================

st.divider()

st.subheader("📈 Forecast Alert Timeline")

timeline = (
    alerts[
        alerts["alert_level"].isin(
            selected_levels
        )
    ]
    .groupby(
        ["valid_time", "alert_level"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

st.line_chart(timeline)


# =========================================================
# TRANSPARENCY
# =========================================================

st.divider()

st.caption(
    "Prototype note: ensemble support is the fraction of "
    "the 5 prototype GEFS members associated with a spatial "
    "consensus cluster. It is not a calibrated probability. "
    "Alert thresholds, footprint radii and confidence scoring "
    "are prototype heuristics and require hindcast/observation "
    "verification before operational use."
)

# =========================================================
# VERIFICATION PANEL
# =========================================================

st.divider()

st.subheader("🧪 Forecast Verification")

VERIFICATION_FILE = (
    PROCESSED
    / "probabilistic_verification.csv"
)

RELIABILITY_FILE = (
    PROCESSED
    / "reliability_diagram.png"
)

if VERIFICATION_FILE.exists():

    verification = pd.read_csv(
        VERIFICATION_FILE
    )

    # --------------------------------------
    # Read metrics
    # --------------------------------------

    metric_values = dict(
        zip(
            verification["metric"],
            verification["value"]
        )
    )

    v1, v2, v3, v4 = st.columns(4)

    with v1:
        st.metric(
            "Brier Score",
            f"{metric_values.get('Brier_Score', 0):.6f}"
        )

    with v2:
        st.metric(
            "Brier Skill Score",
            f"{metric_values.get('Brier_Skill_Score', 0):.4f}"
        )

    with v3:
        st.metric(
            "Observed Event Rate",
            f"{metric_values.get('Observed_Event_Rate', 0) * 100:.3f}%"
        )

    with v4:
        st.metric(
            "Evaluation Points",
            f"{int(metric_values.get('Total_Evaluation_Points', 0)):,}"
        )

    # --------------------------------------
    # Reliability diagram
    # --------------------------------------

    if RELIABILITY_FILE.exists():

        left, right = st.columns([1.3, 1])

        with left:

            st.image(
                RELIABILITY_FILE,
                caption="Reliability Diagram — 5-member GEFS",
                use_container_width=True
            )

        with right:

            st.markdown("### 📊 What this means")

            st.write(
                "The current prototype uses ensemble-member "
                "frequency as event probability."
            )

            st.write(
                "This single-case evaluation indicates that "
                "the raw ensemble probabilities are not yet "
                "calibrated and need multi-event hindcast "
                "calibration."
            )

            st.info(
                "Ensemble Support ≠ calibrated probability"
            )

else:

    st.warning(
        "Verification data not found."
    )