from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ==========================================
# 1. PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "extreme_weather_alerts.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "extremetrack_visualization.html"
)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("📡 Loading alert data...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["valid_time"]
)

print(f"✅ Loaded {len(df)} alert records")


# ==========================================
# 3. HIGH + SEVERE ONLY
# ==========================================

plot_df = df[
    df["alert_level"].isin(["HIGH", "SEVERE"])
].copy()

print(
    f"✅ HIGH/SEVERE records: {len(plot_df)}"
)


# ==========================================
# 4. COLORS
# ==========================================

BLUE = "#2563EB"
ORANGE = "#F59E0B"
RED = "#DC2626"
GREY = "#6B7280"


# ==========================================
# 5. CREATE FIGURE
# ==========================================

fig = go.Figure()


# ==========================================
# 6. FOOTPRINT CIRCLE
# ==========================================

def create_circle(
    lat,
    lon,
    radius_km,
    points=80
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


# ==========================================
# 7. ADD FOOTPRINT OUTLINES
# ==========================================

for _, row in plot_df.iterrows():

    lats, lons = create_circle(
        row["latitude"],
        row["longitude"],
        row["footprint_radius_km"]
    )

    line_color = (
        RED
        if row["alert_level"] == "SEVERE"
        else ORANGE
    )

    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="lines",
            line=dict(
                color=line_color,
                width=1.2,
                dash="dot"
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )


# ==========================================
# 8. TRACK LINES
# ==========================================

track_ids = (
    plot_df["consensus_track_id"]
    .dropna()
    .unique()
)

print(
    f"🚨 Plotting {len(track_ids)} tracks..."
)


for track_id in track_ids:

    track = plot_df[
        plot_df["consensus_track_id"] == track_id
    ].sort_values("valid_time")

    if track.empty:
        continue

    fig.add_trace(
        go.Scattergeo(
            lat=track["latitude"],
            lon=track["longitude"],
            mode="lines+markers",
            line=dict(
                color=BLUE,
                width=3
            ),
            marker=dict(
                size=7,
                color=BLUE,
                line=dict(
                    width=1,
                    color="white"
                )
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )


# ==========================================
# 9. HIGH ALERTS
# ==========================================

high = plot_df[
    plot_df["alert_level"] == "HIGH"
]

if not high.empty:

    fig.add_trace(
        go.Scattergeo(
            lat=high["latitude"],
            lon=high["longitude"],
            mode="markers",
            marker=dict(
                size=11,
                color=ORANGE,
                symbol="circle",
                line=dict(
                    width=1.5,
                    color="white"
                )
            ),
            name="HIGH Alert",

            customdata=np.stack(
                [
                    high["valid_time"].dt.strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    high["max_peak_mm_3h"],
                    high["member_support_pct"],
                    high["confidence_score"],
                    high["footprint_radius_km"]
                ],
                axis=-1
            ),

            hovertemplate=(
                "<b>HIGH ALERT</b><br>"
                "Time: %{customdata[0]}<br>"
                "Peak: %{customdata[1]:.1f} mm/3h<br>"
                "Member support: %{customdata[2]:.0f}%<br>"
                "Confidence: %{customdata[3]:.3f}<br>"
                "Footprint: %{customdata[4]:.0f} km"
                "<extra></extra>"
            )
        )
    )


# ==========================================
# 10. SEVERE ALERTS
# ==========================================

severe = plot_df[
    plot_df["alert_level"] == "SEVERE"
]

if not severe.empty:

    fig.add_trace(
        go.Scattergeo(
            lat=severe["latitude"],
            lon=severe["longitude"],
            mode="markers",
            marker=dict(
                size=15,
                color=RED,
                symbol="diamond",
                line=dict(
                    width=2,
                    color="white"
                )
            ),
            name="SEVERE Alert",

            customdata=np.stack(
                [
                    severe["valid_time"].dt.strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    severe["max_peak_mm_3h"],
                    severe["member_support_pct"],
                    severe["confidence_score"],
                    severe["footprint_radius_km"]
                ],
                axis=-1
            ),

            hovertemplate=(
                "<b>SEVERE ALERT</b><br>"
                "Time: %{customdata[0]}<br>"
                "Peak: %{customdata[1]:.1f} mm/3h<br>"
                "Member support: %{customdata[2]:.0f}%<br>"
                "Confidence: %{customdata[3]:.3f}<br>"
                "Footprint: %{customdata[4]:.0f} km"
                "<extra></extra>"
            )
        )
    )


# ==========================================
# 11. HIGHEST ALERT
# ==========================================

top = plot_df.loc[
    plot_df["alert_score"].idxmax()
]

fig.add_trace(
    go.Scattergeo(
        lat=[top["latitude"]],
        lon=[top["longitude"]],
        mode="markers+text",
        marker=dict(
            size=20,
            color=RED,
            symbol="star",
            line=dict(
                width=2,
                color="white"
            )
        ),
        text=["Highest Alert"],
        textposition="top center",
        textfont=dict(
            size=12
        ),
        name="Highest Alert",

        hovertemplate=(
            "<b>HIGHEST ALERT</b><br>"
            f"Time: {top['valid_time']}<br>"
            f"Peak: {top['max_peak_mm_3h']:.1f} mm/3h<br>"
            f"Member support: {top['member_support_pct']:.0f}%<br>"
            f"Confidence: {top['confidence_score']:.3f}<br>"
            f"Alert score: {top['alert_score']:.3f}"
            "<extra></extra>"
        )
    )
)


# ==========================================
# 12. SUMMARY VALUES
# ==========================================

max_peak = df["max_peak_mm_3h"].max()
max_support = df["member_support_pct"].max()
max_score = df["alert_score"].max()


# ==========================================
# 13. MAP LAYOUT
# ==========================================

fig.update_layout(

    title=dict(
        text=(
            "ExtremeTrack — "
            "Ensemble Consensus & Alert Intelligence"
        ),
        x=0.5,
        xanchor="center",
        font=dict(
            size=22
        )
    ),

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
        countrywidth=0.8,

        showcoastlines=True,
        coastlinecolor="#374151",
        coastlinewidth=1.2,

        showframe=False,

        lataxis=dict(
            range=[5, 37],
            showgrid=True,
            gridcolor="#D1D5DB"
        ),

        lonaxis=dict(
            range=[65, 100],
            showgrid=True,
            gridcolor="#D1D5DB"
        )
    ),

    legend=dict(
        title="Alert Level",
        orientation="h",
        yanchor="bottom",
        y=0.01,
        xanchor="center",
        x=0.5
    ),

    annotations=[
        dict(
            x=0.015,
            y=0.985,
            xref="paper",
            yref="paper",
            text=(
                "<b>ExtremeTrack Prototype</b><br>"
                f"Ensemble members: 5<br>"
                f"Max precipitation: {max_peak:.1f} mm/3h<br>"
                f"Max ensemble support: {max_support:.0f}%<br>"
                f"Max alert score: {max_score:.3f}"
            ),
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            borderpad=8
        )
    ],

    paper_bgcolor="white",
    plot_bgcolor="white",

    margin=dict(
        l=20,
        r=20,
        t=80,
        b=40
    ),

    height=760
)


# ==========================================
# 14. SAVE
# ==========================================

fig.write_html(
    OUTPUT_FILE,
    include_plotlyjs=True
)

print(
    "\n💾 Interactive visualization saved:"
)

print(OUTPUT_FILE)

print(
    "\n✅ ExtremeTrack interactive map completed!"
)