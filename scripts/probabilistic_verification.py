from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORECAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gefs_india_2019010700_3h.nc"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "probabilistic_verification.csv"
)

RELIABILITY_FILE = (
    OUTPUT_DIR
    / "reliability_data.csv"
)

PLOT_FILE = (
    OUTPUT_DIR
    / "reliability_diagram.png"
)


# =========================================================
# 2. SETTINGS
# =========================================================

THRESHOLD_MM = 10.0

START_TIME = "2019-01-07T09:00:00"
END_TIME = "2019-01-17T00:00:00"

ERA5_PATH = (
    "gs://gcp-public-data-arco-era5/"
    "ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)


# =========================================================
# 3. LOAD GEFS
# =========================================================

print("📡 Loading GEFS ensemble...")

forecast_ds = xr.open_dataset(
    FORECAST_FILE
)

forecast = forecast_ds["precip_3h"]

print(
    f"✅ Forecast shape: {forecast.shape}"
)

print(
    f"✅ Members: "
    f"{forecast.member.values}"
)


# =========================================================
# 4. CALCULATE ENSEMBLE EVENT PROBABILITY
# =========================================================

print(
    f"\n📊 Calculating probability of "
    f"precipitation >= {THRESHOLD_MM} mm/3h..."
)

ensemble_event_probability = (
    (forecast >= THRESHOLD_MM)
    .mean(dim="member")
)

print(
    f"✅ Probability values: "
    f"{np.unique(ensemble_event_probability.values)}"
)


# =========================================================
# 5. LOAD ERA5
# =========================================================

print("\n🌍 Opening ERA5...")

era5_ds = xr.open_zarr(
    ERA5_PATH,
    chunks=None,
    storage_options={
        "token": "anon"
    }
)

print("✅ ERA5 opened")


# =========================================================
# 6. SELECT ERA5 TOTAL PRECIPITATION
# =========================================================

era5 = era5_ds[
    "total_precipitation"
]


# =========================================================
# 7. SELECT TIME + REGION
# =========================================================

print("\n🌧️ Selecting ERA5 region...")

era5 = era5.sel(

    time=slice(
        START_TIME,
        END_TIME
    ),

    latitude=slice(
        37,
        5
    ),

    longitude=slice(
        65,
        100
    )
)


# =========================================================
# 8. SORT LATITUDE / LONGITUDE
# =========================================================

era5 = era5.sortby(
    "latitude"
)

era5 = era5.sortby(
    "longitude"
)


print(
    f"✅ ERA5 subset shape: "
    f"{era5.shape}"
)


# =========================================================
# 9. METERS → MILLIMETERS
# =========================================================

print(
    "\n🌧️ Converting ERA5 precipitation "
    "to millimeters..."
)

era5_mm = (
    era5 * 1000.0
)


# =========================================================
# 10. HOURLY → 3-HOUR
# =========================================================

print(
    "⏱️ Aggregating ERA5 to 3-hour precipitation..."
)

era5_3h = (
    era5_mm
    .resample(
        time="3h",
        closed="right",
        label="right"
    )
    .sum()
)


# =========================================================
# 11. ALIGN TIME
# =========================================================

print(
    "\n🔗 Aligning GEFS and ERA5..."
)

common_times = np.intersect1d(

    ensemble_event_probability[
        "valid_time"
    ].values,

    era5_3h[
        "time"
    ].values
)


if len(common_times) == 0:

    raise RuntimeError(
        "No common timestamps found."
    )


forecast_probability = (
    ensemble_event_probability
    .sel(
        valid_time=common_times
    )
)


era5_eval = (
    era5_3h
    .sel(
        time=common_times
    )
)


print(
    f"✅ Common timestamps: "
    f"{len(common_times)}"
)


# =========================================================
# 12. ALIGN SPATIAL GRID
# =========================================================

print(
    "\n🗺️ Aligning spatial grid..."
)

forecast_probability = (
    forecast_probability
    .sel(
        latitude=era5_eval.latitude,
        longitude=era5_eval.longitude
    )
)


print(
    f"✅ Evaluation shape: "
    f"{forecast_probability.shape}"
)


# =========================================================
# 13. OBSERVED BINARY EVENT
# =========================================================

observed_event = (
    era5_eval >= THRESHOLD_MM
)


# =========================================================
# 14. CONVERT TO NUMPY
# =========================================================

p = (
    forecast_probability
    .values
    .astype(float)
)

o = (
    observed_event
    .values
    .astype(float)
)


# Remove invalid values
valid = (
    np.isfinite(p)
    &
    np.isfinite(o)
)

p = p[valid]
o = o[valid]


# =========================================================
# 15. BRIER SCORE
# =========================================================

print(
    "\n📐 Calculating Brier Score..."
)

brier_score = np.mean(
    (p - o) ** 2
)


# =========================================================
# 16. OBSERVED EVENT RATE
# =========================================================

observed_event_rate = np.mean(
    o
)


# =========================================================
# 17. CLIMATOLOGICAL REFERENCE
# =========================================================

climatology_probability = (
    observed_event_rate
)

climatology_brier = (
    np.mean(
        (
            climatology_probability
            - o
        ) ** 2
    )
)


# Brier Skill Score relative to
# constant climatological probability
if climatology_brier > 0:

    brier_skill_score = (
        1
        -
        (
            brier_score
            / climatology_brier
        )
    )

else:

    brier_skill_score = np.nan


# =========================================================
# 18. RELIABILITY DATA
# =========================================================

print(
    "\n📊 Building reliability data..."
)

# With 5 members, probabilities are normally
# multiples of 0.20.
probability_levels = (
    np.arange(0, 1.01, 0.20)
)

reliability_rows = []

for prob in probability_levels:

    # Small tolerance for floating point
    mask = np.isclose(
        p,
        prob,
        atol=1e-6
    )

    count = int(
        mask.sum()
    )

    if count > 0:

        observed_frequency = float(
            o[mask].mean()
        )

    else:

        observed_frequency = np.nan

    reliability_rows.append(
        {
            "forecast_probability": prob,
            "observed_frequency": observed_frequency,
            "sample_count": count
        }
    )


reliability_df = pd.DataFrame(
    reliability_rows
)


# =========================================================
# 19. SAVE RELIABILITY DATA
# =========================================================

reliability_df.to_csv(
    RELIABILITY_FILE,
    index=False
)


# =========================================================
# 20. RELIABILITY DIAGRAM
# =========================================================

print(
    "\n📈 Creating reliability diagram..."
)

plot_df = reliability_df.dropna(
    subset=[
        "observed_frequency"
    ]
)

plt.figure(
    figsize=(8, 7)
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect reliability"
)

plt.plot(
    plot_df[
        "forecast_probability"
    ],
    plot_df[
        "observed_frequency"
    ],
    marker="o",
    linewidth=2,
    label="5-member GEFS"
)

plt.xlabel(
    "Forecast Probability"
)

plt.ylabel(
    "Observed Event Frequency"
)

plt.title(
    f"Reliability Diagram\n"
    f"Precipitation ≥ {THRESHOLD_MM} mm/3h"
)

plt.xlim(
    0,
    1
)

plt.ylim(
    0,
    1
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=200
)

plt.close()


# =========================================================
# 21. RESULTS TABLE
# =========================================================

results = pd.DataFrame(
    [
        {
            "metric": "Brier_Score",
            "value": brier_score
        },
        {
            "metric": "Observed_Event_Rate",
            "value": observed_event_rate
        },
        {
            "metric": "Climatology_Brier",
            "value": climatology_brier
        },
        {
            "metric": "Brier_Skill_Score",
            "value": brier_skill_score
        },
        {
            "metric": "Total_Evaluation_Points",
            "value": len(p)
        }
    ]
)


# =========================================================
# 22. SAVE RESULTS
# =========================================================

results.to_csv(
    RESULT_FILE,
    index=False
)


# =========================================================
# 23. PRINT RESULTS
# =========================================================

print(
    "\n========== PROBABILISTIC VERIFICATION =========="
)

print(
    f"Brier Score: "
    f"{brier_score:.6f}"
)

print(
    f"Observed event rate: "
    f"{observed_event_rate:.6f}"
)

print(
    f"Climatology Brier: "
    f"{climatology_brier:.6f}"
)

print(
    f"Brier Skill Score: "
    f"{brier_skill_score:.6f}"
)


print(
    "\n========== RELIABILITY =========="
)

print(
    reliability_df
)


print(
    "\n💾 Results:"
)

print(RESULT_FILE)

print(
    "\n💾 Reliability data:"
)

print(RELIABILITY_FILE)

print(
    "\n💾 Reliability diagram:"
)

print(PLOT_FILE)


print(
    "\n✅ Probabilistic verification completed!"
)