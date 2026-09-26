from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_verification.csv"
)


# =========================================================
# 2. SETTINGS
# =========================================================

# GEFS forecast verification period
START_TIME = "2019-01-07T09:00:00"
END_TIME = "2019-01-17T00:00:00"

# Prototype extreme threshold
THRESHOLD_MM = 10.0


# =========================================================
# 3. LOAD GEFS FORECAST
# =========================================================

print("📡 Loading GEFS forecast...")

forecast_ds = xr.open_dataset(
    FORECAST_FILE
)

forecast = forecast_ds["precip_3h"]

print(
    f"✅ Forecast shape: {forecast.shape}"
)

print(
    f"✅ Forecast time range: "
    f"{forecast.valid_time.values[0]} → "
    f"{forecast.valid_time.values[-1]}"
)


# =========================================================
# 4. ENSEMBLE MEAN
# =========================================================

print("\n📊 Calculating ensemble mean...")

forecast_mean = (
    forecast
    .mean(dim="member")
)


# =========================================================
# 5. OPEN PUBLIC ERA5
# =========================================================

print("\n🌍 Opening public ERA5 dataset...")

ERA5_PATH = (
    "gs://gcp-public-data-arco-era5/"
    "ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

era5_ds = xr.open_zarr(
    ERA5_PATH,
    chunks=None,
    storage_options={
        "token": "anon"
    }
)

print("✅ ERA5 dataset opened")


# =========================================================
# 6. SELECT ERA5 PRECIPITATION
# =========================================================

print("\n🌧️ Selecting ERA5 total precipitation...")

era5 = era5_ds["total_precipitation"]


# =========================================================
# 7. SELECT TIME + REGION
# =========================================================

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
# 8. CHECK LATITUDE ORDER
# =========================================================

if era5.latitude.values[0] > era5.latitude.values[-1]:

    era5 = era5.sortby(
        "latitude"
    )


# =========================================================
# 9. CHECK LONGITUDE ORDER
# =========================================================

era5 = era5.sortby(
    "longitude"
)


print(
    f"✅ ERA5 subset shape: "
    f"{era5.shape}"
)


# =========================================================
# 10. CONVERT ERA5 TOTAL PRECIPITATION
#     meters → millimeters
# =========================================================

print(
    "\n🌧️ Converting ERA5 precipitation "
    "to millimeters..."
)

era5_mm = (
    era5 * 1000.0
)


# =========================================================
# 11. CONVERT HOURLY ERA5 TO 3-HOUR
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
# 12. ALIGN TIME
# =========================================================

print(
    "\n🔗 Aligning forecast and ERA5..."
)

forecast_mean = forecast_mean.sortby(
    "latitude"
)

forecast_mean = forecast_mean.sortby(
    "longitude"
)

common_times = np.intersect1d(
    forecast_mean.valid_time.values,
    era5_3h.time.values
)

if len(common_times) == 0:

    raise RuntimeError(
        "No common forecast/ERA5 timestamps found."
    )


forecast_eval = forecast_mean.sel(
    valid_time=common_times
)

era5_eval = era5_3h.sel(
    time=common_times
)


print(
    f"✅ Common timestamps: "
    f"{len(common_times)}"
)


# =========================================================
# 13. ALIGN SPATIAL GRID
# =========================================================

# Both products are nominally 0.25°.
# Select common coordinates explicitly.

forecast_eval = forecast_eval.sel(
    latitude=era5_eval.latitude,
    longitude=era5_eval.longitude
)

print(
    f"✅ Evaluation grid: "
    f"{forecast_eval.shape}"
)


# =========================================================
# 14. CALCULATE ERROR
# =========================================================

print(
    "\n📐 Calculating verification metrics..."
)

forecast_values = (
    forecast_eval.values
)

observed_values = (
    era5_eval.values
)

# Remove invalid values
valid = np.isfinite(
    forecast_values
) & np.isfinite(
    observed_values
)

f = forecast_values[valid]
o = observed_values[valid]


# =========================================================
# 15. DETERMINISTIC METRICS
# =========================================================

bias = np.mean(
    f - o
)

mae = np.mean(
    np.abs(f - o)
)

rmse = np.sqrt(
    np.mean(
        (f - o) ** 2
    )
)


# =========================================================
# 16. EXTREME EVENT METRICS
# =========================================================

forecast_event = (
    f >= THRESHOLD_MM
)

observed_event = (
    o >= THRESHOLD_MM
)

hits = np.sum(
    forecast_event
    & observed_event
)

misses = np.sum(
    (~forecast_event)
    & observed_event
)

false_alarms = np.sum(
    forecast_event
    & (~observed_event)
)

correct_negatives = np.sum(
    (~forecast_event)
    & (~observed_event)
)


# Probability of detection
if (
    hits + misses
) > 0:

    pod = (
        hits
        / (hits + misses)
    )

else:

    pod = np.nan


# False alarm ratio
if (
    hits + false_alarms
) > 0:

    far = (
        false_alarms
        / (hits + false_alarms)
    )

else:

    far = np.nan


# CSI
if (
    hits + misses + false_alarms
) > 0:

    csi = (
        hits
        /
        (
            hits
            + misses
            + false_alarms
        )
    )

else:

    csi = np.nan


# =========================================================
# 17. RESULTS
# =========================================================

results = pd.DataFrame(
    [
        {
            "metric": "BIAS_mm_3h",
            "value": bias
        },
        {
            "metric": "MAE_mm_3h",
            "value": mae
        },
        {
            "metric": "RMSE_mm_3h",
            "value": rmse
        },
        {
            "metric": "POD",
            "value": pod
        },
        {
            "metric": "FAR",
            "value": far
        },
        {
            "metric": "CSI",
            "value": csi
        },
        {
            "metric": "HITS",
            "value": hits
        },
        {
            "metric": "MISSES",
            "value": misses
        },
        {
            "metric": "FALSE_ALARMS",
            "value": false_alarms
        }
    ]
)


# =========================================================
# 18. SAVE
# =========================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# 19. PRINT
# =========================================================

print(
    "\n========== VERIFICATION RESULTS =========="
)

for _, row in results.iterrows():

    print(
        f"{row['metric']}: "
        f"{row['value']:.4f}"
        if isinstance(
            row["value"],
            (float, np.floating)
        )
        else
        f"{row['metric']}: "
        f"{row['value']}"
    )


print(
    "\n💾 Saved:"
)

print(
    OUTPUT_FILE
)

print(
    "\n✅ Forecast verification completed!"
)