from pathlib import Path
import xarray as xr

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
# 2. Load 3-hour precipitation
# ==========================================

print("📡 Loading 3-hour precipitation...")

ds = xr.open_dataset(INPUT_FILE)
rain = ds["precip_3h"]

print(rain)

# ==========================================
# 3. Ensemble Mean
# ==========================================

print("\n📊 Calculating ensemble mean...")

ensemble_mean = rain.mean(dim="member")

# ==========================================
# 4. Ensemble Spread
# ==========================================

print("📊 Calculating ensemble spread...")

ensemble_spread = rain.std(dim="member")

# ==========================================
# 5. Probability of precipitation > 10 mm
# ==========================================

THRESHOLD = 10.0

print(
    f"📊 Calculating probability of precipitation > {THRESHOLD} mm / 3h..."
)

probability = (
    (rain >= THRESHOLD)
    .mean(dim="member")
    * 100
)

# ==========================================
# 6. Strongest forecast signal
# ==========================================

max_value = ensemble_mean.max()

location = ensemble_mean.where(
    ensemble_mean == max_value,
    drop=True
)

print("\n========== ENSEMBLE SUMMARY ==========")

print(
    f"Mean precipitation maximum: "
    f"{float(max_value.values):.2f} mm / 3h"
)

print(
    f"Maximum probability of > {THRESHOLD} mm / 3h: "
    f"{float(probability.max().values):.1f}%"
)

print("\nStrongest forecast location:")

print(
    location
    .to_dataframe(name="precipitation")
    .dropna()
    .head()
)

# ==========================================
# 7. General statistics
# ==========================================

print("\n========== STATISTICS ==========")

print(
    f"Overall mean: "
    f"{float(ensemble_mean.mean().values):.3f} mm / 3h"
)

print(
    f"Overall maximum: "
    f"{float(ensemble_mean.max().values):.3f} mm / 3h"
)

print(
    f"Maximum ensemble spread: "
    f"{float(ensemble_spread.max().values):.3f} mm / 3h"
)

print("\n✅ Ensemble analysis completed!")