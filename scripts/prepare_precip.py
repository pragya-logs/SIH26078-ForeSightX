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
    / "gefs_india_2019010700.nc"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gefs_india_2019010700_3h.nc"
)

# ==========================================
# 2. Load data
# ==========================================

print("📡 Loading regional GEFS data...")

ds = xr.open_dataset(INPUT_FILE)

print(ds)

# ==========================================
# 3. Convert accumulated precipitation
#    to 3-hour precipitation
# ==========================================

print("\n🌧️ Converting accumulated precipitation...")

rain_3h = ds["tp"].diff("valid_time")

# Small negative values can occur because of GRIB
# numerical/accumulation behaviour.
rain_3h = rain_3h.clip(min=0)

rain_3h.name = "precip_3h"

rain_3h.attrs["units"] = "kg m-2"
rain_3h.attrs["description"] = (
    "3-hour precipitation derived from accumulated GEFS total precipitation"
)

# ==========================================
# 4. Save
# ==========================================

rain_3h.to_dataset().to_netcdf(OUTPUT_FILE)

print("\n✅ 3-hour precipitation created!")

print("\n========== RESULT ==========")
print(rain_3h)

print("\n========== DIMENSIONS ==========")

for name, size in rain_3h.sizes.items():
    print(f"{name}: {size}")

print("\n💾 Saved:")
print(OUTPUT_FILE)