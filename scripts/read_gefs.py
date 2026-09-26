from pathlib import Path
import xarray as xr

# ==========================================
# 1. Project + GEFS Folder
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEFS_FOLDER = PROJECT_ROOT / "data" / "raw" / "gefs"

# ==========================================
# 2. Find C00 GRIB2 File Automatically
# ==========================================

files = list(GEFS_FOLDER.rglob("*c00*.grib2"))

if not files:
    raise FileNotFoundError(
        f"No C00 GRIB2 file found inside:\n{GEFS_FOLDER}"
    )

FILE_PATH = files[0]

print("✅ GRIB2 file found")
print(f"File: {FILE_PATH}")
print(
    f"Size: {FILE_PATH.stat().st_size / (1024 * 1024):.2f} MB"
)

# ==========================================
# 3. Open GRIB2
# ==========================================

print("\n📡 Reading GEFS GRIB2 file...")

ds = xr.open_dataset(
    FILE_PATH,
    engine="cfgrib",
    backend_kwargs={"indexpath": ""}
)

# ==========================================
# 4. Dataset Information
# ==========================================

print("\n========== DATASET ==========")
print(ds)

# ==========================================
# 5. Variables
# ==========================================

print("\n========== VARIABLES ==========")

for name, data in ds.data_vars.items():
    print(f"\nVariable: {name}")
    print(f"Dimensions: {data.dims}")
    print(f"Shape: {data.shape}")
    print(f"Units: {data.attrs.get('units', 'N/A')}")
    print(f"Long name: {data.attrs.get('long_name', 'N/A')}")

# ==========================================
# 6. Coordinates
# ==========================================

print("\n========== COORDINATES ==========")

for name, coord in ds.coords.items():
    print(f"{name}: shape={coord.shape}")

# ==========================================
# 7. Dimensions
# ==========================================

print("\n========== DIMENSIONS ==========")

for name, size in ds.sizes.items():
    print(f"{name}: {size}")

print("\n✅ GEFS data read successfully!")