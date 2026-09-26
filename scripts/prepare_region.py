from pathlib import Path
import numpy as np
import xarray as xr

# ==========================================
# 1. Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEFS_FOLDER = PROJECT_ROOT / "data" / "raw" / "gefs"

# ==========================================
# 2. Find GEFS files
# ==========================================

files = sorted(GEFS_FOLDER.rglob("*.grib2"))

member_files = {}

for file in files:
    name = file.name.lower()

    for member in ["c00", "p01", "p02", "p03", "p04"]:
        if f"_{member}.grib2" in name:
            member_files[member] = file

print("========== FILES ==========")

for member in ["c00", "p01", "p02", "p03", "p04"]:
    print(member, "→", member_files[member].name)

# ==========================================
# 3. Read regional data
# ==========================================

datasets = []

for member in ["c00", "p01", "p02", "p03", "p04"]:

    data_type = "cf" if member == "c00" else "pf"

    print(f"\n📡 Reading {member}...")

    ds = xr.open_dataset(
        member_files[member],
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {
                "dataType": data_type
            }
        }
    )

    tp = ds["tp"]

    # India + nearby seas
    tp = tp.sel(
        latitude=slice(37, 5),
        longitude=slice(65, 100)
    )

    # Use valid_time instead of step
    tp = tp.swap_dims({
        "step": "valid_time"
    })

    # Remove old step coordinate
    tp = tp.reset_coords(
        "step",
        drop=True
    )

    tp = tp.reset_coords(
    ["number", "time", "surface"],
    drop=True
)

    print(f"   Forecast times: {len(tp.valid_time)}")

    tp = tp.expand_dims(
        member=[member]
    )

    datasets.append(tp)

    print(f"✅ {member} regional data ready")

# ==========================================
# 4. Find common forecast times
# ==========================================

print("\n🔍 Finding common forecast times...")

common_times = datasets[0]["valid_time"].values

for tp in datasets[1:]:
    common_times = np.intersect1d(
        common_times,
        tp["valid_time"].values
    )

if len(common_times) == 0:
    raise RuntimeError(
        "No common valid_time found between ensemble members."
    )

print(f"✅ Common forecast times: {len(common_times)}")

# ==========================================
# 5. Align all members
# ==========================================

aligned_datasets = []

for tp in datasets:
    tp = tp.sel(valid_time=common_times)
    aligned_datasets.append(tp)

# ==========================================
# 6. Combine ensemble
# ==========================================

print("\n🔗 Combining ensemble members...")

ensemble_tp = xr.concat(
    aligned_datasets,
    dim="member",
    join="exact",
    coords="minimal"
)

# ==========================================
# 7. Display result
# ==========================================

print("\n========== REGIONAL ENSEMBLE ==========")
print(ensemble_tp)

print("\n========== MEMBERS ==========")
print(ensemble_tp.member.values)

print("\n========== DIMENSIONS ==========")

for name, size in ensemble_tp.sizes.items():
    print(f"{name}: {size}")

print("\n✅ India regional ensemble ready!")

# ==========================================
# 8. Save regional ensemble
# ==========================================

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "gefs_india_2019010700.nc"

ensemble_tp.to_dataset(name="tp").to_netcdf(
    OUTPUT_FILE
)

print(f"\n💾 Saved regional ensemble:")
print(OUTPUT_FILE)