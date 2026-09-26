from pathlib import Path
import xarray as xr

# ==========================================
# 1. Find GEFS files
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEFS_FOLDER = PROJECT_ROOT / "data" / "raw" / "gefs"

files = sorted(GEFS_FOLDER.rglob("*.grib2"))

print(f"✅ Found {len(files)} GRIB2 files")

if len(files) < 5:
    raise FileNotFoundError(
        f"Expected at least 5 GRIB2 files, but found {len(files)}"
    )

# ==========================================
# 2. Read ensemble members
# ==========================================

datasets = []

for file in files:

    name = file.name.lower()

    if "_c00" in name:
        member = "c00"
        data_type = "cf"

    elif "_p01" in name:
        member = "p01"
        data_type = "pf"

    elif "_p02" in name:
        member = "p02"
        data_type = "pf"

    elif "_p03" in name:
        member = "p03"
        data_type = "pf"

    elif "_p04" in name:
        member = "p04"
        data_type = "pf"

    else:
        continue

    print(f"\n📡 Reading {member} ...")
    print(f"File: {file.name}")

    ds = xr.open_dataset(
        file,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {
                "dataType": data_type
            }
        }
    )

    print(f"   Variable(s): {list(ds.data_vars)}")

    # Add ensemble member dimension
    ds = ds.expand_dims(member=[member])

    datasets.append(ds)

    print(f"✅ {member} loaded")

# ==========================================
# 3. Check datasets
# ==========================================

print(f"\n📦 Datasets collected: {len(datasets)}")

if not datasets:
    raise RuntimeError("No ensemble datasets were loaded.")

# ==========================================
# 4. Combine members
# ==========================================

print("\n🔗 Combining ensemble members...")

ensemble = xr.concat(
    datasets,
    dim="member"
)

# ==========================================
# 5. Display result
# ==========================================

print("\n========== ENSEMBLE DATASET ==========")
print(ensemble)

print("\n========== MEMBERS ==========")
print(ensemble.member.values)

print("\n========== DIMENSIONS ==========")

for name, size in ensemble.sizes.items():
    print(f"{name}: {size}")

print("\n✅ 5-member GEFS ensemble loaded successfully!")