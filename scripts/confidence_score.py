from pathlib import Path
import pandas as pd


# ==========================================
# 1. PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "consensus_objects.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "consensus_confidence.csv"
)


# ==========================================
# 2. LOAD CONSENSUS DATA
# ==========================================

print("📡 Loading consensus objects...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["valid_time"]
)

print(f"✅ Loaded {len(df)} consensus records")


# ==========================================
# 3. SUPPORT SCORE
# ==========================================

df["support_score"] = (
    df["member_support_pct"] / 100
)


# ==========================================
# 4. INTENSITY CONSISTENCY
# ==========================================

# Lower difference between mean and maximum
# means members have more similar intensities.

df["intensity_consistency"] = (
    1
    - (
        (
            df["max_peak_mm_3h"]
            - df["mean_peak_mm_3h"]
        )
        / (
            df["max_peak_mm_3h"] + 1e-6
        )
    )
)

df["intensity_consistency"] = (
    df["intensity_consistency"]
    .clip(0, 1)
)


# ==========================================
# 5. COMBINED CONFIDENCE SCORE
# ==========================================

df["confidence_score"] = (
    0.7 * df["support_score"]
    +
    0.3 * df["intensity_consistency"]
)

df["confidence_score"] = (
    df["confidence_score"]
    .clip(0, 1)
)


# ==========================================
# 6. CONFIDENCE LEVEL
# ==========================================

def confidence_level(score):

    if score >= 0.80:
        return "HIGH"

    elif score >= 0.50:
        return "MEDIUM"

    else:
        return "LOW"


df["confidence_level"] = (
    df["confidence_score"]
    .apply(confidence_level)
)


# ==========================================
# 7. SORT
# ==========================================

df = df.sort_values(
    [
        "valid_time",
        "confidence_score"
    ],
    ascending=[True, False]
)


# ==========================================
# 8. SAVE
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# 9. SUMMARY
# ==========================================

print(
    "\n========== CONFIDENCE SUMMARY =========="
)

print(
    df["confidence_level"]
    .value_counts()
)

print(
    f"\nMaximum confidence score: "
    f"{df['confidence_score'].max():.3f}"
)

print(
    f"Minimum confidence score: "
    f"{df['confidence_score'].min():.3f}"
)

print(
    f"Mean confidence score: "
    f"{df['confidence_score'].mean():.3f}"
)

print(
    "\n💾 Saved:"
)

print(OUTPUT_FILE)

print(
    "\n✅ Confidence layer completed!"
)