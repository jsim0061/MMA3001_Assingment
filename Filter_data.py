
import pandas as pd
import zipfile
from pathlib import Path

# ==============================================
# STEP 1: File locations and settings
# ==============================================

FOLDER = Path(__file__).resolve().parent

ZIP_PATH = FOLDER / "Occupancy_Room.zip"
OUTPUT_PATH = FOLDER / "occupancy_15min_cleaned.csv"

ROOM_ID = "0f3f37f8-0335-41e0-ae87-a1d677dfc7d7"

CHUNK_SIZE = 100_000
MAX_GAP_MINUTES = 30

# ==============================================
# STEP 2: Read the ZIP file
# ==============================================

print("Starting occupancy data extraction...")

parts = []
rows_processed = 0

with zipfile.ZipFile(ZIP_PATH) as archive:

    csv_files = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    ]

    if not csv_files:
        raise ValueError("No CSV file inside ZIP.")

    with archive.open(csv_files[0]) as file:

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            usecols=[
                "floorspaceid",
                "collecteddate",
                "headcount"
            ]
        ):

            rows_processed += len(chunk)

            selected = chunk.loc[
                chunk["floorspaceid"] == ROOM_ID,
                ["collecteddate", "headcount"]
            ].copy()

            if not selected.empty:
                parts.append(selected)

            print(
                f"Processed {rows_processed:,} rows",
                flush=True
            )

if not parts:
    raise ValueError(
        "No occupancy records for selected room."
    )

df = pd.concat(parts, ignore_index=True)

print(f"\nSelected room records: {len(df):,}")

# ==============================================
# STEP 3: Clean the raw observations
# ==============================================

df["collecteddate"] = pd.to_datetime(
    df["collecteddate"],
    errors="coerce",
    utc=True
)

df["headcount"] = pd.to_numeric(
    df["headcount"],
    errors="coerce"
)

df = df.dropna(
    subset=["collecteddate", "headcount"]
)

df = df[df["headcount"] >= 0].copy()

df = df.sort_values("collecteddate")

# Keep the final record where timestamps repeat
df = df.drop_duplicates(
    subset=["collecteddate"],
    keep="last"
)

print(f"Valid room records: {len(df):,}")

# ==============================================
# STEP 4: Make 15-minute snapshots
# ==============================================

df = df.set_index("collecteddate")

# Each timestamp represents the END of an interval.
# For example, 09:10 is in the interval ending 09:15.

occupancy = df["headcount"].resample(
    "15min",
    label="right",
    closed="right"
).last()

# Keep the precise time of the final genuine
# observation in each interval.
measurement_times = pd.Series(
    df.index,
    index=df.index
).resample(
    "15min",
    label="right",
    closed="right"
).last()

data = occupancy.to_frame(
    name="measured_occupancy"
)

data["measurement_time"] = measurement_times

# ==============================================
# STEP 5: Apply the 30-minute missing-data rule
# ==============================================

data["is_measured"] = (
    data["measured_occupancy"].notna()
)

# Remember the time of the last real observation
data["last_measurement"] = (
    data["measurement_time"].ffill()
)

# Calculate how old the last observation is
data["minutes_since_measurement"] = (
    (
        data.index.to_series()
        - data["last_measurement"]
    ).dt.total_seconds() / 60
)

data["assumed_empty"] = (
    data["minutes_since_measurement"]
    > MAX_GAP_MINUTES
)

# Initially retain the last known headcount
data["current_occupancy"] = (
    data["measured_occupancy"].ffill()
)

# Replace occupancy with zero when no reading
# has been received for more than 30 minutes
data.loc[
    data["assumed_empty"],
    "current_occupancy"
] = 0

# A real reading must always take priority
data.loc[
    data["is_measured"],
    "current_occupancy"
] = data.loc[
    data["is_measured"],
    "measured_occupancy"
]

# ==============================================
# STEP 6: Create inputs and prediction target
# ==============================================

data["previous_15min"] = (
    data["current_occupancy"].shift(1)
)

data["previous_30min"] = (
    data["current_occupancy"].shift(2)
)

# Predict occupancy 15 minutes into the future
data["next_15min_occupancy"] = (
    data["current_occupancy"].shift(-1)
)

# Timestamp corresponding to the prediction
data["target_time"] = (
    data.index + pd.Timedelta(minutes=15)
)

# Identify whether the target was actually measured
data["target_is_measured"] = (
    data["is_measured"]
    .shift(-1)
    .fillna(False)
    .astype(bool)
)

# ==============================================
# STEP 7: Local time features
# ==============================================

local_time = data.index.tz_convert(
    "Australia/Melbourne"
)

data["hour"] = local_time.hour
data["minute"] = local_time.minute
data["day_of_week"] = local_time.dayofweek

# ==============================================
# STEP 8: Remove incomplete observations
# ==============================================

data = data.dropna(
    subset=[
        "current_occupancy",
        "previous_15min",
        "previous_30min",
        "next_15min_occupancy"
    ]
)

data = data.reset_index()

# ==============================================
# STEP 9: Save cleaned dataset
# ==============================================

data.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n===================================")
print("DATA PROCESSING COMPLETE")
print("===================================")

print(f"Usable 15-minute rows: {len(data):,}")

print(
    f"Assumed-empty intervals: "
    f"{data['assumed_empty'].sum():,}"
)

print(
    f"Genuinely measured targets: "
    f"{data['target_is_measured'].sum():,}"
)

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst five rows:")
print(data.head())

print(f"\nSaved file: {OUTPUT_PATH}")
