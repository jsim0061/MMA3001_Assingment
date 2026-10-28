
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# STEP 1: LOAD THE DATASET
# ==========================================

FOLDER = Path(__file__).resolve().parent

CSV_PATH = FOLDER / "occupancy_15min_cleaned.csv"

data = pd.read_csv(CSV_PATH)

data["collecteddate"] = pd.to_datetime(
    data["collecteddate"],
    utc=True
)

# Create target times if missing
if "target_time" not in data.columns:
    data["target_time"] = (
        data["collecteddate"]
        + pd.Timedelta(minutes=15)
    )
else:
    data["target_time"] = pd.to_datetime(
        data["target_time"],
        utc=True
    )

data = data.sort_values(
    "collecteddate"
).reset_index(drop=True)

# ==========================================
# STEP 2: DEFINE TIME FEATURES
# ==========================================

# Historical averages must be grouped by
# the time being PREDICTED, not the input time.

target_local = data["target_time"].dt.tz_convert(
    "Australia/Melbourne"
)

data["target_hour"] = target_local.dt.hour

data["target_minute"] = target_local.dt.minute

data["target_day_of_week"] = (
    target_local.dt.dayofweek
)

target = "next_15min_occupancy"

# Check required columns
required = [
    "collecteddate",
    "target_time",
    "target_hour",
    "target_minute",
    "target_day_of_week",
    target
]

missing = [
    column for column in required
    if column not in data.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

data = data.dropna(
    subset=required
).reset_index(drop=True)

# Check genuinely measured targets
has_measured_targets = (
    "target_is_measured" in data.columns
)

if has_measured_targets:
    data["target_is_measured"] = (
        data["target_is_measured"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

print("\nDATASET INFORMATION")
print("-----------------------------")

print("Total observations:", len(data))

print("\nTarget occupancy statistics:")
print(data[target].describe())

zero_percentage = (
    (data[target] == 0).mean() * 100
)

print(f"\nZero occupancy: {zero_percentage:.2f}%")

# ==========================================
# STEP 3: SPLIT 70% / 15% / 15%
# ==========================================

n = len(data)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train = data.iloc[:train_end].copy()

validation = data.iloc[
    train_end:validation_end
].copy()

test = data.iloc[validation_end:].copy()

# Prevent targets crossing split boundaries
validation_start = validation[
    "collecteddate"
].min()

test_start = test[
    "collecteddate"
].min()

train = train[
    train["target_time"] < validation_start
].copy()

validation = validation[
    validation["target_time"] < test_start
].copy()

print("\nDATA SPLIT")
print("-----------------------------")

print("Training:", len(train))
print("Validation:", len(validation))
print("Testing:", len(test))

# ==========================================
# STEP 4: CALCULATE HISTORICAL AVERAGES
# ==========================================

# This groups historical occupancy by:
# 1. Day of week
# 2. Hour
# 3. Minute (00, 15, 30, 45)

group_columns = [
    "target_day_of_week",
    "target_hour",
    "target_minute"
]

# Learn averages from training data only
historical_means = (
    train.groupby(group_columns)[target]
    .mean()
)

# Fallback: average for each target hour
hourly_means = (
    train.groupby("target_hour")[target]
    .mean()
)

# Final fallback: overall training mean
overall_mean = train[target].mean()

print("\nHistorical averages calculated.")
print("Number of time groups:", len(historical_means))

# ==========================================
# STEP 5: PREDICTION FUNCTION
# ==========================================

def predict_historical_average(dataset):

    # Create a matching MultiIndex for lookup
    keys = pd.MultiIndex.from_frame(
        dataset[group_columns]
    )

    predictions = historical_means.reindex(
        keys
    ).to_numpy(dtype=float)

    # If an exact day/time combination
    # does not exist, use the hourly mean.
    missing_predictions = np.isnan(predictions)

    if missing_predictions.any():
        fallback = dataset["target_hour"].map(
            hourly_means
        ).to_numpy(dtype=float)

        predictions[missing_predictions] = (
            fallback[missing_predictions]
        )

    # If an hour has no historical data,
    # use the overall training mean.
    predictions = np.where(
        np.isnan(predictions),
        overall_mean,
        predictions
    )

    return predictions

# ==========================================
# STEP 6: GENERATE PREDICTIONS
# ==========================================

train_predictions = predict_historical_average(
    train
)

validation_predictions = predict_historical_average(
    validation
)

test_predictions = predict_historical_average(
    test
)

# ==========================================
# STEP 7: EVALUATION FUNCTION
# ==========================================

def evaluate_model(actual, predicted, name):

    mae = mean_absolute_error(
        actual, predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(
        actual, predicted
    )

    print(f"\n{name.upper()} RESULTS")
    print("-----------------------------")

    print(f"MAE:  {mae:.3f} occupants")
    print(f"RMSE: {rmse:.3f} occupants")
    print(f"R²:   {r2:.3f}")

    return {
        "Dataset": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

# ==========================================
# STEP 8: EVALUATE MODEL
# ==========================================

results = []

results.append(
    evaluate_model(
        train[target],
        train_predictions,
        "Training"
    )
)

results.append(
    evaluate_model(
        validation[target],
        validation_predictions,
        "Validation"
    )
)

results.append(
    evaluate_model(
        test[target],
        test_predictions,
        "Testing"
    )
)

# ==========================================
# STEP 9: MEASURED-TARGET EVALUATION
# ==========================================

if has_measured_targets:

    measured_mask = (
        test["target_is_measured"].to_numpy()
    )

    print("\nMEASURED-TARGET EVALUATION")
    print("-----------------------------")

    print(
        "Number of genuine targets:",
        measured_mask.sum()
    )

    if measured_mask.sum() >= 2:

        results.append(
            evaluate_model(
                test[target].to_numpy()[measured_mask],
                test_predictions[measured_mask],
                "Measured Targets Only"
            )
        )

    else:
        print("Insufficient measured targets.")

else:
    print(
        "\nMeasured-target flags unavailable."
    )

# ==========================================
# STEP 10: SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    FOLDER / "historical_average_results.csv",
    index=False
)

# Save predictions for comparison
prediction_df = pd.DataFrame({
    "target_time": test["target_time"].to_numpy(),
    "actual_occupancy": test[target].to_numpy(),
    "predicted_occupancy": test_predictions
})

if has_measured_targets:
    prediction_df["target_is_measured"] = (
        test["target_is_measured"].to_numpy()
    )

prediction_df.to_csv(
    FOLDER / "historical_average_predictions.csv",
    index=False
)

# ==========================================
# STEP 11: ACTUAL VS PREDICTED GRAPH
# ==========================================

n_plot = min(150, len(test))

plt.figure(figsize=(12, 5))

plt.plot(
    test["target_time"].iloc[:n_plot],
    test[target].iloc[:n_plot],
    label="Actual Occupancy",
    linewidth=2
)

plt.plot(
    test["target_time"].iloc[:n_plot],
    test_predictions[:n_plot],
    label="Historical Average Prediction",
    linestyle="--",
    linewidth=2
)

plt.title(
    "Historical Average: Actual vs Predicted Occupancy"
)

plt.xlabel("Target Date and Time")
plt.ylabel("Occupancy (people)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    FOLDER / "historical_average_predictions.png",
    dpi=200
)

plt.show()

# ==========================================
# STEP 12: PREDICTION ERROR HISTOGRAM
# ==========================================

errors = (
    test[target].to_numpy() - test_predictions
)

plt.figure(figsize=(10, 5))

plt.hist(
    errors,
    bins=30,
    edgecolor="black"
)

plt.title("Historical Average Prediction Errors")
plt.xlabel("Actual - Predicted Occupancy")
plt.ylabel("Frequency")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FOLDER / "historical_average_errors.png",
    dpi=200
)

plt.show()

# ==========================================
# STEP 13: HISTORICAL OCCUPANCY PROFILE
# ==========================================

# Average occupancy by hour for each weekday
hourly_profile = (
    train.groupby([
        "target_day_of_week",
        "target_hour"
    ])[target]
    .mean()
    .unstack(0)
)

weekday_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

plt.figure(figsize=(12, 6))

for day_number in hourly_profile.columns:

    plt.plot(
        hourly_profile.index,
        hourly_profile[day_number],
        label=weekday_names[int(day_number)]
    )

plt.title("Historical Average Occupancy by Day")

plt.xlabel("Hour of Day")
plt.ylabel("Average Occupancy (people)")

plt.xticks(range(24))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FOLDER / "historical_occupancy_profile.png",
    dpi=200
)

plt.show()

# ==========================================
# STEP 14: FINISH
# ==========================================

print("\n===================================")
print("HISTORICAL AVERAGE MODEL COMPLETE")
print("===================================")

print(results_df)

print("\nFiles saved successfully.")
