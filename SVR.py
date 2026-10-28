
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import perf_counter

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# STEP 1: FILES AND SETTINGS
# ==========================================

FOLDER = Path(__file__).resolve().parent
CSV_PATH = FOLDER / "occupancy_15min_cleaned.csv"

# SVR training can be slow with large datasets.
MAX_TRAIN_SAMPLES = 5000

# A small, manageable parameter investigation
C_VALUES = [1, 10, 100]
EPSILON_VALUES = [0.1, 0.5]
GAMMA_VALUES = ["scale", 0.1]

RANDOM_STATE = 42

# ==========================================
# STEP 2: LOAD AND SORT THE DATA
# ==========================================

data = pd.read_csv(CSV_PATH)

data["collecteddate"] = pd.to_datetime(
    data["collecteddate"],
    utc=True
)

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

# Extract local time variables
local_time = data["collecteddate"].dt.tz_convert(
    "Australia/Melbourne"
)

data["hour"] = local_time.dt.hour
data["minute"] = local_time.dt.minute
data["day_of_week"] = local_time.dt.dayofweek

# ==========================================
# STEP 3: DEFINE FEATURES AND TARGET
# ==========================================

features = [
    "current_occupancy",
    "previous_15min",
    "previous_30min",
    "hour",
    "minute",
    "day_of_week"
]

target = "next_15min_occupancy"

required_columns = features + [target]

missing = [
    column for column in required_columns
    if column not in data.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        "Run data_processing.py first."
    )

data = data.dropna(
    subset=required_columns
).reset_index(drop=True)

# Optional genuine measurement information
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

print("\nOccupancy statistics:")
print(data["current_occupancy"].describe())

zero_percentage = (
    (data["current_occupancy"] == 0).mean() * 100
)

print(f"Zero occupancy: {zero_percentage:.2f}%")

# ==========================================
# STEP 4: SPLIT 70% / 15% / 15%
# ==========================================

n = len(data)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train = data.iloc[:train_end].copy()

validation = data.iloc[
    train_end:validation_end
].copy()

test = data.iloc[
    validation_end:
].copy()

# Avoid target leakage at split boundaries
validation_start = (
    validation["collecteddate"].min()
)

test_start = (
    test["collecteddate"].min()
)

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
# STEP 5: SAMPLE TRAINING DATA FOR SVR
# ==========================================

# Select a reproducible subset from the
# training period, without using future data.

if len(train) > MAX_TRAIN_SAMPLES:

    train_sample = (
        train.sample(
            n=MAX_TRAIN_SAMPLES,
            random_state=RANDOM_STATE
        )
        .sort_values("collecteddate")
    )

else:
    train_sample = train.copy()

print(
    "\nSVR training sample size:",
    len(train_sample)
)

X_train = train_sample[features]
y_train = train_sample[target]

X_val = validation[features]
y_val = validation[target]

X_test = test[features]
y_test = test[target]

# ==========================================
# STEP 6: MODEL EVALUATION FUNCTION
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
# STEP 7: INVESTIGATE SVR PARAMETERS
# ==========================================

parameter_results = []

best_model = None
best_params = None
best_validation_mae = float("inf")

print("\nSVR PARAMETER INVESTIGATION")
print("-----------------------------")

for C in C_VALUES:

    for epsilon in EPSILON_VALUES:

        for gamma in GAMMA_VALUES:

            model = make_pipeline(
                StandardScaler(),
                SVR(
                    kernel="rbf",
                    C=C,
                    epsilon=epsilon,
                    gamma=gamma,
                    cache_size=500
                )
            )

            start_time = perf_counter()

            model.fit(X_train, y_train)

            training_time = (
                perf_counter() - start_time
            )

            # Use validation data for selection
            validation_predictions = (
                model.predict(X_val)
            )

            validation_mae = mean_absolute_error(
                y_val,
                validation_predictions
            )

            parameter_results.append({
                "C": C,
                "epsilon": epsilon,
                "gamma": gamma,
                "Validation_MAE": validation_mae,
                "Training_Time_s": training_time
            })

            print(
                f"C={C}, epsilon={epsilon}, "
                f"gamma={gamma} | "
                f"MAE={validation_mae:.3f} | "
                f"Time={training_time:.2f}s",
                flush=True
            )

            if validation_mae < best_validation_mae:

                best_validation_mae = validation_mae
                best_model = model

                best_params = {
                    "C": C,
                    "epsilon": epsilon,
                    "gamma": gamma
                }

print("\nBEST SVR PARAMETERS")
print(best_params)

# Save parameter investigation
parameter_df = pd.DataFrame(
    parameter_results
)

parameter_df.to_csv(
    FOLDER / "svr_parameter_results.csv",
    index=False
)

# ==========================================
# STEP 8: GENERATE FINAL PREDICTIONS
# ==========================================

train_predictions = best_model.predict(
    X_train
)

validation_predictions = best_model.predict(
    X_val
)

# Use test data only after choosing settings
test_predictions = best_model.predict(
    X_test
)

# ==========================================
# STEP 9: EVALUATE MODEL
# ==========================================

results = []

results.append(
    evaluate_model(
        y_train,
        train_predictions,
        "Training Sample"
    )
)

results.append(
    evaluate_model(
        y_val,
        validation_predictions,
        "Validation"
    )
)

results.append(
    evaluate_model(
        y_test,
        test_predictions,
        "Testing"
    )
)

# ==========================================
# STEP 10: GENUINELY MEASURED TARGETS
# ==========================================

if has_measured_targets:

    measured_mask = (
        test["target_is_measured"].to_numpy()
    )

    print("\nMEASURED TARGETS")
    print("-----------------------------")
    print(
        "Number of measured targets:",
        measured_mask.sum()
    )

    if measured_mask.sum() >= 2:

        results.append(
            evaluate_model(
                y_test.to_numpy()[measured_mask],
                test_predictions[measured_mask],
                "Measured Targets Only"
            )
        )

    else:
        print(
            "Insufficient genuinely measured targets."
        )

else:
    print(
        "\nMeasured-target information unavailable."
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    FOLDER / "svr_results.csv",
    index=False
)

# ==========================================
# STEP 11: SAVE TEST PREDICTIONS
# ==========================================

prediction_df = pd.DataFrame({
    "target_time": test["target_time"].to_numpy(),
    "actual_occupancy": y_test.to_numpy(),
    "predicted_occupancy": test_predictions
})

if has_measured_targets:
    prediction_df["target_is_measured"] = (
        test["target_is_measured"].to_numpy()
    )

prediction_df.to_csv(
    FOLDER / "svr_predictions.csv",
    index=False
)

# ==========================================
# STEP 12: ACTUAL VS PREDICTED GRAPH
# ==========================================

n_plot = min(150, len(test))

plt.figure(figsize=(12, 5))

plt.plot(
    test["target_time"].iloc[:n_plot],
    y_test.iloc[:n_plot].to_numpy(),
    label="Actual Occupancy",
    linewidth=2
)

plt.plot(
    test["target_time"].iloc[:n_plot],
    test_predictions[:n_plot],
    label="SVR Prediction",
    linestyle="--",
    linewidth=2
)

plt.title(
    "SVR: Actual vs Predicted Occupancy"
)

plt.xlabel("Target Date and Time")
plt.ylabel("Occupancy (people)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    FOLDER / "svr_predictions.png",
    dpi=200
)

plt.show()

# ==========================================
# STEP 13: PREDICTION ERROR HISTOGRAM
# ==========================================

errors = (
    y_test.to_numpy() - test_predictions
)

plt.figure(figsize=(10, 5))

plt.hist(
    errors,
    bins=30,
    edgecolor="black"
)

plt.title("SVR Prediction Errors")
plt.xlabel("Actual - Predicted Occupancy")
plt.ylabel("Frequency")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FOLDER / "svr_errors.png",
    dpi=200
)

plt.show()

# ==========================================
# STEP 14: PARAMETER COMPARISON GRAPH
# ==========================================

parameter_df = parameter_df.sort_values(
    "Validation_MAE"
).reset_index(drop=True)

# Display the ten best combinations
top_results = parameter_df.head(10).copy()

top_results["Configuration"] = (
    "C=" + top_results["C"].astype(str)
    + ", e=" + top_results["epsilon"].astype(str)
    + ", g=" + top_results["gamma"].astype(str)
)

plt.figure(figsize=(12, 6))

plt.barh(
    top_results["Configuration"],
    top_results["Validation_MAE"]
)

plt.gca().invert_yaxis()

plt.title("SVR Parameter Comparison")
plt.xlabel("Validation MAE (occupants)")
plt.ylabel("SVR Configuration")

plt.tight_layout()

plt.savefig(
    FOLDER / "svr_parameter_comparison.png",
    dpi=200
)

plt.show()

# ==========================================
# STEP 15: FINAL SUMMARY
# ==========================================

print("\n===================================")
print("SUPPORT VECTOR REGRESSION COMPLETE")
print("===================================")

print("\nSelected parameters:")
print(best_params)

print("\nPerformance summary:")
print(results_df)

print("\nFiles saved successfully.")
