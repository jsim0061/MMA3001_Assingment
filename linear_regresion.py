
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
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

# Create target timestamps if missing
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

# Sort chronologically
data = data.sort_values(
    "collecteddate"
).reset_index(drop=True)

# ==========================================
# STEP 2: PREPARE INPUT VARIABLES
# ==========================================

# Recreate time features consistently
local_time = data["collecteddate"].dt.tz_convert(
    "Australia/Melbourne"
)

data["hour"] = local_time.dt.hour
data["minute"] = local_time.dt.minute
data["day_of_week"] = local_time.dt.dayofweek

# Same input variables as Decision Tree
features = [
    "current_occupancy",
    "previous_15min",
    "previous_30min",
    "hour",
    "minute",
    "day_of_week"
]

target = "next_15min_occupancy"

# Check that the required columns exist
missing = [
    column for column in features + [target]
    if column not in data.columns
]

if missing:
    raise ValueError(
        f"Missing dataset columns: {missing}. "
        "Run data_processing.py first."
    )

# Remove incomplete observations
data = data.dropna(
    subset=features + [target]
).reset_index(drop=True)

# Check for actual sensor measurement information
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

print(f"\nZero occupancy: {zero_percentage:.2f}%")

# ==========================================
# STEP 3: SPLIT DATA 70% / 15% / 15%
# ==========================================

n = len(data)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train = data.iloc[:train_end].copy()

validation = data.iloc[
    train_end:validation_end
].copy()

test = data.iloc[validation_end:].copy()

# Prevent target leakage between periods
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
# STEP 4: DEFINE INPUTS AND OUTPUTS
# ==========================================

X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]

X_test = test[features]
y_test = test[target]

# ==========================================
# STEP 5: CREATE LINEAR REGRESSION MODEL
# ==========================================

model = LinearRegression()

print("\nTraining Linear Regression...")

model.fit(X_train, y_train)

print("Training complete.")

# ==========================================
# STEP 6: GENERATE PREDICTIONS
# ==========================================

train_predictions = model.predict(X_train)

validation_predictions = model.predict(X_val)

test_predictions = model.predict(X_test)

# ==========================================
# STEP 7: CALCULATE PERFORMANCE
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

results = []

results.append(
    evaluate_model(
        y_train,
        train_predictions,
        "Training"
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
# STEP 8: EVALUATE GENUINE MEASUREMENTS
# ==========================================

if has_measured_targets:

    measured_mask = (
        test["target_is_measured"].to_numpy()
    )

    print("\nMEASURED-TARGET EVALUATION")
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
        print("Insufficient measured targets.")

else:
    print(
        "\nGenuine measurement information "
        "is not available."
    )

# ==========================================
# STEP 9: SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    FOLDER / "linear_regression_results.csv",
    index=False
)

# Save test predictions for direct comparison
prediction_df = pd.DataFrame({
    "target_time": test["target_time"],
    "actual_occupancy": y_test,
    "predicted_occupancy": test_predictions
})

prediction_df.to_csv(
    FOLDER / "linear_regression_predictions.csv",
    index=False
)

# ==========================================
# STEP 10: ACTUAL VS PREDICTED GRAPH
# ==========================================

n_plot = min(150, len(test))

plot_times = test["target_time"].iloc[:n_plot]

plt.figure(figsize=(12, 5))

plt.plot(
    plot_times,
    y_test.iloc[:n_plot].to_numpy(),
    label="Actual Occupancy",
    linewidth=2
)

plt.plot(
    plot_times,
    test_predictions[:n_plot],
    label="Linear Regression Prediction",
    linestyle="--",
    linewidth=2
)

plt.title(
    "Linear Regression: Actual vs Predicted Occupancy"
)

plt.xlabel("Target Date and Time")
plt.ylabel("Occupancy (people)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    FOLDER / "linear_regression_predictions.png",
    dpi=200
)

plt.show()

# ==========================================
# STEP 11: PREDICTION ERROR HISTOGRAM
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

plt.title("Linear Regression Prediction Errors")
plt.xlabel("Actual - Predicted Occupancy")
plt.ylabel("Frequency")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FOLDER / "linear_regression_errors.png",
    dpi=200
)

plt.show()

# ==========================================
# STEP 12: REGRESSION COEFFICIENTS
# ==========================================

coefficients = pd.Series(
    model.coef_,
    index=features
).sort_values()

plt.figure(figsize=(10, 5))

coefficients.plot(
    kind="barh"
)

plt.title("Linear Regression Coefficients")
plt.xlabel("Coefficient Value")

plt.axvline(0, linewidth=1)
plt.tight_layout()

plt.savefig(
    FOLDER / "linear_regression_coefficients.png",
    dpi=200
)

plt.show()

# ==========================================
# STEP 13: PRINT MODEL EQUATION
# ==========================================

print("\nLINEAR REGRESSION EQUATION")
print("-----------------------------")

print(f"Intercept: {model.intercept_:.4f}")

for name, coefficient in zip(
    features, model.coef_
):
    print(
        f"{name}: {coefficient:.4f}"
    )

print("\n===================================")
print("LINEAR REGRESSION COMPLETE")
print("===================================")

print(results_df)
