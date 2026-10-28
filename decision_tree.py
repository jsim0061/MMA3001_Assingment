
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==============================================
# STEP 1: Load processed dataset
# ==============================================

FOLDER = Path(__file__).resolve().parent
CSV_PATH = FOLDER / "occupancy_15min_cleaned.csv"

data = pd.read_csv(CSV_PATH)

data["collecteddate"] = pd.to_datetime(
    data["collecteddate"],
    utc=True
)

# Create target_time if it is missing
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

# Check for measured target information
if "target_is_measured" in data.columns:
    data["target_is_measured"] = (
        data["target_is_measured"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
else:
    raise ValueError(
        "target_is_measured is missing. "
        "Run the updated data_processing.py first."
    )

data = data.sort_values(
    "collecteddate"
).reset_index(drop=True)

# ==============================================
# STEP 2: Define input and output variables
# ==============================================

features = [
    "current_occupancy",
    "previous_15min",
    "previous_30min",
    "hour",
    "minute",
    "day_of_week"
]

target = "next_15min_occupancy"

print("\nDATASET INFORMATION")
print("----------------------------")
print("Number of observations:", len(data))

print("\nOccupancy statistics:")
print(data["current_occupancy"].describe())

zero_percentage = (
    (data["current_occupancy"] == 0).mean() * 100
)

print(f"\nZero occupancy: {zero_percentage:.2f}%")

# ==============================================
# STEP 3: Split 70% / 15% / 15%
# ==============================================

n = len(data)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

validation_start = data.loc[
    train_end, "collecteddate"
]

test_start = data.loc[
    validation_end, "collecteddate"
]

train = data.iloc[:train_end].copy()

validation = data.iloc[
    train_end:validation_end
].copy()

test = data.iloc[validation_end:].copy()

# Prevent targets from crossing split boundaries
train = train[
    train["target_time"] < validation_start
].copy()

validation = validation[
    validation["target_time"] < test_start
].copy()

print("\nDATA SPLIT")
print("----------------------------")
print("Training:", len(train))
print("Validation:", len(validation))
print("Testing:", len(test))

# ==============================================
# STEP 4: Prepare model variables
# ==============================================

X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]

X_test = test[features]
y_test = test[target]

# ==============================================
# STEP 5: Create Decision Tree Regression
# ==============================================

model = DecisionTreeRegressor(
    max_depth=20,
    min_samples_leaf=250,
    random_state=42
)

print("\nTraining Decision Tree...")

model.fit(X_train, y_train)

print("Training complete.")

# ==============================================
# STEP 6: Generate predictions
# ==============================================

train_predictions = model.predict(X_train)
validation_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)

# ==============================================
# STEP 7: Performance evaluation function
# ==============================================

def evaluate_model(actual, predicted, name):

    mae = mean_absolute_error(
        actual, predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual, predicted
        )
    )

    r2 = r2_score(
        actual, predicted
    )

    print(f"\n{name}")
    print("----------------------------")
    print(f"MAE:  {mae:.3f} occupants")
    print(f"RMSE: {rmse:.3f} occupants")
    print(f"R²:   {r2:.3f}")

    return {
        "Dataset": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

# ==============================================
# STEP 8: Evaluate training and validation
# ==============================================

results = []

results.append(
    evaluate_model(
        y_train,
        train_predictions,
        "Training Results"
    )
)

results.append(
    evaluate_model(
        y_val,
        validation_predictions,
        "Validation Results"
    )
)

# The final test results should only be used
# after model settings have been selected.
# This is the initial evaluation of fixed settings.

results.append(
    evaluate_model(
        y_test,
        test_predictions,
        "Testing Results"
    )
)

# ==============================================
# STEP 9: Validate against genuine measurements
# ==============================================

measured_mask = (
    test["target_is_measured"].to_numpy()
)

print("\nMEASURED TARGET VALIDATION")
print("----------------------------")

print(
    "Genuinely measured test targets:",
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

# ==============================================
# STEP 10: Plot actual vs predicted occupancy
# ==============================================

# Choose a continuous section of the test data
n_plot = min(150, len(test))

plot_data = test.iloc[:n_plot]

# Plot both lines at the TARGET time,
# rather than when the prediction was made.
prediction_times = plot_data["target_time"]

actual_values = y_test.iloc[:n_plot].to_numpy()
predicted_values = test_predictions[:n_plot]

plt.figure(figsize=(13, 6))

plt.plot(
    prediction_times,
    actual_values,
    label="Actual Occupancy",
    color="blue",
    linewidth=2
)

plt.plot(
    prediction_times,
    predicted_values,
    label="Predicted Occupancy",
    color="red",
    linestyle="--",
    linewidth=2
)

plt.title(
    "Decision Tree: 15-Minute Occupancy Prediction"
)

plt.xlabel("Target Date and Time")
plt.ylabel("Number of Occupants")

plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    FOLDER / "actual_vs_predicted.png",
    dpi=200
)

plt.show()

# ==============================================
# STEP 11: Plot prediction errors
# ==============================================

errors = (
    y_test.to_numpy() - test_predictions
)

plt.figure(figsize=(10, 5))

plt.hist(
    errors,
    bins=30,
    edgecolor="black"
)

plt.title("Decision Tree Prediction Errors")
plt.xlabel("Actual - Predicted Occupancy")
plt.ylabel("Frequency")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FOLDER / "prediction_errors.png",
    dpi=200
)

plt.show()

# ==============================================
# STEP 12: Feature importance
# ==============================================

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values()

plt.figure(figsize=(10, 5))

importance.plot(
    kind="barh"
)

plt.title("Decision Tree Feature Importance")
plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    FOLDER / "feature_importance.png",
    dpi=200
)

plt.show()

# ==============================================
# STEP 13: Visualise the regression tree
# ==============================================

plt.figure(figsize=(18, 9))

plot_tree(
    model,
    feature_names=features,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree Regression Structure")

plt.tight_layout()

plt.savefig(
    FOLDER / "tree_structure.png",
    dpi=200
)

plt.show()

# ==============================================
# STEP 14: Save results
# ==============================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    FOLDER / "decision_tree_results.csv",
    index=False
)

print("\n===================================")
print("DECISION TREE COMPLETE")
print("===================================")

print("\nPerformance Summary:")
print(results_df)

print("\nResults and graphs have been saved.")


