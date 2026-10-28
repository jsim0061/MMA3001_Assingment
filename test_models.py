"""Smoke tests for all four modelling approaches and baseline fallbacks."""
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from occupancy_utils import (FEATURES, TARGET, chronological_split,
    historical_average_fit, historical_average_predict)

@pytest.mark.parametrize('model', [
    DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=42),
    LinearRegression(),
    make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1, epsilon=.1)),
])
def test_regression_predictors_return_finite_outputs(sample_frame, model):
    train, validation, _ = chronological_split(sample_frame)
    model.fit(train[list(FEATURES)], train[TARGET])
    prediction = model.predict(validation[list(FEATURES)])
    assert prediction.shape == (len(validation),)
    assert np.isfinite(prediction).all()


def test_historical_average_training_only(sample_frame):
    train, validation, _ = chronological_split(sample_frame)
    fit = historical_average_fit(train)
    before = historical_average_predict(fit, validation)
    modified = validation.copy()
    modified[TARGET] = 100000  # A future target must not affect existing predictions.
    after = historical_average_predict(fit, modified)
    np.testing.assert_allclose(before, after)


def test_historical_average_unknown_slot_uses_fallback(sample_frame):
    train, _, _ = chronological_split(sample_frame)
    fit = historical_average_fit(train)
    unseen = pd.DataFrame({'target_time': [pd.Timestamp('2024-06-09 23:45:00', tz='UTC')]})
    pred = historical_average_predict(fit, unseen)
    assert len(pred) == 1 and np.isfinite(pred[0])


def test_tree_generalises_to_unseen_room_without_refitting(sample_frame):
    train, validation, _ = chronological_split(sample_frame)
    tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree.fit(train[list(FEATURES)], train[TARGET])
    before_nodes = tree.tree_.node_count
    other_room = validation.copy()
    other_room['current_occupancy'] += 3
    pred = tree.predict(other_room[list(FEATURES)])
    assert np.isfinite(pred).all()
    assert tree.tree_.node_count == before_nodes
