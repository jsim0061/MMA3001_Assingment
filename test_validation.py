"""Unit tests for engineering data constraints, splitting and metrics."""
import numpy as np
import pandas as pd
import pytest
from occupancy_utils import (validate_frame, chronological_split,
                             evaluate_regression, measured_target_mask)


def test_accepts_valid_data(sample_frame):
    assert len(validate_frame(sample_frame)) == 120


def test_missing_feature_rejected(sample_frame):
    with pytest.raises(ValueError, match='Missing required columns'):
        validate_frame(sample_frame.drop(columns=['previous_15min']))


def test_negative_occupancy_rejected(sample_frame):
    sample_frame.loc[0, 'current_occupancy'] = -1
    with pytest.raises(ValueError, match='negative'):
        validate_frame(sample_frame)


def test_invalid_input_rejected(sample_frame):
    sample_frame.loc[0, 'hour'] = np.nan
    with pytest.raises(ValueError, match='finite'):
        validate_frame(sample_frame)


def test_target_is_15_minutes_ahead(sample_frame):
    validate_frame(sample_frame)
    sample_frame.loc[0, 'target_time'] += pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match='15 minutes'):
        validate_frame(sample_frame)


def test_chronological_no_boundary_target_leakage(sample_frame):
    train, validation, test = chronological_split(sample_frame.sample(frac=1, random_state=2))
    assert train.collecteddate.is_monotonic_increasing
    assert train.target_time.max() < validation.collecteddate.min()
    assert validation.target_time.max() < test.collecteddate.min()
    assert len(train) > len(validation) > 0 and len(test) > 0


def test_metrics_perfect_prediction():
    scores = evaluate_regression([1,2,3], [1,2,3])
    assert scores['MAE'] == 0
    assert scores['RMSE'] == 0
    assert scores['R2'] == 1


def test_metrics_known_prediction():
    scores = evaluate_regression([2,4,6], [3,4,5])
    assert scores['MAE'] == pytest.approx(2/3)
    assert scores['RMSE'] == pytest.approx(np.sqrt(2/3))


def test_measured_flags_parse_strings(sample_frame):
    sample_frame['target_is_measured'] = ['True', 'false'] * 60
    flags = measured_target_mask(sample_frame)
    assert flags.dtype == bool and flags.sum() == 60
    assert flags[0] and not flags[1]


def test_invalid_split_fractions(sample_frame):
    with pytest.raises(ValueError):
        chronological_split(sample_frame, .9, .2)
