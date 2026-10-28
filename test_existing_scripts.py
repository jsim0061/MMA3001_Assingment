"""Targeted integration tests for the actual uploaded scripts."""
import zipfile
import numpy as np
import pandas as pd
import pytest
import data_processing
import decision_tree


def test_decision_tree_evaluate_calculates_scores():
    result = decision_tree.evaluate([1, 2, 3], [1, 2, 3], 'unit-test')
    assert result['MAE'] == 0
    assert result['RMSE'] == 0
    assert result['R2'] == 1
    assert result['n'] == 3


def test_decision_tree_evaluate_tiny_dataset_returns_none():
    assert decision_tree.evaluate([1], [1], 'too-short') is None


def test_preprocessing_15min_alignment_and_gap_rule(tmp_path, monkeypatch):
    """Run original preprocessing end-to-end against a tiny synthetic ZIP."""
    zip_path = tmp_path / 'small_occupancy.zip'
    room_id = 'test-room'
    raw = pd.DataFrame({
        'floorspaceid': [room_id, room_id, room_id, 'other-room'],
        'collecteddate': [
            '2024-06-03T00:01:00Z',
            '2024-06-03T00:16:00Z',
            '2024-06-03T02:16:00Z',
            '2024-06-03T00:01:00Z',
        ],
        'headcount': [5, 6, 8, 999],
    })
    with zipfile.ZipFile(zip_path, 'w') as archive:
        archive.writestr('tiny.csv', raw.to_csv(index=False))
    out_path = tmp_path / 'clean.csv'
    monkeypatch.setattr(data_processing, 'ZIP_PATH', zip_path)
    monkeypatch.setattr(data_processing, 'OUTPUT_PATH', out_path)
    monkeypatch.setattr(data_processing, 'ROOM_ID', room_id)
    data_processing.main()
    result = pd.read_csv(out_path)
    assert not result.empty
    assert (result['current_occupancy'] < 999).all()
    assert result['assumed_empty'].astype(str).str.lower().eq('true').any()
    t = pd.to_datetime(result['collecteddate'], utc=True)
    target_t = pd.to_datetime(result['target_time'], utc=True)
    assert ((target_t - t) == pd.Timedelta(minutes=15)).all()
    assert result['target_is_measured'].astype(str).str.lower().eq('true').any()


def test_preprocessing_rejects_missing_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(data_processing, 'ZIP_PATH', tmp_path / 'absent.zip')
    with pytest.raises(FileNotFoundError):
        data_processing.main()
