"""Small artificial dataset shared by unit tests; no private/raw dataset required."""
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def sample_frame():
    """Return 120 valid 15-minute records with known target alignment."""
    times = pd.date_range('2024-06-03', periods=120, freq='15min', tz='UTC')
    local = times.tz_convert('Australia/Melbourne')
    counts = np.arange(120) % 8
    return pd.DataFrame({
        'collecteddate': times,
        'target_time': times + pd.Timedelta(minutes=15),
        'current_occupancy': counts,
        'previous_15min': np.roll(counts, 1),
        'previous_30min': np.roll(counts, 2),
        'hour': local.hour,
        'minute': local.minute,
        'day_of_week': local.dayofweek,
        'next_15min_occupancy': np.roll(counts, -1),
        'target_is_measured': [True, False] * 60,
    })
