# ABOUTME: Tests for shared dose math: converting a requested ml volume into pump-on seconds
# ABOUTME: using the per-pump calibration model {rate_ml_per_s, lag_s}.

import pytest

from shared.dose import seconds_for_dose


def test_dose_is_lag_plus_volume_over_rate():
    assert seconds_for_dose(ml=10, rate_ml_per_s=2.0, lag_s=0.5) == pytest.approx(5.5)


def test_zero_lag():
    assert seconds_for_dose(ml=4, rate_ml_per_s=1.0, lag_s=0.0) == pytest.approx(4.0)


def test_rejects_non_positive_ml():
    with pytest.raises(ValueError):
        seconds_for_dose(ml=0, rate_ml_per_s=1.0, lag_s=0.5)
    with pytest.raises(ValueError):
        seconds_for_dose(ml=-1, rate_ml_per_s=1.0, lag_s=0.5)


def test_rejects_non_positive_rate():
    with pytest.raises(ValueError):
        seconds_for_dose(ml=1, rate_ml_per_s=0.0, lag_s=0.5)


def test_rejects_negative_lag():
    with pytest.raises(ValueError):
        seconds_for_dose(ml=1, rate_ml_per_s=1.0, lag_s=-0.1)
