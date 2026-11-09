# ABOUTME: Pure dose math: converts a requested volume into pump-on seconds using the
# ABOUTME: per-pump calibration model {rate_ml_per_s, lag_s}.


def seconds_for_dose(ml: float, rate_ml_per_s: float, lag_s: float) -> float:
    if ml <= 0:
        raise ValueError(f"ml must be positive, got {ml}")
    if rate_ml_per_s <= 0:
        raise ValueError(f"rate_ml_per_s must be positive, got {rate_ml_per_s}")
    if lag_s < 0:
        raise ValueError(f"lag_s must be non-negative, got {lag_s}")
    return lag_s + ml / rate_ml_per_s
