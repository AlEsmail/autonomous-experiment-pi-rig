# ABOUTME: Tests for scientist settings loading and validation.
# ABOUTME: Validates ScientistConfig fields and load_config error contract.

import json
from pathlib import Path

import pytest

from scientist.settings import ScientistConfig, load_config


def write_config(tmp_path, data):
    path = tmp_path / "scientist.json"
    path.write_text(json.dumps(data))
    return path


VALID = {
    "base_url": "http://127.0.0.1:8000",
    "poll_idle_s": 5.0,
    "poll_active_s": 2.0,
    "request_timeout_s": 10.0,
    "cycle_timeout_s": 900.0,
    "llm_max_tokens": 2000,
    "llm_retries": 2,
    "dose_cap_hint_ml": 10.0,
    "conv_delta_e": 3.0,
    "conv_hold": 3,
    "conv_epsilon": 0.5,
    "conv_window": 5,
    "conv_min_cycles": 5,
}


def test_load_config_reads_all_fields(tmp_path):
    cfg = load_config(write_config(tmp_path, VALID))
    assert isinstance(cfg, ScientistConfig)
    assert cfg.base_url == "http://127.0.0.1:8000"
    assert cfg.poll_idle_s == 5.0
    assert cfg.poll_active_s == 2.0
    assert cfg.request_timeout_s == 10.0
    assert cfg.cycle_timeout_s == 900.0
    assert cfg.llm_max_tokens == 2000
    assert cfg.llm_retries == 2
    assert cfg.dose_cap_hint_ml == 10.0
    assert cfg.conv_delta_e == 3.0
    assert cfg.conv_hold == 3
    assert cfg.conv_epsilon == 0.5
    assert cfg.conv_window == 5
    assert cfg.conv_min_cycles == 5


@pytest.mark.parametrize("key", list(VALID.keys()))
def test_load_config_missing_key_raises_naming_it(tmp_path, key):
    data = dict(VALID)
    del data[key]
    with pytest.raises(ValueError, match=key):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize(
    "key",
    [
        "poll_idle_s",
        "poll_active_s",
        "request_timeout_s",
        "cycle_timeout_s",
        "conv_hold",
        "conv_window",
    ],
)
@pytest.mark.parametrize("bad_value", [0, -1])
def test_load_config_rejects_nonpositive_timeouts_and_windows(tmp_path, key, bad_value):
    data = dict(VALID)
    data[key] = bad_value
    with pytest.raises(ValueError, match=key):
        load_config(write_config(tmp_path, data))
