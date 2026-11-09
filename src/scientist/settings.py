# ABOUTME: Frozen dataclass and loader for the scientist client's runtime configuration,
# ABOUTME: including polling intervals, HTTP timeouts, LLM limits, and convergence thresholds.

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ScientistConfig:
    base_url: str                # "http://127.0.0.1:8000"
    poll_idle_s: float           # 5.0
    poll_active_s: float         # 2.0
    request_timeout_s: float     # 10.0   (GET /status, GET /history)
    cycle_timeout_s: float       # 900.0  (POST /cycle; phase budgets sum to 780 s)
    llm_max_tokens: int          # 2000
    llm_retries: int             # 2      (malformed-reply re-asks per tick)
    dose_cap_hint_ml: float      # 10.0   (prompt text only; rig-service enforces real caps)
    conv_delta_e: float          # 3.0    -- all five conv_* are pre-gamut-sweep placeholders
    conv_hold: int               # 3
    conv_epsilon: float          # 0.5
    conv_window: int             # 5
    conv_min_cycles: int         # 5


def load_config(path: str | Path) -> ScientistConfig:
    """JSON file (config/scientist.json ships the defaults above with a _note naming the
    gamut-sweep tuning debt). Missing key or non-positive timeout/poll/window/hold ->
    ValueError naming the key."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    fields = {f.name for f in ScientistConfig.__dataclass_fields__.values()}
    for key in fields:
        if key not in data:
            raise ValueError(f"missing config key: {key}")

    positive_keys = [
        "poll_idle_s",
        "poll_active_s",
        "request_timeout_s",
        "cycle_timeout_s",
        "conv_hold",
        "conv_window",
    ]
    for key in positive_keys:
        value = data[key]
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"config key {key} must be positive, got {value!r}")

    kwargs = {key: data[key] for key in fields}
    return ScientistConfig(**kwargs)
