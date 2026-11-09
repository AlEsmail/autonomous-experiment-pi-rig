# ABOUTME: Tests for the scientist __main__ bootstrap: config loading and component wiring.

import json

import pytest


@pytest.fixture
def main_fn():
    """Import target under test; red-phase failure surface."""
    from scientist.__main__ import main
    return main


def test_main_builds_components_from_config(main_fn, monkeypatch, tmp_path):
    """__main__.main() wires httpx.Client, AnthropicAdapter, and Scientist.run from config."""
    config_path = tmp_path / "scientist.json"
    distinctive_base_url = "http://distinctive-scientist.local:9999"
    distinctive_max_tokens = 1234
    config_path.write_text(
        json.dumps(
            {
                "base_url": distinctive_base_url,
                "poll_idle_s": 5.0,
                "poll_active_s": 2.0,
                "request_timeout_s": 10.0,
                "cycle_timeout_s": 900.0,
                "llm_max_tokens": distinctive_max_tokens,
                "llm_retries": 2,
                "dose_cap_hint_ml": 10.0,
                "conv_delta_e": 3.0,
                "conv_hold": 3,
                "conv_epsilon": 0.5,
                "conv_window": 5,
                "conv_min_cycles": 5,
            }
        )
    )

    run_calls = []
    captured = {}

    def fake_run(self):
        run_calls.append(self)
        captured["client_base_url"] = str(self._client.base_url)

    monkeypatch.setattr("scientist.loop.Scientist.run", fake_run)

    def fake_adapter_init(self, client=None, max_tokens=2000):
        captured["adapter_max_tokens"] = max_tokens

    monkeypatch.setattr(
        "scientist.adapter.AnthropicAdapter.__init__", fake_adapter_init
    )
    monkeypatch.setenv("SCIENTIST_CONFIG", str(config_path))

    main_fn()

    assert captured["client_base_url"] == distinctive_base_url
    assert captured["adapter_max_tokens"] == distinctive_max_tokens
    assert len(run_calls) == 1
