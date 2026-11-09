# ABOUTME: Tests for scientist prompt building and proposal parsing.
# ABOUTME: Validates prompt content and JSON reply extraction/validation.

import json

import pytest

from scientist.prompt import Proposal, ProposalParseError, build_prompt, parse_proposal
from scientist.settings import ScientistConfig


def make_cfg(**overrides):
    defaults = {
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
    defaults.update(overrides)
    return ScientistConfig(**defaults)


def _status(cycles_used=2):
    return {
        "target_hex": "#ff0000",
        "run": {"target_hex": "#ff0000", "cycle_budget": 30, "cycles": cycles_used},
        "lambda": 0.05,
    }


def _cycles():
    return [
        {
            "n": 1,
            "dose_a_ml": 1.5,
            "dose_b_ml": 2.5,
            "measured_lab": (50.0, 10.0, 20.0),
            "blank_lab": (60.0, 5.0, 5.0),
            "delta_e": 4.5,
            "lambda": 0.05,
            "score": 5.25,
            "aborted": None,
        },
        {
            "n": 2,
            "dose_a_ml": 2.0,
            "dose_b_ml": 1.0,
            "measured_lab": (45.0, 15.0, 10.0),
            "blank_lab": (61.0, 4.0, 6.0),
            "delta_e": 3.5,
            "lambda": 0.05,
            "score": 4.75,
            "aborted": "settle timeout",
        },
    ]


def test_prompt_contains_target_lambda_budget_and_cap_hint():
    cfg = make_cfg(dose_cap_hint_ml=10.0)
    status = _status()
    history = {"cycles": [], "events": []}
    prompt = build_prompt(status, history, cfg, [])
    assert status["run"]["target_hex"] in prompt
    assert "0.05" in prompt
    assert "30" in prompt
    assert "10.0" in prompt


def test_prompt_uses_live_top_level_target_not_stale_run_target():
    cfg = make_cfg()
    status = {
        "target_hex": "#00ff00",
        "run": {"target_hex": "#ff0000", "cycle_budget": 30, "cycles": 2},
        "lambda": 0.05,
    }
    history = {"cycles": [], "events": []}
    prompt = build_prompt(status, history, cfg, [])
    assert "#00ff00" in prompt
    assert "#ff0000" not in prompt


def test_prompt_experiment_table_lists_every_cycle_field():
    cfg = make_cfg()
    cycles = _cycles()
    history = {"cycles": cycles, "events": []}
    prompt = build_prompt(_status(), history, cfg, [])
    for field in [
        "n",
        "dose_a_ml",
        "dose_b_ml",
        "measured_lab",
        "blank_lab",
        "delta_e",
        "lambda",
        "score",
        "aborted",
    ]:
        assert field in prompt
    for cycle in cycles:
        assert str(cycle["n"]) in prompt
        assert str(cycle["dose_a_ml"]) in prompt
        assert str(cycle["dose_b_ml"]) in prompt
        assert str(cycle["measured_lab"]) in prompt
        assert str(cycle["blank_lab"]) in prompt
        assert str(cycle["aborted"]) in prompt or "None" in prompt


def test_prompt_blank_drift_against_first_blank():
    from shared.color import delta_e

    cfg = make_cfg()
    blank1 = (60.0, 5.0, 5.0)
    blank2 = (61.0, 4.0, 6.0)
    cycles = [
        {
            "n": 1,
            "dose_a_ml": 1.5,
            "dose_b_ml": 2.5,
            "measured_lab": (50.0, 10.0, 20.0),
            "blank_lab": blank1,
            "delta_e": 4.5,
            "lambda": 0.05,
            "score": 5.25,
            "aborted": None,
        },
        {
            "n": 2,
            "dose_a_ml": 2.0,
            "dose_b_ml": 1.0,
            "measured_lab": (45.0, 15.0, 10.0),
            "blank_lab": blank2,
            "delta_e": 3.5,
            "lambda": 0.05,
            "score": 4.75,
            "aborted": None,
        },
    ]
    history = {"cycles": cycles, "events": []}
    prompt = build_prompt(_status(), history, cfg, [])
    expected_drift = delta_e(blank2, blank1)
    assert f"{expected_drift:.2f}" in prompt


def test_prompt_includes_visible_events_and_notes():
    cfg = make_cfg()
    # Real event kinds/detail shapes as emitted by rig_service.app._add_event
    # (mode_change: {"mode": ...}, target_change: {"hex": ...}).
    events = [
        {"kind": "mode_change", "detail": {"mode": "auto"}},
        {"kind": "target_change", "detail": {"hex": "#ff0000"}},
    ]
    notes = ["previous proposal over cap", "retrying with smaller doses"]
    history = {"cycles": [], "events": events}
    prompt = build_prompt(_status(), history, cfg, notes)
    assert "mode_change" in prompt
    assert "auto" in prompt
    assert "target_change" in prompt
    assert "#ff0000" in prompt
    for note in notes:
        assert note in prompt


def test_prompt_states_json_reply_contract():
    cfg = make_cfg()
    prompt = build_prompt(_status(), {"cycles": [], "events": []}, cfg, [])
    assert '"dose_a_ml"' in prompt
    assert '"dose_b_ml"' in prompt
    assert '"rationale"' in prompt


def test_parse_plain_json():
    text = '{"dose_a_ml": 1.5, "dose_b_ml": 2.5, "rationale": "try more A"}'
    proposal = parse_proposal(text)
    assert proposal == Proposal(dose_a_ml=1.5, dose_b_ml=2.5, rationale="try more A")


def test_parse_json_in_fences_and_prose():
    text = (
        "Here is my proposal:\n"
        "```json\n"
        '{"dose_a_ml": 1.0, "dose_b_ml": 2.0, "rationale": "first balanced block"}\n'
        "```\n"
        "Hope this helps."
    )
    proposal = parse_proposal(text)
    assert proposal == Proposal(
        dose_a_ml=1.0, dose_b_ml=2.0, rationale="first balanced block"
    )


def test_parse_rejects_negative_dose():
    with pytest.raises(ProposalParseError, match="dose_a_ml"):
        parse_proposal('{"dose_a_ml": -1.0, "dose_b_ml": 2.0, "rationale": "x"}')


@pytest.mark.parametrize("field", ["dose_a_ml", "dose_b_ml", "rationale"])
def test_parse_rejects_missing_field(field):
    data = {"dose_a_ml": 1.0, "dose_b_ml": 2.0, "rationale": "x"}
    del data[field]
    with pytest.raises(ProposalParseError, match=field):
        parse_proposal(json.dumps(data))


@pytest.mark.parametrize("bad_dose", ["1.0", True])
def test_parse_rejects_non_numeric_dose(bad_dose):
    data = {"dose_a_ml": bad_dose, "dose_b_ml": 2.0, "rationale": "x"}
    with pytest.raises(ProposalParseError, match="dose_a_ml"):
        parse_proposal(json.dumps(data))


def test_parse_rejects_empty_rationale():
    with pytest.raises(ProposalParseError, match="rationale"):
        parse_proposal('{"dose_a_ml": 1.0, "dose_b_ml": 2.0, "rationale": ""}')


def test_parse_rejects_no_json_at_all():
    with pytest.raises(ProposalParseError):
        parse_proposal("This response contains no JSON object at all.")
