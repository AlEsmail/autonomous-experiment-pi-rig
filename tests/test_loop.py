# ABOUTME: Tests for the scientist loop state machine (idle/active/resume/stop, /cycle
# ABOUTME: response classification, notes semantics) against the real rig-service app.

import httpx
import pytest
from fastapi.testclient import TestClient

from rig_service.cycle import CycleRecord
from rig_service.drivers.fake import FakeCamera, FakeRelayDriver
from rig_service.volume import VolumeState
from scientist.adapter import AdapterError
from scientist.convergence import should_stop
from scientist.prompt import parse_proposal
from scientist.settings import ScientistConfig
from tests.test_app import (
    CONSOLE,
    SCIENTIST,
    FAST_CAMERA_DEFAULTS,
    _solid_jpeg,
    _app_with_store,
    make_fast_cycle_config,
)


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------


def build_app(tmp_path, cycle_budget=10, model="test-model"):
    """Build a real create_app() instance plus its Store for loop tests."""
    app, store = _app_with_store(
        tmp_path,
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        volumes=VolumeState(0.0, 0.0, 0.0),
        start_run=False,
    )
    return app, store


def _start_run_auto(client, target_hex="#3366cc", cycle_budget=10, model="test-model"):
    """Console action: start a run and flip the arbiter to auto."""
    response = client.post(
        "/run/start",
        json={
            "target_hex": target_hex,
            "lambda": 0.0,
            "cycle_budget": cycle_budget,
            "model": model,
        },
        headers=CONSOLE,
    )
    assert response.status_code == 200
    response = client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
    assert response.status_code == 200


class ScriptedAdapter:
    """Hand-written fake provider adapter: replays queued replies and records calls."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def propose(self, prompt, model):
        self.calls.append((prompt, model))
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class RecordingClient:
    """Thin proxy around a TestClient that logs every (method, path, headers)."""

    def __init__(self, client):
        self._client = client
        self.requests = []

    def get(self, path, **kwargs):
        self.requests.append(("GET", path, kwargs))
        return self._client.get(path, **kwargs)

    def post(self, path, **kwargs):
        self.requests.append(("POST", path, kwargs))
        return self._client.post(path, **kwargs)


class FakeResponse:
    """Minimal response duck-type for status-code classification tests."""

    def __init__(self, status_code, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json


def _make_sleeper(limit):
    """Return a sleep callable that records durations and raises StopIteration at ``limit``."""
    calls = []

    def sleep(duration):
        calls.append(duration)
        if len(calls) >= limit:
            raise StopIteration()

    return sleep, calls


def _seed_completed_cycle(store, run_id, n, delta_e, score):
    """Insert a single completed scientist-view cycle row directly into the store."""
    record = CycleRecord(
        requested_a_ml=1.0,
        requested_b_ml=1.0,
        executed_a_ml=1.0,
        executed_b_ml=1.0,
        lam=0.0,
        target_lab=(50.0, 10.0, -20.0),
        measured_lab=(51.0, 11.0, -19.0),
        blank_lab=(50.5, 10.5, -19.5),
        delta_e=delta_e,
        score=score,
    )
    cycle_id = store.start_cycle(run_id, record, VolumeState(0.0, 0.0, 0.0))
    store.finish_cycle(cycle_id, record, VolumeState(0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scientist_cls():
    """Import target under test; red-phase failure surface."""
    from scientist.loop import Scientist
    return Scientist


@pytest.fixture
def cfg():
    """Distinct poll timings so swapped constants are visible."""
    return ScientistConfig(
        base_url="http://127.0.0.1:8000",
        poll_idle_s=17.0,
        poll_active_s=2.0,
        request_timeout_s=10.0,
        cycle_timeout_s=900.0,
        llm_max_tokens=2000,
        llm_retries=2,
        dose_cap_hint_ml=10.0,
        conv_delta_e=3.0,
        conv_hold=3,
        conv_epsilon=0.5,
        conv_window=5,
        conv_min_cycles=5,
    )


@pytest.fixture
def app_store(tmp_path):
    return build_app(tmp_path)


@pytest.fixture
def client(app_store):
    app, _store = app_store
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def store(app_store):
    _app, store = app_store
    return store


# ---------------------------------------------------------------------------
# Idle branches
# ---------------------------------------------------------------------------


def test_tick_idle_when_mode_manual(scientist_cls, client, cfg):
    client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
    scientist = scientist_cls(client=client, adapter=ScriptedAdapter([]), cfg=cfg)
    assert scientist.tick() == "idle"


def test_tick_idle_when_no_active_run(scientist_cls, client, cfg):
    client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
    scientist = scientist_cls(client=client, adapter=ScriptedAdapter([]), cfg=cfg)
    assert scientist.tick() == "idle"


def test_tick_idle_when_status_unreachable(scientist_cls, client, cfg):
    """A transport error fetching /status is idle, not an exception."""
    class BrokenClient:
        def get(self, *args, **kwargs):
            raise httpx.TransportError("network down")

        def post(self, *args, **kwargs):
            raise httpx.TransportError("network down")

    scientist = scientist_cls(client=BrokenClient(), adapter=ScriptedAdapter([]), cfg=cfg)
    assert scientist.tick() == "idle"


# ---------------------------------------------------------------------------
# Active / propose branches
# ---------------------------------------------------------------------------


def test_tick_active_proposes_and_cycle_runs(scientist_cls, client, cfg):
    model = "claude-distinctive-model-x"
    _start_run_auto(client, cycle_budget=10, model=model)
    reply = (
        '{"dose_a_ml": 2.5, "dose_b_ml": 1.5, "rationale": "distinctive rationale text"}'
    )
    adapter = ScriptedAdapter([reply])
    scientist = scientist_cls(client=client, adapter=adapter, cfg=cfg)

    action = scientist.tick()

    assert action == "proposed"
    assert len(adapter.calls) == 1
    assert adapter.calls[0][1] == model
    status = client.get("/status", headers=CONSOLE).json()
    run_id = status["run"]["id"]
    history = client.get(f"/history?run={run_id}", headers=SCIENTIST).json()
    assert len(history["decisions"]) == 1
    decision = history["decisions"][0]
    assert decision["rationale"] == "distinctive rationale text"
    assert decision["model_id"] == model
    assert isinstance(decision["latency_s"], float)
    assert decision["latency_s"] >= 0.0
    assert len(history["cycles"]) == 1
    cycle = history["cycles"][0]
    assert cycle["dose_a_ml"] == 2.5
    assert cycle["dose_b_ml"] == 1.5


def test_tick_history_unreachable_is_cycle_failed(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    calls = []

    class HistoryTransportError:
        def __init__(self, real_client):
            self._real_client = real_client

        def get(self, path, **kwargs):
            if path.startswith("/history"):
                calls.append(path)
                raise httpx.TransportError("history unreachable")
            return self._real_client.get(path, **kwargs)

        def post(self, path, **kwargs):
            return self._real_client.post(path, **kwargs)

    adapter = ScriptedAdapter([])
    scientist = scientist_cls(
        client=HistoryTransportError(client), adapter=adapter, cfg=cfg
    )

    assert scientist.tick() == "cycle_failed"
    assert adapter.calls == []


def test_tick_sends_x_client_scientist_on_every_request(scientist_cls, client, cfg):
    _start_run_auto(client)
    recording = RecordingClient(client)
    adapter = ScriptedAdapter(
        ['{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "ok"}']
    )
    scientist = scientist_cls(client=recording, adapter=adapter, cfg=cfg)

    scientist.tick()

    assert recording.requests
    for _method, _path, kwargs in recording.requests:
        headers = kwargs.get("headers", {})
        assert headers.get("X-Client") == "scientist", (_method, _path, headers)


# ---------------------------------------------------------------------------
# Stop branches
# ---------------------------------------------------------------------------


def test_tick_stops_at_budget_without_calling_llm(scientist_cls, client, store, cfg):
    cycle_budget = 3
    _start_run_auto(client, cycle_budget=cycle_budget)
    run_id = store.active_run()["id"]
    for n in range(1, cycle_budget + 1):
        _seed_completed_cycle(store, run_id, n, delta_e=1.0, score=10.0)
    adapter = ScriptedAdapter([])
    scientist = scientist_cls(client=client, adapter=adapter, cfg=cfg)

    assert scientist.tick() == "stopped_budget"
    assert adapter.calls == []


def test_tick_stops_when_converged(scientist_cls, client, store, cfg):
    _start_run_auto(client, cycle_budget=10)
    run_id = store.active_run()["id"]
    # Five completed cycles, all under conv_delta_e -> converged_delta_e.
    for n in range(1, 6):
        _seed_completed_cycle(store, run_id, n, delta_e=1.0, score=10.0 - n)
    adapter = ScriptedAdapter([])
    scientist = scientist_cls(client=client, adapter=adapter, cfg=cfg)

    action = scientist.tick()

    assert action == "stopped_converged"
    assert adapter.calls == []


# ---------------------------------------------------------------------------
# Resume / restart
# ---------------------------------------------------------------------------


def test_resume_after_manual_roundtrip_no_duplicate_no_skip(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    adapter = ScriptedAdapter([
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "first"}',
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "second"}',
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "third"}',
    ])
    scientist = scientist_cls(client=client, adapter=adapter, cfg=cfg)

    assert scientist.tick() == "proposed"
    assert scientist.tick() == "proposed"

    client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
    assert scientist.tick() == "idle"

    client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
    assert scientist.tick() == "proposed"

    status = client.get("/status", headers=CONSOLE).json()
    run_id = status["run"]["id"]
    history = client.get(f"/history?run={run_id}", headers=CONSOLE).json()
    assert len(history["cycles"]) == 3
    assert history["cycles"][-1]["n"] == 3


def test_restart_rederives_state(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    first_adapter = ScriptedAdapter([
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "first"}',
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "second"}',
    ])
    first = scientist_cls(client=client, adapter=first_adapter, cfg=cfg)
    assert first.tick() == "proposed"
    assert first.tick() == "proposed"

    second_adapter = ScriptedAdapter(
        ['{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "third"}']
    )
    second = scientist_cls(client=client, adapter=second_adapter, cfg=cfg)
    assert second.tick() == "proposed"

    status = client.get("/status", headers=CONSOLE).json()
    run_id = status["run"]["id"]
    history = client.get(f"/history?run={run_id}", headers=CONSOLE).json()
    assert len(history["cycles"]) == 3
    assert history["cycles"][-1]["n"] == 3


# ---------------------------------------------------------------------------
# /cycle response classification
# ---------------------------------------------------------------------------


def test_tick_cycle_400_appends_note_and_continues(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    adapter = ScriptedAdapter([
        '{"dose_a_ml": 6.0, "dose_b_ml": 1.0, "rationale": "over cap"}',
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "ok"}',
    ])
    scientist = scientist_cls(client=client, adapter=adapter, cfg=cfg)

    assert scientist.tick() == "cycle_rejected"
    assert len(adapter.calls) == 1

    assert scientist.tick() == "proposed"
    assert len(adapter.calls) == 2
    second_prompt = adapter.calls[1][0]
    # The 400 detail from the real /cycle route must appear in the next prompt.
    assert "cap" in second_prompt.lower()


def test_tick_cycle_409_leaves_notes_untouched(scientist_cls, client, cfg):
    """Mode flips to manual between GET /status and POST /cycle -> 409, notes unchanged."""
    _start_run_auto(client, cycle_budget=10)

    class FlipManualThenPost:
        def __init__(self, real_client):
            self._real_client = real_client
            self._flipped = False

        def get(self, path, **kwargs):
            return self._real_client.get(path, **kwargs)

        def post(self, path, **kwargs):
            if path == "/cycle" and not self._flipped:
                self._flipped = True
                self._real_client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
            return self._real_client.post(path, **kwargs)

    adapter = ScriptedAdapter([
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "attempt"}',
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "retry"}',
    ])
    scientist = scientist_cls(
        client=FlipManualThenPost(client), adapter=adapter, cfg=cfg
    )

    assert scientist.tick() == "cycle_refused"

    # Next tick (back in auto) must not carry a 409-derived note.
    client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
    assert scientist.tick() == "proposed"
    prompt = adapter.calls[1][0]
    assert "Notes:" not in prompt


def test_tick_cycle_timeout_no_retry(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    calls = []

    class TimeoutOnce:
        def __init__(self, real_client):
            self._real_client = real_client

        def get(self, path, **kwargs):
            return self._real_client.get(path, **kwargs)

        def post(self, path, **kwargs):
            if path == "/cycle":
                calls.append(path)
                if len(calls) == 1:
                    raise httpx.TimeoutException("cycle timed out")
            return self._real_client.post(path, **kwargs)

    adapter = ScriptedAdapter([
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "try"}',
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "retry"}',
    ])
    scientist = scientist_cls(client=TimeoutOnce(client), adapter=adapter, cfg=cfg)

    assert scientist.tick() == "cycle_failed"
    assert len(calls) == 1

    # A subsequent tick must still propose with no failure note in the prompt.
    assert scientist.tick() == "proposed"
    prompt = adapter.calls[1][0]
    assert "Notes:" not in prompt


def test_tick_cycle_transport_error_no_retry(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    calls = []

    class TransportErrorOnce:
        def __init__(self, real_client):
            self._real_client = real_client

        def get(self, path, **kwargs):
            return self._real_client.get(path, **kwargs)

        def post(self, path, **kwargs):
            if path == "/cycle":
                calls.append(path)
                if len(calls) == 1:
                    raise httpx.TransportError("connection lost")
            return self._real_client.post(path, **kwargs)

    adapter = ScriptedAdapter([
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "try"}',
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "retry"}',
    ])
    scientist = scientist_cls(client=TransportErrorOnce(client), adapter=adapter, cfg=cfg)

    assert scientist.tick() == "cycle_failed"
    assert len(calls) == 1

    # A subsequent tick must still propose with no failure note in the prompt.
    assert scientist.tick() == "proposed"
    prompt = adapter.calls[1][0]
    assert "Notes:" not in prompt


def test_tick_cycle_5xx_is_transient(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)

    class ServerErrorOnCycle:
        def __init__(self, real_client):
            self._real_client = real_client
            self._fired = False

        def get(self, path, **kwargs):
            return self._real_client.get(path, **kwargs)

        def post(self, path, **kwargs):
            if path == "/cycle" and not self._fired:
                self._fired = True
                return FakeResponse(500, text="internal server error")
            return self._real_client.post(path, **kwargs)

    adapter = ScriptedAdapter([
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "try"}',
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "retry"}',
    ])
    scientist = scientist_cls(client=ServerErrorOnCycle(client), adapter=adapter, cfg=cfg)

    assert scientist.tick() == "cycle_failed"

    # A subsequent tick must still propose with no failure note in the prompt.
    assert scientist.tick() == "proposed"
    prompt = adapter.calls[1][0]
    assert "Notes:" not in prompt


# ---------------------------------------------------------------------------
# LLM / adapter failure branches
# ---------------------------------------------------------------------------


def test_tick_adapter_error_is_transient(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    adapter = ScriptedAdapter([
        AdapterError("model down"),
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "ok"}',
    ])
    scientist = scientist_cls(client=client, adapter=adapter, cfg=cfg)

    assert scientist.tick() == "llm_failed"
    assert scientist.tick() == "proposed"

    # The success prompt must not carry an adapter-error note.
    prompt = adapter.calls[1][0]
    assert "Notes:" not in prompt


def test_tick_reasks_on_malformed_then_succeeds(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    adapter = ScriptedAdapter([
        "not json at all",
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "ok"}',
    ])
    scientist = scientist_cls(client=client, adapter=adapter, cfg=cfg)

    assert scientist.tick() == "proposed"
    assert len(adapter.calls) == 2
    second_prompt = adapter.calls[1][0]
    assert "parse" in second_prompt.lower() or "json" in second_prompt.lower()


def test_tick_parse_failed_after_retries_exhausted(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    # llm_retries=2 -> initial ask + 2 retries = 3 calls.
    adapter = ScriptedAdapter(["garbage"] * (cfg.llm_retries + 1))
    scientist = scientist_cls(client=client, adapter=adapter, cfg=cfg)

    assert scientist.tick() == "parse_failed"
    assert len(adapter.calls) == cfg.llm_retries + 1


# ---------------------------------------------------------------------------
# Prompt notes
# ---------------------------------------------------------------------------


def test_notes_capped_at_three(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    adapter = ScriptedAdapter([
        '{"dose_a_ml": 6.0, "dose_b_ml": 1.0, "rationale": "alpha"}',
        '{"dose_a_ml": 7.0, "dose_b_ml": 1.0, "rationale": "bravo"}',
        '{"dose_a_ml": 8.0, "dose_b_ml": 1.0, "rationale": "charlie"}',
        '{"dose_a_ml": 9.0, "dose_b_ml": 1.0, "rationale": "delta"}',
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "clean"}',
    ])
    scientist = scientist_cls(client=client, adapter=adapter, cfg=cfg)

    for _ in range(4):
        assert scientist.tick() == "cycle_rejected"

    # Fifth tick: prompt must contain the last three rejection notes and evict the
    # oldest (the 6.0 ml rejection).
    scientist.tick()
    prompt = adapter.calls[-1][0]
    assert "7.0 ml exceeds" in prompt
    assert "8.0 ml exceeds" in prompt
    assert "9.0 ml exceeds" in prompt
    assert "6.0 ml exceeds" not in prompt


def test_notes_cleared_on_proposed(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)

    adapter = ScriptedAdapter([
        '{"dose_a_ml": 6.0, "dose_b_ml": 1.0, "rationale": "over cap"}',
        '{"dose_a_ml": 6.0, "dose_b_ml": 1.0, "rationale": "over cap again"}',
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "clean"}',
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "still clean"}',
    ])
    scientist = scientist_cls(client=client, adapter=adapter, cfg=cfg)

    # Tick 1: 400 rejection appends the server's cap-rejection detail as a note.
    assert scientist.tick() == "cycle_rejected"

    # Tick 2: the note appears in the prompt before another rejection.
    assert scientist.tick() == "cycle_rejected"
    assert "6.0 ml exceeds" in adapter.calls[1][0]

    # Tick 3: a successful proposal clears notes.
    assert scientist.tick() == "proposed"

    # Tick 4: the transient rejection note is gone (the permanent "Dose cap hint" line
    # is unrelated housekeeping text and stays present in every prompt by design).
    assert scientist.tick() == "proposed"
    prompt = adapter.calls[3][0]
    assert "6.0 ml exceeds" not in prompt


# ---------------------------------------------------------------------------
# run() scheduling and exception resilience
# ---------------------------------------------------------------------------


def test_run_sleeps_idle_after_idle_and_stopped_active_after_proposed(scientist_cls, cfg):
    class FakeClient:
        def get(self, *args, **kwargs):
            return None

        def post(self, *args, **kwargs):
            return None

    sleep, calls = _make_sleeper(3)
    scientist = scientist_cls(
        client=FakeClient(), adapter=ScriptedAdapter([]), cfg=cfg, sleep=sleep
    )
    actions = ["idle", "stopped_budget", "proposed"]
    scientist.tick = lambda: actions.pop(0)

    with pytest.raises(StopIteration):
        scientist.run()

    assert calls == [cfg.poll_idle_s, cfg.poll_idle_s, cfg.poll_active_s]


def test_run_sleeps_idle_after_stopped_converged(scientist_cls, cfg):
    class FakeClient:
        def get(self, *args, **kwargs):
            return None

        def post(self, *args, **kwargs):
            return None

    sleep, calls = _make_sleeper(2)
    scientist = scientist_cls(
        client=FakeClient(), adapter=ScriptedAdapter([]), cfg=cfg, sleep=sleep
    )
    actions = ["stopped_converged", "proposed"]
    scientist.tick = lambda: actions.pop(0)

    with pytest.raises(StopIteration):
        scientist.run()

    assert calls == [cfg.poll_idle_s, cfg.poll_active_s]


def test_run_survives_unexpected_tick_exception(scientist_cls, cfg):
    class FakeClient:
        def get(self, *args, **kwargs):
            return None

        def post(self, *args, **kwargs):
            return None

    sleep, calls = _make_sleeper(2)
    scientist = scientist_cls(
        client=FakeClient(), adapter=ScriptedAdapter([]), cfg=cfg, sleep=sleep
    )

    def bad_tick():
        raise RuntimeError("unexpected boom")

    scientist.tick = bad_tick

    with pytest.raises(StopIteration):
        scientist.run()

    assert len(calls) == 2
    assert calls[0] == cfg.poll_idle_s
    assert calls[1] == cfg.poll_idle_s


def test_scientist_never_calls_run_end_or_mode(scientist_cls, client, cfg):
    """The scientist only ever asks: /status, /history, /cycle."""
    _start_run_auto(client, cycle_budget=10)
    recording = RecordingClient(client)
    adapter = ScriptedAdapter([
        '{"dose_a_ml": 6.0, "dose_b_ml": 1.0, "rationale": "rejected"}',
        '{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "ok"}',
    ])
    scientist = scientist_cls(client=recording, adapter=adapter, cfg=cfg)

    # Cover a rejected cycle and a successful cycle while recording every request.
    scientist.tick()
    scientist.tick()

    forbidden = {"/run/start", "/run/end", "/mode"}
    scientist_paths = {path for _method, path, _kwargs in recording.requests}
    assert not (scientist_paths & forbidden), scientist_paths


# ---------------------------------------------------------------------------
# State reporting (POST /scientist/state, best-effort)
# ---------------------------------------------------------------------------


def _reports(recording):
    """(state, detail) pairs of every /scientist/state report a client sent."""
    return [
        (kwargs["json"]["state"], kwargs["json"].get("detail", ""))
        for _method, path, kwargs in recording.requests
        if path == "/scientist/state"
    ]


def test_tick_reports_idle_when_not_auto(scientist_cls, client, cfg):
    client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
    recording = RecordingClient(client)
    scientist = scientist_cls(client=recording, adapter=ScriptedAdapter([]), cfg=cfg)

    assert scientist.tick() == "idle"
    assert _reports(recording) == [("idle", "")]


def test_tick_reports_idle_when_status_unreachable(scientist_cls, cfg):
    posts = []

    class BrokenClient:
        def get(self, *args, **kwargs):
            raise httpx.TransportError("network down")

        def post(self, path, **kwargs):
            posts.append((path, kwargs))
            raise httpx.TransportError("network down")

    scientist = scientist_cls(client=BrokenClient(), adapter=ScriptedAdapter([]), cfg=cfg)
    assert scientist.tick() == "idle"
    reports = [kwargs["json"]["state"] for path, kwargs in posts if path == "/scientist/state"]
    assert reports == ["idle"]


def test_tick_reports_thinking_then_waiting_on_propose(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    recording = RecordingClient(client)
    adapter = ScriptedAdapter(
        ['{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "ok"}']
    )
    scientist = scientist_cls(client=recording, adapter=adapter, cfg=cfg)

    assert scientist.tick() == "proposed"
    assert _reports(recording) == [("thinking", ""), ("waiting", "")]


def test_tick_reports_stopped_budget_with_reason(scientist_cls, client, store, cfg):
    _start_run_auto(client, cycle_budget=3)
    run_id = store.active_run()["id"]
    for n in range(1, 4):
        _seed_completed_cycle(store, run_id, n, delta_e=1.0, score=10.0)
    recording = RecordingClient(client)
    scientist = scientist_cls(client=recording, adapter=ScriptedAdapter([]), cfg=cfg)

    assert scientist.tick() == "stopped_budget"
    assert _reports(recording) == [("stopped_budget", "budget")]


def test_tick_reports_stopped_converged_with_reason(scientist_cls, client, store, cfg):
    _start_run_auto(client, cycle_budget=10)
    run_id = store.active_run()["id"]
    for n in range(1, 6):
        _seed_completed_cycle(store, run_id, n, delta_e=1.0, score=10.0 - n)
    recording = RecordingClient(client)
    scientist = scientist_cls(client=recording, adapter=ScriptedAdapter([]), cfg=cfg)

    assert scientist.tick() == "stopped_converged"
    assert _reports(recording) == [("stopped_converged", "converged_delta_e")]


def test_tick_reports_cycle_failed_when_history_unreachable(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)

    class HistoryTransportError:
        def __init__(self, real_client):
            self._real_client = real_client
            self.requests = []

        def get(self, path, **kwargs):
            if path.startswith("/history"):
                raise httpx.TransportError("history unreachable")
            return self._real_client.get(path, **kwargs)

        def post(self, path, **kwargs):
            self.requests.append(("POST", path, kwargs))
            return self._real_client.post(path, **kwargs)

    recording = HistoryTransportError(client)
    scientist = scientist_cls(client=recording, adapter=ScriptedAdapter([]), cfg=cfg)

    assert scientist.tick() == "cycle_failed"
    assert _reports(recording) == [("cycle_failed", "")]


def test_tick_reports_llm_error_on_adapter_error(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    recording = RecordingClient(client)
    adapter = ScriptedAdapter([AdapterError("model down")])
    scientist = scientist_cls(client=recording, adapter=adapter, cfg=cfg)

    assert scientist.tick() == "llm_failed"
    assert _reports(recording) == [("thinking", ""), ("llm_error", "")]


def test_tick_reports_parse_error_after_retries_exhausted(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    recording = RecordingClient(client)
    adapter = ScriptedAdapter(["garbage"] * (cfg.llm_retries + 1))
    scientist = scientist_cls(client=recording, adapter=adapter, cfg=cfg)

    assert scientist.tick() == "parse_failed"
    assert _reports(recording) == [("thinking", "")] * (cfg.llm_retries + 1) + [
        ("parse_error", "")
    ]


def test_tick_reports_cycle_refused_on_400_rejection(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)
    recording = RecordingClient(client)
    adapter = ScriptedAdapter(
        ['{"dose_a_ml": 6.0, "dose_b_ml": 1.0, "rationale": "over cap"}']
    )
    scientist = scientist_cls(client=recording, adapter=adapter, cfg=cfg)

    assert scientist.tick() == "cycle_rejected"
    assert _reports(recording) == [("thinking", ""), ("waiting", ""), ("cycle_refused", "")]


def test_tick_reports_cycle_refused_on_409(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)

    class FlipManualThenPost:
        def __init__(self, real_client):
            self._real_client = real_client
            self.requests = []
            self._flipped = False

        def get(self, path, **kwargs):
            return self._real_client.get(path, **kwargs)

        def post(self, path, **kwargs):
            self.requests.append(("POST", path, kwargs))
            if path == "/cycle" and not self._flipped:
                self._flipped = True
                self._real_client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
            return self._real_client.post(path, **kwargs)

    recording = FlipManualThenPost(client)
    adapter = ScriptedAdapter(
        ['{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "attempt"}']
    )
    scientist = scientist_cls(client=recording, adapter=adapter, cfg=cfg)

    assert scientist.tick() == "cycle_refused"
    assert _reports(recording) == [("thinking", ""), ("waiting", ""), ("cycle_refused", "")]


def test_tick_reports_cycle_failed_on_cycle_transport_error(scientist_cls, client, cfg):
    _start_run_auto(client, cycle_budget=10)

    class TransportErrorOnCycle:
        def __init__(self, real_client):
            self._real_client = real_client
            self.requests = []

        def get(self, path, **kwargs):
            return self._real_client.get(path, **kwargs)

        def post(self, path, **kwargs):
            self.requests.append(("POST", path, kwargs))
            if path == "/cycle":
                raise httpx.TransportError("connection lost")
            return self._real_client.post(path, **kwargs)

    recording = TransportErrorOnCycle(client)
    adapter = ScriptedAdapter(
        ['{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "try"}']
    )
    scientist = scientist_cls(client=recording, adapter=adapter, cfg=cfg)

    assert scientist.tick() == "cycle_failed"
    assert _reports(recording) == [("thinking", ""), ("waiting", ""), ("cycle_failed", "")]


def test_failed_report_never_changes_tick_outcome(scientist_cls, client, cfg):
    """A raising report POST is swallowed: the tick still proposes and cycles."""
    _start_run_auto(client, cycle_budget=10)

    class ReportRaises:
        def __init__(self, real_client):
            self._real_client = real_client

        def get(self, path, **kwargs):
            return self._real_client.get(path, **kwargs)

        def post(self, path, **kwargs):
            if path == "/scientist/state":
                raise httpx.TransportError("report endpoint down")
            return self._real_client.post(path, **kwargs)

    adapter = ScriptedAdapter(
        ['{"dose_a_ml": 1.0, "dose_b_ml": 1.0, "rationale": "ok"}']
    )
    scientist = scientist_cls(client=ReportRaises(client), adapter=adapter, cfg=cfg)

    assert scientist.tick() == "proposed"
    assert len(adapter.calls) == 1
    status = client.get("/status", headers=CONSOLE).json()
    history = client.get(f"/history?run={status['run']['id']}", headers=SCIENTIST).json()
    assert len(history["cycles"]) == 1
