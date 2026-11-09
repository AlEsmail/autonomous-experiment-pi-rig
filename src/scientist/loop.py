# ABOUTME: Synchronous poll-loop state machine for the scientist LLM client.
# ABOUTME: Talks to rig-service over HTTP; never self-authorizes (no /mode or /run calls).

import logging
import time
from typing import Callable

import httpx

from scientist.adapter import AdapterError, ProviderAdapter
from scientist.convergence import should_stop
from scientist.prompt import ProposalParseError, build_prompt, parse_proposal
from scientist.settings import ScientistConfig


logger = logging.getLogger(__name__)


class Scientist:
    """Synchronous scientist loop: one tick() is one state-machine step.

    The loop is stateless across ticks except for transient prompt notes
    (decision 8). It never calls /mode or /run/* — all actuation authority
    remains in rig-service's own invariant-1 machinery.
    """

    def __init__(
        self,
        client: "httpx.Client",
        adapter: ProviderAdapter,
        cfg: ScientistConfig,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._adapter = adapter
        self._cfg = cfg
        self._sleep = sleep
        self._notes: list[str] = []

    def tick(self) -> str:
        """Single state-machine step; returns the action taken this tick."""
        status = self._fetch_status()
        if status is None:
            self._report("idle")
            return "idle"

        mode = status.get("mode")
        run = status.get("run")
        if mode != "auto" or run is None:
            self._report("idle")
            return "idle"

        history = self._fetch_history(run["id"])
        if history is None:
            self._report("cycle_failed")
            return "cycle_failed"

        verdict = should_stop(
            history.get("cycles", []), run["cycle_budget"], self._cfg
        )
        if verdict.stop:
            if verdict.reason == "budget":
                self._report("stopped_budget", verdict.reason)
                return "stopped_budget"
            self._report("stopped_converged", verdict.reason)
            return "stopped_converged"

        proposal, latency_s = self._propose(status, history, run["model"])
        if isinstance(proposal, str):
            return proposal

        return self._post_cycle(proposal, run["model"], latency_s)

    def _report(self, state: str, detail: str = "") -> None:
        """Best-effort liveness report to rig-service; a failed report NEVER
        changes loop behavior (all exceptions swallowed, logged at info)."""
        try:
            self._client.post(
                "/scientist/state",
                headers={"X-Client": "scientist"},
                timeout=self._cfg.request_timeout_s,
                json={"state": state, "detail": detail},
            )
        except Exception as error:
            logger.info("state report failed: %s", error)

    def _fetch_status(self) -> dict | None:
        try:
            response = self._client.get(
                "/status",
                headers={"X-Client": "scientist"},
                timeout=self._cfg.request_timeout_s,
            )
        except Exception as error:
            logger.debug("GET /status failed: %s", error)
            return None
        if response.status_code != 200:
            return None
        try:
            return response.json()
        except Exception:
            return None

    def _fetch_history(self, run_id: int) -> dict | None:
        try:
            response = self._client.get(
                f"/history?run={run_id}",
                headers={"X-Client": "scientist"},
                timeout=self._cfg.request_timeout_s,
            )
        except Exception as error:
            logger.debug("GET /history failed: %s", error)
            return None
        if response.status_code != 200:
            return None
        try:
            return response.json()
        except Exception:
            return None

    def _propose(
        self, status: dict, history: dict, model: str
    ) -> tuple[object, float]:
        """Return (Proposal, latency_s) or (action_string, 0.0) on failure."""
        retry_notes: list[str] = []
        latency_s = 0.0
        for attempt in range(self._cfg.llm_retries + 1):
            prompt = build_prompt(status, history, self._cfg, self._notes + retry_notes)
            try:
                self._report("thinking")
                start = time.perf_counter()
                reply = self._adapter.propose(prompt, model)
                latency_s = time.perf_counter() - start
            except AdapterError as error:
                logger.info("LLM adapter error: %s", error)
                self._report("llm_error")
                return "llm_failed", 0.0

            try:
                return parse_proposal(reply), latency_s
            except ProposalParseError as error:
                if attempt == self._cfg.llm_retries:
                    self._report("parse_error")
                    return "parse_failed", 0.0
                retry_notes.append(
                    f"Your previous reply was malformed ({error}). "
                    "Reply with exactly one JSON object."
                )

    def _post_cycle(
        self, proposal, model: str, latency_s: float
    ) -> str:
        body = {
            "dose_a_ml": proposal.dose_a_ml,
            "dose_b_ml": proposal.dose_b_ml,
            "rationale": proposal.rationale,
            "model_id": model,
            "latency_s": latency_s,
        }
        self._report("waiting")
        try:
            response = self._client.post(
                "/cycle",
                headers={"X-Client": "scientist"},
                timeout=self._cfg.cycle_timeout_s,
                json=body,
            )
        except Exception as error:
            logger.info("POST /cycle request failed: %s", error)
            self._report("cycle_failed")
            return "cycle_failed"

        if response.status_code == 200:
            self._notes.clear()
            return "proposed"
        if response.status_code == 400:
            detail = self._extract_detail(response)
            if detail:
                self._notes.append(detail)
                self._notes[:] = self._notes[-3:]
            self._report("cycle_refused")
            return "cycle_rejected"
        if response.status_code == 409:
            self._report("cycle_refused")
            return "cycle_refused"
        self._report("cycle_failed")
        return "cycle_failed"

    @staticmethod
    def _extract_detail(response) -> str:
        try:
            return str(response.json().get("detail", ""))
        except Exception:
            return response.text or ""

    def run(self) -> None:
        """Poll forever: tick, then sleep according to the resulting action."""
        while True:
            try:
                action = self.tick()
            except Exception as error:
                logger.exception("Unexpected error in tick: %s", error)
                action = "idle"

            sleep_s = (
                self._cfg.poll_idle_s
                if action in {"idle", "stopped_budget", "stopped_converged"}
                else self._cfg.poll_active_s
            )
            self._sleep(sleep_s)
