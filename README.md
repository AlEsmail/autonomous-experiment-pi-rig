# autonomous-experiment-pi-rig

Control software for a colour-mixing rig that uses a large language model as its optimiser. Given a target colour, the rig doses two dyes into a cup of water, photographs it, measures the colour in CIELAB, and asks the model for the next pair of doses. The objective is to match the target with as little dye as possible: `score = ΔE(measured, target) + λ · ml dye`, lower is better. Built in July 2026 as a class project (Stochastic Studio).

## How it works

**Hardware.** A Raspberry Pi 5 drives a relay board that switches 12 V to four peristaltic pumps: dye A, dye B, water refill and drain. A Logitech BRIO webcam looks into the cup, with exposure, white balance and focus locked over v4l2.

**One cycle.** dose → refill → settle → measure → drain → rinse + blank → drain. Measure averages an operator-set patch of the frame, converts it to CIE LAB and computes ΔE (CIE76) against the target. The blank reading of the rinsed cup detects camera drift.

**Components.**
- `rig_service` (FastAPI): the only process that touches hardware. Owns relays, camera, safety layer, modes, the SQLite log of every cycle, decision and operator action, and the video stream.
- `scientist`: the LLM loop. Builds a prompt from the target, λ and the full experiment table, gets the next doses, posts the cycle, and stops on a cycle budget or convergence rule. It uses the Anthropic Python SDK; on the rig it ran against an Anthropic-compatible endpoint serving Moonshot's Kimi models.
- `console/`: static `/operate` (operator) and `/wall` (audience) views.

**Safety interlocks.** Relays forced off at boot; caps on dose size and relay on-time; a cup-volume ceiling that rejects rather than clamps; reservoir and waste budgets that pause the rig before a vessel runs dry or overflows; per-phase watchdog timeouts; a camera-lock check every cycle; service-enforced modes (`auto`, `manual`, `paused`) so one client commands doses at a time; volume accounting persisted so a crash leaves it conservative.

**Hidden disruptions.** The operator can secretly skew a pump or add water to the cup. The model only ever sees requested doses, so it has to infer a disruption from the colour readings.

**Remote operation.** Console and SSH run over Tailscale; nothing is exposed to the public internet.

## How it was built

Spec first: `docs/specs/control-software.md` was the first commit and fixes the API, invariants and acceptance criteria. Test first: code under `src/` was written against failing tests (the console is exempt). The suite has 703 tests, including Hypothesis property tests, and runs against fake drivers and a simulated colour world.

The code was written by AI coding agents (Claude Code and Moonshot's Kimi CLI) in a development pipeline I directed: I agreed the spec; each larger feature went through a written plan, agents wrote the failing tests and then the implementation, and reviewer agents checked the result before merge. Wiring, calibration and bench testing were done by hand.

## Repository layout

```
src/rig_service/   hardware service: API, cycle engine, safety, store, drivers
src/scientist/     LLM loop, prompt, adapter, convergence
src/shared/        colour, dose and measurement maths
src/sim/           simulated colour world for off-rig tests
tests/             mirrors src/
console/           operator and wall views
config/            channel map, caps, volumes, camera lock
deploy/            systemd units and Pi deploy notes
docs/specs/        the control-software spec
```

## Running the tests

Python 3.11+. No Pi, camera or API key needed.

```
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # Windows: .venv\Scripts\pip
.venv/bin/python -m pytest
```
