# Pantone Rig Control Software

_Spec agreed 2026-07-13 (brainstorm + Opus 4.8 fresh-eyes review). Implementation and tests
derive from this document; the spec does not document the implementation after the fact.
Format adapted from <name>'s `/spec` skill. Design narrative: vault →
`VSD/Stochastic Studio/Nodes/Control System.md`. **Code freeze: Sunday 2026-07-26 19:00 UK**
(<author> flies to London Monday 07-27); **demo: Wednesday 2026-07-29**, 3-hour science-fair
display, rig in Prague, operator in London. The 07-27/28 gap is for remote rehearsal runs
from London — everything must work without hands on the rig by freeze._

## Purpose

Run the color-mixing optimisation rig fully autonomously and remotely: an LLM ("scientist")
searches the 2D dye-dose space against `score = ΔE(measured, target) + λ·ml dye`, a human
operator supervises and injects disruptions from a web console in London, and every cycle,
decision, and event is logged as the class deliverable (the decision trail). The rig must
survive 3 unattended hours in Prague with only non-expert + one technical human nearby.

## System overview

Three components, all running on the Raspberry Pi 5 (`<pi-hostname>`) in Prague:

1. **rig-service** (Python + FastAPI, systemd unit) — the *only* process that touches
   hardware. Owns relays, camera, calibration table, safety layer, mode state, disruption
   machinery, the SQLite store, the MJPEG stream, and serves the console's static files.
2. **scientist** (Python, systemd unit) — the LLM loop. Talks to rig-service over HTTP on
   localhost *only* (it knows a base URL, nothing else — relocatable to any machine).
   Calls LLM provider APIs through a thin adapter interface.
3. **console** (static HTML/CSS/JS) — three URL-addressable views: `/operate`, `/wall`,
   `/replay`. Browsers in London connect over **Tailscale**; nothing is exposed to the
   public internet. Data via 1–2 s polling of the JSON API (no websockets).

Deploy: `git pull` on the Pi + systemd restart. Development on Windows against fake drivers.

## Modes and authority

Rig-level mode `auto | manual | paused`, owned and **enforced by rig-service** (never a
client-side toggle):

- `auto` — scientist may command cycles; console dose commands are rejected.
- `manual` — console may command cycles/pumps; scientist commands are rejected.
- `paused` — nobody doses; entered by operator or by any watchdog/budget trip.

Clients identify with a client header (`scientist` / `console`) — for arbitration, not
security (single operator inside a private tailnet; no auth beyond the tailnet boundary).

## API contract (rig-service)

| Endpoint | Caller | Notes |
|---|---|---|
| `POST /run/start {target_hex, lambda, cycle_budget, model}` | console | model = per-run LLM selection |
| `POST /run/end` | console | rejected (409) while a cycle is in flight; pause (mode → paused) to abort the cycle first |
| `GET /status` | all | mode, run, cycle phase, health (camera lock, volume budgets, watchdog); the console view also carries the scientist's last reported state (`scientist: {state, detail, age_s}`) |
| `POST /cycle {dose_a_ml, dose_b_ml, rationale?, model_id?, latency_s?}` | scientist (auto) / console (manual) | full cycle: dose → refill → settle → measure → drain → rinse+blank → drain. The optional fields are the proposing client's decision record: when `rationale` is present, rig-service writes the decisions row before the cycle actuates; a command without them (console manual dosing) writes no decision row |
| `POST /pump {channel, ml}` | console (manual only) | bring-up/calibration primitive; caps apply |
| `POST /calibrate {channel, seconds}` | console (manual only) | timed run for the scale protocol |
| `POST /staging {water_remaining_ml, waste_ml}` | console (manual only) | operator declares measured vessel contents; the only reset of the session counters, logged |
| `POST /target {hex}` · `POST /lambda {value}` | console | legitimate mid-run world changes; logged **visibly** |
| `POST /patch {cx, cy, size_px}` | console | sets the measurement region for measure + blank; `cx,cy` normalized [0,1] fractions of frame w/h, `size_px` in px; rejects (400) a center outside [0,1], a size below the floor, or a box not wholly inside the configured frame — never clamps; logged **visibly** (see Measurement region) |
| `POST /disrupt/skew {channel, factor}` | console | secret: executed dose = requested × factor |
| `POST /disrupt/blip {ml_water}` | console | secret one-off perturbation of the current cup |
| `POST /scientist/state {state, detail}` | scientist | liveness report, sent each tick; held in memory only (restart = unknown until the next report) and surfaced in `/status`. `state` ∈ `idle | thinking | waiting | stopped_budget | stopped_converged | llm_error | parse_error | cycle_refused | cycle_failed`; an invalid state or a detail over 200 chars is a 400, never coerced. A transition into a stop verdict writes a `scientist_stop` event (scientist-visible) |
| `GET /history?run=n` | all | **visibility contract below** |
| `GET /stream.mjpg` · `GET /frame/latest` | console (no client gate) | live video; stills are the degradation path. No X-Client gate: a frame shows only cup colour (= scientist-visible measured_lab, no hidden quantity), and an `<img>` tag cannot send the header. The served frame carries a **serve-time** overlay of the current measurement region, drawn on a copy at request time; the cached and captured bytes stay raw, so measure/blank never see annotation. |

**History visibility contract (load-bearing):** the payload returned to the scientist
contains *requested* doses, measured LAB, blank LAB, ΔE, λ, score — and **never** executed
(post-skew) doses or any `scientist_visible=false` event. A skewed cycle's scientist-facing
payload must be byte-identical in its dose fields to an unskewed one. (Without this, the
dilution disruption is transparent in the LLM's own data and the demo's key moment collapses.)

## Safety layer (beneath all callers, indifferent to mode)

- **Hard caps:** max ml per dose, max relay on-time, running **cup-volume accounting** with
  a hard ceiling — an overfill request is rejected, not clamped silently.
- **Session volume budgets:** cumulative water drawn and waste produced per session, with
  configured reservoir/waste capacities. Approaching either limit → `paused` + console
  alert *before* the reservoir runs dry (a dry refill silently corrupts every subsequent
  measurement) or the bucket overflows. Fair math: ~100 ml/cycle × 45–60 cycles ≈ **4.5–6 L**
  each way — stage water and size waste to the 4-run worst case with margin.
- **Watchdog:** every cycle phase has a time budget (derived from *measured* phase timings
  during bring-up + generous margin, per phase — not guessed). Budget exceeded → all pumps
  off, mode → `paused`, console alert.
- **Camera-lock watchdog:** each per-cycle blank (a) re-reads exposure/white-balance/focus
  via v4l2 and asserts they equal the locked values, (b) checks blank-LAB drift against a
  threshold. Either fails → health-strip alarm. (USB webcams silently reset controls; an
  unlocked camera is indistinguishable from "the AI is failing" on the wall view.)
- **Measurement region:** measure and blank sample one operator-set patch — center `cx,cy`
  (normalized, so it survives a resolution change) and `size_px` — not a hard-centered box,
  because the cup is not centred in the frame. `run_cycle` snapshots the patch at cycle start
  and both measure and blank read that snapshot, so an operator nudge between the two phases
  (drain + rinse separate them) never splits one cycle's measured and blank readings across
  two regions. Moving the region is a visibly logged `patch_change` event (scientist-visible):
  as with the camera lock, it changes measurement conditions across a run's cycles, and a
  silent move is indistinguishable from "the AI is failing" on the wall view. Seeded from
  `camera.patch_center` / `camera.patch_px` at boot; persisted and rehydrated alongside the
  volume accounting. A rehydrated patch that no longer fits the configured frame falls back to
  the config default with a logged warning.
- **Boot safety:** first act of rig-service is driving all relay GPIOs HIGH (off on
  low-trigger boards). Channel map (GPIO 4/5/6/21 → A/B/D/W) lives in
  `config/`, never in code.
- **Durable volume accounting:** cup volume and the session water/waste counters are persisted
  and rehydrated at service start — the rig never assumes an empty cup or an untouched
  reservoir. Every change is committed before the next phase begins, in the same transaction as
  the record that caused it. Writes are **ordered so that a crash at any point leaves accounting
  conservative**: believing the cup fuller, the reservoir more depleted, and the waste bucket
  fuller than they physically are. Each quantity is therefore written at whichever side of its
  actuation errs that way — the cup's fill before and its drain after; both session counters
  before. A start that finds a non-zero or unknown cup holds `paused` and raises a console
  alert; the operator drains via `POST /pump` (manual) and then resumes. Unknown is treated as
  non-empty.
- **Staging declaration:** the session counters are unknown until the operator declares
  measured vessel contents via `POST /staging` — the declaration takes real volumes, never a
  "full" flag, and is the **only** thing that resets them; a restart never does. Unknown
  counters reject dosing, so a fresh or lost DB cannot let the rig assume water it does not
  have. The operator restages physically (refill reservoir, empty waste) and re-declares; this
  is the supported mid-run recovery from a budget trip, and it must not require touching the
  database.

## Scientist

State machine against `GET /status`:

- **idle** — no active run, or mode ≠ auto: poll slowly, send no cycle commands (no busy-rejection).
- **active** — run active + auto: fetch history → build prompt (target, λ, full experiment
  table including blank drift) → call the LLM via the provider adapter → `POST /cycle` →
  await completion → repeat.
- **Resume rule:** on any return to auto mid-run, re-read history and continue from the
  current cycle count — never duplicate or skip a cycle.
- **Stop:** cycle budget reached, or **converged** = best ΔE < `conv_delta_e` for
  `conv_hold` consecutive cycles OR no score improvement > `conv_epsilon` over
  `conv_window` cycles, with a floor of `conv_min_cycles`. All four are config, tuned on
  gamut-sweep data.
- **State reporting:** each tick the scientist POSTs its state to `/scientist/state` —
  `thinking` before each LLM call, `waiting` while a cycle executes, a stop verdict
  (`stopped_budget` / `stopped_converged`), an error class (`llm_error`, `parse_error`,
  `cycle_refused`, `cycle_failed`), or `idle` otherwise. Reporting is best-effort: a
  failed report never changes loop behavior. rig-service holds only the latest report in
  memory and writes a `scientist_stop` event on each transition into a stop verdict.

**Provider adapters:** one interface — `propose(prompt: str, model: str) -> str` — with
per-provider implementations: Anthropic (official `anthropic` SDK), Google Gemini
(`google-genai`), Kimi/Moonshot. Every decision row records the model id. **Core scope
builds the Anthropic adapter only**; Gemini/Kimi land after the 15-clean-cycles gate.

Prompt *content* is deliberately out of scope here — it's the subject of the
`Agentic AI Operation` node rewrite and gets exercised in the simulation harness.

## Simulation harness (de-risks the demo before the rig is wet)

A synthetic world behind the same hardware seam (pumps/camera): `f(dose_a, dose_b) → LAB` with
measurement noise, a gamut boundary, and injectable skew — so the **real scientist loop**
runs hundreds of simulated cycles against the **real rig-service** (fake drivers) on
Windows, before the cup is printed. Must support: convergence runs, all four disruptions,
and unreachable targets. This is how the prompt stops being a last-week unknown.

## Data model (SQLite on the Pi, single writer = rig-service)

- **runs** — id, target hex + target LAB, initial λ, cycle budget, model, timestamps, outcome notes
- **cycles** — run id, n, **requested** doses, **executed** doses (post-skew ground truth),
  measured LAB, blank LAB, ΔE, λ-at-cycle, score, phase timestamps, frame paths
- **decisions** — run id, cycle, proposed doses, verbatim LLM rationale, model id, latency
- **events** — mode changes, target/λ changes, disruptions, watchdog/budget trips,
  scientist stop verdicts, operator actions — each with a `scientist_visible` flag
- **rig_state** — single row: durable cup volume, session water/waste counters, and the
  measurement patch (`cx, cy, size_px`), each written in the same transaction as the cycle or
  event row that changed it

Frames (measurement + blank) saved as JPEGs on disk, paths in the DB.

**Backup:** at each run end — (1) export run data as CSV/JSON + frames to a dated folder,
(2) snapshot the DB via SQLite's online backup (`VACUUM INTO` / `.backup`, never a raw file
copy of a live DB) + `PRAGMA integrity_check`, (3) copy to a USB stick on the Pi (SD card
assumed mortal), (4) rsync to the operator's laptop over Tailscale when reachable. The
decision trail is the graded deliverable — treat it accordingly.

## Console

Three static views, all thin clients over the same API the scientist uses:

- **`/wall`** (audience): live video (auto-degrades to per-cycle stills if the stream
  stalls), target patch vs. measured patch, score-vs-cycle chart with event annotations,
  AI rationale ticker, cycle-phase indicator, and a scientist-state indicator beneath it
  (thinking / waiting / converged / budget-reached / generic-error / silent). Error
  *detail* is an `/operate` surface, never the wall. Legible at distance.
- **`/operate`**: mode switch, run start/end, target + λ controls, manual dosing,
  disruption panel, health strip (camera lock, volume budgets, watchdog, last-frame age).
- **`/replay`**: run picker + timeline scrubber replaying frames + charts. *(Post-gate.)*

Charts: score-vs-cycle (headline), ΔE and ml-dye separated, blank-LAB drift, dose-space
scatter colored by score. Exact layout/buttons decided at implementation. No auth, no mobile.

## Build order and scope gate

1. **M0 — walking skeleton (day 1–2):** service stub + real `RelayDriver` + `POST /pump`
   + boot-safe arming, deployed to the Pi, verified against a real relay. Nothing else
   grows until this passes.
2. **Core loop:** safety layer, cycle engine, camera + color path, store, scientist with
   Anthropic adapter, operate + wall views, simulation harness.
3. **Gate: 15 clean autonomous cycles end-to-end on the real rig.**
4. **Post-gate:** replay view, Gemini/Kimi adapters, secondary charts, polish.

## Test strategy

- **Unit (TDD-gated, `src/` ↔ `tests/` mirror):** color math (RGB→LAB, ΔE), scoring, dose
  math, calibration lookup, convergence logic, mode arbitration, cap/budget accounting.
- **Property-based** where invariants allow (e.g. cup volume never exceeds ceiling under
  any command sequence).
- **Service-level:** FastAPI TestClient + fake drivers — mode enforcement, history
  visibility contract (skewed vs. unskewed payload identical in dose fields), watchdog and
  budget trips, blank/camera-lock alarm.
- **Simulation:** scientist convergence + all disruptions against the synthetic world.
- **On-rig:** M0 relay verification; one e2e smoke cycle; measured phase timings feed the
  watchdog budgets.
- **`console/` is exempt from the TDD gate** (decision 2026-07-13); everything under
  `src/` is test-first.

## Invariants

1. No dose executes that would exceed cup ceiling, per-dose cap, or relay on-time cap.
2. Exactly one authority can command doses at any instant (mode arbitration).
3. The scientist-visible history never contains executed doses or hidden events.
4. All relay GPIOs are off (HIGH) at service start, after watchdog trip, and at `paused`.
5. Every executed dose is recorded (requested + executed) before the next phase begins.
6. Session water/waste accounting pauses the rig before physical capacity is reached.
7. A run's exported backup passes `PRAGMA integrity_check`.
8. Volume accounting — cup and session counters — survives restart; it is never silently reset
   to empty.

## Acceptance criteria

- [ ] M0: relay fires from `POST /pump` on the real Pi; boot-safe arming verified across a reboot
- [ ] One full cycle from an HTTP call: doses in → LAB/ΔE/score out, logged (Milestone 3)
- [ ] Simulation: scientist converges on a reachable synthetic target and detects a skew disruption
- [ ] 15 clean autonomous cycles end-to-end on the real rig (scope gate)
- [ ] Mode switching mid-run is clean: manual carries a demo, auto resumes without duplicate/skipped cycles
- [ ] All four disruptions injectable from `/operate`; skew invisible in scientist history (test passes)
- [ ] Volume budgets pause the rig in a forced-depletion test
- [ ] Camera-lock alarm fires on a forced v4l2 control reset
- [ ] Run backup lands on USB + laptop and opens clean
- [ ] Wall view legible from 3 m; stills fallback works with the stream disabled
- [ ] 10–20 cycle AI run with decision trail + ≥1 live disruption captured (Milestone 4)

## Fair-day runbook (ops)

**Pre-fair (Prague, with on-site friend):** stage water to 4-run worst case + margin; waste
bucket sized ≥ staged water and empty; dye stocks topped to a computed volume target (not a
checkbox); return-bus tug test; camera exposure/WB lock verified; Tailscale green both ends;
Prague upload bandwidth measured; one full test cycle end-to-end.
**During:** rolling runs (plan **2–3** full optimisations, one with a live disruption —
derived from measured cycle times, not optimism); timed mid-demo waste/reservoir check by
the on-site friend; disruptions injected when a crowd gathers.
**Failure playbook:** stream chokes → stills; scientist misbehaves → manual mode carries the
demo; Pi reboot → systemd auto-restart + boot-safe arming + resume; anything wet → friend
pulls the 12 V plug (the kill switch is mechanical); London connectivity lost → run
continues in Prague, rejoin when back.

## Related

- Design narrative: vault → `VSD/Stochastic Studio/Nodes/Control System.md`
- Rig physical state (authoritative): vault → `VSD/Stochastic Studio/Build Log.md`
- Project design authority: vault → `VSD/Stochastic Studio/Project Idea.md`
- Objective/prompt design (pending): vault → `VSD/Stochastic Studio/Nodes/Agentic AI Operation.md`
- Methodology source: vault → `Claude Infra/` (<name>'s verified-coding system)
