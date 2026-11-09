// ABOUTME: Wires the /wall audience view -- live video with stills fallback, target vs
// ABOUTME: measured patches, the score-vs-cycle chart with event annotations, and the ticker.

const FRAME_AGE_STALE_S = 5;
const STILLS_POLL_MS = 2000;
const STREAM_RETRY_MS = 30000;

const videoImg = document.getElementById("video");
const videoModeEl = document.getElementById("video-mode");

let videoMode = "stream";
let stillsIntervalId = null;

function renderVideoMode() {
  videoModeEl.textContent = videoMode === "stream" ? "live" : "stills";
}

function startStream() {
  videoMode = "stream";
  if (stillsIntervalId !== null) {
    clearInterval(stillsIntervalId);
    stillsIntervalId = null;
  }
  videoImg.src = "/stream.mjpg?t=" + Date.now();
  renderVideoMode();
}

function pollStill() {
  videoImg.src = "/frame/latest?t=" + Date.now();
}

function switchToStills() {
  if (videoMode === "stills") return;
  videoMode = "stills";
  renderVideoMode();
  pollStill();
  stillsIntervalId = setInterval(pollStill, STILLS_POLL_MS);
}

// MJPEG (multipart/x-mixed-replace) only fires the <img> load event once at stream
// connect, not per frame, so it cannot be used as a freshness signal.
videoImg.addEventListener("error", () => {
  switchToStills();
});

// Retry the live stream periodically while degraded.
setInterval(() => {
  if (videoMode === "stills") {
    startStream();
  }
}, STREAM_RETRY_MS);

startStream();

// --- Patches ---

const targetPatch = document.getElementById("target-patch");
const measuredPatch = document.getElementById("measured-patch");
const targetHexLabel = document.getElementById("target-hex-label");
const measuredHexLabel = document.getElementById("measured-hex-label");

function renderTargetPatch(targetHex) {
  if (!targetHex) {
    targetPatch.style.background = "#000";
    targetHexLabel.textContent = "--";
    return;
  }
  targetPatch.style.background = targetHex;
  targetHexLabel.textContent = targetHex;
}

function renderMeasuredPatch(cycles) {
  const latest = latestMeasuredCycle(cycles);
  if (latest === null) {
    measuredPatch.style.background = "#000";
    measuredHexLabel.textContent = "--";
    return;
  }
  const hex = labToHex(latest.measured_l, latest.measured_a, latest.measured_b);
  measuredPatch.style.background = hex;
  measuredHexLabel.textContent = hex;
}

function latestMeasuredCycle(cycles) {
  for (let i = cycles.length - 1; i >= 0; i--) {
    if (cycles[i].measured_l !== null && cycles[i].measured_l !== undefined) {
      return cycles[i];
    }
  }
  return null;
}

// --- Phase indicator ---

const phaseValueEl = document.getElementById("phase-value");

function renderPhase(phase) {
  phaseValueEl.textContent = phase || "--";
}

// --- Scientist indicator ---

const SCIENTIST_SILENT_S = 900;

const scientistValueEl = document.getElementById("scientist-value");
const scientistDetailEl = document.getElementById("scientist-detail");

function fmtAgeMSS(ageS) {
  const m = Math.floor(ageS / 60);
  const s = Math.floor(ageS % 60);
  return m + ":" + String(s).padStart(2, "0");
}

function renderScientist(scientist) {
  const state = scientist && scientist.state ? scientist.state : "unknown";
  const detail = scientist && scientist.detail ? scientist.detail : "";
  const ageS = scientist ? scientist.age_s : null;

  scientistValueEl.className = "scientist-value";
  scientistDetailEl.textContent = "";

  const terminal = state === "stopped_budget" || state === "stopped_converged";
  if (!terminal && ageS !== null && ageS !== undefined && ageS > SCIENTIST_SILENT_S) {
    scientistValueEl.textContent = "silent — check operator";
    scientistValueEl.classList.add("scientist-warn");
    return;
  }

  switch (state) {
    case "thinking":
      scientistValueEl.textContent =
        ageS === null || ageS === undefined
          ? "thinking…"
          : "thinking… " + fmtAgeMSS(ageS);
      break;
    case "waiting":
      scientistValueEl.textContent = "waiting on cycle";
      break;
    case "stopped_converged":
      scientistValueEl.textContent = "CONVERGED";
      scientistValueEl.classList.add("scientist-good");
      if (detail) scientistDetailEl.textContent = detail;
      break;
    case "stopped_budget":
      scientistValueEl.textContent = "budget reached";
      scientistValueEl.classList.add("scientist-good");
      if (detail) scientistDetailEl.textContent = detail;
      break;
    case "llm_error":
    case "parse_error":
    case "cycle_refused":
    case "cycle_failed":
      scientistValueEl.textContent = "hiccup — retrying";
      scientistValueEl.classList.add("scientist-warn");
      break;
    case "idle":
      scientistValueEl.textContent = "standing by";
      scientistValueEl.classList.add("scientist-muted");
      break;
    default:
      scientistValueEl.textContent = "--";
      scientistValueEl.classList.add("scientist-muted");
  }
}

// --- Chart (hand-rolled SVG score-vs-cycle) ---

const CHART_W = 1000;
const CHART_H = 400;
const CHART_MARGIN = { top: 20, right: 20, bottom: 50, left: 100 };

// The pinned annotation kinds (plan card T5) -- excludes run-boundary and manual-dosing
// events (run_start, run_end, pump, calibrate) that are not mid-run chart annotations.
const ANNOTATION_KINDS = new Set([
  "mode_change",
  "target_change",
  "lambda_change",
  "skew",
  "blip",
  "trip",
  "staging",
]);

const chartSvg = document.getElementById("chart");

function cycleStartTime(cycle) {
  const times = Object.values(cycle.phase_started || {});
  return times.length ? Math.min(...times) : null;
}

// Places an event on the x-axis at the cycle whose phase timing had most recently
// begun when the event fired (cycles carry no timestamp of their own, only
// per-phase epoch times in phase_started).
function eventCycleN(event, cycles) {
  let chosen = null;
  for (const cycle of cycles) {
    const t = cycleStartTime(cycle);
    if (t !== null && t <= event.at) chosen = cycle;
  }
  if (chosen !== null) return chosen.n;
  return cycles.length ? cycles[0].n : null;
}

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const key in attrs) {
    el.setAttribute(key, attrs[key]);
  }
  return el;
}

function fmt1(v) {
  return v === null || v === undefined ? "--" : v.toFixed(1);
}

function renderChart(cycles, events) {
  chartSvg.setAttribute("viewBox", "0 0 " + CHART_W + " " + CHART_H);
  chartSvg.textContent = "";

  if (cycles.length === 0) return;

  const plotW = CHART_W - CHART_MARGIN.left - CHART_MARGIN.right;
  const plotH = CHART_H - CHART_MARGIN.top - CHART_MARGIN.bottom;

  const ns = cycles.map((c) => c.n);
  const scores = cycles.map((c) => c.score).filter((s) => s !== null && s !== undefined);
  const minN = Math.min(...ns);
  const maxN = Math.max(...ns);
  const minScore = scores.length ? Math.min(...scores) : 0;
  const maxScore = scores.length ? Math.max(...scores) : 1;
  const scoreSpan = maxScore - minScore || 1;

  function xFor(n) {
    const span = maxN - minN || 1;
    return CHART_MARGIN.left + ((n - minN) / span) * plotW;
  }
  function yFor(score) {
    return CHART_MARGIN.top + plotH - ((score - minScore) / scoreSpan) * plotH;
  }

  // Axes.
  chartSvg.appendChild(
    svgEl("line", {
      x1: CHART_MARGIN.left,
      y1: CHART_MARGIN.top + plotH,
      x2: CHART_MARGIN.left + plotW,
      y2: CHART_MARGIN.top + plotH,
      stroke: "#3a4258",
      "stroke-width": 2,
    })
  );
  chartSvg.appendChild(
    svgEl("line", {
      x1: CHART_MARGIN.left,
      y1: CHART_MARGIN.top,
      x2: CHART_MARGIN.left,
      y2: CHART_MARGIN.top + plotH,
      stroke: "#3a4258",
      "stroke-width": 2,
    })
  );

  // Y-axis: rotated "score" title plus min/mid/max tick labels.
  const yTitle = svgEl("text", {
    x: 0,
    y: 0,
    fill: "#9fb0c9",
    "font-size": 18,
    "text-anchor": "middle",
    transform: "translate(30 " + (CHART_MARGIN.top + plotH / 2) + ") rotate(-90)",
  });
  yTitle.textContent = "score";
  chartSvg.appendChild(yTitle);
  const midScore = (minScore + maxScore) / 2;
  const tickValues =
    maxScore === minScore ? [maxScore] : [minScore, midScore, maxScore];
  for (const v of tickValues) {
    const tick = svgEl("text", {
      x: CHART_MARGIN.left - 10,
      y: yFor(v) + 6,
      fill: "#9fb0c9",
      "font-size": 18,
      "text-anchor": "end",
    });
    tick.textContent = Number(v.toFixed(1));
    chartSvg.appendChild(tick);
  }

  // Event annotations: vertical markers with kind labels, drawn under the score line.
  for (const event of events) {
    if (!ANNOTATION_KINDS.has(event.kind)) continue;
    const n = eventCycleN(event, cycles);
    if (n === null) continue;
    const x = xFor(n);
    chartSvg.appendChild(
      svgEl("line", {
        x1: x,
        y1: CHART_MARGIN.top,
        x2: x,
        y2: CHART_MARGIN.top + plotH,
        stroke: "#ff8080",
        "stroke-width": 1.5,
        "stroke-dasharray": "5,4",
      })
    );
    const label = svgEl("text", {
      x: x + 4,
      y: CHART_MARGIN.top + 14,
      fill: "#ff8080",
      "font-size": 18,
    });
    label.textContent = event.kind;
    chartSvg.appendChild(label);
  }

  // Score-vs-cycle line + points.
  const points = cycles
    .filter((c) => c.score !== null && c.score !== undefined)
    .map((c) => xFor(c.n) + "," + yFor(c.score))
    .join(" ");
  if (points) {
    chartSvg.appendChild(
      svgEl("polyline", {
        points: points,
        fill: "none",
        stroke: "#3a6ef0",
        "stroke-width": 4,
      })
    );
  }
  for (const cycle of cycles) {
    if (cycle.score === null || cycle.score === undefined) continue;
    const dot = svgEl("circle", {
      cx: xFor(cycle.n),
      cy: yFor(cycle.score),
      r: 6,
      fill: "#3a6ef0",
    });
    const tip = svgEl("title", {});
    tip.textContent =
      "cycle " + cycle.n +
      " · A " + cycle.requested_a_ml + " ml" +
      " · B " + cycle.requested_b_ml + " ml" +
      " · ΔE " + fmt1(cycle.delta_e) +
      " · λ " + fmt1(cycle.lambda_at_cycle) +
      " · score " + fmt1(cycle.score);
    dot.appendChild(tip);
    chartSvg.appendChild(dot);
    const label = svgEl("text", {
      x: xFor(cycle.n),
      y: CHART_MARGIN.top + plotH + 24,
      fill: "#9fb0c9",
      "font-size": 20,
      "text-anchor": "middle",
    });
    label.textContent = cycle.n;
    chartSvg.appendChild(label);
  }
}

// --- Ticker ---

const tickerList = document.getElementById("ticker-list");

function renderTicker(decisions) {
  tickerList.innerHTML = "";
  const newestFirst = decisions.slice().reverse();
  for (const decision of newestFirst) {
    const li = document.createElement("li");
    const cycleSpan = document.createElement("span");
    cycleSpan.className = "cycle-n";
    cycleSpan.textContent = "#" + decision.cycle_n;
    li.appendChild(cycleSpan);
    li.appendChild(document.createTextNode(decision.rationale));
    tickerList.appendChild(li);
  }
}

// --- Data polling ---

let currentRunId = null;
let historyIntervalId = null;

function onStatus(data) {
  renderPhase(data.cycle && data.cycle.phase);
  renderScientist(data.scientist);
  renderTargetPatch(data.target_hex);

  if (videoMode === "stream" && data.last_frame_age_s > FRAME_AGE_STALE_S) {
    switchToStills();
  }

  const runId = data.run ? data.run.id : null;
  if (runId !== currentRunId) {
    currentRunId = runId;
    if (historyIntervalId !== null) {
      clearInterval(historyIntervalId);
      historyIntervalId = null;
    }
    renderChart([], []);
    renderTicker([]);
    renderMeasuredPatch([]);
    if (runId !== null) {
      historyIntervalId = poll("/history?run=" + runId, 2000, onHistory, onHistoryError);
    }
  }
}

function onStatusError(err) {
  renderPhase("status error: " + err.message);
}

function onHistory(data) {
  renderChart(data.cycles, data.events);
  renderMeasuredPatch(data.cycles);
  renderTicker(data.decisions);
}

function onHistoryError(err) {
  // Leave the last-known render in place; the health of /status polling is the
  // primary signal, so a transient /history error does not blank the wall.
}

poll("/status", 1500, onStatus, onStatusError);
