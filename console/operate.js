// ABOUTME: Wires the /operate panels to their endpoints -- every submit renders the
// ABOUTME: success payload or the error's detail, and the health strip polls /status.

function renderResult(resultEl, errorEl, data) {
  errorEl.textContent = "";
  resultEl.textContent = JSON.stringify(data, null, 2);
}

function renderError(resultEl, errorEl, err) {
  resultEl.textContent = "";
  errorEl.textContent = err.message;
}

function wire(buttonId, resultId, errorId, action) {
  const button = document.getElementById(buttonId);
  const resultEl = document.getElementById(resultId);
  const errorEl = document.getElementById(errorId);
  button.addEventListener("click", async () => {
    try {
      const data = await action();
      renderResult(resultEl, errorEl, data);
    } catch (err) {
      renderError(resultEl, errorEl, err);
    }
  });
}

function value(id) {
  return document.getElementById(id).value;
}

function numberValue(id) {
  return Number(document.getElementById(id).value);
}

wire("mode-submit", "mode-result", "mode-error", () =>
  api("POST", "/mode", { mode: value("mode-select") })
);

wire("run-start-submit", "run-result", "run-error", () =>
  api("POST", "/run/start", {
    target_hex: value("run-target-hex"),
    lambda: numberValue("run-lambda"),
    cycle_budget: numberValue("run-cycle-budget"),
    model: value("run-model"),
  })
);

wire("run-end-submit", "run-result", "run-error", () => {
  const outcome = value("run-end-outcome");
  return api("POST", "/run/end", outcome ? { outcome } : {});
});

wire("target-submit", "target-lambda-result", "target-lambda-error", () =>
  api("POST", "/target", { hex: value("target-hex") })
);

wire("lambda-submit", "target-lambda-result", "target-lambda-error", () =>
  api("POST", "/lambda", { value: numberValue("lambda-value") })
);

wire("patch-submit", "patch-result", "patch-error", () =>
  api("POST", "/patch", { cx: numberValue("patch-cx"), cy: numberValue("patch-cy"),
                          size_px: numberValue("patch-size") })
);

wire("pump-submit", "pump-result", "pump-error", () =>
  api("POST", "/pump", { channel: value("pump-channel"), ml: numberValue("pump-ml") })
);

wire("cycle-submit", "cycle-result", "cycle-error", () => {
  const body = {
    dose_a_ml: numberValue("cycle-dose-a"),
    dose_b_ml: numberValue("cycle-dose-b"),
  };
  const rationale = value("cycle-rationale");
  const modelId = value("cycle-model-id");
  const latency = value("cycle-latency");
  if (rationale) body.rationale = rationale;
  if (modelId) body.model_id = modelId;
  if (latency) body.latency_s = Number(latency);
  return api("POST", "/cycle", body);
});

wire("skew-submit", "disrupt-result", "disrupt-error", () =>
  api("POST", "/disrupt/skew", {
    channel: value("skew-channel"),
    factor: numberValue("skew-factor"),
  })
);

wire("blip-submit", "disrupt-result", "disrupt-error", () =>
  api("POST", "/disrupt/blip", { ml_water: numberValue("blip-ml") })
);

wire("staging-submit", "staging-result", "staging-error", () =>
  api("POST", "/staging", {
    water_remaining_ml: numberValue("staging-water"),
    waste_ml: numberValue("staging-waste"),
  })
);

wire("calibrate-submit", "calibrate-result", "calibrate-error", () =>
  api("POST", "/calibrate", {
    channel: value("calibrate-channel"),
    seconds: numberValue("calibrate-seconds"),
  })
);

const healthFields = document.getElementById("health-fields");
const healthError = document.getElementById("health-error");

function field(label, val) {
  return '<div class="field"><b>' + label + "</b>" + val + "</div>";
}

function renderHealth(data) {
  healthError.textContent = "";
  const modeHtml = data.trip_reason
    ? '<span class="tripped">' + data.mode + " -- " + data.trip_reason + "</span>"
    : data.mode;
  healthFields.innerHTML =
    field("mode", modeHtml) +
    field("cycle phase", data.cycle.phase) +
    field("target", data.target_hex || "--") +
    field("lambda", data.lambda) +
    field("relays on", JSON.stringify(data.relays_on || [])) +
    field(
      "volumes",
      JSON.stringify(data.volumes || {})
    ) +
    field("alerts", JSON.stringify(data.alerts || [])) +
    field("alarms", JSON.stringify(data.alarms || [])) +
    field(
      "last frame age (s)",
      data.last_frame_age_s === null || data.last_frame_age_s === undefined
        ? "--"
        : data.last_frame_age_s.toFixed(1)
    );
}

function renderHealthError(err) {
  healthError.textContent = err.message;
}

poll("/status", 1500, renderHealth, renderHealthError);

api("GET", "/status").then(data => {
  if (data.patch) {
    document.getElementById("patch-cx").value = data.patch.cx;
    document.getElementById("patch-cy").value = data.patch.cy;
    document.getElementById("patch-size").value = data.patch.size_px;
  }
}).catch(() => {});
