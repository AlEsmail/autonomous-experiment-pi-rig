// ABOUTME: Shared console helpers -- authenticated fetch wrapper, status polling, and
// ABOUTME: the LAB -> sRGB hex conversion used to render color patches.

async function api(method, path, body) {
  const options = {
    method: method,
    headers: { "X-Client": "console" },
  };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  let data = null;
  try {
    data = await response.json();
  } catch (err) {
    data = null;
  }
  if (!response.ok) {
    const detail = data && data.detail !== undefined ? data.detail : response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function poll(path, intervalMs, onData, onError) {
  const interval = intervalMs || 1500;
  async function tick() {
    try {
      const data = await api("GET", path);
      onData(data);
    } catch (err) {
      onError(err);
    }
  }
  tick();
  return setInterval(tick, interval);
}

// Inverse of shared/color.py's rgb_to_lab pipeline (D65 white point): CIE LAB ->
// XYZ -> linear sRGB -> gamma-encoded 8-bit sRGB, clamping out-of-gamut values
// exactly as shared/color.py's linear_rgb_to_srgb8 does.
function labToHex(l, a, b) {
  const D65 = [0.95047, 1.0, 1.08883];
  const delta = 6 / 29;

  function finv(t) {
    return t > delta ? t * t * t : 3 * delta * delta * (t - 4 / 29);
  }

  const fy = (l + 16) / 116;
  const fx = fy + a / 500;
  const fz = fy - b / 200;

  const x = D65[0] * finv(fx);
  const y = D65[1] * finv(fy);
  const z = D65[2] * finv(fz);

  const rLin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
  const gLin = -0.969266 * x + 1.8760108 * y + 0.041556 * z;
  const bLin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;

  function encode(c) {
    const clamped = Math.min(Math.max(c, 0), 1);
    const srgb =
      clamped <= 0.0031308 ? 12.92 * clamped : 1.055 * Math.pow(clamped, 1 / 2.4) - 0.055;
    return Math.min(Math.max(Math.round(srgb * 255), 0), 255);
  }

  function toHex(n) {
    return n.toString(16).padStart(2, "0");
  }

  return "#" + toHex(encode(rLin)) + toHex(encode(gLin)) + toHex(encode(bLin));
}
