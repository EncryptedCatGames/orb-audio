/* ORB in-browser engine: Pyodide + numpy running the real orb.api.render_spec.
 *
 * Protocol (postMessage):
 *   in : {type: "init"}                     -> {type: "ready", ms, version}
 *   in : {type: "render", id, spec, seed}   -> {type: "result", id, pcm, ...}
 *   in : {type: "plan", id, caption, planner: {index, bin, bytes}}
 *                                           -> {type: "planned", id, spec, planned, ms, tokens}
 *        (first call downloads the planner weights; progress arrives as
 *         {type: "progress", id, loaded, total})
 *   out: {type: "status", text}  progress   |  {type: "error", id?, message}
 */
const PYODIDE_VERSION = "0.27.2";
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let pyodide = null;
let renderFn = null;
let booting = null;
let plannerLoading = null;

function status(text) {
  self.postMessage({ type: "status", text });
}

async function boot() {
  const t0 = performance.now();
  status("Downloading Python runtime");
  importScripts(PYODIDE_BASE + "pyodide.js");
  pyodide = await self.loadPyodide({ indexURL: PYODIDE_BASE });
  status("Loading numpy");
  await pyodide.loadPackage("numpy");
  status("Loading the ORB renderer");
  // worker.js?v=<bundle hash> passes the hash on, so a new renderer is never
  // shadowed by a cached zip.
  const v = new URLSearchParams(self.location.search).get("v") || "";
  const res = await fetch(new URL(`py/orb.zip${v ? `?v=${v}` : ""}`, self.location.href));
  if (!res.ok) throw new Error(`py/orb.zip: HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  pyodide.FS.mkdirTree("/orb_site");
  pyodide.unpackArchive(buf, "zip", { extractDir: "/orb_site" });
  pyodide.runPython(`
import sys
if "/orb_site" not in sys.path:
    sys.path.insert(0, "/orb_site")
import orb_web
import orb_planner
`);
  renderFn = pyodide.globals.get("orb_web").render_js;
  // Warm up the code paths so the first visible render is quick.
  status("Warming up");
  const warm = renderFn(JSON.stringify({ duration: 2.0, texture: "vinyl", mood: "calm",
    events: [{ role: "click", t: 0.5, dur: 0.1, gain_db: -16, pan: 0, seed: 1 }] }), 0);
  warm.destroy();
  const version = pyodide.runPython("import numpy, sys; f'Python {sys.version.split()[0]} / numpy {numpy.__version__}'");
  self.postMessage({ type: "ready", ms: Math.round(performance.now() - t0), version });
}

function render(msg) {
  const t0 = performance.now();
  const out = renderFn(JSON.stringify(msg.spec), msg.seed | 0);
  try {
    const pcmProxy = out.get("pcm");
    const view = pcmProxy.getBuffer("f32");
    const pcm = new Float32Array(view.data); // copy out of the wasm heap
    view.release();
    pcmProxy.destroy();
    const plain = (key) => {
      const v = out.get(key);
      return v && v.toJs ? v.toJs({ dict_converter: Object.fromEntries }) : v;
    };
    const result = {
      type: "result",
      id: msg.id,
      pcm,
      channels: out.get("channels"),
      sampleRate: out.get("sample_rate"),
      ms: out.get("ms"),
      wallMs: Math.round(performance.now() - t0),
      peaks: plain("peaks"),
      spectrogram: out.get("spectrogram"),
      audit: plain("audit"),
      passes: out.get("passes"),
    };
    self.postMessage(result, [pcm.buffer]);
  } finally {
    out.destroy();
  }
}

async function loadPlanner(msg) {
  const meta = msg.planner || {};
  const base = self.location.href;
  const q = meta.v ? `?v=${meta.v}` : "";
  status("Downloading the planner");
  const index = await (await fetch(new URL(meta.index + q, base))).text();
  const res = await fetch(new URL(meta.bin + q, base));
  if (!res.ok) throw new Error(`${meta.bin}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || meta.bytes || 0;
  const buf = new Uint8Array(total || meta.bytes || 0);
  let loaded = 0;
  const chunks = [];
  const reader = res.body.getReader();
  let last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length <= buf.length) buf.set(value, loaded);
    else chunks.push(value);
    loaded += value.length;
    const now = performance.now();
    if (now - last > 80) {
      last = now;
      self.postMessage({ type: "progress", id: msg.id, loaded, total });
    }
  }
  self.postMessage({ type: "progress", id: msg.id, loaded, total: loaded });
  let bytes = buf;
  if (loaded !== buf.length) {  // no or wrong content-length: stitch the pieces
    bytes = new Uint8Array(loaded);
    bytes.set(buf.subarray(0, Math.min(loaded, buf.length)), 0);
    let off = Math.min(loaded, buf.length);
    for (const c of chunks) { bytes.set(c, off); off += c.length; }
  }
  status("Loading the planner");
  const load = pyodide.globals.get("orb_planner").load;
  const py = pyodide.toPy(bytes);
  try {
    load(index, py).destroy();
  } finally {
    py.destroy();
    load.destroy();
  }
}

function plan(msg) {
  const fn = pyodide.globals.get("orb_planner").generate_js;
  try {
    const t0 = performance.now();
    const out = JSON.parse(fn(msg.caption));
    self.postMessage({ type: "planned", id: msg.id, wallMs: Math.round(performance.now() - t0), ...out });
  } finally {
    fn.destroy();
  }
}

self.onmessage = async (e) => {
  const msg = e.data || {};
  try {
    if (msg.type === "init") {
      booting = booting || boot();
      await booting;
    } else if (msg.type === "render") {
      if (!booting) booting = boot();
      await booting;
      render(msg);
    } else if (msg.type === "plan") {
      if (!booting) booting = boot();
      await booting;
      // "no bed" captions never touch the planner, so they skip the download.
      const direct = pyodide.globals.get("orb_planner").needs_planner;
      const needs = direct(msg.caption);
      direct.destroy();
      if (needs) {
        plannerLoading = plannerLoading || loadPlanner(msg);
        try {
          await plannerLoading;
        } catch (err) {
          plannerLoading = null;
          throw err;
        }
      }
      plan(msg);
    }
  } catch (err) {
    self.postMessage({ type: "error", id: msg.id, message: String((err && err.message) || err) });
  }
};
