/* ORB showcase page. Plain JS, no build step.
 *
 * - Prompt box: a caption (or plain words through a keyword helper) goes to
 *   the real planner, ported to numpy (py/orb_planner.py), then the renderer.
 * - Library: pre-rendered practical cues from assets/cues.json (built by
 *   scripts/build_site.py, detail model on).
 * - Engine: worker.js runs the real orb.api.render_spec under Pyodide + numpy.
 * - Playground: preset scores from cues.json with touch controls; every edit
 *   re-renders in the browser and plays.
 */
(() => {
  "use strict";

  const PLAY_D = "M4 2.5v11l9-5.5z";
  const PAUSE_D = "M4 2.5h3v11H4zM9 2.5h3v11H9z";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const round2 = (x) => Math.round(x * 100) / 100;
  const debounce = (fn, ms) => {
    let t = 0;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  };

  let DATA = null; // cues.json

  /* Theme colours come from CSS custom properties, so canvases follow dark mode. */
  let THEME = null;
  function theme() {
    if (THEME) return THEME;
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    THEME = {
      wave: v("--wave"), played: v("--wave-played"), ink: v("--ink"), muted: v("--muted"),
      line: v("--line"), sage: v("--sage"), accent: v("--accent-fill"), accentText: v("--accent"),
      surface2: v("--surface-2"),
    };
    return THEME;
  }

  /* ------------------------------------------------------------------ */
  /* JSON highlighting                                                    */
  /* ------------------------------------------------------------------ */

  function num(x) {
    return Number.isInteger(x) ? x.toFixed(1) : String(round2(x));
  }
  function highlight(json) {
    return esc(json).replace(/(&quot;[^&]*?&quot;)(\s*:)?|(-?\d+(?:\.\d+)?(?:e-?\d+)?)/g, (m, str, colon, n) => {
      if (str) return colon ? `<span class="k">${str}</span>${colon}` : `<span class="s">${str}</span>`;
      return `<span class="n">${n}</span>`;
    });
  }


  /* Canvas sizing */

  function fit(canvas) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(canvas.clientWidth));
    const h = Math.max(1, Math.round(canvas.clientHeight));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }


  /* ------------------------------------------------------------------ */
  /* Waveform drawing                                                    */
  /* ------------------------------------------------------------------ */

  function roundBar(ctx, x, y, w, h) {
    const r = Math.min(w / 2, h / 2);
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
    else ctx.rect(x, y, w, h);
    ctx.fill();
  }

  /** Rounded mirrored waveform. opts: progress (0..1 or -1), bars (count), dim. */
  function drawWave(canvas, cue, opts = {}) {
    const { ctx, w, h } = fit(canvas);
    const T = theme();
    ctx.clearRect(0, 0, w, h);
    const areaH = h;
    ctx.save();
    if (opts.dim) ctx.globalAlpha = 0.4;
    const peaks = cue.peaks || [];
    const progress = opts.progress ?? -1;
    if (peaks.length) {
      const bars = opts.bars || Math.max(24, Math.floor(w / 5));
      const step = w / bars;
      const bw = Math.max(1.5, step * 0.56);
      let maxAbs = 1e-6;
      for (const [a, b] of peaks) maxAbs = Math.max(maxAbs, Math.abs(a), Math.abs(b));
      const mid = areaH / 2;
      for (let i = 0; i < bars; i++) {
        const a = Math.floor((i / bars) * peaks.length);
        const b = Math.max(a + 1, Math.floor(((i + 1) / bars) * peaks.length));
        let amp = 0;
        for (let k = a; k < b && k < peaks.length; k++) amp = Math.max(amp, peaks[k][1] - peaks[k][0]);
        const norm = Math.pow(amp / (2 * maxAbs), 0.7);
        const bh = Math.max(bw, norm * areaH * 0.92);
        ctx.fillStyle = progress >= 0 && (i + 0.5) / bars <= progress ? T.played : T.wave;
        const base = opts.dim ? 0.4 : 1;
        ctx.globalAlpha = norm < 0.06 ? base * 0.5 : base;
        roundBar(ctx, i * step + (step - bw) / 2, mid - bh / 2, bw, bh);
      }
    }
    ctx.restore();
  }

  /* ------------------------------------------------------------------ */
  /* Audio                                                               */
  /* ------------------------------------------------------------------ */

  let actx = null;
  const decoded = new Map();
  const player = { source: null, onEnd: null, onTick: null, start: 0, dur: 0, raf: 0, token: null };

  function audio() {
    if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
    if (actx.state === "suspended") actx.resume();
    return actx;
  }
  async function loadBuffer(url) {
    if (decoded.has(url)) return decoded.get(url);
    const p = fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
      return r.arrayBuffer();
    }).then((buf) => new Promise((res, rej) => audio().decodeAudioData(buf, res, rej)));
    decoded.set(url, p);
    try {
      return await p;
    } catch (err) {
      decoded.delete(url);
      throw err;
    }
  }
  function pcmToBuffer(pcm, channels, sr) {
    const ctx = audio();
    const n = pcm.length / channels;
    const buf = ctx.createBuffer(channels, n, sr);
    for (let c = 0; c < channels; c++) {
      const ch = buf.getChannelData(c);
      for (let i = 0; i < n; i++) ch[i] = pcm[i * channels + c];
    }
    return buf;
  }
  function stop() {
    if (player.source) {
      player.source.onended = null;
      try { player.source.stop(); } catch (_) { /* already stopped */ }
      player.source = null;
    }
    cancelAnimationFrame(player.raf);
    const end = player.onEnd;
    player.onEnd = null;
    player.onTick = null;
    player.token = null;
    if (end) end();
  }
  /** Play a buffer. onTick(progress 0..1, seconds) runs every frame. */
  function play(buffer, token, onTick, onEnd) {
    stop();
    const ctx = audio();
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    player.source = src;
    player.token = token;
    player.onTick = onTick;
    player.onEnd = onEnd;
    player.start = ctx.currentTime;
    player.dur = buffer.duration;
    src.onended = () => {
      if (player.source === src) stop();
    };
    src.start();
    const tick = () => {
      if (player.source !== src) return;
      const t = ctx.currentTime - player.start;
      if (player.onTick) player.onTick(clamp(t / player.dur, 0, 1), t);
      player.raf = requestAnimationFrame(tick);
    };
    player.raf = requestAnimationFrame(tick);
  }
  function setPlayIcon(btn, playing) {
    const path = btn && btn.querySelector(".i-play path");
    if (path) path.setAttribute("d", playing ? PAUSE_D : PLAY_D);
    if (!btn) return;
    btn.classList.toggle("playing", playing);
    btn.setAttribute("aria-pressed", String(playing));
    if (btn.dataset.label) btn.setAttribute("aria-label", `${playing ? "Stop" : "Play"} ${btn.dataset.label}`);
  }

  /* ------------------------------------------------------------------ */
  /* Engine (Pyodide worker)                                             */
  /* ------------------------------------------------------------------ */

  const engine = {
    worker: null, ready: false, failed: false, info: "", pending: new Map(), seq: 0,
    // "server" when the page is served by web/server.py (the HF Space):
    // planning and rendering go over HTTP and Pyodide never boots.
    mode: "probe", probed: null, wantInit: false,
    probe() {
      if (this.probed) return this.probed;
      if (/\.github\.io$/.test(location.hostname) || location.protocol === "file:") {
        // Static hosting: there is no server engine to find.
        this.mode = "browser";
        this.probed = Promise.resolve(this.mode);
        if (this.wantInit) this.init();
        return this.probed;
      }
      const ctl = typeof AbortController === "function" ? new AbortController() : null;
      const timer = setTimeout(() => ctl && ctl.abort(), 3000);
      this.probed = fetch("api/health", { cache: "no-store", signal: ctl ? ctl.signal : undefined })
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null)
        .then((j) => {
          clearTimeout(timer);
          this.mode = j && j.ok && j.mode === "server" ? "server" : "browser";
          if (this.mode === "server") {
            this.ready = true;
            this.status("ready", "server engine ready");
            document.querySelectorAll("[data-where]").forEach((n) => { n.textContent = "Both run on this server, so there is nothing to download first."; });
            document.querySelectorAll("[data-fine]").forEach((n) => { n.textContent = "Generation runs on this server. Every result downloads as a 48 kHz stereo WAV, and the same page runs fully in your browser at stdevmac.github.io/orb-audio."; });
            document.dispatchEvent(new CustomEvent("orb:engine-ready"));
          } else if (this.wantInit) {
            this.init();
          }
          return this.mode;
        });
      return this.probed;
    },
    toBrowser() {
      // The server went away mid-visit: fall back to the in-browser engine.
      this.mode = "browser";
      this.ready = false;
      this.init();
    },
    async post(path, body) {
      const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (r.status === 400) {
        let m = "bad request";
        try { m = (await r.json()).error || m; } catch (_) { /* keep */ }
        const err = new Error(m);
        err.badInput = true;
        throw err;
      }
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r;
    },
    async serverRender(spec, seed) {
      const t0 = performance.now();
      this.status("busy", "rendering");
      const r = await this.post("api/render", { spec, seed: seed | 0 });
      const buf = await r.arrayBuffer();
      const dv = new DataView(buf);
      const n = dv.getUint32(0, true);
      const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
      const off = 4 + n + ((4 - ((4 + n) % 4)) % 4);
      const i16 = new Int16Array(buf, off, (buf.byteLength - off) >> 1);
      const pcm = new Float32Array(i16.length);
      for (let i = 0; i < i16.length; i++) pcm[i] = i16[i] / 32767;
      const wallMs = Math.round(performance.now() - t0);
      this.status("ready", `rendered in ${wallMs} ms`);
      return { pcm, channels: meta.channels, sampleRate: meta.sample_rate, ms: meta.ms, wallMs, peaks: meta.peaks,
        spectrogram: meta.spectrogram, audit: meta.audit, passes: meta.passes };
    },
    async serverPlan(caption) {
      const t0 = performance.now();
      this.status("busy", "planning");
      const r = await this.post("api/plan", { caption });
      const out = await r.json();
      const wallMs = Math.round(performance.now() - t0);
      this.status("ready", out.planned ? `planned in ${wallMs} ms` : "caption read exactly");
      return { type: "planned", wallMs, ...out };
    },
    init() {
      if (this.mode !== "browser") {
        if (this.mode === "probe") { this.wantInit = true; this.probe(); }
        return;
      }
      if (this.worker || this.failed) return;
      try {
        this.worker = new Worker(`worker.js${DATA && DATA.py ? `?v=${DATA.py}` : ""}`);
      } catch (err) {
        this.fail(String(err));
        return;
      }
      this.status("busy", "starting engine");
      this.worker.onmessage = (e) => this.onMessage(e.data);
      this.worker.onerror = (e) => this.fail(e.message || "worker error");
      this.worker.postMessage({ type: "init" });
    },
    onMessage(msg) {
      if (msg.type === "status") this.status("busy", msg.text.toLowerCase());
      else if (msg.type === "ready") {
        this.ready = true;
        this.info = msg.version;
        this.status("ready", `engine ready · ${(msg.ms / 1000).toFixed(1)} s boot`);
        document.dispatchEvent(new CustomEvent("orb:engine-ready"));
      } else if (msg.type === "progress") {
        const p = this.pending.get(msg.id);
        if (p && p.onProgress) p.onProgress(msg.loaded, msg.total);
        if (msg.total) this.status("busy", `planner ${(msg.loaded / 1e6).toFixed(1)} / ${(msg.total / 1e6).toFixed(1)} MB`);
      } else if (msg.type === "result" || msg.type === "planned") {
        const p = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (p) p.resolve(msg);
        if (this.ready && !this.pending.size) {
          this.status("ready", msg.type === "planned" ? (msg.planned ? `planned in ${msg.wallMs} ms` : "caption read exactly") : `rendered in ${msg.wallMs} ms`);
        }
      } else if (msg.type === "error") {
        const p = msg.id != null && this.pending.get(msg.id);
        if (p) {
          this.pending.delete(msg.id);
          p.reject(new Error(lastLine(msg.message)));
          if (this.ready) this.status("ready", "engine ready");
        } else {
          this.fail(lastLine(msg.message));
        }
      }
    },
    async render(spec, seed = 0) {
      if ((await this.probe()) === "server" && this.mode === "server") {
        try { return await this.serverRender(spec, seed); } catch (err) { if (err.badInput) throw err; console.warn("server render failed, using the browser:", err); this.toBrowser(); }
      }
      return this.workerRender(spec, seed);
    },
    workerRender(spec, seed = 0) {
      this.init();
      if (this.failed) return Promise.reject(new Error("The in-browser engine could not start."));
      const id = ++this.seq;
      if (this.ready) this.status("busy", "rendering");
      return new Promise((resolve, reject) => {
        this.pending.set(id, { resolve, reject });
        this.worker.postMessage({ type: "render", id, spec, seed });
      });
    },
    /** Caption -> Spec. The first planned caption downloads the weights;
     * onProgress(loaded, total) follows that download. */
    async plan(caption, onProgress) {
      if ((await this.probe()) === "server" && this.mode === "server") {
        try { return await this.serverPlan(caption); } catch (err) { if (err.badInput) throw err; console.warn("server plan failed, using the browser:", err); this.toBrowser(); }
      }
      return this.workerPlan(caption, onProgress);
    },
    workerPlan(caption, onProgress) {
      this.init();
      if (this.failed) return Promise.reject(new Error("The in-browser engine could not start."));
      if (!DATA.planner) return Promise.reject(new Error("This build of the page ships no planner weights."));
      const id = ++this.seq;
      if (this.ready) this.status("busy", "planning");
      return new Promise((resolve, reject) => {
        this.pending.set(id, { resolve, reject, onProgress });
        this.worker.postMessage({ type: "plan", id, caption, planner: DATA.planner });
      });
    },
    fail(message) {
      this.failed = true;
      this.status("err", "engine unavailable");
      console.warn("ORB engine:", message);
      for (const p of this.pending.values()) p.reject(new Error(message));
      this.pending.clear();
    },
    status(kind, text) {
      for (const pill of $$("[data-engine-status]")) {
        pill.className = `status ${kind}`;
        pill.textContent = text;
      }
    },
  };
  function lastLine(s) {
    const lines = String(s).trim().split("\n").filter(Boolean);
    return lines[lines.length - 1] || String(s);
  }


  /* ------------------------------------------------------------------ */
  /* Film                                                                */
  /* ------------------------------------------------------------------ */

  async function initVideo() {
    const video = $("#demo-video");
    const fallback = $("#video-fallback");
    const showFallback = () => {
      video.hidden = true;
      fallback.hidden = false;
      drawFallback($("#video-fallback-canvas"));
    };
    const media = DATA.media || {};
    let hasVideo = media.video === true;
    let hasPoster = media.poster === true;
    if (!hasVideo) {
      // The manifest may predate the film; ask the server before giving up.
      try {
        const [v, p] = await Promise.all([video.dataset.src, video.dataset.poster].map((u) => fetch(u, { method: "HEAD" })));
        hasVideo = v.ok;
        hasPoster = hasPoster || p.ok;
      } catch (_) { hasVideo = false; }
    }
    if (!hasVideo) return showFallback();
    if (hasPoster) video.poster = video.dataset.poster;
    const cover = $("#video-cover");
    const source = document.createElement("source");
    source.src = video.dataset.src;
    source.type = "video/mp4";
    source.addEventListener("error", () => { cover.hidden = true; showFallback(); });
    video.appendChild(source);
    video.hidden = false;
    cover.hidden = false;
    const reveal = () => {
      cover.hidden = true;
      video.controls = true;
    };
    cover.addEventListener("click", () => {
      reveal();
      const p = video.play();
      if (p && p.catch) p.catch(() => {});
      video.focus();
    });
    video.addEventListener("play", () => { reveal(); stop(); });
  }
  function drawFallback(canvas) {
    const cues = (DATA && DATA.cues) || [];
    const draw = () => {
      const { ctx, w, h } = fit(canvas);
      ctx.clearRect(0, 0, w, h);
      // one soft waveform of the first cue behind the note
      const T = theme();
      const c = cues[0];
      if (!c) return;
      const bars = Math.floor(w / 9);
      const step = w / bars;
      for (let i = 0; i < bars; i++) {
        const k = Math.floor((i / bars) * c.peaks.length);
        const [lo, hi] = c.peaks[k];
        const bh = Math.max(4, Math.pow(hi - lo, 0.6) * h * 0.5);
        ctx.fillStyle = T.wave;
        ctx.globalAlpha = 0.55;
        roundBar(ctx, i * step + step * 0.25, h / 2 - bh / 2, step * 0.5, bh);
      }
      ctx.globalAlpha = 1;
    };
    draw();
    window.addEventListener("resize", debounce(draw, 150));
  }

  /* ------------------------------------------------------------------ */
  /* Prompt helper: plain words -> a grammar caption (rules, not a model) */
  /* ------------------------------------------------------------------ */

  // The planner reads one caption grammar. Anything else goes through these
  // keyword rules first, and the page shows the caption they wrote.
  const PAN_TOKENS = ["L4", "L3", "L2", "L1", "C", "R1", "R2", "R3", "R4"];
  const CAPTION_RE = /^\s*\d+(?:\.\d+)?-second\s+[a-z]+\s+cue\b/i;

  // Each sound phrase is a small recipe: events [role, offset s, dur s, gain dB, pan],
  // the time to the next sound, and optional mood / bed hints. Order matters:
  // longer phrases are matched (and blanked out) before the words inside them.
  const SOUNDS = [
    { re: /\b(?:level(?:ing)?[ -]?up|power[ -]?up|achievement|unlock(?:ed)?)\b/, label: "level up",
      ev: [["riser", 0, 1.0, -16, 0], ["ui_pop", 1.0, 0.1, -10, -0.5], ["ui_pop", 1.12, 0.1, -10, 0.5], ["bell", 1.24, 1.8, -10, 0]], next: 2.4, mood: "hopeful", ui: true },
    { re: /\b(?:game[ -]?over|you (?:lose|lost|died)|defeat(?:ed)?|death)\b/, label: "game over",
      ev: [["glitch", 0, 0.3, -12, 0], ["impact", 0.35, 0.9, -6, 0], ["sub_drop", 0.35, 2.0, -8, 0]], next: 2.0, mood: "dark", ui: true },
    { re: /\b(?:payment|purchase|checkout|success(?:ful)?|completed?|confirm(?:ed|ation)?|correct|saved|done)\b/, label: "success",
      ev: [["ui_pop", 0, 0.1, -10, -0.25], ["ui_pop", 0.12, 0.1, -10, 0.25], ["bell", 0.24, 1.6, -12, 0]], next: 1.6, mood: "bright", ui: true },
    { re: /\b(?:jump[ -]?scare|scare[ds]?|horror|scream)\b/, label: "scare",
      ev: [["swell", 0, 2.4, -18, 0], ["glitch", 2.3, 0.25, -10, 0.25], ["impact", 2.4, 0.9, -4, 0], ["sub_drop", 2.4, 1.8, -8, 0]], next: 3.0, mood: "eerie", tex: "drone" },
    { re: /\b(?:logo(?: reveal| sting| intro)?|brand(?: reveal| sting)|sting(?:er)?|reveal)\b/, label: "logo reveal",
      ev: [["whoosh", 0, 1.0, -8, -0.5], ["impact", 0.8, 0.9, -6, 0], ["bell", 0.8, 2.0, -12, 0]], next: 2.2, mood: "warm" },
    { re: /\b(?:(?:epic |cinematic |movie )?trailer(?: hit| impact| boom)?|cinematic (?:hit|impact|boom)|blockbuster)\b/, label: "trailer hit",
      ev: [["riser", 0, 2.4, -18, 0], ["impact", 2.4, 0.9, -4, 0], ["sub_drop", 2.4, 1.6, -8, 0]], next: 2.8, mood: "intense", tex: "drone" },
    { re: /\b(?:count ?down|timer|3,? ?2,? ?1)\b/, label: "countdown",
      ev: [["click", 0, 0.06, -8, 0], ["click", 1, 0.06, -8, 0], ["click", 2, 0.06, -8, 0], ["bell", 3, 1.4, -10, 0]], next: 3.8, mood: "tense", ui: true },
    { re: /\b(?:typing|keyboard|typewriter|keystrokes?)\b/, label: "typing",
      ev: [["keyboard_typing", 0, 3.0, -14, 0]], next: 3.0, ui: true },
    { re: /\b(?:(?:notification|notify|alert|new message|message|inbox|reminder)(?: ding| chime| ping| pop| sound| tone| bell)?|ping)\b/, label: "notification",
      ev: [["ui_pop", 0, 0.12, -8, -0.25], ["bell", 0.12, 1.2, -12, 0.25]], next: 1.2, mood: "bright", ui: true },
    // Natural and foley roles are physically modelled; an optional sixth
    // element [mod, value] pins the seed's variant (thunder seed % 4 = distance,
    // 0 close .. 3 far; explosion seed % 3 = distance; water drop seed % 2 = drip/plop).
    { re: /\b(?:thunder ?storms?|storms?|stormy|tempest|lightning storm)\b/, label: "thunderstorm",
      ev: [["thunder", 0, 7.0, -10, -0.5, [4, 3]], ["thunder", 5.6, 6.5, -4, 0.25, [4, 0]]], next: 11.5, mood: "dark", tex: "rain" },
    { re: /\b(?:thunder(?:clap|s)?|lightning)\b/, label: "thunder",
      ev: [["thunder", 0, 6.0, -6, 0.25]], next: 5.6, mood: "tense", tex: "rain" },
    { re: /\b(?:explosions?|explodes?|exploding|blast|boom|detonation|grenade|bomb)\b/, label: "explosion",
      ev: [["explosion", 0, 3.0, -4, 0, [3, 0]]], next: 2.8, mood: "intense" },
    { re: /\b(?:door ?slam(?:s|med)?|slam(?:s|med|ming)?(?: the door)?|(?:door|gate) (?:shuts|closes|slams))\b/, label: "door slam",
      ev: [["door_slam", 0, 0.9, -6, 0]], next: 1.0, mood: "tense" },
    { re: /\b(?:knock(?:s|ing)?(?: on the door| at the door)?|rap(?:s|ping)? on)\b/, label: "knock",
      ev: [["knock", 0, 0.7, -10, 0]], next: 0.9 },
    { re: /\b(?:foot ?steps?|walk(?:s|ing)?(?: up| in)?|steps|stomps?|running|jogging)\b/, label: "footsteps",
      ev: [["footsteps", 0, 3.0, -12, -0.25]], next: 3.0 },
    { re: /\b(?:water ?drops?|drips?|dripping|droplets?|drop of water|plops?|leak(?:ing|y)?)\b/, label: "water drip",
      ev: [["water_drop", 0, 0.35, -14, -0.5, [2, 0]], ["water_drop", 1.1, 0.35, -18, 0.5, [2, 0]], ["water_drop", 2.3, 0.35, -12, 0, [2, 1]]], next: 2.8, mood: "eerie" },
    // Everyday sounds (orb.render.natural): one physical model per role.
    { re: /\b(?:alarm clock|alarm(?: bell| ringing| goes off)?|wake[ -]?up alarm|school bell)\b/, label: "alarm clock",
      ev: [["clock_alarm", 0, 3.0, -12, 0]], next: 3.0, mood: "tense" },
    { re: /\b(?:church bells?|bell tower|tolling|tolls?|cathedral|big ben)\b/, label: "church bell",
      ev: [["church_bell", 0, 6.0, -10, 0]], next: 5.0, mood: "calm" },
    { re: /\b(?:phone(?: ring\w*| call)?|telephone|ringtone|incoming call|ringing)\b/, label: "phone ring",
      ev: [["phone_ring", 0, 4.0, -12, 0]], next: 4.0 },
    { re: /\b(?:clock(?: ticking| ticks?)?|tick[ -]?tock|ticking)\b/, label: "clock tick",
      ev: [["clock_tick", 0, 4.0, -14, 0]], next: 4.0 },
    { re: /\b(?:fireworks?|firecrackers?|new year'?s?(?: eve)?|(?:4th|fourth) of july)\b/, label: "fireworks",
      ev: [["fireworks", 0, 4.0, -8, 0]], next: 3.6, mood: "playful" },
    { re: /\b(?:bird ?song|birds?(?: singing| chirp\w*| tweet\w*)?|chirp\w*|tweet\w*|songbirds?|dawn chorus|sparrows?|robins?)\b/, label: "birds",
      ev: [["bird_chirp", 0, 3.0, -12, -0.5], ["bird_chirp", 1.8, 3.0, -16, 0.5]], next: 4.8, mood: "calm", tex: "forest" },
    { re: /\b(?:crickets?|cicadas?|insects?)\b/, label: "crickets",
      ev: [["crickets", 0, 5.0, -14, 0, [3, 2]]], next: 5.0, mood: "calm", tex: "night" },
    { re: /\b(?:frogs?|toads?|ribbit\w*|croak\w*|bullfrogs?)\b/, label: "frog",
      ev: [["frog", 0, 3.0, -12, -0.25]], next: 3.0, tex: "night" },
    { re: /\b(?:owls?|hoot\w*)\b/, label: "owl",
      ev: [["owl", 0, 3.0, -14, 0.25]], next: 3.0, mood: "eerie", tex: "night" },
    { re: /\b(?:waves?(?: crash\w*| break\w*)?|crashing waves|surf|breakers?)\b/, label: "waves",
      ev: [["wave_crash", 0, 6.0, -8, -0.25]], next: 5.0, mood: "calm", tex: "ocean" },
    { re: /\b(?:wind ?gusts?|gusts?(?: of wind)?|gusty|howling wind|wind howl\w*)\b/, label: "gust",
      ev: [["wind_gust", 0, 4.0, -10, 0]], next: 3.5, tex: "wind" },
    { re: /\b(?:splash\w*|cannonball|dives?|diving|jumps? in(?:to)?(?: the)? (?:water|pool|lake))\b/, label: "splash",
      ev: [["splash", 0, 1.2, -8, 0]], next: 1.2 },
    { re: /\b(?:glass(?:es)? (?:break\w*|shatter\w*|smash\w*)|(?:break\w*|smash\w*|drop\w*) (?:a |the )?glass(?:es)?|shatter\w*|broken (?:glass|window)|window break\w*)\b/, label: "glass break",
      ev: [["glass_break", 0, 1.6, -6, 0]], next: 1.6 },
    { re: /\b(?:mouse ?clicks?|clicking the mouse|trackpad)\b/, label: "mouse click",
      ev: [["mouse_click", 0, 0.3, -12, 0], ["mouse_click", 0.45, 0.3, -12, 0]], next: 0.9, ui: true },
    { re: /\b(?:camera(?: shutter| click| snap)?|shutter|photos?|snapshot|selfie|taking a picture)\b/, label: "camera",
      ev: [["camera_shutter", 0, 0.6, -10, 0]], next: 0.8, ui: true },
    { re: /\b(?:paper(?:s| rustl\w*)?|rustl\w*|page turn\w*|turning pages?|newspaper|crumpl\w*)\b/, label: "paper",
      ev: [["paper_rustle", 0, 2.0, -14, 0]], next: 2.0 },
    { re: /\b(?:cork(?: pops?)?|champagne|celebrat\w*|cheers|prosecco)\b/, label: "cork pop",
      ev: [["cork_pop", 0, 1.0, -10, 0]], next: 1.2, mood: "playful" },
    { re: /\b(?:coins? (?:drop\w*|fall\w*|jingl\w*|clink\w*)|coins?(?! ?(?:pick|collect|pop))|loose change|spare change|jingl\w*|clink\w*|tip jar)\b/, label: "coins",
      ev: [["coin_drop", 0, 1.5, -12, 0]], next: 1.5 },
    { re: /\b(?:metal(?: clang| hit| bang)?|clang\w*|clank\w*|anvil|pipe|pots? and pans|frying pan)\b/, label: "metal clang",
      ev: [["metal_clang", 0, 2.5, -8, 0]], next: 2.0 },
    { re: /\b(?:creak\w*|squeak\w*|old door|floorboards?|rocking chair)\b/, label: "creak",
      ev: [["wood_creak", 0, 2.0, -12, 0]], next: 2.0, mood: "eerie" },
    { re: /\b(?:heart ?beats?|heart beating|heart pounding|pulse|heart)\b/, label: "heartbeat",
      ev: [["heartbeat", 0, 4.0, -8, 0]], next: 4.0, mood: "tense" },
    { re: /\b(?:applause|applaud\w*|clapping|claps?|ovation|audience)\b/, label: "applause",
      ev: [["applause", 0, 5.0, -10, 0]], next: 4.5, mood: "hopeful" },
    { re: /\b(?:pour\w*|glass of water|filling (?:a|the) glass)\b/, label: "pouring",
      ev: [["pouring_water", 0, 3.0, -12, 0]], next: 3.0 },
    { re: /\b(?:can open\w*|opens? a can|soda|cola|beer can|fizz\w*|crack(?:s|ing)? open)\b/, label: "can opening",
      ev: [["can_opening", 0, 1.0, -10, 0]], next: 1.0 },
    { re: /\b(?:punch(?:es|ed|ing)?|kick(?:s|ed|ing)?|fist ?fight|brawl|uppercut|jab|slap)\b/, label: "punch",
      ev: [["punch", 0, 0.5, -6, 0]], next: 0.6, mood: "intense" },
    { re: /\b(?:swords?(?: fight\w*| clash\w*)?|blades?|clash\w*|duel\w*|fencing|parry|katana|sabre|saber)\b/, label: "sword clash",
      ev: [["sword_clash", 0, 1.5, -8, -0.25], ["sword_clash", 0.9, 1.5, -8, 0.25]], next: 2.2, mood: "intense" },
    { re: /\b(?:gun ?shots?|guns?|shots? fired|shoot\w*|pistol|rifle|bullets?)\b/, label: "gunshot",
      ev: [["gunshot", 0, 1.5, -6, 0]], next: 1.5, mood: "intense" },
    { re: /\b(?:car horns?|horns?|honk\w*)\b/, label: "car horn",
      ev: [["car_horn", 0, 1.0, -10, 0]], next: 1.0, tex: "traffic" },
    { re: /\b(?:sirens?|police|ambulance|fire ?truck|emergency)\b/, label: "siren",
      ev: [["siren", 0, 6.0, -12, 0]], next: 5.0, mood: "tense", tex: "traffic" },
    { re: /\b(?:engines?|motor\w*|rev(?:s|ving|ved)?|cars? (?:passing|pass(?:es)? by|driving by|drives? by)|motorbikes?|motorcycles?|trucks?)\b/, label: "engine",
      ev: [["engine", 0, 4.0, -10, 0]], next: 3.5, tex: "traffic" },
    { re: /\b(?:impacts?|hits?|thud|bang|crash(?:es)?|smash)\b/, label: "hit",
      ev: [["impact", 0, 0.9, -6, 0]], next: 0.9 },
    { re: /\b(?:whoosh(?:es)?|swoosh|swish|swipe|fly[ -]?by|flies by|pass(?:es)? by|transition|wipe|zoom|dash)\b/, label: "whoosh",
      ev: [["whoosh", 0, 0.9, -8, 0]], next: 0.8 },
    { re: /\b(?:riser|rising|rises|build[ -]?up|builds|tension|suspense|anticipation)\b/, label: "riser",
      ev: [["riser", 0, 2.4, -16, 0]], next: 2.4, mood: "tense" },
    { re: /\b(?:swell|bloom|pad|wash|shimmer)\b/, label: "swell",
      ev: [["swell", 0, 1.8, -16, 0]], next: 1.6 },
    { re: /\b(?:sub ?drop|drop|bass|rumble|rumbling|thump|earthquake)\b/, label: "sub drop",
      ev: [["sub_drop", 0, 1.8, -8, 0]], next: 1.2 },
    { re: /\b(?:glitch(?:y|es)?|error|fail(?:ed|ure)?|static|corrupted?|malfunction|robot(?:ic)?|denied|wrong|zaps?|buzz)\b/, label: "glitch",
      ev: [["glitch", 0, 0.25, -12, 0]], next: 0.5, ui: true },
    { re: /\b(?:noise sweep|sweep|hiss|steam|airlock|depressuri[sz]e|vent)\b/, label: "sweep",
      ev: [["noise_sweep", 0, 1.8, -14, 0]], next: 1.4 },
    { re: /\b(?:bell|chimes?|ding|rings?|gong)\b/, label: "bell",
      ev: [["bell", 0, 1.4, -12, 0]], next: 1.2 },
    { re: /\b(?:clicks?|taps?|ticks?|button|press|switch|toggle)\b/, label: "click",
      ev: [["click", 0, 0.06, -10, 0]], next: 0.4, ui: true },
    { re: /\b(?:(?:coin|item|gem) (?:pickup|pick[ -]up|collect(?:ed)?)|ui pop|pops?|bubble|blip|bloop|pickup|pick[ -]up|collect(?:ed|ing)?|select|like)\b/, label: "pop",
      ev: [["ui_pop", 0, 0.12, -10, 0]], next: 0.4, ui: true },
  ];
  const BEDS = [
    [/\b(?:rain\w*|storm\w*|drizzl\w*|downpour|wet)\b/, "rain"],
    [/\b(?:wind\w*|breez\w*|blizzard|desert|mountains?|gale)\b/, "wind"],
    [/\b(?:fire|fireplace|campfire|burn\w*|flames?|crackl\w*|hearth)\b/, "fire"],
    [/\b(?:traffic|street\w*|roads?|highway|cars?|downtown|intersection|crosswalk)\b/, "traffic"],
    [/\b(?:city|urban|subway|metro|skyline)\b/, "city_hum"],
    [/\b(?:ocean|sea|beach|shore|seaside|coast\w*|waves?|surf)\b/, "ocean"],
    [/\b(?:stream|brook|creek|rivers?|riverside|babbling|waterfall)\b/, "stream"],
    [/\b(?:forest|woods|woodland|jungle|park|garden|trees?|dawn|countryside)\b/, "forest"],
    [/\b(?:night\w*|swamp|pond|camp(?:ing|site)?|nocturnal)\b/, "night"],
    [/\b(?:vinyl|record player|lo-?fi|retro|nostalg\w*|old film)\b/, "vinyl"],
    [/\b(?:sci-?fi|space\w*|starship|airlock|lasers?|futur\w*|aliens?|cyber\w*|reactor)\b/, "sci_fi"],
    [/\b(?:drone|dread|dungeon|cave|underground|void|abyss|haunt\w*|ghost\w*|crypt|graveyard)\b/, "drone"],
    [/\b(?:room|indoors?|hall\w*|office|corridor|warehouse|basement|house|home|kitchen)\b/, "room_cold"],
    [/\b(?:no bed|dry|silent|silence|no background)\b/, "none"],
  ];
  const MOOD_RULES = [
    [/\b(?:calm\w*|relax\w*|gentle|soft\w*|peace\w*|quiet|sleep\w*|meditat\w*)\b/, "calm"],
    [/\b(?:warm\w*|cozy|cosy|friendly|soothing)\b/, "warm"],
    [/\b(?:playful|fun|funny|cute|cartoon\w*|bouncy|silly|quirky)\b/, "playful"],
    [/\b(?:hope\w*|uplift\w*|inspir\w*|optimis\w*|victor\w*|triumph\w*)\b/, "hopeful"],
    [/\b(?:bright|happy|cheer\w*|joy\w*|positive|crisp)\b/, "bright"],
    [/\b(?:tense|nervous|alarm\w*|urgent|danger\w*|anxious)\b/, "tense"],
    [/\b(?:dark|ominous|sad|gloom\w*|doom\w*|grim|melanchol\w*|night)\b/, "dark"],
    [/\b(?:eerie|creep\w*|spook\w*|ghost\w*|haunt\w*|weird|unsettling|myster\w*)\b/, "eerie"],
    [/\b(?:neutral|plain|simple|minimal)\b/, "neutral"],
    [/\b(?:intense|epic|action|aggressive|huge|massive|powerful|heavy|big|loud)\b/, "intense"],
  ];
  const WORD_EXAMPLES = ["door slam in the rain", "spaceship airlock opens, tense", "coin pickup, then level up", "thunderstorm", "footsteps, door slam, then a knock",
    "birds singing in the forest at dawn", "waves on the beach", "crickets and an owl at night", "glass breaking in the kitchen", "phone ringing in the office",
    "sword fight", "car horn in traffic", "applause", "fireworks", "heartbeat, tense", "pouring a glass of water"];

  function hashText(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
    return h >>> 0;
  }
  const firstMatch = (rules, text) => {
    let best = null;
    for (const [re, value] of rules) {
      const m = re.exec(text);
      if (m && (!best || m.index < best.at)) best = { at: m.index, value, word: m[0] };
    }
    return best;
  };

  /** Plain words -> { caption, notes } or null when nothing maps. */
  function translate(raw) {
    const text = ` ${String(raw).toLowerCase().replace(/[^a-z0-9.,' -]+/g, " ")} `;
    // 1. sound phrases, in the order they appear
    let rest = text;
    const found = [];
    for (const s of SOUNDS) {
      const re = new RegExp(s.re.source, "g");
      let m;
      while ((m = re.exec(rest))) {
        found.push({ s, at: m.index, word: m[0] });
        rest = rest.slice(0, m.index) + " ".repeat(m[0].length) + rest.slice(m.index + m[0].length);
      }
    }
    found.sort((a, b) => a.at - b.at);
    // "typing on a keyboard", "a ding, a chime": one sound named twice, close together.
    for (let i = found.length - 1; i > 0; i--) {
      const a = found[i - 1], b = found[i];
      const gap = text.slice(a.at + a.word.length, b.at);
      if (a.s === b.s && gap.length <= 12 && !/\bthen\b|\bafter\b/.test(gap)) found.splice(i, 1);
    }
    // 2. bed and mood
    const bedHit = firstMatch(BEDS, text);
    const moodHit = firstMatch(MOOD_RULES, text);
    if (!found.length && (!bedHit || bedHit.value === "none")) return null;
    let texture = bedHit ? bedHit.value : null;
    const hinted = found.find((f) => f.s.tex);
    if (!texture) {
      if (hinted) texture = hinted.s.tex;
      else if (found.every((f) => f.s.ui)) texture = "none";
    }
    const moodFrom = found.find((f) => f.s.mood);
    let mood = moodHit ? moodHit.value : moodFrom && moodFrom.s.mood;
    if (!mood) mood = texture === "rain" || texture === "fire" ? "calm" : texture === "drone" ? "dark" : "neutral";
    if (!texture) texture = ["tense", "dark", "eerie", "intense"].includes(mood) ? "drone" : "room_cold";
    const bed = texture !== "none";
    // 3. sounds in sequence
    const side = /\bleft\b/.test(text) && !/\bright\b/.test(text) ? -1 : /\bright\b/.test(text) && !/\bleft\b/.test(text) ? 1 : 0;
    const seed0 = hashText(text);
    const events = [];
    const used = [];
    let at = bed ? 0.5 : 0.1;
    for (const f of found) {
      if (events.length + f.s.ev.length > MAX_EVENTS) continue;
      for (const [role, off, dur, gain, pan, variant] of f.s.ev) {
        events.push({ role, t: round2(at + off), dur, gain_db: gain, pan: pan === 0 && side ? side * 0.5 : pan, variant });
      }
      used.push(f);
      at += f.s.next + 0.15;
    }
    // 4. duration: asked for, or long enough for the last tail
    const asked = /(\d+(?:\.\d+)?)\s*(?:s|secs?|seconds?)\b/.exec(text);
    let duration;
    if (asked) duration = clamp(Math.round(parseFloat(asked[1]) * 2) / 2, 2, 30);
    else {
      const end = events.reduce((m, e) => Math.max(m, e.t + e.dur), 0);
      duration = events.length ? clamp(Math.ceil((end + (bed ? 1.0 : 0.4)) * 2) / 2, 2, 30) : 8;
    }
    const pinSeed = (seed, variant) => {
      if (!variant) return seed;
      const [mod, value] = variant;
      let out = seed - (seed % mod) + value;
      while (out > 0xff) out -= mod;
      return out;
    };
    const kept = events.filter((e) => e.t <= duration - 0.05).map(({ variant, ...e }, i) => ({
      ...e,
      dur: round2(Math.max(0.05, Math.min(e.dur, duration - e.t))),
      seed: pinSeed((seed0 + i * 37) & 0xff, variant),
    }));
    kept.sort((a, b) => a.t - b.t);
    const evText = kept.map((e) => `${words(e.role)} @${e.t.toFixed(2)} ${e.dur.toFixed(2)}s ${e.gain_db}dB ${PAN_TOKENS[clamp(Math.round(e.pan * 4) + 4, 0, 8)]} #${e.seed.toString(16).padStart(2, "0")}`);
    const caption = `${duration.toFixed(2)}-second ${mood} cue, ${bed ? `${words(texture)} bed` : "no bed"}${evText.length ? `, with ${evText.join(", ")}` : ""}`;
    const notes = [
      ...used.map((f) => `“${f.word.trim()}” → ${[...new Set(f.s.ev.map((e) => words(e[0])))].join(" + ")}`),
      `bed: ${bed ? words(texture) : "none"}${bedHit ? "" : hinted && hinted.s.tex === texture ? ` (from “${hinted.word.trim()}”)` : " (default)"}`,
      `mood: ${mood}${moodHit ? "" : moodFrom ? ` (from “${moodFrom.word.trim()}”)` : " (default)"}`,
      `${duration.toFixed(1)} s`,
    ];
    if (found.length > used.length) notes.push(`dropped ${found.length - used.length} sound(s): max ${MAX_EVENTS} events`);
    return { caption, notes: notes.join(" · "), planned: bed };
  }

  /* ------------------------------------------------------------------ */
  /* Hero: prompt box -> planner -> renderer                              */
  /* ------------------------------------------------------------------ */

  let loadIntoPlayground = null;

  function wavBlob(pcm, channels, sr) {
    const n = pcm.length;
    const buf = new ArrayBuffer(44 + n * 2);
    const v = new DataView(buf);
    const str = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); str(8, "WAVE");
    str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, channels, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * channels * 2, true); v.setUint16(32, channels * 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++) {
      const x = clamp(pcm[i], -1, 1);
      v.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  function slug(text, fallback = "sound") {
    let s = String(text).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    if (s.length > 48) s = s.slice(0, 48).replace(/-[^-]*$/, "");
    return s || fallback;
  }
  function saveBlob(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = el("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }
  /** AudioBuffer -> interleaved PCM -> WAV blob. */
  function bufferWav(buf) {
    const ch = buf.numberOfChannels, n = buf.length;
    const pcm = new Float32Array(n * ch);
    for (let c = 0; c < ch; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < n; i++) pcm[i * ch + c] = d[i];
    }
    return wavBlob(pcm, ch, buf.sampleRate);
  }
  /** Fresh event seeds for another take of the same score. Thunder keeps
   * its distance (seed % 4). */
  function reseed(spec) {
    const s = clone(spec);
    for (const e of s.events) {
      const keep = e.role === "thunder" ? 3 : 0;
      let seed;
      do { seed = (Math.floor(Math.random() * 256) & ~keep) | ((e.seed | 0) & keep); } while (seed === e.seed);
      e.seed = seed;
    }
    return s;
  }

  function initGenerator() {
    const form = $("#gen-form");
    const input = $("#gen-input");
    const go = $("#gen-go");
    const helper = $("#gen-helper");
    const capBox = $("#gen-caption");
    const notes = $("#gen-notes");
    const runBtn = $("#gen-run");
    const msg = $("#gen-msg");
    const prog = $("#gen-progress");
    const fill = $("#gen-progress-fill");
    const progText = $("#gen-progress-text");
    const out = $("#gen-out");
    const playBtn = $("#gen-play");
    const wave = $("#gen-wave");
    const pathEl = $("#gen-path");
    const capEl = $("#gen-cap");
    const specEl = $("#gen-spec");
    const specSum = $("#gen-spec-sum");
    let busy = false;
    let result = null;

    const say = (text, isErr = false) => {
      msg.hidden = !text;
      msg.textContent = text || "";
      msg.classList.toggle("err", isErr);
    };
    const chip = (root, label, title, onClick) => {
      const c = el("button", "gen-chip", label);
      c.type = "button";
      if (title) c.title = title;
      c.addEventListener("click", onClick);
      root.appendChild(c);
      return c;
    };
    for (const ex of DATA.prompts || []) {
      chip($("#gen-chips"), ex.label, ex.caption, () => {
        input.value = ex.caption;
        run(ex.caption, false);
      });
    }
    for (const w of WORD_EXAMPLES) {
      chip($("#gen-words"), w, null, () => {
        input.value = w;
        submit();
      });
    }

    const redraw = (p = -1) => result && drawWave(wave, result.res, { progress: p });
    function playResult() {
      if (!result) return;
      play(result.buffer, "gen", (p) => redraw(p), () => { setPlayIcon(playBtn, false); redraw(); });
      setPlayIcon(playBtn, true);
    }

    function submit() {
      const text = input.value.trim();
      if (!text) {
        say("Type a few words about the sound, or tap an example below.");
        input.focus();
        return;
      }
      if (CAPTION_RE.test(text)) {
        helper.hidden = true;
        run(text, false);
        return;
      }
      // v8 reads plain words itself: the planner gets the raw text first. The
      // keyword helper's reading is offered next to the result as a second
      // opinion (it is better at soft/loud and ignores words it has no sound for).
      const t = translate(text);
      if (t) {
        capBox.value = t.caption;
        notes.textContent = t.notes;
      }
      helper.hidden = !t;
      run(text.slice(0, 240), false, true);
    }

    async function run(caption, viaHelper, keepHelper = false) {
      if (busy) return;
      caption = caption.trim();
      if (!caption) return;
      busy = true;
      go.disabled = runBtn.disabled = true;
      go.textContent = "Generating…";
      if (!viaHelper && !keepHelper) helper.hidden = true;
      if (player.token === "gen") stop();
      say(engine.ready || engine.mode === "server" ? "" : "Starting the in-browser engine (Python + numpy, once per visit)…");
      try {
        const planned = await engine.plan(caption, (loaded, total) => {
          prog.hidden = false;
          const pct = total ? loaded / total : 0;
          fill.style.width = `${(pct * 100).toFixed(1)}%`;
          progText.textContent = `Downloading the planner · ${(loaded / 1e6).toFixed(1)} of ${(total / 1e6).toFixed(1)} MB`;
          if (pct >= 1) {
            progText.textContent = "Planner downloaded · writing the score…";
          }
          say("");
        });
        prog.hidden = true;
        say("Rendering…");
        const res = await engine.render(planned.spec, 0);
        result = { caption, planned, res, buffer: pcmToBuffer(res.pcm, res.channels, res.sampleRate) };
        show();
        say("");
        playResult();
      } catch (err) {
        console.warn(err);
        prog.hidden = true;
        say(`That did not work: ${err.message}`, true);
      } finally {
        busy = false;
        go.disabled = runBtn.disabled = false;
        go.textContent = "Generate";
      }
    }

    function show() {
      const { planned, res, caption } = result;
      const spec = planned.spec;
      out.hidden = false;
      const how = planned.planned
        ? `Planner · ${planned.tokens} tokens in ${(planned.ms / 1000).toFixed(1)} s`
        : "No bed: read exactly from the caption, planner skipped";
      pathEl.textContent = `${how} · rendered in ${res.wallMs} ms · audit ${res.passes ? "passed" : "flagged"}`;
      specSum.textContent = `${num(spec.duration)} s · ${spec.texture === "none" ? "no bed" : words(spec.texture)} · ${spec.mood} · ${spec.events.length} event${spec.events.length === 1 ? "" : "s"}`;
      capEl.textContent = caption;
      specEl.innerHTML = highlight(prettySpec(spec));
      requestAnimationFrame(() => {
        redraw();
        const r = out.getBoundingClientRect();
        if (r.bottom > window.innerHeight || r.top < 0) out.scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
    }

    form.addEventListener("submit", (e) => { e.preventDefault(); submit(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
    });
    runBtn.addEventListener("click", () => run(capBox.value, true));
    capBox.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); run(capBox.value, true); }
    });
    playBtn.addEventListener("click", () => (player.token === "gen" ? stop() : playResult()));
    wave.addEventListener("click", () => (player.token === "gen" ? stop() : playResult()));
    $("#gen-dl").addEventListener("click", () => {
      if (!result) return;
      const { res, caption } = result;
      saveBlob(wavBlob(res.pcm, res.channels, res.sampleRate), `orb-${slug(caption)}${result.take ? `-take${result.take + 1}` : ""}.wav`);
    });
    const againBtn = $("#gen-again");
    againBtn.addEventListener("click", async () => {
      if (!result || busy) return;
      busy = true;
      againBtn.disabled = go.disabled = true;
      againBtn.textContent = "Rendering…";
      if (player.token === "gen") stop();
      try {
        const spec = reseed(result.planned.spec);
        const res = await engine.render(spec, 0);
        result = { ...result, planned: { ...result.planned, spec }, res, take: (result.take || 0) + 1,
          buffer: pcmToBuffer(res.pcm, res.channels, res.sampleRate) };
        show();
        playResult();
      } catch (err) {
        console.warn(err);
        say(`That did not work: ${err.message}`, true);
      } finally {
        busy = false;
        againBtn.disabled = go.disabled = false;
        againBtn.textContent = "New take";
      }
    });
    $("#gen-remix").addEventListener("click", () => {
      if (result && loadIntoPlayground) loadIntoPlayground(result.planned.spec, { peaks: result.res.peaks, buffer: result.buffer });
    });
    window.addEventListener("resize", debounce(() => redraw(), 120));
  }

  /* ------------------------------------------------------------------ */
  /* Library: practical cues, grouped                                    */
  /* ------------------------------------------------------------------ */

  function initGallery() {
    const root = $("#gallery");
    const tabsEl = $("#lib-tabs");
    const playAll = $("#lib-playall");
    const order = DATA.groups || [];
    const lib = DATA.cues.filter((c) => c.group && c.group !== "Building blocks")
      .map((c, i) => [c, i]).sort((a, b) => (order.indexOf(a[0].group) - order.indexOf(b[0].group)) || a[1] - b[1]).map((x) => x[0]);
    const blocks = DATA.cues.filter((c) => c.group === "Building blocks");
    const groups = (DATA.groups || []).filter((g) => lib.some((c) => c.group === g));
    let current = "All";
    let chain = 0;          // id of the running "play all", 0 when idle
    const cards = new Map(); // name -> {card, btn, redraw}

    function stopChain() {
      chain = 0;
      setPlayIcon(playAll, false);
      $("span", playAll).textContent = "Play all";
    }

    function makeCard(cue, small) {
      const card = el("article", small ? "cue block" : "cue");
      const evN = (cue.spec && cue.spec.events.length) || 0;
      card.innerHTML = `
        <div class="cue-top">
          <div class="cue-head">
            ${small ? "" : `<div class="cue-group">${esc(cue.group)}</div>`}
            <h3 class="cue-title">${esc(cue.title)}</h3>
          </div>
          <button class="play" type="button" data-label="${esc(cue.title)}" aria-label="Play ${esc(cue.title)}" aria-pressed="false">
            <svg class="i-play" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="${PLAY_D}"/></svg>
          </button>
        </div>
        <p class="cue-cap">${esc(cue.use || cue.blurb || "")}</p>
        <canvas aria-hidden="true"></canvas>
        ${small ? "" : `
        <div class="cue-foot">
          <span class="cue-len">${Number(cue.duration).toFixed(1)} s · ${evN} event${evN === 1 ? "" : "s"}${cue.spec && cue.spec.texture !== "none" ? ` · ${esc(words(cue.spec.texture))}` : ""}</span>
          <span class="cue-tools">
            <a class="cue-dl" href="${esc(cue.audio)}" download="orb-${esc(cue.name)}.mp3" aria-label="Download ${esc(cue.title)} as MP3"><svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M7.25 1.5h1.5v7.19l2.47-2.47 1.06 1.06L8 11.56 3.72 7.28l1.06-1.06 2.47 2.47V1.5zM2.5 12.5h11V14h-11z"/></svg>MP3</a>
            <button class="cue-spec-btn" type="button" aria-expanded="false">Spec</button>
          </span>
        </div>
        <div class="cue-spec" hidden>
          <p class="cue-caption">${esc(cue.caption || "")}</p>
          <pre class="try-spec">${cue.spec ? highlight(prettySpec(cue.spec)) : ""}</pre>
          <button class="cue-copy" type="button">Copy JSON</button>
        </div>`}`;
      const canvas = $("canvas", card);
      const btn = $(".play", card);
      const redraw = (p = -1) => drawWave(canvas, cue, { progress: p, bars: small ? 28 : 40 });
      const start = async (fromChain) => {
        if (!fromChain && chain) stopChain();
        if (player.token === cue.name && !fromChain) return stop();
        try {
          const buf = await loadBuffer(cue.audio);
          const myChain = fromChain ? chain : 0;
          play(buf, cue.name, (p) => redraw(p), () => {
            const natural = actx && actx.currentTime - player.start >= player.dur - 0.05;
            setPlayIcon(btn, false);
            card.classList.remove("on");
            redraw();
            if (myChain && myChain === chain) {
              if (natural) setTimeout(() => myChain === chain && advance(cue), 450);
              else stopChain();
            }
          });
          setPlayIcon(btn, true);
          card.classList.add("on");
        } catch (err) {
          console.warn(err);
          if (fromChain) stopChain();
        }
      };
      btn.addEventListener("click", () => start(false));
      canvas.addEventListener("click", () => start(false));
      const specBtn = $(".cue-spec-btn", card);
      if (specBtn) {
        const box = $(".cue-spec", card);
        specBtn.addEventListener("click", () => {
          const open = box.hidden;
          box.hidden = !open;
          specBtn.setAttribute("aria-expanded", String(open));
          specBtn.textContent = open ? "Hide" : "Spec";
        });
        $(".cue-copy", card).addEventListener("click", async (e) => {
          const b = e.currentTarget;
          try {
            await navigator.clipboard.writeText(prettySpec(cue.spec));
            b.textContent = "Copied";
          } catch (_) {
            b.textContent = "Select the JSON above";
          }
          setTimeout(() => { b.textContent = "Copy JSON"; }, 1600);
        });
      }
      cards.set(cue.name, { card, start, redraw });
      return card;
    }

    const visible = () => lib.filter((c) => current === "All" || c.group === current);
    function advance(prev) {
      const list = visible();
      const i = list.indexOf(prev);
      if (i < 0 || i + 1 >= list.length) return stopChain();
      cards.get(list[i + 1].name).start(true);
    }
    playAll.addEventListener("click", () => {
      if (chain) { stopChain(); stop(); return; }
      const list = visible();
      if (!list.length) return;
      chain = Date.now();
      setPlayIcon(playAll, true);
      $("span", playAll).textContent = "Stop";
      cards.get(list[0].name).start(true);
    });

    const tabs = ["All", ...groups].map((g) => {
      const n = g === "All" ? lib.length : lib.filter((c) => c.group === g).length;
      const b = el("button", "lib-tab");
      b.type = "button";
      b.setAttribute("role", "tab");
      b.innerHTML = `${esc(g)}<span>${n}</span>`;
      b.addEventListener("click", () => select(g));
      b.addEventListener("keydown", (e) => {
        const all = ["All", ...groups];
        const i = all.indexOf(g);
        const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: all.length - 1 }[e.key];
        if (to == null) return;
        e.preventDefault();
        const j = (to + all.length) % all.length;
        select(all[j]);
        tabs[j].focus();
      });
      tabsEl.appendChild(b);
      return b;
    });
    function select(g) {
      if (chain) { stopChain(); stop(); }
      current = g;
      tabs.forEach((t, k) => {
        const on = (k === 0 ? "All" : groups[k - 1]) === g;
        t.setAttribute("aria-selected", String(on));
        t.tabIndex = on ? 0 : -1;
      });
      for (const c of lib) cards.get(c.name).card.hidden = !(g === "All" || c.group === g);
      $("span", playAll).textContent = "Play all";
      playAll.setAttribute("aria-label", g === "All" ? "Play all cues" : `Play all ${g} cues`);
      requestAnimationFrame(() => lib.forEach((c) => cards.get(c.name).redraw()));
    }

    root.innerHTML = "";
    for (const cue of lib) root.appendChild(makeCard(cue, false));
    if (blocks.length) {
      const row = $("#blocks");
      for (const cue of blocks) row.appendChild(makeCard(cue, true));
      $("#blocks-wrap").hidden = false;
    }
    select("All");
    requestAnimationFrame(() => blocks.forEach((c) => cards.get(c.name).redraw()));
    window.addEventListener("resize", debounce(() => cards.forEach((c) => c.redraw()), 120));
  }

  /* ------------------------------------------------------------------ */
  /* Playground: preset scores, touch controls, live render              */
  /* ------------------------------------------------------------------ */

  // Mirrors orb.grammar / orb.score; the engine validates again on render.
  const ROLES = ["whoosh", "impact", "ui_pop", "riser", "swell", "click", "bell", "sub_drop", "glitch", "noise_sweep"];
  const TEXTURES = ["room_cold", "wind", "city_hum", "vinyl", "drone", "rain", "fire", "sci_fi", "none"];
  const MOODS = ["calm", "warm", "playful", "hopeful", "bright", "tense", "dark", "eerie", "neutral", "intense"];
  const MAX_EVENTS = 8;
  const LANE = 22;   // px per marker-label lane
  const AXIS = 22;   // px for the time labels
  const words = (s) => String(s).replace(/_/g, " ");
  const clone = (x) => JSON.parse(JSON.stringify(x));
  const panPct = (p) => Math.round(Math.abs(clamp(p, -1, 1)) * 100);
  const panLabel = (p) => (panPct(p) === 0 ? "centre" : `${p < 0 ? "L" : "R"} ${panPct(p)}%`);
  const panWords = (p) => (panPct(p) === 0 ? "centre" : `${panPct(p)}% ${p < 0 ? "left" : "right"}`);
  // Apex fractions of the rising roles (orb.grammar.EVENT_STYLE): a new take
  // keeps their peak where it was and only stretches the build.
  const APEX = { riser: 0.92, swell: 0.9 };
  const GRIP = '<svg class="pg-grip" viewBox="0 0 12 8" aria-hidden="true"><path d="M3.5 1 1 4l2.5 3M8.5 1 11 4 8.5 7" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const DICE = '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="2.5" y="2.5" width="11" height="11" rx="2.5" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="5.8" cy="5.8" r="1.1" fill="currentColor"/><circle cx="10.2" cy="10.2" r="1.1" fill="currentColor"/><circle cx="10.2" cy="5.8" r="1.1" fill="currentColor"/><circle cx="5.8" cy="10.2" r="1.1" fill="currentColor"/></svg>';

  /** Waveform for the playground: bars in the wave band, time axis below,
   * faint spans for the events of the active marker. */
  function drawScore(canvas, peaks, spec, opts) {
    const { ctx, w, h } = fit(canvas);
    const T = theme();
    ctx.clearRect(0, 0, w, h);
    const top = opts.top;
    const areaH = Math.max(20, h - top - AXIS);
    const d = spec.duration;
    const X = (t) => (t / d) * w;
    for (const i of opts.spans || []) {
      const e = spec.events[i];
      if (!e) continue;
      ctx.fillStyle = T.sage;
      ctx.globalAlpha = 0.16;
      const x0 = X(e.t), x1 = X(Math.min(d, e.t + e.dur));
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x0, top, Math.max(2, x1 - x0), areaH, 6);
      else ctx.rect(x0, top, Math.max(2, x1 - x0), areaH);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    const bars = Math.max(24, Math.floor(w / 5));
    const step = w / bars;
    const bw = Math.max(1.5, step * 0.56);
    let maxAbs = 1e-6;
    for (const [a, b] of peaks) maxAbs = Math.max(maxAbs, Math.abs(a), Math.abs(b));
    const mid = top + areaH / 2;
    const base = opts.dim ? 0.38 : 1;
    for (let i = 0; i < bars; i++) {
      const a = Math.floor((i / bars) * peaks.length);
      const b = Math.max(a + 1, Math.floor(((i + 1) / bars) * peaks.length));
      let amp = 0;
      for (let k = a; k < b && k < peaks.length; k++) amp = Math.max(amp, peaks[k][1] - peaks[k][0]);
      const norm = Math.pow(amp / (2 * maxAbs), 0.7);
      const bh = Math.max(bw, norm * areaH * 0.94);
      ctx.fillStyle = opts.progress >= 0 && (i + 0.5) / bars <= opts.progress ? T.played : T.wave;
      ctx.globalAlpha = norm < 0.06 ? base * 0.5 : base;
      roundBar(ctx, i * step + (step - bw) / 2, mid - bh / 2, bw, bh);
    }
    ctx.globalAlpha = 1;
    ctx.font = "400 11px 'DM Mono', ui-monospace, monospace";
    ctx.fillStyle = T.muted;
    ctx.textBaseline = "alphabetic";
    const every = w / d < 34 ? 2 : 1;
    for (let s = 0; s <= d + 1e-9; s += every) {
      const x = X(s);
      ctx.textAlign = s === 0 ? "left" : s / d > 0.97 ? "right" : "center";
      ctx.fillText(`${s}s`, x, h - 4);
    }
  }

  function fmtNum(x, k) {
    return Number.isInteger(x) && k !== "seed" ? x.toFixed(1) : String(round2(x));
  }

  /** The Spec as compact, highlighted JSON. Values that differ from the
   * preset are marked; the active marker's events get a line tint. */
  function specHTML(spec, base, active, narrow) {
    const changed = (a, b) => JSON.stringify(a) !== JSON.stringify(b);
    const lit = (v, k) => (typeof v === "string" ? `"${esc(v)}"` : fmtNum(v, k));
    const field = (k, v, was) => changed(v, was)
      ? `<span class="k">"${k}"</span>: <span class="hot">${lit(v, k)}</span>`
      : `<span class="k">"${k}"</span>: <span class="${typeof v === "string" ? "s" : "n"}">${lit(v, k)}</span>`;
    const head = [field("duration", spec.duration, base.duration), field("texture", spec.texture, base.texture)];
    if (spec.bed_db != null) head.push(field("bed_db", spec.bed_db, base.bed_db));
    head.push(field("mood", spec.mood, base.mood));
    const lines = [];
    if (narrow) {
      lines.push(`{ ${head.slice(0, 2).join(", ")},`);
      lines.push(`  ${head.slice(2).join(", ")},`);
    } else {
      lines.push(`{ ${head.join(", ")},`);
    }
    lines.push(`  <span class="k">"events"</span>: [`);
    const sameShape = spec.events.length === base.events.length && spec.events.every((e, i) => e.role === base.events[i].role);
    spec.events.forEach((e, i) => {
      const b = sameShape ? base.events[i] : {};
      const keys = narrow
        ? ["role", "t", ...["dur", "gain_db", "pan", "seed"].filter((k) => !sameShape || changed(e[k], b[k]))]
        : ["role", "t", "dur", "gain_db", "pan", "seed"];
      const body = keys.map((k) => field(k, e[k], sameShape ? b[k] : e[k])).join(", ");
      const comma = i < spec.events.length - 1 ? "," : "";
      const on = active && active.includes(i);
      lines.push(`<span class="ln${on ? " on" : ""}">    { ${body} }${comma}</span>`);
    });
    lines.push("  ] }");
    return lines.join("\n");
  }

  /** Client-side check, so a typo gets a clear message before the engine. */
  function validateSpec(s) {
    const isNum = (x) => typeof x === "number" && Number.isFinite(x);
    if (!s || typeof s !== "object" || Array.isArray(s)) return "The Spec must be a JSON object.";
    const extra = Object.keys(s).filter((k) => !["duration", "texture", "mood", "events", "bed_db"].includes(k));
    if (extra.length) return `Unknown field "${extra[0]}".`;
    if (!isNum(s.duration) || s.duration < 0.5 || s.duration > 30) return "duration must be a number from 0.5 to 30 seconds.";
    if (!TEXTURES.includes(s.texture)) return `texture must be one of: ${TEXTURES.join(", ")}.`;
    if (!MOODS.includes(s.mood)) return `mood must be one of: ${MOODS.join(", ")}.`;
    if (s.bed_db != null) {
      if (!isNum(s.bed_db) || s.bed_db < -60 || s.bed_db > -6) return "bed_db must be a number from -60 to -6.";
      if (s.texture === "none") return 'bed_db has no effect with texture "none"; drop one of them.';
    }
    if (!Array.isArray(s.events)) return "events must be a list.";
    if (s.events.length > MAX_EVENTS) return `At most ${MAX_EVENTS} events.`;
    if (s.texture === "none" && !s.events.length) return 'texture "none" needs at least one event.';
    for (const [i, e] of s.events.entries()) {
      const at = `events[${i}]`;
      if (!e || typeof e !== "object") return `${at} must be an object.`;
      const bad = Object.keys(e).filter((k) => !["role", "t", "dur", "gain_db", "pan", "seed"].includes(k));
      if (bad.length) return `${at}: unknown field "${bad[0]}".`;
      if (!ROLES.includes(e.role)) return `${at}.role must be one of: ${ROLES.join(", ")}.`;
      if (!isNum(e.t) || e.t < 0 || e.t >= s.duration) return `${at}.t must be inside the cue (0 to ${s.duration}).`;
      if (!isNum(e.dur) || e.dur <= 0.02 || e.dur > 10) return `${at}.dur must be from 0.03 to 10 seconds.`;
      if (e.gain_db != null && (!isNum(e.gain_db) || e.gain_db < -60 || e.gain_db > 0)) return `${at}.gain_db must be from -60 to 0.`;
      if (e.pan != null && (!isNum(e.pan) || e.pan < -1 || e.pan > 1)) return `${at}.pan must be from -1 to 1.`;
      if (e.seed != null && (!Number.isInteger(e.seed) || e.seed < 0 || e.seed > 255)) return `${at}.seed must be an integer from 0 to 255.`;
    }
    return null;
  }

  function prettySpec(spec) {
    const head = { ...spec };
    delete head.events;
    const top = JSON.stringify(head).slice(1, -1).replace(/,"/g, ', "').replace(/":/g, '": ');
    const evs = spec.events.map((e) => "    " + JSON.stringify(e).replace(/,"/g, ', "').replace(/":/g, '": ').replace(/^\{/, "{ ").replace(/\}$/, " }"));
    return `{\n  ${top},\n  "events": [\n${evs.join(",\n")}\n  ]\n}`;
  }

  function initPlayground() {
    const presets = (DATA.playground || []).filter((p) => p && p.spec && p.peaks);
    const root = $("#try");
    if (!presets.length) { root.hidden = true; return; }
    const chipsEl = $("#pg-chips");
    const card = $("#pg-card");
    const blurb = $("#pg-blurb");
    const resetBtn = $("#pg-reset");
    const dlBtn = $("#pg-dl");
    const viz = $("#pg-viz");
    const canvas = $("#try-canvas");
    const marksEl = $("#pg-marks");
    const head = $("#try-playhead");
    const btn = $("#try-play");
    const controlsEl = $("#pg-controls");
    const specEl = $("#try-spec");
    const editToggle = $("#pg-edit-toggle");
    const jsonBox = $("#pg-json");
    const jsonText = $("#pg-json-text");
    const jsonMsg = $("#pg-json-msg");
    const jsonGo = $("#pg-json-go");
    btn.dataset.label = "the score";

    const states = presets.map((p) => ({
      preset: p,
      spec: clone(p.spec),
      current: { peaks: p.peaks, url: p.audio, buffer: null },
      renderedKey: JSON.stringify(p.spec),
      busy: false, dirty: false, failedKey: null,
    }));
    let st = states[0];
    let marks = [];
    let activeMark = -1;
    let lanes = 1;
    let syncers = [];
    const keyOf = (s) => JSON.stringify(s.spec);
    const isStale = () => keyOf(st) !== st.renderedKey;
    const topPx = () => lanes * LANE + 6;

    /* -- chips ------------------------------------------------------- */
    const chips = presets.map((p, i) => {
      const c = el("button", "pg-chip");
      c.type = "button";
      c.id = `pg-tab-${i}`;
      c.setAttribute("role", "tab");
      c.setAttribute("aria-controls", "pg-card");
      c.innerHTML = `<span class="pg-chip-name">${esc(p.title)}</span><span class="pg-chip-tag">${esc(p.tag || "")}</span>`;
      c.addEventListener("click", () => {
        const same = states[i] === st;
        select(i);
        // A tap is a "touch and hear" gesture: play the preset right away.
        if (!same || player.token !== "try") playCurrent();
      });
      c.addEventListener("keydown", (e) => {
        const n = presets.length;
        const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: n - 1 }[e.key];
        if (to == null) return;
        e.preventDefault();
        const j = (to + n) % n;
        select(j);
        chips[j].focus();
      });
      chipsEl.appendChild(c);
      return c;
    });

    /* -- markers ----------------------------------------------------- */
    function computeMarks() {
      const spec = st.spec;
      const base = st.preset.spec;
      const sameShape = spec.events.length === base.events.length && spec.events.every((e, i) => e.role === base.events[i].role);
      const out = [];
      const used = new Set();
      if (sameShape) {
        for (const m of st.preset.marks || []) {
          out.push({ label: m.label, events: m.events.slice() });
          m.events.forEach((i) => used.add(i));
        }
      }
      // Anything left: events that start together share a marker.
      const rest = spec.events.map((e, i) => i).filter((i) => !used.has(i));
      const byT = new Map();
      for (const i of rest) {
        const k = spec.events[i].t.toFixed(2);
        if (!byT.has(k)) byT.set(k, []);
        byT.get(k).push(i);
      }
      for (const evs of byT.values()) {
        const roles = evs.map((i) => words(spec.events[i].role));
        out.push({ label: roles.length > 2 ? `${roles[0]} +${roles.length - 1}` : roles.join(" + "), events: evs });
      }
      out.sort((a, b) => spec.events[a.events[0]].t - spec.events[b.events[0]].t);
      return out;
    }

    function markChanged(m) {
      const base = st.preset.spec.events;
      return m.events.some((i) => base[i] && base[i].role === st.spec.events[i].role && base[i].t !== st.spec.events[i].t);
    }

    function buildMarks() {
      marksEl.innerHTML = "";
      marks = computeMarks();
      marks.forEach((m, k) => {
        const b = el("button", "pg-mark");
        b.type = "button";
        b.setAttribute("role", "slider");
        b.setAttribute("aria-valuemin", "0");
        b.setAttribute("aria-valuemax", String(st.spec.duration));
        b.setAttribute("aria-label", `${m.label} time`);
        b.innerHTML = `<span class="pg-mark-label">${GRIP}${esc(m.label)}</span><span class="pg-mark-line" aria-hidden="true"></span>`;
        m.node = b;
        m.labelNode = b.firstChild;
        bindMark(m, k);
        marksEl.appendChild(b);
      });
      layoutMarks();
    }

    function layoutMarks() {
      const w = viz.clientWidth;
      const d = st.spec.duration;
      const ends = [];
      const placed = marks.map((m) => {
        const x = (st.spec.events[m.events[0]].t / d) * w;
        const lw = m.labelNode.offsetWidth || 40;
        const left = clamp(x - lw / 2, 0, Math.max(0, w - lw));
        let lane = ends.findIndex((r) => r + 6 <= left);
        if (lane < 0) {
          if (ends.length < 3) { lane = ends.length; ends.push(0); }
          else lane = ends.indexOf(Math.min(...ends));
        }
        ends[lane] = left + lw;
        return { m, x, lw, left, lane };
      });
      lanes = Math.max(1, ends.length);
      for (const { m, x, lw, left, lane } of placed) {
        const t = st.spec.events[m.events[0]].t;
        m.node.style.left = `${x}px`;
        m.node.style.top = `${lane * LANE}px`;
        m.node.style.setProperty("--shift", `${left - (x - lw / 2)}px`);
        m.node.classList.toggle("changed", markChanged(m));
        m.node.setAttribute("aria-valuenow", t.toFixed(2));
        m.node.setAttribute("aria-valuetext", `${m.label} at ${t.toFixed(2)} seconds`);
      }
      head.style.top = `${topPx()}px`;
    }

    function shiftMark(m, starts, delta) {
      const d = st.spec.duration;
      let lo = -Infinity, hi = Infinity;
      m.events.forEach((i, k) => {
        const e = st.spec.events[i];
        lo = Math.max(lo, -starts[k]);
        hi = Math.min(hi, Math.max(starts[k], d - e.dur) - starts[k]);
      });
      const dd = clamp(delta, lo, hi);
      m.events.forEach((i, k) => { st.spec.events[i].t = round2(starts[k] + dd); });
    }

    const commitSoon = debounce(() => commit(), 450);
    let nudged = null;
    function stopNudge() {
      if (nudged) nudged.classList.remove("nudge");
      nudged = null;
    }

    function bindMark(m, k) {
      const b = m.node;
      let drag = null;
      const activate = () => { activeMark = k; b.classList.add("active"); refresh(); };
      const deactivate = () => {
        if (drag) return;
        if (activeMark === k) activeMark = -1;
        b.classList.remove("active");
        refresh();
      };
      m.start = (e) => {
        b.focus({ preventScroll: true });
        engine.init();
        b.setPointerCapture(e.pointerId);
        drag = { x: e.clientX, starts: m.events.map((i) => st.spec.events[i].t), moved: false };
        b.classList.add("dragging");
        activate();
      };
      b.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        // A label is unambiguous (labels never overlap). On the line, markers
        // a few px apart overlap their hit areas, so take the nearest line.
        let pick = m;
        if (!e.target.closest(".pg-mark-label")) {
          let best = Infinity;
          for (const o of marks) {
            const r = o.node.getBoundingClientRect();
            const dist = Math.abs(e.clientX - (r.left + r.width / 2));
            if (dist < best) { best = dist; pick = o; }
          }
        }
        stopNudge();
        pick.start(e);
      });
      b.addEventListener("pointermove", (e) => {
        if (!drag) return;
        const dx = e.clientX - drag.x;
        if (!drag.moved && Math.abs(dx) < 2) return;
        drag.moved = true;
        shiftMark(m, drag.starts, (dx / viz.clientWidth) * st.spec.duration);
        refresh();
      });
      const end = () => {
        if (!drag) return;
        const moved = drag.moved;
        drag = null;
        b.classList.remove("dragging");
        if (moved) commit();
      };
      b.addEventListener("pointerup", end);
      b.addEventListener("pointercancel", end);
      b.addEventListener("focus", activate);
      b.addEventListener("blur", deactivate);
      b.addEventListener("mouseenter", () => { if (activeMark !== k) { activeMark = k; refresh(); } });
      b.addEventListener("mouseleave", () => { if (document.activeElement !== b) deactivate(); });
      b.addEventListener("keydown", (e) => {
        const big = e.shiftKey ? 0.25 : 0.05;
        const t0 = st.spec.events[m.events[0]].t;
        const delta = {
          ArrowRight: big, ArrowUp: big, ArrowLeft: -big, ArrowDown: -big,
          PageUp: 1, PageDown: -1, Home: -t0, End: st.spec.duration,
        }[e.key];
        if (delta == null) return;
        e.preventDefault();
        engine.init();
        shiftMark(m, m.events.map((i) => st.spec.events[i].t), delta);
        refresh();
        commitSoon();
      });
    }

    /* -- controls ---------------------------------------------------- */
    function field(label, cls) {
      const f = el("div", `pg-field ${cls || ""}`);
      if (label) f.appendChild(el("span", "pg-field-label", label));
      return f;
    }
    function range(f, { min, max, step, aria, onInput, fmt }) {
      const row = el("label", "pg-range");
      const input = document.createElement("input");
      input.type = "range";
      input.min = String(min); input.max = String(max); input.step = String(step);
      input.setAttribute("aria-label", aria);
      const out = el("output", "pg-out");
      row.append(input, out);
      f.appendChild(row);
      const show = (v) => {
        input.value = String(v);
        input.style.setProperty("--p", `${((v - min) / (max - min)) * 100}%`);
        out.textContent = fmt(v);
        input.setAttribute("aria-valuetext", fmt(v));
      };
      input.addEventListener("input", () => { engine.init(); const v = parseFloat(input.value); onInput(v); show(v); refresh(); });
      input.addEventListener("change", () => commit());
      return { input, show };
    }

    function buildControls() {
      controlsEl.innerHTML = "";
      syncers = [];
      const spec = () => st.spec;
      const base = st.preset.spec;
      const sameShape = st.spec.events.length === base.events.length && st.spec.events.every((e, i) => e.role === base.events[i].role);
      for (const c of st.preset.controls || []) {
        const evs = c.events || [];
        if (evs.length && !sameShape) continue; // the Spec was rewritten; its indices no longer apply
        if (c.kind === "pan") {
          const f = field(`${c.label} · pan`);
          const r = range(f, {
            min: -1, max: 1, step: 0.25, aria: `${c.label} pan`,
            fmt: (v) => panLabel(v),
            onInput: (v) => evs.forEach((i) => { spec().events[i].pan = v; }),
          });
          r.input.setAttribute("aria-valuetext", "");
          const show = r.show;
          r.show = (v) => { show(v); r.input.setAttribute("aria-valuetext", panWords(v)); };
          syncers.push(() => r.show(spec().events[evs[0]].pan));
          controlsEl.appendChild(f);
        } else if (c.kind === "gain") {
          const f = field(c.label);
          const r = range(f, {
            min: c.min ?? -30, max: c.max ?? 0, step: 1, aria: `${c.label} gain`,
            fmt: (v) => `${v > 0 ? "+" : ""}${Math.round(v)} dB`,
            onInput: (v) => {
              const d = v - spec().events[evs[0]].gain_db;
              evs.forEach((i) => { spec().events[i].gain_db = round2(clamp(spec().events[i].gain_db + d, -60, 0)); });
            },
          });
          syncers.push(() => r.show(spec().events[evs[0]].gain_db));
          controlsEl.appendChild(f);
        } else if (c.kind === "bed_db") {
          const f = field(`${c.label} · level`);
          const fallback = base.bed_db ?? -40;
          const r = range(f, {
            min: c.min ?? -60, max: c.max ?? -6, step: 1, aria: `${c.label} level`,
            fmt: (v) => `${Math.round(v)} dB`,
            onInput: (v) => { if (spec().texture !== "none") spec().bed_db = v; },
          });
          syncers.push(() => {
            const none = spec().texture === "none";
            r.input.disabled = none;
            f.classList.toggle("off", none);
            r.show(spec().bed_db ?? fallback);
          });
          controlsEl.appendChild(f);
        } else if (c.kind === "seed") {
          const f = field("", "pg-field-seed");
          const b = el("button", "pg-seed");
          b.type = "button";
          b.innerHTML = `${DICE}<span>${esc(c.label)}</span>`;
          b.addEventListener("click", () => {
            engine.init();
            newTake(evs);
            refresh();
            commit();
          });
          f.append(b);
          controlsEl.appendChild(f);
        } else if (c.kind === "texture") {
          const f = field(c.label);
          const group = el("div", "pg-seg");
          group.setAttribute("role", "radiogroup");
          group.setAttribute("aria-label", c.label);
          const opts = c.options.map((o) => {
            const b = el("button", "pg-seg-opt", words(o));
            b.type = "button";
            b.setAttribute("role", "radio");
            b.addEventListener("click", () => setTexture(o));
            b.addEventListener("keydown", (e) => {
              const n = c.options.length;
              const k = c.options.indexOf(o);
              const to = { ArrowRight: k + 1, ArrowDown: k + 1, ArrowLeft: k - 1, ArrowUp: k - 1 }[e.key];
              if (to == null) return;
              e.preventDefault();
              const j = (to + n) % n;
              setTexture(c.options[j]);
              opts[j].focus();
            });
            group.appendChild(b);
            return b;
          });
          f.appendChild(group);
          syncers.push(() => opts.forEach((b, k) => {
            const on = spec().texture === c.options[k];
            b.setAttribute("aria-checked", String(on));
            b.tabIndex = on ? 0 : -1;
          }));
          controlsEl.appendChild(f);
        }
      }
      controlsEl.hidden = !controlsEl.children.length;
      sync();
    }

    /** Fresh seeds, plus a bounded jitter of length and level: impact and
     * sub_drop are modal/sine bodies the seed alone barely changes. The
     * jitter is stored per event and replaced (not stacked) on each take, so
     * takes never drift and the user's own edits stay. Rising roles keep
     * their apex, so the hit stays on time. */
    function newTake(evs) {
      const s = st.spec;
      const jit = st.jit || (st.jit = {});
      const rnd = (a, b) => a + Math.random() * (b - a);
      for (const i of evs) {
        const e = s.events[i];
        const prev = jit[i] || { f: 1, g: 0 };
        let seed;
        // Thunder keeps its distance (seed % 4); everything else re-rolls freely.
        const keep = e.role === "thunder" ? 3 : 0;
        do { seed = (Math.floor(Math.random() * 256) & ~keep) | (e.seed & keep); } while (seed === e.seed);
        e.seed = seed;
        const baseDur = e.dur / prev.f;
        if (e.role in APEX) {
          const A = APEX[e.role];
          const apex = e.t + A * e.dur;
          const dur = round2(clamp(baseDur * rnd(0.8, 1.2), 0.1, apex / A));
          e.t = round2(Math.max(0, apex - A * dur));
          e.dur = dur;
          jit[i] = { f: dur / baseDur, g: 0 };
        } else if (["impact", "sub_drop", "bell", "whoosh", "noise_sweep"].includes(e.role)) {
          const dur = round2(clamp(baseDur * rnd(0.75, 1.3), 0.05, Math.max(0.05, s.duration - e.t)));
          const g = rnd(-2, 2);
          const gain = round2(clamp(e.gain_db - prev.g + g, -60, 0));
          jit[i] = { f: dur / baseDur, g: gain - (e.gain_db - prev.g) };
          e.dur = dur;
          e.gain_db = gain;
        }
      }
    }

    function setTexture(tex) {
      engine.init();
      const s = st.spec;
      if (s.texture === tex) return;
      if (tex === "none") {
        delete s.bed_db;
      } else if (s.bed_db == null && st.preset.spec.bed_db != null) {
        s.bed_db = st.preset.spec.bed_db;
      }
      s.texture = tex;
      refresh();
      commit();
    }

    function sync() { syncers.forEach((f) => f()); }

    /* -- drawing ----------------------------------------------------- */
    function paint(progress = -1) {
      drawScore(canvas, st.current.peaks, st.spec, {
        top: topPx(),
        progress,
        dim: isStale() || st.busy,
        spans: activeMark >= 0 && marks[activeMark] ? marks[activeMark].events : [],
      });
    }

    function refresh() {
      layoutMarks();
      sync();
      const active = activeMark >= 0 && marks[activeMark] ? marks[activeMark].events : null;
      specEl.innerHTML = specHTML(st.spec, st.preset.spec, active, specEl.clientWidth < 600);
      resetBtn.disabled = keyOf(st) === JSON.stringify(st.preset.spec);
      viz.classList.toggle("stale", isStale());
      if (player.token !== "try") paint();
    }

    /* -- audio ------------------------------------------------------- */
    async function playCurrent() {
      const s = st;
      try {
        const buf = s.current.buffer || (s.current.buffer = await loadBuffer(s.current.url));
        if (s !== st) return;
        head.hidden = false;
        play(buf, "try", (p) => {
          head.style.left = `${p * viz.clientWidth}px`;
          paint(p);
        }, () => {
          head.hidden = true;
          setPlayIcon(btn, false);
          paint();
        });
        setPlayIcon(btn, true);
      } catch (err) {
        console.warn(err);
      }
    }

    async function commit(s = st) {
      const key = keyOf(s);
      if (key === s.renderedKey || key === s.failedKey) { if (s === st) refresh(); return; }
      if (s.busy) { s.dirty = true; return; }
      s.busy = true;
      if (s === st) refresh();
      try {
        const res = await engine.render(JSON.parse(key));
        s.current = { peaks: res.peaks, url: null, buffer: pcmToBuffer(res.pcm, res.channels, res.sampleRate) };
        s.renderedKey = key;
        s.failedKey = null;
        if (s === st) {
          if (!jsonBox.hidden) { jsonMsg.textContent = "Rendered. Drag a marker, or edit again."; jsonMsg.classList.remove("err"); }
          if (keyOf(s) === key) playCurrent();
        }
      } catch (err) {
        console.warn(err);
        s.failedKey = key;
        if (s === st) {
          const msg = `The engine refused this Spec: ${err.message}`;
          if (!jsonBox.hidden) { jsonMsg.textContent = msg; jsonMsg.classList.add("err"); }
          else engine.status("ready", "render failed, try another value");
        }
      } finally {
        s.busy = false;
        if (s === st) refresh();
        if (s.dirty) { s.dirty = false; commit(s); }
      }
    }

    dlBtn.addEventListener("click", async () => {
      const s = st;
      try {
        const buf = s.current.buffer || (s.current.buffer = await loadBuffer(s.current.url));
        const edited = keyOf(s) !== JSON.stringify(s.preset.spec);
        saveBlob(bufferWav(buf), `orb-${slug(s.preset.title || "score")}${edited ? "-edit" : ""}.wav`);
      } catch (err) {
        console.warn(err);
      }
    });

    /* -- preset selection, reset, JSON ------------------------------- */
    function select(i) {
      if (player.token === "try") stop();
      st = states[i];
      activeMark = -1;
      chips.forEach((c, k) => {
        c.setAttribute("aria-selected", String(k === i));
        c.tabIndex = k === i ? 0 : -1;
      });
      card.setAttribute("aria-labelledby", chips[i].id);
      blurb.textContent = st.preset.blurb || "";
      viz.setAttribute("aria-label", `${st.preset.title}, ${st.spec.duration} seconds`);
      rebuild();
      if (!jsonBox.hidden) fillJson();
    }

    function rebuild() {
      buildMarks();
      buildControls();
      refresh();
    }

    function fillJson() {
      jsonText.value = prettySpec(st.spec);
      jsonMsg.textContent = "Valid JSON in, a new render out.";
      jsonMsg.classList.remove("err");
    }

    resetBtn.addEventListener("click", () => {
      if (player.token === "try") stop();
      st.spec = clone(st.preset.spec);
      st.jit = null;
      st.current = { peaks: st.preset.peaks, url: st.preset.audio, buffer: null };
      st.renderedKey = JSON.stringify(st.preset.spec);
      st.failedKey = null;
      st.dirty = false;
      activeMark = -1;
      rebuild();
      if (!jsonBox.hidden) fillJson();
    });

    editToggle.addEventListener("click", () => {
      const open = jsonBox.hidden;
      jsonBox.hidden = !open;
      specEl.hidden = open;
      editToggle.setAttribute("aria-pressed", String(open));
      editToggle.textContent = open ? "Done" : "Edit";
      if (open) {
        fillJson();
        engine.init();
        jsonText.focus({ preventScroll: true });
      }
    });

    const applyJson = () => {
      let obj;
      try {
        obj = JSON.parse(jsonText.value);
      } catch (err) {
        jsonMsg.textContent = `Not valid JSON: ${err.message}`;
        jsonMsg.classList.add("err");
        return;
      }
      const problem = validateSpec(obj);
      if (problem) {
        jsonMsg.textContent = problem;
        jsonMsg.classList.add("err");
        return;
      }
      for (const e of obj.events) {
        e.gain_db = e.gain_db ?? -12;
        e.pan = e.pan ?? 0;
        e.seed = e.seed ?? 0;
      }
      jsonMsg.textContent = "Rendering…";
      jsonMsg.classList.remove("err");
      if (player.token === "try") stop();
      st.spec = obj;
      st.jit = null;
      st.failedKey = null;
      activeMark = -1;
      rebuild();
      commit();
    };
    jsonGo.addEventListener("click", applyJson);
    jsonText.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); applyJson(); }
    });

    btn.addEventListener("click", () => (player.token === "try" ? stop() : playCurrent()));
    window.addEventListener("resize", debounce(() => refresh(), 100));
    select(0);
    // "Shape it by hand" under a generated sound: load its Spec into the
    // current preset slot as an edit (Reset brings the preset back).
    loadIntoPlayground = (spec, current) => {
      if (player.token === "try") stop();
      st.spec = clone(spec);
      st.jit = null;
      st.current = { peaks: current.peaks, url: null, buffer: current.buffer };
      st.renderedKey = keyOf(st);
      st.failedKey = null;
      st.dirty = false;
      activeMark = -1;
      rebuild();
      blurb.textContent = "Your generated sound. Drag a marker or a slider; Reset brings the preset back.";
      if (!jsonBox.hidden) fillJson();
      $("#shape").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    };
    // One quiet hint that the markers move: nudge one label sideways, once,
    // when the playground first comes into view. Any touch of a marker ends it.
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries) => {
        if (!entries.some((en) => en.isIntersecting)) return;
        io.disconnect();
        setTimeout(() => {
          const m = marks[Math.min(2, marks.length - 1)];
          if (!m || activeMark >= 0 || keyOf(st) !== JSON.stringify(st.preset.spec)) return;
          nudged = m.labelNode;
          nudged.classList.add("nudge");
          nudged.addEventListener("animationend", stopNudge, { once: true });
        }, 900);
      }, { threshold: 0.6 });
      io.observe(viz);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Misc                                                                */
  /* ------------------------------------------------------------------ */

  let revealObs = null;
  function observeReveal(root = document) {
    if (reduceMotion || !("IntersectionObserver" in window)) {
      $$(".reveal", root).forEach((n) => n.classList.add("in"));
      return;
    }
    revealObs = revealObs || new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          revealObs.unobserve(e.target);
        }
      }
    }, { rootMargin: "0px 0px -6% 0px" });
    $$(".reveal:not(.in)", root).forEach((n) => revealObs.observe(n));
  }

  function initNav() {
    const nav = $(".nav");
    const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 8);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  function initEngineAutostart() {
    // Start the Python runtime as the visitor nears the live block, unless
    // they asked the browser to save data.
    const saveData = navigator.connection && navigator.connection.saveData;
    if (saveData || !("IntersectionObserver" in window)) return;
    const obs = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        obs.disconnect();
        engine.init();
      }
    }, { rootMargin: "300px 0px" });
    obs.observe($("#gen"));
  }

  /* ------------------------------------------------------------------ */
  /* Boot                                                                */
  /* ------------------------------------------------------------------ */

  async function boot() {
    engine.probe();
    initNav();
    observeReveal();
    try {
      const r = await fetch("assets/cues.json");
      if (!r.ok) throw new Error(`cues.json: HTTP ${r.status}`);
      DATA = await r.json();
    } catch (err) {
      $("#gallery").innerHTML = `<p class="loading">Could not load the cue gallery (${esc(err.message)}).</p>`;
      console.error(err);
      return;
    }
    initVideo();
    initGenerator();
    initPlayground();
    initGallery();
    initEngineAutostart();
    darkQuery.addEventListener && darkQuery.addEventListener("change", () => {
      THEME = null;
      window.dispatchEvent(new Event("resize"));
    });
    document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
