(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const rootStyle = getComputedStyle(document.documentElement);
  const COLORS = {
    border: rootStyle.getPropertyValue("--border").trim() || "#232327",
    faint: rootStyle.getPropertyValue("--faint").trim() || "#33333a",
    accent: rootStyle.getPropertyValue("--accent").trim() || "#d9ff3d",
  };

  const PLAY_SVG =
    '<svg viewBox="0 0 12 12" aria-hidden="true" focusable="false"><path d="M3.2 1.8 10.2 6 3.2 10.2Z" fill="currentColor"/></svg>';
  const PAUSE_SVG =
    '<svg viewBox="0 0 12 12" aria-hidden="true" focusable="false"><rect x="3" y="2" width="2" height="8" fill="currentColor"/><rect x="7" y="2" width="2" height="8" fill="currentColor"/></svg>';

  const fmt = {
    dur: (seconds) => `${Number(seconds || 0).toFixed(1)}s`,
    peak: (value) => Number(value || 0).toFixed(3),
    dc: (value) => {
      const n = Number(value);
      if (!Number.isFinite(n)) return "—";
      if (n === 0) return "0";
      if (Math.abs(n) < 1e-3) return n.toExponential(1);
      return n.toFixed(4);
    },
  };

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const chip = (text, extra) => el("span", `chip${extra ? ` ${extra}` : ""}`, text);

  let uid = 0;

  // -- waveform painting --------------------------------------------------

  function paintWave(peaks, width, height, dpr, color, centerLine) {
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const mid = height / 2;
    if (centerLine) {
      ctx.strokeStyle = COLORS.border;
      ctx.globalAlpha = 0.7;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, mid + 0.5);
      ctx.lineTo(width, mid + 0.5);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    if (!peaks || !peaks.length) return canvas;

    const shape = (v) => Math.sign(v) * Math.pow(Math.abs(v), 0.6);
    const amp = (height / 2) * 0.88;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.lineCap = "butt";
    ctx.beginPath();
    for (let i = 0; i < peaks.length; i += 1) {
      const x = ((i + 0.5) / peaks.length) * width;
      // Peaks are normalised to the global max, so a cue with one loud
      // transient leaves the rest visually flat. A mild power curve lifts
      // the quiet body without turning noise into a solid block.
      const lo = shape(Number(peaks[i][0]) || 0);
      const hi = shape(Number(peaks[i][1]) || 0);
      let y0 = mid - hi * amp;
      let y1 = mid - lo * amp;
      if (y1 - y0 < 1) {
        const centre = (y0 + y1) / 2;
        y0 = centre - 0.5;
        y1 = centre + 0.5;
      }
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y1);
    }
    ctx.stroke();
    return canvas;
  }

  function setupCardCanvas(card) {
    const { canvas } = card;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return false;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.max(1, Math.round(width * dpr));
    const pixelHeight = Math.max(1, Math.round(height * dpr));
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }

    card.w = width;
    card.h = height;
    card.dpr = dpr;
    card.baseMuted = paintWave(card.peaks, width, height, dpr, COLORS.border, true);
    card.baseAccent = paintWave(card.peaks, width, height, dpr, COLORS.accent, false);
    return true;
  }

  function drawFrame(card, progress) {
    if (!card.baseMuted || !card.baseAccent) return;
    const ctx = card.ctx;
    const { w, h, dpr } = card;
    const p = Math.min(1, Math.max(0, Number(progress) || 0));

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, card.canvas.width, card.canvas.height);
    ctx.drawImage(card.baseMuted, 0, 0);

    if (p > 0) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, Math.round(p * w * dpr), h * dpr);
      ctx.clip();
      ctx.drawImage(card.baseAccent, 0, 0);
      ctx.restore();
    }

    if (p > 0 && p < 1) {
      ctx.fillStyle = COLORS.accent;
      ctx.fillRect(Math.round(p * w * dpr), 0, Math.max(1, Math.round(dpr)), h * dpr);
    }
  }

  // -- audio player -------------------------------------------------------

  const player = new Audio();
  player.preload = "auto";
  let activeCard = null;
  let rafId = 0;
  let pendingPlay = false;

  function progressOf(card) {
    const duration =
      Number.isFinite(player.duration) && player.duration > 0
        ? player.duration
        : Number(card.data.duration_s) || 0;
    if (!duration) return 0;
    return Math.min(1, Math.max(0, player.currentTime / duration));
  }

  function setPlaying(card, playing) {
    card.playing = playing;
    card.playBtn.classList.toggle("is-playing", playing);
    card.playBtn.setAttribute("aria-label", `${playing ? "Pause" : "Play"} ${card.label}`);
    card.playBtn.innerHTML = playing ? PAUSE_SVG : PLAY_SVG;
  }

  function stopRaf() {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
  }

  function tick() {
    if (!activeCard || player.paused) {
      rafId = 0;
      return;
    }
    drawFrame(activeCard, progressOf(activeCard));
    rafId = requestAnimationFrame(tick);
  }

  function startRaf() {
    stopRaf();
    rafId = requestAnimationFrame(tick);
  }

  function playCard(card) {
    if (activeCard === card && !player.paused) {
      player.pause();
      return;
    }

    if (activeCard && activeCard !== card) {
      setPlaying(activeCard, false);
      drawFrame(activeCard, 0);
    }

    if (activeCard !== card) {
      pendingPlay = true;
      player.src = card.data.url;
      activeCard = card;
      drawFrame(card, 0);
    }

    setPlaying(card, true);
    const promise = player.play();
    if (promise && typeof promise.catch === "function") {
      promise.catch(() => {
        pendingPlay = false;
        setPlaying(card, false);
        stopRaf();
        showError("Playback failed. The audio file may be missing.");
      });
    }
    startRaf();
  }

  function seekCard(card, fraction) {
    if (activeCard !== card) {
      if (activeCard) {
        setPlaying(activeCard, false);
        drawFrame(activeCard, 0);
      }
      player.src = card.data.url;
      activeCard = card;
      pendingPlay = false;
      stopRaf();
    }

    const apply = () => {
      const duration =
        Number.isFinite(player.duration) && player.duration > 0
          ? player.duration
          : Number(card.data.duration_s) || 0;
      try {
        player.currentTime = fraction * duration;
      } catch (_) {
        /* seeking before metadata is ready is harmless */
      }
      drawFrame(card, fraction);
    };

    if (player.readyState >= 1) apply();
    else player.addEventListener("loadedmetadata", apply, { once: true });
  }

  player.addEventListener("playing", () => {
    pendingPlay = false;
  });

  player.addEventListener("pause", () => {
    if (pendingPlay) return;
    if (activeCard) {
      setPlaying(activeCard, false);
      drawFrame(activeCard, progressOf(activeCard));
    }
    stopRaf();
  });

  player.addEventListener("ended", () => {
    if (activeCard) {
      setPlaying(activeCard, false);
      drawFrame(activeCard, 0);
      try {
        player.currentTime = 0;
      } catch (_) {
        /* ignore */
      }
    }
    stopRaf();
  });

  player.addEventListener("error", () => {
    if (!activeCard) return;
    setPlaying(activeCard, false);
    stopRaf();
    showError("Could not load the audio file.");
  });

  // -- cards --------------------------------------------------------------

  function createCard(data, options = {}) {
    const root = el("article", `card${options.isNew ? " is-new" : ""}`);
    const label = data.name || "generated";

    const head = el("div", "card-head");
    const title = el("div", "card-title");
    if (options.index) title.appendChild(el("span", "card-index", options.index));
    title.appendChild(el("h3", "card-name", label));

    const tags = el("div", "card-tags");
    if (options.isNew) tags.appendChild(chip("new", "new"));
    tags.appendChild(chip(data.detail ? "detail" : "dsp"));
    if (data.exact === false) {
      const drift = chip("spec drift", "drift");
      drift.title = "The planner output differs from the intended spec.";
      tags.appendChild(drift);
    }
    head.append(title, tags);

    const caption = el("p", "card-caption", data.caption || "");

    const wave = el("div", "wave");
    const canvas = document.createElement("canvas");
    canvas.className = "wave-canvas";
    canvas.setAttribute("aria-hidden", "true");
    wave.appendChild(canvas);

    const controls = el("div", "card-controls");
    const playBtn = document.createElement("button");
    playBtn.type = "button";
    playBtn.className = "play";
    playBtn.innerHTML = PLAY_SVG;
    playBtn.setAttribute("aria-label", `Play ${label}`);

    const duration = el("span", "card-duration", fmt.dur(data.duration_s));

    const audit = data.audit || {};
    const meta = el("div", "card-meta");
    const peakChip = chip(`peak ${fmt.peak(audit.peak)}`);
    peakChip.title = `peak ${Number(audit.peak || 0)}`;
    const dcChip = chip(`DC ${fmt.dc(audit.dc)}`);
    dcChip.title = `DC ${Number(audit.dc || 0)}`;
    meta.append(
      peakChip,
      dcChip,
      chip(data.passes ? "pass" : "fail", data.passes ? "pass" : "fail")
    );
    controls.append(playBtn, duration, meta);

    const pre = document.createElement("pre");
    pre.className = "spec";
    pre.id = `spec-${++uid}`;
    pre.hidden = true;
    pre.textContent = JSON.stringify(data.spec || {}, null, 2);

    const foot = el("div", "card-foot");
    const specBtn = document.createElement("button");
    specBtn.type = "button";
    specBtn.className = "spec-toggle";
    specBtn.setAttribute("aria-expanded", "false");
    specBtn.setAttribute("aria-controls", pre.id);
    const glyph = el("span", "glyph", "+");
    glyph.setAttribute("aria-hidden", "true");
    specBtn.append(el("span", "", "spec"), glyph);
    specBtn.addEventListener("click", () => {
      const open = pre.hidden;
      pre.hidden = !open;
      specBtn.setAttribute("aria-expanded", String(open));
      specBtn.querySelector(".glyph").textContent = open ? "−" : "+";
    });

    const download = document.createElement("a");
    download.className = "download";
    download.href = data.url;
    download.download = `${label.replace(/\s+/g, "_")}.wav`;
    download.textContent = "download";

    foot.append(specBtn, download);
    root.append(head, caption, wave, controls, pre, foot);

    const card = {
      root,
      data,
      label,
      canvas,
      ctx: canvas.getContext("2d"),
      playBtn,
      peaks: Array.isArray(data.peaks) ? data.peaks : [],
      playing: false,
      w: 0,
      h: 0,
      dpr: 1,
      baseMuted: null,
      baseAccent: null,
      resizePending: false,
    };

    playBtn.addEventListener("click", () => playCard(card));
    canvas.addEventListener("click", (event) => {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width) return;
      const fraction = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
      seekCard(card, fraction);
    });

    const observer = new ResizeObserver(() => {
      if (card.resizePending) return;
      card.resizePending = true;
      requestAnimationFrame(() => {
        card.resizePending = false;
        if (setupCardCanvas(card)) {
          drawFrame(card, activeCard === card ? progressOf(card) : 0);
        }
      });
    });
    observer.observe(canvas);
    card.observer = observer;

    return card;
  }

  function mountCard(card) {
    if (setupCardCanvas(card)) drawFrame(card, 0);
  }

  // -- examples -----------------------------------------------------------

  const examplesGrid = $("examples-grid");
  const examplesCount = $("examples-count");

  async function loadExamples() {
    try {
      const response = await fetch("/api/examples", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      const list = Array.isArray(payload.examples) ? payload.examples : [];
      examplesGrid.textContent = "";

      if (!list.length) {
        examplesGrid.appendChild(
          el(
            "div",
            "empty",
            "No examples built yet. Run: .venv-torch/bin/python scripts/build_web_examples.py"
          )
        );
        examplesCount.textContent = "0 cues";
        return;
      }

      list.forEach((item, index) => {
        const card = createCard(item, {
          index: String(index + 1).padStart(2, "0"),
        });
        examplesGrid.appendChild(card.root);
        mountCard(card);
      });
      examplesCount.textContent = `${list.length} cue${list.length === 1 ? "" : "s"}`;
    } catch (error) {
      examplesGrid.textContent = "";
      examplesGrid.appendChild(
        el("div", "empty", `Could not load examples (${error.message || error}).`)
      );
      examplesCount.textContent = "—";
    }
  }

  // -- hero strip ---------------------------------------------------------

  function drawHero() {
    const canvas = $("hero-wave");
    if (!canvas) return;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const mid = height / 2;
    const amp = height * 0.46;
    const bars = Math.max(60, Math.floor(width / 4));

    ctx.strokeStyle = COLORS.border;
    ctx.globalAlpha = 0.8;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid + 0.5);
    ctx.lineTo(width, mid + 0.5);
    ctx.stroke();
    ctx.globalAlpha = 1;

    let seed = 0x9e3779b9;
    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };

    const barWidth = Math.max(1, (width / bars) * 0.55);
    for (let i = 0; i < bars; i += 1) {
      const envelope = Math.pow(Math.sin((i / (bars - 1)) * Math.PI), 0.55);
      const top = (0.08 + 0.92 * Math.pow(random(), 1.7)) * envelope;
      const bottom = (0.08 + 0.92 * Math.pow(random(), 1.7)) * envelope;
      const x = ((i + 0.5) / bars) * width;
      const accent = i % 41 === 20;
      ctx.fillStyle = accent ? COLORS.accent : COLORS.faint;
      ctx.globalAlpha = accent ? 0.9 : 0.9;
      ctx.fillRect(x - barWidth / 2, mid - top * amp, barWidth, Math.max(1, (top + bottom) * amp));
    }
    ctx.globalAlpha = 1;
  }

  // -- playground ---------------------------------------------------------

  const captionInput = $("caption");
  const chipsWrap = $("caption-chips");
  const detailToggle = $("detail-toggle");
  const strengthInput = $("strength");
  const strengthValue = $("strength-value");
  const strengthControl = $("strength-control");
  const generateBtn = $("generate");
  const panel = $("playground-panel");
  const spinner = $("spinner");
  const statusText = $("status-text");
  const errorBox = $("error");
  const results = $("playground-results");

  const PRESETS = [
    {
      label: "ui pop",
      caption: "4.00-second playful cue, vinyl bed, with ui pop @0.60 0.16s -12dB R1 #73",
    },
    {
      label: "whoosh",
      caption: "6.00-second bright cue, wind bed, with whoosh @1.00 0.90s -6dB L2 #2a",
    },
    {
      label: "ambience",
      caption: "8.00-second calm cue, rain bed",
    },
    {
      label: "riser",
      caption: "8.00-second tense cue, drone bed, with riser @1.00 2.40s -16dB C #51",
    },
    {
      label: "trailer",
      caption:
        "12.00-second intense cue, drone bed, with riser @1.00 2.00s -18dB C #55, impact @3.20 0.80s -4dB C #66, whoosh @5.00 0.90s -8dB L2 #77, sub drop @7.00 1.20s -8dB C #88, impact @9.50 0.90s -4dB C #99",
    },
  ];

  let detailOn = false;
  let busy = false;
  let timer = null;

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.hidden = true;
  }

  function setDetail(on) {
    detailOn = on;
    detailToggle.setAttribute("aria-checked", String(on));
    detailToggle.classList.toggle("is-on", on);
    strengthInput.disabled = !on;
    strengthControl.classList.toggle("is-disabled", !on);
  }

  function updateStrength() {
    strengthValue.textContent = Number(strengthInput.value).toFixed(2);
  }

  async function loadHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) return;
      const data = await res.json();
      if (data.detail_available === false) {
        // The detail weights are CC BY-NC and ship separately; without them
        // the DSP path is still a complete cue, so just disable the switch.
        detailToggle.disabled = true;
        detailToggle.setAttribute("aria-disabled", "true");
        detailToggle.title = "detail model not installed";
        const hint = detailToggle.parentElement.querySelector(".hint");
        if (hint) hint.textContent = "detail model not installed · DSP only";
        setDetail(false);
      }
    } catch (_) {
      /* health is advisory; the page still works without it */
    }
  }

  function isInViewport(node) {
    const rect = node.getBoundingClientRect();
    return rect.top >= 0 && rect.bottom <= window.innerHeight;
  }

  async function generate() {
    if (busy) return;
    clearError();

    const caption = captionInput.value.trim();
    if (!caption) {
      showError("Enter a caption first.");
      captionInput.focus();
      return;
    }
    if (caption.length > 400) {
      showError(`Caption is ${caption.length} characters; the limit is 400.`);
      return;
    }

    busy = true;
    generateBtn.disabled = true;
    panel.setAttribute("aria-busy", "true");
    spinner.hidden = false;

    const started = performance.now();
    const update = () => {
      statusText.textContent = `generating… ${((performance.now() - started) / 1000).toFixed(1)}s`;
    };
    update();
    timer = window.setInterval(update, 100);

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          caption,
          detail: detailOn,
          strength: detailOn ? Number(strengthInput.value) : 1.0,
        }),
      });

      let payload = null;
      try {
        payload = await response.json();
      } catch (_) {
        payload = null;
      }
      if (!response.ok) {
        throw new Error((payload && payload.error) || `request failed (HTTP ${response.status})`);
      }
      if (!payload || !payload.url) throw new Error("server returned an empty result");

      const card = createCard({ ...payload, name: "generated" }, { isNew: true });
      results.prepend(card.root);
      mountCard(card);

      const seconds = (Number(payload.ms || 0) / 1000).toFixed(1);
      statusText.textContent = payload.cached ? `cached · ${seconds}s` : `done in ${seconds}s`;

      if (!isInViewport(card.root)) {
        const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        card.root.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "center" });
      }
    } catch (error) {
      statusText.textContent = "";
      showError(error && error.message ? error.message : String(error));
    } finally {
      if (timer) window.clearInterval(timer);
      timer = null;
      busy = false;
      generateBtn.disabled = false;
      panel.removeAttribute("aria-busy");
      spinner.hidden = true;
    }
  }

  PRESETS.forEach((preset) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip-btn";
    button.textContent = preset.label;
    button.title = preset.caption;
    button.addEventListener("click", () => {
      captionInput.value = preset.caption;
      captionInput.focus();
      captionInput.setSelectionRange(preset.caption.length, preset.caption.length);
      clearError();
    });
    chipsWrap.appendChild(button);
  });

  detailToggle.addEventListener("click", () => setDetail(!detailOn));
  strengthInput.addEventListener("input", updateStrength);
  generateBtn.addEventListener("click", generate);

  captionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      generate();
    }
  });

  // -- boot ---------------------------------------------------------------

  setDetail(false);
  updateStrength();
  loadHealth();

  const hero = $("hero-wave");
  if (hero) {
    const heroObserver = new ResizeObserver(() => drawHero());
    heroObserver.observe(hero);
    drawHero();
  }

  loadExamples();
})();
