// Event grayscale player: all images and traces are fetched from the
// backend, which computes them from the actual event integration.
const $ = (id) => document.getElementById(id);

let state = null;
let playing = false;
let playTimer = null;
let frameIdx = 0;
let picking = false;

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function loadState() {
  state = await getJSON("/api/state");
  const sel = $("scene-select");
  sel.innerHTML = "";
  for (const name of state.scenes) {
    const o = document.createElement("option");
    o.value = o.textContent = name;
    if (name === state.current_scene) o.selected = true;
    sel.appendChild(o);
  }
  $("time-slider").max = state.n_frames - 1;
  $("dataset-info").textContent =
    `${state.shape[1]}x${state.shape[0]} 传感器 · ` +
    `${state.n_events} 事件 · ${state.n_frames} 输出帧 · ` +
    `阈值 C=${state.contrast.toFixed(2)} · ${state.anchor_times.length} 个锚点`;
  renderAnomalies();
  renderAnchors();
  frameIdx = 0;
  await updateView();
  loadPixelTrace();
}

function currentTime() {
  return state.output_times[frameIdx];
}

async function updateView() {
  const t = currentTime();
  $("time-slider").value = frameIdx;
  $("t-label").textContent = t.toFixed(3);

  const win = parseFloat($("window").value) || 0.05;
  const t0 = Math.max(0, t - win);
  const showRef = $("show-ref").checked ? 1 : 0;

  $("event-img").src = `/api/events?t0=${t0}&t1=${t}&r=${Math.random()}`;
  $("grey-img").src = `/api/frame?t=${t}&ref=${showRef}&r=${Math.random()}`;

  const stats = await getJSON(`/api/event_stats?t0=${t0}&t1=${t}`);
  $("event-stats").innerHTML =
    `<span>窗内事件 <b>${stats.n_events}</b></span>` +
    `<span>ON <b style="color:#ef6b6b">${stats.n_on}</b></span>` +
    `<span>OFF <b style="color:#7aa2ff">${stats.n_off}</b></span>` +
    `<span>净计数 <b>${stats.n_on - stats.n_off}</b></span>` +
    `<span>窗 [${stats.t0.toFixed(3)}, ${stats.t1.toFixed(3)}) s</span>`;
}

function renderAnomalies() {
  const ul = $("anomaly-list");
  ul.innerHTML = "";
  if (!state.anomalies.length) {
    ul.innerHTML = '<li class="info">未检测到异常</li>';
    return;
  }
  for (const a of state.anomalies) {
    const li = document.createElement("li");
    li.className = a.level;
    li.textContent = a.message;
    ul.appendChild(li);
  }
}

function renderAnchors() {
  const ul = $("anchor-list");
  ul.innerHTML = "";
  for (const a of state.anchor_summary) {
    const li = document.createElement("li");
    li.className = a.failed ? "failed" : "";
    li.textContent = a.failed
      ? `t=${a.t.toFixed(3)}s 失效：${a.reason}`
      : `t=${a.t.toFixed(3)}s 漂移 ${a.global_drift >= 0 ? "+" : ""}${a.global_drift.toFixed(3)} · ` +
        `软校正 ${a.blended_pixels}px · 重置 ${a.reset_pixels}px · 支持 ${a.supported_pixels}px`;
    ul.appendChild(li);
  }
}

async function loadPixelTrace() {
  const x = parseInt($("px").value, 10);
  const y = parseInt($("py").value, 10);
  const tr = await getJSON(`/api/pixel?x=${x}&y=${y}`);
  drawTrace(tr);
  $("pixel-stats").innerHTML =
    `<span>总事件 <b>${tr.count_total}</b></span>` +
    `<span>ON <b style="color:#ef6b6b">${tr.count_on}</b></span>` +
    `<span>OFF <b style="color:#7aa2ff">${tr.count_off}</b></span>` +
    (tr.count_on === tr.count_off && tr.count_total > 0
      ? '<span style="color:#f59e0b">净计数为 0：曲线仍保留中间峰值（按极性时序积分）</span>' : "");
}

function drawTrace(tr) {
  const c = $("trace-canvas");
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height, pad = 46;
  ctx.clearRect(0, 0, W, H);
  const times = tr.times;
  const tMin = times[0], tMax = times[times.length - 1] || 1;
  const allV = tr.raw.concat(tr.corrected).filter((v) => !isNaN(v));
  let vMin = Math.min(...allV), vMax = Math.max(...allV);
  const padV = (vMax - vMin) * 0.1 || 0.5;
  vMin -= padV; vMax += padV;
  const X = (t) => pad + (t - tMin) / (tMax - tMin || 1) * (W - 2 * pad);
  const Y = (v) => H - pad - (v - vMin) / (vMax - vMin) * (H - 2 * pad);

  // grid + axes
  ctx.strokeStyle = "#2c323d"; ctx.lineWidth = 1;
  ctx.fillStyle = "#8a93a3"; ctx.font = "11px sans-serif";
  for (let i = 0; i <= 4; i++) {
    const v = vMin + (vMax - vMin) * i / 4;
    const yy = Y(v);
    ctx.beginPath(); ctx.moveTo(pad, yy); ctx.lineTo(W - pad, yy); ctx.stroke();
    ctx.fillText(v.toFixed(2), 6, yy + 4);
  }
  ctx.strokeStyle = "#5b6472";
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad);
  ctx.lineTo(W - pad, H - pad); ctx.stroke();
  ctx.fillText("log L", 8, pad - 8);
  ctx.fillText("t (s)", W - 46, H - 16);

  // anchor vertical markers
  for (const ap of tr.anchor_points) {
    ctx.strokeStyle = ap.kind === "reset" ? "rgba(239,68,68,.55)" : "rgba(245,158,11,.55)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(X(ap.t), pad); ctx.lineTo(X(ap.t), H - pad); ctx.stroke();
    ctx.setLineDash([]);
  }

  const drawLine = (data, color, width) => {
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.beginPath();
    let started = false;
    data.forEach((v, i) => {
      if (isNaN(v)) return;
      const xx = X(times[i]), yy = Y(v);
      started ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy);
      started = true;
    });
    ctx.stroke();
  };
  drawLine(tr.raw, "#7aa2ff", 1.6);        // raw event-only integration
  drawLine(tr.corrected, "#34d399", 2.2);  // anchor-corrected output

  // events as ticks along the bottom
  for (const e of tr.events) {
    ctx.strokeStyle = e.p === 1 ? "#ef6b6b" : "#60a5fa";
    const xx = X(e.t);
    ctx.beginPath();
    ctx.moveTo(xx, H - pad);
    ctx.lineTo(xx, H - pad + (e.p === 1 ? 7 : -7));
    ctx.stroke();
  }
  // anchor correction points
  for (const ap of tr.anchor_points) {
    ctx.fillStyle = ap.kind === "reset" ? "#ef4444" : "#f59e0b";
    ctx.beginPath(); ctx.arc(X(ap.t), Y(ap.logL), 4.5, 0, Math.PI * 2); ctx.fill();
  }

  // legend
  const legend = [
    ["#7aa2ff", "纯事件积分（无绝对亮度）"],
    ["#34d399", "锚点校正后重建值"],
    ["#f59e0b", "软校正锚点"], ["#ef4444", "重置锚点"],
  ];
  legend.forEach(([col, txt], i) => {
    ctx.fillStyle = col; ctx.fillRect(pad + 8 + i * 190, 14, 12, 4);
    ctx.fillStyle = "#c4cad6"; ctx.fillText(txt, pad + 24 + i * 190, 19);
  });
}

// ---------------- controls ----------------
$("time-slider").addEventListener("input", async () => {
  frameIdx = parseInt($("time-slider").value, 10);
  await updateView();
});
$("window").addEventListener("change", updateView);
$("show-ref").addEventListener("change", updateView);
$("play").addEventListener("click", () => {
  playing = !playing;
  $("play").textContent = playing ? "⏸ 暂停" : "▶ 播放";
  if (playing) {
    playTimer = setInterval(async () => {
      frameIdx = (frameIdx + 1) % state.n_frames;
      await updateView();
    }, 1000 / state.fps);
  } else {
    clearInterval(playTimer);
  }
});
$("reload").addEventListener("click", async () => {
  const params = new URLSearchParams({
    scene: $("scene-select").value,
    fps: $("fps").value,
    coverage: $("coverage").value,
    corrupt: $("corrupt").value,
  });
  await getJSON(`/api/reload?${params}`);
  await loadState();
});
$("pick-pixel").addEventListener("click", () => {
  picking = !picking;
  $("pick-pixel").textContent = picking ? "点击灰度图选像素…" : "在灰度图上点选";
});
$("grey-img").addEventListener("click", (ev) => {
  if (!picking || !state) return;
  const rect = ev.target.getBoundingClientRect();
  const x = Math.floor((ev.clientX - rect.left) / rect.width * state.shape[1]);
  const y = Math.floor((ev.clientY - rect.top) / rect.height * state.shape[0]);
  $("px").value = x; $("py").value = y;
  picking = false;
  $("pick-pixel").textContent = "在灰度图上点选";
  loadPixelTrace();
});
$("px").addEventListener("change", loadPixelTrace);
$("py").addEventListener("change", loadPixelTrace);

loadState().catch((e) => {
  document.body.innerHTML = `<pre style="padding:20px">加载失败: ${e.message}</pre>`;
});
