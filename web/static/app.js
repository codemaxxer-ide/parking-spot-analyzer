"use strict";

const $ = (id) => document.getElementById(id);
const player = $("video-player");
let videoFile = null, slotsFile = null, config = null, previewUrl = null;
let busy = false, processed = false, filter = "all", showRois = true, selected = null;
let probabilities = {};
let states = {}, activeJob = null, configVersion = 0, insightsLoaded = false;
const pct = (value, places = 2) => (100 * Number(value)).toFixed(places);
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (c) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));

function alertUser(message) { $("alert-text").textContent = message; $("alert").hidden = false; }
$("dismiss-alert").onclick = () => { $("alert").hidden = true; };
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Request failed. Check your files and try again.");
  return data;
}
function ready() { $("analyse").disabled = busy || !videoFile || !slotsFile; }
function setBusy(value) {
  busy = value; $("input-controls").disabled = value;
  ["top-upload", "empty-upload"].forEach((id) => { $(id).disabled = value; });
  $("processing").hidden = !value; ready();
}
function showView(name) {
  document.querySelectorAll(".view").forEach((view) => { view.hidden = view.id !== `view-${name}`; });
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === name);
    if (button.dataset.view === name) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  });
  if (name === "insights" && !insightsLoaded) loadInsights();
  if (name !== "dashboard") player.pause();
}
document.querySelectorAll("[data-view]").forEach((button) => { button.onclick = () => showView(button.dataset.view); });
$("toggle-sidebar").onclick = () => {
  const collapsed = document.body.classList.toggle("sidebar-collapsed");
  $("toggle-sidebar").setAttribute("aria-expanded", String(!collapsed));
};
for (const id of ["top-upload", "empty-upload"]) $(id).onclick = () => { showView("dashboard"); $("video-input").click(); };
$("calibrate").onclick = () => $("calibration-dialog").showModal();
document.querySelectorAll(".dialog-close,.dialog-close-button").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });

function resetResult() {
  processed = false; states = {}; probabilities = {}; activeJob = null;
  sessionStorage.removeItem("cloudforge-job");
  $("downloads").hidden = true; $("summary-scope").textContent = "Awaiting analysis";
  for (const id of ["occupied", "available", "occupancy"]) $(id).textContent = "—";
  $("total").textContent = config?.slots.length ?? "—"; $("occupancy-fill").style.width = "0%";
  $("stage-label").textContent = "INPUT PREVIEW";
  if (previewUrl) { player.src = previewUrl; $("video-shell").hidden = false; }
  updateHud(); renderSlots();
}
function chooseVideo(file) {
  if (busy || !file) return;
  if (!/\.mp4$/i.test(file.name) || !file.size || file.size > 500 * 1024 * 1024) {
    videoFile = null; ready(); alertUser("Choose a non-empty MP4 smaller than 500 MiB."); return;
  }
  videoFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  resetResult(); $("empty-state").hidden = true;
  $("file-info").hidden = false; $("file-name").textContent = file.name;
  $("search-title").textContent = file.name;
  $("file-meta").textContent = `${(file.size / 1024 / 1024).toFixed(1)} MiB · reading metadata…`;
  $("alert").hidden = true; ready();
}
$("video-input").onchange = (event) => chooseVideo(event.target.files[0]);
for (const event of ["dragenter", "dragover"]) $("drop-zone").addEventListener(event, (e) => { e.preventDefault(); if (!busy) $("drop-zone").classList.add("dragging"); });
for (const event of ["dragleave", "drop"]) $("drop-zone").addEventListener(event, (e) => { e.preventDefault(); $("drop-zone").classList.remove("dragging"); });
$("drop-zone").addEventListener("drop", (event) => chooseVideo(event.dataTransfer.files[0]));
player.addEventListener("loadedmetadata", () => {
  $("stage-resolution").textContent = `${player.videoWidth} × ${player.videoHeight}`;
  if (!processed && videoFile) $("file-meta").textContent = `${(videoFile.size / 1024 / 1024).toFixed(1)} MiB · ${player.videoWidth} × ${player.videoHeight} · ${Number.isFinite(player.duration) ? player.duration.toFixed(1) + " sec" : "duration unavailable"}`;
  drawRois();
});
player.addEventListener("error", () => {
  if (player.getAttribute("src")) alertUser(processed ? "This browser could not play the result. Download the processed MP4 below." : "Browser preview unavailable for this MP4 codec. You can still submit it; the server will check whether OpenCV can decode it.");
});
async function chooseConfig(file) {
  if (busy || !file) return;
  const version = ++configVersion;
  slotsFile = null; config = null; resetResult(); ready();
  $("config-name").textContent = file.name; $("config-status").textContent = "Validating parking polygons…";
  $("config-status").classList.remove("ready");
  try {
    const form = new FormData(); form.append("slots", file);
    const validated = await api("/api/validate-config", {method: "POST", body: form});
    if (version !== configVersion) return;
    config = validated; slotsFile = file;
    $("config-status").textContent = `${config.slots.length} spaces · ${config.width ? `${config.width} × ${config.height}` : "legacy config; no resolution metadata"}`;
    $("config-status").classList.add("ready"); resetResult(); ready();
    if (!idsTouched) $("show-ids").checked = config.slots.length <= 50;
  } catch (error) { if (version === configVersion) { $("config-status").textContent = "Configuration rejected"; alertUser(error.message); } }
}
$("slots-input").onchange = (event) => chooseConfig(event.target.files[0]);

let idsTouched = false;
function applyEngine() {
  const classifier = $("engine").value === "classifier";
  $("yolo-controls").hidden = classifier; $("classifier-controls").hidden = !classifier;
  $("boxes-row").hidden = classifier;
  $("every-label").textContent = classifier ? "Classify every N frames" : "Detect every N frames";
  $("engine-micro").textContent = classifier ? "MobileNetV2" : "YOLO11n";
}
function updateHud(settings) {
  const classifier = (settings?.engine ?? $("engine").value) === "classifier";
  $("hud-engine").textContent = classifier ? "Slot Classifier — MobileNetV2" : "YOLO11n + ROI";
  $("hud-confidence").textContent = classifier ? `Threshold ${Number(settings?.classification_threshold ?? $("threshold").value).toFixed(2)}` : `Confidence ${Number(settings?.confidence ?? $("confidence").value).toFixed(2)}`;
  $("hud-method").textContent = classifier ? "All slots · batch" : (settings?.occupancy_method ?? $("method").value) === "center" ? "Center point" : `Polygon overlap ${Math.round(100 * Number(settings?.overlap_threshold ?? $("overlap").value))}%`;
  $("hud-smoothing").textContent = `Smoothing ${(settings ? settings.smoothing_window > 1 : $("smoothing").checked) ? "ON" : "OFF"}`;
}
$("show-ids").onchange = () => { idsTouched = true; };
for (const id of ["confidence", "overlap", "method", "smoothing", "engine", "threshold"]) $(id).addEventListener("input", () => {
  applyEngine(); $("threshold-value").textContent = Number($("threshold").value).toFixed(2);
  $("confidence-value").textContent = Number($("confidence").value).toFixed(2);
  $("overlap-value").textContent = `${Math.round(100 * $("overlap").value)}%`;
  $("overlap-row").hidden = $("method").value !== "overlap";
  if (!processed && !busy) updateHud();
});
function renderSlots() {
  $("space-count").textContent = config?.slots.length ?? 0;
  $("space-scope").textContent = processed ? "Final processed frame; not live playback counts." : busy ? "Latest processed frame" : "Configured spaces; occupancy pending analysis.";
  $("space-list").replaceChildren();
  for (const slot of config?.slots ?? []) {
    const occupied = states[slot.id];
    if (filter === "occupied" && occupied !== true || filter === "available" && occupied !== false) continue;
    const card = document.createElement("button"); card.type = "button"; card.className = "space-card";
    card.classList.toggle("selected", selected === slot.id);
    const dot = document.createElement("i"); dot.className = `dot ${occupied === true ? "red" : occupied === false ? "green" : ""}`;
    const name = document.createElement("strong"); name.textContent = `${slot.label} · ${slot.id}`;
    const status = document.createElement("small"); const probability = probabilities[slot.id];
    status.textContent = (occupied === true ? "OCCUPIED" : occupied === false ? "AVAILABLE" : "READY") +
      (selected === slot.id && probability != null ? ` · P(occupied) ${Math.round(100 * probability)}%` : "");
    card.append(dot, name, status); card.onclick = () => { selected = slot.id; renderSlots(); };
    $("space-list").append(card);
  }
  drawRois();
}
function drawRois() {
  const svg = $("roi-svg"); svg.replaceChildren(); svg.hidden = processed || !showRois || !config || !player.videoWidth;
  if (svg.hidden) return;
  const w = player.videoWidth, h = player.videoHeight;
  const scale = Math.min(player.clientWidth / w, player.clientHeight / h);
  svg.style.width = `${w * scale}px`; svg.style.height = `${h * scale}px`; svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  for (const slot of config.slots) {
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.setAttribute("points", slot.points.map((p) => p.join(",")).join(" "));
    polygon.classList.toggle("selected", selected === slot.id); svg.append(polygon);
  }
}
new ResizeObserver(drawRois).observe($("video-shell"));
document.querySelectorAll("[data-filter]").forEach((button) => { button.onclick = () => {
  filter = button.dataset.filter;
  document.querySelectorAll("[data-filter]").forEach((item) => item.classList.toggle("active", item === button));
  renderSlots();
}; });
for (const id of ["roi-chip", "toggle-rois"]) $(id).onclick = () => {
  if (processed) { alertUser("ROI annotations are embedded in the processed video. This switch controls the input preview overlay."); return; }
  showRois = !showRois;
  for (const control of ["roi-chip", "toggle-rois"]) $(control).setAttribute("aria-pressed", String(showRois));
  drawRois();
};
$("fit-video").onclick = () => { player.style.objectFit = "contain"; drawRois(); };
$("fullscreen").onclick = async () => {
  try { await $("video-shell").requestFullscreen(); } catch { alertUser("Fullscreen is unavailable in this browser window."); }
};
function counts(total, occupied, scope) {
  $("total").textContent = total; $("occupied").textContent = occupied; $("available").textContent = total - occupied;
  const percent = total ? 100 * occupied / total : 0;
  $("occupancy").textContent = `${percent.toFixed(1)}%`; $("occupancy-fill").style.width = `${percent}%`;
  $("summary-scope").textContent = scope;
}
function finish(job) {
  processed = true; setBusy(false); const result = job.result;
  counts(result.total, result.occupied, "Final processed frame");
  config = job.metadata; states = job.occupied ?? {}; updateHud(job.settings);
  probabilities = Object.fromEntries(result.slots.map((slot) => [slot.id, slot.occupied_probability]));
  renderSlots();
  $("stage-label").textContent = "ANNOTATED RESULT"; $("empty-state").hidden = true;
  $("downloads").hidden = false;
  $("download-video").href = `/api/output/${job.job_id}?download=true`;
  $("download-config").href = `/api/files/${job.job_id}/config`;
  $("download-csv").href = `/api/files/${job.job_id}/csv`;
  $("reference").href = `/api/files/${job.job_id}/reference`;
  $("video-shell").hidden = false; player.poster = `/api/files/${job.job_id}/reference`;
  if (result.video.browser_playable) player.src = `/api/output/${job.job_id}`;
  else { player.removeAttribute("src"); player.load(); }
  if (result.video.warning) alertUser(result.video.warning);
}
async function pollJob(id) {
  let failures = 0;
  while (activeJob === id) {
    try {
      const job = await api(`/api/status/${id}`); failures = 0;
      $("processing-stage").textContent = job.stage;
      $("search-title").textContent = job.filename;
      config = job.metadata; states = job.occupied ?? {}; updateHud(job.settings); renderSlots();
      $("file-info").hidden = false; $("file-name").textContent = job.filename;
      $("file-meta").textContent = `${job.metadata.width} × ${job.metadata.height} · ${job.metadata.fps?.toFixed(2) ?? "unknown"} FPS · ${job.metadata.frame_count ?? "unknown"} frames`;
      if (job.frames_written) counts(config.slots.length, Object.values(states).filter(Boolean).length, `Processed frame ${job.frames_written}`);
      if (job.state === "complete") { finish(job); return; }
      if (job.state === "failed") { setBusy(false); alertUser(job.error); sessionStorage.removeItem("cloudforge-job"); return; }
      const encoding = job.stage === "Generating browser video";
      if (job.expected_frames && !encoding) {
        $("progress").value = Math.min(100, 100 * job.frames_written / job.expected_frames);
        $("progress-text").textContent = `${job.frames_written} / ${job.expected_frames} frames processed (${Math.floor($("progress").value)}%)`;
      } else {
        $("progress").removeAttribute("value");
        $("progress-text").textContent = encoding ? "Frames complete. Preparing browser playback…" : `${job.frames_written} frames processed · waiting for frame totals`;
      }
    } catch (error) {
      if (++failures >= 5) { setBusy(false); alertUser(`${error.message} Reload to reconnect to the saved job.`); return; }
      $("progress-text").textContent = "Connection interrupted; reconnecting to analysis…";
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}
$("analysis-form").onsubmit = (event) => {
  event.preventDefault(); if (busy || !videoFile || !slotsFile) return;
  resetResult(); setBusy(true); player.pause(); showView("dashboard"); $("alert").hidden = true;
  $("processing-stage").textContent = "Uploading footage"; $("progress").removeAttribute("value");
  const form = new FormData(); form.append("video", videoFile); form.append("slots", slotsFile);
  form.append("confidence", $("confidence").value); form.append("occupancy_method", $("method").value);
  form.append("overlap_threshold", $("overlap").value); form.append("smoothing", $("smoothing").checked);
  form.append("process_every", $("process-every").value);
  form.append("engine", $("engine").value); form.append("classification_threshold", $("threshold").value);
  form.append("show_rois", $("show-rois").checked); form.append("show_ids", $("show-ids").checked);
  form.append("show_boxes", $("show-boxes").checked);
  const xhr = new XMLHttpRequest(); xhr.open("POST", "/api/process-video");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) { $("progress").value = 100 * e.loaded / e.total; $("progress-text").textContent = `${Math.floor($("progress").value)}% uploaded`; }
  };
  xhr.upload.onload = () => { $("processing-stage").textContent = "Validating video and parking spaces"; $("progress").removeAttribute("value"); };
  xhr.onerror = () => { setBusy(false); alertUser("Cannot reach the local server. Check that it is running before retrying."); };
  xhr.onload = () => {
    let response; try { response = JSON.parse(xhr.responseText); } catch { setBusy(false); alertUser("The server returned an unreadable response."); return; }
    if (xhr.status !== 202) { setBusy(false); alertUser(response.detail || "Upload failed. Check your files."); return; }
    activeJob = response.job_id; sessionStorage.setItem("cloudforge-job", activeJob); pollJob(activeJob);
  };
  xhr.send(form);
};

async function loadInsights() {
  try {
    const data = await api("/api/insights");
    const model = (key, title, accent) => {
      const m = data.models[key].external;
      return `<article class="panel model-card ${accent ? "accent-panel" : ""}"><span class="eyebrow">EXTERNAL EVALUATION · CNR-EXT</span><h3>${title}</h3><div class="metric-pair"><div><strong>${pct(m.accuracy, key === "mobilenet" ? 3 : 2)}<em>%</em></strong><small>External accuracy</small></div><div><strong>${pct(m.f1)}<em>%</em></strong><small>External F1</small></div></div><div class="extra-metrics"><span>Precision <b>${pct(m.precision)}%</b></span><span>Occupied recall <b>${pct(m.recall)}%</b></span></div></article>`;
    };
    const bars = (rows, occlusion) => rows.map((row) => `<div class="bar-row ${row.condition === "HEAVY" ? "warning" : ""}"><span>${escapeHtml(row.condition.toLowerCase())}</span><div class="bar-track"><i style="width:${pct(row.accuracy)}%"></i></div><strong>${occlusion ? `${row.correct}/${row.sample_count}` : `${pct(row.accuracy)}%`}</strong></div>`).join("");
    const weather = data.lighting.filter((row) => row.model === "mobilenet" && row.dimension === "weather");
    const occlusion = data.occlusion.filter((row) => row.model === "mobilenet").sort((a,b) => ["NONE","PARTIAL","HEAVY"].indexOf(a.condition) - ["NONE","PARTIAL","HEAVY"].indexOf(b.condition));
    $("insights-content").innerHTML = `<div class="comparison-grid">${model("baseline", "Baseline CNN", false)}${model("mobilenet", "MobileNetV2", true)}</div><div class="gain-strip"><span><strong>+${data.accuracy_gain_pp.toFixed(3)}</strong>accuracy points</span><span><strong>+${data.f1_gain_pp.toFixed(2)}</strong>F1 points</span><span>MobileNetV2 vs. baseline · external test</span></div><div class="dataset-strip"><div><strong>CNR-EXT</strong><span>External test · never used for training</span></div><div><strong>${data.dataset.selected.toLocaleString()}</strong><span>Parking crops</span></div><div><strong>${Object.keys(data.dataset.camera_counts).length}</strong><span>Cameras</span></div><div><strong>${data.dataset.day_count}</strong><span>Days</span></div></div><p class="insights-note">OCCUPIED is the positive class. MobileNetV2 has higher external accuracy but lower occupied recall than the baseline: some occupied spaces are predicted vacant. These crop-classifier scores do not measure the YOLO video demo's occupancy accuracy.</p><div class="insight-grid"><article class="panel"><span class="eyebrow">ROBUSTNESS · MOBILENETV2</span><h3>Across the weather.</h3><p>Accuracy on the external weather groups.</p>${bars(weather, false)}</article><article class="panel"><span class="eyebrow">ROBUSTNESS · REVIEWED CROPS</span><h3>When the view is obstructed.</h3><p>Correct / reviewed examples by occlusion.</p>${bars(occlusion, true)}<p class="warning-note">Heavy occlusion contains only seven reviewed examples (3/7 correct). These small, subjective groups are descriptive, not a general performance guarantee.</p></article></div><p class="insights-note">Dataset A is a repetitive fixed parking scene. Its in-domain 100% results are less informative about generalization than the external CNR-EXT evaluation. Custom YOLO mAP was not measured.</p><section class="artifact-section"><span class="eyebrow">SAVED EXPERIMENT ARTIFACTS</span><h3>Inspect the evidence.</h3><div class="artifact-grid" id="artifact-grid"></div></section>`;
    for (const artifact of data.artifacts) {
      if (artifact.name.endsWith(".md")) {
        const link = document.createElement("a"); link.href = artifact.url; link.textContent = "Read the verified results summary ↗"; link.target = "_blank"; link.rel = "noopener"; $("artifact-grid").append(link); continue;
      }
      const button = document.createElement("button"); button.className = "artifact-card";
      const img = document.createElement("img"); img.src = artifact.url; img.alt = artifact.title; img.loading = "lazy";
      const title = document.createElement("span"); title.textContent = artifact.title; button.append(img, title);
      button.onclick = () => { $("artifact-title").textContent = artifact.title; $("artifact-image").src = artifact.url; $("artifact-image").alt = artifact.title; $("artifact-link").href = artifact.url; $("artifact-dialog").showModal(); };
      $("artifact-grid").append(button);
    }
    insightsLoaded = true;
  } catch (error) { $("insights-content").textContent = error.message; alertUser(error.message); }
}
api("/api/health").then((health) => {
  $("connection").classList.add("online"); $("connection").querySelector("span").textContent = "Local · online";
  if (!health.model_file_available) alertUser("YOLO11n weights are missing. Place yolo11n.pt in the project folder before analysing.");
}).catch(() => alertUser("Local server unavailable. Start the app and reload this page."));
const savedJob = sessionStorage.getItem("cloudforge-job");
if (savedJob) { activeJob = savedJob; setBusy(true); pollJob(savedJob); }
applyEngine();
