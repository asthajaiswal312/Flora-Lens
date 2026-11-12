"use strict";

const state = {
  file: null,
  previewUrl: null,
  prediction: null,
  status: null,
  classes: [],
  sightings: [],
  stats: { total: 0, verified: 0, pending: 0, species: 0 },
  map: null,
  mapLayer: null,
  toastTimer: null,
};

const byId = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch (_) { /* Keep the HTTP error message. */ }
    throw new Error(message);
  }
  return response.json();
}

function showToast(message, isError = false) {
  const toast = byId("toast");
  clearTimeout(state.toastTimer);
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("show");
  state.toastTimer = setTimeout(() => toast.classList.remove("show"), 4500);
}

function setBusy(button, busy, busyText = "Working…") {
  if (busy) {
    button.dataset.originalHtml = button.innerHTML;
    button.disabled = true;
    button.textContent = busyText;
  } else {
    button.innerHTML = button.dataset.originalHtml || button.innerHTML;
    button.disabled = false;
  }
}

function setObservationNow() {
  const now = new Date();
  now.setSeconds(0, 0);
  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60000);
  byId("observed-at").value = local.toISOString().slice(0, 16);
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown date";
  return new Intl.DateTimeFormat(undefined, {
    day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
  }).format(date);
}

async function refreshStatus() {
  try {
    state.status = await api("/api/status");
    if (state.status.model_loaded) {
      const classData = await api("/api/classes");
      state.classes = classData.classes;
    }
    updateStatusUI();
  } catch (error) {
    byId("model-status-short").textContent = "Status unavailable";
    byId("model-banner-title").textContent = "Could not check model status";
    byId("model-banner-text").textContent = error.message;
  }
}

function updateStatusUI() {
  const status = state.status;
  if (!status) return;
  const chip = byId("model-status-button");
  const banner = byId("model-banner");
  chip.className = `status-chip ${status.mode}`;
  banner.className = `model-banner ${status.mode}`;
  byId("model-status-short").textContent = status.model_loaded
    ? `${status.class_count}-class model connected`
    : status.mode === "error" ? "Model setup error" : "Interface preview";
  byId("model-banner-title").textContent = status.model_loaded
    ? "Your flower classifier is connected."
    : status.mode === "error" ? "The model checkpoint needs attention." : "Preview mode — no model inference yet.";
  byId("model-banner-text").textContent = status.mode === "error" && status.error
    ? status.error : status.message;
  byId("setup-error").textContent = status.error || "";
  const identifyText = byId("identify-button").querySelector("span");
  identifyText.textContent = status.model_loaded ? "Identify flower" : "Preview identification flow";
}

function switchView(view) {
  document.querySelectorAll(".app-view").forEach((element) => {
    element.hidden = element.id !== `${view}-view`;
  });
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  if (view === "dashboard") {
    requestAnimationFrame(() => {
      initMap();
      if (state.map) state.map.invalidateSize();
      renderDashboard();
    });
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function resetPrediction() {
  state.prediction = null;
  byId("prediction-empty").hidden = false;
  byId("prediction-content").hidden = true;
  byId("save-button").disabled = true;
}

function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showToast("Please choose a JPG, PNG, or WebP image.", true);
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast("The image is larger than the 10 MB limit.", true);
    return;
  }
  state.file = file;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  byId("photo-preview").src = state.previewUrl;
  byId("photo-preview").hidden = false;
  byId("photo-overlay").hidden = false;
  byId("upload-empty").hidden = true;
  byId("identify-button").disabled = false;
  resetPrediction();
}

function resetCollector() {
  byId("sighting-form").reset();
  state.file = null;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  byId("photo-preview").removeAttribute("src");
  byId("photo-preview").hidden = true;
  byId("photo-overlay").hidden = true;
  byId("upload-empty").hidden = false;
  byId("identify-button").disabled = true;
  byId("location-feedback").textContent = "Location permission is requested only when you press the button.";
  resetPrediction();
  setObservationNow();
}

async function identifyFlower() {
  if (!state.file) return;
  const button = byId("identify-button");
  setBusy(button, true, state.status?.model_loaded ? "Running ResNet50…" : "Loading illustrative result…");
  const data = new FormData();
  data.append("file", state.file);
  try {
    state.prediction = await api("/api/predict", { method: "POST", body: data });
    renderPrediction();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(button, false);
  }
}

function renderPrediction() {
  const result = state.prediction;
  if (!result || !result.predictions.length) return;
  const top = result.predictions[0];
  byId("prediction-empty").hidden = true;
  byId("prediction-content").hidden = false;
  byId("top-species").textContent = top.species;
  byId("top-score").textContent = `${Math.round(top.confidence * 100)}%`;
  document.querySelector(".score-ring span").textContent = result.demo ? "sample score" : "model score";
  byId("review-alert").hidden = !result.needs_review;
  const demoNotice = byId("demo-notice");
  demoNotice.hidden = !result.demo;
  demoNotice.textContent = result.notice;
  byId("prediction-list").innerHTML = result.predictions.map((prediction, index) => {
    const percentage = Math.max(0, Math.min(100, prediction.confidence * 100));
    return `<div class="prediction-row">
      <div class="prediction-meta"><span>${index + 1}. ${escapeHtml(prediction.species)}</span><span>${percentage.toFixed(1)}%</span></div>
      <div class="score-track"><div class="score-fill" style="width:${percentage.toFixed(2)}%"></div></div>
    </div>`;
  }).join("");

  const select = byId("confirmed-species");
  select.replaceChildren();
  const suggestions = document.createElement("optgroup");
  suggestions.label = result.demo ? "Illustrative suggestions" : "Top model suggestions";
  const added = new Set();
  result.predictions.forEach((prediction) => {
    const option = document.createElement("option");
    option.value = prediction.species;
    option.textContent = prediction.species;
    suggestions.appendChild(option);
    added.add(prediction.species);
  });
  select.appendChild(suggestions);
  if (state.classes.length) {
    const allClasses = document.createElement("optgroup");
    allClasses.label = "All model classes";
    state.classes.forEach((name) => {
      if (added.has(name)) return;
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      allClasses.appendChild(option);
    });
    select.appendChild(allClasses);
  }
  const unknown = document.createElement("option");
  unknown.value = "Unknown / needs expert review";
  unknown.textContent = "Unknown / needs expert review";
  select.appendChild(unknown);
  select.value = top.species;
  byId("save-button").disabled = false;
}

function useCurrentLocation() {
  const button = byId("location-button");
  const feedback = byId("location-feedback");
  if (!navigator.geolocation) {
    feedback.textContent = "Geolocation is unavailable in this browser. Enter coordinates manually.";
    return;
  }
  setBusy(button, true, "Requesting GPS…");
  feedback.textContent = "Waiting for location permission…";
  navigator.geolocation.getCurrentPosition(
    (position) => {
      byId("latitude").value = position.coords.latitude.toFixed(6);
      byId("longitude").value = position.coords.longitude.toFixed(6);
      feedback.textContent = `Location captured · approximately ${Math.round(position.coords.accuracy)} m accuracy.`;
      setBusy(button, false);
    },
    (error) => {
      const messages = {
        1: "Location permission was denied. Enter coordinates manually.",
        2: "Your location is currently unavailable. Enter coordinates manually.",
        3: "Location request timed out. Try again or enter coordinates manually.",
      };
      feedback.textContent = messages[error.code] || "Could not obtain GPS coordinates.";
      setBusy(button, false);
    },
    { enableHighAccuracy: true, timeout: 12000, maximumAge: 60000 },
  );
}

async function saveSighting(event) {
  event.preventDefault();
  if (!state.file || !state.prediction) {
    showToast("Add a photo and review its identification first.", true);
    return;
  }
  const latitude = Number(byId("latitude").value);
  const longitude = Number(byId("longitude").value);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    showToast("Capture or enter valid GPS coordinates.", true);
    return;
  }
  const observedValue = byId("observed-at").value;
  if (!observedValue) {
    showToast("Choose the observation date and time.", true);
    return;
  }
  const button = byId("save-button");
  setBusy(button, true, "Saving sighting…");
  const data = new FormData();
  data.append("file", state.file);
  data.append("species", byId("confirmed-species").value);
  data.append("predicted_species", state.prediction.predictions[0].species);
  data.append("confidence", state.prediction.predictions[0].confidence);
  data.append("top_predictions", JSON.stringify(state.prediction.predictions));
  data.append("latitude", latitude);
  data.append("longitude", longitude);
  data.append("observed_at", new Date(observedValue).toISOString());
  data.append("observer", byId("observer").value);
  data.append("notes", byId("notes").value);
  try {
    await api("/api/sightings", { method: "POST", body: data });
    showToast(state.prediction.demo ? "Illustrative sighting saved and labelled as sample data." : "Sighting saved for verification.");
    resetCollector();
    await loadSightings();
    switchView("dashboard");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(button, false);
    button.disabled = !state.prediction;
  }
}

async function loadSightings() {
  try {
    const [records, stats] = await Promise.all([api("/api/sightings"), api("/api/stats")]);
    state.sightings = records.sightings;
    state.stats = stats;
    populateSpeciesFilter();
    renderDashboard();
  } catch (error) {
    showToast(error.message, true);
  }
}

function populateSpeciesFilter() {
  const select = byId("species-filter");
  const current = select.value;
  const names = [...new Set(state.sightings.map((sighting) => sighting.species))].sort();
  select.innerHTML = '<option value="">All labels</option>' + names.map((name) =>
    `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
  if (names.includes(current)) select.value = current;
}

function filteredSightings() {
  const species = byId("species-filter").value;
  const status = byId("status-filter").value;
  const fromValue = byId("from-date").value;
  const toValue = byId("to-date").value;
  const from = fromValue ? new Date(`${fromValue}T00:00:00`).getTime() : -Infinity;
  const to = toValue ? new Date(`${toValue}T23:59:59.999`).getTime() : Infinity;
  return state.sightings.filter((sighting) => {
    const observed = new Date(sighting.observed_at).getTime();
    return (!species || sighting.species === species)
      && (!status || sighting.status === status)
      && observed >= from && observed <= to;
  });
}

function renderDashboard() {
  byId("stat-total").textContent = state.stats.total;
  byId("stat-verified").textContent = state.stats.verified;
  byId("stat-pending").textContent = state.stats.pending;
  byId("stat-species").textContent = state.stats.species;
  const hasSamples = state.sightings.some((sighting) => sighting.source === "sample");
  byId("sample-dashboard-note").hidden = !hasSamples;
  byId("clear-samples").hidden = !hasSamples;
  const sightings = filteredSightings();
  byId("map-result-count").textContent = `${sightings.length} sighting${sightings.length === 1 ? "" : "s"}`;
  byId("record-count").textContent = `Showing ${sightings.length} of ${state.sightings.length} records`;
  renderSpeciesChart(sightings);
  renderRecords(sightings);
  renderMap(sightings);
}

function renderSpeciesChart(sightings) {
  const counts = {};
  sightings.forEach((sighting) => { counts[sighting.species] = (counts[sighting.species] || 0) + 1; });
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const maximum = rows.length ? rows[0][1] : 1;
  byId("species-chart").innerHTML = rows.length ? rows.map(([name, count]) =>
    `<div class="species-chart-row"><div class="prediction-meta"><span>${escapeHtml(name)}</span><span>${count}</span></div><div class="score-track"><div class="score-fill" style="width:${(count / maximum * 100).toFixed(1)}%"></div></div></div>`).join("")
    : '<p class="field-help">No labels to summarise for these filters.</p>';
}

function renderRecords(sightings) {
  byId("records-empty").hidden = sightings.length > 0;
  byId("records-body").innerHTML = sightings.map((sighting) => {
    const image = sighting.image_url
      ? `<img class="record-thumb" src="${escapeHtml(sighting.image_url)}" alt="">`
      : '<span class="record-thumb" aria-hidden="true">✣</span>';
    const sample = sighting.source === "sample" ? '<span class="sample-tag">sample</span>' : "";
    const score = sighting.confidence == null ? "—" : `${(sighting.confidence * 100).toFixed(1)}%`;
    const verify = sighting.status !== "verified"
      ? `<button data-action="verified" data-id="${sighting.id}" type="button">Verify</button>` : "";
    const reject = sighting.status !== "rejected"
      ? `<button data-action="rejected" data-id="${sighting.id}" type="button">Reject</button>` : "";
    return `<tr>
      <td><div class="species-cell">${image}<div><strong>${escapeHtml(sighting.species)}${sample}</strong><small>${escapeHtml(sighting.observer || "Unnamed observer")}</small></div></div></td>
      <td>${escapeHtml(formatDate(sighting.observed_at))}</td>
      <td>${Number(sighting.latitude).toFixed(4)}, ${Number(sighting.longitude).toFixed(4)}</td>
      <td>${score}</td>
      <td><span class="status-pill ${escapeHtml(sighting.status)}">${escapeHtml(sighting.status)}</span></td>
      <td><div class="review-actions">${verify}${reject}<button class="delete" data-action="delete" data-id="${sighting.id}" type="button">Delete</button></div></td>
    </tr>`;
  }).join("");
}

function initMap() {
  if (state.map) return;
  if (!window.L) {
    byId("map-notice").hidden = false;
    byId("map-notice").textContent = "The map library did not load. Coordinates remain available in the records table.";
    return;
  }
  state.map = L.map("map", { scrollWheelZoom: false }).setView([26.8467, 80.9462], 11);
  const tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  });
  tiles.on("tileerror", () => { byId("map-notice").hidden = false; });
  tiles.addTo(state.map);
  L.control.scale({ imperial: false }).addTo(state.map);
  state.mapLayer = L.layerGroup().addTo(state.map);
}

function popupFor(sighting) {
  const content = document.createElement("div");
  content.className = "popup-content";
  if (sighting.image_url) {
    const image = document.createElement("img");
    image.src = sighting.image_url;
    image.alt = "Uploaded sighting";
    content.appendChild(image);
  }
  const title = document.createElement("strong");
  title.textContent = sighting.species;
  const status = document.createElement("span");
  status.textContent = `${sighting.status.toUpperCase()}${sighting.source === "sample" ? " · SAMPLE" : ""}`;
  const date = document.createElement("span");
  date.textContent = formatDate(sighting.observed_at);
  const coordinates = document.createElement("span");
  coordinates.textContent = `${Number(sighting.latitude).toFixed(5)}, ${Number(sighting.longitude).toFixed(5)}`;
  content.append(title, status, date, coordinates);
  if (sighting.notes) {
    const notes = document.createElement("span");
    notes.textContent = sighting.notes;
    content.appendChild(notes);
  }
  return content;
}

function renderMap(sightings) {
  if (!state.map || !state.mapLayer) return;
  state.mapLayer.clearLayers();
  const bounds = [];
  sightings.forEach((sighting) => {
    const position = [Number(sighting.latitude), Number(sighting.longitude)];
    const icon = L.divIcon({
      className: "sighting-div-icon",
      html: `<span class="map-pin ${escapeHtml(sighting.status)} ${escapeHtml(sighting.source)}"></span>`,
      iconSize: [20, 20], iconAnchor: [10, 19], popupAnchor: [0, -18],
    });
    const marker = L.marker(position, { icon }).bindPopup(popupFor(sighting));
    marker.addTo(state.mapLayer);
    bounds.push(position);
  });
  if (bounds.length === 1) state.map.setView(bounds[0], 14);
  else if (bounds.length > 1) state.map.fitBounds(bounds, { padding: [35, 35], maxZoom: 14 });
}

async function reviewRecord(id, action) {
  if (action === "delete") {
    if (!window.confirm("Delete this sighting permanently?")) return;
    try {
      await api(`/api/sightings/${id}`, { method: "DELETE" });
      showToast("Sighting deleted.");
      await loadSightings();
    } catch (error) { showToast(error.message, true); }
    return;
  }
  try {
    await api(`/api/sightings/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: action }),
    });
    showToast(action === "verified" ? "Sighting marked as verified." : "Sighting marked as rejected.");
    await loadSightings();
  } catch (error) { showToast(error.message, true); }
}

async function clearSamples() {
  if (!window.confirm("Remove all records labelled as sample data? This cannot be undone.")) return;
  try {
    const result = await api("/api/sightings/samples", { method: "DELETE" });
    showToast(`${result.deleted} sample record${result.deleted === 1 ? "" : "s"} removed.`);
    await loadSightings();
  } catch (error) { showToast(error.message, true); }
}

function openSetup() {
  const dialog = byId("setup-dialog");
  if (typeof dialog.showModal === "function") dialog.showModal();
}

function bindEvents() {
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  document.querySelector(".brand").addEventListener("click", (event) => { event.preventDefault(); switchView("collect"); });
  byId("new-sighting").addEventListener("click", () => switchView("collect"));
  byId("file-input").addEventListener("change", (event) => handleFile(event.target.files[0]));
  const dropZone = byId("drop-zone");
  ["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => {
    event.preventDefault(); dropZone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => {
    event.preventDefault(); dropZone.classList.remove("dragging");
  }));
  dropZone.addEventListener("drop", (event) => handleFile(event.dataTransfer.files[0]));
  byId("identify-button").addEventListener("click", identifyFlower);
  byId("location-button").addEventListener("click", useCurrentLocation);
  byId("sighting-form").addEventListener("submit", saveSighting);
  ["species-filter", "status-filter", "from-date", "to-date"].forEach((id) => {
    byId(id).addEventListener("change", renderDashboard);
  });
  byId("clear-filters").addEventListener("click", () => {
    ["species-filter", "status-filter", "from-date", "to-date"].forEach((id) => { byId(id).value = ""; });
    renderDashboard();
  });
  byId("records-body").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    if (button) reviewRecord(Number(button.dataset.id), button.dataset.action);
  });
  byId("clear-samples").addEventListener("click", clearSamples);
  byId("clear-samples-inline").addEventListener("click", clearSamples);
  byId("open-setup").addEventListener("click", openSetup);
  byId("model-status-button").addEventListener("click", openSetup);
  byId("close-setup").addEventListener("click", () => byId("setup-dialog").close());
  byId("setup-dialog").addEventListener("click", (event) => {
    if (event.target === byId("setup-dialog")) byId("setup-dialog").close();
  });
}

async function initialise() {
  bindEvents();
  setObservationNow();
  await Promise.all([refreshStatus(), loadSightings()]);
}

document.addEventListener("DOMContentLoaded", initialise);
