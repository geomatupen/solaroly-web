(() => {
  "use strict";

  const state = {
    initialized: false,
    loaded: false,
    analysis: null,
    workflowId: null,
    pollToken: 0,
    geojsonFiles: [],
    map: null,
    previewLayers: new Map(),
    previewLoading: new Set(),
    workflows: [],
    scanComplete: false,
  };

  const PREVIEW_STYLES = {
    source: { label: "Source", color: "#38bdf8", weight: 1, fillOpacity: 0.08 },
    combined: { label: "Combined", color: "#f59e0b", weight: 2, fillOpacity: 0.13 },
    regularized: { label: "Regularized", color: "#22c55e", weight: 2, fillOpacity: 0.16 },
  };

  const byId = id => document.getElementById(id);

  async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    let payload = {};
    try { payload = await response.json(); } catch (_) { /* response without JSON */ }
    if (!response.ok) {
      const detail = payload.detail;
      const message = Array.isArray(detail)
        ? detail.map(item => item.msg || String(item)).join("; ")
        : detail || payload.error || `Request failed (${response.status})`;
      throw new Error(message);
    }
    return payload;
  }

  function setMessage(message = "", type = "") {
    const element = byId("ppMessage");
    element.textContent = message;
    element.className = `statusLine${type ? ` ${type}` : ""}`;
  }

  function numberValue(id) {
    return Number(byId(id).value);
  }

  function startIndeterminate(message) {
    const progress = byId("ppProgressWrap");
    progress.hidden = false;
    progress.classList.add("ppIndeterminate");
    byId("ppProgressText").textContent = message;
  }

  function stopIndeterminate() {
    const progress = byId("ppProgressWrap");
    progress.classList.remove("ppIndeterminate");
    progress.hidden = true;
  }

  function resetAnalysis() {
    state.analysis = null;
    state.scanComplete = false;
    state.workflowId = null;
    state.pollToken += 1;
    byId("ppSummary").hidden = true;
    byId("ppSummary").replaceChildren();
    byId("ppCombine").disabled = true;
    byId("ppRegularize").disabled = true;
    byId("ppCombineStep").hidden = true;
    byId("ppRegularizeStep").hidden = true;
    byId("ppProgressWrap").hidden = true;
    byId("ppOutputs").hidden = true;
    byId("ppOutputs").replaceChildren();
    byId("ppLogWrap").hidden = true;
    byId("ppLog").textContent = "";
    clearPreviewLayers();
  }

  function ensurePreviewMap() {
    if (state.map || !window.L || !byId("ppMap")) return state.map;
    state.map = window.L.map("ppMap", { preferCanvas: true, zoomControl: true }).setView([0, 0], 2);
    window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 22,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(state.map);
    return state.map;
  }

  function clearPreviewLayers() {
    if (state.map) {
      for (const item of state.previewLayers.values()) {
        if (state.map.hasLayer(item.layer)) state.map.removeLayer(item.layer);
      }
    }
    state.previewLayers.clear();
    state.previewLoading.clear();
    byId("ppFitLayers").disabled = true;
    byId("ppLayerList").innerHTML = '<div class="muted tiny">No preview layers loaded.</div>';
    byId("ppMapStatus").textContent = "Scan a GeoJSON to preview it.";
  }

  function fitVisibleLayers() {
    const map = ensurePreviewMap();
    if (!map) return;
    const bounds = [];
    for (const item of state.previewLayers.values()) {
      if (map.hasLayer(item.layer)) {
        const layerBounds = item.layer.getBounds?.();
        if (layerBounds?.isValid()) bounds.push(layerBounds);
      }
    }
    if (bounds.length) {
      const combined = bounds.slice(1).reduce((value, next) => value.extend(next), bounds[0]);
      map.fitBounds(combined, { padding: [18, 18], maxZoom: 21 });
    }
  }

  function renderPreviewLayers() {
    const container = byId("ppLayerList");
    container.replaceChildren();
    for (const [key, item] of state.previewLayers) {
      const row = document.createElement("div");
      row.className = "postprocessLayerItem";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = state.map?.hasLayer(item.layer) || false;
      checkbox.title = `Show ${item.label}`;
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) item.layer.addTo(state.map);
        else state.map.removeLayer(item.layer);
      });
      const swatch = document.createElement("span");
      swatch.className = "postprocessLayerSwatch";
      swatch.style.color = item.color;
      const text = document.createElement("div");
      text.className = "postprocessLayerText";
      const name = document.createElement("strong");
      name.textContent = item.label;
      const detail = document.createElement("small");
      detail.textContent = `${Number(item.count).toLocaleString()} polygons`;
      text.append(name, detail);
      const focus = document.createElement("button");
      focus.type = "button";
      focus.className = "secondary tiny";
      focus.textContent = "Focus";
      focus.addEventListener("click", () => {
        const bounds = item.layer.getBounds?.();
        if (bounds?.isValid()) state.map.fitBounds(bounds, { padding: [18, 18], maxZoom: 21 });
      });
      const download = document.createElement("a");
      download.className = "postprocessLayerDownload";
      download.href = item.url;
      download.download = "";
      download.title = `Download ${item.label} GeoJSON`;
      download.textContent = "Download";
      row.append(checkbox, swatch, text, focus, download);
      container.appendChild(row);
    }
    byId("ppFitLayers").disabled = !state.previewLayers.size;
  }

  async function loadPreviewLayer(stage, url, expectedCount = null) {
    if (!url || state.previewLayers.get(stage)?.url === url || state.previewLoading.has(stage)) return;
    const map = ensurePreviewMap();
    if (!map) {
      byId("ppMapStatus").textContent = "Map preview is unavailable because Leaflet did not load.";
      return;
    }
    state.previewLoading.add(stage);
    const style = PREVIEW_STYLES[stage] || PREVIEW_STYLES.source;
    byId("ppMapStatus").textContent = `Loading ${style.label.toLowerCase()} polygons…`;
    try {
      const geojson = await requestJson(url, { cache: "no-store" });
      const existing = state.previewLayers.get(stage);
      if (existing && map.hasLayer(existing.layer)) map.removeLayer(existing.layer);
      const baseStyle = {
        color: style.color,
        weight: style.weight,
        fillColor: style.color,
        fillOpacity: style.fillOpacity,
      };
      const layer = window.L.geoJSON(geojson, {
        renderer: window.L.canvas({ padding: 0.5 }),
        style: () => baseStyle,
        onEachFeature: (feature, polygonLayer) => {
          polygonLayer.on("mouseover", event => {
            event.target.setStyle({
              color: "#ffffff",
              weight: style.weight + 2,
              fillColor: style.color,
              fillOpacity: 0.38,
            });
            event.target.bringToFront?.();
          });
          polygonLayer.on("mouseout", event => event.target.setStyle(baseStyle));
          const fragments = Number(feature.properties?.source_feature_count || 1);
          const suffix = fragments > 1 ? ` · ${fragments} source fragments` : "";
          polygonLayer.bindTooltip(`${style.label} panel${suffix}`, {
            sticky: true,
            direction: "top",
            opacity: 0.95,
          });
        },
      });
      // Show the newest stage by itself initially. Users can turn earlier stages
      // back on for comparison using the layer list.
      for (const item of state.previewLayers.values()) {
        if (map.hasLayer(item.layer)) map.removeLayer(item.layer);
      }
      layer.addTo(map);
      state.previewLayers.set(stage, {
        label: style.label,
        color: style.color,
        count: expectedCount ?? geojson.features?.length ?? 0,
        layer,
        url,
      });
      renderPreviewLayers();
      const bounds = layer.getBounds();
      if (bounds.isValid()) map.fitBounds(bounds, { padding: [18, 18], maxZoom: 21 });
      byId("ppMapStatus").textContent = `${style.label} preview loaded.`;
    } catch (error) {
      byId("ppMapStatus").textContent = `Could not load ${style.label.toLowerCase()} preview: ${error.message}`;
    } finally {
      state.previewLoading.delete(stage);
    }
  }

  function syncOutputPreviews(status) {
    if (status.outputs?.combined?.url) {
      void loadPreviewLayer("combined", status.outputs.combined.url, status.combine_stats?.output_features);
    }
    if (status.outputs?.regularized?.url) {
      void loadPreviewLayer("regularized", status.outputs.regularized.url, status.regularize_stats?.output_features);
    }
  }

  function workflowDisplayName(workflow) {
    return workflow.display_name || workflow.parameters?.output_name || workflow.id;
  }

  function sourceIdentity(path) {
    return String(path || "")
      .replace(/(^|\/)anomalies\.geojson$/i, "$1predictions.geojson")
      .replace(/(^|\/)final_anomalies\.geojson$/i, "$1filtered_predictions.geojson");
  }

  function populateRegularizeSources(preferredId = null) {
    const select = byId("ppRegularizeSource");
    const combined = state.workflows.filter(workflow => workflow.outputs?.combined);
    select.replaceChildren();
    addOption(select, "", "Select a combined output…");
    combined.forEach((workflow, index) => {
      const created = workflow.created_at
        ? new Date(workflow.created_at).toLocaleString()
        : workflow.id;
      const latest = index === 0 ? " · Latest" : "";
      addOption(select, workflow.id, `${workflowDisplayName(workflow)}${latest} · ${created}`);
    });
    const selected = combined.find(workflow => workflow.id === preferredId)
      || combined[0];
    if (selected) select.value = selected.id;
    select.disabled = !combined.length;
  }

  function renderWorkflowList() {
    const container = byId("ppWorkflowList");
    container.replaceChildren();
    if (!state.workflows.length) {
      container.innerHTML = '<div class="muted tiny">No saved outputs.</div>';
      populateRegularizeSources();
      return;
    }
    for (const workflow of state.workflows) {
      const item = document.createElement("div");
      item.className = `postprocessWorkflowItem${workflow.id === state.workflowId ? " active" : ""}`;
      const info = document.createElement("div");
      info.className = "postprocessWorkflowInfo";
      const name = document.createElement("strong");
      name.textContent = workflowDisplayName(workflow);
      name.title = workflowDisplayName(workflow);
      const id = document.createElement("small");
      id.textContent = `ID: ${workflow.id}`;
      id.title = workflow.id;
      const stages = Object.keys(workflow.outputs || {}).join(" + ") || workflow.stage || "No output";
      const status = document.createElement("small");
      status.textContent = `${workflow.status || "unknown"} · ${stages}`;
      info.append(name, id, status);
      info.tabIndex = 0;
      info.setAttribute("role", "button");
      const open = () => {
        state.workflowId = workflow.id;
        applyWorkflow(workflow);
        if (workflow.outputs?.combined) populateRegularizeSources(workflow.id);
        renderWorkflowList();
      };
      info.addEventListener("click", open);
      info.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); }
      });

      const dots = document.createElement("button");
      dots.type = "button";
      dots.className = "iconDots";
      dots.textContent = "⋮";
      dots.setAttribute("aria-label", `Options for ${workflowDisplayName(workflow)}`);
      const menu = document.createElement("div");
      menu.className = "postprocessWorkflowMenu";
      menu.hidden = true;
      const rename = document.createElement("button");
      rename.type = "button";
      rename.textContent = "Rename";
      rename.addEventListener("click", async () => {
        menu.hidden = true;
        const next = window.prompt("Output name", workflowDisplayName(workflow));
        if (next == null || !next.trim() || next.trim() === workflowDisplayName(workflow)) return;
        try {
          await requestJson(`/api/results/${encodeURIComponent(byId("ppResult").value)}/postprocess/${encodeURIComponent(workflow.id)}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: next.trim() }),
          });
          await restoreLatestWorkflow(byId("ppResult").value, byId("ppGeojson").value, workflow.id);
          setMessage("Saved output renamed.", "ok");
        } catch (error) {
          setMessage(error.message, "err");
        }
      });
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "danger";
      remove.textContent = "Delete";
      remove.addEventListener("click", async () => {
        menu.hidden = true;
        if (!window.confirm(`Delete “${workflowDisplayName(workflow)}” and all of its post-processing files?`)) return;
        try {
          await requestJson(`/api/results/${encodeURIComponent(byId("ppResult").value)}/postprocess/${encodeURIComponent(workflow.id)}`, { method: "DELETE" });
          if (state.workflowId === workflow.id) {
            state.workflowId = null;
            clearPreviewLayers();
            byId("ppRegularize").disabled = true;
            byId("ppOutputs").hidden = true;
          }
          await restoreLatestWorkflow(byId("ppResult").value, byId("ppGeojson").value);
          setMessage("Saved output deleted.", "ok");
        } catch (error) {
          setMessage(error.message, "err");
        }
      });
      menu.append(rename, remove);
      dots.addEventListener("click", event => {
        event.stopPropagation();
        document.querySelectorAll(".postprocessWorkflowMenu").forEach(other => {
          if (other !== menu) other.hidden = true;
        });
        menu.hidden = !menu.hidden;
      });
      item.append(info, dots, menu);
      container.appendChild(item);
    }
    populateRegularizeSources(state.workflowId);
  }

  function addOption(select, value, label) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  }

  async function loadResults(force = false) {
    if (state.loaded && !force) return;
    const select = byId("ppResult");
    const previous = select.value;
    setMessage("Loading test results…");
    try {
      const payload = await requestJson("/api/sessions", { cache: "no-store" });
      select.replaceChildren();
      addOption(select, "", "Select a result…");
      for (const result of payload.sessions || []) {
        const name = result.display_name || result.name;
        const status = result.status === "complete" ? "Complete" : "Incomplete";
        addOption(select, result.id || result.name, `${name} · ${status} · ID: ${result.id || result.name}`);
      }
      if (previous && [...select.options].some(option => option.value === previous)) {
        select.value = previous;
      }
      state.loaded = true;
      setMessage(select.options.length > 1 ? "Select a result and GeoJSON to begin." : "No test results are available.", select.options.length > 1 ? "" : "warn");
      if (select.value) await loadGeojsons();
    } catch (error) {
      setMessage(error.message, "err");
    }
  }

  async function restoreLatestWorkflow(resultId, inputPath, preferredId = null) {
    if (!resultId || !inputPath) return false;
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess`, { cache: "no-store" });
      state.workflows = (payload.workflows || []).sort((first, second) =>
        String(second.created_at || "").localeCompare(String(first.created_at || ""))
      );
      const workflow = state.workflows.find(item => item.id === preferredId)
        || state.workflows.find(item => sourceIdentity(item.input_path) === sourceIdentity(inputPath) && item.outputs?.combined);
      renderWorkflowList();
      if (!workflow) {
        byId("ppRegularizeStep").hidden = true;
        return false;
      }
      state.workflowId = workflow.id;
      applyWorkflow(workflow);
      populateRegularizeSources(workflow.id);
      renderWorkflowList();
      return true;
    } catch (_) {
      return false;
    }
  }

  async function loadGeojsons() {
    resetAnalysis();
    state.workflows = [];
    renderWorkflowList();
    const resultId = byId("ppResult").value;
    const select = byId("ppGeojson");
    select.disabled = true;
    byId("ppAnalyze").disabled = true;
    select.replaceChildren();
    addOption(select, "", "Select a GeoJSON…");
    if (!resultId) {
      setMessage("Select a test result.");
      return;
    }
    setMessage("Looking for GeoJSON files in the selected result…");
    startIndeterminate("Loading the available GeoJSON file list…");
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/geojsons`);
      state.geojsonFiles = payload.files || [];
      for (const file of state.geojsonFiles) {
        addOption(select, file.path, `${file.name} · ${file.stage}`);
      }
      const predictions = [...select.options].find(option => option.value.toLowerCase() === "predictions.geojson")
        || [...select.options].find(option => option.value.toLowerCase() === "anomalies.geojson");
      if (predictions) select.value = predictions.value;
      else if (select.options.length > 1) select.selectedIndex = 1;
      select.disabled = select.options.length <= 1;
      byId("ppAnalyze").disabled = !select.value;
      const restored = select.value
        ? await restoreLatestWorkflow(resultId, select.value)
        : false;
      if (!restored) {
        setMessage(select.value ? "Click Scan GeoJSON to inspect tile edges. Nothing runs until you start it." : "This result has no GeoJSON files.", select.value ? "" : "warn");
      }
    } catch (error) {
      setMessage(error.message, "err");
    } finally {
      stopIndeterminate();
    }
  }

  function metric(container, value, label) {
    const item = document.createElement("div");
    item.className = "postprocessMetric";
    const strong = document.createElement("strong");
    strong.textContent = value;
    const caption = document.createElement("span");
    caption.textContent = label;
    item.append(strong, caption);
    container.appendChild(item);
  }

  function renderAnalysis(summary) {
    const container = byId("ppSummary");
    container.replaceChildren();
    metric(container, Number(summary.feature_count || 0).toLocaleString(), "GeoJSON features");
    metric(container, Number(summary.features_on_tile_edges || 0).toLocaleString(), "On tile edges");
    metric(container, Number(summary.tile_count || 0).toLocaleString(), "Referenced tiles found");
    const gsd = summary.median_pixel_size_m == null ? "—" : `${Number(summary.median_pixel_size_m).toFixed(4)} m`;
    metric(container, gsd, "Derived pixel size");
    metric(container, Number(summary.invalid_feature_count || 0).toLocaleString(), "Invalid features");
    container.hidden = false;
  }

  async function analyze() {
    const resultId = byId("ppResult").value;
    const inputPath = byId("ppGeojson").value;
    if (!resultId || !inputPath) return;
    resetAnalysis();
    byId("ppAnalyze").disabled = true;
    startIndeterminate("Scanning polygons and tile metadata…");
    setMessage("Scanning polygons and saved tile metadata…");
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input_path: inputPath, edge_tolerance_px: numberValue("ppEdgeTolerance") }),
      });
      state.analysis = payload.summary;
      state.scanComplete = true;
      renderAnalysis(payload.summary);
      byId("ppCombineStep").hidden = false;
      const source = state.geojsonFiles.find(file => file.path === inputPath);
      if (source?.url) void loadPreviewLayer("source", source.url, payload.summary.feature_count);
      const restored = await restoreLatestWorkflow(resultId, inputPath);
      if (!payload.summary.tile_metadata_available) {
        setMessage("The GeoJSON can be read, but its referenced result tiles were not found. Grid-aware combining cannot run.", "warn");
        return;
      }
      if (!payload.summary.features_on_tile_edges) {
        setMessage("No polygons touch a tile edge at this tolerance. There are no tile fragments to combine.", "warn");
        return;
      }
      byId("ppCombine").disabled = false;
      setMessage(
        restored
          ? "Scan complete. A saved combined output is available; you can combine again or continue to regularization."
          : "Scan complete. Review the settings, then start fragment combining.",
        "ok",
      );
    } catch (error) {
      setMessage(error.message, "err");
    } finally {
      stopIndeterminate();
      byId("ppAnalyze").disabled = !byId("ppGeojson").value;
    }
  }

  function showProgress(progress, message) {
    const value = Math.max(0, Math.min(100, Number(progress) || 0));
    byId("ppProgressWrap").hidden = false;
    byId("ppProgressWrap").classList.remove("ppIndeterminate");
    byId("ppProgressBar").style.width = `${value}%`;
    byId("ppProgressText").textContent = `${message || "Processing…"} ${Math.round(value)}%`;
  }

  function renderOutputs(outputs = {}) {
    const container = byId("ppOutputs");
    container.replaceChildren();
    const entries = Object.entries(outputs).filter(([, output]) => output?.url);
    if (!entries.length) {
      container.hidden = true;
      return;
    }
    const title = document.createElement("strong");
    title.textContent = "Outputs:";
    container.appendChild(title);
    for (const [stage, output] of entries) {
      const link = document.createElement("a");
      link.href = output.url;
      link.target = "_blank";
      link.rel = "noopener";
      link.download = "";
      link.textContent = stage === "regularized" ? "Regularized GeoJSON" : "Combined GeoJSON";
      container.appendChild(link);
    }
    container.hidden = false;
  }

  function applyWorkflow(status) {
    showProgress(status.progress, status.message);
    if (Array.isArray(status.log)) {
      byId("ppLog").textContent = status.log.join("\n");
      byId("ppLogWrap").hidden = !status.log.length;
    }
    renderOutputs(status.outputs);
    syncOutputPreviews(status);
    const running = status.status === "queued" || status.status === "running";
    byId("ppCombine").disabled = running || !state.analysis;
    byId("ppRegularize").disabled = running || !status.outputs?.combined;
    if (state.scanComplete && status.outputs?.combined) {
      byId("ppRegularizeStep").hidden = false;
    }
    if (status.status === "failed") setMessage(status.error || status.message || "Post-processing failed.", "err");
    else if (status.status === "complete") setMessage(status.message || "Post-processing complete.", "ok");
    else setMessage(status.message || "Post-processing…");
  }

  async function pollWorkflow(token) {
    const resultId = byId("ppResult").value;
    let consecutiveErrors = 0;
    while (token === state.pollToken && state.workflowId) {
      await new Promise(resolve => setTimeout(resolve, 900));
      if (token !== state.pollToken) return;
      try {
        const status = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/${encodeURIComponent(state.workflowId)}`);
        consecutiveErrors = 0;
        applyWorkflow(status);
        if (status.status !== "queued" && status.status !== "running") {
          await restoreLatestWorkflow(resultId, byId("ppGeojson").value, state.workflowId);
          return;
        }
      } catch (error) {
        consecutiveErrors += 1;
        if (consecutiveErrors < 5) {
          setMessage("Workflow is still running; waiting for the next status update…", "warn");
          continue;
        }
        setMessage(error.message, "err");
        return;
      }
    }
  }

  async function combine() {
    const resultId = byId("ppResult").value;
    const inputPath = byId("ppGeojson").value;
    byId("ppCombine").disabled = true;
    byId("ppRegularize").disabled = true;
    showProgress(0, "Starting fragment combining…");
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/combine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_path: inputPath,
          output_name: byId("ppOutputName").value.trim() || "solar_panels",
          edge_tolerance_px: numberValue("ppEdgeTolerance"),
          gap_tolerance_px: numberValue("ppGapTolerance"),
          min_boundary_overlap: numberValue("ppBoundaryOverlap"),
          remove_contained_polygons: byId("ppRemoveContained").checked,
        }),
      });
      state.workflowId = payload.id;
      state.pollToken += 1;
      applyWorkflow(payload);
      await pollWorkflow(state.pollToken);
    } catch (error) {
      byId("ppCombine").disabled = false;
      setMessage(error.message, "err");
    }
  }

  async function regularize() {
    const selectedWorkflow = byId("ppRegularizeSource").value;
    if (!selectedWorkflow) {
      setMessage("Select a combined output to regularize.", "warn");
      return;
    }
    state.workflowId = selectedWorkflow;
    const resultId = byId("ppResult").value;
    byId("ppRegularize").disabled = true;
    showProgress(0, "Starting polygon regularization…");
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/${encodeURIComponent(state.workflowId)}/regularize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_area_change_percent: numberValue("ppAreaChange") }),
      });
      state.pollToken += 1;
      applyWorkflow(payload);
      await pollWorkflow(state.pollToken);
    } catch (error) {
      byId("ppRegularize").disabled = false;
      setMessage(error.message, "err");
    }
  }

  function init() {
    if (state.initialized) return;
    state.initialized = true;
    // Keep activation self-contained. This also works if an older cached copy of
    // the shared tab controller is still present in the browser.
    byId("btnPostprocess")?.addEventListener("click", () => loadResults(false));
    byId("ppRefresh").addEventListener("click", () => loadResults(true));
    byId("ppResult").addEventListener("change", loadGeojsons);
    byId("ppGeojson").addEventListener("change", async () => {
      resetAnalysis();
      byId("ppAnalyze").disabled = !byId("ppGeojson").value;
      setMessage(byId("ppGeojson").value ? "Click Scan GeoJSON to inspect tile edges." : "Select a GeoJSON.");
      await restoreLatestWorkflow(byId("ppResult").value, byId("ppGeojson").value);
    });
    byId("ppAnalyze").addEventListener("click", analyze);
    byId("ppCombine").addEventListener("click", combine);
    byId("ppRegularize").addEventListener("click", regularize);
    byId("ppRegularizeSource").addEventListener("change", () => {
      const workflow = state.workflows.find(item => item.id === byId("ppRegularizeSource").value);
      if (!workflow) {
        byId("ppRegularize").disabled = true;
        return;
      }
      state.workflowId = workflow.id;
      applyWorkflow(workflow);
      renderWorkflowList();
    });
    byId("ppFitLayers").addEventListener("click", fitVisibleLayers);
    byId("ppRefreshOutputs").addEventListener("click", () =>
      restoreLatestWorkflow(byId("ppResult").value, byId("ppGeojson").value, state.workflowId)
    );
  }

  async function activate() {
    init();
    ensurePreviewMap();
    setTimeout(() => state.map?.invalidateSize(), 30);
    await loadResults(false);
  }

  window.PostProcessTab = { init, activate };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
