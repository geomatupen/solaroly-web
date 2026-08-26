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
    editing: null,
    loadingPromise: null,
    referenceLayers: new Map(),
    referenceToken: 0,
    temporarySequence: 0,
    previewHoverReset: null,
    mode: "segmentation",
  };

  const GENERATED_STAGES = new Set([
    "combined", "regularized", "panel_rows", "identified_panels",
    "deduplicated", "associated", "edited",
  ]);

  const PREVIEW_STYLES = {
    source: { label: "Source", color: "#38bdf8", weight: 1, fillOpacity: 0.08 },
    combined: { label: "Combined", color: "#f59e0b", weight: 2, fillOpacity: 0.13 },
    regularized: { label: "Regularized", color: "#22c55e", weight: 2, fillOpacity: 0.16 },
    panel_rows: { label: "Panel rows", color: "#ef4444", weight: 3, fillOpacity: 0.05 },
    identified_panels: { label: "Identified panels", color: "#14b8a6", weight: 2, fillOpacity: 0.14 },
    panel_reference: { label: "Panel reference", color: "#14b8a6", weight: 2, fillOpacity: 0.08 },
    deduplicated: { label: "Deduplicated anomalies", color: "#f97316", weight: 2, fillOpacity: 0.22 },
    associated: { label: "Associated anomalies", color: "#eab308", weight: 2, fillOpacity: 0.25 },
    edited: { label: "Edited", color: "#a855f7", weight: 2, fillOpacity: 0.18 },
  };

  const byId = id => document.getElementById(id);

  function showListLoading(id, message) {
    const container = byId(id);
    container.replaceChildren();
    const row = document.createElement("div");
    row.className = "mapListLoading";
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    spinner.setAttribute("aria-hidden", "true");
    const text = document.createElement("span");
    text.textContent = message;
    row.append(spinner, text);
    container.appendChild(row);
  }

  function updateFitButton() {
    const hasVisibleReference = state.map && [...state.referenceLayers.values()]
      .some(item => item.layer && state.map.hasLayer(item.layer));
    byId("ppFitLayers").disabled = !state.previewLayers.size && !hasVisibleReference;
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, character => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[character]);
  }

  function polygonPopupHtml(feature, label) {
    const properties = Object.entries(feature?.properties || {}).slice(0, 20);
    const rows = properties.map(([key, value]) =>
      `<tr><th>${escapeHtml(key)}</th><td>${escapeHtml(typeof value === "object" ? JSON.stringify(value) : value)}</td></tr>`
    ).join("");
    return `<div class="mini"><strong>${escapeHtml(label)} polygon</strong>${rows ? `<table class="propTable"><tbody>${rows}</tbody></table>` : ""}</div>`;
  }

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
    byId("ppHierarchyStep").hidden = true;
    byId("ppProgressWrap").hidden = true;
    byId("ppLogWrap").hidden = true;
    byId("ppLog").textContent = "";
    clearPreviewLayers();
  }

  function ensurePreviewMap() {
    if (state.map || !window.L || !byId("ppMap")) return state.map;
    state.map = window.L.map("ppMap", { preferCanvas: true, zoomControl: true }).setView([0, 0], 2);
    const rasterPane = state.map.createPane("ppRasterPane");
    rasterPane.style.zIndex = "250";
    const referencePane = state.map.createPane("ppReferencePane");
    referencePane.style.zIndex = "350";
    window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 22,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(state.map);
    return state.map;
  }

  function clearPreviewLayers() {
    stopEditing(false);
    if (state.map) {
      for (const item of state.previewLayers.values()) {
        if (state.map.hasLayer(item.layer)) state.map.removeLayer(item.layer);
      }
    }
    state.previewLayers.clear();
    state.previewLoading.clear();
    updateFitButton();
    byId("ppLayerList").innerHTML = '<div class="muted tiny">No preview layers loaded.</div>';
    byId("ppMapStatus").textContent = "Scan a GeoJSON to preview it.";
  }

  function clearReferenceLayers() {
    state.referenceToken += 1;
    if (state.map) {
      for (const item of state.referenceLayers.values()) {
        if (item.layer && state.map.hasLayer(item.layer)) state.map.removeLayer(item.layer);
      }
    }
    state.referenceLayers.clear();
    byId("ppReferenceLayers").innerHTML = '<div class="muted tiny">No reference layers loaded.</div>';
    updateFitButton();
  }

  function setReferenceOpacity(item, opacity) {
    item.opacity = opacity;
    item.layer?.eachLayer?.(layer => {
      if (layer.setOpacity && !layer.setStyle) layer.setOpacity(opacity);
      else if (layer.setStyle && layer.options?.pane === "ppReferencePane") {
        layer.setStyle({ opacity: Math.max(opacity, 0.25), fillOpacity: 0.04 });
      }
    });
  }

  function renderReferenceLayers() {
    const container = byId("ppReferenceLayers");
    container.replaceChildren();
    if (!state.referenceLayers.size) {
      container.innerHTML = '<div class="muted tiny">No linked or temporary reference layers.</div>';
      return;
    }
    for (const [id, item] of state.referenceLayers) {
      const row = document.createElement("div");
      row.className = "postprocessLayerItem";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = Boolean(item.layer && state.map?.hasLayer(item.layer));
      checkbox.disabled = Boolean(item.loading);
      checkbox.title = `Show ${item.label}`;
      checkbox.addEventListener("change", async () => {
        try {
          if (checkbox.checked && item.loader && !item.loaded) {
            checkbox.disabled = true;
            item.loading = true;
            byId("ppReferenceStatus").textContent = `Loading ${item.label}…`;
            await item.loader(item);
            if (state.referenceLayers.get(id) !== item) return;
            item.loaded = true;
            item.loading = false;
          }
          if (checkbox.checked) item.layer.addTo(state.map);
          else if (state.map.hasLayer(item.layer)) state.map.removeLayer(item.layer);
          byId("ppReferenceStatus").textContent = checkbox.checked
            ? `${item.label} is visible as a read-only reference.`
            : `${item.label} is hidden.`;
        } catch (error) {
          checkbox.checked = false;
          item.loading = false;
          byId("ppReferenceStatus").textContent = `Could not load ${item.label}: ${error.message}`;
        } finally {
          checkbox.disabled = false;
          renderReferenceLayers();
        }
      });
      const swatch = document.createElement("span");
      swatch.className = "postprocessLayerSwatch";
      swatch.style.color = item.color;
      const text = document.createElement("div");
      text.className = "postprocessLayerText";
      const name = document.createElement("strong");
      name.textContent = item.label;
      const detail = document.createElement("small");
      detail.textContent = item.detail;
      text.append(name, detail);
      const actions = document.createElement("div");
      actions.className = "postprocessLayerActions";
      const opacity = document.createElement("input");
      opacity.type = "range";
      opacity.min = "0.1";
      opacity.max = "1";
      opacity.step = "0.05";
      opacity.value = String(item.opacity ?? 0.75);
      opacity.className = "postprocessReferenceOpacity";
      opacity.title = "Reference opacity";
      opacity.addEventListener("input", () => setReferenceOpacity(item, Number(opacity.value)));
      actions.appendChild(opacity);
      const focus = document.createElement("button");
      focus.type = "button";
      focus.className = "secondary tiny";
      focus.textContent = "Focus";
      focus.disabled = !item.loaded;
      focus.addEventListener("click", () => {
        const bounds = item.bounds || item.layer?.getBounds?.();
        if (bounds?.isValid()) state.map.fitBounds(bounds, { padding: [18, 18], maxZoom: 21 });
      });
      actions.appendChild(focus);
      if (item.temporary) {
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "secondary tiny";
        remove.textContent = "Remove";
        remove.addEventListener("click", () => {
          if (state.map.hasLayer(item.layer)) state.map.removeLayer(item.layer);
          state.referenceLayers.delete(id);
          renderReferenceLayers();
          byId("ppReferenceStatus").textContent = "Temporary layer removed. No project file was changed.";
        });
        actions.appendChild(remove);
      }
      row.append(checkbox, swatch, text, actions);
      container.appendChild(row);
    }
    updateFitButton();
  }

  function imageBounds(feature) {
    const properties = feature?.properties || {};
    const coordinates = feature?.geometry?.coordinates || [];
    const longitude = Number(coordinates[0]);
    const latitude = Number(coordinates[1]);
    const corners = properties.corners;
    if (Array.isArray(corners) && corners.length >= 4) {
      const points = corners.map(point => window.L.latLng(Number(point[1]), Number(point[0])));
      const bounds = window.L.latLngBounds(points);
      if (bounds.isValid()) return bounds;
    }
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
    const halfWidth = Number(properties.width_m || 50) / 2;
    const halfHeight = Number(properties.height_m || 50) / 2;
    const latitudeDelta = halfHeight / 111320;
    const longitudeDelta = halfWidth / (111320 * Math.max(Math.cos(latitude * Math.PI / 180), 0.01));
    return window.L.latLngBounds(
      [latitude - latitudeDelta, longitude - longitudeDelta],
      [latitude + latitudeDelta, longitude + longitudeDelta],
    );
  }

  function imageOverlayUrl(properties, overlayByName) {
    const direct = properties?.overlay;
    if (typeof direct === "string" && (direct.startsWith("/") || /^https?:/i.test(direct))) return direct;
    const source = properties?.src || properties?.image || properties?.file || properties?.name;
    const basename = String(source || "").split(/[\\/]/).pop();
    if (!basename) return null;
    return overlayByName.get(`${basename.replace(/\.[^.]+$/, "")}.png`) || null;
  }

  async function loadLinkedImages(item, summary, imagesUrl) {
    const geojson = await requestJson(imagesUrl, { cache: "no-store" });
    const overlayByName = new Map();
    for (const url of summary?.assets?.overlays || []) {
      const basename = decodeURIComponent(String(url).split("?")[0].split("/").pop() || "");
      if (basename) overlayByName.set(basename, url);
    }
    let count = 0;
    for (const feature of geojson.features || []) {
      if (feature?.geometry?.type !== "Point") continue;
      const bounds = imageBounds(feature);
      if (!bounds) continue;
      const properties = feature.properties || {};
      const imageUrl = imageOverlayUrl(properties, overlayByName);
      const footprint = window.L.rectangle(bounds, {
        pane: "ppReferencePane",
        color: "#38bdf8",
        weight: 1,
        opacity: 0.75,
        fillOpacity: 0.04,
        pmIgnore: true,
      });
      footprint.on("mouseover", event => event.target.setStyle({ color: "#fff", weight: 2, fillOpacity: 0.12 }));
      footprint.on("mouseout", event => event.target.setStyle({ color: "#38bdf8", weight: 1, opacity: Math.max(item.opacity, 0.25), fillOpacity: 0.04 }));
      footprint.on("click", () => {
        if (!imageUrl) {
          byId("ppReferenceStatus").textContent = "This image footprint has no linked overlay image.";
          return;
        }
        let image = item.imageLayers.get(imageUrl);
        if (!image) {
          image = window.L.imageOverlay(imageUrl, bounds, { pane: "ppRasterPane", opacity: item.opacity, interactive: false });
          item.imageLayers.set(imageUrl, image);
        }
        if (item.layer.hasLayer(image)) {
          item.layer.removeLayer(image);
          byId("ppReferenceStatus").textContent = "Linked image hidden.";
        } else {
          item.layer.addLayer(image);
          byId("ppReferenceStatus").textContent = "Linked image loaded. Click its footprint again to hide it.";
        }
      });
      item.layer.addLayer(footprint);
      count += 1;
    }
    item.detail = `${count.toLocaleString()} footprints · click a footprint to load its image`;
    item.bounds = item.layer.getBounds?.();
  }

  async function loadReferenceSources(resultId) {
    const token = ++state.referenceToken;
    showListLoading("ppReferenceLayers", "Loading reference layers…");
    byId("ppReferenceStatus").textContent = "Finding linked orthophoto and image references…";
    const [summaryResult, tilesResult] = await Promise.allSettled([
      requestJson(`/api/session_summary?session=${encodeURIComponent(resultId)}`, { cache: "no-store" }),
      requestJson(`/api/session_tiles?session=${encodeURIComponent(resultId)}`, { cache: "no-store" }),
    ]);
    if (token !== state.referenceToken) return;
    const summary = summaryResult.status === "fulfilled" ? summaryResult.value : {};
    const tiles = tilesResult.status === "fulfilled" ? tilesResult.value : {};
    if (Array.isArray(tiles.layers) && tiles.layers.length) {
      const group = window.L.layerGroup();
      let combinedBounds = null;
      for (const definition of tiles.layers) {
        const bounds = Array.isArray(definition.bounds) ? window.L.latLngBounds(definition.bounds) : undefined;
        if (bounds?.isValid()) combinedBounds = combinedBounds ? combinedBounds.extend(bounds) : bounds;
        window.L.tileLayer(definition.template, {
          pane: "ppRasterPane",
          bounds,
          minZoom: definition.minzoom ?? 0,
          maxZoom: definition.maxzoom ?? 22,
          tileSize: 256,
          noWrap: true,
          opacity: 0.8,
        }).addTo(group);
      }
      state.referenceLayers.set("orthophoto", {
        label: "Orthophoto",
        detail: `${tiles.layers.length.toLocaleString()} linked tile layer${tiles.layers.length === 1 ? "" : "s"} · read-only`,
        color: "#f8fafc",
        layer: group,
        bounds: combinedBounds,
        loaded: true,
        opacity: 0.8,
      });
    }
    const imagesUrl = summary.images_geojson_url || summary.images_geojson || summary.images || summary.images_gj;
    if (imagesUrl) {
      const item = {
        label: "Individual images",
        detail: "Linked footprints · images load only when clicked",
        color: "#38bdf8",
        layer: window.L.featureGroup(),
        loaded: false,
        loading: false,
        opacity: 0.75,
        imageLayers: new Map(),
      };
      item.loader = () => loadLinkedImages(item, summary, imagesUrl);
      state.referenceLayers.set("images", item);
    }
    renderReferenceLayers();
    byId("ppReferenceStatus").textContent = state.referenceLayers.size
      ? "Reference sources found. Enable only the imagery needed for editing."
      : "No linked orthophoto or geolocated image references were found.";
  }

  async function addTemporaryGeoJson(file) {
    const payload = JSON.parse(await file.text());
    if (payload?.type !== "FeatureCollection" || !Array.isArray(payload.features)) {
      throw new Error("Temporary GeoJSON must be a FeatureCollection.");
    }
    const id = `temporary-${++state.temporarySequence}`;
    const layer = window.L.geoJSON(payload, {
      pane: "ppReferencePane",
      pmIgnore: true,
      style: { color: "#e879f9", weight: 2, dashArray: "6 4", fillColor: "#e879f9", fillOpacity: 0.08 },
      pointToLayer: (_feature, latlng) => window.L.circleMarker(latlng, { pane: "ppReferencePane", radius: 5, color: "#e879f9" }),
      onEachFeature: (feature, featureLayer) => {
        featureLayer.options.pmIgnore = true;
        featureLayer.on("click", event => {
          window.L.popup({ maxWidth: 320, maxHeight: 210 })
            .setLatLng(event.latlng)
            .setContent(polygonPopupHtml(feature, file.name))
            .openOn(state.map);
        });
      },
    });
    state.referenceLayers.set(id, {
      label: file.name,
      detail: `${Number(payload.features.length).toLocaleString()} features · temporary · read-only`,
      color: "#e879f9",
      layer,
      loaded: true,
      temporary: true,
      opacity: 0.8,
    });
    layer.addTo(state.map);
    renderReferenceLayers();
    const bounds = layer.getBounds?.();
    if (bounds?.isValid()) state.map.fitBounds(bounds, { padding: [18, 18], maxZoom: 21 });
    byId("ppReferenceStatus").textContent = "Temporary GeoJSON added in browser memory only.";
  }

  function createPreviewGeoJsonLayer(stage, geojson, label = null) {
    const style = { ...(PREVIEW_STYLES[stage] || PREVIEW_STYLES.source) };
    if (label) style.label = label;
    const editableStage = GENERATED_STAGES.has(stage);
    const baseStyle = {
      color: style.color,
      weight: style.weight,
      fillColor: style.color,
      fillOpacity: style.fillOpacity,
      opacity: 1,
    };
    const layer = window.L.geoJSON(geojson, {
      renderer: window.L.svg({ padding: 0.5 }),
      pmIgnore: !editableStage,
      style: () => baseStyle,
      onEachFeature: (feature, polygonLayer) => {
        polygonLayer.options.pmIgnore = !editableStage;
        let hovered = false;
        const resetHover = () => {
          if (!hovered) return;
          hovered = false;
          if (state.previewHoverReset === resetHover) state.previewHoverReset = null;
          if (state.editing?.selectedLayer === polygonLayer) return;
          if (!state.editing) polygonLayer.setStyle(baseStyle);
          else if (state.editing.stage === stage) {
            polygonLayer.setStyle({ ...baseStyle, weight: baseStyle.weight + 1, fillOpacity: 0.28, opacity: 1 });
          }
        };
        polygonLayer.on("mouseover", event => {
          if (state.editing && state.editing.stage !== stage) return;
          if (state.editing?.selectedLayer === event.target) return;
          if (state.previewHoverReset && state.previewHoverReset !== resetHover) state.previewHoverReset();
          hovered = true;
          event.target.setStyle({
            color: "#ffffff",
            weight: style.weight + 2,
            fillColor: style.color,
            fillOpacity: 0.38,
          });
          state.previewHoverReset = resetHover;
        });
        polygonLayer.on("mouseout", resetHover);
        polygonLayer.on("remove", resetHover);
        polygonLayer.on("click", event => {
          resetHover();
          if (state.editing) return;
          window.L.popup({
            className: "postprocessPolygonPopup",
            maxWidth: 340,
            maxHeight: 210,
          })
            .setLatLng(event.latlng)
            .setContent(polygonPopupHtml(feature, style.label))
            .openOn(state.map);
        });
      },
    });
    return { layer, baseStyle, label: style.label };
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
    for (const item of state.referenceLayers.values()) {
      if (item.layer && map.hasLayer(item.layer)) {
        const layerBounds = item.bounds || item.layer.getBounds?.();
        if (layerBounds?.isValid()) bounds.push(layerBounds);
      }
    }
    if (bounds.length) {
      const combined = bounds.slice(1).reduce((value, next) => value.extend(next), bounds[0]);
      map.fitBounds(combined, { padding: [18, 18], maxZoom: 21 });
    }
  }

  function setPreviewFullscreen(active) {
    const preview = document.querySelector(".postprocessPreview");
    if (!preview) return;
    preview.classList.toggle("postprocessFullscreen", active);
    document.body.classList.toggle("postprocessFullscreenOpen", active);
    const button = byId("ppFullscreen");
    button.textContent = active ? "Exit fullscreen" : "Fullscreen";
    button.setAttribute("aria-pressed", String(active));
    window.setTimeout(() => state.map?.invalidateSize(), 50);
  }

  async function togglePreviewFullscreen() {
    const preview = document.querySelector(".postprocessPreview");
    if (!preview) return;
    const active = preview.classList.contains("postprocessFullscreen") || document.fullscreenElement === preview;
    if (active) {
      if (document.fullscreenElement === preview && document.exitFullscreen) {
        await document.exitFullscreen().catch(() => {});
      }
      setPreviewFullscreen(false);
      return;
    }
    setPreviewFullscreen(true);
    if (preview.requestFullscreen) {
      await preview.requestFullscreen().catch(() => {
        // The fixed-position fallback remains active when browser fullscreen is unavailable.
      });
    }
  }

  function exitPreviewFullscreen() {
    const preview = document.querySelector(".postprocessPreview");
    if (!preview?.classList.contains("postprocessFullscreen") && document.fullscreenElement !== preview) return;
    if (document.fullscreenElement === preview && document.exitFullscreen) void document.exitFullscreen();
    setPreviewFullscreen(false);
  }

  async function sendLayerToMap(stage, button) {
    if (!state.workflowId || !GENERATED_STAGES.has(stage)) return;
    const original = button.textContent;
    button.disabled = true;
    button.textContent = "Sending…";
    try {
      const resultId = byId("ppResult").value;
      const payload = await requestJson(
        `/api/results/${encodeURIComponent(resultId)}/postprocess/${encodeURIComponent(state.workflowId)}/${encodeURIComponent(stage)}/share`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        },
      );
      if (typeof window.showSharedOverlayOnMap !== "function") {
        throw new Error("Map integration is not available.");
      }
      await window.showSharedOverlayOnMap(payload.overlay);
    } catch (error) {
      setMessage(error.message, "err");
    } finally {
      button.disabled = false;
      button.textContent = original;
    }
  }

  function updateEditHistoryControls(message = "") {
    const editing = state.editing;
    if (!editing) return;
    editing.dirty = editing.historyIndex > 0;
    byId("ppUndoEdits").disabled = editing.historyIndex <= 0;
    byId("ppRedoEdits").disabled = editing.historyIndex >= editing.history.length - 1;
    byId("ppSaveEdits").disabled = !editing.dirty;
    if (message) byId("ppEditStatus").textContent = message;
  }

  function recordEditState(message = "Change recorded. You can undo or redo it.") {
    const editing = state.editing;
    if (!editing) return;
    if (typeof message !== "string") message = "Vertex edit recorded. You can undo or redo it.";
    if (editing.historyTimer) {
      window.clearTimeout(editing.historyTimer);
      editing.historyTimer = null;
    }
    const snapshot = JSON.parse(JSON.stringify(editing.item.layer.toGeoJSON()));
    if (JSON.stringify(snapshot) === JSON.stringify(editing.history[editing.historyIndex])) return;
    editing.history.splice(editing.historyIndex + 1);
    editing.history.push(snapshot);
    if (editing.history.length > 12) editing.history.splice(1, editing.history.length - 12);
    editing.historyIndex = editing.history.length - 1;
    updateEditHistoryControls(message);
  }

  function scheduleEditHistory() {
    if (!state.editing) return;
    if (state.editing.historyTimer) window.clearTimeout(state.editing.historyTimer);
    state.editing.historyTimer = window.setTimeout(() => {
      if (!state.editing) return;
      state.editing.historyTimer = null;
      recordEditState("Vertex edit recorded. You can undo or redo it.");
    }, 120);
  }

  function lockProcessingControls(locked) {
    byId("ppResult").disabled = locked;
    byId("ppGeojson").disabled = locked || !byId("ppResult").value;
    byId("ppAnalyze").disabled = locked || !byId("ppGeojson").value;
    byId("ppRefresh").disabled = locked;
    byId("ppSegmentationTab").disabled = locked;
    byId("ppAnomalyTab").disabled = locked;
    byId("ppAnomalyGeojson").disabled = locked || !byId("ppResult").value;
    byId("ppRegularizeSource").disabled = locked || !state.workflows.some(item => item.outputs?.combined);
    if (locked) {
      byId("ppCombine").disabled = true;
      byId("ppRegularize").disabled = true;
    } else {
      byId("ppCombine").disabled = !state.analysis;
      const workflow = state.workflows.find(item => item.id === state.workflowId);
      byId("ppRegularize").disabled = !workflow?.outputs?.combined;
    }
  }

  function disableEditingTools() {
    const editing = state.editing;
    if (!editing) return;
    if (editing.historyTimer) {
      window.clearTimeout(editing.historyTimer);
      editing.historyTimer = null;
      recordEditState("Vertex edit recorded. You can undo or redo it.");
    }
    for (const entry of editing.deleteHandlers || []) {
      entry.layer.off("click", entry.handler);
    }
    editing.deleteHandlers = [];
    editing.mode = null;
    editing.selectedLayer = null;
    editing.item.layer.eachLayer(layer => {
      try { layer.pm?.disable(); } catch (_) {}
      try { layer.pm?.disableLayerDrag?.(); } catch (_) {}
      try { layer.pm?.disableRotate?.(); } catch (_) {}
      layer.off("pm:edit", scheduleEditHistory);
      layer.off("pm:dragend", recordEditState);
      layer.off("pm:rotateend", recordEditState);
      layer.off("pm:markerdragend", recordEditState);
      layer.off("pm:vertexadded", scheduleEditHistory);
      layer.off("pm:vertexremoved", scheduleEditHistory);
    });
    byId("ppMap").classList.remove("deleteMode");
    byId("ppMap").classList.remove("vertexMode");
    byId("ppMap").classList.remove("moveMode");
    byId("ppMap").classList.remove("rotateMode");
    byId("ppEditVertices").classList.remove("active");
    byId("ppMovePolygons").classList.remove("active");
    byId("ppRotatePolygons").classList.remove("active");
    byId("ppDeletePolygons").classList.remove("active");
    applyEditingEmphasis();
  }

  function applyEditingEmphasis() {
    for (const [stage, item] of state.previewLayers) {
      const selected = state.editing?.stage === stage;
      item.layer.eachLayer(layer => {
        if (!layer.setStyle) return;
        layer.setStyle(selected
          ? { ...item.baseStyle, weight: item.baseStyle.weight + 1, fillOpacity: 0.28, opacity: 1 }
          : { ...item.baseStyle, fillOpacity: 0.02, opacity: 0.18 });
      });
    }
  }

  function beginEditing(stage) {
    if (!GENERATED_STAGES.has(stage)) return;
    const item = state.previewLayers.get(stage);
    if (!item || !state.workflowId) return;
    if (state.editing?.dirty && !window.confirm("Discard the unsaved edits on the current layer?")) return;
    stopEditing(false);
    const visibleStages = new Set(
      [...state.previewLayers].filter(([, candidate]) => state.map.hasLayer(candidate.layer)).map(([key]) => key)
    );
    if (!state.map.hasLayer(item.layer)) item.layer.addTo(state.map);
    for (const [key, candidate] of state.previewLayers) {
      if (key !== stage && state.map.hasLayer(candidate.layer)) state.map.removeLayer(candidate.layer);
    }
    const initialSnapshot = JSON.parse(JSON.stringify(item.layer.toGeoJSON()));
    state.editing = {
      stage,
      item,
      snapshot: initialSnapshot,
      dirty: false,
      deleteHandlers: [],
      mode: null,
      selectedLayer: null,
      visibleStages,
      history: [initialSnapshot],
      historyIndex: 0,
      historyTimer: null,
    };
    lockProcessingControls(true);
    byId("ppEditPanel").hidden = false;
    const workflow = state.workflows.find(candidate => candidate.id === state.workflowId);
    byId("ppEditLayerName").textContent = `${workflow ? workflowDisplayName(workflow) : state.workflowId} · ${item.label}`;
    byId("ppEditLayerIdentity").textContent = `Workflow ID: ${state.workflowId} · Stage: ${stage}`;
    byId("ppEditStatus").textContent = "Only this highlighted layer is editable. Other layers are temporarily hidden and locked.";
    byId("ppUndoEdits").disabled = true;
    byId("ppRedoEdits").disabled = true;
    byId("ppSaveEdits").disabled = true;
    applyEditingEmphasis();
    renderPreviewLayers();
    enableVertexEditing();
  }

  function enableVertexEditing() {
    if (!state.editing) return;
    disableEditingTools();
    state.editing.mode = "vertices";
    let supported = false;
    state.editing.item.layer.eachLayer(layer => {
      layer.options.pmIgnore = false;
      try { window.L.PM?.reInitLayer(layer); } catch (_) {}
      if (layer.pm?.enable) {
        supported = true;
        const handler = event => {
          if (event.originalEvent) window.L.DomEvent.stopPropagation(event.originalEvent);
          state.editing.item.layer.eachLayer(other => {
            if (other !== layer) {
              try { other.pm?.disable(); } catch (_) {}
              other.off("pm:edit", scheduleEditHistory);
              other.off("pm:markerdragend", recordEditState);
              other.off("pm:vertexadded", scheduleEditHistory);
              other.off("pm:vertexremoved", scheduleEditHistory);
            }
          });
          applyEditingEmphasis();
          state.editing.selectedLayer = layer;
          layer.setStyle?.({ color: "#ffffff", weight: state.editing.item.baseStyle.weight + 2, fillOpacity: 0.42 });
          layer.on("pm:edit", scheduleEditHistory);
          layer.on("pm:markerdragend", recordEditState);
          layer.on("pm:vertexadded", scheduleEditHistory);
          layer.on("pm:vertexremoved", scheduleEditHistory);
          layer.pm.enable({ allowSelfIntersection: false, snappable: true });
          const markerCount = layer.pm?._markers?.length || 0;
          byId("ppEditStatus").textContent = markerCount
            ? `Polygon selected · ${markerCount} vertex handles. Drag a handle to edit.`
            : "Polygon selected, but vertex handles could not be created.";
        };
        layer.on("click", handler);
        state.editing.deleteHandlers.push({ layer, handler });
      }
    });
    byId("ppMap").classList.add("vertexMode");
    byId("ppEditVertices").classList.add("active");
    byId("ppEditStatus").textContent = supported
      ? "Vertex mode: click one polygon in the highlighted layer, then drag its vertices."
      : "Vertex editing support did not load; refresh the page and try again.";
  }

  function enablePolygonDeletion() {
    if (!state.editing) return;
    disableEditingTools();
    state.editing.mode = "delete";
    const group = state.editing.item.layer;
    group.eachLayer(layer => {
      const handler = event => {
        if (event.originalEvent) window.L.DomEvent.stopPropagation(event.originalEvent);
        group.removeLayer(layer);
        state.editing.item.count = group.getLayers().length;
        recordEditState("Polygon deleted. You can undo or redo this change.");
        renderPreviewLayers();
      };
      layer.on("click", handler);
      state.editing.deleteHandlers.push({ layer, handler });
    });
    byId("ppMap").classList.add("deleteMode");
    byId("ppDeletePolygons").classList.add("active");
    byId("ppEditStatus").textContent = "Delete mode: click polygons only in the highlighted layer.";
  }

  function enablePolygonMovement() {
    if (!state.editing) return;
    disableEditingTools();
    state.editing.mode = "move";
    let supported = false;
    state.editing.item.layer.eachLayer(layer => {
      layer.options.pmIgnore = false;
      try { window.L.PM?.reInitLayer(layer); } catch (_) {}
      if (!layer.pm?.enableLayerDrag) return;
      supported = true;
      const handler = event => {
        if (event.originalEvent) window.L.DomEvent.stopPropagation(event.originalEvent);
        state.editing.item.layer.eachLayer(other => {
          if (other === layer) return;
          try { other.pm?.disableLayerDrag?.(); } catch (_) {}
          other.off("pm:dragend", recordEditState);
        });
        state.editing.selectedLayer = layer;
        layer.pm.enableLayerDrag();
        layer.on("pm:dragend", recordEditState);
        byId("ppEditStatus").textContent = "Polygon selected. Drag it to a corrected position; the move is recorded when released.";
      };
      layer.on("click", handler);
      state.editing.deleteHandlers.push({ layer, handler });
    });
    byId("ppMap").classList.add("moveMode");
    byId("ppMovePolygons").classList.add("active");
    byId("ppEditStatus").textContent = supported
      ? "Move mode: click one polygon in the highlighted layer, then drag it."
      : "Polygon movement support did not load; refresh the page and try again.";
  }

  function enablePolygonRotation() {
    if (!state.editing) return;
    disableEditingTools();
    state.editing.mode = "rotate";
    let supported = false;
    state.editing.item.layer.eachLayer(layer => {
      layer.options.pmIgnore = false;
      try { window.L.PM?.reInitLayer(layer); } catch (_) {}
      if (!layer.pm?.enableRotate) return;
      supported = true;
      const handler = event => {
        if (event.originalEvent) window.L.DomEvent.stopPropagation(event.originalEvent);
        state.editing.item.layer.eachLayer(other => {
          if (other === layer) return;
          try { other.pm?.disableRotate?.(); } catch (_) {}
          other.off("pm:rotateend", recordEditState);
        });
        state.editing.selectedLayer = layer;
        layer.pm.enableRotate();
        layer.on("pm:rotateend", recordEditState);
        byId("ppEditStatus").textContent = "Polygon selected. Drag a rotation handle; the rotation is recorded when released.";
      };
      layer.on("click", handler);
      state.editing.deleteHandlers.push({ layer, handler });
    });
    byId("ppMap").classList.add("rotateMode");
    byId("ppRotatePolygons").classList.add("active");
    byId("ppEditStatus").textContent = supported
      ? "Rotate mode: click one polygon, then use its rotation handles."
      : "Polygon rotation support did not load; refresh the page and try again.";
  }

  function restoreEditHistory(targetIndex) {
    if (!state.editing) return;
    const editing = state.editing;
    if (targetIndex < 0 || targetIndex >= editing.history.length || targetIndex === editing.historyIndex) return;
    const mode = editing.mode || "vertices";
    disableEditingTools();
    const wasVisible = state.map.hasLayer(editing.item.layer);
    if (wasVisible) state.map.removeLayer(editing.item.layer);
    const snapshot = JSON.parse(JSON.stringify(editing.history[targetIndex]));
    const created = createPreviewGeoJsonLayer(editing.stage, snapshot);
    editing.item.layer = created.layer;
    editing.item.geojson = snapshot;
    editing.item.baseStyle = created.baseStyle;
    editing.item.count = snapshot.features?.length || 0;
    if (wasVisible) editing.item.layer.addTo(state.map);
    editing.historyIndex = targetIndex;
    updateEditHistoryControls(targetIndex === 0 ? "Returned to the original layer." : "Edit history restored.");
    applyEditingEmphasis();
    renderPreviewLayers();
    if (mode === "delete") enablePolygonDeletion();
    else if (mode === "move") enablePolygonMovement();
    else if (mode === "rotate") enablePolygonRotation();
    else enableVertexEditing();
  }

  function undoEdits() {
    if (state.editing) restoreEditHistory(state.editing.historyIndex - 1);
  }

  function redoEdits() {
    if (state.editing) restoreEditHistory(state.editing.historyIndex + 1);
  }

  function stopEditing(confirmDiscard = true) {
    if (!state.editing) return true;
    if (confirmDiscard && state.editing.dirty && !window.confirm("Exit and discard these unsaved polygon edits?")) return false;
    const editing = state.editing;
    disableEditingTools();
    for (const [stage, item] of state.previewLayers) {
      item.layer.eachLayer(layer => {
        layer.options.pmIgnore = !GENERATED_STAGES.has(stage);
        layer.setStyle?.(item.baseStyle);
      });
    }
    state.editing = null;
    for (const [stage, item] of state.previewLayers) {
      const shouldShow = editing.visibleStages.has(stage);
      if (shouldShow && !state.map.hasLayer(item.layer)) item.layer.addTo(state.map);
      else if (!shouldShow && state.map.hasLayer(item.layer)) state.map.removeLayer(item.layer);
    }
    byId("ppEditPanel").hidden = true;
    lockProcessingControls(false);
    renderPreviewLayers();
    return true;
  }

  async function saveEditedRevision() {
    if (!state.editing?.dirty) return;
    disableEditingTools();
    const editing = state.editing;
    const button = byId("ppSaveEdits");
    button.disabled = true;
    button.textContent = "Saving…";
      byId("ppEditStatus").textContent = "Validating and updating the edited GeoJSON…";
    try {
      const resultId = byId("ppResult").value;
      const payload = await requestJson(
        `/api/results/${encodeURIComponent(resultId)}/postprocess/${encodeURIComponent(state.workflowId)}/${encodeURIComponent(editing.stage)}/revisions`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ geojson: editing.item.layer.toGeoJSON() }),
        },
      );
      stopEditing(false);
      applyWorkflow(payload);
      await restoreLatestWorkflow(resultId, byId("ppGeojson").value, state.workflowId);
      setMessage("Edited polygons saved. The edited layer has been updated.", "ok");
    } catch (error) {
      byId("ppEditStatus").textContent = error.message;
      setMessage(error.message, "err");
    } finally {
      button.textContent = "Save edits";
      if (state.editing) button.disabled = !state.editing.dirty;
    }
  }

  function renderPreviewLayers() {
    const container = byId("ppLayerList");
    container.replaceChildren();
    for (const [key, item] of state.previewLayers) {
      const row = document.createElement("div");
      const isEditing = state.editing?.stage === key;
      row.className = `postprocessLayerItem${isEditing ? " editing" : ""}${state.editing && !isEditing ? " locked" : ""}`;
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = state.map?.hasLayer(item.layer) || false;
      checkbox.disabled = Boolean(state.editing);
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
      detail.textContent = `${Number(item.count).toLocaleString()} polygons${key === "source" ? " · Read-only" : ""}`;
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
      const actions = document.createElement("div");
      actions.className = "postprocessLayerActions";
      actions.appendChild(focus);
      if (GENERATED_STAGES.has(key)) {
        const edit = document.createElement("button");
        edit.type = "button";
        edit.className = "secondary tiny";
        edit.textContent = isEditing ? "Editing" : "Edit this layer";
        edit.disabled = Boolean(state.editing);
        edit.addEventListener("click", () => beginEditing(key));
        actions.appendChild(edit);
        const send = document.createElement("button");
        send.type = "button";
        send.className = "secondary tiny";
        send.textContent = "Link to Map";
        send.disabled = Boolean(state.editing);
        send.addEventListener("click", () => sendLayerToMap(key, send));
        actions.appendChild(send);
      }
      actions.appendChild(download);
      row.append(checkbox, swatch, text, actions);
      container.appendChild(row);
    }
    for (const stage of state.previewLoading) {
      if (state.previewLayers.has(stage)) continue;
      const style = PREVIEW_STYLES[stage] || PREVIEW_STYLES.source;
      const loading = document.createElement("div");
      loading.className = "mapListLoading";
      loading.innerHTML = `<span class="spinner" aria-hidden="true"></span><span>Loading ${escapeHtml(style.label)} layer…</span>`;
      container.appendChild(loading);
    }
    if (!container.childElementCount) {
      container.innerHTML = '<div class="muted tiny">No processing layers loaded.</div>';
    }
    updateFitButton();
  }

  async function loadPreviewLayer(stage, url, expectedCount = null, label = null) {
    if (!url || state.previewLayers.get(stage)?.url === url || state.previewLoading.has(stage)) return;
    const map = ensurePreviewMap();
    if (!map) {
      byId("ppMapStatus").textContent = "Map preview is unavailable because Leaflet did not load.";
      return;
    }
    state.previewLoading.add(stage);
    renderPreviewLayers();
    const style = PREVIEW_STYLES[stage] || PREVIEW_STYLES.source;
    byId("ppMapStatus").textContent = `Loading ${style.label.toLowerCase()} polygons…`;
    try {
      const geojson = await requestJson(url, { cache: "no-store" });
      const existing = state.previewLayers.get(stage);
      if (existing && map.hasLayer(existing.layer)) map.removeLayer(existing.layer);
      const created = createPreviewGeoJsonLayer(stage, geojson, label);
      const layer = created.layer;
      // Show the newest stage by itself initially. Users can turn earlier stages
      // back on for comparison using the layer list.
      for (const item of state.previewLayers.values()) {
        if (map.hasLayer(item.layer)) map.removeLayer(item.layer);
      }
      layer.addTo(map);
      state.previewLayers.set(stage, {
        label: created.label,
        color: style.color,
        count: expectedCount ?? geojson.features?.length ?? 0,
        layer,
        url,
        geojson,
        baseStyle: created.baseStyle,
      });
      renderPreviewLayers();
      const bounds = layer.getBounds();
      if (bounds.isValid()) map.fitBounds(bounds, { padding: [18, 18], maxZoom: 21 });
      byId("ppMapStatus").textContent = `${style.label} preview loaded.`;
    } catch (error) {
      byId("ppMapStatus").textContent = `Could not load ${style.label.toLowerCase()} preview: ${error.message}`;
    } finally {
      state.previewLoading.delete(stage);
      renderPreviewLayers();
    }
  }

  function syncOutputPreviews(status) {
    const counts = {
      combined: status.combine_stats?.output_features,
      regularized: status.regularize_stats?.output_features,
      panel_rows: status.hierarchy_stats?.row_count,
      identified_panels: status.hierarchy_stats?.panel_count,
      deduplicated: status.deduplicate_stats?.output_features,
      associated: status.association_stats?.output_features,
    };
    for (const stage of GENERATED_STAGES) {
      if (stage === "edited" || !status.outputs?.[stage]?.url) continue;
      void loadPreviewLayer(stage, status.outputs[stage].url, counts[stage]);
    }
    if (status.outputs?.edited?.url) {
      const latest = (status.manual_revisions || []).at(-1);
      void loadPreviewLayer("edited", status.outputs.edited.url, latest?.feature_count, editedLayerName(status));
    }
  }

  function workflowDisplayName(workflow) {
    return workflow.display_name || workflow.parameters?.output_name || workflow.id;
  }

  function editedLayerName(workflow) {
    const latest = (workflow?.manual_revisions || []).at(-1);
    let sourceStage = String(latest?.source_stage || "").toLowerCase();
    if (!GENERATED_STAGES.has(sourceStage) || sourceStage === "edited") {
      const path = String(workflow?.outputs?.edited?.path || "");
      sourceStage = path.split("/").pop().replace(/_edited\.geojson$/i, "") || "combined";
    }
    return `${sourceStage}_edited`;
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
    const visibleWorkflows = state.workflows.filter(workflow =>
      state.mode === "anomaly" ? workflow.workflow_kind === "anomaly" : workflow.workflow_kind !== "anomaly"
    );
    if (!visibleWorkflows.length) {
      container.innerHTML = '<div class="muted tiny">No saved outputs.</div>';
      populateRegularizeSources();
      return;
    }
    for (const workflow of visibleWorkflows) {
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
      const stages = Object.keys(workflow.outputs || {})
        .map(stage => stage === "edited" ? editedLayerName(workflow) : stage)
        .join(" + ") || workflow.stage || "No output";
      const status = document.createElement("small");
      status.textContent = `${workflow.status || "unknown"} · ${stages}`;
      info.append(name, id, status);
      info.tabIndex = 0;
      info.setAttribute("role", "button");
      const open = () => {
        if (state.editing) return;
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
      dots.disabled = Boolean(state.editing);
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
    if (state.loadingPromise) return state.loadingPromise;
    const select = byId("ppResult");
    const previous = select.value;
    select.replaceChildren();
    addOption(select, "", "Loading test results…");
    select.disabled = true;
    setMessage("Loading test results…");
    state.loadingPromise = (async () => {
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
        select.disabled = select.options.length <= 1;
        state.loaded = true;
        setMessage(select.options.length > 1 ? "Select a result and GeoJSON to begin." : "No test results are available.", select.options.length > 1 ? "" : "warn");
        if (select.value) await loadGeojsons();
      } catch (error) {
        select.replaceChildren();
        addOption(select, "", "Could not load test results");
        select.disabled = true;
        setMessage(error.message, "err");
      }
    })().finally(() => { state.loadingPromise = null; });
    return state.loadingPromise;
  }

  async function restoreLatestWorkflow(resultId, inputPath, preferredId = null) {
    if (!resultId) return false;
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess`, { cache: "no-store" });
      state.workflows = (payload.workflows || []).sort((first, second) =>
        String(second.created_at || "").localeCompare(String(first.created_at || ""))
      );
      const workflow = selectSavedWorkflow(inputPath, preferredId, false);
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

  function selectSavedWorkflow(inputPath = "", preferredId = null, allowLatest = false) {
    const visible = state.workflows.filter(item =>
      state.mode === "anomaly" ? item.workflow_kind === "anomaly" : item.workflow_kind !== "anomaly"
    );
    return visible.find(item => item.id === preferredId)
      || visible.find(item => sourceIdentity(item.input_path) === sourceIdentity(inputPath) && (item.outputs?.combined || item.outputs?.deduplicated))
      || (allowLatest ? visible.find(item => Object.keys(item.outputs || {}).length) : null);
  }

  async function loadSavedOutputsFirst(resultId) {
    showListLoading("ppWorkflowList", "Loading saved outputs…");
    showListLoading("ppLayerList", "Loading output layers…");
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess`, { cache: "no-store" });
      state.workflows = (payload.workflows || []).sort((first, second) =>
        String(second.created_at || "").localeCompare(String(first.created_at || ""))
      );
      const workflow = selectSavedWorkflow("", null, true);
      if (workflow) {
        state.workflowId = workflow.id;
        applyWorkflow(workflow);
        populateRegularizeSources(workflow.id);
      } else {
        renderPreviewLayers();
      }
      renderWorkflowList();
      return workflow;
    } catch (error) {
      state.workflows = [];
      renderWorkflowList();
      renderPreviewLayers();
      byId("ppMapStatus").textContent = `Could not load saved outputs: ${error.message}`;
      return null;
    }
  }

  async function loadGeojsons() {
    resetAnalysis();
    clearReferenceLayers();
    state.workflows = [];
    const resultId = byId("ppResult").value;
    const select = byId("ppGeojson");
    select.disabled = true;
    byId("ppAnalyze").disabled = true;
    select.replaceChildren();
    addOption(select, "", resultId ? "Looking for GeoJSON files…" : "Select a GeoJSON…");
    if (!resultId) {
      setMessage("Select a test result.");
      byId("ppReferenceStatus").textContent = "Select a test result to find linked imagery.";
      return;
    }
    setMessage("Loading saved outputs first…");
    const earlyWorkflow = await loadSavedOutputsFirst(resultId);
    void loadReferenceSources(resultId);
    setMessage("Looking for GeoJSON files in the selected result…");
    startIndeterminate("Loading the available GeoJSON file list…");
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/geojsons`);
      state.geojsonFiles = payload.files || [];
      select.replaceChildren();
      addOption(select, "", "Select a GeoJSON…");
      for (const file of state.geojsonFiles) {
        addOption(select, file.path, `${file.name} · ${file.stage}`);
      }
      const previousSource = earlyWorkflow
        ? [...select.options].find(option => sourceIdentity(option.value) === sourceIdentity(earlyWorkflow.input_path))
        : null;
      const predictions = [...select.options].find(option => option.value.toLowerCase() === "predictions.geojson")
        || [...select.options].find(option => option.value.toLowerCase() === "anomalies.geojson");
      if (previousSource) select.value = previousSource.value;
      else if (predictions) select.value = predictions.value;
      else if (select.options.length > 1) select.selectedIndex = 1;
      select.disabled = select.options.length <= 1;
      byId("ppAnalyze").disabled = !select.value;
      const matchingWorkflow = selectSavedWorkflow(select.value, null, false);
      const restored = Boolean(matchingWorkflow);
      if (matchingWorkflow && matchingWorkflow.id !== state.workflowId) {
        state.workflowId = matchingWorkflow.id;
        applyWorkflow(matchingWorkflow);
        populateRegularizeSources(matchingWorkflow.id);
        renderWorkflowList();
      }
      if (!restored) {
        setMessage(select.value ? "Click Scan GeoJSON to inspect tile edges. Nothing runs until you start it." : "This result has no GeoJSON files.", select.value ? "" : "warn");
      }
      document.dispatchEvent(new CustomEvent("postprocess:data", { detail: getContext() }));
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

  function applyWorkflow(status) {
    showProgress(status.progress, status.message);
    if (Array.isArray(status.log)) {
      byId("ppLog").textContent = status.log.join("\n");
      byId("ppLogWrap").hidden = !status.log.length;
    }
    syncOutputPreviews(status);
    const running = status.status === "queued" || status.status === "running";
    const hasCombined = Boolean(status.outputs?.combined);
    byId("ppCombine").disabled = running || !state.analysis;
    byId("ppRegularize").disabled = running || !hasCombined;
    if (hasCombined) {
      byId("ppCombineStep").hidden = false;
      byId("ppRegularizeStep").hidden = false;
    }
    if (status.status === "failed") setMessage(status.error || status.message || "Post-processing failed.", "err");
    else if (status.status === "complete") setMessage(status.message || "Post-processing complete.", "ok");
    else setMessage(status.message || "Post-processing…");
    document.dispatchEvent(new CustomEvent("postprocess:workflow", { detail: { status, context: getContext() } }));
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
    if (state.editing) return;
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
    if (state.editing) return;
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
    try { window.L?.PM?.setOptIn(true); } catch (_) {}
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
    byId("ppFullscreen").addEventListener("click", togglePreviewFullscreen);
    byId("ppRefreshOutputs").addEventListener("click", () =>
      restoreLatestWorkflow(byId("ppResult").value, byId("ppGeojson").value, state.workflowId)
    );
    byId("ppEditVertices").addEventListener("click", enableVertexEditing);
    byId("ppMovePolygons").addEventListener("click", enablePolygonMovement);
    byId("ppRotatePolygons").addEventListener("click", enablePolygonRotation);
    byId("ppDeletePolygons").addEventListener("click", enablePolygonDeletion);
    byId("ppUndoEdits").addEventListener("click", undoEdits);
    byId("ppRedoEdits").addEventListener("click", redoEdits);
    byId("ppSaveEdits").addEventListener("click", saveEditedRevision);
    byId("ppExitEditing").addEventListener("click", () => stopEditing(true));
    byId("ppAddTemporaryLayer").addEventListener("click", () => byId("ppTemporaryLayerFile").click());
    byId("ppTemporaryLayerFile").addEventListener("change", async event => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (!file) return;
      try {
        ensurePreviewMap();
        await addTemporaryGeoJson(file);
      } catch (error) {
        byId("ppReferenceStatus").textContent = `Could not add temporary layer: ${error.message}`;
      }
    });
    document.querySelector(".tabs")?.addEventListener("click", event => {
      const navigation = event.target.closest("button[data-tab], a[href]");
      if (!navigation || navigation.id === "btnPostprocess") return;
      exitPreviewFullscreen();
      if (!state.editing) return;
      if (state.editing.dirty && !window.confirm("Leave Post-process and discard the unsaved polygon edits?")) {
        event.preventDefault();
        event.stopImmediatePropagation();
        return;
      }
      stopEditing(false);
    }, true);
    window.addEventListener("beforeunload", event => {
      if (!state.editing?.dirty) return;
      event.preventDefault();
      event.returnValue = "";
    });
    document.addEventListener("fullscreenchange", () => {
      const preview = document.querySelector(".postprocessPreview");
      setPreviewFullscreen(document.fullscreenElement === preview);
    });
    document.addEventListener("keydown", event => {
      if (event.key === "Escape" && !document.fullscreenElement) exitPreviewFullscreen();
    });
    document.addEventListener("keydown", event => {
      if (!state.editing || !(event.ctrlKey || event.metaKey)) return;
      const key = event.key.toLowerCase();
      if (key === "z" && event.shiftKey) {
        event.preventDefault();
        redoEdits();
      } else if (key === "z") {
        event.preventDefault();
        undoEdits();
      } else if (key === "y") {
        event.preventDefault();
        redoEdits();
      }
    });
  }

  async function activate() {
    init();
    ensurePreviewMap();
    setTimeout(() => state.map?.invalidateSize(), 30);
    await loadResults(false);
  }

  function getContext() {
    return {
      resultId: byId("ppResult")?.value || "",
      sourcePath: byId("ppGeojson")?.value || "",
      workflowId: state.workflowId,
      workflows: state.workflows.slice(),
      geojsonFiles: state.geojsonFiles.slice(),
      mode: state.mode,
    };
  }

  function setMode(mode) {
    state.mode = mode === "anomaly" ? "anomaly" : "segmentation";
    if (state.editing) stopEditing(true);
    clearPreviewLayers();
    const workflow = selectSavedWorkflow("", null, true);
    state.workflowId = workflow?.id || null;
    if (workflow) applyWorkflow(workflow);
    renderWorkflowList();
    document.dispatchEvent(new CustomEvent("postprocess:data", { detail: getContext() }));
  }

  async function runWorkflow(payload) {
    state.workflowId = payload.id;
    state.pollToken += 1;
    applyWorkflow(payload);
    await pollWorkflow(state.pollToken);
  }

  function selectWorkflow(workflowId) {
    const workflow = state.workflows.find(item => item.id === workflowId);
    if (!workflow) return false;
    state.workflowId = workflow.id;
    applyWorkflow(workflow);
    renderWorkflowList();
    return true;
  }

  window.PostprocessWorkspace = {
    getContext,
    loadPreviewLayer,
    requestJson,
    runWorkflow,
    selectWorkflow,
    setMessage,
    setMode,
  };
  window.PostProcessTab = { init, activate };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
