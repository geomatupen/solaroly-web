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
    measureActive: false,
    measureDoubleClickZoomWasEnabled: false,
    measureLayer: null,
    measurePoints: [],
    measurePreviousMapStatus: "",
    anomalyReviewPairs: [],
    anomalyReviewSelectedIndex: null,
    anomalyReviewAllLayer: null,
    anomalyReviewSelectedLayer: null,
    previewLayers: new Map(),
    previewLoading: new Map(),
    previewLayerCache: new Map(),
    previewDataCache: new Map(),
    previewDataPromises: new Map(),
    workflowStatusCache: new Map(),
    workflowStatusPromises: new Map(),
    workflows: [],
    scanComplete: false,
    editing: null,
    loadingPromise: null,
    referenceLayers: new Map(),
    referenceToken: 0,
    referenceSourceCache: new Map(),
    referenceLoadPromises: new Map(),
    temporarySequence: 0,
    previewHoverReset: null,
    mode: "segmentation",
    segmentationStepPhase: null,
    currentJobId: null,
    currentJob: null,
    jobs: [],
    jobsLoaded: false,
    jobsLoadingPromise: null,
    modeCache: new Map(),
    stepsLoading: false,
    modeLoadToken: 0,
    modeAbortController: new AbortController(),
    modeLoadTimer: null,
    deferredProcessingPromise: Promise.resolve(),
  };

  const GENERATED_STAGES = new Set([
    "combined", "regularized", "solar_rows", "deduplicated", "associated",
  ]);
  const SHAREABLE_STAGES = new Set(GENERATED_STAGES);
  const GENERATION_DETAIL_STAGES = new Set([
    "combined", "regularized", "solar_rows",
  ]);

  const PREVIEW_STYLES = {
    source: { label: "Source", color: "#38bdf8", weight: 1, fillOpacity: 0.08 },
    combined: { label: "Combined", color: "#f59e0b", weight: 2, fillOpacity: 0.13 },
    regularized: { label: "Regularized", color: "#22c55e", weight: 2, fillOpacity: 0.16 },
    solar_rows: { label: "Rows", color: "#ef4444", weight: 3, fillOpacity: 0.05 },
    panel_reference: { label: "Panel reference", color: "#14b8a6", weight: 2, fillOpacity: 0.08 },
    segmentation_regularized_reference: { label: "Final regularized panels", color: "#22c55e", weight: 2, fillOpacity: 0.10 },
    segmentation_rows_reference: { label: "Final rows", color: "#ef4444", weight: 3, fillOpacity: 0.035 },
    deduplicated: { label: "Deduplicated anomalies", color: "#f97316", weight: 2, fillOpacity: 0.22 },
    associated: { label: "Associated anomalies", color: "#eab308", weight: 2, fillOpacity: 0.25 },
  };

  const internalControls = new Map();
  for (const id of ["ppResult", "ppGeojson", "ppAnomalyGeojson"]) {
    const select = document.createElement("select");
    select.id = id;
    select.disabled = true;
    internalControls.set(id, select);
  }
  const resultSpinner = document.createElement("span");
  resultSpinner.id = "ppResultSpinner";
  resultSpinner.hidden = true;
  internalControls.set(resultSpinner.id, resultSpinner);
  const byId = id => document.getElementById(id) || internalControls.get(id) || null;

  function loadContext() {
    return {
      token: state.modeLoadToken,
      mode: state.mode,
      jobId: state.currentJobId,
      signal: state.modeAbortController.signal,
      persistent: Boolean(state.currentJob),
    };
  }

  function invalidateModeLoad() {
    if (state.modeLoadTimer) {
      window.clearTimeout(state.modeLoadTimer);
      state.modeLoadTimer = null;
    }
    state.modeAbortController.abort();
    state.modeAbortController = new AbortController();
    state.modeLoadToken += 1;
  }

  function isCurrentLoad(context) {
    return !context || (
      context.token === state.modeLoadToken
      && context.mode === state.mode
      && context.jobId === state.currentJobId
    );
  }

  function setStepsLoading(loading, message = "Loading configured GeoJSON data…") {
    state.stepsLoading = Boolean(loading);
    byId("ppSegmentationTab").disabled = false;
    byId("ppAnomalyTab").disabled = false;
    const indicator = byId("ppStepsLoading");
    if (!indicator) return;
    indicator.hidden = !loading;
    const text = indicator.querySelector("span:last-child");
    if (text) text.textContent = message;
  }
  let replacementResolver = null;

  function finishReplacementConfirmation(confirmed) {
    const modal = byId("ppReplacementModal");
    modal?.classList.remove("show");
    modal?.classList.add("hidden");
    const resolve = replacementResolver;
    replacementResolver = null;
    if (resolve) resolve(Boolean(confirmed));
  }

  function confirmReplacement(title, message) {
    if (replacementResolver) finishReplacementConfirmation(false);
    byId("ppReplacementTitle").textContent = title;
    byId("ppReplacementMessage").textContent = message;
    const modal = byId("ppReplacementModal");
    modal.classList.remove("hidden");
    modal.classList.add("show");
    byId("ppReplacementConfirm").focus();
    return new Promise(resolve => { replacementResolver = resolve; });
  }

  function closeGenerationDetails() {
    const modal = byId("ppGenerationDetailsModal");
    modal?.classList.remove("show");
    modal?.classList.add("hidden");
  }

  function detailValue(value, kind = "text") {
    if (value == null || value === "") return "Not recorded";
    if (kind === "boolean") return value ? "Yes" : "No";
    if (kind === "percent") return `${(Number(value) * 100).toLocaleString()}%`;
    if (typeof value === "number") return value.toLocaleString();
    return String(value);
  }

  function appendGenerationSection(container, title, entries) {
    const section = document.createElement("section");
    const heading = document.createElement("h4");
    heading.textContent = title;
    const grid = document.createElement("dl");
    grid.className = "postprocessGenerationDetailsGrid";
    for (const [label, value] of entries) {
      const term = document.createElement("dt");
      term.textContent = label;
      const description = document.createElement("dd");
      description.textContent = value;
      grid.append(term, description);
    }
    section.append(heading, grid);
    container.appendChild(section);
  }

  function renderGenerationDetails(status, stage, fallbackLabel) {
    const group = stage === "combined" ? "combined" : stage === "regularized" ? "regularized" : "hierarchy";
    const title = group === "combined"
      ? "Combined — generation details"
      : group === "regularized"
        ? "Regularized — generation details"
        : "Rows — generation details";
    byId("ppGenerationDetailsTitle").textContent = title;
    const container = byId("ppGenerationDetailsContent");
    container.replaceChildren();
    const introduction = document.createElement("p");
    introduction.className = "muted tiny";
    introduction.textContent = "These are the settings and source files used to generate this layer.";
    container.appendChild(introduction);

    const output = status.outputs?.[stage] || {};
    const source = group === "combined"
      ? status.input_path
      : group === "regularized"
        ? status.outputs?.combined?.path
        : status.hierarchy_parameters?.input_path || status.outputs?.regularized?.path;
    appendGenerationSection(container, "Layer", [
      ["Layer", fallbackLabel],
      ["Workflow", status.display_name || status.id],
      ["Workflow ID", status.id],
      ["Source GeoJSON", detailValue(source)],
      ["Output GeoJSON", detailValue(output.path)],
      ["Workflow created", detailValue(status.created_at)],
      ["Last workflow update", detailValue(status.updated_at)],
      ["Last manual save", detailValue(status.manual_edits?.[stage]?.updated_at)],
    ]);

    if (group === "combined") {
      const parameters = status.parameters || {};
      const stats = status.combine_stats || {};
      appendGenerationSection(container, "Configuration", [
        ["Edge tolerance", `${detailValue(parameters.edge_tolerance_px)} px`],
        ["Join gap", `${detailValue(parameters.gap_tolerance_px)} px`],
        ["Minimum boundary overlap", detailValue(parameters.min_boundary_overlap, "percent")],
        ["Maximum dimension factor", detailValue(parameters.max_dimension_factor)],
        ["Maximum area factor", detailValue(parameters.max_area_factor)],
        ["Remove contained polygons", detailValue(parameters.remove_contained_polygons, "boolean")],
      ]);
      appendGenerationSection(container, "Result summary", [
        ["Input polygons", detailValue(stats.input_features)],
        ["Output polygons", detailValue(stats.output_features)],
        ["Merged components", detailValue(stats.merged_components)],
        ["Contained polygons removed", detailValue(stats.contained_polygons_removed)],
        ["Accepted joins", detailValue(stats.accepted_links)],
      ]);
    } else if (group === "regularized") {
      const parameters = status.regularize_parameters || {};
      const stats = status.regularize_stats || {};
      appendGenerationSection(container, "Configuration", [
        ["Flag area change above", `${detailValue(parameters.max_area_change_percent)}%`],
        ["Geometry method", "Minimum-area oriented rectangle"],
      ]);
      appendGenerationSection(container, "Result summary", [
        ["Input polygons", detailValue(stats.input_features)],
        ["Output polygons", detailValue(stats.output_features)],
        ["Flagged for review", detailValue(stats.review_required)],
        ["Invalid polygons", detailValue(stats.invalid_input_features)],
      ]);
      if (status.hierarchy_stats) {
        const hierarchy = status.hierarchy_parameters || {};
        appendGenerationSection(container, "ID assignment", [
          ["Orientation tolerance", `${detailValue(hierarchy.max_orientation_difference_deg)}°`],
          ["Lateral distance factor", detailValue(hierarchy.max_lateral_distance_factor)],
          ["Along-row gap factor", detailValue(hierarchy.max_along_gap_factor)],
          ["Merge inner-row gap factor", detailValue(hierarchy.max_inner_row_gap_factor)],
          ["Panels assigned IDs", detailValue(status.hierarchy_stats.panel_count)],
          ["ID ordering", "Map reading order: top-to-bottom, then left-to-right"],
        ]);
      }
    } else {
      const parameters = status.hierarchy_parameters || {};
      const stats = status.hierarchy_stats || {};
      appendGenerationSection(container, "Configuration", [
        ["Orientation tolerance", `${detailValue(parameters.max_orientation_difference_deg)}°`],
        ["Lateral distance factor", detailValue(parameters.max_lateral_distance_factor)],
        ["Along-row gap factor", detailValue(parameters.max_along_gap_factor)],
        ["Merge inner-row gap factor", detailValue(parameters.max_inner_row_gap_factor)],
        ["ID ordering", "Map reading order: top-to-bottom, then left-to-right"],
      ]);
      appendGenerationSection(container, "Result summary", [
        ["Input polygons", detailValue(stats.input_features)],
        ["Solar panels", detailValue(stats.panel_count)],
        ["Rows", detailValue(stats.row_count)],
        ["Inner rows", detailValue(stats.inner_row_count)],
        ["Singleton rows", detailValue(stats.singleton_rows)],
      ]);
    }
  }

  async function showGenerationDetails(stage, fallbackLabel) {
    if (!state.workflowId) return;
    const modal = byId("ppGenerationDetailsModal");
    byId("ppGenerationDetailsTitle").textContent = "Generation details";
    byId("ppGenerationDetailsContent").innerHTML = '<div class="mapListLoading"><span class="spinner" aria-hidden="true"></span><span>Loading generation details…</span></div>';
    modal.classList.remove("hidden");
    modal.classList.add("show");
    try {
      const resultId = byId("ppResult").value;
      const status = await requestJson(
        `/api/results/${encodeURIComponent(resultId)}/postprocess/${encodeURIComponent(state.workflowId)}`,
        { cache: "no-store" },
      );
      renderGenerationDetails(status, stage, fallbackLabel);
    } catch (error) {
      byId("ppGenerationDetailsContent").textContent = `Could not load generation details: ${error.message}`;
    }
  }

  function initCollapsibleSteps() {
    document.querySelectorAll(".postprocessStep").forEach(step => {
      const body = step.querySelector(":scope > .postprocessStepBody");
      const heading = body?.querySelector(":scope > h3");
      if (!body || !heading || body.querySelector(":scope > .postprocessStepTitleRow")) return;
      const titleRow = document.createElement("div");
      titleRow.className = "postprocessStepTitleRow";
      body.insertBefore(titleRow, heading);
      titleRow.appendChild(heading);
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "secondary tiny postprocessStepCollapse";
      toggle.textContent = "−";
      toggle.title = "Minimize step";
      toggle.setAttribute("aria-expanded", "true");
      toggle.addEventListener("click", () => {
        const collapsed = step.classList.toggle("collapsed");
        toggle.textContent = collapsed ? "+" : "−";
        toggle.title = collapsed ? "Expand step" : "Minimize step";
        toggle.setAttribute("aria-expanded", String(!collapsed));
      });
      titleRow.appendChild(toggle);
    });
  }

  function setSourceStepExpanded(step, expanded) {
    if (!step) return;
    const body = step.querySelector(":scope > .postprocessSourceStepBody");
    const toggle = step.querySelector(":scope > .postprocessSourceStepHeader .postprocessSourceStepToggle");
    step.classList.toggle("collapsed", !expanded);
    if (body) body.hidden = !expanded;
    if (toggle) {
      toggle.textContent = expanded ? "−" : "+";
      toggle.title = expanded ? "Minimize step" : "Expand step";
      toggle.setAttribute("aria-expanded", String(expanded));
    }
  }

  function resetSourceSteps(modal) {
    modal?.querySelectorAll(".postprocessSourceStep").forEach((step, index) => {
      setSourceStepExpanded(step, index === 0);
    });
  }

  function resetSourceInfo(modal) {
    modal?.querySelectorAll(".postprocessSourceInfoToggle").forEach(toggle => {
      const info = byId(toggle.getAttribute("aria-controls"));
      if (info) info.hidden = true;
      toggle.setAttribute("aria-expanded", "false");
      toggle.classList.remove("active");
    });
  }

  function revealSourceField(field) {
    setSourceStepExpanded(field?.closest(".postprocessSourceStep"), true);
  }

  function resetSourceStepCompletion(field) {
    const step = field?.closest(".postprocessSourceStep");
    step?.classList.remove("complete");
    step?.querySelector(":scope > .postprocessSourceStepSummary")?.classList.remove("complete");
  }

  function completeSourceStep(summary, advance = false) {
    const step = summary?.closest(".postprocessSourceStep");
    if (!step) return;
    step.classList.add("complete");
    summary.classList.add("complete");
    if (!advance) return;
    const nextStep = step.nextElementSibling?.classList.contains("postprocessSourceStep")
      ? step.nextElementSibling
      : null;
    setSourceStepExpanded(step, false);
    setSourceStepExpanded(nextStep, true);
    nextStep?.querySelector("select")?.focus();
  }

  function setSourceStepMessage(summary, message, loading = false) {
    if (!summary) return;
    summary.replaceChildren();
    summary.classList.toggle("loading", loading);
    if (loading) {
      const spinner = document.createElement("span");
      spinner.className = "spinner";
      spinner.setAttribute("aria-hidden", "true");
      summary.appendChild(spinner);
    }
    const text = document.createElement("span");
    text.textContent = message;
    summary.appendChild(text);
  }

  function initSourceStepToggles() {
    document.querySelectorAll(".postprocessSourceStepToggle").forEach(toggle => {
      toggle.addEventListener("click", () => {
        const step = toggle.closest(".postprocessSourceStep");
        setSourceStepExpanded(step, step?.classList.contains("collapsed"));
      });
    });
    document.querySelectorAll(".postprocessSourceInfoToggle").forEach(toggle => {
      toggle.addEventListener("click", () => {
        const info = byId(toggle.getAttribute("aria-controls"));
        if (!info) return;
        if (info.hidden) setSourceStepExpanded(toggle.closest(".postprocessSourceStep"), true);
        info.hidden = !info.hidden;
        toggle.setAttribute("aria-expanded", String(!info.hidden));
        toggle.classList.toggle("active", !info.hidden);
      });
    });
  }

  function setStepCollapsed(step, collapsed) {
    if (!step) return;
    step.classList.toggle("collapsed", collapsed);
    const toggle = step.querySelector(".postprocessStepCollapse");
    if (!toggle) return;
    toggle.textContent = collapsed ? "+" : "−";
    toggle.title = collapsed ? "Expand step" : "Minimize step";
    toggle.setAttribute("aria-expanded", String(!collapsed));
  }

  function syncSegmentationStepProgress(status = null) {
    const outputs = status?.outputs || {};
    const hasCombined = Boolean(outputs.combined);
    const hasRegularized = Boolean(outputs.regularized);
    const hasHierarchy = Boolean(outputs.solar_rows);
    const canShow = state.scanComplete || hasCombined || hasRegularized || hasHierarchy;
    if (!canShow) return;
    const steps = [byId("ppCombineStep"), byId("ppRegularizeStep"), byId("ppHierarchyStep")];
    steps.forEach(step => { step.hidden = false; });
    const available = [state.scanComplete || hasCombined, hasCombined, hasRegularized];
    steps.forEach((step, index) => {
      step.classList.toggle("locked", !available[index]);
      step.setAttribute("aria-disabled", String(!available[index]));
    });
    const phase = hasHierarchy ? 0 : hasRegularized ? 3 : hasCombined ? 2 : 1;
    if (phase === state.segmentationStepPhase) return;
    state.segmentationStepPhase = phase;
    steps.forEach((step, index) => setStepCollapsed(step, phase === 0 || index + 1 !== phase));
  }

  function closeLayerMenus(except = null) {
    document.querySelectorAll(".postprocessLayerMenu, .postprocessWorkflowMenu").forEach(menu => {
      if (menu !== except) menu.hidden = true;
    });
  }

  function removePortalLayerMenus(owner) {
    document.querySelectorAll(`.postprocessLayerMenu[data-menu-owner="${owner}"]`).forEach(menu => menu.remove());
  }

  function layerMenuButton(label, menu, action, { disabled = false, danger = false } = {}) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.disabled = disabled;
    if (danger) button.classList.add("danger");
    button.addEventListener("click", event => {
      event.stopPropagation();
      menu.hidden = true;
      action();
    });
    return button;
  }

  function createLayerMenu(label, disabled = false, owner = "processing") {
    const wrapper = document.createElement("div");
    wrapper.className = "postprocessLayerMenuWrap";
    const dots = document.createElement("button");
    dots.type = "button";
    dots.className = "iconDots";
    dots.textContent = "⋮";
    dots.disabled = disabled;
    dots.setAttribute("aria-label", `Options for ${label}`);
    const menu = document.createElement("div");
    menu.className = "postprocessLayerMenu";
    menu.dataset.menuOwner = owner;
    menu.hidden = true;
    dots.addEventListener("click", event => {
      event.stopPropagation();
      const willOpen = menu.hidden;
      closeLayerMenus(menu);
      if (!willOpen) {
        menu.hidden = true;
        return;
      }
      const fullscreenPreview = document.fullscreenElement === document.querySelector(".postprocessPreview")
        ? document.fullscreenElement
        : null;
      (fullscreenPreview || document.body).appendChild(menu);
      menu.hidden = false;
      const buttonBounds = dots.getBoundingClientRect();
      const menuBounds = menu.getBoundingClientRect();
      const margin = 8;
      const left = Math.max(margin, Math.min(
        buttonBounds.right - menuBounds.width,
        window.innerWidth - menuBounds.width - margin,
      ));
      const roomBelow = window.innerHeight - buttonBounds.bottom;
      const top = roomBelow >= menuBounds.height + margin
        ? buttonBounds.bottom + 4
        : Math.max(margin, buttonBounds.top - menuBounds.height - 4);
      menu.style.left = `${Math.round(left)}px`;
      menu.style.top = `${Math.round(top)}px`;
    });
    wrapper.appendChild(dots);
    document.body.appendChild(menu);
    return { wrapper, menu };
  }

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
    const values = feature?.properties || {};
    const priorityKeys = [
      "anomaly_count", "anomaly_ids", "anomaly_id",
      "panel_id", "row_id", "association_method",
      "panel_overlap_fraction", "panel_distance_m", "review_required",
    ];
    const orderedKeys = [
      ...priorityKeys.filter(key => Object.prototype.hasOwnProperty.call(values, key)),
      ...Object.keys(values).filter(key => !priorityKeys.includes(key)),
    ];
    const properties = orderedKeys.slice(0, 30).map(key => [key, values[key]]);
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
    element.hidden = !message;
  }

  function numberValue(id) {
    return Number(byId(id).value);
  }

  function setIconButtonLabel(id, label) {
    const button = byId(id);
    if (!button) return;
    button.title = label;
    button.setAttribute("aria-label", label);
  }

  function stopIndeterminate() {
    const progress = byId("ppProgressWrap");
    progress.classList.remove("ppIndeterminate");
    progress.hidden = true;
  }

  function showConfiguredInitialStep() {
    if (!state.currentJob?.sources?.[state.mode]?.path) return;
    if (state.mode === "anomaly") {
      const steps = [byId("ppOverlapDeduplicateStep"), byId("ppDeduplicateStep"), byId("ppAdjustAnomaliesStep"), byId("ppAssociateStep")];
      steps.forEach((step, index) => {
        step.hidden = false;
        step.classList.add("locked");
        step.setAttribute("aria-disabled", "true");
        setStepCollapsed(step, index !== 0);
      });
      byId("ppDeduplicate").disabled = true;
      return;
    }
    const steps = [byId("ppCombineStep"), byId("ppRegularizeStep"), byId("ppHierarchyStep")];
    steps.forEach((step, index) => {
      step.hidden = false;
      step.classList.add("locked");
      step.setAttribute("aria-disabled", "true");
      setStepCollapsed(step, index !== 0);
    });
  }

  function resetAnalysis({ showConfiguredStep = false } = {}) {
    state.analysis = null;
    state.scanComplete = false;
    state.workflowId = null;
    state.segmentationStepPhase = null;
    state.pollToken += 1;
    byId("ppCombine").disabled = true;
    byId("ppRegularize").disabled = true;
    byId("ppCombineStep").hidden = true;
    byId("ppRegularizeStep").hidden = true;
    byId("ppHierarchyStep").hidden = true;
    byId("ppOverlapDeduplicateStep").hidden = true;
    byId("ppDeduplicateStep").hidden = true;
    byId("ppAdjustAnomaliesStep").hidden = true;
    byId("ppAssociateStep").hidden = true;
    byId("ppDeduplicate").disabled = true;
    byId("ppRemoveOverlappingAnomalies").disabled = true;
    byId("ppSkipVisualDeduplication").disabled = true;
    byId("ppAssociate").disabled = true;
    byId("ppProgressWrap").hidden = true;
    byId("ppLogWrap").hidden = true;
    byId("ppLog").textContent = "";
    clearPreviewLayers();
    if (showConfiguredStep) showConfiguredInitialStep();
  }

  function formatMeasuredDistance(distanceMetres) {
    if (distanceMetres >= 1000) {
      const kilometres = distanceMetres / 1000;
      return `${kilometres >= 10 ? kilometres.toFixed(1) : kilometres.toFixed(2)} km`;
    }
    return `${distanceMetres >= 100 ? distanceMetres.toFixed(0) : distanceMetres.toFixed(1)} m`;
  }

  function measuredDistance() {
    if (!state.map || state.measurePoints.length < 2) return 0;
    return state.measurePoints.slice(1).reduce((total, point, index) => (
      total + state.map.distance(state.measurePoints[index], point)
    ), 0);
  }

  function measureControlButton() {
    return document.getElementById("ppMeasureDistance");
  }

  function renderMeasurement() {
    if (!state.map) return;
    if (!state.measureLayer) state.measureLayer = window.L.layerGroup().addTo(state.map);
    state.measureLayer.clearLayers();
    if (!state.measurePoints.length) return;

    if (state.measurePoints.length > 1) {
      window.L.polyline(state.measurePoints, {
        color: "#38bdf8",
        weight: 3,
        opacity: 0.95,
        dashArray: "7 6",
        interactive: false,
      }).addTo(state.measureLayer);
    }
    state.measurePoints.forEach(point => {
      window.L.circleMarker(point, {
        radius: 4,
        color: "#e0f2fe",
        weight: 2,
        fillColor: "#0284c7",
        fillOpacity: 1,
        interactive: false,
      }).addTo(state.measureLayer);
    });
    window.L.tooltip({
      permanent: true,
      direction: "top",
      offset: [0, -8],
      className: "postprocessMeasureTooltip",
    })
      .setLatLng(state.measurePoints[state.measurePoints.length - 1])
      .setContent(state.measurePoints.length === 1 ? "Add another point" : formatMeasuredDistance(measuredDistance()))
      .addTo(state.measureLayer);
  }

  function finishMeasurement({ clear = false } = {}) {
    if (!state.map) return;
    state.measureActive = false;
    state.map.getContainer().classList.remove("postprocessMeasureMode");
    state.map.off("click", addMeasurementPoint);
    state.map.off("dblclick", finishMeasurementFromMap);
    document.removeEventListener("keydown", handleMeasurementKeydown);
    if (state.measureDoubleClickZoomWasEnabled) state.map.doubleClickZoom.enable();
    const button = measureControlButton();
    if (button) {
      button.classList.remove("active");
      button.setAttribute("aria-pressed", "false");
      button.title = "Measure distance";
      button.setAttribute("aria-label", "Measure distance");
    }
    const mapStatus = byId("ppMapStatus");
    if (mapStatus && state.measurePreviousMapStatus) {
      mapStatus.textContent = state.measurePreviousMapStatus;
      state.measurePreviousMapStatus = "";
    }
    if (clear) {
      state.measurePoints = [];
      state.measureLayer?.clearLayers();
    }
  }

  function addMeasurementPoint(event) {
    const previous = state.measurePoints[state.measurePoints.length - 1];
    if (!previous || previous.distanceTo(event.latlng) > 0.01) state.measurePoints.push(event.latlng);
    renderMeasurement();
  }

  function finishMeasurementFromMap(event) {
    window.L.DomEvent.stop(event.originalEvent);
    finishMeasurement({ clear: state.measurePoints.length < 2 });
  }

  function handleMeasurementKeydown(event) {
    if (event.key === "Escape") finishMeasurement({ clear: true });
  }

  function startMeasurement() {
    if (!state.map) return;
    if (!stopEditing(true)) return;
    finishMeasurement({ clear: true });
    state.measureActive = true;
    state.measureDoubleClickZoomWasEnabled = state.map.doubleClickZoom.enabled();
    state.map.doubleClickZoom.disable();
    state.map.getContainer().classList.add("postprocessMeasureMode");
    const button = measureControlButton();
    button?.classList.add("active");
    button?.setAttribute("aria-pressed", "true");
    if (button) {
      const finishLabel = "Finish measuring (double-click map or press Escape to cancel)";
      button.title = finishLabel;
      button.setAttribute("aria-label", finishLabel);
    }
    const mapStatus = byId("ppMapStatus");
    if (mapStatus) {
      state.measurePreviousMapStatus = mapStatus.textContent;
      mapStatus.textContent = "Measurement active — click the map to add points, then double-click to finish.";
    }
    state.map.on("click", addMeasurementPoint);
    state.map.on("dblclick", finishMeasurementFromMap);
    document.addEventListener("keydown", handleMeasurementKeydown);
  }

  function addMeasureControl(map) {
    const MeasureControl = window.L.Control.extend({
      options: { position: "topleft" },
      onAdd() {
        const container = window.L.DomUtil.create("div", "leaflet-bar postprocessMeasureControl");
        const button = window.L.DomUtil.create("button", "postprocessMeasureButton", container);
        button.id = "ppMeasureDistance";
        button.type = "button";
        button.title = "Measure distance";
        button.setAttribute("aria-label", "Measure distance");
        button.setAttribute("aria-pressed", "false");
        button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 17 17 5l2 2L7 19 5 17Zm3-1-2-2m5-1-2-2m5-1-2-2m5-1-2-2"/></svg>';
        window.L.DomEvent.disableClickPropagation(container);
        return container;
      },
    });
    new MeasureControl().addTo(map);
    const button = document.getElementById("ppMeasureDistance");
    if (!button) {
      console.error("Measure control button #ppMeasureDistance was not created.");
      return;
    }
    button.onclick = event => {
      event.preventDefault();
      event.stopPropagation();
      if (state.measureActive) finishMeasurement({ clear: state.measurePoints.length < 2 });
      else startMeasurement();
    };
  }

  function ensurePreviewMap() {
    if (state.map || !window.L || !byId("ppMap")) return state.map;
    const street = window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 22,
      attribution: "&copy; OpenStreetMap contributors",
    });
    const satellite = window.L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 22,
      attribution: "&copy; Esri",
    });
    state.map = window.L.map("ppMap", {
      preferCanvas: true,
      zoomControl: true,
      layers: [street],
    }).setView([48.8566, 2.3522], 5);
    const rasterPane = state.map.createPane("ppRasterPane");
    rasterPane.style.zIndex = "250";
    const referencePane = state.map.createPane("ppReferencePane");
    referencePane.style.zIndex = "350";
    const rowsPane = state.map.createPane("ppRowsPane");
    rowsPane.style.zIndex = "410";
    const panelsPane = state.map.createPane("ppPanelsPane");
    panelsPane.style.zIndex = "420";
    const anomaliesPane = state.map.createPane("ppAnomaliesPane");
    anomaliesPane.style.zIndex = "430";
    const anomalyReviewPane = state.map.createPane("ppAnomalyReviewPane");
    anomalyReviewPane.style.zIndex = "440";
    window.L.control.layers({ Street: street, Satellite: satellite }, {}, { position: "topleft" }).addTo(state.map);
    addMeasureControl(state.map);
    return state.map;
  }

  function geoJsonFeatureCenter(feature) {
    const coordinates = feature?.geometry?.coordinates;
    if (!Array.isArray(coordinates)) return null;
    let minimumLongitude = Infinity;
    let minimumLatitude = Infinity;
    let maximumLongitude = -Infinity;
    let maximumLatitude = -Infinity;
    const visit = value => {
      if (!Array.isArray(value)) return;
      if (value.length >= 2 && Number.isFinite(Number(value[0])) && Number.isFinite(Number(value[1]))) {
        const longitude = Number(value[0]);
        const latitude = Number(value[1]);
        minimumLongitude = Math.min(minimumLongitude, longitude);
        maximumLongitude = Math.max(maximumLongitude, longitude);
        minimumLatitude = Math.min(minimumLatitude, latitude);
        maximumLatitude = Math.max(maximumLatitude, latitude);
        return;
      }
      value.forEach(visit);
    };
    visit(coordinates);
    if (![minimumLongitude, minimumLatitude, maximumLongitude, maximumLatitude].every(Number.isFinite)) return null;
    return window.L.latLng(
      (minimumLatitude + maximumLatitude) / 2,
      (minimumLongitude + maximumLongitude) / 2,
    );
  }

  function anomalySourceFeatures() {
    return state.previewLayers.get("source")?.geojson?.features || [];
  }

  function anomalyReviewFeature(pair, side, sourceFeatures = anomalySourceFeatures()) {
    const source = sourceFeatures[Number(pair?.[`${side}_index`])];
    const geometry = pair?.[`${side}_geometry`];
    if (geometry) {
      return {
        type: "Feature",
        geometry,
        properties: {
          ...(source?.properties || {}),
          anomaly_id: pair?.[`${side}_anomaly_id`],
          ...(pair?.[`${side}_image`] ? { image: pair[`${side}_image`] } : {}),
        },
      };
    }
    return source;
  }

  function clearAnomalyReviewMap() {
    if (state.map) {
      for (const layer of [state.anomalyReviewAllLayer, state.anomalyReviewSelectedLayer]) {
        if (layer && state.map.hasLayer(layer)) state.map.removeLayer(layer);
      }
    }
    state.anomalyReviewAllLayer = null;
    state.anomalyReviewSelectedLayer = null;
    state.anomalyReviewPairs = [];
    state.anomalyReviewSelectedIndex = null;
    byId("ppAnomalyMapReview").hidden = true;
    byId("ppAnomalyMapPairFilter").value = "active";
  }

  function anomalyPairColor(pair) {
    if (pair.review_status === "accepted") return "#22c55e";
    if (pair.review_status === "rejected") return "#ef6b73";
    if (pair.review_status === "duplicate") return "#ef4444";
    if (pair.review_status === "review") return "#f59e0b";
    return "#64748b";
  }

  function renderAllAnomalyReviewPairs() {
    if (!state.map) return;
    if (state.anomalyReviewAllLayer && state.map.hasLayer(state.anomalyReviewAllLayer)) {
      state.map.removeLayer(state.anomalyReviewAllLayer);
    }
    state.anomalyReviewAllLayer = null;
    const features = anomalySourceFeatures();
    const group = window.L.layerGroup();
    const featureStyles = new Map();
    const filter = byId("ppAnomalyMapPairFilter")?.value || "active";
    const pairs = state.anomalyReviewPairs.filter(pair => {
      if (filter === "all") return true;
      if (filter === "active") return pair.review_status !== "rejected";
      return pair.review_status === filter;
    });
    for (const pair of pairs) {
      const first = anomalyReviewFeature(pair, "first", features);
      const second = anomalyReviewFeature(pair, "second", features);
      const firstCenter = geoJsonFeatureCenter(first);
      const secondCenter = geoJsonFeatureCenter(second);
      const color = anomalyPairColor(pair);
      if (firstCenter && secondCenter) {
        window.L.polyline([firstCenter, secondCenter], {
          pane: "ppAnomalyReviewPane",
          color,
          weight: 1.5,
          opacity: 0.5,
          dashArray: "5 5",
          interactive: false,
        }).addTo(group);
      }
      for (const [index, feature] of [
        [Number(pair.first_index), first],
        [Number(pair.second_index), second],
      ]) {
        const priority = pair.review_status === "accepted" ? 4
          : pair.review_status === "duplicate" ? 3
            : pair.review_status === "review" || pair.review_status === "rejected" ? 2 : 1;
        if (!featureStyles.has(index) || featureStyles.get(index).priority < priority) {
          featureStyles.set(index, { color, priority, feature });
        }
      }
    }
    for (const style of featureStyles.values()) {
      const feature = style.feature;
      if (!feature) continue;
      window.L.geoJSON(feature, {
        pane: "ppAnomalyReviewPane",
        renderer: window.L.svg({ pane: "ppAnomalyReviewPane", padding: 0.5 }),
        interactive: false,
        style: { color: style.color, weight: 1.5, opacity: 0.6, fillColor: style.color, fillOpacity: 0.08 },
      }).addTo(group);
    }
    group.addTo(state.map);
    state.anomalyReviewAllLayer = group;
    state.anomalyReviewSelectedLayer?.bringToFront?.();
  }

  function showAnomalyReviewPair(pair, comparisonIndex = 0) {
    const map = ensurePreviewMap();
    const features = anomalySourceFeatures();
    if (!map || !pair || (!features.length && !pair.first_geometry && !pair.second_geometry)) {
      setMessage("The configured anomaly source layer must be loaded before a pair can be shown on the map.", "err");
      return false;
    }
    if (state.anomalyReviewSelectedLayer && map.hasLayer(state.anomalyReviewSelectedLayer)) {
      map.removeLayer(state.anomalyReviewSelectedLayer);
    }
    state.anomalyReviewPairs = [pair];
    state.anomalyReviewSelectedIndex = comparisonIndex;
    const first = anomalyReviewFeature(pair, "first", features);
    const second = anomalyReviewFeature(pair, "second", features);
    if (!first || !second) {
      setMessage("This comparison no longer matches the loaded anomaly source ordering.", "err");
      return false;
    }
    renderAllAnomalyReviewPairs();
    const selected = window.L.featureGroup();
    const firstLayer = window.L.geoJSON(first, {
      pane: "ppAnomalyReviewPane",
      renderer: window.L.svg({ pane: "ppAnomalyReviewPane", padding: 0.5 }),
      style: { color: "#22d3ee", weight: 4, opacity: 1, fillColor: "#22d3ee", fillOpacity: 0.22 },
      onEachFeature: (feature, layer) => {
        layer.on("click", event => {
          window.L.popup({ className: "postprocessPolygonPopup", maxWidth: 340, maxHeight: 210 })
            .setLatLng(event.latlng)
            .setContent(polygonPopupContent(feature, "Left anomaly"))
            .openOn(state.map);
        });
      },
    }).bindTooltip(`Left anomaly · ID ${pair.first_anomaly_id ?? Number(pair.first_index) + 1}`, { sticky: true });
    const secondLayer = window.L.geoJSON(second, {
      pane: "ppAnomalyReviewPane",
      renderer: window.L.svg({ pane: "ppAnomalyReviewPane", padding: 0.5 }),
      style: { color: "#e879f9", weight: 4, opacity: 1, fillColor: "#e879f9", fillOpacity: 0.22 },
      onEachFeature: (feature, layer) => {
        layer.on("click", event => {
          window.L.popup({ className: "postprocessPolygonPopup", maxWidth: 340, maxHeight: 210 })
            .setLatLng(event.latlng)
            .setContent(polygonPopupContent(feature, "Right anomaly"))
            .openOn(state.map);
        });
      },
    }).bindTooltip(`Right anomaly · ID ${pair.second_anomaly_id ?? Number(pair.second_index) + 1}`, { sticky: true });
    firstLayer.addTo(selected);
    secondLayer.addTo(selected);
    const centers = [geoJsonFeatureCenter(first), geoJsonFeatureCenter(second)].filter(Boolean);
    if (centers.length === 2) {
      window.L.polyline(centers, { pane: "ppAnomalyReviewPane", interactive: false, color: "#ffffff", weight: 2.5, dashArray: "7 6", opacity: 0.95 }).addTo(selected);
    }
    selected.addTo(map);
    state.anomalyReviewSelectedLayer = selected;
    const bounds = selected.getBounds();
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [60, 60], maxZoom: 21 });
    byId("ppAnomalyMapReview").hidden = false;
    byId("ppAnomalyMapReviewTitle").textContent = `Comparison ${comparisonIndex + 1}`;
    byId("ppAnomalyMapReviewDetails").textContent = `IDs ${pair.first_anomaly_id ?? Number(pair.first_index) + 1} / ${pair.second_anomaly_id ?? Number(pair.second_index) + 1} · ${pair.display_score == null ? "Score unavailable" : `${Math.round(pair.display_score)}% duplicate`} · ${Number(pair.center_distance_m || 0).toFixed(2)} m apart`;
    window.requestAnimationFrame(() => map.invalidateSize());
    return true;
  }

  function updateAnomalyReviewPairDecision(firstIndex, secondIndex, reviewStatus) {
    const edge = [Number(firstIndex), Number(secondIndex)].sort((first, second) => first - second).join(":");
    for (const pair of state.anomalyReviewPairs) {
      const pairEdge = [Number(pair.first_index), Number(pair.second_index)].sort((first, second) => first - second).join(":");
      if (pairEdge === edge) pair.review_status = reviewStatus || "below";
    }
    renderAllAnomalyReviewPairs();
  }

  function clearPreviewLayers() {
    stopEditing(false);
    finishMeasurement({ clear: true });
    clearAnomalyReviewMap();
    if (state.map) {
      for (const item of state.previewLayers.values()) {
        if (state.map.hasLayer(item.layer)) state.map.removeLayer(item.layer);
      }
    }
    state.previewLayers.clear();
    state.previewLoading.clear();
    updateFitButton();
    byId("ppLayerList").innerHTML = '<div class="muted tiny">No preview layers loaded.</div>';
    byId("ppMapStatus").textContent = "Configured layers will appear here.";
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

  function visibleLayerKeys(collection) {
    const visible = new Set();
    if (!state.map) return visible;
    for (const [key, item] of collection) {
      if (item.layer && state.map.hasLayer(item.layer)) visible.add(key);
    }
    return visible;
  }

  function detachLayerCollection(collection) {
    if (!state.map) return;
    for (const item of collection.values()) {
      if (item.layer && state.map.hasLayer(item.layer)) state.map.removeLayer(item.layer);
    }
  }

  function cacheCurrentModeState() {
    if (!state.currentJobId) return;
    const activeWorkflow = state.workflows.find(item => item.id === state.workflowId);
    const workflowRunning = activeWorkflow?.status === "queued" || activeWorkflow?.status === "running";
    if (state.stepsLoading || workflowRunning) {
      state.modeCache.delete(state.mode);
      detachLayerCollection(state.previewLayers);
      detachLayerCollection(state.referenceLayers);
      return;
    }
    state.modeCache.set(state.mode, {
      resultId: byId("ppResult").value,
      sourcePath: byId("ppGeojson").value,
      geojsonFiles: state.geojsonFiles.slice(),
      workflows: state.workflows.slice(),
      analysis: state.analysis,
      scanComplete: state.scanComplete,
      workflowId: state.workflowId,
      segmentationStepPhase: state.segmentationStepPhase,
      previewLayers: state.previewLayers,
      previewVisible: visibleLayerKeys(state.previewLayers),
      referenceLayers: state.referenceLayers,
      referenceVisible: visibleLayerKeys(state.referenceLayers),
      message: byId("ppMessage").textContent,
      messageClass: byId("ppMessage").className,
      mapStatus: byId("ppMapStatus").textContent,
      referenceStatus: byId("ppReferenceStatus").textContent,
    });
    detachLayerCollection(state.previewLayers);
    detachLayerCollection(state.referenceLayers);
  }

  function restoreModeState(mode) {
    const cached = state.modeCache.get(mode);
    if (!cached) return false;
    state.geojsonFiles = cached.geojsonFiles.slice();
    state.workflows = cached.workflows.slice();
    state.analysis = cached.analysis;
    state.scanComplete = cached.scanComplete;
    state.workflowId = cached.workflowId;
    state.segmentationStepPhase = cached.segmentationStepPhase;
    state.previewLayers = cached.previewLayers;
    state.referenceLayers = cached.referenceLayers;
    const result = byId("ppResult");
    if ([...result.options].some(option => option.value === cached.resultId)) result.value = cached.resultId;
    const source = byId("ppGeojson");
    source.replaceChildren();
    addOption(source, "", "Select a GeoJSON…");
    for (const file of state.geojsonFiles) addOption(source, file.path, `${file.name} · ${file.stage}`);
    if ([...source.options].some(option => option.value === cached.sourcePath)) source.value = cached.sourcePath;
    source.disabled = true;
    for (const [key, item] of state.previewLayers) {
      if (cached.previewVisible.has(key) && item.layer && state.map && !state.map.hasLayer(item.layer)) item.layer.addTo(state.map);
    }
    for (const [key, item] of state.referenceLayers) {
      if (cached.referenceVisible.has(key) && item.layer && state.map && !state.map.hasLayer(item.layer)) item.layer.addTo(state.map);
    }
    if (mode === "segmentation") {
      showConfiguredInitialStep();
      const workflow = state.workflows.find(item => item.id === state.workflowId);
      syncSegmentationStepProgress(workflow || null);
    }
    renderReferenceLayers();
    renderPreviewLayers();
    renderWorkflowList();
    byId("ppMessage").textContent = cached.message;
    byId("ppMessage").className = cached.messageClass;
    byId("ppMessage").hidden = !cached.message;
    byId("ppMapStatus").textContent = cached.mapStatus;
    byId("ppReferenceStatus").textContent = cached.referenceStatus;
    setStepsLoading(false);
    document.dispatchEvent(new CustomEvent("postprocess:data", { detail: getContext() }));
    window.requestAnimationFrame(() => state.map?.invalidateSize());
    return true;
  }

  function resetModeCache() {
    invalidateModeLoad();
    for (const cached of state.modeCache.values()) {
      detachLayerCollection(cached.previewLayers);
      detachLayerCollection(cached.referenceLayers);
    }
    state.modeCache.clear();
    state.workflowStatusCache.clear();
    state.workflowStatusPromises.clear();
    document.dispatchEvent(new CustomEvent("postprocess:cache-reset"));
  }

  function invalidateCachedMode(mode) {
    const cached = state.modeCache.get(mode);
    if (cached) {
      detachLayerCollection(cached.previewLayers);
      state.modeCache.delete(mode);
    }
    const source = state.currentJob?.sources?.[mode];
    const binding = state.currentJob?.workflows?.[mode];
    if (source?.workspace_result_id && binding?.workflow_id) {
      state.workflowStatusCache.delete(workflowStatusKey(source.workspace_result_id, binding.workflow_id));
    }
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
    removePortalLayerMenus("reference");
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
      const layerMenu = createLayerMenu(item.label, Boolean(item.loading), "reference");
      layerMenu.menu.appendChild(layerMenuButton("Focus", layerMenu.menu, () => {
        const bounds = item.bounds || item.layer?.getBounds?.();
        if (bounds?.isValid()) state.map.fitBounds(bounds, { padding: [18, 18], maxZoom: 21 });
      }, { disabled: !item.loaded }));
      const labelFields = item.geojson ? layerLabelFields(item) : [];
      if (labelFields.length) {
        layerMenu.menu.appendChild(layerMenuButton(
          item.labelControlOpen ? "Hide label options" : "Labels…",
          layerMenu.menu,
          () => {
            item.labelControlOpen = !item.labelControlOpen;
            renderReferenceLayers();
          },
        ));
      }
      if (item.temporary) {
        layerMenu.menu.appendChild(layerMenuButton("Remove", layerMenu.menu, () => {
          if (state.map.hasLayer(item.layer)) state.map.removeLayer(item.layer);
          state.referenceLayers.delete(id);
          renderReferenceLayers();
          byId("ppReferenceStatus").textContent = "Temporary layer removed. No project file was changed.";
        }, { danger: true }));
      }
      actions.appendChild(layerMenu.wrapper);
      row.append(checkbox, swatch, text, actions);
      container.appendChild(row);
      if (item.labelControlOpen && labelFields.length) {
        const control = document.createElement("div");
        control.className = "postprocessLayerLabelControl";
        const label = document.createElement("label");
        label.textContent = "Label field";
        const select = document.createElement("select");
        addOption(select, "", "Off");
        for (const field of labelFields) addOption(select, field, field);
        select.value = item.labelField || "";
        select.addEventListener("change", () => applyLayerLabels(item, select.value));
        label.appendChild(select);
        control.appendChild(label);
        container.appendChild(control);
      }
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

  function imageMatchTokens(...values) {
    const tokens = new Set();
    const add = value => {
      const basename = String(value || "").split(/[\\/]/).pop().toLowerCase();
      if (!basename) return;
      tokens.add(basename);
      tokens.add(basename.replace(/\.[^.]+$/, ""));
    };
    for (const value of values.flat(Infinity)) add(value);
    return tokens;
  }

  function linkedImageRecordForFeature(feature) {
    const properties = feature?.properties || {};
    const primaryTargets = imageMatchTokens(
      properties.image,
      properties.file,
      properties.src,
      properties.name,
    );
    const fallbackTargets = imageMatchTokens(properties.source_images || []);
    for (const targets of [primaryTargets, fallbackTargets]) {
      if (!targets.size) continue;
      for (const item of state.referenceLayers.values()) {
        for (const record of item.imageRecords || []) {
          if ([...record.matchTokens].some(token => targets.has(token))) return { item, record };
        }
      }
    }
    return null;
  }

  async function ensureLinkedImagesLoaded(feature) {
    if (linkedImageRecordForFeature(feature)) return;
    for (const item of state.referenceLayers.values()) {
      if (!item.loader || !item.imageLayers) continue;
      if (!item.loaded) {
        if (!item.loadingPromise) {
          item.loadingPromise = Promise.resolve(item.loader(item)).then(() => {
            item.loaded = true;
            renderReferenceLayers();
          }).finally(() => {
            item.loading = false;
            item.loadingPromise = null;
          });
        }
        item.loading = true;
        await item.loadingPromise;
      }
      if (linkedImageRecordForFeature(feature)) return;
    }
  }

  function linkedImageIsVisible(match) {
    const image = match?.item?.imageLayers?.get(match.record.imageUrl);
    return Boolean(image && match.item.layer.hasLayer(image) && state.map?.hasLayer(match.item.layer));
  }

  async function toggleLinkedImageForFeature(feature, button) {
    button.disabled = true;
    button.textContent = "Loading image…";
    try {
      await ensureLinkedImagesLoaded(feature);
      const match = linkedImageRecordForFeature(feature);
      if (!match) throw new Error("No linked source image was found for this anomaly.");
      let image = match.item.imageLayers.get(match.record.imageUrl);
      if (!image) {
        image = window.L.imageOverlay(match.record.imageUrl, match.record.bounds, {
          pane: "ppRasterPane",
          opacity: match.item.opacity,
          interactive: false,
        });
        match.item.imageLayers.set(match.record.imageUrl, image);
      }
      if (linkedImageIsVisible(match)) {
        match.item.layer.removeLayer(image);
        button.textContent = "View image";
        byId("ppReferenceStatus").textContent = "Linked anomaly image removed from the map.";
      } else {
        match.item.layer.addLayer(image);
        if (!state.map.hasLayer(match.item.layer)) match.item.layer.addTo(state.map);
        button.textContent = "Remove image";
        byId("ppReferenceStatus").textContent = "Linked anomaly image is visible on the map.";
      }
      renderReferenceLayers();
    } catch (error) {
      button.textContent = "View image";
      byId("ppReferenceStatus").textContent = error.message;
    } finally {
      button.disabled = false;
    }
  }

  function polygonPopupContent(feature, label) {
    const wrapper = document.createElement("div");
    wrapper.innerHTML = polygonPopupHtml(feature, label);
    const properties = feature?.properties || {};
    if (!imageMatchTokens(properties.image, properties.file, properties.src, properties.name, properties.source_images || []).size) {
      return wrapper;
    }
    const actions = document.createElement("div");
    actions.className = "postprocessPolygonPopupActions";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary tiny";
    const match = linkedImageRecordForFeature(feature);
    button.textContent = linkedImageIsVisible(match) ? "Remove image" : "View image";
    button.addEventListener("click", () => void toggleLinkedImageForFeature(feature, button));
    actions.appendChild(button);
    const content = wrapper.firstElementChild || wrapper;
    content.insertBefore(actions, content.querySelector("table"));
    return wrapper;
  }

  async function loadLinkedImages(item, summary, imagesUrl) {
    const geojson = await requestJson(imagesUrl, { cache: "no-store" });
    const overlayByName = new Map();
    for (const url of summary?.assets?.overlays || []) {
      const basename = decodeURIComponent(String(url).split("?")[0].split("/").pop() || "");
      if (basename) overlayByName.set(basename, url);
    }
    let count = 0;
    item.imageRecords = [];
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
      if (imageUrl) {
        item.imageRecords.push({
          imageUrl,
          bounds,
          matchTokens: imageMatchTokens(properties.image, properties.file, properties.src, properties.name),
        });
      }
      item.layer.addLayer(footprint);
      count += 1;
    }
    item.detail = `${count.toLocaleString()} footprints · click a footprint to load its image`;
    item.bounds = item.layer.getBounds?.();
  }

  function labeledReferenceItem(item, sourceLabel) {
    if (!sourceLabel) return item;
    const labeled = {
      ...item,
      label: `${sourceLabel} · ${item.label}`,
    };
    if (item.loader) {
      labeled.loader = async () => {
        if (!item.loaded) await item.loader();
        item.loaded = true;
        labeled.loaded = true;
        labeled.loading = item.loading;
        labeled.detail = item.detail;
        labeled.bounds = item.bounds;
        labeled.geojson = item.geojson;
      };
    }
    return labeled;
  }

  function applyReferenceSource(loaded, { append = false, sourceKey = "", sourceLabel = "" } = {}) {
    const layers = append ? new Map(state.referenceLayers) : new Map();
    for (const [key, item] of loaded.layers) {
      layers.set(sourceKey ? `${sourceKey}:${key}` : key, labeledReferenceItem(item, sourceLabel));
    }
    state.referenceLayers = layers;
    renderReferenceLayers();
    byId("ppReferenceStatus").textContent = layers.size
      ? "Reference sources found. Enable only the imagery needed for editing."
      : loaded.status;
  }

  async function loadReferenceSources(resultId, options = {}) {
    const token = ++state.referenceToken;
    const cached = state.referenceSourceCache.get(resultId);
    if (cached) {
      applyReferenceSource(cached, options);
      return;
    }
    if (!options.append) showListLoading("ppReferenceLayers", "Loading reference layers…");
    byId("ppReferenceStatus").textContent = options.sourceLabel
      ? `Finding ${options.sourceLabel.toLowerCase()} orthophoto and image references…`
      : "Finding linked orthophoto and image references…";
    let pending = state.referenceLoadPromises.get(resultId);
    if (!pending) {
      pending = (async () => {
        const [summaryResult, tilesResult] = await Promise.allSettled([
          requestJson(`/api/session_summary?session=${encodeURIComponent(resultId)}`, { cache: "no-store" }),
          requestJson(`/api/session_tiles?session=${encodeURIComponent(resultId)}`, { cache: "no-store" }),
        ]);
        const summary = summaryResult.status === "fulfilled" ? summaryResult.value : {};
        const tiles = tilesResult.status === "fulfilled" ? tilesResult.value : {};
        const layers = new Map();
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
          layers.set("orthophoto", {
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
          layers.set("images", item);
        }
        const status = layers.size
          ? "Reference sources found. Enable only the imagery needed for editing."
          : "No linked orthophoto or geolocated image references were found.";
        return { layers, status };
      })();
      state.referenceLoadPromises.set(resultId, pending);
    }
    let loaded;
    try {
      loaded = await pending;
      state.referenceSourceCache.set(resultId, loaded);
    } finally {
      if (state.referenceLoadPromises.get(resultId) === pending) state.referenceLoadPromises.delete(resultId);
    }
    if (token !== state.referenceToken) return;
    applyReferenceSource(loaded, options);
  }

  async function loadConfiguredReferenceSources() {
    const sources = state.currentJob?.sources || {};
    const segmentationId = String(sources.segmentation?.result_id || sources.segmentation?.workspace_result_id || "");
    const anomalyId = String(sources.anomaly?.result_id || sources.anomaly?.workspace_result_id || "");
    if (state.mode !== "anomaly") {
      if (segmentationId) await loadReferenceSources(segmentationId);
      return;
    }
    if (anomalyId && anomalyId === segmentationId) {
      await loadReferenceSources(anomalyId, { sourceKey: "shared", sourceLabel: "Shared run" });
      return;
    }
    let appended = false;
    if (anomalyId) {
      await loadReferenceSources(anomalyId, { sourceKey: "anomaly", sourceLabel: "Anomaly run" });
      appended = true;
    }
    if (segmentationId) {
      await loadReferenceSources(segmentationId, {
        append: appended,
        sourceKey: "segmentation",
        sourceLabel: "Segmentation run",
      });
    }
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
      geojson: payload,
    });
    layer.addTo(state.map);
    renderReferenceLayers();
    const bounds = layer.getBounds?.();
    if (bounds?.isValid()) state.map.fitBounds(bounds, { padding: [18, 18], maxZoom: 21 });
    byId("ppReferenceStatus").textContent = "Temporary GeoJSON added in browser memory only.";
  }

  function previewPane(stage) {
    if (stage === "solar_rows" || stage === "segmentation_rows_reference") return "ppRowsPane";
    if (state.mode === "anomaly" && (stage === "source" || stage === "deduplicated" || stage === "associated")) {
      return "ppAnomaliesPane";
    }
    return "ppPanelsPane";
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
    const featureStyle = feature => {
      if (stage === "solar_rows" || stage === "segmentation_rows_reference") {
        return {
          ...baseStyle,
          color: "#ef4444",
          fillColor: "#ef4444",
          weight: 3,
          dashArray: "8 4",
          fillOpacity: 0.035,
        };
      }
      return baseStyle;
    };
    const pane = previewPane(stage);
    const layer = window.L.geoJSON(geojson, {
      pane,
      renderer: window.L.svg({ pane, padding: 0.5 }),
      pmIgnore: !editableStage,
      style: featureStyle,
      onEachFeature: (feature, polygonLayer) => {
        const polygonBaseStyle = featureStyle(feature);
        polygonLayer.options.pmIgnore = !editableStage;
        let hovered = false;
        const resetHover = () => {
          if (!hovered) return;
          hovered = false;
          if (state.previewHoverReset === resetHover) state.previewHoverReset = null;
          if (state.editing?.selectedLayer === polygonLayer) return;
          if (!state.editing) polygonLayer.setStyle(polygonBaseStyle);
          else if (state.editing.stage === stage) {
            polygonLayer.setStyle({ ...polygonBaseStyle, weight: polygonBaseStyle.weight + 1, fillOpacity: 0.28, opacity: 1 });
          }
        };
        polygonLayer.on("mouseover", event => {
          if (state.editing && state.editing.stage !== stage) return;
          if (state.editing?.selectedLayer === event.target) return;
          if (state.previewHoverReset && state.previewHoverReset !== resetHover) state.previewHoverReset();
          hovered = true;
          event.target.setStyle({
            color: "#ffffff",
            weight: polygonBaseStyle.weight + 2,
            fillColor: polygonBaseStyle.fillColor,
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
            .setContent(polygonPopupContent(feature, style.label))
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
    button.title = active ? "Exit fullscreen" : "Open fullscreen";
    button.setAttribute("aria-label", button.title);
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
    if (!state.workflowId || !SHAREABLE_STAGES.has(stage)) return;
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
    byId("ppResult").disabled = locked || Boolean(state.currentJob);
    byId("ppGeojson").disabled = locked || Boolean(state.currentJob) || !byId("ppResult").value;
    byId("ppRefresh").disabled = locked;
    byId("ppSegmentationTab").disabled = locked;
    byId("ppAnomalyTab").disabled = locked;
    byId("ppAnomalyGeojson").disabled = locked || Boolean(state.currentJob) || !byId("ppResult").value;
    byId("ppRegularizeSource").disabled = locked || !state.workflows.some(item => item.outputs?.combined);
    if (locked) {
      byId("ppCombine").disabled = true;
      byId("ppRegularize").disabled = true;
    } else {
      const workflow = state.workflows.find(item => item.id === state.workflowId);
      const hasSelectedSource = Boolean(byId("ppGeojson").value);
      byId("ppCombine").disabled = !hasSelectedSource || (!state.analysis && !workflow?.outputs?.combined);
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
    byId("ppMap").classList.remove("mergeMode");
    byId("ppEditVertices").classList.remove("active");
    byId("ppMovePolygons").classList.remove("active");
    byId("ppRotatePolygons").classList.remove("active");
    byId("ppMergePolygons").classList.remove("active");
    setIconButtonLabel("ppMergePolygons", "Merge polygons");
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
    if (stage !== "source" && !GENERATED_STAGES.has(stage)) return;
    const item = state.previewLayers.get(stage);
    if (!item || !state.workflowId) return;
    if (state.editing?.dirty && !window.confirm("Discard the unsaved edits on the current layer?")) return;
    finishMeasurement();
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
    byId("ppEditLayerName").textContent = item.label;
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

  function enablePolygonMerging() {
    if (!state.editing) return;
    const existingSelection = state.editing.mode === "merge"
      ? [...(state.editing.mergeSelection || [])]
      : [];
    if (existingSelection.length >= 2) {
      const editing = state.editing;
      const polygons = [];
      const firstFeature = JSON.parse(JSON.stringify(existingSelection[0].toGeoJSON()));
      for (const layer of existingSelection) {
        const geometry = layer.toGeoJSON()?.geometry;
        if (geometry?.type === "Polygon") polygons.push(geometry.coordinates);
        else if (geometry?.type === "MultiPolygon") polygons.push(...geometry.coordinates);
        editing.item.layer.removeLayer(layer);
      }
      firstFeature.geometry = { type: "MultiPolygon", coordinates: polygons };
      firstFeature.properties = { ...(firstFeature.properties || {}), manually_merged: true };
      const merged = window.L.geoJSON(firstFeature, {
        pmIgnore: false,
        style: editing.item.baseStyle,
      });
      merged.eachLayer(layer => editing.item.layer.addLayer(layer));
      editing.item.count = editing.item.layer.getLayers().length;
      recordEditState(`${existingSelection.length} polygons merged. You can undo or redo this change.`);
    }
    disableEditingTools();
    state.editing.mode = "merge";
    state.editing.mergeSelection = new Set();
    state.editing.item.layer.eachLayer(layer => {
      const handler = event => {
        if (event.originalEvent) window.L.DomEvent.stopPropagation(event.originalEvent);
        const selected = state.editing.mergeSelection;
        if (selected.has(layer)) {
          selected.delete(layer);
          layer.setStyle?.(state.editing.item.baseStyle);
        } else {
          selected.add(layer);
          layer.setStyle?.({ color: "#ffffff", weight: state.editing.item.baseStyle.weight + 2, fillOpacity: 0.48 });
        }
        setIconButtonLabel("ppMergePolygons", selected.size >= 2 ? `Merge selected (${selected.size})` : "Merge polygons");
        byId("ppEditStatus").textContent = selected.size >= 2
          ? "Click Merge selected to combine the highlighted polygons into one feature."
          : `Merge mode: select at least two polygons (${selected.size} selected).`;
      };
      layer.on("click", handler);
      state.editing.deleteHandlers.push({ layer, handler });
    });
    byId("ppMap").classList.add("mergeMode");
    byId("ppMergePolygons").classList.add("active");
    setIconButtonLabel("ppMergePolygons", "Merge polygons");
    byId("ppEditStatus").textContent = "Merge mode: select at least two polygons, then click Merge polygons again.";
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
    else if (mode === "merge") enablePolygonMerging();
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

  async function saveLayerEdits() {
    if (!state.editing?.dirty) return;
    disableEditingTools();
    const editing = state.editing;
    const button = byId("ppSaveEdits");
    button.disabled = true;
    button.textContent = "Saving…";
      byId("ppEditStatus").textContent = `Validating and updating ${editing.item.label}…`;
    try {
      const resultId = byId("ppResult").value;
      if (editing.stage === "source") {
        const payload = await requestJson(
          `/api/results/${encodeURIComponent(resultId)}/postprocess/source-edits`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              input_path: byId("ppGeojson").value,
              output_name: `${byId("ppOutputName").value.trim() || "combined_solar_panels"}_source`,
              geojson: editing.item.layer.toGeoJSON(),
            }),
          },
        );
        stopEditing(false);
        state.workflowId = payload.id;
        await bindCurrentJobWorkflow(payload.id);
        if (!state.workflows.some(item => item.id === payload.id)) state.workflows.unshift(payload);
        const sourcePath = payload.outputs?.source?.path || payload.input_path;
        if (sourcePath && ![...byId("ppGeojson").options].some(option => option.value === sourcePath)) {
          addOption(byId("ppGeojson"), sourcePath, "Edited source working copy · source");
        }
        byId("ppGeojson").value = sourcePath;
        if (payload.outputs?.source?.url) {
          await loadPreviewLayer("source", payload.outputs.source.url, editing.item.count, "Edited source");
        }
        renderWorkflowList();
        setMessage("Source working copy saved. The configured test-run GeoJSON remains unchanged.", "ok");
        return;
      }
      const payload = await requestJson(
        `/api/results/${encodeURIComponent(resultId)}/postprocess/${encodeURIComponent(state.workflowId)}/${encodeURIComponent(editing.stage)}/edits`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ geojson: editing.item.layer.toGeoJSON() }),
        },
      );
      stopEditing(false);
      applyWorkflow(payload);
      await restoreLatestWorkflow(resultId, byId("ppGeojson").value, state.workflowId);
      setMessage(`${editing.item.label} updated. No additional layer was created.`, "ok");
    } catch (error) {
      byId("ppEditStatus").textContent = error.message;
      setMessage(error.message, "err");
    } finally {
      button.textContent = "Save edits";
      if (state.editing) button.disabled = !state.editing.dirty;
    }
  }

  function layerLabelFields(item) {
    const fields = new Set();
    for (const feature of item.geojson?.features || []) {
      for (const [key, value] of Object.entries(feature?.properties || {})) {
        if (value != null && typeof value !== "object") fields.add(key);
      }
      if (fields.size >= 40) break;
    }
    return [...fields].sort((first, second) => first.localeCompare(second));
  }

  function applyLayerLabels(item, field = "") {
    item.labelField = field;
    item.layer.eachLayer(layer => {
      try { layer.unbindTooltip?.(); } catch (_) {}
      if (!field) return;
      const value = layer.feature?.properties?.[field];
      if (value == null || value === "") return;
      try {
        layer.bindTooltip(String(value), {
          className: "postprocessFeatureLabel",
          direction: "center",
          permanent: true,
          opacity: 0.94,
        });
      } catch (_) {}
    });
  }

  function renderPreviewLayers() {
    const container = byId("ppLayerList");
    removePortalLayerMenus("processing");
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
      const readOnlyReference = key === "segmentation_regularized_reference" || key === "segmentation_rows_reference";
      detail.textContent = `${Number(item.count).toLocaleString()} polygons${key === "source" ? " · Original preserved" : readOnlyReference ? " · Read-only" : ""}`;
      text.append(name, detail);
      const layerMenu = createLayerMenu(item.label, Boolean(state.editing), "processing");
      layerMenu.menu.appendChild(layerMenuButton("Focus", layerMenu.menu, () => {
        const bounds = item.layer.getBounds?.();
        if (bounds?.isValid()) state.map.fitBounds(bounds, { padding: [18, 18], maxZoom: 21 });
      }));
      if (GENERATION_DETAIL_STAGES.has(key)) {
        layerMenu.menu.appendChild(layerMenuButton(
          "Generation details",
          layerMenu.menu,
          () => showGenerationDetails(key, item.label),
        ));
      }
      const labelFields = layerLabelFields(item);
      if (labelFields.length) {
        layerMenu.menu.appendChild(layerMenuButton(
          item.labelControlOpen ? "Hide label options" : "Labels…",
          layerMenu.menu,
          () => {
            item.labelControlOpen = !item.labelControlOpen;
            renderPreviewLayers();
          },
        ));
      }
      const download = document.createElement("a");
      download.href = item.url;
      download.download = "";
      download.title = `Download ${item.label} GeoJSON`;
      download.textContent = "Download";
      download.addEventListener("click", () => { layerMenu.menu.hidden = true; });
      const actions = document.createElement("div");
      actions.className = "postprocessLayerActions";
      if (key === "source" || GENERATED_STAGES.has(key)) {
        layerMenu.menu.appendChild(layerMenuButton(
          isEditing ? "Editing" : key === "source" ? "Edit a working copy" : "Edit this layer",
          layerMenu.menu,
          () => beginEditing(key),
          { disabled: Boolean(state.editing) },
        ));
      }
      if (GENERATED_STAGES.has(key)) {
        const send = layerMenuButton("Link to Map", layerMenu.menu, () => sendLayerToMap(key, send), {
          disabled: Boolean(state.editing),
        });
        layerMenu.menu.appendChild(send);
      }
      layerMenu.menu.appendChild(download);
      actions.appendChild(layerMenu.wrapper);
      row.append(checkbox, swatch, text, actions);
      container.appendChild(row);
      if (item.labelControlOpen && labelFields.length) {
        const control = document.createElement("div");
        control.className = "postprocessLayerLabelControl";
        const label = document.createElement("label");
        label.textContent = "Label field";
        const select = document.createElement("select");
        const off = document.createElement("option");
        off.value = "";
        off.textContent = "Off";
        select.appendChild(off);
        for (const field of labelFields) addOption(select, field, field);
        select.value = item.labelField || "";
        select.addEventListener("change", () => applyLayerLabels(item, select.value));
        label.appendChild(select);
        control.appendChild(label);
        container.appendChild(control);
      }
    }
    for (const [stage, loadingCount] of state.previewLoading) {
      if (!loadingCount) continue;
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

  async function loadPreviewLayer(stage, url, expectedCount = null, label = null, showAlongside = false, version = null, requestedContext = null, visible = true) {
    if (!url) return;
    const context = requestedContext || loadContext();
    if (!isCurrentLoad(context)) return;
    const map = ensurePreviewMap();
    if (!map) {
      byId("ppMapStatus").textContent = "Map preview is unavailable because Leaflet did not load.";
      return;
    }
    const alreadyLoaded = state.previewLayers.get(stage);
    if (alreadyLoaded?.url === url && alreadyLoaded?.version === version) {
      if (visible && showAlongside && !map.hasLayer(alreadyLoaded.layer)) alreadyLoaded.layer.addTo(map);
      renderPreviewLayers();
      return alreadyLoaded;
    }
    const style = PREVIEW_STYLES[stage] || PREVIEW_STYLES.source;
    const renderedKey = `${stage}::${url}::${version || ""}::${label || ""}`;
    const cachedLayer = state.previewLayerCache.get(renderedKey);
    if (cachedLayer) {
      const existing = state.previewLayers.get(stage);
      if (existing && existing !== cachedLayer && map.hasLayer(existing.layer)) map.removeLayer(existing.layer);
      if (visible && !showAlongside) {
        for (const item of state.previewLayers.values()) {
          if (item !== cachedLayer && map.hasLayer(item.layer)) map.removeLayer(item.layer);
        }
      }
      if (visible && !map.hasLayer(cachedLayer.layer)) cachedLayer.layer.addTo(map);
      state.previewLayers.set(stage, cachedLayer);
      renderPreviewLayers();
      if (visible) byId("ppMapStatus").textContent = `${cachedLayer.label} preview reused.`;
      return cachedLayer;
    }
    state.previewLoading.set(stage, (state.previewLoading.get(stage) || 0) + 1);
    renderPreviewLayers();
    if (visible) byId("ppMapStatus").textContent = `Loading ${style.label.toLowerCase()} polygons…`;
    try {
      const dataKey = `${url}::${version || ""}`;
      let geojson = state.previewDataCache.get(dataKey);
      if (!geojson) {
        let pending = state.previewDataPromises.get(dataKey);
        if (!pending || pending.signal?.aborted) {
          const persistentSignal = context.persistent ? null : context.signal;
          pending = {
            signal: persistentSignal,
            promise: requestJson(url, {
              cache: "no-store",
              ...(persistentSignal ? { signal: persistentSignal } : {}),
            }),
          };
          state.previewDataPromises.set(dataKey, pending);
        }
        try {
          geojson = await pending.promise;
          state.previewDataCache.set(dataKey, geojson);
        } finally {
          if (state.previewDataPromises.get(dataKey) === pending) state.previewDataPromises.delete(dataKey);
        }
      }
      if (!isCurrentLoad(context)) return;
      const existing = state.previewLayers.get(stage);
      if (existing && map.hasLayer(existing.layer)) map.removeLayer(existing.layer);
      const created = createPreviewGeoJsonLayer(stage, geojson, label);
      const layer = created.layer;
      // Show the newest stage by itself initially. Users can turn earlier stages
      // back on for comparison using the layer list.
      if (visible && !showAlongside) {
        for (const item of state.previewLayers.values()) {
          if (map.hasLayer(item.layer)) map.removeLayer(item.layer);
        }
      }
      if (visible) layer.addTo(map);
      const item = {
        label: created.label,
        color: style.color,
        count: expectedCount ?? geojson.features?.length ?? 0,
        layer,
        url,
        version,
        geojson,
        baseStyle: created.baseStyle,
      };
      state.previewLayers.set(stage, item);
      state.previewLayerCache.set(renderedKey, item);
      renderPreviewLayers();
      if (visible) {
        const bounds = layer.getBounds();
        if (bounds.isValid()) map.fitBounds(bounds, { padding: [18, 18], maxZoom: 21 });
        byId("ppMapStatus").textContent = `${style.label} preview loaded.`;
      }
      return item;
    } catch (error) {
      if (!isCurrentLoad(context)) return;
      if (visible) byId("ppMapStatus").textContent = `Could not load ${style.label.toLowerCase()} preview: ${error.message}`;
    } finally {
      const remaining = (state.previewLoading.get(stage) || 1) - 1;
      if (remaining > 0) state.previewLoading.set(stage, remaining);
      else state.previewLoading.delete(stage);
      if (isCurrentLoad(context)) renderPreviewLayers();
    }
  }

  async function syncOutputPreviews(status, requestedContext = null) {
    const removableStages = [
      "combined", "regularized", "solar_rows", "deduplicated", "associated",
      "panel_hierarchy", "panel_rows", "identified_panels", "edited",
    ];
    for (const stage of removableStages) {
      if (status.outputs?.[stage] || !state.previewLayers.has(stage)) continue;
      const stale = state.previewLayers.get(stage);
      if (state.map?.hasLayer(stale.layer)) state.map.removeLayer(stale.layer);
      state.previewLayers.delete(stage);
    }
    renderPreviewLayers();
    const counts = {
      combined: status.manual_edits?.combined?.feature_count ?? status.combine_stats?.output_features,
      regularized: status.manual_edits?.regularized?.feature_count ?? status.hierarchy_stats?.panel_count ?? status.regularize_stats?.output_features,
      solar_rows: status.manual_edits?.solar_rows?.feature_count ?? status.hierarchy_stats?.row_count,
      deduplicated: status.manual_edits?.deduplicated?.feature_count
        ?? status.deduplicate_stats?.output_features
        ?? status.overlap_deduplicate_stats?.output_features,
      associated: status.manual_edits?.associated?.feature_count ?? status.association_stats?.output_features,
    };
    const availableStages = [...GENERATED_STAGES].filter(stage => status.outputs?.[stage]?.url);
    const preferredStage = state.mode === "anomaly"
      ? (availableStages.includes("associated") ? "associated" : "deduplicated")
      : (availableStages.includes("regularized") ? "regularized" : availableStages.includes("combined") ? "combined" : "solar_rows");
    const immediateStages = preferredStage && availableStages.includes(preferredStage) ? [preferredStage] : [];
    const deferredStages = availableStages.filter(stage => !immediateStages.includes(stage));
    const loadStages = async (stages, visible) => {
      for (const stage of stages) {
        if (!isCurrentLoad(requestedContext)) return;
        const output = status.outputs[stage];
        const version = output.mtime
          || status.manual_edits?.[stage]?.updated_at
          || (stage === "regularized" && status.hierarchy_stats ? status.updated_at : null);
        await loadPreviewLayer(stage, output.url, counts[stage], null, false, version, requestedContext, visible);
        if (!visible) await new Promise(resolve => window.requestAnimationFrame(resolve));
      }
    };
    await loadStages(immediateStages, true);
    state.deferredProcessingPromise = new Promise(resolve => window.setTimeout(resolve, 50))
      .then(() => loadStages(deferredStages, false));
  }

  function whenProcessingLayersReady() {
    return state.deferredProcessingPromise || Promise.resolve();
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
    byId("ppRegularize").disabled = !selected || ["queued", "running"].includes(selected.status);
  }

  function renderWorkflowList() {
    populateRegularizeSources(state.workflowId);
  }

  function addOption(select, value, label) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  }

  function setResultLoading(select, loading) {
    select.toggleAttribute("aria-busy", loading);
    byId("ppResultSpinner").hidden = !loading;
  }

  async function loadResults(force = false) {
    if (state.loaded && !force) return;
    if (state.loadingPromise) return state.loadingPromise;
    const select = byId("ppResult");
    const previous = select.value;
    select.replaceChildren();
    addOption(select, "", "Loading test results…");
    select.disabled = true;
    setResultLoading(select, true);
    if (state.currentJob) setMessage("");
    state.loadingPromise = (async () => {
      try {
        const payload = await requestJson("/api/sessions", { cache: "no-store" });
        setResultLoading(select, false);
        select.replaceChildren();
        addOption(select, "", "Select a result…");
        for (const result of payload.sessions || []) {
          const name = result.display_name || result.name;
          const status = result.status === "complete" ? "Complete" : "Incomplete";
          addOption(select, result.id || result.name, `${name} · ${status} · ID: ${result.id || result.name}`);
        }
        const desiredResultId = state.currentJob?.sources?.[state.mode]?.result_id || previous;
        if (desiredResultId && [...select.options].some(option => option.value === desiredResultId)) {
          select.value = desiredResultId;
        }
        select.disabled = Boolean(state.currentJob) || select.options.length <= 1;
        state.loaded = true;
        if (select.value) await loadGeojsons();
      } catch (error) {
        setResultLoading(select, false);
        select.replaceChildren();
        addOption(select, "", "Could not load test results");
        select.disabled = true;
        setMessage(error.message, "err");
      }
    })().finally(() => {
      setResultLoading(select, false);
      state.loadingPromise = null;
    });
    return state.loadingPromise;
  }

  async function restoreLatestWorkflow(resultId, inputPath, preferredId = null, requestedContext = loadContext()) {
    if (!resultId) return false;
    try {
      if (state.currentJob) {
        const workflowId = preferredId || state.currentJob.workflows?.[state.mode]?.workflow_id;
        if (!workflowId) {
          state.workflows = [];
          state.workflowId = null;
          renderWorkflowList();
          return false;
        }
        const workflow = await requestJson(
          `/api/results/${encodeURIComponent(resultId)}/postprocess/${encodeURIComponent(workflowId)}`,
          { cache: "no-store", signal: requestedContext?.signal },
        );
        if (!isCurrentLoad(requestedContext)) return false;
        state.workflows = [workflow];
        state.workflowId = workflow.id;
        await applyWorkflow(workflow, requestedContext);
        populateRegularizeSources(workflow.id);
        renderWorkflowList();
        return true;
      }
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess`, {
        cache: "no-store",
        signal: requestedContext?.signal,
      });
      if (!isCurrentLoad(requestedContext)) return false;
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
      applyWorkflow(workflow, requestedContext);
      populateRegularizeSources(workflow.id);
      renderWorkflowList();
      if (workflow.source_changed) {
        setMessage("The source GeoJSON changed after this output was created. Existing derived layers are preserved; review them before creating an updated version.", "warn");
      }
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
      || visible.find(item => sourceIdentity(item.input_path) === sourceIdentity(inputPath) && (item.outputs?.source || item.outputs?.combined || item.outputs?.deduplicated))
      || (allowLatest ? visible.find(item => Object.keys(item.outputs || {}).length) : null);
  }

  async function loadSavedOutputsFirst(resultId, requestedContext = null) {
    if (!isCurrentLoad(requestedContext)) return null;
    showListLoading("ppLayerList", "Loading output layers…");
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess`, {
        cache: "no-store",
        signal: requestedContext?.signal,
      });
      if (!isCurrentLoad(requestedContext)) return null;
      state.workflows = (payload.workflows || []).sort((first, second) =>
        String(second.created_at || "").localeCompare(String(first.created_at || ""))
      );
      const workflow = selectSavedWorkflow("", null, true);
      if (workflow) {
        state.workflowId = workflow.id;
        applyWorkflow(workflow, requestedContext, { loadPreviews: false });
        populateRegularizeSources(workflow.id);
      } else {
        renderPreviewLayers();
      }
      renderWorkflowList();
      return workflow;
    } catch (error) {
      if (!isCurrentLoad(requestedContext)) return null;
      state.workflows = [];
      renderWorkflowList();
      renderPreviewLayers();
      byId("ppMapStatus").textContent = `Could not load saved outputs: ${error.message}`;
      return null;
    }
  }

  async function loadGeojsons() {
    invalidateModeLoad();
    state.deferredProcessingPromise = Promise.resolve();
    const requestedContext = loadContext();
    if (state.currentJob) setStepsLoading(true);
    resetAnalysis({ showConfiguredStep: true });
    clearReferenceLayers();
    state.workflows = [];
    const resultId = byId("ppResult").value;
    const select = byId("ppGeojson");
    select.disabled = true;
    select.replaceChildren();
    addOption(select, "", resultId ? "Looking for GeoJSON files…" : "Select a GeoJSON…");
    if (!resultId) {
      setMessage("Select a test result.");
      byId("ppReferenceStatus").textContent = "Select a test result to find linked imagery.";
      return;
    }
    setMessage("");
    // Both modes use the segmentation run's imagery as their common spatial
    // reference, even when anomaly predictions come from a different run.
    const referenceResultId = state.currentJob?.sources?.segmentation?.result_id || resultId;
    if (state.currentJob) {
      await loadConfiguredJobMode(requestedContext);
      return;
    }
    try {
      const [earlyWorkflow, payload] = await Promise.all([
        loadSavedOutputsFirst(resultId, requestedContext),
        requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/geojsons`, { signal: requestedContext.signal }),
      ]);
      if (!isCurrentLoad(requestedContext)) return;
      state.geojsonFiles = payload.files || [];
      select.replaceChildren();
      addOption(select, "", "Select a GeoJSON…");
      for (const file of state.geojsonFiles) {
        addOption(select, file.path, `${file.name} · ${file.stage}`);
      }
      const previousSource = earlyWorkflow
        ? [...select.options].find(option => sourceIdentity(option.value) === sourceIdentity(earlyWorkflow.input_path))
        : null;
      const configuredModePath = state.currentJob?.sources?.[state.mode]?.path || "";
      const configuredSource = configuredModePath
        ? [...select.options].find(option => sourceIdentity(option.value) === sourceIdentity(configuredModePath))
        : null;
      const configuredWorkingCopy = state.mode === "segmentation" && earlyWorkflow?.outputs?.source
        && sourceIdentity(earlyWorkflow.original_input_path) === sourceIdentity(configuredModePath)
        ? previousSource
        : null;
      const predictions = [...select.options].find(option => option.value.toLowerCase() === "predictions.geojson")
        || [...select.options].find(option => option.value.toLowerCase() === "anomalies.geojson");
      if (configuredWorkingCopy) select.value = configuredWorkingCopy.value;
      else if (configuredSource) select.value = configuredSource.value;
      else if (previousSource) select.value = previousSource.value;
      else if (predictions) select.value = predictions.value;
      else if (select.options.length > 1) select.selectedIndex = 1;
      select.disabled = Boolean(state.currentJob) || select.options.length <= 1;
      let deferredSource = null;
      if (state.currentJob && state.mode === "segmentation") {
        await initializeConfiguredSegmentation({ loadSource: !earlyWorkflow });
        if (earlyWorkflow) {
          const source = state.geojsonFiles.find(file => file.path === select.value);
          if (source?.url) deferredSource = { source, count: state.currentJob.sources?.segmentation?.summary?.feature_count, label: null };
        }
      } else if (state.currentJob && state.mode === "anomaly") {
        const configuredAnomaly = state.currentJob.sources?.anomaly;
        const source = state.geojsonFiles.find(file => file.path === select.value);
        if (source?.url && !earlyWorkflow) {
          await loadPreviewLayer(
            "source",
            source.url,
            configuredAnomaly?.summary?.feature_count,
            "Anomaly predictions",
            false,
            source.mtime,
          );
        } else if (source?.url) {
          deferredSource = { source, count: configuredAnomaly?.summary?.feature_count, label: "Anomaly predictions" };
        }
      }
      const matchingWorkflow = selectSavedWorkflow(select.value, null, false);
      const restored = Boolean(matchingWorkflow);
      if (matchingWorkflow) {
        state.workflowId = matchingWorkflow.id;
        await applyWorkflow(matchingWorkflow, requestedContext);
        populateRegularizeSources(matchingWorkflow.id);
        renderWorkflowList();
      }
      if (deferredSource) {
        const processingReady = whenProcessingLayersReady();
        state.deferredProcessingPromise = processingReady.then(() => loadPreviewLayer(
          "source",
          deferredSource.source.url,
          deferredSource.count,
          deferredSource.label,
          false,
          deferredSource.source.mtime,
          requestedContext,
          false,
        ));
      }
      if (!restored) {
        if (!state.currentJob) {
          setMessage(select.value ? "Select a source to continue." : "This result has no GeoJSON files.", select.value ? "" : "warn");
        } else if (!select.value) {
          setMessage("The configured segmentation GeoJSON is unavailable. Edit the job configuration.", "err");
        }
      }
      document.dispatchEvent(new CustomEvent("postprocess:data", { detail: getContext() }));
    } catch (error) {
      if (!isCurrentLoad(requestedContext)) return;
      setMessage(error.message, "err");
    } finally {
      if (!isCurrentLoad(requestedContext)) return;
      stopIndeterminate();
      setStepsLoading(false);
      void whenProcessingLayersReady().finally(() => {
        if (!isCurrentLoad(requestedContext)) return;
        void loadReferenceSources(referenceResultId);
      });
    }
  }

  async function loadConfiguredJobMode(requestedContext) {
    const sourceConfig = state.currentJob?.sources?.[state.mode] || {};
    const resultId = String(sourceConfig.workspace_result_id || "");
    const sourcePath = String(sourceConfig.workspace_path || "source.geojson");
    const sourceUrl = String(sourceConfig.workspace_url || "");
    const select = byId("ppGeojson");
    try {
      if (!resultId || !sourceUrl) {
        throw new Error("This job has no saved source snapshot. Recreate the job or save its configuration again.");
      }
      const sourceFile = {
        name: sourcePath.split("/").pop() || "source.geojson",
        path: sourcePath,
        stage: state.mode === "anomaly" ? "anomalies" : "segmentation",
        url: sourceUrl,
        mtime: sourceConfig.workspace_mtime || "",
      };
      state.geojsonFiles = [sourceFile];
      select.replaceChildren();
      addOption(select, sourcePath, sourceFile.name);
      select.value = sourcePath;
      select.disabled = true;
      const binding = state.currentJob.workflows?.[state.mode];
      let workflow = null;
      if (binding?.workflow_id) {
        workflow = await loadBoundWorkflowStatus(resultId, binding.workflow_id);
        if (!isCurrentLoad(requestedContext)) return;
      }
      state.workflows = workflow ? [workflow] : [];
      state.workflowId = workflow?.id || null;
      if (state.mode === "segmentation") {
        await initializeConfiguredSegmentation({ loadSource: !workflow });
      } else if (!workflow) {
        await loadPreviewLayer(
          "source", sourceUrl, sourceConfig.summary?.feature_count,
          "Anomaly predictions", false, sourceConfig.workspace_mtime,
          requestedContext,
        );
      }
      if (workflow) {
        await applyWorkflow(workflow, requestedContext);
        populateRegularizeSources(workflow.id);
        renderWorkflowList();
        const processingReady = whenProcessingLayersReady();
        state.deferredProcessingPromise = processingReady.then(() => loadPreviewLayer(
          "source", sourceUrl, sourceConfig.summary?.feature_count,
          state.mode === "anomaly" ? "Anomaly predictions" : null,
          false, sourceConfig.workspace_mtime, requestedContext, false,
        ));
      } else {
        renderWorkflowList();
      }
      document.dispatchEvent(new CustomEvent("postprocess:data", { detail: getContext() }));
    } catch (error) {
      if (isCurrentLoad(requestedContext)) setMessage(error.message, "err");
    } finally {
      if (!isCurrentLoad(requestedContext)) return;
      stopIndeterminate();
      setStepsLoading(false);
      void whenProcessingLayersReady().finally(() => {
        if (!isCurrentLoad(requestedContext)) return;
        void loadConfiguredReferenceSources();
      });
    }
  }

  function workflowStatusKey(resultId, workflowId) {
    return `${resultId}::${workflowId}`;
  }

  async function loadBoundWorkflowStatus(resultId, workflowId) {
    const key = workflowStatusKey(resultId, workflowId);
    if (state.workflowStatusCache.has(key)) return state.workflowStatusCache.get(key);
    let pending = state.workflowStatusPromises.get(key);
    if (!pending) {
      pending = requestJson(
        `/api/results/${encodeURIComponent(resultId)}/postprocess/${encodeURIComponent(workflowId)}`,
        { cache: "no-store" },
      ).then(status => {
        state.workflowStatusCache.set(key, status);
        return status;
      }).finally(() => {
        state.workflowStatusPromises.delete(key);
      });
      state.workflowStatusPromises.set(key, pending);
    }
    return pending;
  }

  async function initializeConfiguredSegmentation({ loadSource = true } = {}) {
    const configured = state.currentJob?.sources?.segmentation;
    const summary = configured?.summary;
    const sourcePath = byId("ppGeojson").value;
    if (!configured || !summary || !sourcePath) return false;
    state.analysis = summary;
    state.scanComplete = true;
    syncSegmentationStepProgress();
    const source = state.geojsonFiles.find(file => file.path === sourcePath);
    if (loadSource && source?.url) await loadPreviewLayer("source", source.url, summary.feature_count, null, false, source.mtime);
    const canCombine = Boolean(summary.tile_metadata_available && summary.features_on_tile_edges);
    byId("ppCombine").disabled = !canCombine;
    setMessage("");
    return true;
  }

  function showProgress(progress, message) {
    const value = Math.max(0, Math.min(100, Number(progress) || 0));
    byId("ppProgressWrap").hidden = false;
    byId("ppProgressWrap").classList.remove("ppIndeterminate");
    byId("ppProgressBar").style.width = `${value}%`;
    byId("ppProgressText").textContent = `${message || "Processing…"} ${Math.round(value)}%`;
  }

  function applyWorkflow(status, requestedContext = null, { loadPreviews = true } = {}) {
    if (!isCurrentLoad(requestedContext)) return Promise.resolve();
    if (status?.id) {
      const workflowIndex = state.workflows.findIndex(item => item.id === status.id);
      if (workflowIndex >= 0) {
        state.workflows[workflowIndex] = { ...state.workflows[workflowIndex], ...status };
        status = state.workflows[workflowIndex];
      } else {
        state.workflows.unshift(status);
      }
    }
    showProgress(status.progress, status.message);
    if (Array.isArray(status.log)) {
      byId("ppLog").textContent = status.log.join("\n");
      byId("ppLogWrap").hidden = !status.log.length;
    }
    const previewPromise = loadPreviews
      ? syncOutputPreviews(status, requestedContext)
      : Promise.resolve();
    if (state.mode === "segmentation") syncSegmentationStepProgress(status);
    const running = status.status === "queued" || status.status === "running";
    const hasCombined = Boolean(status.outputs?.combined);
    const hasSelectedSource = Boolean(byId("ppGeojson").value);
    byId("ppCombine").disabled = running || !hasSelectedSource || (!state.analysis && !hasCombined);
    byId("ppRegularize").disabled = running || !hasCombined;
    if (status.status === "failed") setMessage(status.error || status.message || "Post-processing failed.", "err");
    else if (status.status === "complete") setMessage(status.message || "Post-processing complete.", "ok");
    else setMessage(status.message || "Post-processing…");
    document.dispatchEvent(new CustomEvent("postprocess:workflow", { detail: { status, context: getContext() } }));
    return previewPromise;
  }

  async function pollWorkflow(token, requestedContext, resultId, workflowId) {
    let consecutiveErrors = 0;
    while (token === state.pollToken && isCurrentLoad(requestedContext)) {
      await new Promise(resolve => setTimeout(resolve, 900));
      if (token !== state.pollToken || !isCurrentLoad(requestedContext)) return;
      try {
        const status = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/${encodeURIComponent(workflowId)}`);
        state.workflowStatusCache.set(workflowStatusKey(resultId, workflowId), status);
        consecutiveErrors = 0;
        await applyWorkflow(status, requestedContext);
        if (status.status !== "queued" && status.status !== "running") {
          await restoreLatestWorkflow(resultId, byId("ppGeojson").value, workflowId, requestedContext);
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
    let existingWorkflow = state.workflows.find(item =>
      item.workflow_kind !== "anomaly"
      && sourceIdentity(item.input_path) === sourceIdentity(inputPath)
      && item.outputs?.combined
    );
    if (existingWorkflow) {
      setMessage("Checking the existing Combined GeoJSON before replacement…");
      try {
        existingWorkflow = await requestJson(
          `/api/results/${encodeURIComponent(resultId)}/postprocess/${encodeURIComponent(existingWorkflow.id)}`,
          { cache: "no-store" },
        );
      } catch (error) {
        byId("ppCombine").disabled = false;
        setMessage(`Could not check the existing combined output: ${error.message}`, "err");
        return;
      }
      const confirmed = await confirmReplacement(
        "Replace combined fragments?",
        "The current Combined GeoJSON will be replaced by the newly generated file. Regularized and Rows outputs derived from it will also be removed and must be generated again. Manual changes in those replaced files will be lost.",
      );
      if (!confirmed) {
        byId("ppCombine").disabled = false;
        byId("ppRegularize").disabled = !existingWorkflow.outputs?.combined;
        return;
      }
    }
    showProgress(0, "Starting fragment combining…");
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/combine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_path: inputPath,
          workflow_id: existingWorkflow?.id || null,
          output_name: byId("ppOutputName").value.trim() || "combined_solar_panels",
          edge_tolerance_px: numberValue("ppEdgeTolerance"),
          gap_tolerance_px: numberValue("ppGapTolerance"),
          min_boundary_overlap: numberValue("ppBoundaryOverlap"),
          remove_contained_polygons: byId("ppRemoveContained").checked,
        }),
      });
      await runWorkflow(payload);
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
    let workflow = state.workflows.find(item => item.id === selectedWorkflow);
    setMessage("Checking the existing Regularized GeoJSON before replacement…");
    try {
      workflow = await requestJson(
        `/api/results/${encodeURIComponent(resultId)}/postprocess/${encodeURIComponent(selectedWorkflow)}`,
        { cache: "no-store" },
      );
    } catch (error) {
      setMessage(`Could not check the existing regularized output: ${error.message}`, "err");
      return;
    }
    if (workflow?.outputs?.regularized) {
      const confirmed = await confirmReplacement(
        "Replace regularized polygons?",
        "The current Regularized GeoJSON will be replaced by the newly generated file. Its Rows output will also be removed and must be generated again. Manual changes in those replaced files will be lost.",
      );
      if (!confirmed) return;
    }
    byId("ppRegularize").disabled = true;
    showProgress(0, "Starting polygon regularization…");
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/${encodeURIComponent(state.workflowId)}/regularize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_area_change_percent: numberValue("ppAreaChange") }),
      });
      await runWorkflow(payload);
    } catch (error) {
      byId("ppRegularize").disabled = false;
      setMessage(error.message, "err");
    }
  }

  function init() {
    if (state.initialized) return;
    state.initialized = true;
    try { window.L?.PM?.setOptIn(true); } catch (_) {}
    initCollapsibleSteps();
    initSourceStepToggles();
    byId("ppReplacementConfirm")?.addEventListener("click", () => finishReplacementConfirmation(true));
    byId("ppReplacementCancel")?.addEventListener("click", () => finishReplacementConfirmation(false));
    byId("ppReplacementClose")?.addEventListener("click", () => finishReplacementConfirmation(false));
    byId("ppReplacementModal")?.addEventListener("click", event => {
      if (event.target === byId("ppReplacementModal")) finishReplacementConfirmation(false);
    });
    byId("ppGenerationDetailsClose")?.addEventListener("click", closeGenerationDetails);
    byId("ppGenerationDetailsDone")?.addEventListener("click", closeGenerationDetails);
    byId("ppGenerationDetailsModal")?.addEventListener("click", event => {
      if (event.target === byId("ppGenerationDetailsModal")) closeGenerationDetails();
    });
    // Keep activation self-contained. This also works if an older cached copy of
    // the shared tab controller is still present in the browser.
    byId("ppRefresh").addEventListener("click", () => void refreshCurrentJob());
    byId("ppBackToJobs")?.addEventListener("click", showJobLanding);
    byId("ppEditJobConfig")?.addEventListener("click", () => {
      if (state.currentJob) void openJobConfiguration(state.currentJob);
    });
    byId("ppJobSearch")?.addEventListener("input", renderJobList);
    byId("ppResult").addEventListener("change", loadGeojsons);
    byId("ppGeojson").addEventListener("change", async () => {
      resetAnalysis();
      setMessage(byId("ppGeojson").value ? "Configured segmentation source selected." : "No configured segmentation source is available.");
      await restoreLatestWorkflow(byId("ppResult").value, byId("ppGeojson").value);
    });
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
    byId("ppAnomalyMapPairFilter").addEventListener("change", renderAllAnomalyReviewPairs);
    byId("ppReturnToComparison").addEventListener("click", () => {
      document.dispatchEvent(new CustomEvent("postprocess:return-comparisons", {
        detail: { pairIndex: state.anomalyReviewSelectedIndex },
      }));
    });
    byId("ppClearAnomalyPair").addEventListener("click", clearAnomalyReviewMap);
    byId("ppEditVertices").addEventListener("click", enableVertexEditing);
    byId("ppMovePolygons").addEventListener("click", enablePolygonMovement);
    byId("ppRotatePolygons").addEventListener("click", enablePolygonRotation);
    byId("ppMergePolygons").addEventListener("click", enablePolygonMerging);
    byId("ppDeletePolygons").addEventListener("click", enablePolygonDeletion);
    byId("ppUndoEdits").addEventListener("click", undoEdits);
    byId("ppRedoEdits").addEventListener("click", redoEdits);
    byId("ppSaveEdits").addEventListener("click", saveLayerEdits);
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
    document.addEventListener("click", () => closeLayerMenus());
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
    if (state.currentJobId && state.currentJob) return;
    await loadJobs(false);
  }

  async function loadJobs(force = false) {
    const landing = byId("ppJobLanding");
    const workspace = document.querySelector(".postprocessWorkspace");
    const list = byId("ppJobList");
    if (!landing || !workspace || !list) return loadResults(false);
    landing.hidden = false;
    landing.style.display = "block";
    workspace.hidden = true;
    workspace.style.display = "none";
    byId("ppBackToJobs").hidden = true;
    byId("ppHeaderTitle").textContent = "Post-processing";
    byId("ppHeaderJobId").hidden = true;
    byId("ppHeaderDescription").hidden = false;
    byId("ppRefresh").hidden = true;
    byId("ppEditJobConfig").hidden = true;
    if (state.jobsLoaded && !force) {
      renderJobList();
      return;
    }
    if (state.jobsLoadingPromise) return state.jobsLoadingPromise;
    list.replaceChildren();
    const loading = document.createElement("div");
    loading.className = "mapListLoading";
    loading.innerHTML = '<span class="spinner" aria-hidden="true"></span><span>Loading jobs…</span>';
    list.appendChild(loading);
    state.jobsLoadingPromise = (async () => {
      try {
        const payload = await requestJson("/api/postprocess-jobs", { cache: "no-store" });
        state.jobs = payload.jobs || [];
        state.jobsLoaded = true;
        renderJobList();
      } catch (error) {
        state.jobs = [];
        state.jobsLoaded = false;
        list.replaceChildren();
        list.textContent = error.message;
      } finally {
        state.jobsLoadingPromise = null;
      }
    })();
    return state.jobsLoadingPromise;
  }

  function renderJobList() {
    const list = byId("ppJobList");
    if (!list) return;
    const search = byId("ppJobSearch")?.value.trim().toLocaleLowerCase() || "";
    const jobs = state.jobs.filter(job => {
      if (!search) return true;
      return `${job.name || ""} ${job.id || ""}`.toLocaleLowerCase().includes(search);
    });
    list.replaceChildren();
    if (!jobs.length) {
        const empty = document.createElement("div");
        empty.className = "muted tiny";
        empty.textContent = state.jobs.length ? "No jobs match your search." : "No post-processing jobs yet. Create one to begin.";
        list.appendChild(empty);
    }
    for (const job of jobs) renderJobItem(list, job);
  }

  function renderJobItem(container, job) {
    const item = document.createElement("div");
    item.className = "postprocessWorkflowItem";
    const info = document.createElement("div");
    info.className = "postprocessWorkflowInfo";
    const icon = document.createElement("span");
    icon.className = "postprocessJobListIcon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "📁";
    const text = document.createElement("div");
    text.className = "postprocessJobListText";
    const name = document.createElement("strong");
    name.textContent = job.name || job.id;
    const id = document.createElement("small");
    id.className = "postprocessJobListId";
    id.textContent = `ID: ${job.id}`;
    const segmentationSource = document.createElement("small");
    segmentationSource.className = "postprocessJobListSource";
    segmentationSource.textContent = job.sources?.segmentation?.result_id
      ? `Segmentation: ${job.sources.segmentation.result_id} · ${job.sources.segmentation.path}`
      : "Segmentation source not configured";
    const anomalySource = document.createElement("small");
    anomalySource.className = "postprocessJobListSource";
    anomalySource.textContent = job.sources?.anomaly?.result_id
      ? `Anomalies: ${job.sources.anomaly.result_id} · ${job.sources.anomaly.path}`
      : "Anomaly source not configured";
    const details = document.createElement("small");
    details.className = "postprocessJobListDates";
    const formatTime = value => value ? new Date(value).toLocaleString() : "Not recorded";
    details.textContent = `Created ${formatTime(job.created_at)} · Modified ${formatTime(job.updated_at)}`;
    text.append(name, id, segmentationSource, anomalySource, details);
    info.append(icon, text);
    info.addEventListener("click", () => openJob(job));
    const menu = document.createElement("button");
    menu.type = "button";
    menu.className = "iconDots";
    menu.textContent = "⋮";
    menu.title = "Job options";
    const actions = document.createElement("div");
    actions.className = "postprocessWorkflowMenu";
    actions.hidden = true;
    const rename = document.createElement("button");
    rename.type = "button";
    rename.textContent = "Rename";
    rename.addEventListener("click", async () => {
        actions.hidden = true;
        const renamed = await showRenameJobModal(job.name || job.id);
        if (!renamed) return;
        const payload = await requestJson(`/api/postprocess-jobs/${encodeURIComponent(job.id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: renamed }) });
        state.jobs = state.jobs.map(item => item.id === job.id ? payload.job : item);
        state.jobsLoaded = true;
        renderJobList();
    });
    const config = document.createElement("button");
    config.type = "button";
    config.textContent = "Edit configuration";
    config.addEventListener("click", () => {
      actions.hidden = true;
      void openJobConfiguration(job);
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger";
    remove.textContent = "Delete";
    remove.addEventListener("click", async () => {
        actions.hidden = true;
        if (!await showDeleteJobModal(job.name || job.id)) return;
        await requestJson(`/api/postprocess-jobs/${encodeURIComponent(job.id)}`, { method: "DELETE" });
        state.jobs = state.jobs.filter(item => item.id !== job.id);
        state.jobsLoaded = true;
        renderJobList();
    });
    actions.append(rename, config, remove);
    menu.addEventListener("click", event => {
      event.stopPropagation();
      document.querySelectorAll("#ppJobList .postprocessWorkflowMenu").forEach(other => {
        if (other !== actions) other.hidden = true;
      });
      actions.hidden = !actions.hidden;
    });
    item.append(info, menu, actions);
    container.appendChild(item);
  }

  function closeJobModal() {
    const modal = byId("ppJobModal");
    modal.classList.remove("show");
    modal.classList.add("hidden");
  }

  function lockJobCreationFields(locked) {
    byId("ppJobName").disabled = locked;
    byId("ppJobSourceFields")?.querySelectorAll("select, button").forEach(control => {
      control.disabled = locked;
    });
  }

  function showJobCreationProgress(action) {
    const modal = byId("ppJobModal");
    const message = byId("ppJobModalMessage");
    byId("ppJobModalTitle").textContent = "Creating post-processing job";
    message.replaceChildren();
    message.className = "postprocessJobCreationProgress";
    const heading = document.createElement("div");
    heading.className = "postprocessJobCreationStatus";
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    spinner.setAttribute("aria-hidden", "true");
    const status = document.createElement("strong");
    status.textContent = `Creating ${action.name}…`;
    heading.append(spinner, status);
    const details = document.createElement("div");
    details.className = "postprocessJobCreationDetails muted tiny";
    const segmentation = document.createElement("span");
    segmentation.textContent = `Segmentation · ${action.sourceResultId} · ${action.sourcePath}`;
    const anomaly = document.createElement("span");
    anomaly.textContent = `Anomalies · ${action.anomalyResultId} · ${action.anomalyPath}`;
    const note = document.createElement("span");
    note.textContent = "Copying only the selected segmentation and anomaly GeoJSONs into the job. Images and tile assets remain linked read-only from their test results, which cannot be deleted while this job uses them.";
    details.append(segmentation, anomaly, note);
    message.append(heading, details);
    byId("ppJobSourceFields").after(message);
    byId("ppJobNameField").hidden = false;
    byId("ppJobSourceFields").hidden = false;
    lockJobCreationFields(true);
    byId("ppJobModalCancel").disabled = true;
    byId("ppJobModalSave").disabled = true;
    byId("ppJobModalSave").textContent = "Creating job…";
    byId("ppJobModalClose").disabled = true;
    modal.classList.remove("hidden");
    modal.classList.add("show");
    window.requestAnimationFrame(() => {
      const body = modal.querySelector(".modalBody");
      body?.scrollTo({ top: body.scrollHeight, behavior: "smooth" });
    });
  }

  function showJobCreationFailure(error) {
    const message = byId("ppJobModalMessage");
    byId("ppJobModalTitle").textContent = "Could not create job";
    message.replaceChildren();
    message.className = "statusLine err postprocessJobCreationError";
    message.textContent = error.message || String(error);
    const cancel = byId("ppJobModalCancel");
    const save = byId("ppJobModalSave");
    const close = byId("ppJobModalClose");
    cancel.hidden = true;
    save.disabled = false;
    save.textContent = "Close";
    save.onclick = closeJobModal;
    close.disabled = false;
    close.onclick = closeJobModal;
  }

  async function createJob() {
    const action = await showJobModal("create", "Post-processing job");
    if (!action?.name) return;
    showJobCreationProgress(action);
    const matchesCreatedJob = job => (job.name === action.name || job.name?.startsWith(`${action.name} (`))
      && job.sources?.segmentation?.result_id === action.sourceResultId
      && job.sources?.segmentation?.path === action.sourcePath
      && job.sources?.anomaly?.result_id === action.anomalyResultId
      && job.sources?.anomaly?.path === action.anomalyPath;
    try {
      const payload = await requestJson("/api/postprocess-jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: action.name,
          segmentation_result_id: action.sourceResultId,
          segmentation_path: action.sourcePath,
          anomaly_result_id: action.anomalyResultId,
          anomaly_path: action.anomalyPath,
        }),
      });
      if (!payload.job?.id) throw new Error("The job was saved without a valid job response.");
      state.jobs = [payload.job, ...state.jobs.filter(job => job.id !== payload.job.id)];
      state.jobsLoaded = true;
      renderJobList();
      closeJobModal();
      await openJob(payload.job);
    } catch (error) {
      // A response can be interrupted after the backend has already persisted
      // the job. Reload once and recover that job before reporting a failure.
      try {
        const payload = await requestJson("/api/postprocess-jobs", { cache: "no-store" });
        state.jobs = payload.jobs || [];
        state.jobsLoaded = true;
        const recovered = state.jobs.find(matchesCreatedJob);
        if (recovered) {
          renderJobList();
          closeJobModal();
          await openJob(recovered);
          setMessage("Job created. The saved job was recovered after the creation response was interrupted.", "warn");
          return;
        }
      } catch (_) {
        // Preserve and display the original creation error below.
      }
      renderJobList();
      showJobCreationFailure(error);
    }
  }

  function showRenameJobModal(name) {
    const modal = byId("ppJobRenameModal");
    const input = byId("ppJobRenameName");
    input.value = name;
    modal.classList.remove("hidden");
    modal.classList.add("show");
    input.focus();
    input.select();
    return new Promise(resolve => {
      const finish = value => { modal.classList.remove("show"); modal.classList.add("hidden"); resolve(value); };
      byId("ppJobRenameSave").onclick = () => input.value.trim() && finish(input.value.trim());
      byId("ppJobRenameCancel").onclick = () => finish(null);
      byId("ppJobRenameClose").onclick = () => finish(null);
      input.onkeydown = event => { if (event.key === "Enter") byId("ppJobRenameSave").click(); if (event.key === "Escape") finish(null); };
    });
  }

  function showDeleteJobModal(name) {
    const modal = byId("ppJobDeleteModal");
    byId("ppJobDeleteMessage").textContent = `Are you sure you want to delete “${name}”? All derived outputs in this job will be removed. Original test results will remain unchanged.`;
    modal.classList.remove("hidden");
    modal.classList.add("show");
    return new Promise(resolve => {
      const finish = value => { modal.classList.remove("show"); modal.classList.add("hidden"); resolve(value); };
      byId("ppJobDeleteConfirm").onclick = () => finish(true);
      byId("ppJobDeleteCancel").onclick = () => finish(false);
      byId("ppJobDeleteClose").onclick = () => finish(false);
    });
  }

  async function openJobConfiguration(job) {
    try {
      const payload = await requestJson(`/api/postprocess-jobs/${encodeURIComponent(job.id)}/config`, { cache: "no-store" });
      const action = await showConfigJobModal(payload.job);
      if (!action?.segmentationResultId || !action?.segmentationPath || !action?.anomalyResultId || !action?.anomalyPath) return;
      const currentSegmentation = payload.job.sources?.segmentation || {};
      const currentAnomaly = payload.job.sources?.anomaly || {};
      const segmentationChanged = action.segmentationResultId !== currentSegmentation.result_id
        || action.segmentationPath !== currentSegmentation.path;
      const anomalyChanged = action.anomalyResultId !== currentAnomaly.result_id
        || action.anomalyPath !== currentAnomaly.path;
      if (!segmentationChanged && !anomalyChanged) return;
      const changedSources = [segmentationChanged ? "segmentation" : "", anomalyChanged ? "anomaly" : ""].filter(Boolean);
      const confirm = await showJobModal("reset", { ...job, changedSources });
      if (confirm?.type !== "reset") return;
      const updated = await requestJson(`/api/postprocess-jobs/${encodeURIComponent(job.id)}/config`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          segmentation_result_id: action.segmentationResultId,
          segmentation_path: action.segmentationPath,
          anomaly_result_id: action.anomalyResultId,
          anomaly_path: action.anomalyPath,
          confirm_reset: true,
        }),
      });
      state.currentJob = updated.job;
      state.jobs = state.jobs.map(item => item.id === updated.job.id ? updated.job : item);
      state.jobsLoaded = true;
      resetModeCache();
      const changedLabels = (updated.changed_sources || []).map(kind => kind === "anomaly" ? "anomaly" : kind).join(" and ");
      setMessage(`${changedLabels || "Job"} source updated. Dependent derived outputs were removed; original test results were not changed.`, "ok");
      const workspaceResultId = state.currentJob.sources?.[state.mode]?.workspace_result_id || "";
      byId("ppResult").replaceChildren();
      addOption(byId("ppResult"), workspaceResultId, workspaceResultId);
      byId("ppResult").value = workspaceResultId;
      await loadGeojsons();
    } catch (error) {
      setMessage(error.message, "err");
    }
  }

  async function showConfigJobModal(job) {
    const modal = byId("ppJobConfigModal");
    const fields = {
      segmentation: {
        result: byId("ppConfigSourceResult"),
        geojson: byId("ppConfigSourceGeojson"),
        summary: byId("ppConfigSourceSummary"),
        fullSummary: byId("ppConfigSourceFullSummary"),
        source: job.sources?.segmentation || {},
      },
      anomaly: {
        result: byId("ppConfigAnomalyResult"),
        geojson: byId("ppConfigAnomalyGeojson"),
        summary: byId("ppConfigAnomalySummary"),
        fullSummary: byId("ppConfigAnomalyFullSummary"),
        source: job.sources?.anomaly || {},
      },
    };
    const sessions = await requestJson("/api/sessions", { cache: "no-store" });
    for (const field of Object.values(fields)) {
      field.result.replaceChildren();
      addOption(field.result, "", "Select a result…");
      for (const item of sessions.sessions || []) addOption(field.result, item.id, item.display_name || item.name || item.id);
      field.result.value = field.source.result_id || "";
    }
    const loadFiles = async (kind, selectedPath = "") => {
      const field = fields[kind];
      field.fullSummary.hidden = true;
      field.fullSummary.replaceChildren();
      setSourceStepMessage(field.summary, kind === "anomaly" ? "Loading anomaly GeoJSON files…" : "Loading segmentation GeoJSON files…", true);
      field.geojson.replaceChildren();
      addOption(field.geojson, "", field.result.value ? "Loading GeoJSON files…" : "Select a GeoJSON…");
      field.geojson.disabled = true;
      if (!field.result.value) {
        setSourceStepMessage(field.summary, "Select a test result first.");
        return;
      }
      const payload = await requestJson(`/api/results/${encodeURIComponent(field.result.value)}/postprocess/geojsons`);
      const files = kind === "anomaly"
        ? (payload.files || []).filter(file => /anomal|predict/i.test(file.name))
        : (payload.files || []);
      field.geojson.replaceChildren();
      addOption(field.geojson, "", kind === "anomaly" ? "Select an anomalies GeoJSON…" : "Select a GeoJSON…");
      for (const file of files) addOption(field.geojson, file.path, `${file.name} · ${file.stage}`);
      field.geojson.disabled = false;
      if (selectedPath && [...field.geojson.options].some(option => option.value === selectedPath)) field.geojson.value = selectedPath;
      if (!selectedPath) setSourceStepMessage(field.summary, "Select a GeoJSON to scan the source.");
    };
    const renderFullSummary = (container, sourceSummary) => {
      container.replaceChildren();
      const geometryTypes = Object.entries(sourceSummary.geometry_types || {})
        .map(([type, count]) => `${type}: ${Number(count).toLocaleString()}`)
        .join(", ") || "—";
      const pixelSize = sourceSummary.median_pixel_size_m == null
        ? "—"
        : `${Number(sourceSummary.median_pixel_size_m).toFixed(4)} m`;
      const rows = [
        ["GeoJSON features", Number(sourceSummary.feature_count || 0).toLocaleString()],
        ["Valid polygons", Number(sourceSummary.valid_polygon_count || 0).toLocaleString()],
        ["On tile edges", Number(sourceSummary.features_on_tile_edges || 0).toLocaleString()],
        ["Away from tile edges", Number(sourceSummary.features_away_from_tile_edges || 0).toLocaleString()],
        ["Referenced tiles", Number(sourceSummary.tile_count || 0).toLocaleString()],
        ["Invalid features", Number(sourceSummary.invalid_feature_count || 0).toLocaleString()],
        ["Geometry", geometryTypes],
        ["Metric CRS", sourceSummary.metric_crs || "—"],
        ["Median pixel size", pixelSize],
      ];
      for (const [label, value] of rows) {
        const row = document.createElement("div");
        const name = document.createElement("span");
        name.textContent = label;
        const detail = document.createElement("strong");
        detail.textContent = value;
        row.append(name, detail);
        container.appendChild(row);
      }
    };
    const showSummary = (kind, sourceSummary) => {
      const field = fields[kind];
      setSourceStepMessage(field.summary, kind === "anomaly"
        ? `${Number(sourceSummary.feature_count || 0).toLocaleString()} anomaly features`
        : `${Number(sourceSummary.feature_count || 0).toLocaleString()} features · ${Number(sourceSummary.features_on_tile_edges || 0).toLocaleString()} on tile edges`);
      renderFullSummary(field.fullSummary, sourceSummary);
      const separator = document.createElement("span");
      separator.textContent = " · ";
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "postprocessSummaryLink";
      toggle.textContent = "View full summary";
      toggle.addEventListener("click", () => {
        const opening = field.fullSummary.hidden;
        if (opening) revealSourceField(field.result);
        field.fullSummary.hidden = !opening;
        toggle.textContent = opening ? "Hide full summary" : "View full summary";
      });
      field.summary.append(separator, toggle);
    };
    const scan = async kind => {
      const field = fields[kind];
      if (!field.result.value || !field.geojson.value) return;
      field.fullSummary.hidden = true;
      resetSourceStepCompletion(field.geojson);
      setSourceStepMessage(field.summary, kind === "anomaly" ? "Scanning anomalies GeoJSON…" : "Scanning source GeoJSON…", true);
      try {
        const scanned = await requestJson(`/api/results/${encodeURIComponent(field.result.value)}/postprocess/analyze`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input_path: field.geojson.value }),
        });
        showSummary(kind, scanned.summary || {});
        completeSourceStep(field.summary, kind === "segmentation");
      } catch (error) { setSourceStepMessage(field.summary, error.message); }
    };
    await Promise.all([
      loadFiles("segmentation", fields.segmentation.source.path || ""),
      loadFiles("anomaly", fields.anomaly.source.path || ""),
    ]);
    showSummary("segmentation", fields.segmentation.source.summary || {});
    showSummary("anomaly", fields.anomaly.source.summary || {});
    completeSourceStep(fields.segmentation.summary);
    completeSourceStep(fields.anomaly.summary);
    for (const kind of Object.keys(fields)) {
      fields[kind].result.onchange = () => {
        resetSourceStepCompletion(fields[kind].result);
        setSourceStepMessage(fields[kind].summary, "Select a GeoJSON to scan the source.");
        void loadFiles(kind);
      };
      fields[kind].geojson.onchange = () => void scan(kind);
    }
    resetSourceSteps(modal);
    modal.classList.remove("hidden");
    modal.classList.add("show");
    return new Promise(resolve => {
      const finish = value => { modal.classList.remove("show"); modal.classList.add("hidden"); resolve(value); };
      byId("ppJobConfigSave").onclick = () => {
        const segmentationMissing = !fields.segmentation.result.value || !fields.segmentation.geojson.value;
        const anomalyMissing = !fields.anomaly.result.value || !fields.anomaly.geojson.value;
        if (segmentationMissing || anomalyMissing) {
          if (segmentationMissing) {
            setSourceStepMessage(fields.segmentation.summary, "Select the segmentation test result and GeoJSON.");
            revealSourceField(fields.segmentation.result);
          }
          if (anomalyMissing) {
            setSourceStepMessage(fields.anomaly.summary, "Select the anomaly test result and GeoJSON.");
            revealSourceField(fields.anomaly.result);
          }
          return;
        }
        finish({
          segmentationResultId: fields.segmentation.result.value,
          segmentationPath: fields.segmentation.geojson.value,
          anomalyResultId: fields.anomaly.result.value,
          anomalyPath: fields.anomaly.geojson.value,
        });
      };
      byId("ppJobConfigCancel").onclick = () => finish(null);
      byId("ppJobConfigClose").onclick = () => finish(null);
    });
  }

  function showJobModal(mode, value) {
    const modal = byId("ppJobModal");
    const input = byId("ppJobName");
    const nameField = byId("ppJobNameField");
    const save = byId("ppJobModalSave");
    const deleteButton = byId("ppJobModalDelete");
    const sourceFields = byId("ppJobSourceFields");
    const sourceResult = byId("ppJobSourceResult");
    const sourceGeojson = byId("ppJobSourceGeojson");
    const anomalyResult = byId("ppJobAnomalyResult");
    const anomalyGeojson = byId("ppJobAnomalyGeojson");
    if (!modal || !input || !save) return Promise.resolve(null);
    const modalBody = modal.querySelector(".modalBody");
    modalBody.insertBefore(byId("ppJobModalMessage"), nameField);
    lockJobCreationFields(false);
    byId("ppJobModalMessage").className = "muted tiny";
    byId("ppJobModalCancel").hidden = false;
    byId("ppJobModalCancel").disabled = false;
    byId("ppJobModalClose").disabled = false;
    save.disabled = false;
    const config = mode === "config";
    byId("ppJobModalTitle").textContent = mode === "create" ? "Create post-processing job"
      : config ? "Edit job configuration"
        : mode === "delete" ? "Delete post-processing job"
          : mode === "reset" ? "Change job source"
            : "Rename post-processing job";
    const resetScope = mode === "reset" ? value?.changedSources || [] : [];
    const resetMessage = resetScope.length === 2
      ? "Both sources will change. All segmentation and anomaly outputs in this job will be removed."
      : resetScope[0] === "segmentation"
        ? "The segmentation source will change. Segmentation outputs and anomaly-to-panel associations derived from it will be removed."
        : "The anomaly source will change. Anomaly outputs derived from it will be removed.";
    byId("ppJobModalMessage").textContent = mode === "delete"
      ? `Delete “${value}”? This removes the job metadata and derived outputs.`
      : mode === "reset"
        ? `Change the configuration for “${value?.name || value}”? ${resetMessage} Original test results are not changed.`
      : "Give this workspace a name. Test-run source files remain unchanged.";
    const automaticJobName = () => {
      const now = new Date();
      const month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][now.getMonth()];
      return `Post_Process_${month}_${now.getDate()}`;
    };
    input.value = mode === "rename" ? value || "" : mode === "create" ? automaticJobName() : "";
    nameField.hidden = !["create", "rename"].includes(mode);
    sourceFields.hidden = !["create", "config"].includes(mode);
    save.textContent = mode === "create" ? "Create job" : mode === "delete" ? "Delete job" : mode === "reset" ? "Change source" : mode === "config" ? "Save configuration" : "Save name";
    save.classList.toggle("danger", ["delete", "reset"].includes(mode));
    deleteButton.hidden = mode !== "menu";
    if (mode === "menu") {
      byId("ppJobModalMessage").textContent = "Choose an operation for this job.";
      save.textContent = "Rename";
    }
    if (mode === "create") resetSourceSteps(modal);
    if (mode === "create") resetSourceInfo(modal);
    modal.classList.remove("hidden");
    modal.classList.add("show");
    if (mode === "create" || config) {
      if (mode === "create") {
        resetSourceStepCompletion(sourceResult);
        resetSourceStepCompletion(anomalyResult);
        setSourceStepMessage(byId("ppJobSourceSummary"), "Loading test results…", true);
        setSourceStepMessage(byId("ppJobAnomalySummary"), "Loading test results…", true);
      }
      void loadJobSourceResults(config ? value?.source?.result_id : "", config ? value?.source?.path : "");
      if (mode === "create") void loadJobAnomalyResults();
      sourceResult.onchange = () => {
        resetSourceStepCompletion(sourceResult);
        void loadJobSourceGeojsons();
      };
      sourceGeojson.onchange = () => void scanJobSource();
      anomalyResult.onchange = () => {
        resetSourceStepCompletion(anomalyResult);
        void loadJobAnomalyGeojsons();
      };
      anomalyGeojson.onchange = () => void scanJobAnomalySource();
    }
    input.focus();
    if (mode === "create") input.select();
    return new Promise(resolve => {
      const finish = result => {
        const creationSubmitted = mode === "create" && result?.type === "create";
        if (!creationSubmitted) closeJobModal();
        save.onclick = null;
        deleteButton.onclick = null;
        byId("ppJobModalCancel").onclick = null;
        byId("ppJobModalClose").onclick = null;
        resolve(result);
      };
      save.onclick = () => {
        if (mode === "delete") return finish({ type: "delete" });
        if (mode === "reset") return finish({ type: "reset" });
        const name = input.value.trim();
        if ((mode === "create" || config) && (!sourceResult.value || !sourceGeojson.value || (mode === "create" && (!anomalyResult.value || !anomalyGeojson.value)))) {
          setSourceStepMessage(byId("ppJobSourceSummary"), "Select the required segmentation and anomaly sources.");
          if (!sourceResult.value || !sourceGeojson.value) revealSourceField(sourceResult);
          if (!anomalyResult.value || !anomalyGeojson.value) revealSourceField(anomalyResult);
          return;
        }
        if (mode === "config") finish({ type: mode, sourceResultId: sourceResult.value, sourcePath: sourceGeojson.value });
        else if (name) finish({ type: mode, name, sourceResultId: sourceResult.value, sourcePath: sourceGeojson.value, anomalyResultId: anomalyResult.value, anomalyPath: anomalyGeojson.value });
      };
      deleteButton.onclick = () => finish({ type: "delete" });
      byId("ppJobModalCancel").onclick = () => finish(null);
      byId("ppJobModalClose").onclick = () => finish(null);
      input.onkeydown = event => { if (event.key === "Enter") save.click(); if (event.key === "Escape") finish(null); };
    });
  }

  async function loadJobSourceResults(selectedResultId = "", selectedPath = "") {
    const result = byId("ppJobSourceResult");
    setSourceStepMessage(byId("ppJobSourceSummary"), "Loading test results…", true);
    result.replaceChildren();
    addOption(result, "", "Loading results…");
    result.disabled = true;
    try {
      const payload = await requestJson("/api/sessions", { cache: "no-store" });
      result.replaceChildren();
      addOption(result, "", "Select a result…");
      for (const item of payload.sessions || []) addOption(result, item.id, item.display_name || item.name || item.id);
      result.disabled = false;
      setSourceStepMessage(byId("ppJobSourceSummary"), "Select a result and GeoJSON to scan the source.");
      if (selectedResultId && [...result.options].some(option => option.value === selectedResultId)) {
        result.value = selectedResultId;
        await loadJobSourceGeojsons(selectedPath);
      }
    } catch (error) {
      result.replaceChildren();
      addOption(result, "", "Could not load results");
      setSourceStepMessage(byId("ppJobSourceSummary"), error.message);
    }
  }

  async function loadJobSourceGeojsons(selectedPath = "") {
    const resultId = byId("ppJobSourceResult").value;
    const geojson = byId("ppJobSourceGeojson");
    setSourceStepMessage(byId("ppJobSourceSummary"), "Loading segmentation GeoJSON files…", true);
    geojson.replaceChildren();
    addOption(geojson, "", resultId ? "Loading GeoJSON files…" : "Select a GeoJSON…");
    geojson.disabled = true;
    if (!resultId) {
      setSourceStepMessage(byId("ppJobSourceSummary"), "Select a test result first.");
      return;
    }
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/geojsons`);
      geojson.replaceChildren();
      addOption(geojson, "", "Select a GeoJSON…");
      for (const file of payload.files || []) addOption(geojson, file.path, `${file.name} · ${file.stage}`);
      geojson.disabled = false;
      setSourceStepMessage(byId("ppJobSourceSummary"), "Select a GeoJSON to scan the source.");
      if (selectedPath && [...geojson.options].some(option => option.value === selectedPath)) {
        geojson.value = selectedPath;
        await scanJobSource();
      } else if (!selectedPath) {
        const predictions = (payload.files || []).find(file => file.name.toLowerCase() === "predictions.geojson");
        if (predictions) {
          geojson.value = predictions.path;
          await scanJobSource();
        }
      }
    } catch (error) {
      setSourceStepMessage(byId("ppJobSourceSummary"), error.message);
    }
  }

  async function scanJobSource() {
    const resultId = byId("ppJobSourceResult").value;
    const inputPath = byId("ppJobSourceGeojson").value;
    if (!resultId || !inputPath) return;
    const summary = byId("ppJobSourceSummary");
    resetSourceStepCompletion(byId("ppJobSourceGeojson"));
    setSourceStepMessage(summary, "Scanning source GeoJSON…", true);
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/analyze`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input_path: inputPath }),
      });
      setSourceStepMessage(summary, `${Number(payload.summary?.feature_count || 0).toLocaleString()} features · ${Number(payload.summary?.features_on_tile_edges || 0).toLocaleString()} on tile edges`);
      completeSourceStep(summary, true);
    } catch (error) {
      setSourceStepMessage(summary, error.message);
    }
  }

  async function loadJobAnomalyResults() {
    const select = byId("ppJobAnomalyResult");
    setSourceStepMessage(byId("ppJobAnomalySummary"), "Loading test results…", true);
    select.replaceChildren(); addOption(select, "", "Loading results…"); select.disabled = true;
    try {
      const payload = await requestJson("/api/sessions", { cache: "no-store" });
      select.replaceChildren(); addOption(select, "", "Select a result…");
      for (const item of payload.sessions || []) addOption(select, item.id, item.display_name || item.name || item.id);
      select.disabled = false;
      setSourceStepMessage(byId("ppJobAnomalySummary"), "Select a result and anomalies GeoJSON to scan the source.");
    } catch (error) { setSourceStepMessage(byId("ppJobAnomalySummary"), error.message); }
  }

  async function loadJobAnomalyGeojsons() {
    const resultId = byId("ppJobAnomalyResult").value;
    const select = byId("ppJobAnomalyGeojson");
    setSourceStepMessage(byId("ppJobAnomalySummary"), "Loading anomaly GeoJSON files…", true);
    select.replaceChildren(); addOption(select, "", "Loading GeoJSON files…"); select.disabled = true;
    if (!resultId) {
      setSourceStepMessage(byId("ppJobAnomalySummary"), "Select a test result first.");
      return;
    }
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/geojsons`);
      const files = (payload.files || []).filter(file => /anomal|predict/i.test(file.name));
      select.replaceChildren(); addOption(select, "", "Select an anomalies GeoJSON…");
      for (const file of files) addOption(select, file.path, `${file.name} · ${file.stage}`);
      select.disabled = false;
      setSourceStepMessage(byId("ppJobAnomalySummary"), "Select an anomalies GeoJSON to scan the source.");
      const predictions = files.find(file => file.name.toLowerCase() === "predictions.geojson");
      if (predictions) {
        select.value = predictions.path;
        await scanJobAnomalySource();
      }
    } catch (error) { setSourceStepMessage(byId("ppJobAnomalySummary"), error.message); }
  }

  async function scanJobAnomalySource() {
    const resultId = byId("ppJobAnomalyResult").value;
    const inputPath = byId("ppJobAnomalyGeojson").value;
    if (!resultId || !inputPath) return;
    const summary = byId("ppJobAnomalySummary");
    resetSourceStepCompletion(byId("ppJobAnomalyGeojson"));
    setSourceStepMessage(summary, "Scanning anomalies GeoJSON…", true);
    try {
      const payload = await requestJson(`/api/results/${encodeURIComponent(resultId)}/postprocess/analyze`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input_path: inputPath }),
      });
      setSourceStepMessage(summary, `${Number(payload.summary?.feature_count || 0).toLocaleString()} anomaly features`);
      completeSourceStep(summary);
    } catch (error) { setSourceStepMessage(summary, error.message); }
  }

  async function openJob(job) {
    if (state.currentJobId !== job.id) {
      resetModeCache();
      clearPreviewLayers();
      clearReferenceLayers();
    }
    state.currentJobId = job.id;
    state.currentJob = job;
    byId("ppJobLanding").hidden = true;
    byId("ppJobLanding").style.display = "none";
    document.querySelector(".postprocessWorkspace").hidden = false;
    document.querySelector(".postprocessWorkspace").style.display = "grid";
    ensurePreviewMap();
    byId("ppBackToJobs").hidden = false;
    byId("ppHeaderTitle").textContent = job.name || job.id;
    byId("ppHeaderJobId").textContent = `ID: ${job.id}`;
    byId("ppHeaderJobId").hidden = false;
    byId("ppHeaderDescription").hidden = true;
    byId("ppRefresh").hidden = false;
    byId("ppEditJobConfig").hidden = false;
    resetAnalysis({ showConfiguredStep: true });
    setStepsLoading(true);
    window.requestAnimationFrame(() => state.map?.invalidateSize());
    const configuredResultId = job.sources?.[state.mode]?.workspace_result_id || "";
    const resultSelect = byId("ppResult");
    resultSelect.replaceChildren();
    addOption(resultSelect, configuredResultId, configuredResultId || "Job snapshot unavailable");
    resultSelect.value = configuredResultId;
    resultSelect.disabled = true;
    await loadGeojsons();
  }

  async function refreshCurrentJob() {
    if (!state.currentJobId) return;
    try {
      const payload = await requestJson(
        `/api/postprocess-jobs/${encodeURIComponent(state.currentJobId)}/config`,
        { cache: "no-store" },
      );
      state.currentJob = payload.job;
      const index = state.jobs.findIndex(job => job.id === payload.job.id);
      if (index >= 0) state.jobs[index] = payload.job;
      resetModeCache();
      const workspaceResultId = payload.job.sources?.[state.mode]?.workspace_result_id || "";
      byId("ppResult").replaceChildren();
      addOption(byId("ppResult"), workspaceResultId, workspaceResultId || "Job snapshot unavailable");
      byId("ppResult").value = workspaceResultId;
      await loadGeojsons();
    } catch (error) {
      setMessage(error.message, "err");
    }
  }

  function showJobLanding() {
    resetModeCache();
    clearPreviewLayers();
    clearReferenceLayers();
    state.currentJobId = null;
    state.currentJob = null;
    void loadJobs();
  }

  function getContext() {
    const configuredSource = state.currentJob?.sources?.[state.mode] || {};
    return {
      resultId: byId("ppResult")?.value || "",
      sourcePath: byId("ppGeojson")?.value || "",
      configuredResultId: configuredSource.workspace_result_id || configuredSource.result_id || "",
      configuredSourcePath: configuredSource.workspace_path || configuredSource.path || "",
      segmentationResultId: state.currentJob?.sources?.segmentation?.workspace_result_id || state.currentJob?.sources?.segmentation?.result_id || "",
      job: state.currentJob,
      workflowId: state.workflowId,
      workflows: state.workflows.slice(),
      geojsonFiles: state.geojsonFiles.slice(),
      mode: state.mode,
    };
  }

  function setMode(mode) {
    const nextMode = mode === "anomaly" ? "anomaly" : "segmentation";
    if (nextMode === state.mode) return;
    if (state.editing) stopEditing(true);
    cacheCurrentModeState();
    state.pollToken += 1;
    invalidateModeLoad();
    state.mode = nextMode;
    if (restoreModeState(nextMode)) return;
    setStepsLoading(true, `Loading configured ${nextMode === "anomaly" ? "anomaly" : "segmentation"} GeoJSON data…`);
    state.previewLayers = new Map();
    state.referenceLayers = new Map();
    state.geojsonFiles = [];
    state.workflows = [];
    resetAnalysis({ showConfiguredStep: true });
    clearReferenceLayers();
    const configuredResultId = state.currentJob?.sources?.[state.mode]?.workspace_result_id || "";
    const resultSelect = byId("ppResult");
    if (configuredResultId) {
      resultSelect.replaceChildren();
      addOption(resultSelect, configuredResultId, configuredResultId);
      resultSelect.value = configuredResultId;
      resultSelect.disabled = true;
      const scheduledContext = loadContext();
      state.modeLoadTimer = window.setTimeout(() => {
        state.modeLoadTimer = null;
        if (isCurrentLoad(scheduledContext)) void loadGeojsons();
      }, 100);
      return;
    }
    const workflow = selectSavedWorkflow("", null, true);
    state.workflowId = workflow?.id || null;
    if (workflow) applyWorkflow(workflow);
    renderWorkflowList();
    setStepsLoading(false);
    document.dispatchEvent(new CustomEvent("postprocess:data", { detail: getContext() }));
  }

  async function runWorkflow(payload) {
    const requestedContext = loadContext();
    const resultId = byId("ppResult").value;
    state.workflowId = payload.id;
    state.workflowStatusCache.set(workflowStatusKey(resultId, payload.id), payload);
    await bindCurrentJobWorkflow(payload.id);
    state.pollToken += 1;
    applyWorkflow(payload, requestedContext);
    await pollWorkflow(state.pollToken, requestedContext, resultId, payload.id);
  }

  async function bindCurrentJobWorkflow(workflowId) {
    if (!state.currentJobId || !state.currentJob) return;
    const payload = await requestJson(
      `/api/postprocess-jobs/${encodeURIComponent(state.currentJobId)}/workflow`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: state.mode, workflow_id: workflowId || "" }),
      },
    );
    state.currentJob = payload.job;
    const source = payload.job.sources?.[state.mode];
    if (source?.workspace_result_id && workflowId) {
      state.workflowStatusCache.delete(workflowStatusKey(source.workspace_result_id, workflowId));
    }
    const index = state.jobs.findIndex(job => job.id === payload.job.id);
    if (index >= 0) state.jobs[index] = payload.job;
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
    loadBoundWorkflowStatus,
    loadPreviewLayer,
    requestJson,
    runWorkflow,
    selectWorkflow,
    showAnomalyReviewPair,
    updateAnomalyReviewPairDecision,
    clearAnomalyReviewMap,
    whenProcessingLayersReady,
    setMessage,
    setMode,
    setStepsLoading,
    confirmReplacement,
    invalidateCachedMode,
    getInternalControl: id => internalControls.get(id) || null,
  };
  window.PostProcessTab = { init, activate, createJob };
  const bindJobCreateButton = () => {
    const button = document.getElementById("ppCreateJob");
    if (!button) {
      console.warn("Post-process: Create job button was not found after DOM ready.");
      return;
    }
    if (button.dataset.jobHandlerBound === "true") return;
    button.dataset.jobHandlerBound = "true";
    button.onclick = () => { void createJob(); };
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bindJobCreateButton, { once: true });
  else bindJobCreateButton();
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
