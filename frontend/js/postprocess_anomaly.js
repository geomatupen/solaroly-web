(() => {
  "use strict";

  const byId = id => document.getElementById(id) || window.PostprocessWorkspace?.getInternalControl?.(id) || null;
  const api = () => window.PostprocessWorkspace;
  let scannedPath = "";
  let scanningPath = "";
  let panelLayers = [];
  let panelLayersLoaded = false;

  function addOption(select, value, label, data = {}) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    Object.assign(option.dataset, data);
    select.appendChild(option);
  }

  function switchMode(mode) {
    const anomaly = mode === "anomaly";
    byId("ppSegmentationWorkflow").hidden = anomaly;
    byId("ppAnomalyControls").hidden = !anomaly;
    byId("ppSegmentationTab").classList.toggle("active", !anomaly);
    byId("ppSegmentationTab").classList.toggle("secondary", anomaly);
    byId("ppSegmentationTab").setAttribute("aria-selected", String(!anomaly));
    byId("ppAnomalyTab").classList.toggle("active", anomaly);
    byId("ppAnomalyTab").classList.toggle("secondary", !anomaly);
    byId("ppAnomalyTab").setAttribute("aria-selected", String(anomaly));
    api()?.setMode(mode);
    refresh(api()?.getContext());
    if (anomaly) void loadPanelLayers();
  }

  async function loadPanelLayers() {
    if (panelLayersLoaded) {
      refresh(api()?.getContext());
      return;
    }
    const select = byId("ppPanelReference");
    select.replaceChildren();
    addOption(select, "", "Loading identified panel layers…");
    select.disabled = true;
    try {
      const payload = await api().requestJson("/api/results/postprocess/panel-layers", { cache: "no-store" });
      panelLayers = payload.layers || [];
      panelLayersLoaded = true;
    } catch (error) {
      panelLayers = [];
      api().setMessage(`Could not load panel references: ${error.message}`, "err");
    }
    refresh(api()?.getContext());
  }

  async function showSegmentationReferences(option, context = api()?.getContext()) {
    if (!option?.dataset.url || context?.mode !== "anomaly") return;
    await api()?.loadPreviewLayer(
      "segmentation_regularized_reference",
      option.dataset.url,
      null,
      "Final regularized panels (read-only)",
      true,
      option.dataset.mtime,
    );
    if (option.dataset.rowsUrl) {
      await api()?.loadPreviewLayer(
        "segmentation_rows_reference",
        option.dataset.rowsUrl,
        null,
        "Final rows (visual reference)",
        true,
        option.dataset.rowsMtime,
      );
    }
  }

  function refresh(context = api()?.getContext()) {
    if (!context) return;
    const workflow = context.workflows.find(item => item.id === context.workflowId && item.workflow_kind === "anomaly");
    const hasDeduplicated = Boolean(workflow?.outputs?.deduplicated);
    const anomalySelect = byId("ppAnomalyGeojson");
    const previousAnomaly = anomalySelect.value;
    const candidates = context.geojsonFiles.filter(file =>
      /(?:^|[_-])(anomal(?:y|ies)|predictions?)(?:[_-]|\.|$)/i.test(file.name)
    );
    const configuredFile = context.geojsonFiles.find(file => file.path === context.configuredSourcePath);
    if (configuredFile && !candidates.some(file => file.path === configuredFile.path)) candidates.unshift(configuredFile);
    anomalySelect.replaceChildren();
    addOption(anomalySelect, "", context.resultId ? "Select an anomalies GeoJSON…" : "Select a test result first…");
    for (const file of candidates) addOption(anomalySelect, file.path, `${file.name} · ${file.stage}`);
    if (context.configuredSourcePath && candidates.some(file => file.path === context.configuredSourcePath)) {
      anomalySelect.value = context.configuredSourcePath;
    }
    else if (candidates.some(file => file.path === previousAnomaly)) anomalySelect.value = previousAnomaly;
    else if (workflow?.input_path && candidates.some(file => file.path === workflow.input_path)) anomalySelect.value = workflow.input_path;
    else {
      const likely = candidates.find(file => file.name.toLowerCase() === "anomalies.geojson")
        || candidates.find(file => file.name.toLowerCase() === "predictions.geojson")
        || candidates[0];
      if (likely) anomalySelect.value = likely.path;
    }
    anomalySelect.disabled = Boolean(context.configuredSourcePath) || !context.resultId || candidates.length === 0;
    byId("ppScanAnomalies").disabled = !anomalySelect.value;
    const selectedKey = anomalySelect.value ? `${context.resultId}::${anomalySelect.value}` : "";
    const configuredReady = Boolean(
      context.mode === "anomaly"
      && context.resultId === context.configuredResultId
      && context.configuredSourcePath
      && anomalySelect.value === context.configuredSourcePath
    );
    if (configuredReady && selectedKey !== scannedPath) {
      scannedPath = selectedKey;
      api()?.setMessage("");
    }
    const scanned = Boolean(configuredReady || (selectedKey && selectedKey === scannedPath) || hasDeduplicated);
    byId("ppDeduplicateStep").hidden = false;
    byId("ppDeduplicateStep").classList.toggle("locked", !scanned);
    byId("ppDeduplicateStep").setAttribute("aria-disabled", String(!scanned));
    byId("ppDeduplicate").disabled = !scanned;

    const panelSelect = byId("ppPanelReference");
    const previousPanel = panelSelect.value;
    const availablePanelLayers = context.segmentationResultId
      ? panelLayers.filter(layer => layer.result_id === context.segmentationResultId)
      : panelLayers;
    panelSelect.replaceChildren();
    addOption(panelSelect, "", "Select final regularized panels…");
    availablePanelLayers.forEach((layer, index) => addOption(
      panelSelect,
      `${layer.result_id}::${layer.path}`,
      `${layer.workflow_name}${index === 0 ? " · Latest" : ""} · ${layer.result_id} · Regularized`,
      {
        resultId: layer.result_id,
        path: layer.path,
        url: layer.url,
        mtime: layer.mtime,
        rowsPath: layer.rows_path,
        rowsUrl: layer.rows_url,
        rowsMtime: layer.rows_mtime,
      },
    ));
    if ([...panelSelect.options].some(option => option.value === previousPanel)) panelSelect.value = previousPanel;
    else if (availablePanelLayers.length) panelSelect.selectedIndex = 1;
    panelSelect.disabled = availablePanelLayers.length === 0;
    if (context.mode === "anomaly" && panelSelect.selectedOptions[0]?.dataset.url) {
      void showSegmentationReferences(panelSelect.selectedOptions[0], context);
    }

    byId("ppAdjustAnomaliesStep").hidden = false;
    byId("ppAssociateStep").hidden = false;
    byId("ppAdjustAnomaliesStep").classList.toggle("locked", !hasDeduplicated);
    byId("ppAssociateStep").classList.toggle("locked", !hasDeduplicated);
    byId("ppAdjustAnomaliesStep").setAttribute("aria-disabled", String(!hasDeduplicated));
    byId("ppAssociateStep").setAttribute("aria-disabled", String(!hasDeduplicated));
    byId("ppAssociate").disabled = !hasDeduplicated || !panelSelect.value;
    if (context.mode === "anomaly" && context.resultId && context.geojsonFiles.length) {
      api()?.setStepsLoading(false);
    }
  }

  async function scanPredictions() {
    const workspace = api();
    const context = workspace.getContext();
    const path = byId("ppAnomalyGeojson").value;
    const file = context.geojsonFiles.find(item => item.path === path);
    if (!path || !file?.url) return;
    const sourceKey = `${context.resultId}::${path}`;
    scanningPath = sourceKey;
    byId("ppScanAnomalies").disabled = true;
    workspace.setMessage("Loading prediction polygons into the map…");
    try {
      await workspace.loadPreviewLayer("source", file.url, null, "Anomaly predictions", false, file.mtime);
      const current = workspace.getContext();
      if (current.mode !== "anomaly" || current.resultId !== context.resultId) return;
      scannedPath = sourceKey;
      byId("ppDeduplicateStep").hidden = false;
      byId("ppDeduplicateStep").classList.remove("locked");
      byId("ppDeduplicateStep").setAttribute("aria-disabled", "false");
      byId("ppDeduplicate").disabled = false;
      workspace.setMessage("Anomalies GeoJSON is ready. Review it, then deduplicate overlapping-image detections.", "ok");
    } catch (error) {
      workspace.setMessage(error.message, "err");
    } finally {
      scanningPath = "";
      byId("ppScanAnomalies").disabled = !byId("ppAnomalyGeojson").value;
    }
  }

  async function deduplicate() {
    const workspace = api();
    const context = workspace.getContext();
    const source = byId("ppAnomalyGeojson").value;
    if (!context.resultId || !source) return;
    byId("ppDeduplicate").disabled = true;
    workspace.setMessage("Starting anomaly deduplication…");
    try {
      const payload = await workspace.requestJson(
        `/api/results/${encodeURIComponent(context.resultId)}/postprocess/anomalies/deduplicate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            input_path: source,
            output_name: byId("ppAnomalyOutputName").value.trim() || "anomaly_postprocess",
            minimum_iou: Number(byId("ppAnomalyIou").value),
            maximum_center_distance_m: Number(byId("ppAnomalyDistance").value),
            minimum_smaller_overlap: Number(byId("ppAnomalyOverlap").value),
          }),
        },
      );
      await workspace.runWorkflow(payload);
    } catch (error) {
      workspace.setMessage(error.message, "err");
      byId("ppDeduplicate").disabled = false;
    }
  }

  async function associate() {
    const workspace = api();
    const context = workspace.getContext();
    const panelOption = byId("ppPanelReference").selectedOptions[0];
    if (!context.resultId || !context.workflowId || !panelOption?.dataset.path) return;
    byId("ppAssociate").disabled = true;
    workspace.setMessage("Starting anomaly-to-panel association…");
    try {
      const payload = await workspace.requestJson(
        `/api/results/${encodeURIComponent(context.resultId)}/postprocess/${encodeURIComponent(context.workflowId)}/associate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            panel_path: panelOption.dataset.path,
            panel_result_id: panelOption.dataset.resultId,
            minimum_overlap: Number(byId("ppAssociationOverlap").value),
            maximum_distance_m: Number(byId("ppAssociationDistance").value),
          }),
        },
      );
      await workspace.runWorkflow(payload);
    } catch (error) {
      workspace.setMessage(error.message, "err");
      byId("ppAssociate").disabled = false;
    }
  }

  function init() {
    byId("ppSegmentationTab")?.addEventListener("click", () => switchMode("segmentation"));
    byId("ppAnomalyTab")?.addEventListener("click", () => switchMode("anomaly"));
    byId("ppAnomalyGeojson")?.addEventListener("change", event => {
      scannedPath = "";
      byId("ppScanAnomalies").disabled = !event.target.value;
      for (const id of ["ppDeduplicateStep", "ppAdjustAnomaliesStep", "ppAssociateStep"]) {
        byId(id).hidden = false;
        byId(id).classList.add("locked");
        byId(id).setAttribute("aria-disabled", "true");
      }
    });
    byId("ppPanelReference")?.addEventListener("change", event => {
      const context = api()?.getContext();
      const workflow = context?.workflows.find(item => item.id === context.workflowId);
      byId("ppAssociate").disabled = !event.target.value || !workflow?.outputs?.deduplicated;
      const option = event.target.selectedOptions[0];
      if (option?.dataset.url) void showSegmentationReferences(option, context);
    });
    byId("ppScanAnomalies")?.addEventListener("click", scanPredictions);
    byId("ppDeduplicate")?.addEventListener("click", deduplicate);
    byId("ppAssociate")?.addEventListener("click", associate);
    document.addEventListener("postprocess:data", event => refresh(event.detail));
    document.addEventListener("postprocess:workflow", event => refresh(event.detail.context));
    document.addEventListener("postprocess:cache-reset", () => {
      scannedPath = "";
      scanningPath = "";
      panelLayers = [];
      panelLayersLoaded = false;
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
