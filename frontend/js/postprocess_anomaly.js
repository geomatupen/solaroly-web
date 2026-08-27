(() => {
  "use strict";

  const byId = id => document.getElementById(id);
  const api = () => window.PostprocessWorkspace;
  let scannedPath = "";
  let panelLayers = [];

  function addOption(select, value, label, data = {}) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    Object.assign(option.dataset, data);
    select.appendChild(option);
  }

  function switchMode(mode) {
    const anomaly = mode === "anomaly";
    byId("ppSegmentationControls").hidden = anomaly;
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
    api()?.setMessage(anomaly
      ? "Select a test result, then choose its anomalies GeoJSON to deduplicate overlapping-image detections."
      : "Select a segmentation GeoJSON and scan it to begin.");
  }

  async function loadPanelLayers() {
    const select = byId("ppPanelReference");
    select.replaceChildren();
    addOption(select, "", "Loading identified panel layers…");
    select.disabled = true;
    try {
      const payload = await api().requestJson("/api/results/postprocess/panel-layers", { cache: "no-store" });
      panelLayers = payload.layers || [];
    } catch (error) {
      panelLayers = [];
      api().setMessage(`Could not load panel references: ${error.message}`, "err");
    }
    refresh(api()?.getContext());
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
    anomalySelect.replaceChildren();
    addOption(anomalySelect, "", context.resultId ? "Select an anomalies GeoJSON…" : "Select a test result first…");
    for (const file of candidates) addOption(anomalySelect, file.path, `${file.name} · ${file.stage}`);
    if (candidates.some(file => file.path === previousAnomaly)) anomalySelect.value = previousAnomaly;
    else if (workflow?.input_path && candidates.some(file => file.path === workflow.input_path)) anomalySelect.value = workflow.input_path;
    else {
      const likely = candidates.find(file => file.name.toLowerCase() === "anomalies.geojson")
        || candidates.find(file => file.name.toLowerCase() === "predictions.geojson")
        || candidates[0];
      if (likely) anomalySelect.value = likely.path;
    }
    anomalySelect.disabled = !context.resultId || candidates.length === 0;
    byId("ppScanAnomalies").disabled = !anomalySelect.value;
    const scanned = Boolean((anomalySelect.value && anomalySelect.value === scannedPath) || hasDeduplicated);
    byId("ppDeduplicateStep").hidden = !scanned;
    byId("ppDeduplicate").disabled = !scanned;

    const panelSelect = byId("ppPanelReference");
    const previousPanel = panelSelect.value;
    panelSelect.replaceChildren();
    addOption(panelSelect, "", "Select identified panels…");
    panelLayers.forEach((layer, index) => addOption(
      panelSelect,
      `${layer.result_id}::${layer.path}`,
      `${layer.workflow_name}${index === 0 ? " · Latest" : ""} · ${layer.result_id} · ${layer.stage}`,
      { resultId: layer.result_id, path: layer.path, url: layer.url },
    ));
    if ([...panelSelect.options].some(option => option.value === previousPanel)) panelSelect.value = previousPanel;
    else if (panelLayers.length) panelSelect.selectedIndex = 1;
    panelSelect.disabled = panelLayers.length === 0;

    byId("ppAdjustAnomaliesStep").hidden = !hasDeduplicated;
    byId("ppAssociateStep").hidden = !hasDeduplicated;
    byId("ppAssociate").disabled = !hasDeduplicated || !panelSelect.value;
  }

  async function scanPredictions() {
    const workspace = api();
    const context = workspace.getContext();
    const path = byId("ppAnomalyGeojson").value;
    const file = context.geojsonFiles.find(item => item.path === path);
    if (!path || !file?.url) return;
    byId("ppScanAnomalies").disabled = true;
    workspace.setMessage("Loading prediction polygons into the map…");
    try {
      await workspace.loadPreviewLayer("source", file.url, null, "Anomaly predictions");
      scannedPath = path;
      byId("ppDeduplicateStep").hidden = false;
      byId("ppDeduplicate").disabled = false;
      workspace.setMessage("Anomalies GeoJSON is ready. Review it, then deduplicate overlapping-image detections.", "ok");
    } catch (error) {
      workspace.setMessage(error.message, "err");
    } finally {
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
      byId("ppDeduplicateStep").hidden = true;
      byId("ppAdjustAnomaliesStep").hidden = true;
      byId("ppAssociateStep").hidden = true;
    });
    byId("ppPanelReference")?.addEventListener("change", event => {
      const context = api()?.getContext();
      const workflow = context?.workflows.find(item => item.id === context.workflowId);
      byId("ppAssociate").disabled = !event.target.value || !workflow?.outputs?.deduplicated;
      const option = event.target.selectedOptions[0];
      if (option?.dataset.url) void api()?.loadPreviewLayer("panel_reference", option.dataset.url, null, "Panel reference (read-only)");
    });
    byId("ppScanAnomalies")?.addEventListener("click", scanPredictions);
    byId("ppDeduplicate")?.addEventListener("click", deduplicate);
    byId("ppAssociate")?.addEventListener("click", associate);
    document.addEventListener("postprocess:data", event => refresh(event.detail));
    document.addEventListener("postprocess:workflow", event => refresh(event.detail.context));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
