(() => {
  "use strict";

  const byId = id => document.getElementById(id) || window.PostprocessWorkspace?.getInternalControl?.(id) || null;
  const api = () => window.PostprocessWorkspace;
  let scannedPath = "";
  let scanningPath = "";
  let panelLayers = [];
  let panelLayersLoaded = false;
  let loadedScoringWorkflowId = "";
  let neighborStatsTimer = null;
  let neighborStatsToken = 0;
  let lastNeighborStatsKey = "";
  let pendingNeighborStatsKey = "";
  const scoringDefaults = {
    ppVisualSimilarity: 80,
    ppAppearanceWeight: 50,
    ppContextWeight: 20,
    ppShapeWeight: 15,
    ppSizeWeight: 10,
    ppProximityWeight: 5,
    ppMinimumAppearance: 75,
    ppMinimumContext: 60,
    ppManualReviewThreshold: 60,
  };
  const representativeDefaults = {
    ppRepresentativeImageCenterWeight: 40,
    ppRepresentativeSpatialWeight: 35,
    ppRepresentativeConfidenceWeight: 25,
  };
  let representativeModalSnapshot = null;
  const manualDuplicateDecisions = new Map();
  let loadedComparisonPairs = [];
  let loadedComparisonTotal = 0;
  let loadedComparisonWorkflowId = "";

  function comparisonPairs(workflow) {
    return loadedComparisonWorkflowId === workflow?.id && loadedComparisonPairs.length
      ? loadedComparisonPairs
      : workflow?.visual_review?.pairs || [];
  }

  function visualThreshold() {
    return Math.max(0, Math.min(100, Number(byId("ppVisualSimilarity")?.value || 80)));
  }

  function scoringConfig() {
    const percent = id => Math.max(0, Math.min(100, Number(byId(id)?.value || 0)));
    return {
      threshold: visualThreshold(),
      reviewThreshold: percent("ppManualReviewThreshold"),
      minimumAppearance: percent("ppMinimumAppearance"),
      minimumContext: percent("ppMinimumContext"),
      weights: {
        appearance: percent("ppAppearanceWeight"),
        context: percent("ppContextWeight"),
        shape: percent("ppShapeWeight"),
        size: percent("ppSizeWeight"),
        proximity: percent("ppProximityWeight"),
      },
      representativeWeights: {
        imageCenter: percent("ppRepresentativeImageCenterWeight"),
        spatialCentrality: percent("ppRepresentativeSpatialWeight"),
        modelConfidence: percent("ppRepresentativeConfidenceWeight"),
      },
    };
  }

  function validateRepresentativeWeights(config = scoringConfig()) {
    const total = Object.values(config.representativeWeights).reduce((sum, value) => sum + value, 0);
    const valid = Math.abs(total - 100) < 0.01;
    const validation = byId("ppRepresentativeWeightsValidation");
    validation.textContent = valid
      ? "Weights total 100%."
      : `Weights total ${total.toLocaleString()}%; they must total 100%.`;
    validation.classList.toggle("err", !valid);
    byId("ppRepresentativeWeightsSave").disabled = !valid;
    return valid;
  }

  function refreshScoringControls() {
    const context = api()?.getContext();
    const workflow = context?.workflows.find(item => item.id === context.workflowId && item.workflow_kind === "anomaly");
    if (workflow) updateThresholdEffect(workflow);
    else scoringIsValid();
  }

  function closeRepresentativeWeightsModal(save = false) {
    const modal = document.getElementById("ppRepresentativeWeightsModal");
    if (!save && representativeModalSnapshot) {
      for (const [id, value] of Object.entries(representativeModalSnapshot)) {
        document.getElementById(id).value = value;
      }
    }
    representativeModalSnapshot = null;
    modal.classList.add("hidden");
    refreshScoringControls();
  }

  function openRepresentativeWeightsModal() {
    const modal = document.getElementById("ppRepresentativeWeightsModal");
    representativeModalSnapshot = Object.fromEntries(
      Object.keys(representativeDefaults).map(id => [id, document.getElementById(id).value]),
    );
    validateRepresentativeWeights();
    modal.classList.remove("hidden");
    document.getElementById("ppRepresentativeImageCenterWeight").focus();
  }

  function bindRepresentativeWeightsModal() {
    const button = document.getElementById("ppConfigureRepresentativeWeights");
    const modal = document.getElementById("ppRepresentativeWeightsModal");
    if (!button || !modal || button.dataset.weightsHandlerBound === "true") return;
    button.dataset.weightsHandlerBound = "true";
    button.onclick = openRepresentativeWeightsModal;
    for (const id of Object.keys(representativeDefaults)) {
      document.getElementById(id).oninput = () => validateRepresentativeWeights();
    }
    document.getElementById("ppRepresentativeWeightsReset").onclick = () => {
      for (const [id, value] of Object.entries(representativeDefaults)) {
        document.getElementById(id).value = String(value);
      }
      validateRepresentativeWeights();
    };
    document.getElementById("ppRepresentativeWeightsSave").onclick = () => {
      if (validateRepresentativeWeights()) closeRepresentativeWeightsModal(true);
    };
    document.getElementById("ppRepresentativeWeightsClose").onclick = () => closeRepresentativeWeightsModal(false);
    document.getElementById("ppRepresentativeWeightsCancel").onclick = () => closeRepresentativeWeightsModal(false);
    modal.onclick = event => {
      if (event.target === modal) closeRepresentativeWeightsModal(false);
    };
  }

  function compositeScore(pair, config = scoringConfig()) {
    if (pair?.appearance_similarity == null || pair?.context_similarity == null) return null;
    const total = Object.values(config.weights).reduce((sum, value) => sum + value, 0);
    if (!total) return null;
    return Object.entries(config.weights).reduce(
      (sum, [name, weight]) => sum + Number(pair[`${name}_similarity`] || 0) * weight,
      0,
    ) / total * 100;
  }

  function scoringIsValid(config = scoringConfig()) {
    const total = Object.values(config.weights).reduce((sum, value) => sum + value, 0);
    const representativeValid = validateRepresentativeWeights(config);
    const valid = Math.abs(total - 100) < 0.01 && config.reviewThreshold <= config.threshold && representativeValid;
    const validation = byId("ppScoringValidation");
    validation.textContent = Math.abs(total - 100) >= 0.01
      ? `Weights total ${total.toLocaleString()}%; they must total 100%.`
      : config.reviewThreshold > config.threshold
        ? "Manual-review threshold must not exceed the automatic threshold."
        : !representativeValid
          ? "Representative-selection weights must total 100%."
          : "Weights total 100%.";
    validation.classList.toggle("err", !valid);
    const warning = byId("ppVisualThresholdWarning");
    warning.hidden = config.threshold >= 70;
    warning.textContent = config.threshold < 70
      ? "A duplicate threshold below 70% can merge different nearby anomalies. Review the suggested pairs carefully."
      : "";
    return valid;
  }

  function updateThresholdEffect(workflow) {
    const pairs = comparisonPairs(workflow);
    const config = scoringConfig();
    const manualMode = byId("ppDeduplicationMode").value === "manual";
    const valid = manualMode ? manualDuplicateDecisions.size > 0 : scoringIsValid(config);
    const matching = pairs.filter(pair => {
      const score = compositeScore(pair, config);
      return score != null
        && score >= config.threshold
        && Number(pair.appearance_similarity || 0) * 100 >= config.minimumAppearance
        && Number(pair.context_similarity || 0) * 100 >= config.minimumContext;
    }).length;
    const reviewCount = pairs.filter(pair => {
      const score = compositeScore(pair, config);
      return score != null && score >= config.reviewThreshold && score < config.threshold;
    }).length;
    const comparedTotal = Number(workflow?.visual_analysis_stats?.visually_compared_pairs || 0);
    const displayed = pairs.length;
    const totalPairs = Number(workflow?.visual_review?.total_pairs || pairs.length);
    const estimate = displayed && totalPairs > displayed
      ? ` At least ${matching} of the ${displayed} displayed pairs qualify; all ${comparedTotal.toLocaleString()} comparisons will be evaluated when applied.`
      : ` ${matching.toLocaleString()} pair${matching === 1 ? "" : "s"} will be deduplicated.`;
    byId("ppVisualThresholdEffect").textContent = manualMode
      ? `${manualDuplicateDecisions.size.toLocaleString()} pair${manualDuplicateDecisions.size === 1 ? " is" : "s are"} marked as duplicate. Open image comparisons to change manual selections.`
      : `${config.threshold.toLocaleString()}% duplicate score or higher.${estimate} ${reviewCount.toLocaleString()} displayed pair${reviewCount === 1 ? " is" : "s are"} in the manual-review range.`;
    for (const card of byId("ppVisualReviewPairs").querySelectorAll(".postprocessVisualPair")) {
      const pair = pairs[Number(card.dataset.pairIndex)];
      const score = compositeScore(pair, config);
      const qualifies = score != null
        && score >= config.threshold
        && Number(pair.appearance_similarity || 0) * 100 >= config.minimumAppearance
        && Number(pair.context_similarity || 0) * 100 >= config.minimumContext;
      const edgeKey = `${Math.min(pair.first_index, pair.second_index)}:${Math.max(pair.first_index, pair.second_index)}`;
      card.classList.toggle("isDuplicate", manualMode ? manualDuplicateDecisions.has(edgeKey) : qualifies);
      card.classList.toggle("needsReview", !manualMode && !qualifies && score != null && score >= config.reviewThreshold);
      card.querySelector(".postprocessManualPairControls")?.toggleAttribute("hidden", !manualMode);
      const scoreLabel = card.querySelector(".postprocessVisualSimilarity");
      if (scoreLabel) scoreLabel.textContent = score == null ? "Not compared" : `${Math.round(score)}% duplicate`;
    }
    byId("ppApplyVisualDeduplication").disabled = !valid || workflow?.status !== "complete";
    byId("ppApplyVisualDeduplication").textContent = manualMode ? "Apply manual selections" : "Apply visual deduplication";
    byId("ppVisualSimilarity").disabled = manualMode;
    byId("ppDuplicateScoringAdvanced").classList.toggle("locked", manualMode);
  }

  function renderVisualReview(workflow) {
    const review = byId("ppVisualReview");
    const pairsHost = byId("ppVisualReviewPairs");
    const stats = workflow?.visual_analysis_stats;
    if (!stats || !workflow?.visual_review) {
      review.hidden = true;
      pairsHost.replaceChildren();
      byId("ppViewVisualComparisons").hidden = true;
      return;
    }
    if (loadedScoringWorkflowId !== workflow.id) {
      loadedScoringWorkflowId = workflow.id;
      loadedComparisonPairs = [];
      loadedComparisonTotal = 0;
      loadedComparisonWorkflowId = workflow.id;
      const saved = workflow.scoring_parameters || {};
      const fields = {
        ppDeduplicationMode: saved.deduplication_mode,
        ppVisualSimilarity: saved.duplicate_score_percent,
        ppAppearanceWeight: saved.appearance_weight_percent,
        ppContextWeight: saved.context_weight_percent,
        ppShapeWeight: saved.shape_weight_percent,
        ppSizeWeight: saved.size_weight_percent,
        ppProximityWeight: saved.proximity_weight_percent,
        ppMinimumAppearance: saved.minimum_appearance_percent,
        ppMinimumContext: saved.minimum_context_percent,
        ppManualReviewThreshold: saved.manual_review_percent,
        ppRepresentativeImageCenterWeight: saved.representative_image_center_weight_percent,
        ppRepresentativeSpatialWeight: saved.representative_spatial_centrality_weight_percent,
        ppRepresentativeConfidenceWeight: saved.representative_confidence_weight_percent,
      };
      for (const [id, value] of Object.entries(fields)) {
        if (value != null) byId(id).value = String(value);
      }
      manualDuplicateDecisions.clear();
      for (const decision of saved.manual_decisions || []) {
        const key = `${Math.min(decision.first_index, decision.second_index)}:${Math.max(decision.first_index, decision.second_index)}`;
        manualDuplicateDecisions.set(key, decision);
      }
    }
    review.hidden = false;
    const candidateCount = Number(stats.spatial_candidate_pairs || 0);
    const comparedCount = Number(stats.visually_compared_pairs || 0);
    const missingCount = Number(stats.missing_image_pairs || 0);
    byId("ppVisualReviewSummary").textContent = `${candidateCount.toLocaleString()} candidate pairs from images within ${Number(stats.neighbor_image_radius_m || 0).toLocaleString()} m · anomaly shift up to ${Number(stats.maximum_location_shift_m || 0).toLocaleString()} m · ${comparedCount.toLocaleString()} visually compared${missingCount ? ` · ${missingCount.toLocaleString()} kept because imagery was unavailable` : ""}`;
    byId("ppVisualComparisonsSummary").textContent = byId("ppVisualReviewSummary").textContent;
    pairsHost.replaceChildren();
    const pairs = comparisonPairs(workflow);
    for (const [pairIndex, pair] of pairs.entries()) {
      const card = document.createElement("article");
      card.className = "postprocessVisualPair";
      card.dataset.pairIndex = String(pairIndex);
      const images = document.createElement("div");
      images.className = "postprocessVisualPairImages";
      for (const [url, name] of [[pair.first_crop_url, pair.first_image], [pair.second_crop_url, pair.second_image]]) {
        if (url) {
          const image = document.createElement("img");
          image.src = url;
          image.alt = `Anomaly crop from ${name || "source image"}`;
          image.loading = "lazy";
          image.title = name || "Source image";
          images.appendChild(image);
        } else {
          const missing = document.createElement("div");
          missing.className = "postprocessVisualPairMissing";
          missing.textContent = "Image or footprint unavailable";
          images.appendChild(missing);
        }
      }
      const meta = document.createElement("div");
      meta.className = "postprocessVisualPairMeta";
      const spatial = document.createElement("span");
      spatial.textContent = `IoU ${Math.round(Number(pair.iou || 0) * 100)}%`;
      const similarity = document.createElement("span");
      similarity.className = "postprocessVisualSimilarity";
      similarity.textContent = "Calculating score…";
      meta.append(spatial, similarity);
      const components = document.createElement("small");
      components.className = "postprocessVisualComponents muted";
      components.textContent = pair.appearance_similarity == null
        ? "Component scores unavailable"
        : `Appearance ${Math.round(Number(pair.appearance_similarity) * 100)}% · Context ${Math.round(Number(pair.context_similarity) * 100)}% · Shape ${Math.round(Number(pair.shape_similarity) * 100)}% · Size ${Math.round(Number(pair.size_similarity) * 100)}% · Distance ${Number(pair.center_distance_m || 0).toFixed(2)} m`;
      const manualControls = document.createElement("div");
      manualControls.className = "postprocessManualPairControls";
      const edgeKey = `${Math.min(pair.first_index, pair.second_index)}:${Math.max(pair.first_index, pair.second_index)}`;
      const existingDecision = manualDuplicateDecisions.get(edgeKey);
      const markLabel = document.createElement("label");
      const mark = document.createElement("input");
      mark.type = "checkbox";
      mark.checked = Boolean(existingDecision);
      mark.disabled = !pair.first_crop_url || !pair.second_crop_url;
      if (mark.disabled) mark.title = "Both image crops are required for manual duplicate selection.";
      markLabel.append(mark, document.createTextNode("Duplicate"));
      const keep = document.createElement("select");
      addOption(keep, String(pair.first_index), `Keep left · ${Number(pair.first_index) + 1}`);
      addOption(keep, String(pair.second_index), `Keep right · ${Number(pair.second_index) + 1}`);
      keep.value = String(existingDecision?.keep_index ?? pair.first_index);
      keep.disabled = !mark.checked || mark.disabled;
      const saveManualDecision = () => {
        keep.disabled = !mark.checked || mark.disabled;
        if (mark.checked) {
          manualDuplicateDecisions.set(edgeKey, {
            first_index: Number(pair.first_index),
            second_index: Number(pair.second_index),
            keep_index: Number(keep.value),
          });
        } else manualDuplicateDecisions.delete(edgeKey);
        updateThresholdEffect(workflow);
      };
      mark.addEventListener("change", saveManualDecision);
      keep.addEventListener("change", saveManualDecision);
      manualControls.append(markLabel, keep);
      card.append(images, meta, components, manualControls);
      pairsHost.appendChild(card);
    }
    const totalPairs = loadedComparisonTotal || Number(review.total_pairs || 0);
    if (totalPairs > pairs.length) {
      const note = document.createElement("p");
      note.className = "muted tiny";
      note.textContent = `Showing ${pairs.length.toLocaleString()} of ${totalPairs.toLocaleString()} candidate pairs.`;
      pairsHost.appendChild(note);
    }
    const viewButton = byId("ppViewVisualComparisons");
    viewButton.hidden = totalPairs === 0;
    viewButton.textContent = `View image comparisons (${totalPairs.toLocaleString()})`;
    const loadMore = byId("ppVisualComparisonsLoadMore");
    loadMore.hidden = pairs.length >= totalPairs;
    loadMore.disabled = false;
    updateThresholdEffect(workflow);
  }

  function addOption(select, value, label, data = {}) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    Object.assign(option.dataset, data);
    select.appendChild(option);
  }

  function scheduleNeighborStats(force = false) {
    if (neighborStatsTimer) clearTimeout(neighborStatsTimer);
    neighborStatsTimer = setTimeout(() => void loadNeighborStats(force), 350);
  }

  async function loadNeighborStats(force = false) {
    const workspace = api();
    const context = workspace?.getContext();
    const source = byId("ppAnomalyGeojson")?.value || context?.configuredSourcePath || "";
    const radius = Number(byId("ppAnomalyNeighborRadius")?.value);
    const estimate = byId("ppAnomalyNeighborEstimate");
    if (!context?.resultId || !source || !Number.isFinite(radius) || radius <= 0) {
      estimate.classList.remove("isLoading");
      estimate.textContent = "Select an anomaly source and enter a positive radius.";
      return;
    }
    const key = `${context.resultId}::${source}::${radius}`;
    if (!force && (key === lastNeighborStatsKey || key === pendingNeighborStatsKey)) return;
    const token = ++neighborStatsToken;
    pendingNeighborStatsKey = key;
    estimate.classList.add("isLoading");
    estimate.textContent = "Calculating neighboring-image coverage…";
    try {
      const payload = await workspace.requestJson(
        `/api/results/${encodeURIComponent(context.resultId)}/postprocess/anomalies/neighbor-stats`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ input_path: source, neighbor_image_radius_m: radius }),
        },
      );
      if (token !== neighborStatsToken) return;
      lastNeighborStatsKey = key;
      const average = Number(payload.average_neighbors || 0);
      const isolated = Number(payload.isolated_images || 0);
      estimate.textContent = `${average.toLocaleString()} neighboring images on average across ${Number(payload.image_count || 0).toLocaleString()} source images · range ${Number(payload.minimum_neighbors || 0).toLocaleString()}–${Number(payload.maximum_neighbors || 0).toLocaleString()}${isolated ? ` · ${isolated.toLocaleString()} isolated` : ""}${average > 12 ? " · Large radius; analysis may take longer" : ""}`;
    } catch (error) {
      if (token !== neighborStatsToken) return;
      estimate.textContent = `Could not estimate neighboring images: ${error.message}`;
    } finally {
      if (token === neighborStatsToken) {
        pendingNeighborStatsKey = "";
        estimate.classList.remove("isLoading");
      }
    }
  }

  async function loadComparisonPage(reset = false) {
    const workspace = api();
    const context = workspace?.getContext();
    if (!context?.resultId || !context.workflowId) return;
    const workflowId = context.workflowId;
    const button = byId("ppVisualComparisonsLoadMore");
    button.disabled = true;
    button.textContent = "Loading comparisons…";
    const offset = reset || loadedComparisonWorkflowId !== workflowId ? 0 : loadedComparisonPairs.length;
    try {
      const payload = await workspace.requestJson(
        `/api/results/${encodeURIComponent(context.resultId)}/postprocess/${encodeURIComponent(workflowId)}/visual-review?offset=${offset}&limit=48`,
        { cache: "no-store" },
      );
      const current = workspace.getContext();
      if (current.workflowId !== workflowId) return;
      loadedComparisonWorkflowId = workflowId;
      loadedComparisonPairs = offset === 0
        ? (payload.pairs || [])
        : [...loadedComparisonPairs, ...(payload.pairs || [])];
      loadedComparisonTotal = Number(payload.total_pairs || loadedComparisonPairs.length);
      const workflow = current.workflows.find(item => item.id === workflowId && item.workflow_kind === "anomaly");
      if (workflow) renderVisualReview(workflow);
    } catch (error) {
      api()?.setMessage(`Could not load image comparisons: ${error.message}`, "err");
    } finally {
      button.disabled = false;
      button.textContent = "Load more";
    }
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
    renderVisualReview(workflow);
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
    const workflowRunning = workflow?.status === "queued" || workflow?.status === "running";
    byId("ppDeduplicateStep").hidden = false;
    byId("ppDeduplicateStep").classList.toggle("locked", !scanned);
    byId("ppDeduplicateStep").setAttribute("aria-disabled", String(!scanned));
    byId("ppDeduplicate").disabled = !scanned || workflowRunning;
    byId("ppDeduplicate").textContent = workflowRunning && workflow?.stage === "deduplicate"
      ? "Analyzing visual duplicates…"
      : "Analyze visual duplicates";

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
        workflowId: layer.workflow_id,
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
      scheduleNeighborStats();
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
            maximum_center_distance_m: Number(byId("ppAnomalyDistance").value),
            neighbor_image_radius_m: Number(byId("ppAnomalyNeighborRadius").value),
          }),
        },
      );
      await workspace.runWorkflow(payload);
    } catch (error) {
      workspace.setMessage(error.message, "err");
      byId("ppDeduplicate").disabled = false;
    }
  }

  async function applyVisualDeduplication() {
    const workspace = api();
    const context = workspace.getContext();
    if (!context.resultId || !context.workflowId) return;
    const button = byId("ppApplyVisualDeduplication");
    button.disabled = true;
    const config = scoringConfig();
    const deduplicationMode = byId("ppDeduplicationMode").value;
    if (deduplicationMode === "threshold" && !scoringIsValid(config)) return;
    if (deduplicationMode === "manual" && !manualDuplicateDecisions.size) return;
    workspace.setMessage(deduplicationMode === "manual"
      ? "Applying manual duplicate selections…"
      : `Applying ${config.threshold}% duplicate score threshold…`);
    try {
      const payload = await workspace.requestJson(
        `/api/results/${encodeURIComponent(context.resultId)}/postprocess/${encodeURIComponent(context.workflowId)}/deduplicate/apply`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            deduplication_mode: deduplicationMode,
            manual_decisions: deduplicationMode === "manual" ? [...manualDuplicateDecisions.values()] : [],
            duplicate_score_percent: config.threshold,
            manual_review_percent: config.reviewThreshold,
            minimum_appearance_percent: config.minimumAppearance,
            minimum_context_percent: config.minimumContext,
            appearance_weight_percent: config.weights.appearance,
            context_weight_percent: config.weights.context,
            shape_weight_percent: config.weights.shape,
            size_weight_percent: config.weights.size,
            proximity_weight_percent: config.weights.proximity,
            representative_image_center_weight_percent: config.representativeWeights.imageCenter,
            representative_spatial_centrality_weight_percent: config.representativeWeights.spatialCentrality,
            representative_confidence_weight_percent: config.representativeWeights.modelConfidence,
          }),
        },
      );
      await workspace.runWorkflow(payload);
    } catch (error) {
      workspace.setMessage(error.message, "err");
      button.disabled = false;
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
            panel_workflow_id: panelOption.dataset.workflowId,
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
      lastNeighborStatsKey = "";
      scheduleNeighborStats(true);
    });
    byId("ppAnomalyNeighborRadius")?.addEventListener("input", () => {
      lastNeighborStatsKey = "";
      scheduleNeighborStats(true);
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
    byId("ppApplyVisualDeduplication")?.addEventListener("click", applyVisualDeduplication);
    for (const id of Object.keys(scoringDefaults)) byId(id)?.addEventListener("input", refreshScoringControls);
    byId("ppDeduplicationMode")?.addEventListener("change", refreshScoringControls);
    byId("ppResetScoringDefaults")?.addEventListener("click", () => {
      for (const [id, value] of Object.entries(scoringDefaults)) byId(id).value = String(value);
      refreshScoringControls();
    });
    const comparisonsModal = byId("ppVisualComparisonsModal");
    const closeComparisonsModal = () => comparisonsModal.classList.add("hidden");
    byId("ppViewVisualComparisons")?.addEventListener("click", () => {
      comparisonsModal.classList.remove("hidden");
      byId("ppVisualComparisonsClose").focus();
      void loadComparisonPage(true);
    });
    byId("ppVisualComparisonsLoadMore")?.addEventListener("click", () => void loadComparisonPage(false));
    byId("ppVisualComparisonsClose")?.addEventListener("click", closeComparisonsModal);
    byId("ppVisualComparisonsDone")?.addEventListener("click", closeComparisonsModal);
    comparisonsModal?.addEventListener("click", event => {
      if (event.target === comparisonsModal) closeComparisonsModal();
    });
    document.addEventListener("keydown", event => {
      const representativeModal = document.getElementById("ppRepresentativeWeightsModal");
      if (event.key === "Escape" && !representativeModal?.classList.contains("hidden")) {
        closeRepresentativeWeightsModal(false);
      }
      if (event.key === "Escape" && !comparisonsModal?.classList.contains("hidden")) {
        closeComparisonsModal();
      }
    });
    byId("ppAssociate")?.addEventListener("click", associate);
    document.addEventListener("postprocess:data", event => refresh(event.detail));
    document.addEventListener("postprocess:workflow", event => {
      refresh(event.detail.context);
      const status = event.detail.status;
      if (event.detail.context?.mode !== "anomaly"
        || status?.status !== "complete"
        || !status?.outputs?.associated
        || !status?.association_stats?.panel_updated_mtime) return;
      const option = byId("ppPanelReference").selectedOptions[0];
      if (!option?.dataset.url) return;
      option.dataset.mtime = String(status.association_stats.panel_updated_mtime);
      void showSegmentationReferences(option, event.detail.context);
      api()?.invalidateCachedMode("segmentation");
    });
    document.addEventListener("postprocess:cache-reset", () => {
      scannedPath = "";
      scanningPath = "";
      panelLayers = [];
      panelLayersLoaded = false;
      loadedScoringWorkflowId = "";
      lastNeighborStatsKey = "";
      pendingNeighborStatsKey = "";
      neighborStatsToken += 1;
      manualDuplicateDecisions.clear();
      loadedComparisonPairs = [];
      loadedComparisonTotal = 0;
      loadedComparisonWorkflowId = "";
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindRepresentativeWeightsModal, { once: true });
    document.addEventListener("DOMContentLoaded", init);
  } else {
    bindRepresentativeWeightsModal();
    init();
  }
})();
