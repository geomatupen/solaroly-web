(() => {
  "use strict";

  const byId = id => document.getElementById(id) || window.PostprocessWorkspace?.getInternalControl?.(id) || null;
  const api = () => window.PostprocessWorkspace;
  let scannedPath = "";
  let panelLayers = [];
  let panelLayersLoaded = false;
  let panelLayersJobKey = "";
  let loadedPanelReferenceKey = "";
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
    imageCenter: 40,
    spatialCentrality: 35,
    modelConfidence: 25,
  };
  const duplicateWeightDefaults = Object.fromEntries(
    Object.entries(scoringDefaults).filter(([id]) => id.endsWith("Weight")),
  );
  const modalWeightDefaults = { ...duplicateWeightDefaults };
  const thresholdDefaultIds = [
    "ppVisualSimilarity", "ppMinimumAppearance", "ppMinimumContext", "ppManualReviewThreshold",
  ];
  let representativeModalSnapshot = null;
  const manualDuplicateDecisions = new Map();
  let loadedComparisonPairs = [];
  let loadedComparisonTotal = 0;
  let loadedComparisonWorkflowId = "";
  let activeComparisonIndex = null;
  let comparisonImageZoom = 1;

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
      representativeWeights: { ...representativeDefaults },
    };
  }

  function setWeightValidation(id, weights) {
    const total = Object.values(weights).reduce((sum, value) => sum + value, 0);
    const valid = Math.abs(total - 100) < 0.01;
    const validation = byId(id);
    validation.textContent = valid
      ? "Weights total 100%."
      : `Weights total ${total.toLocaleString()}%; they must total 100%.`;
    validation.classList.toggle("err", !valid);
    return valid;
  }

  function validateDuplicateWeights(config = scoringConfig()) {
    return setWeightValidation("ppDuplicateWeightsValidation", config.weights);
  }

  function validateWeightConfiguration(config = scoringConfig()) {
    const valid = validateDuplicateWeights(config);
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
    modal.classList.remove("show");
    modal.classList.add("hidden");
    refreshScoringControls();
  }

  function openRepresentativeWeightsModal() {
    const modal = document.getElementById("ppRepresentativeWeightsModal");
    representativeModalSnapshot = Object.fromEntries(
      Object.keys(modalWeightDefaults).map(id => [id, document.getElementById(id).value]),
    );
    validateWeightConfiguration();
    modal.classList.remove("hidden");
    modal.classList.add("show");
    document.getElementById("ppAppearanceWeight").focus();
  }

  function openComparisonsModalLoading() {
    const modal = document.getElementById("ppVisualComparisonsModal");
    const pairs = document.getElementById("ppVisualReviewPairs");
    const summary = document.getElementById("ppVisualComparisonsSummary");
    const loadMore = document.getElementById("ppVisualComparisonsLoadMore");
    const loading = document.createElement("div");
    loading.className = "postprocessComparisonsLoading";
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    spinner.setAttribute("aria-hidden", "true");
    const message = document.createElement("span");
    message.textContent = "Analyzing visual duplicate candidates…";
    loading.append(spinner, message);
    activeComparisonIndex = null;
    byId("ppVisualComparisonDetail").hidden = true;
    pairs.hidden = false;
    pairs.dataset.analysisLoading = "true";
    pairs.replaceChildren(loading);
    summary.textContent = "Comparing anomaly crops from neighboring images. This may take a moment.";
    loadMore.hidden = true;
    modal.classList.remove("hidden");
    modal.classList.add("show");
    document.getElementById("ppVisualComparisonsClose")?.focus();
  }

  function bindRepresentativeWeightsModal() {
    const button = document.getElementById("ppConfigureRepresentativeWeights");
    const modal = document.getElementById("ppRepresentativeWeightsModal");
    if (!button || !modal || button.dataset.weightsHandlerBound === "true") return;
    button.dataset.weightsHandlerBound = "true";
    button.onclick = openRepresentativeWeightsModal;
    for (const id of Object.keys(modalWeightDefaults)) {
      document.getElementById(id).oninput = () => validateWeightConfiguration();
    }
    document.getElementById("ppRepresentativeWeightsReset").onclick = () => {
      for (const [id, value] of Object.entries(modalWeightDefaults)) {
        document.getElementById(id).value = String(value);
      }
      validateWeightConfiguration();
    };
    document.getElementById("ppRepresentativeWeightsSave").onclick = () => {
      if (validateWeightConfiguration()) closeRepresentativeWeightsModal(true);
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
    const duplicateValid = validateDuplicateWeights(config);
    const valid = duplicateValid && config.reviewThreshold <= config.threshold;
    const validation = byId("ppScoringValidation");
    validation.textContent = !duplicateValid
      ? "Duplicate-matching weights must total 100%. Open Configure weights to correct them."
      : config.reviewThreshold > config.threshold
        ? "Manual-review threshold must not exceed the automatic threshold."
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

  function comparisonEdgeKey(pair) {
    return `${Math.min(pair.first_index, pair.second_index)}:${Math.max(pair.first_index, pair.second_index)}`;
  }

  function currentAnomalyWorkflow() {
    const context = api()?.getContext();
    return context?.workflows.find(item => item.id === context.workflowId && item.workflow_kind === "anomaly");
  }

  function syncPairCardDecision(pair, pairIndex) {
    const card = byId("ppVisualReviewPairs").querySelector(`[data-pair-index="${pairIndex}"]`);
    if (!card) return;
    const decision = manualDuplicateDecisions.get(comparisonEdgeKey(pair));
    const mark = card.querySelector('[data-role="manual-duplicate"]');
    const keep = card.querySelector('[data-role="manual-keep"]');
    if (mark) mark.checked = Boolean(decision);
    if (keep) {
      keep.value = String(decision?.keep_index ?? pair.first_index);
      keep.disabled = !decision;
    }
  }

  function saveManualPairDecision(pair, pairIndex, marked, keepIndex) {
    const edgeKey = comparisonEdgeKey(pair);
    if (marked) {
      manualDuplicateDecisions.set(edgeKey, {
        first_index: Number(pair.first_index),
        second_index: Number(pair.second_index),
        keep_index: Number(keepIndex),
      });
    } else manualDuplicateDecisions.delete(edgeKey);
    syncPairCardDecision(pair, pairIndex);
    const workflow = currentAnomalyWorkflow();
    if (workflow) updateThresholdEffect(workflow);
  }

  function mapReviewPair(pair) {
    const config = scoringConfig();
    const score = compositeScore(pair, config);
    const qualifies = score != null
      && score >= config.threshold
      && Number(pair.appearance_similarity || 0) * 100 >= config.minimumAppearance
      && Number(pair.context_similarity || 0) * 100 >= config.minimumContext;
    return {
      ...pair,
      display_score: score,
      review_status: qualifies ? "duplicate" : score != null && score >= config.reviewThreshold ? "review" : "below",
    };
  }

  function viewComparisonOnMap(pairIndex) {
    const pairs = loadedComparisonPairs.map(mapReviewPair);
    if (!api()?.showAnomalyReviewPair(pairs, pairIndex, false)) return;
    const modal = byId("ppVisualComparisonsModal");
    modal.classList.remove("show");
    modal.classList.add("hidden");
  }

  async function loadAllComparisonMapPairs() {
    const workspace = api();
    const context = workspace?.getContext();
    if (!context?.resultId || !context.workflowId) return;
    try {
      const payload = await workspace.requestJson(
        `/api/results/${encodeURIComponent(context.resultId)}/postprocess/${encodeURIComponent(context.workflowId)}/visual-review-map`,
        { cache: "no-store" },
      );
      workspace.updateAnomalyReviewPairs((payload.pairs || []).map(mapReviewPair));
    } catch (error) {
      workspace.setMessage(`Could not show all duplicate candidates on the map: ${error.message}`, "err");
    }
  }

  function showComparisonGrid() {
    activeComparisonIndex = null;
    byId("ppVisualComparisonDetail").hidden = true;
    byId("ppVisualReviewPairs").hidden = false;
    byId("ppVisualComparisonsTitle").textContent = "Image comparisons";
    const loadMore = byId("ppVisualComparisonsLoadMore");
    loadMore.hidden = loadedComparisonPairs.length >= loadedComparisonTotal;
  }

  function focusBoxScale(box) {
    if (!Array.isArray(box) || box.length !== 4) return null;
    const width = Math.max(0.001, Number(box[2]) - Number(box[0]));
    const height = Math.max(0.001, Number(box[3]) - Number(box[1]));
    return Math.max(1, Math.min(8, 0.38 / Math.max(width, height)));
  }

  function setComparisonImageZoom(zoom) {
    comparisonImageZoom = Math.max(1, Math.min(10, Number(zoom) || 1));
    for (const image of byId("ppVisualComparisonFullImages").querySelectorAll("img[data-focus-x]")) {
      image.style.transformOrigin = `${image.dataset.focusX}% ${image.dataset.focusY}%`;
      image.style.transform = `scale(${comparisonImageZoom})`;
    }
    byId("ppVisualComparisonZoomLevel").textContent = `${Math.round(comparisonImageZoom * 100)}%`;
    byId("ppVisualComparisonZoomOut").disabled = comparisonImageZoom <= 1;
  }

  function renderComparisonDetail(pairIndex) {
    const pair = loadedComparisonPairs[pairIndex];
    if (!pair) return;
    activeComparisonIndex = pairIndex;
    byId("ppVisualReviewPairs").hidden = true;
    byId("ppVisualComparisonDetail").hidden = false;
    byId("ppVisualComparisonsLoadMore").hidden = true;
    byId("ppVisualComparisonsTitle").textContent = "Full-image comparison";
    byId("ppVisualComparisonPosition").textContent = `Comparison ${(pairIndex + 1).toLocaleString()} of ${loadedComparisonTotal.toLocaleString()}`;
    byId("ppVisualComparisonPrevious").disabled = pairIndex <= 0;
    byId("ppVisualComparisonNext").disabled = pairIndex + 1 >= loadedComparisonTotal;
    const focusScales = [
      pair.first_image_url ? focusBoxScale(pair.first_focus_box) : null,
      pair.second_image_url ? focusBoxScale(pair.second_focus_box) : null,
    ].filter(Number.isFinite);
    comparisonImageZoom = focusScales.length ? Math.min(...focusScales) : 1;

    const images = byId("ppVisualComparisonFullImages");
    images.replaceChildren();
    for (const [side, imageUrl, cropUrl, name, focusBox] of [
      ["Left", pair.first_image_url, pair.first_crop_url, pair.first_image, pair.first_focus_box],
      ["Right", pair.second_image_url, pair.second_crop_url, pair.second_image, pair.second_focus_box],
    ]) {
      const figure = document.createElement("figure");
      const frame = document.createElement("div");
      frame.className = "postprocessComparisonImageFrame";
      const availableUrl = imageUrl || cropUrl;
      if (availableUrl) {
        const image = document.createElement("img");
        image.src = availableUrl;
        image.alt = `${side} full source image showing anomaly ${name || "prediction"}`;
        const validFocus = imageUrl && Array.isArray(focusBox) && focusBox.length === 4;
        const focusX = validFocus ? (Number(focusBox[0]) + Number(focusBox[2])) * 50 : 50;
        const focusY = validFocus ? (Number(focusBox[1]) + Number(focusBox[3])) * 50 : 50;
        image.dataset.focusX = String(focusX);
        image.dataset.focusY = String(focusY);
        frame.appendChild(image);
      } else {
        const missing = document.createElement("div");
        missing.className = "postprocessComparisonImageMissing";
        missing.textContent = "Full image unavailable";
        frame.appendChild(missing);
      }
      const caption = document.createElement("figcaption");
      const label = document.createElement("strong");
      label.textContent = side;
      const filename = document.createElement("span");
      filename.textContent = name || "Source image unavailable";
      caption.append(label, filename);
      if (!imageUrl && cropUrl) {
        const cropOnly = document.createElement("small");
        cropOnly.className = "muted";
        cropOnly.textContent = "Showing crop because the full image is unavailable";
        caption.appendChild(cropOnly);
      }
      figure.append(frame, caption);
      images.appendChild(figure);
    }
    setComparisonImageZoom(comparisonImageZoom);

    const details = byId("ppVisualComparisonDetails");
    details.replaceChildren();
    const config = scoringConfig();
    const score = compositeScore(pair, config);
    const metrics = document.createElement("p");
    metrics.className = "muted tiny";
    metrics.textContent = pair.appearance_similarity == null
      ? `IoU ${Math.round(Number(pair.iou || 0) * 100)}% · Component scores unavailable`
      : `${score == null ? "No duplicate score" : `${Math.round(score)}% duplicate`} · Appearance ${Math.round(Number(pair.appearance_similarity) * 100)}% · Context ${Math.round(Number(pair.context_similarity) * 100)}% · Shape ${Math.round(Number(pair.shape_similarity) * 100)}% · Size ${Math.round(Number(pair.size_similarity) * 100)}% · Distance ${Number(pair.center_distance_m || 0).toFixed(2)} m`;
    details.appendChild(metrics);

    const manualControls = document.createElement("div");
    manualControls.className = "postprocessManualPairControls postprocessComparisonManualControls";
    manualControls.hidden = byId("ppDeduplicationMode").value !== "manual";
    const decision = manualDuplicateDecisions.get(comparisonEdgeKey(pair));
    const markLabel = document.createElement("label");
    const mark = document.createElement("input");
    mark.type = "checkbox";
    mark.checked = Boolean(decision);
    mark.disabled = !(pair.first_image_url || pair.first_crop_url) || !(pair.second_image_url || pair.second_crop_url);
    markLabel.append(mark, document.createTextNode("Mark as duplicate"));
    const keep = document.createElement("select");
    addOption(keep, String(pair.first_index), `Keep left · ${pair.first_image || Number(pair.first_index) + 1}`);
    addOption(keep, String(pair.second_index), `Keep right · ${pair.second_image || Number(pair.second_index) + 1}`);
    keep.value = String(decision?.keep_index ?? pair.first_index);
    keep.disabled = !mark.checked || mark.disabled;
    const persist = () => {
      keep.disabled = !mark.checked || mark.disabled;
      saveManualPairDecision(pair, pairIndex, mark.checked, keep.value);
    };
    mark.onchange = persist;
    keep.onchange = persist;
    manualControls.append(markLabel, keep);
    details.appendChild(manualControls);
  }

  async function navigateComparison(direction) {
    if (activeComparisonIndex == null) return;
    const target = activeComparisonIndex + direction;
    if (target < 0 || target >= loadedComparisonTotal) return;
    if (target >= loadedComparisonPairs.length) await loadComparisonPage(false);
    if (loadedComparisonPairs[target]) renderComparisonDetail(target);
  }

  function renderVisualReview(workflow) {
    const review = byId("ppVisualReview");
    const pairsHost = byId("ppVisualReviewPairs");
    const stats = workflow?.visual_analysis_stats;
    if (!stats || !workflow?.visual_review) {
      review.hidden = true;
      if (pairsHost.dataset.analysisLoading !== "true") pairsHost.replaceChildren();
      byId("ppViewVisualComparisons").hidden = true;
      return;
    }
    delete pairsHost.dataset.analysisLoading;
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
      };
      for (const [id, value] of Object.entries(fields)) {
        const field = byId(id);
        if (field && value != null) field.value = String(value);
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
      card.tabIndex = 0;
      card.setAttribute("role", "button");
      card.setAttribute("aria-label", `Open full images for comparison ${pairIndex + 1}`);
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
      const mapButton = document.createElement("button");
      mapButton.className = "secondary tiny postprocessPairMapButton";
      mapButton.type = "button";
      mapButton.textContent = "⌖ Map";
      mapButton.title = "View this pair on the map";
      mapButton.onclick = event => {
        event.preventDefault();
        event.stopPropagation();
        viewComparisonOnMap(pairIndex);
      };
      meta.appendChild(mapButton);
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
      mark.dataset.role = "manual-duplicate";
      mark.checked = Boolean(existingDecision);
      mark.disabled = !(pair.first_image_url || pair.first_crop_url) || !(pair.second_image_url || pair.second_crop_url);
      if (mark.disabled) mark.title = "Both source images are required for manual duplicate selection.";
      markLabel.append(mark, document.createTextNode("Duplicate"));
      const keep = document.createElement("select");
      keep.dataset.role = "manual-keep";
      addOption(keep, String(pair.first_index), `Keep left · ${Number(pair.first_index) + 1}`);
      addOption(keep, String(pair.second_index), `Keep right · ${Number(pair.second_index) + 1}`);
      keep.value = String(existingDecision?.keep_index ?? pair.first_index);
      keep.disabled = !mark.checked || mark.disabled;
      const saveManualDecision = () => {
        keep.disabled = !mark.checked || mark.disabled;
        saveManualPairDecision(pair, pairIndex, mark.checked, keep.value);
      };
      mark.addEventListener("change", saveManualDecision);
      keep.addEventListener("change", saveManualDecision);
      manualControls.append(markLabel, keep);
      card.append(images, meta, components, manualControls);
      card.onclick = event => {
        if (!manualControls.contains(event.target)) renderComparisonDetail(pairIndex);
      };
      card.onkeydown = event => {
        if ((event.key === "Enter" || event.key === " ") && !manualControls.contains(event.target)) {
          event.preventDefault();
          renderComparisonDetail(pairIndex);
        }
      };
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
    if (activeComparisonIndex != null) renderComparisonDetail(activeComparisonIndex);
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
    if (api()?.getContext()?.mode === mode) return;
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
    if (anomaly) void loadPanelLayers();
  }

  async function loadPanelLayers() {
    const context = api()?.getContext();
    const binding = context?.job?.workflows?.segmentation;
    const source = context?.job?.sources?.segmentation;
    const jobKey = `${context?.job?.id || ""}::${binding?.workflow_id || ""}`;
    if (panelLayersLoaded && panelLayersJobKey === jobKey) {
      refresh(api()?.getContext());
      return;
    }
    const select = byId("ppPanelReference");
    select.replaceChildren();
    addOption(select, "", "Loading identified panel layers…");
    select.disabled = true;
    try {
      if (!binding?.workflow_id || !source?.workspace_result_id) {
        panelLayers = [];
      } else {
        const status = await api().loadBoundWorkflowStatus(
          source.workspace_result_id,
          binding.workflow_id,
        );
        const panels = status.outputs?.regularized || status.outputs?.solar_panels;
        const rows = status.outputs?.solar_rows;
        panelLayers = panels?.url ? [{
          result_id: source.workspace_result_id,
          workflow_id: status.id,
          workflow_name: status.display_name || status.id,
          stage: status.outputs?.regularized ? "regularized" : "solar_panels",
          path: panels.path,
          url: panels.url,
          mtime: panels.mtime,
          rows_path: rows?.path || "",
          rows_url: rows?.url || null,
          rows_mtime: rows?.mtime || null,
        }] : [];
      }
      panelLayersLoaded = true;
      panelLayersJobKey = jobKey;
    } catch (error) {
      panelLayers = [];
      panelLayersJobKey = "";
      api().setMessage(`Could not load panel references: ${error.message}`, "err");
    }
    refresh(api()?.getContext());
  }

  async function showSegmentationReferences(option, context = api()?.getContext()) {
    if (!option?.dataset.url || context?.mode !== "anomaly") return;
    const referenceKey = `${option.dataset.url}::${option.dataset.mtime || ""}::${option.dataset.rowsUrl || ""}::${option.dataset.rowsMtime || ""}`;
    if (referenceKey === loadedPanelReferenceKey) return;
    loadedPanelReferenceKey = referenceKey;
    try {
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
          false,
          option.dataset.rowsMtime,
          null,
          false,
        );
      }
    } catch (_) {
      loadedPanelReferenceKey = "";
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
      const selectedOption = panelSelect.selectedOptions[0];
      void showSegmentationReferences(selectedOption, context);
    }

    byId("ppAdjustAnomaliesStep").hidden = false;
    byId("ppAssociateStep").hidden = false;
    byId("ppAdjustAnomaliesStep").classList.toggle("locked", !hasDeduplicated);
    byId("ppAssociateStep").classList.toggle("locked", !hasDeduplicated);
    byId("ppAdjustAnomaliesStep").setAttribute("aria-disabled", String(!hasDeduplicated));
    byId("ppAssociateStep").setAttribute("aria-disabled", String(!hasDeduplicated));
    byId("ppAssociate").disabled = !hasDeduplicated || !panelSelect.value;
    if (context.mode === "anomaly" && context.resultId && context.geojsonFiles.length) {
      void api()?.whenProcessingLayersReady().then(() => {
        if (api()?.getContext()?.mode === "anomaly") scheduleNeighborStats();
      });
    }
  }

  async function deduplicate() {
    const workspace = api();
    const context = workspace.getContext();
    const resultId = context.resultId || context.configuredResultId;
    const source = byId("ppAnomalyGeojson")?.value || context.configuredSourcePath;
    const button = document.getElementById("ppDeduplicate");
    if (!resultId || !source) {
      workspace.setMessage("The configured anomaly result or GeoJSON source is unavailable. Open Edit config and verify the anomaly source.", "err");
      return;
    }
    button.disabled = true;
    button.textContent = "Starting visual analysis…";
    workspace.setMessage("Starting anomaly deduplication…");
    openComparisonsModalLoading();
    try {
      const payload = await workspace.requestJson(
        `/api/results/${encodeURIComponent(resultId)}/postprocess/anomalies/deduplicate`,
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
      await loadComparisonPage(true);
    } catch (error) {
      workspace.setMessage(error.message, "err");
      button.disabled = false;
      button.textContent = "Analyze visual duplicates";
      const pairs = document.getElementById("ppVisualReviewPairs");
      delete pairs.dataset.analysisLoading;
      const errorMessage = document.createElement("p");
      errorMessage.className = "statusLine err";
      errorMessage.textContent = error.message;
      pairs.replaceChildren(errorMessage);
      document.getElementById("ppVisualComparisonsSummary").textContent = "Visual duplicate analysis could not be completed.";
    }
  }

  function bindAnalyzeVisualDuplicates() {
    const button = document.getElementById("ppDeduplicate");
    if (!button || button.dataset.analyzeHandlerBound === "true") return;
    button.dataset.analyzeHandlerBound = "true";
    button.onclick = () => void deduplicate();
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
    byId("ppApplyVisualDeduplication")?.addEventListener("click", applyVisualDeduplication);
    for (const id of Object.keys(scoringDefaults)) byId(id)?.addEventListener("input", refreshScoringControls);
    byId("ppDeduplicationMode")?.addEventListener("change", () => {
      refreshScoringControls();
      if (activeComparisonIndex != null) renderComparisonDetail(activeComparisonIndex);
    });
    byId("ppResetScoringDefaults")?.addEventListener("click", () => {
      for (const id of thresholdDefaultIds) byId(id).value = String(scoringDefaults[id]);
      refreshScoringControls();
    });
    const comparisonsModal = byId("ppVisualComparisonsModal");
    const closeComparisonsModal = () => {
      comparisonsModal.classList.remove("show");
      comparisonsModal.classList.add("hidden");
      showComparisonGrid();
    };
    byId("ppViewVisualComparisons")?.addEventListener("click", () => {
      showComparisonGrid();
      comparisonsModal.classList.remove("hidden");
      comparisonsModal.classList.add("show");
      byId("ppVisualComparisonsClose").focus();
      void loadComparisonPage(true);
    });
    byId("ppVisualComparisonsLoadMore")?.addEventListener("click", () => void loadComparisonPage(false));
    byId("ppVisualComparisonBack").onclick = showComparisonGrid;
    byId("ppVisualComparisonPrevious").onclick = () => void navigateComparison(-1);
    byId("ppVisualComparisonNext").onclick = () => void navigateComparison(1);
    byId("ppVisualComparisonZoomOut").onclick = () => setComparisonImageZoom(comparisonImageZoom - 0.5);
    byId("ppVisualComparisonZoomIn").onclick = () => setComparisonImageZoom(comparisonImageZoom + 0.5);
    byId("ppVisualComparisonFit").onclick = () => setComparisonImageZoom(1);
    byId("ppVisualComparisonViewOnMap").onclick = () => {
      if (activeComparisonIndex != null) viewComparisonOnMap(activeComparisonIndex);
    };
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
      if (!comparisonsModal?.classList.contains("hidden") && activeComparisonIndex != null) {
        if (event.key === "ArrowLeft") void navigateComparison(-1);
        else if (event.key === "ArrowRight") void navigateComparison(1);
      }
    });
    byId("ppAssociate")?.addEventListener("click", associate);
    document.addEventListener("postprocess:data", event => {
      const context = event.detail;
      const binding = context?.job?.workflows?.segmentation;
      const jobKey = `${context?.job?.id || ""}::${binding?.workflow_id || ""}`;
      if (context?.mode === "anomaly" && (!panelLayersLoaded || panelLayersJobKey !== jobKey)) {
        void loadPanelLayers();
      } else {
        refresh(context);
      }
    });
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
      panelLayers = [];
      panelLayersLoaded = false;
      panelLayersJobKey = "";
      loadedPanelReferenceKey = "";
      loadedScoringWorkflowId = "";
      lastNeighborStatsKey = "";
      pendingNeighborStatsKey = "";
      neighborStatsToken += 1;
      manualDuplicateDecisions.clear();
      loadedComparisonPairs = [];
      loadedComparisonTotal = 0;
      loadedComparisonWorkflowId = "";
      activeComparisonIndex = null;
    });
    document.addEventListener("postprocess:request-all-anomaly-pairs", () => void loadAllComparisonMapPairs());
    document.addEventListener("postprocess:return-comparisons", event => {
      const pairIndex = Number(event.detail?.pairIndex);
      comparisonsModal.classList.remove("hidden");
      comparisonsModal.classList.add("show");
      if (Number.isInteger(pairIndex) && loadedComparisonPairs[pairIndex]) renderComparisonDetail(pairIndex);
      else showComparisonGrid();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindRepresentativeWeightsModal, { once: true });
    document.addEventListener("DOMContentLoaded", bindAnalyzeVisualDuplicates, { once: true });
    document.addEventListener("DOMContentLoaded", init);
  } else {
    bindRepresentativeWeightsModal();
    bindAnalyzeVisualDuplicates();
    init();
  }
})();
