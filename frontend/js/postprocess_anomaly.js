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
  let representativeModalSnapshot = null;
  const manualDuplicateDecisions = new Map();
  let loadedComparisonPairs = [];
  let loadedComparisonTotal = 0;
  let loadedComparisonWorkflowId = "";
  let loadedComparisonMapPairs = [];
  let loadedComparisonMapWorkflowId = "";
  let activeComparisonIndex = null;
  let comparisonImageZoom = 1;
  let anomalyStepPhase = null;

  function setAnomalyStepCollapsed(step, collapsed) {
    if (!step) return;
    step.classList.toggle("collapsed", collapsed);
    const toggle = step.querySelector(".postprocessStepCollapse");
    if (!toggle) return;
    toggle.textContent = collapsed ? "+" : "−";
    toggle.title = collapsed ? "Expand step" : "Minimize step";
    toggle.setAttribute("aria-expanded", String(!collapsed));
  }

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

  function showComparisonsWorkspace() {
    byId("ppAnomalyControls").querySelector(".postprocessSteps").hidden = true;
    byId("ppVisualComparisonsWorkspace").hidden = false;
  }

  function closeComparisonsWorkspace() {
    byId("ppVisualComparisonsWorkspace").hidden = true;
    byId("ppAnomalyControls").querySelector(".postprocessSteps").hidden = false;
    showComparisonGrid();
    api()?.clearAnomalyReviewMap();
  }

  function openComparisonsWorkspaceLoading() {
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
    showComparisonsWorkspace();
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
    const valid = validateDuplicateWeights(config);
    const validation = byId("ppScoringValidation");
    validation.textContent = !valid
      ? "Duplicate-matching weights must total 100%. Open Configure weights to correct them."
      : "The weighted score is the only automatic deduplication threshold.";
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
      return score != null && score >= config.threshold;
    }).length;
    const comparedTotal = Number(workflow?.visual_analysis_stats?.visually_compared_pairs || 0);
    const displayed = pairs.length;
    const totalPairs = Number(workflow?.visual_review?.total_pairs || pairs.length);
    const estimate = displayed && totalPairs > displayed
      ? ` At least ${matching} of the ${displayed} displayed pairs qualify; all ${comparedTotal.toLocaleString()} comparisons will be evaluated when applied.`
      : ` ${matching.toLocaleString()} pair${matching === 1 ? "" : "s"} will be deduplicated.`;
    byId("ppVisualThresholdEffect").textContent = manualMode
      ? `${manualDuplicateDecisions.size.toLocaleString()} pair${manualDuplicateDecisions.size === 1 ? " is" : "s are"} marked as duplicate. Open image comparisons to change manual selections.`
      : `${config.threshold.toLocaleString()}% weighted duplicate score or higher.${estimate}`;
    for (const card of byId("ppVisualReviewPairs").querySelectorAll(".postprocessVisualPair")) {
      const pair = pairs[Number(card.dataset.pairIndex)];
      const score = compositeScore(pair, config);
      const qualifies = score != null && score >= config.threshold;
      const edgeKey = `${Math.min(pair.first_index, pair.second_index)}:${Math.max(pair.first_index, pair.second_index)}`;
      card.classList.toggle("isDuplicate", manualMode ? manualDuplicateDecisions.has(edgeKey) : qualifies);
      const scoreLabel = card.querySelector(".postprocessVisualSimilarity");
      if (scoreLabel) scoreLabel.textContent = score == null ? "Not compared" : `${Math.round(score)}% duplicate`;
    }
    byId("ppApplyVisualDeduplication").disabled = !valid || workflow?.status !== "complete";
    byId("ppApplyVisualDeduplication").textContent = manualMode ? "Apply manual selections" : "Apply visual deduplication";
    byId("ppVisualSimilarity").disabled = manualMode;
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
    const decision = manualDuplicateDecisions.get(comparisonEdgeKey(pair));
    const status = pair.manual_review_status || "unreviewed";
    const detailControls = activeComparisonIndex === pairIndex
      ? byId("ppVisualComparisonDetails")?.querySelector(".postprocessComparisonManualControls")
      : null;
    if (card) card.classList.toggle("isRejected", status === "rejected");
    for (const controls of [card?.querySelector(".postprocessManualPairControls"), detailControls].filter(Boolean)) {
      const keep = controls.querySelector('[data-role="manual-keep"]');
      if (keep) {
        keep.value = String(decision?.keep_index ?? pair.first_index);
        keep.disabled = false;
      }
      for (const button of controls.querySelectorAll("[data-decision]")) {
        const active = button.dataset.decision === status;
        const action = button.dataset.decision === "accepted" ? "accept duplicate" : "reject duplicate";
        button.classList.toggle("isActive", active);
        button.setAttribute("aria-pressed", String(active));
        button.title = active ? `Undo ${action}` : `${action[0].toUpperCase()}${action.slice(1)}`;
        button.setAttribute("aria-label", button.title);
      }
    }
    applyComparisonFilter();
  }

  function setLocalPairDecision(pair, status, keepIndex) {
    const edgeKey = comparisonEdgeKey(pair);
    pair.manual_review_status = status === "unreviewed" ? undefined : status;
    if (status === "accepted") {
      pair.manual_keep_index = Number(keepIndex);
      manualDuplicateDecisions.set(edgeKey, {
        first_index: Number(pair.first_index),
        second_index: Number(pair.second_index),
        keep_index: Number(keepIndex),
      });
    } else {
      delete pair.manual_keep_index;
      manualDuplicateDecisions.delete(edgeKey);
    }
  }

  function syncCachedMapPairDecision(pair) {
    const edgeKey = comparisonEdgeKey(pair);
    const cached = loadedComparisonMapPairs.find(candidate => comparisonEdgeKey(candidate) === edgeKey);
    if (!cached) return;
    if (pair.manual_review_status) cached.manual_review_status = pair.manual_review_status;
    else delete cached.manual_review_status;
    if (pair.manual_keep_index != null) cached.manual_keep_index = pair.manual_keep_index;
    else delete cached.manual_keep_index;
  }

  function switchToManualReview() {
    const mode = byId("ppDeduplicationMode");
    if (mode.value !== "manual") mode.value = "manual";
  }

  async function saveManualPairDecision(pair, pairIndex, status, keepIndex) {
    const previousStatus = pair.manual_review_status || "unreviewed";
    const previousKeep = pair.manual_keep_index;
    switchToManualReview();
    setLocalPairDecision(pair, status, keepIndex);
    syncCachedMapPairDecision(pair);
    api()?.updateAnomalyReviewPairDecision(
      pair.first_index,
      pair.second_index,
      mapReviewPair(pair).review_status,
    );
    syncPairCardDecision(pair, pairIndex);
    const workflow = currentAnomalyWorkflow();
    if (workflow) updateThresholdEffect(workflow);
    try {
      const context = api()?.getContext();
      await api()?.requestJson(
        `/api/results/${encodeURIComponent(context.resultId)}/postprocess/${encodeURIComponent(context.workflowId)}/visual-review/decision`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            first_index: Number(pair.first_index),
            second_index: Number(pair.second_index),
            status,
            keep_index: status === "accepted" ? Number(keepIndex) : null,
          }),
        },
      );
      return true;
    } catch (error) {
      setLocalPairDecision(pair, previousStatus, previousKeep ?? pair.first_index);
      syncCachedMapPairDecision(pair);
      api()?.updateAnomalyReviewPairDecision(
        pair.first_index,
        pair.second_index,
        mapReviewPair(pair).review_status,
      );
      syncPairCardDecision(pair, pairIndex);
      if (workflow) updateThresholdEffect(workflow);
      api()?.setMessage(`Could not save the comparison decision: ${error.message}`, "err");
      return false;
    }
  }

  function pairMatchesFilter(pair, filter = byId("ppVisualComparisonFilter")?.value || "active") {
    const status = pair.manual_review_status || "unreviewed";
    if (filter === "all") return true;
    if (filter === "active") return status !== "rejected";
    return status === filter;
  }

  function applyComparisonFilter() {
    const pairs = comparisonPairs(currentAnomalyWorkflow());
    const filter = byId("ppVisualComparisonFilter")?.value || "active";
    let visible = 0;
    for (const card of byId("ppVisualReviewPairs")?.querySelectorAll(".postprocessVisualPair") || []) {
      const show = pairMatchesFilter(pairs[Number(card.dataset.pairIndex)], filter);
      card.classList.toggle("isFilteredOut", !show);
      if (show) visible += 1;
    }
    const empty = byId("ppVisualReviewPairs")?.querySelector(".postprocessComparisonFilterEmpty");
    if (empty) empty.hidden = visible > 0;
  }

  function setComparisonFilter(value) {
    const select = byId("ppVisualComparisonFilter");
    const menu = byId("ppVisualComparisonFilterMenu");
    const button = byId("ppVisualComparisonFilterButton");
    if (!select || !menu || !button) return;
    select.value = value;
    const label = select.selectedOptions[0]?.textContent || "Active";
    button.title = `Filter comparisons: ${label}`;
    for (const option of menu.querySelectorAll("[data-comparison-filter]")) {
      const active = option.dataset.comparisonFilter === value;
      option.classList.toggle("isActive", active);
      option.querySelector("span").textContent = active ? "✓" : "";
    }
    menu.hidden = true;
    button.setAttribute("aria-expanded", "false");
    applyComparisonFilter();
  }

  function leaveFilteredComparison(pairIndex) {
    const pairs = comparisonPairs(currentAnomalyWorkflow());
    const next = pairs.findIndex((pair, index) => index > pairIndex && pairMatchesFilter(pair));
    if (next >= 0) {
      renderComparisonDetail(next);
      return;
    }
    for (let index = pairIndex - 1; index >= 0; index -= 1) {
      if (!pairMatchesFilter(pairs[index])) continue;
      renderComparisonDetail(index);
      return;
    }
    showComparisonGrid();
  }

  function buildPairDecisionControls(pair, pairIndex, detailed = false) {
    const controls = document.createElement("div");
    controls.className = `postprocessManualPairControls${detailed ? " postprocessComparisonManualControls" : ""}`;
    const buttons = document.createElement("div");
    buttons.className = "postprocessPairDecisionButtons";
    const keep = document.createElement("select");
    keep.dataset.role = "manual-keep";
    addOption(keep, String(pair.first_index), `Keep left · ${pair.first_image || Number(pair.first_index) + 1}`);
    addOption(keep, String(pair.second_index), `Keep right · ${pair.second_image || Number(pair.second_index) + 1}`);
    keep.value = String(pair.manual_keep_index ?? pair.first_index);
    keep.disabled = false;
    keep.title = "Choose which prediction to keep. Changing this accepts the duplicate pair.";
    for (const [status, symbol, title] of [
      ["accepted", "✓", "Accept duplicate"],
      ["rejected", "×", "Reject duplicate"],
    ]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "secondary postprocessPairDecisionButton";
      button.dataset.decision = status;
      button.textContent = symbol;
      button.title = pair.manual_review_status === status ? `Undo ${title.toLowerCase()}` : title;
      button.setAttribute("aria-label", button.title);
      button.setAttribute("aria-pressed", String(pair.manual_review_status === status));
      button.onclick = async event => {
        event.preventDefault();
        event.stopPropagation();
        const nextStatus = pair.manual_review_status === status ? "unreviewed" : status;
        const saved = await saveManualPairDecision(pair, pairIndex, nextStatus, keep.value);
        if (saved && detailed && !pairMatchesFilter(pair)) leaveFilteredComparison(pairIndex);
      };
      buttons.appendChild(button);
    }
    keep.onchange = async event => {
      event.stopPropagation();
      const saved = await saveManualPairDecision(pair, pairIndex, "accepted", keep.value);
      if (saved && detailed && !pairMatchesFilter(pair)) leaveFilteredComparison(pairIndex);
    };
    controls.append(buttons, keep);
    return controls;
  }

  function mapReviewPair(pair) {
    const config = scoringConfig();
    const score = compositeScore(pair, config);
    const qualifies = score != null && score >= config.threshold;
    return {
      ...pair,
      display_score: score,
      review_status: pair.manual_review_status || (qualifies ? "duplicate" : "below"),
    };
  }

  function viewComparisonOnMap(pairIndex) {
    const workflow = currentAnomalyWorkflow();
    const pairs = comparisonPairs(workflow).map(mapReviewPair);
    api()?.showAnomalyReviewPair(pairs, pairIndex);
    void loadAllComparisonMapPairs();
  }

  async function loadAllComparisonMapPairs() {
    const workspace = api();
    const context = workspace?.getContext();
    if (!context?.resultId || !context.workflowId) return;
    if (loadedComparisonMapWorkflowId === context.workflowId && loadedComparisonMapPairs.length) {
      workspace.updateAnomalyReviewPairs(loadedComparisonMapPairs.map(mapReviewPair));
      return;
    }
    try {
      const payload = await workspace.requestJson(
        `/api/results/${encodeURIComponent(context.resultId)}/postprocess/${encodeURIComponent(context.workflowId)}/visual-review-map`,
        { cache: "no-store" },
      );
      loadedComparisonMapWorkflowId = context.workflowId;
      loadedComparisonMapPairs = payload.pairs || [];
      workspace.updateAnomalyReviewPairs(loadedComparisonMapPairs.map(mapReviewPair));
    } catch (error) {
      workspace.setMessage(`Could not show all duplicate candidates on the map: ${error.message}`, "err");
    }
  }

  function showComparisonGrid() {
    activeComparisonIndex = null;
    byId("ppVisualComparisonsHeader").hidden = false;
    byId("ppVisualComparisonDetail").hidden = true;
    byId("ppVisualReviewPairs").hidden = false;
    byId("ppVisualComparisonsTitle").textContent = "Image comparisons";
    const loadMore = byId("ppVisualComparisonsLoadMore");
    loadMore.hidden = loadedComparisonPairs.length >= loadedComparisonTotal;
    byId("ppVisualComparisonFilterWrap").hidden = false;
    applyComparisonFilter();
  }

  function focusBoxScale(box) {
    if (!Array.isArray(box) || box.length !== 4) return null;
    const width = Math.max(0.001, Number(box[2]) - Number(box[0]));
    const height = Math.max(0.001, Number(box[3]) - Number(box[1]));
    return Math.max(1, Math.min(8, 0.38 / Math.max(width, height)));
  }

  function applyComparisonImageTransform(image) {
    const panX = Number(image.dataset.panX || 0);
    const panY = Number(image.dataset.panY || 0);
    const basePanX = comparisonImageZoom > 1 ? Number(image.dataset.basePanX || 0) : 0;
    const basePanY = comparisonImageZoom > 1 ? Number(image.dataset.basePanY || 0) : 0;
    image.style.transform = `translate(${basePanX + panX}px, ${basePanY + panY}px) scale(${comparisonImageZoom})`;
  }

  function centerComparisonImage(image) {
    const frame = image.parentElement;
    if (!frame || !image.naturalWidth || !image.naturalHeight) return;
    const frameWidth = frame.clientWidth;
    const frameHeight = frame.clientHeight;
    const containedScale = Math.min(frameWidth / image.naturalWidth, frameHeight / image.naturalHeight);
    const contentWidth = image.naturalWidth * containedScale;
    const contentHeight = image.naturalHeight * containedScale;
    const contentLeft = (frameWidth - contentWidth) / 2;
    const contentTop = (frameHeight - contentHeight) / 2;
    const focusX = contentLeft + Number(image.dataset.focusX || 50) / 100 * contentWidth;
    const focusY = contentTop + Number(image.dataset.focusY || 50) / 100 * contentHeight;
    image.style.transformOrigin = `${focusX}px ${focusY}px`;
    image.dataset.basePanX = String(frameWidth / 2 - focusX);
    image.dataset.basePanY = String(frameHeight / 2 - focusY);
    applyComparisonImageTransform(image);
  }

  function setComparisonImageZoom(zoom, resetPan = false) {
    comparisonImageZoom = Math.max(1, Math.min(10, Number(zoom) || 1));
    for (const image of byId("ppVisualComparisonFullImages").querySelectorAll("img[data-focus-x]")) {
      if (resetPan || comparisonImageZoom === 1) {
        image.dataset.panX = "0";
        image.dataset.panY = "0";
      }
      applyComparisonImageTransform(image);
    }
    byId("ppVisualComparisonZoomLevel").textContent = `${Math.round(comparisonImageZoom * 100)}%`;
    byId("ppVisualComparisonZoomOut").disabled = comparisonImageZoom <= 1;
  }

  function bindComparisonImageInteraction(frame, image) {
    let drag = null;
    image.dataset.panX = "0";
    image.dataset.panY = "0";
    frame.addEventListener("wheel", event => {
      event.preventDefault();
      setComparisonImageZoom(comparisonImageZoom + (event.deltaY < 0 ? 0.5 : -0.5));
    }, { passive: false });
    frame.addEventListener("pointerdown", event => {
      if (comparisonImageZoom <= 1) return;
      drag = {
        x: event.clientX,
        y: event.clientY,
        panX: Number(image.dataset.panX || 0),
        panY: Number(image.dataset.panY || 0),
      };
      frame.setPointerCapture(event.pointerId);
      frame.classList.add("isPanning");
    });
    frame.addEventListener("pointermove", event => {
      if (!drag) return;
      image.dataset.panX = String(drag.panX + event.clientX - drag.x);
      image.dataset.panY = String(drag.panY + event.clientY - drag.y);
      applyComparisonImageTransform(image);
    });
    const stop = event => {
      if (!drag) return;
      drag = null;
      frame.classList.remove("isPanning");
      if (frame.hasPointerCapture(event.pointerId)) frame.releasePointerCapture(event.pointerId);
    };
    frame.addEventListener("pointerup", stop);
    frame.addEventListener("pointercancel", stop);
  }

  function renderComparisonDetail(pairIndex) {
    const pairs = comparisonPairs(currentAnomalyWorkflow());
    const pair = pairs[pairIndex];
    if (!pair) return;
    activeComparisonIndex = pairIndex;
    byId("ppVisualComparisonsHeader").hidden = true;
    byId("ppVisualReviewPairs").hidden = true;
    byId("ppVisualComparisonDetail").hidden = false;
    byId("ppVisualComparisonsLoadMore").hidden = true;
    byId("ppVisualComparisonFilterWrap").hidden = true;
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
    for (const [side, imageUrl, cropUrl, name, focusBox, anomalyId] of [
      ["Left", pair.first_image_url, pair.first_crop_url, pair.first_image, pair.first_focus_box, pair.first_anomaly_id ?? Number(pair.first_index) + 1],
      ["Right", pair.second_image_url, pair.second_crop_url, pair.second_image, pair.second_focus_box, pair.second_anomaly_id ?? Number(pair.second_index) + 1],
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
        bindComparisonImageInteraction(frame, image);
        image.addEventListener("load", () => {
          window.requestAnimationFrame(() => centerComparisonImage(image));
        }, { once: true });
        if (image.complete) window.requestAnimationFrame(() => centerComparisonImage(image));
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
      const identifier = document.createElement("small");
      identifier.className = "postprocessComparisonAnomalyId";
      identifier.textContent = `Anomaly ID: ${anomalyId}`;
      caption.appendChild(identifier);
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

    details.appendChild(buildPairDecisionControls(pair, pairIndex, true));
    syncPairCardDecision(pair, pairIndex);
    viewComparisonOnMap(pairIndex);
  }

  async function toggleComparisonFullscreen() {
    const workspace = byId("ppVisualComparisonsWorkspace");
    try {
      if (document.fullscreenElement === workspace) await document.exitFullscreen();
      else await workspace.requestFullscreen();
    } catch (error) {
      api()?.setMessage(`Could not open the comparison fullscreen: ${error.message}`, "err");
    }
  }

  async function navigateComparison(direction) {
    if (activeComparisonIndex == null) return;
    const target = activeComparisonIndex + direction;
    if (target < 0 || target >= loadedComparisonTotal) return;
    if (target >= loadedComparisonPairs.length) await loadComparisonPage(false);
    if (comparisonPairs(currentAnomalyWorkflow())[target]) renderComparisonDetail(target);
  }

  function renderVisualReview(workflow) {
    const review = byId("ppVisualReview");
    const pairsHost = byId("ppVisualReviewPairs");
    const applyButton = byId("ppApplyVisualDeduplication");
    const stats = workflow?.visual_analysis_stats;
    const viewButton = byId("ppViewVisualComparisons");
    const savedTotal = Number(
      workflow?.visual_review?.total_pairs
      ?? workflow?.visual_review_total_pairs
      ?? 0
    );
    const hasSavedReview = Boolean(
      workflow?.visual_review_available
      || workflow?.visual_review_path
      || workflow?.visual_review
    );
    const workflowRunning = workflow?.status === "queued" || workflow?.status === "running";
    viewButton.hidden = !hasSavedReview;
    viewButton.disabled = !hasSavedReview || workflowRunning;
    viewButton.textContent = savedTotal > 0
      ? `Review image comparisons (${savedTotal.toLocaleString()})`
      : "Review saved image comparisons";
    applyButton.hidden = !workflow?.visual_review;
    if (!workflow?.visual_review) {
      review.hidden = true;
      if (pairsHost.dataset.analysisLoading !== "true") pairsHost.replaceChildren();
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
    const candidateCount = Number(stats?.spatial_candidate_pairs ?? workflow.visual_review.total_pairs ?? 0);
    const comparedCount = Number(stats?.visually_compared_pairs || 0);
    const missingCount = Number(stats?.missing_image_pairs || 0);
    byId("ppVisualReviewSummary").textContent = stats
      ? `${candidateCount.toLocaleString()} candidate pairs from images within ${Number(stats.neighbor_image_radius_m || 0).toLocaleString()} m · anomaly shift up to ${Number(stats.maximum_location_shift_m || 0).toLocaleString()} m · ${comparedCount.toLocaleString()} visually compared${missingCount ? ` · ${missingCount.toLocaleString()} kept because imagery was unavailable` : ""}`
      : `${candidateCount.toLocaleString()} saved comparison pair${candidateCount === 1 ? "" : "s"} restored from this job.`;
    byId("ppVisualComparisonsSummary").textContent = byId("ppVisualReviewSummary").textContent;
    pairsHost.replaceChildren();
    const pairs = comparisonPairs(workflow);
    for (const [pairIndex, pair] of pairs.entries()) {
      if (pair.manual_review_status === "accepted") {
        setLocalPairDecision(pair, "accepted", pair.manual_keep_index ?? pair.first_index);
      } else if (pair.manual_review_status === "rejected") {
        setLocalPairDecision(pair, "rejected", pair.first_index);
      }
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
      const manualControls = buildPairDecisionControls(pair, pairIndex);
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
      syncPairCardDecision(pair, pairIndex);
    }
    const filterEmpty = document.createElement("p");
    filterEmpty.className = "muted tiny postprocessComparisonFilterEmpty";
    filterEmpty.textContent = "No comparisons match this filter.";
    filterEmpty.hidden = true;
    pairsHost.appendChild(filterEmpty);
    const totalPairs = loadedComparisonTotal || Number(review.total_pairs || 0);
    if (totalPairs > pairs.length) {
      const note = document.createElement("p");
      note.className = "muted tiny";
      note.textContent = `Showing ${pairs.length.toLocaleString()} of ${totalPairs.toLocaleString()} candidate pairs.`;
      pairsHost.appendChild(note);
    }
    viewButton.hidden = false;
    viewButton.disabled = false;
    viewButton.textContent = `Review image comparisons (${totalPairs.toLocaleString()})`;
    const loadMore = byId("ppVisualComparisonsLoadMore");
    loadMore.hidden = pairs.length >= totalPairs;
    loadMore.disabled = false;
    updateThresholdEffect(workflow);
    applyComparisonFilter();
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
      for (const pair of payload.pairs || []) {
        if (pair.manual_review_status === "accepted") {
          switchToManualReview();
          setLocalPairDecision(pair, "accepted", pair.manual_keep_index ?? pair.first_index);
        } else if (pair.manual_review_status === "rejected") {
          switchToManualReview();
          setLocalPairDecision(pair, "rejected", pair.first_index);
        }
      }
      loadedComparisonTotal = Number(payload.total_pairs || loadedComparisonPairs.length);
      const workflow = current.workflows.find(item => item.id === workflowId && item.workflow_kind === "anomaly");
      if (workflow) {
        workflow.visual_review = {
          ...(workflow.visual_review || {}),
          pairs: loadedComparisonPairs.slice(0, 12),
          total_pairs: loadedComparisonTotal,
          displayed_pairs: Math.min(12, loadedComparisonPairs.length),
        };
        renderVisualReview(workflow);
      }
    } catch (error) {
      api()?.setMessage(`Could not load image comparisons: ${error.message}`, "err");
      const errorMessage = document.createElement("p");
      errorMessage.className = "statusLine err";
      errorMessage.textContent = `Could not load saved comparisons: ${error.message}`;
      byId("ppVisualReviewPairs").replaceChildren(errorMessage);
    } finally {
      button.disabled = false;
      button.textContent = "Load more";
    }
  }

  function switchMode(mode) {
    if (api()?.getContext()?.mode === mode) return;
    closeComparisonsWorkspace();
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
    byId("ppPanelReferenceStatus").textContent = "Loading this job’s final regularized panels…";
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

  function currentPanelLayer(context = api()?.getContext()) {
    const available = context?.segmentationResultId
      ? panelLayers.filter(layer => layer.result_id === context.segmentationResultId)
      : panelLayers;
    return available[0] || null;
  }

  async function showSegmentationReferences(layer, context = api()?.getContext()) {
    if (!layer?.url || context?.mode !== "anomaly") return;
    const referenceKey = `${layer.url}::${layer.mtime || ""}::${layer.rows_url || ""}::${layer.rows_mtime || ""}`;
    if (referenceKey === loadedPanelReferenceKey) return;
    try {
      const panels = await api()?.loadPreviewLayer(
        "segmentation_regularized_reference",
        layer.url,
        null,
        "Final regularized panels (read-only)",
        true,
        layer.mtime,
      );
      if (!panels) throw new Error("The panel layer request was interrupted.");
      if (layer.rows_url) {
        const rows = await api()?.loadPreviewLayer(
          "segmentation_rows_reference",
          layer.rows_url,
          null,
          "Final rows (visual reference)",
          true,
          layer.rows_mtime,
          null,
          false,
        );
        if (!rows) throw new Error("The row layer request was interrupted.");
      }
      loadedPanelReferenceKey = referenceKey;
      byId("ppPanelReferenceStatus").textContent = layer.rows_url
        ? "Using this job’s final regularized panels and rows as read-only references."
        : "Using this job’s final regularized panels. Final rows will appear after panel and row IDs are assigned in Segmentation.";
    } catch (error) {
      loadedPanelReferenceKey = "";
      byId("ppPanelReferenceStatus").textContent = `Could not load final segmentation references: ${error.message}`;
    }
  }

  function refresh(context = api()?.getContext()) {
    if (!context) return;
    const workflow = context.workflows.find(item => item.id === context.workflowId && item.workflow_kind === "anomaly");
    const hasOverlapDeduplicated = Boolean(workflow?.overlap_deduplicate_stats);
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
    byId("ppOverlapDeduplicateStep").hidden = false;
    byId("ppOverlapDeduplicateStep").classList.toggle("locked", !scanned);
    byId("ppOverlapDeduplicateStep").setAttribute("aria-disabled", String(!scanned));
    byId("ppRemoveOverlappingAnomalies").disabled = !scanned || workflowRunning;
    byId("ppRemoveOverlappingAnomalies").textContent = workflowRunning && workflow?.stage === "overlap_deduplicate"
      ? "Removing overlapping polygons…"
      : hasOverlapDeduplicated ? "Re-run overlap removal" : "Remove overlapping polygons";
    byId("ppDeduplicateStep").hidden = false;
    byId("ppDeduplicateStep").classList.toggle("locked", !hasOverlapDeduplicated);
    byId("ppDeduplicateStep").setAttribute("aria-disabled", String(!hasOverlapDeduplicated));
    byId("ppDeduplicate").disabled = !hasOverlapDeduplicated || workflowRunning;
    byId("ppDeduplicate").textContent = workflowRunning && workflow?.stage === "deduplicate"
      ? "Analyzing visual duplicates…"
      : "Analyze visual duplicates";
    byId("ppSkipVisualDeduplication").disabled = !hasOverlapDeduplicated || workflowRunning;

    const panelLayer = currentPanelLayer(context);
    byId("ppPanelReferenceStatus").textContent = panelLayer
      ? panelLayer.rows_url
        ? "Using this job’s final regularized panels and rows as read-only references."
        : "Using this job’s final regularized panels. Final rows will appear after panel and row IDs are assigned in Segmentation."
      : panelLayersLoaded
        ? "Final regularized panels are unavailable. Complete segmentation for this job first."
        : "Loading this job’s final regularized panels…";
    if (context.mode === "anomaly" && panelLayer?.url) void showSegmentationReferences(panelLayer, context);

    byId("ppAdjustAnomaliesStep").hidden = false;
    byId("ppAssociateStep").hidden = false;
    byId("ppAdjustAnomaliesStep").classList.toggle("locked", !hasDeduplicated);
    byId("ppAssociateStep").classList.toggle("locked", !hasDeduplicated);
    byId("ppAdjustAnomaliesStep").setAttribute("aria-disabled", String(!hasDeduplicated));
    byId("ppAssociateStep").setAttribute("aria-disabled", String(!hasDeduplicated));
    byId("ppAssociate").disabled = !hasDeduplicated || !panelLayer;
    const phase = workflow?.outputs?.associated
      ? 0
      : workflow?.deduplicate_stats ? 3
        : hasOverlapDeduplicated ? 2 : 1;
    if (phase !== anomalyStepPhase) {
      anomalyStepPhase = phase;
      [
        byId("ppOverlapDeduplicateStep"),
        byId("ppDeduplicateStep"),
        byId("ppAdjustAnomaliesStep"),
        byId("ppAssociateStep"),
      ].forEach((step, index) => setAnomalyStepCollapsed(step, phase === 0 || index + 1 !== phase));
    }
    if (context.mode === "anomaly" && context.resultId && context.geojsonFiles.length) {
      void api()?.whenProcessingLayersReady().then(() => {
        if (api()?.getContext()?.mode === "anomaly") scheduleNeighborStats();
      });
    }
  }

  async function removeOverlappingAnomalies() {
    const workspace = api();
    const context = workspace.getContext();
    const resultId = context.resultId || context.configuredResultId;
    const source = byId("ppAnomalyGeojson")?.value || context.configuredSourcePath;
    const button = byId("ppRemoveOverlappingAnomalies");
    if (!resultId || !source) {
      workspace.setMessage("The configured anomaly source is unavailable. Open Edit config and verify it.", "err");
      return;
    }
    button.disabled = true;
    button.textContent = "Removing overlapping polygons…";
    workspace.setMessage("Removing overlapping anomaly polygons…");
    try {
      const payload = await workspace.requestJson(
        `/api/results/${encodeURIComponent(resultId)}/postprocess/anomalies/overlap-deduplicate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            input_path: source,
            output_name: byId("ppAnomalyOutputName").value.trim() || "anomaly_postprocess",
            minimum_overlap_percent: Number(byId("ppAnomalyOverlapPercent").value),
          }),
        },
      );
      await workspace.runWorkflow(payload);
    } catch (error) {
      workspace.setMessage(error.message, "err");
      button.disabled = false;
      button.textContent = "Remove overlapping polygons";
    }
  }

  async function deduplicate() {
    const workspace = api();
    const context = workspace.getContext();
    const resultId = context.resultId || context.configuredResultId;
    const workflow = context.workflows.find(item => item.id === context.workflowId && item.workflow_kind === "anomaly");
    const source = workflow?.overlap_input_path || workflow?.outputs?.deduplicated?.path;
    const button = document.getElementById("ppDeduplicate");
    if (!resultId || !source) {
      workspace.setMessage("The configured anomaly result or GeoJSON source is unavailable. Open Edit config and verify the anomaly source.", "err");
      return;
    }
    button.disabled = true;
    button.textContent = "Starting visual analysis…";
    workspace.setMessage("Starting anomaly deduplication…");
    loadedComparisonPairs = [];
    loadedComparisonTotal = 0;
    loadedComparisonWorkflowId = context.workflowId || "";
    activeComparisonIndex = null;
    manualDuplicateDecisions.clear();
    openComparisonsWorkspaceLoading();
    try {
      const payload = await workspace.requestJson(
        `/api/results/${encodeURIComponent(resultId)}/postprocess/anomalies/deduplicate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            input_path: source,
            workflow_id: context.workflowId,
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
    const panelLayer = currentPanelLayer(context);
    if (!context.resultId || !context.workflowId || !panelLayer?.path) return;
    byId("ppAssociate").disabled = true;
    workspace.setMessage("Starting anomaly-to-panel association…");
    try {
      const payload = await workspace.requestJson(
        `/api/results/${encodeURIComponent(context.resultId)}/postprocess/${encodeURIComponent(context.workflowId)}/associate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            panel_path: panelLayer.path,
            panel_result_id: panelLayer.result_id,
            panel_workflow_id: panelLayer.workflow_id,
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
    byId("ppApplyVisualDeduplication")?.addEventListener("click", applyVisualDeduplication);
    byId("ppRemoveOverlappingAnomalies")?.addEventListener("click", removeOverlappingAnomalies);
    byId("ppSkipVisualDeduplication")?.addEventListener("click", () => {
      setAnomalyStepCollapsed(byId("ppDeduplicateStep"), true);
      setAnomalyStepCollapsed(byId("ppAdjustAnomaliesStep"), false);
      byId("ppAdjustAnomaliesStep")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    for (const id of Object.keys(scoringDefaults)) byId(id)?.addEventListener("input", refreshScoringControls);
    byId("ppDeduplicationMode")?.addEventListener("change", () => {
      refreshScoringControls();
      if (activeComparisonIndex != null) renderComparisonDetail(activeComparisonIndex);
    });
    byId("ppViewVisualComparisons")?.addEventListener("click", () => {
      showComparisonGrid();
      showComparisonsWorkspace();
      if (!comparisonPairs(currentAnomalyWorkflow()).length) {
        const loading = document.createElement("div");
        loading.className = "postprocessComparisonsLoading";
        loading.innerHTML = '<span class="spinner" aria-hidden="true"></span><span>Loading saved image comparisons…</span>';
        byId("ppVisualReviewPairs").replaceChildren(loading);
      }
      void loadComparisonPage(true);
    });
    byId("ppVisualComparisonsBackToSteps")?.addEventListener("click", closeComparisonsWorkspace);
    byId("ppVisualComparisonsLoadMore")?.addEventListener("click", () => void loadComparisonPage(false));
    byId("ppVisualComparisonFilterButton")?.addEventListener("click", event => {
      event.stopPropagation();
      const menu = byId("ppVisualComparisonFilterMenu");
      const open = menu.hidden;
      menu.hidden = !open;
      event.currentTarget.setAttribute("aria-expanded", String(open));
    });
    for (const option of byId("ppVisualComparisonFilterMenu")?.querySelectorAll("[data-comparison-filter]") || []) {
      option.addEventListener("click", () => setComparisonFilter(option.dataset.comparisonFilter));
    }
    document.addEventListener("click", event => {
      const wrap = byId("ppVisualComparisonFilterWrap");
      if (wrap?.contains(event.target)) return;
      const menu = byId("ppVisualComparisonFilterMenu");
      const button = byId("ppVisualComparisonFilterButton");
      if (menu) menu.hidden = true;
      button?.setAttribute("aria-expanded", "false");
    });
    setComparisonFilter("active");
    byId("ppVisualComparisonBack").onclick = () => {
      showComparisonGrid();
      api()?.clearAnomalyReviewMap();
    };
    byId("ppVisualComparisonPrevious").onclick = () => void navigateComparison(-1);
    byId("ppVisualComparisonNext").onclick = () => void navigateComparison(1);
    byId("ppVisualComparisonZoomOut").onclick = () => setComparisonImageZoom(comparisonImageZoom - 0.5);
    byId("ppVisualComparisonZoomIn").onclick = () => setComparisonImageZoom(comparisonImageZoom + 0.5);
    byId("ppVisualComparisonFit").onclick = () => setComparisonImageZoom(1, true);
    byId("ppVisualComparisonViewOnMap").onclick = () => {
      if (activeComparisonIndex != null) viewComparisonOnMap(activeComparisonIndex);
    };
    byId("ppVisualComparisonFullscreen").onclick = () => void toggleComparisonFullscreen();
    document.addEventListener("fullscreenchange", () => {
      const button = byId("ppVisualComparisonFullscreen");
      const active = document.fullscreenElement === byId("ppVisualComparisonsWorkspace");
      button?.setAttribute("aria-pressed", String(active));
      if (button) {
        button.title = active ? "Exit comparison fullscreen" : "View comparison fullscreen";
        button.setAttribute("aria-label", button.title);
      }
      for (const image of byId("ppVisualComparisonFullImages")?.querySelectorAll("img[data-focus-x]") || []) {
        centerComparisonImage(image);
      }
    });
    window.addEventListener("resize", () => {
      if (activeComparisonIndex == null) return;
      for (const image of byId("ppVisualComparisonFullImages")?.querySelectorAll("img[data-focus-x]") || []) {
        centerComparisonImage(image);
      }
    });
    document.addEventListener("keydown", event => {
      const representativeModal = document.getElementById("ppRepresentativeWeightsModal");
      if (event.key === "Escape" && !representativeModal?.classList.contains("hidden")) {
        closeRepresentativeWeightsModal(false);
      }
      if (event.key === "Escape" && !byId("ppVisualComparisonsWorkspace")?.hidden) {
        closeComparisonsWorkspace();
      }
      if (!byId("ppVisualComparisonsWorkspace")?.hidden && activeComparisonIndex != null) {
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
      if (event.detail.context?.mode === "segmentation"
        && status?.status === "complete"
        && (status?.outputs?.regularized || status?.outputs?.solar_panels)) {
        panelLayersLoaded = false;
        panelLayersJobKey = "";
        loadedPanelReferenceKey = "";
        return;
      }
      if (event.detail.context?.mode !== "anomaly"
        || status?.status !== "complete"
        || !status?.outputs?.associated
        || !status?.association_stats?.panel_updated_mtime) return;
      const panelLayer = currentPanelLayer(event.detail.context);
      if (!panelLayer?.url) return;
      panelLayer.mtime = String(status.association_stats.panel_updated_mtime);
      loadedPanelReferenceKey = "";
      void showSegmentationReferences(panelLayer, event.detail.context);
      api()?.invalidateCachedMode("segmentation");
    });
    document.addEventListener("postprocess:cache-reset", () => {
      closeComparisonsWorkspace();
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
      loadedComparisonMapPairs = [];
      loadedComparisonMapWorkflowId = "";
      activeComparisonIndex = null;
      anomalyStepPhase = null;
    });
    document.addEventListener("postprocess:return-comparisons", event => {
      const pairIndex = Number(event.detail?.pairIndex);
      showComparisonsWorkspace();
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
