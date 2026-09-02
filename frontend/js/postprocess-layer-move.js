(() => {
  "use strict";

  const DEPENDENCIES = {
    segmentation: {
      source: ["Combined panels", "Regularized panels", "Rows", "Final anomaly associations"],
      combined: ["Regularized panels", "Rows", "Final anomaly associations"],
      regularized: ["Rows", "Final anomaly associations"],
      solar_rows: [],
    },
    anomaly: {
      source: ["Overlap-filtered anomalies", "Visual comparisons", "Visually deduplicated anomalies", "Final anomalies"],
      overlap_deduplicated: ["Visual comparisons", "Visually deduplicated anomalies", "Final anomalies"],
      deduplicated: ["Final anomalies"],
      associated: [],
    },
  };

  function create(options) {
    const {
      state,
      byId,
      confirmAction,
      requestJson,
      setMessage,
      lockControls,
      renderLayers,
      stopEditing,
      reloadAfterSave,
    } = options;

    function elementsFor(item) {
      const elements = new Set();
      item.layer?.eachLayer?.(layer => {
        const tileContainer = layer.getContainer?.();
        if (tileContainer) elements.add(tileContainer);
        const rendererContainer = layer._renderer?._container;
        if (rendererContainer) elements.add(rendererContainer);
        const element = layer.getElement?.();
        if (element && !element.classList?.contains("leaflet-interactive")) elements.add(element);
      });
      const directContainer = item.layer?.getContainer?.();
      if (directContainer) elements.add(directContainer);
      return [...elements];
    }

    function metresBetween(first, second) {
      const earthRadius = 6378137;
      const meanLatitude = ((first.lat + second.lat) / 2) * Math.PI / 180;
      return {
        east: (second.lng - first.lng) * Math.PI / 180 * earthRadius * Math.cos(meanLatitude),
        north: (second.lat - first.lat) * Math.PI / 180 * earthRadius,
      };
    }

    function pixelOffset(editing) {
      const bounds = editing.item.bounds || editing.item.layer?.getBounds?.();
      const center = bounds?.isValid?.() ? bounds.getCenter() : state.map.getCenter();
      const earthRadius = 6378137;
      const target = window.L.latLng(
        center.lat + editing.currentOffset.north / earthRadius * 180 / Math.PI,
        center.lng + editing.currentOffset.east / (earthRadius * Math.max(Math.cos(center.lat * Math.PI / 180), 0.01)) * 180 / Math.PI,
      );
      return state.map.latLngToLayerPoint(target).subtract(state.map.latLngToLayerPoint(center));
    }

    function updatePreview(message = "") {
      const editing = state.editing;
      if (!editing?.wholeLayer) return;
      const offset = pixelOffset(editing);
      for (const record of editing.previewElements) {
        record.element.style.transform = `${record.baseTransform} translate3d(${offset.x}px, ${offset.y}px, 0)`;
      }
      const eastInput = byId("ppLayerMoveEast");
      const northInput = byId("ppLayerMoveNorth");
      if (document.activeElement !== eastInput) eastInput.value = editing.currentOffset.east.toFixed(2);
      if (document.activeElement !== northInput) northInput.value = editing.currentOffset.north.toFixed(2);
      editing.dirty = Math.hypot(editing.currentOffset.east, editing.currentOffset.north) >= 0.001;
      byId("ppLayerMoveUndo").disabled = editing.historyIndex <= 0;
      byId("ppLayerMoveRedo").disabled = editing.historyIndex >= editing.history.length - 1;
      byId("ppLayerMoveSave").disabled = !editing.dirty;
      if (message) byId("ppLayerMoveStatus").textContent = message;
    }

    function recordOffset(message) {
      const editing = state.editing;
      if (!editing?.wholeLayer) return;
      const previous = editing.history[editing.historyIndex];
      if (Math.hypot(
        editing.currentOffset.east - previous.east,
        editing.currentOffset.north - previous.north,
      ) >= 0.001) {
        editing.history.splice(editing.historyIndex + 1);
        editing.history.push({ ...editing.currentOffset });
        if (editing.history.length > 20) editing.history.shift();
        editing.historyIndex = editing.history.length - 1;
      }
      updatePreview(message);
    }

    function updateOffsetFromInput(input, axis, commit) {
      const editing = state.editing;
      if (!editing?.wholeLayer) return;
      const value = Number(input.value);
      if (input.value.trim() === "" || !Number.isFinite(value) || Math.abs(value) > 10000) {
        if (commit) {
          editing.currentOffset[axis] = editing.history[editing.historyIndex][axis];
          input.value = editing.currentOffset[axis].toFixed(2);
          updatePreview("Enter a movement between -10,000 and 10,000 metres.");
        }
        return;
      }
      editing.currentOffset[axis] = value;
      if (commit) {
        input.value = value.toFixed(2);
        recordOffset("Manual movement recorded. Use Undo, Redo or Save movement.");
      } else {
        updatePreview("Previewing the entered movement. Finish editing the value to add it to history.");
      }
    }

    function detach({ restoreTransform = true } = {}) {
      const editing = state.editing;
      if (!editing?.wholeLayer) return;
      const container = state.map?.getContainer?.();
      if (container && editing.pointerHandlers) {
        for (const [event, handler] of Object.entries(editing.pointerHandlers)) {
          container.removeEventListener(`pointer${event}`, handler, true);
        }
      }
      if (editing.zoomHandler) state.map?.off("zoomend", editing.zoomHandler);
      if (restoreTransform) {
        for (const record of editing.previewElements || []) {
          record.element.style.transform = record.baseTransform;
          record.element.style.willChange = record.baseWillChange;
        }
      }
      for (const [name, wasEnabled] of Object.entries(editing.mapInteractions || {})) {
        if (wasEnabled) state.map?.[name]?.enable?.();
      }
      byId("ppMap").classList.remove("wholeLayerMoveMode", "wholeLayerDragging");
    }

    function install(editing) {
      const container = state.map.getContainer();
      let pointerId = null;
      let startLatLng = null;
      let startOffset = null;
      const down = event => {
        if (event.button !== 0 || event.target.closest(".leaflet-control, .leaflet-popup")) return;
        pointerId = event.pointerId;
        startLatLng = state.map.containerPointToLatLng(state.map.mouseEventToContainerPoint(event));
        startOffset = { ...editing.currentOffset };
        container.setPointerCapture?.(pointerId);
        byId("ppMap").classList.add("wholeLayerDragging");
        event.preventDefault();
        event.stopPropagation();
      };
      const move = event => {
        if (pointerId !== event.pointerId || !startLatLng) return;
        const current = state.map.containerPointToLatLng(state.map.mouseEventToContainerPoint(event));
        const delta = metresBetween(startLatLng, current);
        editing.currentOffset = {
          east: startOffset.east + delta.east,
          north: startOffset.north + delta.north,
        };
        updatePreview();
        event.preventDefault();
        event.stopPropagation();
      };
      const finish = (event, cancelled = false) => {
        if (pointerId !== event.pointerId) return;
        if (cancelled) editing.currentOffset = { ...editing.history[editing.historyIndex] };
        else recordOffset();
        container.releasePointerCapture?.(pointerId);
        pointerId = null;
        startLatLng = null;
        startOffset = null;
        byId("ppMap").classList.remove("wholeLayerDragging");
        updatePreview(cancelled
          ? "Movement cancelled. The previous position was restored."
          : "Whole-layer movement recorded. Use Undo, Redo or Save movement.");
        event.preventDefault();
        event.stopPropagation();
      };
      editing.pointerHandlers = {
        down,
        move,
        up: event => finish(event, false),
        cancel: event => finish(event, true),
      };
      editing.zoomHandler = () => updatePreview();
      for (const [event, handler] of Object.entries(editing.pointerHandlers)) {
        container.addEventListener(`pointer${event}`, handler, true);
      }
      state.map.on("zoomend", editing.zoomHandler);
    }

    for (const [id, axis] of [["ppLayerMoveEast", "east"], ["ppLayerMoveNorth", "north"]]) {
      const input = byId(id);
      input.addEventListener("input", () => updateOffsetFromInput(input, axis, false));
      input.addEventListener("change", () => updateOffsetFromInput(input, axis, true));
    }

    async function begin(target) {
      if (!state.currentJobId || !target?.item || state.editing) return;
      const isRaster = target.layerType === "raster";
      const confirmed = await confirmAction({
        title: `Move ${target.item.label}?`,
        message: isRaster
          ? "You are about to move the complete orthophoto or mosaic. Saving will create or replace one job-owned GeoTIFF working copy; the original raster remains unchanged."
          : "You are about to move every feature in this GeoJSON layer together. Individual polygons will not be draggable in this mode.",
        details: DEPENDENCIES[target.kind]?.[target.stage] || [],
        confirmLabel: "Start moving",
      });
      if (!confirmed) return;
      state.map?._solarolyMeasureControl?.cancel();
      const item = target.item;
      if (!state.map.hasLayer(item.layer)) item.layer.addTo(state.map);
      const previewElements = elementsFor(item).map(element => ({
        element,
        baseTransform: element.style.transform || "",
        baseWillChange: element.style.willChange || "",
      }));
      if (!previewElements.length) {
        setMessage("This layer has not finished rendering. Turn it on and try again.", "warn");
        return;
      }
      state.editing = {
        wholeLayer: true,
        owner: target.owner,
        kind: target.kind,
        stage: target.stage,
        layerType: target.layerType,
        item,
        dirty: false,
        currentOffset: { east: 0, north: 0 },
        history: [{ east: 0, north: 0 }],
        historyIndex: 0,
        previewElements,
        pointerHandlers: null,
        zoomHandler: null,
        mapInteractions: {},
      };
      for (const record of previewElements) record.element.style.willChange = "transform";
      for (const name of ["dragging", "scrollWheelZoom", "doubleClickZoom", "touchZoom", "boxZoom", "keyboard"]) {
        const handler = state.map?.[name];
        state.editing.mapInteractions[name] = Boolean(handler?.enabled?.());
        handler?.disable?.();
      }
      lockControls(true);
      byId("ppLayerMovePanel").hidden = false;
      byId("ppLayerMoveName").textContent = item.label;
      byId("ppLayerMoveIdentity").textContent = isRaster ? "Raster reference" : "GeoJSON layer";
      byId("ppLayerMoveSave").textContent = "Save movement";
      byId("ppLayerMoveStatus").textContent = "Move mode: drag anywhere on the map. Other layers remain visible for alignment.";
      byId("ppMap").classList.add("wholeLayerMoveMode");
      install(state.editing);
      updatePreview();
      renderLayers();
    }

    function restore(targetIndex) {
      const editing = state.editing;
      if (!editing?.wholeLayer || targetIndex < 0 || targetIndex >= editing.history.length || targetIndex === editing.historyIndex) return;
      editing.historyIndex = targetIndex;
      editing.currentOffset = { ...editing.history[targetIndex] };
      updatePreview(targetIndex === 0 ? "Returned to the original layer position." : "Movement history restored.");
    }

    async function save() {
      const editing = state.editing;
      if (!editing?.wholeLayer || !editing.dirty) return;
      const details = DEPENDENCIES[editing.kind]?.[editing.stage] || [];
      const existingRasterCopy = Boolean(
        editing.layerType === "raster"
        && state.currentJob?.raster_copies?.[editing.kind]?.paths?.length
      );
      let message;
      if (editing.layerType === "raster") {
        message = existingRasterCopy
          ? "The existing moved orthophoto/mosaic in this job will be replaced. The original test-run raster remains unchanged."
          : "The entire orthophoto/mosaic will be copied into this job and saved at the new position. This can take time and use approximately the raster's file size. The original test-run raster remains unchanged.";
      } else {
        message = details.length
          ? "The current job-owned vector layer will be replaced by its moved version, and dependent outputs will be archived as outdated. The original test-run GeoJSON remains unchanged."
          : "The current job-owned vector layer will be replaced by its moved version. The original test-run GeoJSON remains unchanged.";
      }
      const confirmed = await confirmAction({
        title: "Save whole-layer movement?",
        message,
        details,
        confirmLabel: "Save movement",
        danger: editing.layerType === "raster" || details.length > 0,
      });
      if (!confirmed || state.editing !== editing) return;
      const button = byId("ppLayerMoveSave");
      button.disabled = true;
      button.textContent = "Saving…";
      byId("ppLayerMoveStatus").textContent = `Saving ${editing.item.label} movement…`;
      try {
        const payload = await requestJson(`/api/postprocess-jobs/${encodeURIComponent(state.currentJobId)}/move-layer`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            kind: editing.kind,
            layer_type: editing.layerType,
            stage: editing.stage,
            east_m: editing.currentOffset.east,
            north_m: editing.currentOffset.north,
            confirm_move: true,
          }),
        });
        const archivedCount = payload.archived?.length || 0;
        state.currentJob = payload.job;
        stopEditing(false);
        await reloadAfterSave();
        setMessage(
          archivedCount
            ? `${editing.item.label} moved. ${archivedCount} dependent file${archivedCount === 1 ? " was" : "s were"} archived as outdated.`
            : `${editing.item.label} moved and saved.`,
          "ok",
        );
      } catch (error) {
        if (state.editing === editing) {
          button.disabled = false;
          button.textContent = "Save movement";
          byId("ppLayerMoveStatus").textContent = error.message;
        }
        setMessage(error.message, "err");
      }
    }

    return { begin, detach, restore, save, updatePreview };
  }

  window.PostprocessLayerMove = { create };
})();
