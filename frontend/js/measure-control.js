(function () {
  "use strict";

  const EARTH_RADIUS_METRES = 6378137;
  let controlSequence = 0;

  function formatLength(metres) {
    if (metres >= 1000) return `${(metres / 1000).toFixed(metres >= 10000 ? 1 : 2)} km`;
    return `${metres.toFixed(metres >= 100 ? 0 : 1)} m`;
  }

  function formatArea(squareMetres) {
    if (squareMetres >= 1000000) return `${(squareMetres / 1000000).toFixed(2)} km²`;
    if (squareMetres >= 10000) return `${(squareMetres / 10000).toFixed(2)} ha`;
    return `${squareMetres.toFixed(squareMetres >= 100 ? 0 : 1)} m²`;
  }

  function polygonArea(points) {
    if (points.length < 3) return 0;
    const radians = Math.PI / 180;
    let area = 0;
    for (let index = 0; index < points.length; index += 1) {
      const current = points[index];
      const next = points[(index + 1) % points.length];
      area += (next.lng - current.lng) * radians
        * (2 + Math.sin(current.lat * radians) + Math.sin(next.lat * radians));
    }
    return Math.abs(area * EARTH_RADIUS_METRES * EARTH_RADIUS_METRES / 2);
  }

  function addMeasureControl(map, { position = "topleft", onStart = null } = {}) {
    if (!map || !window.L) return null;
    if (map._solarolyMeasureControl) return map._solarolyMeasureControl;

    const id = `solarolyMeasure${++controlSequence}`;
    const completedLayers = window.L.layerGroup().addTo(map);
    const draftLayer = window.L.layerGroup().addTo(map);
    let mode = null;
    let points = [];
    let button = null;
    let menu = null;
    let doubleClickZoomWasEnabled = false;

    function renderDraft(cursorPoint = null) {
      draftLayer.clearLayers();
      const previewPoints = cursorPoint && points.length ? [...points, cursorPoint] : points;
      if (!previewPoints.length) return;
      previewPoints.forEach((point, index) => {
        if (cursorPoint && index === previewPoints.length - 1) return;
        window.L.circleMarker(point, {
          radius: 4, color: "#e0f2fe", weight: 2, fillColor: "#0284c7", fillOpacity: 1, interactive: false,
        }).addTo(draftLayer);
      });
      if (previewPoints.length > 1) {
        const style = { color: "#38bdf8", weight: 3, opacity: 0.95, dashArray: "7 6", interactive: false };
        if (mode === "area" && previewPoints.length > 2) {
          window.L.polygon(previewPoints, { ...style, fillOpacity: 0.12 }).addTo(draftLayer);
        } else window.L.polyline(previewPoints, style).addTo(draftLayer);
      }
    }

    function setInactive() {
      mode = null;
      points = [];
      draftLayer.clearLayers();
      map.off("click", addPoint);
      map.off("mousemove", previewPoint);
      map.off("dblclick", finishFromMap);
      document.removeEventListener("keydown", handleKeydown);
      map.getContainer().classList.remove("solarolyMeasureActive");
      if (doubleClickZoomWasEnabled) map.doubleClickZoom.enable();
      button?.classList.remove("active");
      button?.setAttribute("aria-pressed", "false");
      button?.setAttribute("aria-expanded", "false");
    }

    function cancel() {
      setInactive();
      if (menu) menu.hidden = true;
    }

    function resultPopup(content, group) {
      const wrapper = document.createElement("div");
      wrapper.className = "solarolyMeasureResult";
      const result = document.createElement("div");
      result.innerHTML = content;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "solarolyMeasureRemove";
      remove.textContent = "Remove measurement";
      remove.onclick = () => {
        completedLayers.removeLayer(group);
        map.closePopup();
      };
      wrapper.append(result, remove);
      return wrapper;
    }

    function finish() {
      const minimumPoints = mode === "area" ? 3 : 2;
      if (!mode || points.length < minimumPoints) {
        cancel();
        return;
      }
      const resultPoints = [...points];
      const resultMode = mode;
      setInactive();
      const group = window.L.layerGroup().addTo(completedLayers);
      const style = { color: "#0284c7", weight: 3, opacity: 0.95, fillOpacity: 0.12 };
      let shape;
      let content;
      if (resultMode === "area") {
        shape = window.L.polygon(resultPoints, style).addTo(group);
        const perimeter = resultPoints.reduce((total, point, index) => (
          total + map.distance(point, resultPoints[(index + 1) % resultPoints.length])
        ), 0);
        content = `<strong>Area</strong><br>${formatArea(polygonArea(resultPoints))}<br><span>Perimeter: ${formatLength(perimeter)}</span>`;
      } else {
        shape = window.L.polyline(resultPoints, style).addTo(group);
        const length = resultPoints.slice(1).reduce((total, point, index) => (
          total + map.distance(resultPoints[index], point)
        ), 0);
        content = `<strong>Length</strong><br>${formatLength(length)}`;
      }
      resultPoints.forEach(point => window.L.circleMarker(point, {
        radius: 3, color: "#e0f2fe", weight: 1, fillColor: "#0284c7", fillOpacity: 1, interactive: false,
      }).addTo(group));
      shape.bindPopup(resultPopup(content, group), { autoPan: false }).openPopup();
    }

    function addPoint(event) {
      if (!mode) return;
      const previous = points[points.length - 1];
      if (!previous || previous.distanceTo(event.latlng) > 0.01) points.push(event.latlng);
      renderDraft();
    }

    function previewPoint(event) {
      if (mode && points.length) renderDraft(event.latlng);
    }

    function finishFromMap(event) {
      window.L.DomEvent.preventDefault(event.originalEvent);
      window.L.DomEvent.stopPropagation(event.originalEvent);
      finish();
    }

    function handleKeydown(event) {
      if (event.key === "Escape") cancel();
      if (event.key === "Enter") finish();
    }

    function start(nextMode) {
      if (typeof onStart === "function" && onStart(nextMode) === false) return;
      cancel();
      mode = nextMode;
      doubleClickZoomWasEnabled = map.doubleClickZoom.enabled();
      map.doubleClickZoom.disable();
      map.getContainer().classList.add("solarolyMeasureActive");
      button?.classList.add("active");
      button?.setAttribute("aria-pressed", "true");
      map.on("click", addPoint);
      map.on("mousemove", previewPoint);
      map.on("dblclick", finishFromMap);
      document.addEventListener("keydown", handleKeydown);
    }

    const MeasureControl = window.L.Control.extend({
      options: { position },
      onAdd() {
        const container = window.L.DomUtil.create("div", "leaflet-bar solarolyMeasureControl");
        button = window.L.DomUtil.create("button", "solarolyMeasureToggle", container);
        button.id = `${id}Toggle`;
        button.type = "button";
        button.title = "Measure length or area";
        button.setAttribute("aria-label", "Measure length or area");
        button.setAttribute("aria-expanded", "false");
        button.setAttribute("aria-pressed", "false");
        button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 17 17 5l2 2L7 19 5 17Zm3-1-2-2m5-1-2-2m5-1-2-2m5-1-2-2"/></svg>';
        menu = window.L.DomUtil.create("div", "solarolyMeasureMenu", container);
        menu.hidden = true;
        [["length", "Measure length"], ["area", "Measure area"]].forEach(([value, label]) => {
          const item = window.L.DomUtil.create("button", "solarolyMeasureMenuItem", menu);
          item.type = "button";
          item.textContent = label;
          item.onclick = event => {
            event.preventDefault();
            menu.hidden = true;
            button.setAttribute("aria-expanded", "false");
            start(value);
          };
        });
        button.onclick = event => {
          event.preventDefault();
          if (mode) cancel();
          menu.hidden = !menu.hidden;
          button.setAttribute("aria-expanded", String(!menu.hidden));
        };
        window.L.DomEvent.disableClickPropagation(container);
        window.L.DomEvent.disableScrollPropagation(container);
        return container;
      },
      onRemove() {
        cancel();
        map.removeLayer(draftLayer);
        map.removeLayer(completedLayers);
      },
    });

    const control = new MeasureControl();
    control.addTo(map);
    control.cancel = cancel;
    map._solarolyMeasureControl = control;
    return control;
  }

  window.addStandardMeasureControl = addMeasureControl;
})();
