(function () {
  "use strict";

  function addStandardMeasureControl(map, options = {}) {
    if (!map) return null;
    if (map._solarolyMeasureControl) return map._solarolyMeasureControl;

    if (!window.L?.control?.measure) {
      console.error("Leaflet Measure is unavailable; the measurement control was not added.");
      return null;
    }

    const control = window.L.control.measure({
      position: "topleft",
      primaryLengthUnit: "meters",
      secondaryLengthUnit: "kilometers",
      primaryAreaUnit: "sqmeters",
      secondaryAreaUnit: "hectares",
      activeColor: "#38bdf8",
      completedColor: "#0284c7",
      captureZIndex: 10000,
      ...options,
    });

    control.addTo(map);
    map._solarolyMeasureControl = control;
    return control;
  }

  window.addStandardMeasureControl = addStandardMeasureControl;
})();
