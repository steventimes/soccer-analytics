"use strict";

// Resolve the page's semantic colors once; SVG exports retain explicit colors.
(function () {
  const style = getComputedStyle(document.documentElement);
  const color = (name) => style.getPropertyValue(name).trim();
  window.TouchlineVisuals = Object.freeze({
    comparison: Object.freeze([
      Object.freeze({label: "A", symbol: "●", color: color("--series-a"), textColor: color("--series-a")}),
      Object.freeze({label: "B", symbol: "◇", color: color("--series-b"), textColor: color("--series-b-text")}),
    ]),
    timeMetrics: Object.freeze({
      shots: Object.freeze({field: "shots", label: "射门", unit: "次"}),
      network: Object.freeze({field: "completed_passes", label: "完成传球", unit: "次"}),
      matrix: Object.freeze({field: "completed_passes", label: "完成传球", unit: "次"}),
    }),
  });
})();
