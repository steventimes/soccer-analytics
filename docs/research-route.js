"use strict";
// Historical research links keep their selection parameters and specific section.
const researchRoutes = {
  "match-explorer": "matches.html", "fifa-offers": "matches.html", case: "matches.html",
  comparisons: "entities.html", "style-retrieval": "entities.html",
  representations: "models.html", "representation-paired": "models.html", "vaep-calibration": "models.html", "xt-grid": "models.html",
  figures: "library.html", coverage: "library.html", methods: "library.html", literature: "library.html",
};
function researchRoute(href) {
  const target = new URL(href), section = target.hash.slice(1);
  target.pathname = target.pathname.replace(/[^/]*$/, researchRoutes[section] || "matches.html");
  if (!researchRoutes[section]) target.hash = "match-explorer";
  return target.href;
}
if (typeof module !== "undefined") module.exports = {researchRoute};
if (typeof location !== "undefined") location.replace(researchRoute(location.href));
