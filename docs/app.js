async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderList(container, items, formatter, emptyMessage = "No data available yet.") {
  container.innerHTML = "";
  if (!items || items.length === 0) {
    container.innerHTML = `<p class="empty-state">${escapeHtml(emptyMessage)}</p>`;
    return;
  }
  items.forEach((item) => {
    const div = document.createElement("div");
    div.className = "item";
    div.innerHTML = formatter(item);
    container.appendChild(div);
  });
}

function formatConfidence(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatDateTime(value) {
  if (!value) {
    return "Not generated yet";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function predictionTitle(item) {
  return `${escapeHtml(item.home_team || "TBD")} vs ${escapeHtml(item.away_team || "TBD")}`;
}

function probabilityBars(probabilities = {}) {
  const entries = Object.entries(probabilities);
  if (entries.length === 0) {
    return "";
  }
  return `
    <div class="probabilities">
      ${entries.map(([label, value]) => `
        <div class="probability-row">
          <span>${escapeHtml(label)}</span>
          <div class="probability-track"><span style="width: ${Math.max(0, Math.min(100, Number(value) * 100))}%"></span></div>
          <strong>${formatConfidence(value)}</strong>
        </div>
      `).join("")}
    </div>
  `;
}

function updateMetrics(predictions, recommendedItems) {
  const fixtureMetric = document.getElementById("metric-fixtures");
  const pickMetric = document.getElementById("metric-picks");
  const confidenceMetric = document.getElementById("metric-confidence");

  fixtureMetric.textContent = String(predictions.length);
  pickMetric.textContent = String(recommendedItems.filter((item) => item.confidence !== null && item.confidence !== undefined).length);

  const confidenceValues = predictions
    .map((item) => Number(item.confidence))
    .filter((value) => !Number.isNaN(value));
  const average = confidenceValues.length
    ? confidenceValues.reduce((sum, value) => sum + value, 0) / confidenceValues.length
    : null;
  confidenceMetric.textContent = formatConfidence(average);
}

function populateCompetitionFilter(select, predictions) {
  const competitions = [...new Set(predictions.map((item) => item.competition).filter(Boolean))].sort();
  competitions.forEach((competition) => {
    const option = document.createElement("option");
    option.value = competition;
    option.textContent = competition;
    select.appendChild(option);
  });
}

function filterPredictions(predictions) {
  const competition = document.getElementById("competition-filter").value;
  const minimumConfidence = Number(document.getElementById("confidence-filter").value);
  return predictions.filter((item) => {
    const competitionMatches = competition === "all" || item.competition === competition;
    const confidenceMatches = Number(item.confidence || 0) >= minimumConfidence;
    return competitionMatches && confidenceMatches;
  });
}

function renderPredictions(predictions) {
  const predictionsContainer = document.getElementById("predictions");
  renderList(predictionsContainer, filterPredictions(predictions), (item) => `
    <div class="item-header">
      <div>
        <div class="item-title">${predictionTitle(item)}</div>
        <div class="item-meta">${escapeHtml(item.competition || "Unknown")} • ${escapeHtml(item.utc_date || "Date TBD")}</div>
      </div>
      <span class="badge">${escapeHtml(item.prediction || "Pending")}</span>
    </div>
    <div class="confidence-line">Confidence ${formatConfidence(item.confidence)}</div>
    ${probabilityBars(item.probabilities)}
  `, "No predictions match the selected filters.");
}


function renderReleaseGovernance(release) {
  const status = document.getElementById("release-status");
  const summary = document.getElementById("release-summary");
  const blockers = document.getElementById("release-blockers");
  const disclosure = document.getElementById("risk-disclosure");

  const decision = release?.release_decision || "unknown";
  status.textContent = decision.toUpperCase();
  status.className = `status-pill status-${decision === "approved" ? "healthy" : "attention"}`;

  summary.innerHTML = `
    <div class="operation-stat">
      <span>Can publish</span>
      <strong>${release?.can_publish_recommendations ? "Yes" : "No"}</strong>
    </div>
    <div class="operation-stat">
      <span>Prediction coverage</span>
      <strong>${escapeHtml(release?.audit?.prediction_count ?? 0)}</strong>
    </div>
    <div class="operation-stat">
      <span>High-confidence picks</span>
      <strong>${escapeHtml(release?.audit?.high_confidence_count ?? 0)}</strong>
    </div>
    <div class="operation-stat">
      <span>Average confidence</span>
      <strong>${formatConfidence(release?.audit?.average_confidence)}</strong>
    </div>
  `;

  renderList(blockers, release?.blockers || [], (item) => `
    <div class="item-header">
      <div>
        <div class="item-title">${escapeHtml(item.code || "BLOCKER")}</div>
        <div class="item-meta">${escapeHtml(item.message || "No details available.")}</div>
      </div>
      <span class="badge">blocker</span>
    </div>
  `, "No release blockers for the latest export.");

  const responsibleUse = release?.risk_disclosure?.responsible_use || [];
  disclosure.innerHTML = `
    <h3>Risk disclosure</h3>
    <p>${escapeHtml(release?.risk_disclosure?.summary || "Model output is probabilistic and requires human review.")}</p>
    <ul>
      ${responsibleUse.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
    </ul>
  `;
}

function renderOperationsSummary(summary) {
  const status = document.getElementById("operations-status");
  const summaryContainer = document.getElementById("operations-summary");
  const alertsContainer = document.getElementById("operations-alerts");

  const statusText = summary?.status || "unknown";
  status.textContent = statusText.toUpperCase();
  status.className = `status-pill status-${escapeHtml(statusText)}`;

  const coverage = summary?.coverage_by_competition || [];
  summaryContainer.innerHTML = `
    <div class="operation-stat">
      <span>Predictions</span>
      <strong>${escapeHtml(summary?.prediction_count ?? 0)}</strong>
    </div>
    <div class="operation-stat">
      <span>Competitions</span>
      <strong>${escapeHtml(summary?.competition_count ?? 0)}</strong>
    </div>
    <div class="operation-stat">
      <span>High-confidence picks</span>
      <strong>${escapeHtml(summary?.high_confidence_count ?? 0)}</strong>
    </div>
    <div class="operation-stat">
      <span>Run confidence</span>
      <strong>${formatConfidence(summary?.average_confidence)}</strong>
    </div>
  `;

  if (coverage.length > 0) {
    const coverageDiv = document.createElement("div");
    coverageDiv.className = "coverage-list";
    coverageDiv.innerHTML = coverage.map((item) => `
      <span>${escapeHtml(item.competition)} ${escapeHtml(item.prediction_count)}</span>
    `).join("");
    summaryContainer.appendChild(coverageDiv);
  }

  renderList(alertsContainer, summary?.alerts || [], (alert) => `
    <div class="item-header">
      <div>
        <div class="item-title">${escapeHtml(alert.code || "ALERT")}</div>
        <div class="item-meta">${escapeHtml(alert.message || "No details available.")}</div>
      </div>
      <span class="badge">${escapeHtml(alert.level || "info")}</span>
    </div>
  `, "No operational alerts for the latest export.");
}

async function init() {
  const recommendedContainer = document.getElementById("recommended-bets");
  const scoresContainer = document.getElementById("scores");
  const dataHealth = document.getElementById("data-health");
  const operationsSummary = document.getElementById("operations-summary");
  const competitionFilter = document.getElementById("competition-filter");
  const confidenceFilter = document.getElementById("confidence-filter");

  let predictions = [];
  let recommendedItems = [];

  try {
    const manifest = await loadJson("./data/manifest.json");
    dataHealth.textContent = `Last export: ${formatDateTime(manifest.generated_at)} • ${manifest.prediction_count ?? 0} predictions`;
  } catch (error) {
    dataHealth.textContent = "Export manifest unavailable. Showing cached data when present.";
  }

  try {
    const operationsData = await loadJson("./data/operations.json");
    renderOperationsSummary(operationsData);
  } catch (error) {
    operationsSummary.innerHTML = `<p class="empty-state">Operations summary is unavailable.</p>`;
  }

  try {
    const releaseData = await loadJson("./data/release.json");
    renderReleaseGovernance(releaseData);
  } catch (error) {
    document.getElementById("release-summary").innerHTML = `<p class="empty-state">Release governance is unavailable.</p>`;
  }

  try {
    const predictionsData = await loadJson("./data/predictions.json");
    predictions = predictionsData.predictions || [];
    populateCompetitionFilter(competitionFilter, predictions);
    renderPredictions(predictions);
    competitionFilter.addEventListener("change", () => renderPredictions(predictions));
    confidenceFilter.addEventListener("change", () => renderPredictions(predictions));
  } catch (error) {
    renderList(document.getElementById("predictions"), [], () => "", "Predictions are unavailable.");
  }

  try {
    const preset = await loadJson("./data/preset_questions.json");
    const recommended = (preset.questions || []).find(
      (question) => question.id === "recommended_bets_today"
    );
    recommendedItems = recommended?.items || [];
    renderList(recommendedContainer, recommendedItems, (item) => `
      <div class="item-header">
        <div>
          <div class="item-title">${escapeHtml(item.match || "No high-confidence picks yet")}</div>
          <div class="item-meta">${escapeHtml(item.competition || "All competitions")} • ${formatConfidence(item.confidence)}</div>
        </div>
        <span class="badge">${escapeHtml(item.prediction || "Hold")}</span>
      </div>
    `, "Recommended picks are unavailable.");
  } catch (error) {
    renderList(recommendedContainer, [], () => "", "Recommended picks are unavailable.");
  }

  updateMetrics(predictions, recommendedItems);

  try {
    const scoresData = await loadJson("./data/scores.json");
    renderList(scoresContainer, scoresData.scores, (item) => `
      <div class="item-header">
        <div>
          <div class="item-title">${escapeHtml(item.home_team)} ${escapeHtml(item.home_score ?? "")} - ${escapeHtml(item.away_score ?? "")} ${escapeHtml(item.away_team)}</div>
          <div class="item-meta">${escapeHtml(item.league || "")} • ${escapeHtml(item.date || "")} ${escapeHtml(item.time || "")}</div>
        </div>
      </div>
    `, "Scores are unavailable.");
  } catch (error) {
    renderList(scoresContainer, [], () => "", "Scores are unavailable.");
  }
}

init();
