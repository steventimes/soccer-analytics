const state = {
  predictions: [],
  predictionsAvailable: false,
};

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  const raw = await response.text();
  return { payload: JSON.parse(raw), raw };
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function percent(value, digits = 0) {
  const numeric = finiteNumber(value);
  return numeric === null ? "—" : `${(numeric * 100).toFixed(digits)}%`;
}

function formatDateTime(value) {
  if (!value) return "Snapshot time unavailable";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function formatKickoff(value) {
  if (!value) return "Kickoff pending";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function setEmpty(container, title, message) {
  container.innerHTML = `
    <div class="empty-state">
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(message)}</span>
    </div>
  `;
}

function outcomeLabel(value) {
  return {
    HOME_WIN: "Home win",
    DRAW: "Draw",
    AWAY_WIN: "Away win",
  }[value] || value || "Pending";
}

function renderCoverage(manifest, operations) {
  const coverageAvailable = Boolean(manifest || operations);
  const coverage = operations?.model_coverage || manifest?.model_coverage || {};
  const loaded = Math.max(0, finiteNumber(coverage.loaded) ?? 0);
  const requested = Math.max(0, finiteNumber(coverage.requested) ?? 0);
  const ratio = requested > 0 ? Math.min(1, loaded / requested) : 0;
  const track = document.getElementById("model-coverage-track");
  document.getElementById("coverage-fill").style.width = `${ratio * 100}%`;
  document.getElementById("coverage-ratio").textContent = requested ? `${loaded} / ${requested}` : "— / —";
  document.getElementById("coverage-copy").textContent = requested
    ? loaded === requested
      ? "Every requested competition is ready"
      : `${requested - loaded} competition model${requested - loaded === 1 ? " needs" : "s need"} attention`
    : coverageAvailable ? "No competition models reported" : "Coverage artifact unavailable";
  track.setAttribute("aria-valuenow", String(Math.round(ratio * 100)));

  const labels = operations?.coverage_by_competition || [];
  document.getElementById("coverage-labels").innerHTML = labels.length
    ? labels.map((item) => `<span><b>${escapeHtml(item.competition)}</b>${escapeHtml(item.prediction_count)} fixtures</span>`).join("")
    : coverageAvailable
      ? "<span><b>No fixture coverage</b>Run training and prediction to populate this line.</span>"
      : "<span><b>Coverage unknown</b>Restore manifest.json or operations.json.</span>";
}

function recommendationsApproved(release) {
  return release?.release_decision === "approved"
    && release?.can_publish_recommendations === true;
}

function renderMetrics(predictions, operations, predictionsAvailable, release) {
  if (!predictionsAvailable) {
    ["metric-fixtures", "metric-picks", "metric-confidence", "metric-cold-start"].forEach((id) => {
      document.getElementById(id).textContent = "—";
    });
    return;
  }
  const confidences = predictions.map((item) => finiteNumber(item.confidence)).filter((value) => value !== null);
  const average = confidences.length
    ? confidences.reduce((sum, value) => sum + value, 0) / confidences.length
    : null;
  document.getElementById("metric-fixtures").textContent = String(predictions.length);
  document.getElementById("metric-picks").textContent = recommendationsApproved(release)
    ? String(confidences.filter((value) => value >= 0.6).length)
    : "0";
  document.getElementById("metric-confidence").textContent = percent(average, 1);
  document.getElementById("metric-cold-start").textContent = predictions.length
    ? percent(operations?.cold_start_ratio, 0)
    : "—";
}

function signalMarkup(item, fallbackLevel = "warning") {
  const level = item.level || fallbackLevel;
  return `
    <article class="signal signal-${escapeHtml(level)}">
      <span class="signal-mark" aria-hidden="true"></span>
      <div>
        <strong>${escapeHtml(item.code || "NOTICE")}</strong>
        <p>${escapeHtml(item.message || "No additional detail was supplied.")}</p>
      </div>
    </article>
  `;
}

function renderOperations(operations) {
  const container = document.getElementById("operations-summary");
  const alerts = document.getElementById("operations-alerts");
  const status = operations?.status || "unknown";
  const statusNode = document.getElementById("operations-status");
  statusNode.textContent = status;
  statusNode.className = `status-token status-${status}`;
  if (!operations) {
    setEmpty(container, "Operations unavailable", "The operations artifact could not be loaded.");
    setEmpty(alerts, "Operations artifact unavailable", "Quality checks are unknown; do not treat this snapshot as passed.");
    return;
  }
  const coverage = operations?.model_coverage || {};
  const hasFixtures = (finiteNumber(operations.prediction_count) ?? 0) > 0;
  const qualityRows = operations.model_quality_by_competition || [];
  const backtestsReady = qualityRows.filter((item) => finiteNumber(item.backtest_folds) !== null).length;
  const fallbackModels = (operations.models_recovered_from_previous || []).length;
  const staleModels = (operations.models_stale || []).length;
  const unstableModels = (operations.models_temporally_unstable || []).length;
  container.innerHTML = `
    <div><span>Models ready</span><strong>${escapeHtml(coverage.loaded ?? "—")} / ${escapeHtml(coverage.requested ?? "—")}</strong></div>
    <div><span>Cold-start share</span><strong>${hasFixtures ? percent(operations.cold_start_ratio) : "—"}</strong></div>
    <div><span>Mean uncertainty</span><strong>${percent(operations?.average_uncertainty)}</strong></div>
    <div><span>Backtests ready</span><strong>${escapeHtml(backtestsReady)} / ${escapeHtml(coverage.loaded ?? "—")}</strong></div>
    <div><span>Fallback models</span><strong>${escapeHtml(fallbackModels)}</strong></div>
    <div><span>Stale models</span><strong>${escapeHtml(staleModels)}</strong></div>
    <div><span>Unstable models</span><strong>${escapeHtml(unstableModels)}</strong></div>
  `;
  const qualityByCompetition = new Map(
    qualityRows.map((item) => [item.competition, item]),
  );
  const qualityItems = (operations.models_below_baseline || []).map((competition) => {
    const item = qualityByCompetition.get(competition);
    if (!item) {
      return {
        level: "critical",
        code: `MODEL_BELOW_BASELINE / ${competition}`,
        message: "Quality detail unavailable; do not publish until validation evidence is restored.",
      };
    }
    return {
      level: "critical",
      code: `MODEL_BELOW_BASELINE / ${item.competition}`,
      message: `Validation ${percent(item.validation_accuracy)} versus baseline ${percent(item.baseline_accuracy)} (${percent(item.accuracy_lift, 1)} lift).`,
    };
  });
  const backtestItems = qualityRows
    .filter((item) => finiteNumber(item.backtest_folds) !== null)
    .map((item) => {
      const technology = [item.estimator, item.estimator_version]
        .filter((value) => typeof value === "string" && value.trim())
        .map((value) => escapeHtml(value.trim()))
        .join(" ");
      const explanationMethod = typeof item.explanation_method === "string" && item.explanation_method.trim()
        ? ` · ${escapeHtml(item.explanation_method.trim().replaceAll("_", " "))}`
        : "";
      const boostingRounds = finiteNumber(item.boosting_rounds) === null
        ? ""
        : ` · ${escapeHtml(Math.round(finiteNumber(item.boosting_rounds)))} rounds`;
      const eloBaseline = finiteNumber(item.backtest_elo_baseline_accuracy);
      const majorityBaseline = finiteNumber(item.backtest_majority_baseline_accuracy);
      const domainBaselines = eloBaseline === null || majorityBaseline === null
        ? ""
        : ` · Elo ${percent(eloBaseline)} / majority ${percent(majorityBaseline)}`;
      const classLabels = ["Away", "Draw", "Home"];
      const reliability = Array.isArray(item.backtest_reliability)
        ? item.backtest_reliability.map((entry) => {
          const classIndex = Number(entry.class_index);
          const label = classLabels[classIndex] || `Class ${classIndex}`;
          return `${label} ECE ${percent(entry.calibration_error)} (${escapeHtml(entry.sample_count ?? "—")} samples)`;
        }).join(" · ")
        : "Class reliability unavailable";
      return {
        level: "healthy",
        code: `BACKTEST / ${item.competition || "UNKNOWN"}`,
        message: `${technology ? `${technology}${explanationMethod}${boostingRounds} · ` : ""}${escapeHtml(item.backtest_folds)} expanding folds · accuracy ${percent(item.backtest_accuracy)} versus ${percent(item.backtest_baseline_accuracy)} strongest baseline${domainBaselines} · RPS ${finiteNumber(item.backtest_ranked_probability_score) === null ? "—" : escapeHtml(finiteNumber(item.backtest_ranked_probability_score).toFixed(3))} · calibration error ${percent(item.backtest_calibration_error)} · ${reliability} · sharpness ${percent(item.backtest_sharpness)} · worst-fold lift ${percent(item.backtest_worst_fold_accuracy_lift, 1)} · fold σ ${percent(item.backtest_accuracy_lift_stddev, 1)}${finiteNumber(item.model_age_hours) === null ? "" : ` · model age ${escapeHtml(Math.round(finiteNumber(item.model_age_hours)))}h`}.`,
      };
    });
  const items = [...qualityItems, ...backtestItems, ...(operations.alerts || [])];
  if (items.length) {
    alerts.innerHTML = items.map((item) => signalMarkup(item)).join("");
  } else {
    setEmpty(alerts, "No quality alerts", "Coverage and model checks passed for this snapshot.");
  }
}

function renderRelease(release) {
  const decision = release?.release_decision || "unknown";
  const approved = decision === "approved";
  const status = document.getElementById("release-status");
  status.textContent = approved ? "Publish" : decision === "blocked" ? "Hold" : "Unknown";
  status.className = `status-token status-${approved ? "healthy" : decision === "blocked" ? "critical" : "unknown"}`;
  document.getElementById("release-command-title").textContent = approved
    ? "Recommendations cleared"
    : decision === "blocked" ? "Hold recommendations" : "Release status unavailable";
  document.getElementById("release-command-copy").textContent = approved
    ? "Coverage, model quality, and confidence gates passed."
    : decision === "blocked"
      ? "At least one quality gate must be resolved before publishing."
      : "Load the latest release snapshot before publishing any pick.";

  const audit = release?.audit || {};
  document.getElementById("release-command-facts").innerHTML = `
    <span><b>${escapeHtml(audit.prediction_count ?? 0)}</b> fixtures</span>
    <span><b>${escapeHtml(audit.high_confidence_count ?? 0)}</b> shortlist</span>
    <span><b>${percent(audit.average_confidence)}</b> confidence</span>
  `;
  document.getElementById("release-summary").innerHTML = `
    <div><span>Decision</span><strong>${escapeHtml(decision)}</strong></div>
    <div><span>Publishable</span><strong>${release?.can_publish_recommendations ? "Yes" : "No"}</strong></div>
    <div><span>Blockers</span><strong>${escapeHtml((release?.blockers || []).length)}</strong></div>
    <div><span>Warnings</span><strong>${escapeHtml((release?.warnings || []).length)}</strong></div>
  `;

  const blockers = document.getElementById("release-blockers");
  if (release?.blockers?.length) {
    blockers.innerHTML = release.blockers.map((item) => signalMarkup(item, "critical")).join("");
  } else if (approved) {
    setEmpty(blockers, "Release gate passed", "This snapshot may be published with the risk disclosure visible.");
  } else {
    setEmpty(blockers, "No gate file loaded", "Generate release.json through the site export pipeline.");
  }

  const disclosure = release?.risk_disclosure;
  if (disclosure) {
    document.getElementById("risk-disclosure").innerHTML = `
      <p>${escapeHtml(disclosure.summary)}</p>
      <ul>${(disclosure.responsible_use || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    `;
  }
}

function populateCompetitionFilter(predictions) {
  const select = document.getElementById("competition-filter");
  select.innerHTML = '<option value="all">All competitions</option>';
  [...new Set(predictions.map((item) => item.competition).filter(Boolean))]
    .sort()
    .forEach((competition) => {
      const option = document.createElement("option");
      option.value = competition;
      option.textContent = competition;
      select.appendChild(option);
    });
}

function filteredPredictions() {
  const query = document.getElementById("prediction-search").value.trim().toLowerCase();
  const competition = document.getElementById("competition-filter").value;
  const confidence = Number(document.getElementById("confidence-filter").value);
  const sort = document.getElementById("prediction-sort").value;
  const filtered = state.predictions.filter((item) => {
    const teams = `${item.home_team || ""} ${item.away_team || ""}`.toLowerCase();
    return (!query || teams.includes(query))
      && (competition === "all" || item.competition === competition)
      && (finiteNumber(item.confidence) ?? 0) >= confidence;
  });
  return filtered.sort((left, right) => {
    if (sort === "confidence") return (finiteNumber(right.confidence) ?? -1) - (finiteNumber(left.confidence) ?? -1);
    if (sort === "uncertainty") return (finiteNumber(left.uncertainty) ?? 2) - (finiteNumber(right.uncertainty) ?? 2);
    return new Date(left.utc_date || 0) - new Date(right.utc_date || 0);
  });
}

function probabilityMarkup(probabilities = {}) {
  const rows = [
    ["Home", probabilities.home_win ?? probabilities.Win],
    ["Draw", probabilities.draw ?? probabilities.Draw],
    ["Away", probabilities.away_win ?? probabilities.Loss],
  ];
  if (rows.every(([, value]) => finiteNumber(value) === null)) return "";
  return `<div class="probability-table">${rows.map(([label, value]) => {
    const numeric = Math.max(0, Math.min(1, finiteNumber(value) ?? 0));
    return `
      <div class="probability-row">
        <span>${label}</span>
        <i><b style="--probability:${numeric * 100}%"></b></i>
        <strong>${percent(numeric)}</strong>
      </div>`;
  }).join("")}</div>`;
}

function evidenceMarkup(item) {
  const margin = finiteNumber(item.probability_margin);
  const explanation = item.explanation || {};
  const drivers = explanation.available === true && Array.isArray(explanation.top_drivers)
    ? explanation.top_drivers
    : [];
  const driverRows = drivers.map((driver) => {
    const contribution = finiteNumber(driver.contribution);
    const signed = contribution === null
      ? "—"
      : `${contribution >= 0 ? "+" : ""}${contribution.toFixed(2)}`;
    const label = String(driver.feature || "unknown signal").replaceAll("_", " ");
    return `<li><span>${escapeHtml(label)}</span><b class="${contribution !== null && contribution < 0 ? "driver-negative" : ""}">${escapeHtml(signed)} logit</b></li>`;
  }).join("");
  const predictedAt = item.predicted_at ? formatDateTime(item.predicted_at) : "Unavailable";
  const forecastId = String(item.forecast_id || "unavailable").slice(0, 8);
  const modelSha256 = String(item.model?.model_sha256 || "unavailable").slice(0, 8);
  const estimator = [item.model?.estimator, item.model?.estimator_version]
    .filter((value) => typeof value === "string" && value.trim())
    .map((value) => value.trim())
    .join(" ");
  return `
    <details class="prediction-evidence">
      <summary>Why this call</summary>
      <div class="evidence-facts">
        <span>Probability lead <b>${percent(margin)}</b></span>
        <span>Meaning <b>top outcome probability</b></span>
        <span>Forecast <b>${escapeHtml(forecastId)}</b></span>
        <span>Model <b>${escapeHtml(modelSha256)}</b></span>
        ${estimator ? `<span>Estimator <b>${escapeHtml(estimator)}</b></span>` : ""}
        <span>Predicted <b>${escapeHtml(predictedAt)}</b></span>
      </div>
      ${driverRows
        ? `<p>${escapeHtml(explanation.predicted_vs_runner_up || "Winner vs runner-up")}</p><ol>${driverRows}</ol>`
        : '<p>Feature-level explanation is unavailable for this estimator.</p>'}
    </details>
  `;
}

function renderPredictions() {
  const predictions = filteredPredictions();
  const container = document.getElementById("predictions");
  if (!state.predictionsAvailable) {
    document.getElementById("fixture-result-count").textContent = "Prediction artifact unavailable";
    setEmpty(container, "Predictions unavailable", "Restore predictions.json before inspecting or publishing fixtures.");
    return;
  }
  document.getElementById("fixture-result-count").textContent = `${predictions.length} of ${state.predictions.length} fixtures shown`;
  if (!predictions.length) {
    setEmpty(container, "No fixtures match", "Change a filter, or run the pipeline when new schedules are available.");
    return;
  }
  container.innerHTML = predictions.map((item) => {
    const confidence = Math.max(0, Math.min(1, finiteNumber(item.confidence) ?? 0));
    const coldStart = item.data_quality?.is_cold_start === true;
    const validation = finiteNumber(item.model?.validation_accuracy);
    const baseline = finiteNumber(item.model?.baseline_accuracy);
    const lift = validation !== null && baseline !== null ? validation - baseline : null;
    return `
      <article class="fixture-card">
        <div class="fixture-meta">
          <span>${escapeHtml(item.competition || "Unlabelled")}</span>
          <time datetime="${escapeHtml(item.utc_date || "")}">${escapeHtml(formatKickoff(item.utc_date))}</time>
        </div>
        <div class="matchup">
          <h3>${escapeHtml(item.home_team || "Home TBD")} <span>vs</span> ${escapeHtml(item.away_team || "Away TBD")}</h3>
          <div class="confidence-disc" style="--confidence:${confidence * 100}%">
            <strong>${percent(confidence)}</strong><span>confidence</span>
          </div>
        </div>
        <div class="model-call">
          <span>Model call</span><strong>${escapeHtml(outcomeLabel(item.prediction))}</strong>
        </div>
        ${probabilityMarkup(item.probabilities)}
        ${evidenceMarkup(item)}
        <div class="quality-flags">
          <span class="${coldStart ? "flag-warning" : ""}">${coldStart ? "Cold start" : "History ready"}</span>
          <span>Uncertainty ${percent(item.uncertainty)}</span>
          <span>Validation lift ${lift === null ? "—" : `${lift >= 0 ? "+" : ""}${percent(lift, 1)}`}</span>
        </div>
      </article>
    `;
  }).join("");
}

function renderRecommended(preset, release) {
  const container = document.getElementById("recommended-bets");
  if (!recommendationsApproved(release) || preset?.can_publish_recommendations !== true) {
    setEmpty(container, "Recommendations on hold", "The release gate is blocked or unavailable; raw probabilities remain visible for diagnosis only.");
    return;
  }
  if (!preset) {
    setEmpty(container, "Shortlist unavailable", "The preset questions artifact could not be loaded.");
    return;
  }
  const question = (preset?.questions || []).find((item) => item.id === "recommended_bets_today");
  const items = (question?.items || []).filter((item) => finiteNumber(item.confidence) !== null);
  if (!items.length) {
    setEmpty(container, "No shortlist yet", "Only verified picks above the recommendation threshold appear here.");
    return;
  }
  container.innerHTML = items.map((item) => `
    <article class="compact-row">
      <div><strong>${escapeHtml(item.match)}</strong><span>${escapeHtml(item.competition || "Competition pending")}</span></div>
      <div class="compact-result"><b>${escapeHtml(outcomeLabel(item.prediction))}</b><span>${percent(item.confidence, 1)}</span></div>
    </article>
  `).join("");
}

function renderScores(scorePayload) {
  const container = document.getElementById("scores");
  if (!scorePayload) {
    setEmpty(container, "Score data unavailable", "The scores artifact could not be loaded.");
    return;
  }
  const scores = scorePayload?.scores || [];
  if (!scores.length) {
    setEmpty(container, "No score wire", "Latest results will appear after the score source responds.");
    return;
  }
  container.innerHTML = scores.slice(0, 12).map((item) => `
    <article class="compact-row score-row">
      <div><strong>${escapeHtml(item.home_team || "Home")}</strong><span>${escapeHtml(item.away_team || "Away")}</span></div>
      <div class="scoreline"><b>${escapeHtml(item.home_score ?? "–")}</b><i>:</i><b>${escapeHtml(item.away_score ?? "–")}</b></div>
    </article>
  `).join("");
}

function renderErrors(predictionPayload, scorePayload, artifactFailures) {
  const container = document.getElementById("prediction-errors");
  const errors = predictionPayload?.errors || [];
  const refreshErrors = predictionPayload?.refresh_errors || [];
  const scoreErrors = scorePayload?.errors || [];
  const signals = [
    ...artifactFailures,
    ...errors.map((item) => ({
      level: "critical",
      code: `${item.competition || "UNKNOWN"} / ${item.code || "PREDICTION_FAILED"}`,
      message: item.message,
    })),
    ...refreshErrors.map((item) => ({
      level: "warning",
      code: `${item.competition || "UNKNOWN"} / REFRESH / ${item.code || "TRAINING_FAILED"}`,
      message: item.message,
    })),
    ...scoreErrors.map((item) => ({
      level: "warning",
      code: item.code || "SCORE_FETCH_FAILED",
      message: item.message,
    })),
  ];
  if (!signals.length) {
    setEmpty(container, "No partial failures", "Every requested competition completed without a structured prediction or model-refresh error.");
    return;
  }
  container.innerHTML = signals.map((item) => signalMarkup(item, "critical")).join("");
}

function settledValue(results, index) {
  return results[index].status === "fulfilled" ? results[index].value.payload : null;
}

function settledRaw(results, index) {
  return results[index].status === "fulfilled" ? results[index].value.raw : null;
}

function artifactFailures(paths, results) {
  return results.flatMap((result, index) => {
    if (result.status === "fulfilled") return [];
    const name = paths[index].toUpperCase();
    return [{
      level: "critical",
      code: `${name}_ARTIFACT_UNAVAILABLE`,
      message: `${paths[index]}.json could not be loaded: ${result.reason?.message || "unknown error"}`,
    }];
  });
}

async function sha256(raw) {
  if (!globalThis.crypto?.subtle || typeof TextEncoder === "undefined") return null;
  const bytes = new TextEncoder().encode(raw);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function snapshotFailures(paths, values, rawValues) {
  const manifest = values[0];
  if (!manifest) return [];
  const expected = manifest.generated_at;
  if (!expected) {
    return [{
      level: "critical",
      code: "MANIFEST_SNAPSHOT_MISMATCH",
      message: "manifest.json has no generated_at value; the snapshot cannot be verified.",
    }];
  }
  const manifestKeys = {
    operations: "operations",
    release: "release_governance",
    predictions: "predictions",
    preset_questions: "preset_questions",
    scores: "scores",
  };
  const failures = [];
  for (let index = 1; index < values.length; index += 1) {
    const value = values[index];
    if (value === null) continue;
    const name = paths[index].toUpperCase();
    if (value.generated_at !== expected) {
      failures.push({
        level: "critical",
        code: `${name}_SNAPSHOT_MISMATCH`,
        message: `${paths[index]}.json belongs to ${value.generated_at || "an unknown generation"}, expected ${expected}.`,
      });
      continue;
    }
    const metadata = manifest.artifacts?.[manifestKeys[paths[index]]];
    const raw = rawValues[index];
    const bytes = typeof raw === "string" && typeof TextEncoder !== "undefined"
      ? new TextEncoder().encode(raw)
      : null;
    const digest = typeof raw === "string" ? await sha256(raw) : null;
    if (
      !metadata
      || bytes === null
      || digest === null
      || metadata.size_bytes !== bytes.byteLength
      || metadata.sha256 !== digest
    ) {
      failures.push({
        level: "critical",
        code: `${name}_SNAPSHOT_MISMATCH`,
        message: `${paths[index]}.json does not match the manifest SHA-256 and size contract.`,
      });
    }
  }
  return failures;
}

async function init() {
  const paths = ["manifest", "operations", "release", "predictions", "preset_questions", "scores"];
  const results = await Promise.allSettled(paths.map((name) => loadJson(`./data/${name}.json`)));
  const values = paths.map((_, index) => settledValue(results, index));
  const rawValues = paths.map((_, index) => settledRaw(results, index));
  const manifest = values[0];
  const unavailable = artifactFailures(paths, results);
  const inconsistent = await snapshotFailures(paths, values, rawValues);
  const failures = [...unavailable, ...inconsistent];
  const snapshotUsable = Boolean(manifest) && failures.length === 0;
  const operations = snapshotUsable ? values[1] : null;
  const release = snapshotUsable ? values[2] : null;
  const predictionPayload = snapshotUsable ? values[3] : null;
  const preset = snapshotUsable ? values[4] : null;
  const scores = snapshotUsable ? values[5] : null;
  state.predictionsAvailable = Boolean(predictionPayload);
  state.predictions = predictionPayload?.predictions || [];

  document.getElementById("data-health").textContent = manifest
    ? snapshotUsable
      ? `Snapshot ${formatDateTime(manifest.generated_at)} · ${manifest.prediction_count ?? 0} fixtures`
      : `Snapshot ${formatDateTime(manifest.generated_at)} · integrity check failed`
    : "Manifest unavailable · cached panels may be out of sync";
  document.getElementById("footer-run-id").textContent = manifest?.run_id
    ? `Run ${manifest.run_id}`
    : "Legacy or incomplete snapshot";

  renderCoverage(manifest, operations);
  renderMetrics(state.predictions, operations, state.predictionsAvailable, release);
  renderOperations(operations);
  renderRelease(release);
  populateCompetitionFilter(state.predictions);
  renderPredictions();
  renderRecommended(preset, release);
  renderScores(scores);
  renderErrors(predictionPayload, scores, failures);
  document.body.classList.add("is-ready");
}

["prediction-search", "competition-filter", "confidence-filter", "prediction-sort"].forEach((id) => {
  document.getElementById(id).addEventListener(id === "prediction-search" ? "input" : "change", renderPredictions);
});

init().catch((error) => {
  document.getElementById("data-health").textContent = `Dashboard failed to initialize: ${error.message}`;
  document.body.classList.add("is-ready");
});
