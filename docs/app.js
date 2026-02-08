async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }
  return response.json();
}

function renderList(container, items, formatter) {
  container.innerHTML = "";
  if (!items || items.length === 0) {
    container.innerHTML = "<p>No data available yet.</p>";
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
  if (value === null || value === undefined) {
    return "N/A";
  }
  return `${(value * 100).toFixed(1)}%`;
}

async function init() {
  const recommendedContainer = document.getElementById("recommended-bets");
  const predictionsContainer = document.getElementById("predictions");
  const scoresContainer = document.getElementById("scores");

  try {
    const preset = await loadJson("./data/preset_questions.json");
    const recommended = preset.questions.find(
      (question) => question.id === "recommended_bets_today"
    );
    renderList(recommendedContainer, recommended.items, (item) => `
      <div class="item-title">${item.match}</div>
      <div class="item-meta">${item.competition || ""} • ${item.prediction || ""} • ${formatConfidence(item.confidence)}</div>
    `);
  } catch (error) {
    recommendedContainer.innerHTML = "<p>Recommended picks are unavailable.</p>";
  }

  try {
    const predictionsData = await loadJson("./data/predictions.json");
    renderList(predictionsContainer, predictionsData.predictions, (item) => `
      <div class="item-title">${item.home_team} vs ${item.away_team}</div>
      <div class="item-meta">${item.competition} • ${item.prediction} • ${formatConfidence(item.confidence)}</div>
    `);
  } catch (error) {
    predictionsContainer.innerHTML = "<p>Predictions are unavailable.</p>";
  }

  try {
    const scoresData = await loadJson("./data/scores.json");
    renderList(scoresContainer, scoresData.scores, (item) => `
      <div class="item-title">${item.home_team} ${item.home_score ?? ""} - ${item.away_score ?? ""} ${item.away_team}</div>
      <div class="item-meta">${item.league || ""} • ${item.date || ""} ${item.time || ""}</div>
    `);
  } catch (error) {
    scoresContainer.innerHTML = "<p>Scores are unavailable.</p>";
  }
}

init();