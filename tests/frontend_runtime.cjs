"use strict";

const { createHash } = require("node:crypto");
const path = require("node:path");
const scenario = process.argv[2];
const newScenarios = ["approved-shortlist", "expired-shortlist", "inconsistent-approval", "malformed-predictions", "future-snapshot", "expiry-timer", "missing-expiry"];
const html = require("node:fs").readFileSync(path.resolve(__dirname, "../docs/predictions.html"), "utf8");
const htmlIds = new Set(Array.from(html.matchAll(/\bid="([^"]+)"/g), match => match[1]));
const elements = new Map();
let now = Date.parse("2026-08-10T09:00:00Z");
Date.now = () => now;
let expireCallback;
const nativeTimeout = global.setTimeout;
global.setTimeout = (callback, delay) => {
  if (delay > 1000) { expireCallback = callback; return 0; }
  return nativeTimeout(callback, delay);
};

function element(id = "") {
  return {
    id, innerHTML: "", textContent: "", className: "",
    value: id === "competition-filter"
      ? "all"
      : id === "prediction-sort"
        ? "kickoff"
        : id === "prediction-search" ? "" : "0",
    style: {}, children: [], classList: { add() {} },
    setAttribute() {}, addEventListener() {},
    appendChild(child) { this.children.push(child); },
  };
}

global.document = {
  body: element("body"),
  getElementById(id) {
    if (!htmlIds.has(id)) throw new Error(`Dashboard references missing HTML element: ${id}`);
    if (!elements.has(id)) elements.set(id, element(id));
    return elements.get(id);
  },
  createElement() { return element(); },
};

const common = {
  "manifest.json": { generated_at: "2026-08-10T08:30:00Z", prediction_count: 1, run_id: "test" },
  "release.json": {
    release_decision: "blocked", can_publish_recommendations: false,
    blockers: [], warnings: [],
    audit: { prediction_count: 1, high_confidence_count: 1, average_confidence: 0.72 },
  },
  "predictions.json": {
    predictions: [{
      competition: "PL", utc_date: "2026-08-11T18:00:00Z",
      home_team: '<img src=x onerror="globalThis.pwned=true">', away_team: "Safe FC",
      prediction: "HOME_WIN", confidence: 0.72, uncertainty: 0.4,
      probabilities: { home_win: 0.72, draw: 0.18, away_win: 0.1 },
      probability_margin: 0.54,
      confidence_semantics: "top_outcome_probability",
      explanation: {
        available: true, method: "tree_shap_margin",
        predicted_vs_runner_up: "HOME_WIN vs DRAW",
        top_drivers: [{ feature: "elo_diff", contribution: 0.31, direction: "supports_prediction" }],
      },
      data_quality: { is_cold_start: false },
      model: {
        estimator: "xgboost", estimator_version: "3.4.1",
        validation_accuracy: 0.4, baseline_accuracy: 0.5,
      },
    }],
    errors: [],
  },
  "preset_questions.json": { questions: [] },
  "scores.json": { scores: [], errors: [] },
};

if (scenario === "missing") {
  common["manifest.json"].prediction_count = 0;
  common["predictions.json"] = { predictions: [], errors: [] };
} else if (scenario === "quality" || scenario === "quality-mismatch") {
  common["operations.json"] = {
    status: "critical", prediction_count: 1, average_uncertainty: 0.4,
    cold_start_ratio: 0, high_confidence_count: 1,
    model_coverage: { loaded: 1, requested: 1, ratio: 1 },
    models_below_baseline: ["PL"], models_without_quality_metrics: [],
    model_quality_by_competition: scenario === "quality-mismatch" ? [] : [{
      competition: "PL", validation_accuracy: 0.4,
      baseline_accuracy: 0.5, accuracy_lift: -0.1,
    }],
    coverage_by_competition: [{ competition: "PL", prediction_count: 1 }],
    alerts: [],
  };
} else if (scenario === "score-error") {
  common["operations.json"] = {
    status: "attention", prediction_count: 1, average_uncertainty: 0.4,
    cold_start_ratio: 0, high_confidence_count: 1,
    model_coverage: { loaded: 1, requested: 1, ratio: 1 },
    models_below_baseline: [], models_without_quality_metrics: [],
    model_quality_by_competition: [], coverage_by_competition: [],
    alerts: [{ level: "warning", code: "SCORE_FETCH_ERRORS", message: "1 score source error" }],
  };
  common["scores.json"].errors = [{
    code: "SCORE_SOURCE_UNAVAILABLE",
    message: "Score source timed out",
  }];
} else if (scenario === "backtest") {
  common["operations.json"] = {
    status: "healthy", prediction_count: 1, average_uncertainty: 0.4,
    cold_start_ratio: 0, high_confidence_count: 1,
    model_coverage: { loaded: 1, requested: 1, ratio: 1 },
    models_below_baseline: [], models_without_quality_metrics: [],
    models_below_backtest_baseline: [], models_with_poor_calibration: [],
    models_recovered_from_previous: [],
    model_quality_by_competition: [{
      competition: "PL", validation_accuracy: 0.61, baseline_accuracy: 0.48,
      estimator: "xgboost", estimator_version: "3.4.1",
      explanation_method: "tree_shap_margin", boosting_rounds: 84,
      accuracy_lift: 0.13, backtest_folds: 3, backtest_accuracy: 0.58,
      backtest_baseline_accuracy: 0.47, backtest_accuracy_lift: 0.11,
      backtest_majority_baseline_accuracy: 0.44,
      backtest_elo_baseline_accuracy: 0.47,
      backtest_log_loss: 0.94, backtest_brier_score: 0.59,
      backtest_calibration_error: 0.09,
      backtest_ranked_probability_score: 0.22,
      backtest_worst_fold_accuracy_lift: 0.02,
      backtest_accuracy_lift_stddev: 0.03,
      backtest_sharpness: 0.61,
      backtest_reliability: [
        { class_index: 0, calibration_error: 0.07, sample_count: 60 },
        { class_index: 1, calibration_error: 0.1, sample_count: 60 },
        { class_index: 2, calibration_error: 0.07, sample_count: 60 },
      ],
    }],
    coverage_by_competition: [{ competition: "PL", prediction_count: 1 }],
    alerts: [],
  };
} else if (scenario === "refresh-error") {
  common["operations.json"] = {
    status: "attention", prediction_count: 1, average_uncertainty: 0.4,
    cold_start_ratio: 0, high_confidence_count: 1,
    model_coverage: { loaded: 1, requested: 1, ratio: 1 },
    models_below_baseline: [], models_without_quality_metrics: [],
    models_below_backtest_baseline: [], models_with_poor_calibration: [],
    models_recovered_from_previous: [], model_quality_by_competition: [],
    coverage_by_competition: [],
    alerts: [{ level: "warning", code: "MODEL_REFRESH_ERRORS", message: "1 refresh error" }],
  };
  common["predictions.json"].refresh_errors = [{
    competition: "PL", code: "TrainingError", message: "backtest below baseline",
  }];
} else if (scenario === "stale") {
  common["operations.json"] = {
    status: "critical", prediction_count: 1, average_uncertainty: 0.4,
    cold_start_ratio: 0, high_confidence_count: 1,
    model_coverage: { loaded: 1, requested: 1, ratio: 1 },
    models_below_baseline: [], models_without_quality_metrics: [],
    models_below_backtest_baseline: [], models_with_poor_calibration: [],
    models_recovered_from_previous: [], models_stale: ["PL"],
    model_quality_by_competition: [{
      competition: "PL", validation_accuracy: 0.61, baseline_accuracy: 0.48,
      backtest_folds: 3, backtest_accuracy: 0.58,
      backtest_baseline_accuracy: 0.47, backtest_calibration_error: 0.09,
      model_age_hours: 216,
    }],
    coverage_by_competition: [{ competition: "PL", prediction_count: 1 }],
    alerts: [{ level: "critical", code: "MODEL_STALE", message: "PL is stale" }],
  };
} else if (
  scenario === "blocked-shortlist"
  || scenario === "mixed-snapshot"
  || scenario === "tampered"
) {
  common["operations.json"] = {
    status: "healthy", prediction_count: 1, average_uncertainty: 0.4,
    cold_start_ratio: 0, high_confidence_count: 1,
    model_coverage: { loaded: 1, requested: 1, ratio: 1 },
    models_below_baseline: [], models_without_quality_metrics: [],
    model_quality_by_competition: [], coverage_by_competition: [], alerts: [],
  };
  common["preset_questions.json"] = {
    can_publish_recommendations: true,
    questions: [{
      id: "recommended_bets_today",
      items: [{
        match: "North FC vs South FC", competition: "PL",
        prediction: "HOME_WIN", confidence: 0.72,
      }],
    }],
  };
} else if (!newScenarios.includes(scenario)) {
  throw new Error(`unknown scenario: ${scenario}`);
}

if (newScenarios.includes(scenario)) {
  common["operations.json"] = {
    status: "healthy", prediction_count: 1, alerts: [],
    model_coverage: { loaded: 1, requested: 1 },
  };
  Object.assign(common["release.json"], {
    release_decision: "approved", can_publish_recommendations: true,
    expires_at: "2026-08-10T10:00:00Z", thresholds: { high_confidence: 0.6 },
  });
  common["preset_questions.json"] = {
    can_publish_recommendations: true,
    questions: [{ id: "recommended_bets_today", items: [
      { match: "Approved FC vs Safe FC", confidence: 0.72, prediction: "HOME_WIN" },
      { match: "Below threshold FC", confidence: 0.56, prediction: "DRAW" },
    ] }],
  };
  if (scenario === "expired-shortlist") common["release.json"].expires_at = "2026-08-10T08:45:00Z";
  if (scenario === "inconsistent-approval") common["release.json"].can_publish_recommendations = false;
  if (scenario === "missing-expiry") delete common["release.json"].expires_at;
  if (scenario === "malformed-predictions") common["predictions.json"].predictions = { invalid: true };
  if (scenario === "future-snapshot") common["manifest.json"].generated_at = "2026-08-11T08:30:00Z";
}

const snapshotTime = common["manifest.json"].generated_at;
Object.values(common).forEach((payload) => {
  if (payload && typeof payload === "object" && !payload.generated_at) {
    payload.generated_at = snapshotTime;
  }
});
if (scenario === "mixed-snapshot") {
  common["operations.json"].generated_at = "2026-08-09T08:30:00Z";
}
const manifestArtifacts = {
  operations: "operations.json",
  release_governance: "release.json",
  predictions: "predictions.json",
  preset_questions: "preset_questions.json",
  scores: "scores.json",
};
common["manifest.json"].artifacts = Object.fromEntries(
  Object.entries(manifestArtifacts).map(([key, file]) => {
    const raw = JSON.stringify(common[file] || {});
    return [key, {
      sha256: createHash("sha256").update(raw).digest("hex"),
      size_bytes: Buffer.byteLength(raw),
    }];
  }),
);
if (scenario === "tampered") {
  common["predictions.json"].predictions[0].home_team = "Tampered FC";
}

global.fetch = async (url) => {
  const name = String(url).split("/").pop();
  if (scenario === "missing" && name === "operations.json") {
    return { ok: false, status: 404, async text() { return "{}"; } };
  }
  return {
    ok: true,
    status: 200,
    async text() { return JSON.stringify(common[name]); },
  };
};

require(path.resolve(process.cwd(), "docs/app.js"));

setTimeout(() => {
  if (scenario === "missing") {
    const operations = elements.get("operations-alerts").innerHTML;
    const diagnostics = elements.get("prediction-errors").innerHTML;
    if (!operations.includes("运行摘要文件未就绪")) throw new Error(`missing operations was not diagnosed: ${operations}`);
    if (operations.includes("检查已通过")) throw new Error("missing operations was reported as passed");
    if (!diagnostics.includes("OPERATIONS_ARTIFACT_UNAVAILABLE")) throw new Error(`artifact failure missing from diagnostics: ${diagnostics}`);
    if (elements.get("metric-cold-start").textContent !== "—") throw new Error("zero-fixture cold-start value must be unknown");
  } else if (scenario === "quality") {
    const operations = elements.get("operations-alerts").innerHTML;
    const predictions = elements.get("predictions").innerHTML;
    if (!operations.includes("MODEL_BELOW_BASELINE / PL") || !operations.includes("40%") || !operations.includes("50%")) throw new Error(`model quality detail missing: ${operations}`);
    if (!predictions.includes("&lt;img") || predictions.includes("<img src=x")) throw new Error(`dynamic team name was not escaped: ${predictions}`);
    if (!predictions.includes("领先第二赛果") || !predictions.includes("54%") || !predictions.includes("xgboost 3.4.1") || !predictions.includes("elo diff") || !predictions.includes("+0.31")) {
      throw new Error(`prediction evidence missing: ${predictions}`);
    }
  } else if (scenario === "quality-mismatch") {
    const operations = elements.get("operations-alerts").innerHTML;
    if (!operations.includes("MODEL_BELOW_BASELINE / PL") || !operations.includes("质量详情未就绪")) {
      throw new Error(`missing quality fallback did not fail closed: ${operations}`);
    }
  } else if (scenario === "score-error") {
    const diagnostics = elements.get("prediction-errors").innerHTML;
    if (!diagnostics.includes("SCORE_SOURCE_UNAVAILABLE") || !diagnostics.includes("Score source timed out")) {
      throw new Error(`score source detail missing from diagnostics: ${diagnostics}`);
    }
  } else if (scenario === "backtest") {
    const summary = elements.get("operations-summary").innerHTML;
    const evidence = elements.get("operations-alerts").innerHTML;
    if (!summary.includes("已有回测") || !summary.includes("1 / 1")) {
      throw new Error(`backtest coverage missing from summary: ${summary}`);
    }
    if (!evidence.includes("BACKTEST / PL") || !evidence.includes("xgboost 3.4.1") || !evidence.includes("tree shap margin") || !evidence.includes("84 轮") || !evidence.includes("58%") || !evidence.includes("47%") || !evidence.includes("Elo 47% / 多数类 44%") || !evidence.includes("RPS 0.220") || !evidence.includes("最差窗口提升 2.0%") || !evidence.includes("9%")) {
      throw new Error(`backtest evidence missing from quality panel: ${evidence}`);
    }
    if (!evidence.includes("客胜 ECE 7%") || !evidence.includes("平局 ECE 10%") || !evidence.includes("主胜 ECE 7%") || !evidence.includes("60 个样本") || !evidence.includes("平均最高概率 61%") || !evidence.includes("窗口标准差 3.0%")) {
      throw new Error(`classwise reliability evidence missing: ${evidence}`);
    }
  } else if (scenario === "refresh-error") {
    const diagnostics = elements.get("prediction-errors").innerHTML;
    if (!diagnostics.includes("PL / REFRESH / TrainingError") || !diagnostics.includes("backtest below baseline")) {
      throw new Error(`refresh failure detail missing from diagnostics: ${diagnostics}`);
    }
  } else if (scenario === "stale") {
    const summary = elements.get("operations-summary").innerHTML;
    const evidence = elements.get("operations-alerts").innerHTML;
    if (!summary.includes("过期模型") || !summary.includes(">1<")) {
      throw new Error(`stale model count missing from summary: ${summary}`);
    }
    if (!evidence.includes("BACKTEST / PL") || !evidence.includes("216h")) {
      throw new Error(`model age missing from quality evidence: ${evidence}`);
    }
  } else if (scenario === "blocked-shortlist") {
    const shortlist = elements.get("recommended-bets").innerHTML;
    if (!shortlist.includes("暂停显示推荐") || shortlist.includes("North FC vs South FC")) {
      throw new Error(`blocked release exposed actionable shortlist: ${shortlist}`);
    }
    if (elements.get("metric-picks").textContent !== "0") {
      throw new Error("blocked release exposed publishable count");
    }
  } else if (scenario === "mixed-snapshot") {
    const diagnostics = elements.get("prediction-errors").innerHTML;
    const operations = elements.get("operations-alerts").innerHTML;
    const releaseTitle = elements.get("release-command-title").textContent;
    if (!diagnostics.includes("OPERATIONS_SNAPSHOT_MISMATCH")) {
      throw new Error(`mixed snapshot was not diagnosed: ${diagnostics}`);
    }
    if (!operations.includes("运行摘要文件未就绪") || releaseTitle !== "发布状态未就绪") {
      throw new Error("mixed snapshot did not fail closed across dependent panels");
    }
  } else if (newScenarios.includes(scenario)) {
    if (scenario === "expiry-timer") {
      if (typeof expireCallback !== "function") throw new Error("release expiry was not scheduled");
      now = Date.parse("2026-08-10T10:00:01Z");
      expireCallback();
    }
    const shortlist = elements.get("recommended-bets").innerHTML;
    const status = elements.get("release-status").textContent;
    if (scenario === "approved-shortlist") {
      if (status !== "允许发布" || !shortlist.includes("Approved FC") || shortlist.includes("Below threshold FC")) throw new Error("approved shortlist or threshold is incorrect");
    } else {
      if (status === "允许发布" || shortlist.includes("Approved FC")) throw new Error(`${scenario} exposed recommendations`);
      if (["expired-shortlist", "expiry-timer"].includes(scenario) && !elements.get("release-blockers").innerHTML.includes("RELEASE_EXPIRED")) throw new Error("expiry reason missing");
      if (scenario === "malformed-predictions" && !elements.get("prediction-errors").innerHTML.includes("PREDICTIONS_ARTIFACT_UNAVAILABLE")) throw new Error("malformed data diagnosis missing");
      if (scenario === "future-snapshot" && !elements.get("prediction-errors").innerHTML.includes("MANIFEST_SNAPSHOT_MISMATCH")) throw new Error("future snapshot diagnosis missing");
    }
  } else {
    const diagnostics = elements.get("prediction-errors").innerHTML;
    const predictions = elements.get("predictions").innerHTML;
    if (!diagnostics.includes("PREDICTIONS_SNAPSHOT_MISMATCH")) {
      throw new Error(`tampered snapshot was not diagnosed: ${diagnostics}`);
    }
    if (!predictions.includes("预测未就绪") || predictions.includes("Tampered FC")) {
      throw new Error("tampered prediction payload was rendered");
    }
  }
  process.stdout.write(`${scenario}: ok\n`);
}, 50);
