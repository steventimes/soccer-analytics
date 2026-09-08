from __future__ import annotations

FRONTEND_RUNTIME_HARNESS = r'''"use strict";

const { createHash } = require("node:crypto");
const path = require("node:path");
const scenario = process.argv[1];
const newScenarios = ["approved-shortlist", "expired-shortlist", "inconsistent-approval", "malformed-predictions", "future-snapshot", "expiry-timer", "missing-expiry"];
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
    if (!operations.includes("Operations artifact unavailable")) throw new Error(`missing operations was not diagnosed: ${operations}`);
    if (operations.includes("checks passed")) throw new Error("missing operations was reported as passed");
    if (!diagnostics.includes("OPERATIONS_ARTIFACT_UNAVAILABLE")) throw new Error(`artifact failure missing from diagnostics: ${diagnostics}`);
    if (elements.get("metric-cold-start").textContent !== "—") throw new Error("zero-fixture cold-start value must be unknown");
  } else if (scenario === "quality") {
    const operations = elements.get("operations-alerts").innerHTML;
    const predictions = elements.get("predictions").innerHTML;
    if (!operations.includes("MODEL_BELOW_BASELINE / PL") || !operations.includes("40%") || !operations.includes("50%")) throw new Error(`model quality detail missing: ${operations}`);
    if (!predictions.includes("&lt;img") || predictions.includes("<img src=x")) throw new Error(`dynamic team name was not escaped: ${predictions}`);
    if (!predictions.includes("Probability lead") || !predictions.includes("54%") || !predictions.includes("xgboost 3.4.1") || !predictions.includes("elo diff") || !predictions.includes("+0.31")) {
      throw new Error(`prediction evidence missing: ${predictions}`);
    }
  } else if (scenario === "quality-mismatch") {
    const operations = elements.get("operations-alerts").innerHTML;
    if (!operations.includes("MODEL_BELOW_BASELINE / PL") || !operations.includes("Quality detail unavailable")) {
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
    if (!summary.includes("Backtests ready") || !summary.includes("1 / 1")) {
      throw new Error(`backtest coverage missing from summary: ${summary}`);
    }
    if (!evidence.includes("BACKTEST / PL") || !evidence.includes("xgboost 3.4.1") || !evidence.includes("tree shap margin") || !evidence.includes("84 rounds") || !evidence.includes("58%") || !evidence.includes("47%") || !evidence.includes("Elo 47% / majority 44%") || !evidence.includes("RPS 0.220") || !evidence.includes("worst-fold lift 2.0%") || !evidence.includes("9%")) {
      throw new Error(`backtest evidence missing from quality panel: ${evidence}`);
    }
    if (!evidence.includes("Away ECE 7%") || !evidence.includes("Draw ECE 10%") || !evidence.includes("Home ECE 7%") || !evidence.includes("60 samples") || !evidence.includes("sharpness 61%") || !evidence.includes("fold σ 3.0%")) {
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
    if (!summary.includes("Stale models") || !summary.includes(">1<")) {
      throw new Error(`stale model count missing from summary: ${summary}`);
    }
    if (!evidence.includes("BACKTEST / PL") || !evidence.includes("216h")) {
      throw new Error(`model age missing from quality evidence: ${evidence}`);
    }
  } else if (scenario === "blocked-shortlist") {
    const shortlist = elements.get("recommended-bets").innerHTML;
    if (!shortlist.includes("Recommendations on hold") || shortlist.includes("North FC vs South FC")) {
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
    if (!operations.includes("Operations artifact unavailable") || releaseTitle !== "Release status unavailable") {
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
      if (status !== "Publish" || !shortlist.includes("Approved FC") || shortlist.includes("Below threshold FC")) throw new Error("approved shortlist or threshold is incorrect");
    } else {
      if (status === "Publish" || shortlist.includes("Approved FC")) throw new Error(`${scenario} exposed recommendations`);
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
    if (!predictions.includes("Predictions unavailable") || predictions.includes("Tampered FC")) {
      throw new Error("tampered prediction payload was rendered");
    }
  }
  process.stdout.write(`${scenario}: ok\n`);
}, 50);'''


import unittest
from datetime import UTC, datetime
from unittest.mock import patch

import requests

from app.web import scores as scores_module


class _Response:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _InvalidJsonResponse(_Response):
    def json(self) -> object:
        raise requests.exceptions.JSONDecodeError("invalid JSON", "<html>", 0)


class TestDailyScores(unittest.TestCase):
    def test_source_failure_contracts(self) -> None:
        for name in (
            "_check_requests_json_decode_error_is_not_misclassified_as_network_failure",
            "_check_events_field_must_be_a_list",
            "_check_non_object_payload_becomes_invalid_payload_diagnostic",
            "_check_transport_failure_returns_structured_diagnostic",
        ):
            with self.subTest(contract=name):
                getattr(self, name)()

    def test_event_filtering_contract(self) -> None:
        self._check_malformed_event_is_reported_while_valid_event_is_preserved()

    def _check_requests_json_decode_error_is_not_misclassified_as_network_failure(self) -> None:
        with patch(
            "app.web.scores.requests.get",
            return_value=_InvalidJsonResponse(None),
        ):
            result = scores_module.fetch_daily_scores_with_diagnostics(
                datetime(2026, 8, 10, tzinfo=UTC)
            )

        self.assertEqual(result.scores, ())
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["code"], "SCORE_SOURCE_INVALID_PAYLOAD")

    def _check_malformed_event_is_reported_while_valid_event_is_preserved(self) -> None:
        payload = {
            "events": [
                {
                    "idEvent": "901",
                    "strLeague": "Premier League",
                    "strHomeTeam": "North FC",
                    "strAwayTeam": "South FC",
                    "intHomeScore": "2",
                    "intAwayScore": "1",
                    "strTime": "19:30:00",
                    "dateEvent": "2026-08-10",
                },
                {"idEvent": "broken", "strLeague": "Premier League"},
            ]
        }
        with patch("app.web.scores.requests.get", return_value=_Response(payload)):
            result = scores_module.fetch_daily_scores_with_diagnostics(
                datetime(2026, 8, 10, tzinfo=UTC)
            )

        self.assertEqual(len(result.scores), 1)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["code"], "SCORE_EVENT_INVALID")
        self.assertIn("broken", result.errors[0]["message"])

    def _check_events_field_must_be_a_list(self) -> None:
        with patch(
            "app.web.scores.requests.get",
            return_value=_Response({"events": {"unexpected": "mapping"}}),
        ):
            result = scores_module.fetch_daily_scores_with_diagnostics(
                datetime(2026, 8, 10, tzinfo=UTC)
            )

        self.assertEqual(result.scores, ())
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["code"], "SCORE_SOURCE_INVALID_PAYLOAD")

    def _check_non_object_payload_becomes_invalid_payload_diagnostic(self) -> None:
        fetch = scores_module.fetch_daily_scores_with_diagnostics
        with patch(
            "app.web.scores.requests.get",
            return_value=_Response(["unexpected", "shape"]),
        ):
            try:
                result = fetch(datetime(2026, 8, 10, tzinfo=UTC))
            except Exception as exc:  # pragma: no cover - 失败分支用于形成清晰红测
                self.fail(f"畸形比分响应不得中断导出: {exc}")

        self.assertEqual(result.scores, ())
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["code"], "SCORE_SOURCE_INVALID_PAYLOAD")

    def _check_transport_failure_returns_structured_diagnostic(self) -> None:
        fetch = getattr(scores_module, "fetch_daily_scores_with_diagnostics", None)
        self.assertIsNotNone(fetch, "比分抓取必须暴露带诊断的结果接口")

        with patch(
            "app.web.scores.requests.get",
            side_effect=requests.Timeout("source timed out"),
        ):
            result = fetch(datetime(2026, 8, 10, tzinfo=UTC))

        self.assertEqual(result.scores, ())
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["code"], "SCORE_SOURCE_UNAVAILABLE")
        self.assertIn("source timed out", result.errors[0]["message"])


import unittest

from app.web.preset_questions import build_preset_questions


class TestPresetQuestions(unittest.TestCase):
    def test_default_shortlist_uses_the_operations_threshold(self):
        payload = build_preset_questions([
            {"home_team": "Home", "away_team": "Away", "confidence": confidence}
            for confidence in (0.55, 0.59, 0.6, 0.7)
        ], can_publish_recommendations=True)
        self.assertEqual([item["confidence"] for item in payload["questions"][0]["items"]], [0.7, 0.6])

    def test_non_object_cache_is_ignored(self):
        from app.web.preset_questions import load_cached_questions
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "preset_questions.json"
            for raw in ("[]", "null", "42"):
                path.write_text(raw)
                self.assertIsNone(load_cached_questions(path))

    def test_invalid_confidence_never_enters_recommendations(self) -> None:
        payload = build_preset_questions(
            [
                {
                    "home_team": "North FC",
                    "away_team": "South FC",
                    "competition": "PL",
                    "prediction": "HOME_WIN",
                    "confidence": 2.0,
                }
            ],
            can_publish_recommendations=True,
        )

        recommendations = payload["questions"][0]["items"]
        self.assertEqual(recommendations[0]["match"], "No high-confidence picks yet.")


import hashlib
import json
import stat
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from app.ml.prediction import PredictionRun
from app.web import export_site
from app.web.scores import ScoreFetchResult


class TestExportSiteData(unittest.TestCase):
    def test_non_finite_json_is_rejected_without_replacing_an_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.json"
            path.write_text('{"old": true}\n')
            for value in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    export_site._atomic_write_json(path, {"invalid": value})
                self.assertEqual(path.read_text(), '{"old": true}\n')
                self.assertEqual(list(Path(temp_dir).glob(".*.tmp")), [])

    def _run_contracts(self, names: tuple[str, ...]) -> None:
        for name in names:
            with self.subTest(contract=name):
                getattr(self, name)()

    def test_diagnostic_export_contracts(self) -> None:
        self._run_contracts(
            (
                "_check_refresh_failure_is_exported_while_existing_model_snapshot_survives",
                "_check_score_source_failure_is_exported_for_operations_diagnosis",
            )
        )

    def test_snapshot_export_contracts(self) -> None:
        self._run_contracts(
            (
                "_check_export_writes_one_auditable_snapshot",
                "_check_export_refreshes_same_day_preset_cache",
            )
        )

    def test_atomic_writer_contracts(self) -> None:
        self._run_contracts(
            (
                "_check_atomic_write_preserves_old_target_when_replace_fails",
                "_check_atomic_write_aligns_public_artifact_owner_with_output_directory",
            )
        )

    def _check_refresh_failure_is_exported_while_existing_model_snapshot_survives(self) -> None:
        now = datetime(2026, 8, 10, 9, 0, tzinfo=UTC)
        refresh_error = {
            "competition": "PL",
            "code": "TrainingError",
            "message": "backtest below baseline",
        }
        run = PredictionRun(
            generated_at=now.isoformat(),
            predictions=(),
            errors=(),
            models_loaded=1,
            competitions_requested=1,
            model_reports=(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = export_site.export_site_data(
                prediction_run=run,
                scores=[],
                refresh_errors=[refresh_error],
                output_dir=Path(tmpdir),
                now=now,
            )
            predictions = json.loads(outputs["predictions"].read_text(encoding="utf-8"))
            operations = json.loads(outputs["operations"].read_text(encoding="utf-8"))
            release = json.loads(outputs["release_governance"].read_text(encoding="utf-8"))
            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))

        self.assertEqual(predictions["refresh_errors"], [refresh_error])
        self.assertEqual(operations["model_refresh_error_count"], 1)
        self.assertEqual(manifest["error_count"], 1)
        self.assertEqual(manifest["model_refresh_error_count"], 1)
        self.assertEqual(manifest["prediction_error_count"], 0)
        self.assertEqual(manifest["score_error_count"], 0)
        self.assertIn("MODEL_REFRESH_ERRORS", {item["code"] for item in operations["alerts"]})
        self.assertEqual(release["release_decision"], "blocked")

    def _check_score_source_failure_is_exported_for_operations_diagnosis(self) -> None:
        now = datetime(2026, 8, 10, 9, 0, tzinfo=UTC)
        run = PredictionRun(
            generated_at=now.isoformat(),
            predictions=(),
            errors=(),
            models_loaded=0,
            competitions_requested=0,
        )
        score_error = {
            "code": "SCORE_SOURCE_UNAVAILABLE",
            "message": "比分源请求失败: timed out",
        }
        score_result = ScoreFetchResult(scores=(), errors=(score_error,))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                export_site,
                "fetch_daily_scores_with_diagnostics",
                return_value=score_result,
            ):
                outputs = export_site.export_site_data(
                    prediction_run=run,
                    output_dir=Path(tmpdir),
                    now=now,
                )

            scores_payload = json.loads(outputs["scores"].read_text(encoding="utf-8"))
            operations = json.loads(outputs["operations"].read_text(encoding="utf-8"))
            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))

        self.assertEqual(scores_payload.get("errors"), [score_error])
        self.assertEqual(operations.get("score_error_count"), 1)
        self.assertEqual(manifest["error_count"], 1)
        self.assertEqual(manifest["score_error_count"], 1)
        self.assertIn(
            "SCORE_FETCH_ERRORS",
            {alert["code"] for alert in operations["alerts"]},
        )

    def _check_export_writes_one_auditable_snapshot(self) -> None:
        generated_at = "2026-07-10T08:30:00+00:00"
        predictions = (
            {
                "match_id": 101,
                "forecast_id": "1" * 64,
                "competition": "PL",
                "utc_date": "2026-07-10T19:00:00+00:00",
                "home_team": "North FC",
                "away_team": "South FC",
                "prediction": "HOME_WIN",
                "confidence": 0.72,
                "uncertainty": 0.51,
                "probabilities": {"home_win": 0.72, "draw": 0.18, "away_win": 0.10},
                "data_quality": {"is_cold_start": False},
                "model": {
                    "schema_version": 3,
                    "model_sha256": "a" * 64,
                    "validation_accuracy": 0.61,
                    "baseline_accuracy": 0.48,
                },
            },
        )
        run = PredictionRun(
            generated_at=generated_at,
            predictions=predictions,
            errors=({"competition": "SA", "code": "MODEL_UNAVAILABLE", "message": "missing"},),
            models_loaded=1,
            competitions_requested=2,
            model_reports=(
                {
                    "competition": "PL",
                    "schema_version": 3,
                    "model_sha256": "a" * 64,
                    "trained_at": "2026-07-10T07:00:00+00:00",
                    "validation_accuracy": 0.61,
                    "validation_log_loss": 0.9,
                    "validation_brier_score": 0.57,
                    "validation_ranked_probability_score": 0.21,
                    "baseline_accuracy": 0.48,
                    "backtest_folds": 3,
                    "backtest_rows": 60,
                    "backtest_accuracy": 0.58,
                    "backtest_baseline_accuracy": 0.47,
                    "backtest_log_loss": 0.94,
                    "backtest_brier_score": 0.59,
                    "backtest_ranked_probability_score": 0.22,
                    "backtest_calibration_error": 0.09,
                    "backtest_classwise_calibration_error": 0.08,
                    "backtest_max_class_calibration_error": 0.1,
                    "backtest_sharpness": 0.61,
                    "backtest_worst_fold_accuracy_lift": 0.02,
                    "backtest_accuracy_lift_stddev": 0.03,
                    "backtest_reliability": [
                        {
                            "class_index": class_index,
                            "calibration_error": error,
                            "mean_probability": 1 / 3,
                            "observed_rate": 1 / 3,
                            "bins": [{
                                "lower_bound": 0.0,
                                "upper_bound": 1.0,
                                "count": 60,
                                "mean_probability": 1 / 3,
                                "observed_rate": 1 / 3 + error,
                            }],
                        }
                        for class_index, error in enumerate((0.07, 0.1, 0.07))
                    ],
                    "recovered_from_previous": False,
                },
            ),
        )
        scores = [{"league": "Premier League", "home_team": "East FC", "away_team": "West FC"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "docs" / "data"
            outputs = export_site.export_site_data(
                days=4,
                prediction_run=run,
                scores=scores,
                output_dir=output_dir,
                now=datetime(2026, 7, 10, 8, 30, tzinfo=UTC),
            )

            self.assertEqual(
                set(outputs),
                {"manifest", "predictions", "preset_questions", "scores", "operations", "release_governance"},
            )
            payloads = {
                name: json.loads(path.read_text(encoding="utf-8"))
                for name, path in outputs.items()
            }

            manifest = payloads["manifest"]
            self.assertEqual(manifest["schema"], "SoccerAnalytics.SiteManifest.v3")
            self.assertEqual(manifest["prediction_count"], 1)
            self.assertEqual(manifest["error_count"], 1)
            self.assertEqual(manifest["prediction_error_count"], 1)
            self.assertEqual(manifest["model_refresh_error_count"], 0)
            self.assertEqual(manifest["score_error_count"], 0)
            self.assertEqual(manifest["model_coverage"], {"loaded": 1, "requested": 2, "ratio": 0.5})
            self.assertEqual(payloads["predictions"]["schema"], "SoccerAnalytics.Predictions.v3")
            self.assertEqual(payloads["predictions"]["errors"], list(run.errors))
            self.assertEqual(payloads["predictions"]["model_reports"], list(run.model_reports))
            self.assertEqual(
                payloads["operations"]["model_quality_by_competition"][0]["backtest_accuracy"],
                0.58,
            )
            self.assertEqual(payloads["operations"]["status"], "critical")
            self.assertEqual(payloads["release_governance"]["schema"], "SoccerAnalytics.ReleaseGovernance.v3")
            self.assertEqual(payloads["release_governance"]["release_decision"], "blocked")
            recommended = next(
                question
                for question in payloads["preset_questions"]["questions"]
                if question["id"] == "recommended_bets_today"
            )
            self.assertEqual(recommended["items"], [])
            self.assertFalse(payloads["preset_questions"]["can_publish_recommendations"])

            # 清单最后写入，并以摘要约束其余五个制品的真实字节内容。
            for name, metadata in manifest["artifacts"].items():
                raw = outputs[name].read_bytes()
                self.assertEqual(metadata["sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(metadata["size_bytes"], len(raw))
            timestamps = {
                payloads[name]["generated_at"]
                for name in ("manifest", "predictions", "preset_questions", "scores", "operations", "release_governance")
            }
            self.assertEqual(timestamps, {generated_at})
            self.assertEqual(list(output_dir.glob(".*.tmp")), [])
            for path in outputs.values():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

    def _check_export_refreshes_same_day_preset_cache(self) -> None:
        now = datetime(2026, 7, 10, 9, 0, tzinfo=UTC)
        run = PredictionRun(
            generated_at=now.isoformat(),
            predictions=(),
            errors=(),
            models_loaded=0,
            competitions_requested=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_dir.mkdir(parents=True, exist_ok=True)
            preset_path = output_dir / "preset_questions.json"
            preset_path.write_text(
                json.dumps({"date": "2026-07-10", "generated_at": "old", "questions": []}),
                encoding="utf-8",
            )
            export_site.export_site_data(
                prediction_run=run,
                scores=[],
                output_dir=output_dir,
                now=now,
            )
            refreshed = json.loads(preset_path.read_text(encoding="utf-8"))
            self.assertEqual(refreshed["generated_at"], now.isoformat())
            self.assertNotEqual(refreshed["questions"], [])

    def _check_atomic_write_preserves_old_target_when_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "payload.json"
            target.write_text('{"old": true}\n', encoding="utf-8")
            with patch.object(export_site.os, "replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    export_site._atomic_write_json(target, {"new": True})
            self.assertEqual(target.read_text(encoding="utf-8"), '{"old": true}\n')
            self.assertEqual(list(Path(tmpdir).glob(".*.tmp")), [])

    def _check_atomic_write_aligns_public_artifact_owner_with_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "payload.json"
            parent = target.parent.stat()
            with patch.object(export_site.os, "fchown") as fchown:
                export_site._atomic_write_json(target, {"ready": True})

        self.assertEqual(fchown.call_count, 1)
        self.assertEqual(fchown.call_args.args[1:], (parent.st_uid, parent.st_gid))


import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "docs" / "index.html"
JAVASCRIPT = ROOT / "docs" / "app.js"
STYLES = ROOT / "docs" / "styles.css"


class TestFrontendContract(unittest.TestCase):
    def test_static_dashboard_contracts(self):
        for name in (
            "_check_dashboard_exposes_quality_first_controls_and_states",
            "_check_script_understands_v2_quality_and_partial_failure_fields",
            "_check_styles_cover_keyboard_mobile_and_reduced_motion",
        ):
            with self.subTest(contract=name):
                getattr(self, name)()

    def _check_dashboard_exposes_quality_first_controls_and_states(self):
        html = HTML.read_text(encoding="utf-8")
        for element_id in (
            "release-command",
            "model-coverage-track",
            "prediction-search",
            "competition-filter",
            "confidence-filter",
            "prediction-sort",
            "prediction-errors",
        ):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn('class="skip-link"', html)
        self.assertNotIn("bettings", html.lower())

    def _check_script_understands_v2_quality_and_partial_failure_fields(self):
        javascript = JAVASCRIPT.read_text(encoding="utf-8")
        for field in (
            "model_coverage",
            "cold_start_ratio",
            "models_below_baseline",
            "backtest_calibration_error",
            "models_recovered_from_previous",
            "SNAPSHOT_MISMATCH",
            "can_publish_recommendations",
            "sha256",
            "errors",
            "uncertainty",
            "probability_margin",
            "top_drivers",
            "backtest_ranked_probability_score",
            "predicted_at",
            "forecast_id",
            "model_sha256",
        ):
            self.assertIn(field, javascript)
        result = subprocess.run(
            ["node", "--check", str(JAVASCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def _check_styles_cover_keyboard_mobile_and_reduced_motion(self):
        styles = STYLES.read_text(encoding="utf-8")
        self.assertIn(":focus-visible", styles)
        self.assertIn("prefers-reduced-motion", styles)
        self.assertRegex(styles, r"@media\s*\(max-width:\s*760px\)")

    def test_runtime_diagnoses_missing_artifacts_and_shows_quality_detail(self):
        for scenario in (
            "missing",
            "quality",
            "quality-mismatch",
            "score-error",
            "backtest",
            "refresh-error",
            "stale",
            "blocked-shortlist",
            "mixed-snapshot",
            "tampered",
            "approved-shortlist",
            "expired-shortlist",
            "inconsistent-approval",
            "malformed-predictions",
            "future-snapshot",
            "expiry-timer",
            "missing-expiry",
        ):
            result = subprocess.run(
                ["node", "-e", FRONTEND_RUNTIME_HARNESS, scenario],
                check=False,
                capture_output=True,
                text=True,
            )
            with self.subTest(scenario=scenario):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"{scenario}: ok", result.stdout)
