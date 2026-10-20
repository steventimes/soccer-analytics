from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import requests

from app.ml.prediction import PredictionRun
from app.web import scores as scores_module, export_site
from app.web.preset_questions import build_preset_questions
from app.web.scores import ScoreFetchResult

from tests.support import ContractTestCase


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


class TestDailyScores(ContractTestCase):
    def test_source_failure_contracts(self) -> None:
        self._run_contracts(
            (
                '_check_requests_json_decode_error_is_not_misclassified_as_network_failure',
                '_check_events_field_must_be_a_list',
                '_check_non_object_payload_becomes_invalid_payload_diagnostic',
                '_check_transport_failure_returns_structured_diagnostic',
            )
        )

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


class TestExportSiteData(ContractTestCase):
    def test_non_finite_json_is_rejected_without_replacing_an_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.json"
            path.write_text('{"old": true}\n')
            for value in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    export_site._atomic_write_json(path, {"invalid": value})
                self.assertEqual(path.read_text(), '{"old": true}\n')
                self.assertEqual(list(Path(temp_dir).glob(".*.tmp")), [])


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


ROOT = Path(__file__).resolve().parents[1]


class TestFrontendContract(unittest.TestCase):


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
                ["node", str(ROOT / "tests/frontend_runtime.cjs"), scenario],
                check=False,
                capture_output=True,
                text=True,
            )
            with self.subTest(scenario=scenario):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"{scenario}: ok", result.stdout)
