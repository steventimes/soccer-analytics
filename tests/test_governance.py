from __future__ import annotations

import math

from app.ml.forecast_identity import forecast_id_for_prediction
from app.web.operations import build_operations_summary
from app.web.release_governance import build_release_governance

from tests.support import ContractTestCase


def _quality_evidence(
    competition: str,
    *,
    validation_accuracy: float = 0.62,
    baseline_accuracy: float = 0.48,
) -> dict:
    return {
        "competition": competition,
        "schema_version": 3,
        "model_sha256": "a" * 64,
        "estimator": "xgboost",
        "estimator_version": "3.4.1",
        "explanation_method": "tree_shap_margin",
        "boosting_rounds": 84,
        "trained_at": "2026-07-08T00:00:00+00:00",
        "validation_accuracy": validation_accuracy,
        "validation_log_loss": 0.89,
        "validation_brier_score": 0.56,
        "validation_ranked_probability_score": 0.21,
        "baseline_accuracy": baseline_accuracy,
        "backtest_folds": 3,
        "backtest_rows": 60,
        "backtest_accuracy": 0.58,
        "backtest_baseline_accuracy": 0.47,
        "majority_baseline_accuracy": 0.48,
        "elo_baseline_accuracy": 0.51,
        "backtest_majority_baseline_accuracy": 0.44,
        "backtest_elo_baseline_accuracy": 0.47,
        "backtest_log_loss": 0.92,
        "backtest_brier_score": 0.59,
        "backtest_calibration_error": 0.09,
        "backtest_ranked_probability_score": 0.22,
        "backtest_classwise_calibration_error": 0.08,
        "backtest_max_class_calibration_error": 0.1,
        "backtest_sharpness": 0.61,
        "backtest_worst_fold_accuracy_lift": 0.02,
        "backtest_accuracy_lift_stddev": 0.03,
        "backtest_reliability": [
            {
                "class_index": index,
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
            for index, error in enumerate((0.07, 0.1, 0.07))
        ],
        "recovered_from_previous": False,
    }


def _valid_prediction(
    match_id: int,
    competition: str = "PL",
    confidence: float = 0.72,
) -> dict:
    remainder = (1.0 - confidence) / 2.0
    prediction = {
        "match_id": match_id,
        "competition": competition,
        "utc_date": "2026-07-10T19:00:00+00:00",
        "home_team_id": 100 + match_id,
        "away_team_id": 200 + match_id,
        "feature_fingerprint": "f" * 64,
        "prediction": "HOME_WIN",
        "confidence": confidence,
        "uncertainty": 0.4,
        "probabilities": {
            "home_win": confidence,
            "draw": remainder,
            "away_win": remainder,
        },
        "data_quality": {
            "is_cold_start": False,
            "home_history_matches": 5,
            "away_history_matches": 5,
        },
        "model": _quality_evidence(competition),
    }
    prediction["forecast_id"] = forecast_id_for_prediction(prediction)
    return prediction


class TestOperationsSummary(ContractTestCase):

    def test_prediction_integrity_contracts(self) -> None:
        self._run_contracts(
            (
                "_check_complete_inline_model_metadata_cannot_replace_explicit_report",
                "_check_cold_start_lineage_tampering_is_critical",
                "_check_forecast_id_is_recomputed_from_exported_lineage",
                "_check_forecast_digest_case_variants_are_one_conflicting_identity",
                "_check_invalid_probability_contract_is_critical",
                "_check_duplicate_forecasts_and_conflicts_are_critical",
            )
        )

    def test_model_evidence_contracts(self) -> None:
        self._run_contracts(
            (
                "_check_quality_evidence_preserves_model_identity",
                "_check_malformed_reliability_bins_fail_quality_closed",
                "_check_worst_class_calibration_blocks_otherwise_healthy_model",
                "_check_temporal_instability_is_a_critical_model_signal",
                "_check_validation_only_inline_model_metadata_fails_closed",
                "_check_fallback_and_staleness_survive_incomplete_quality_evidence",
                "_check_model_reports_drive_quality_without_upcoming_fixtures",
                "_check_stale_model_is_a_critical_quality_signal",
                "_check_missing_model_quality_metrics_are_critical",
            )
        )

    def test_summary_health_contracts(self) -> None:
        self._run_contracts(
            (
                "_check_builds_healthy_summary_for_covered_quality_run",
                "_check_flags_model_coverage_cold_start_and_below_baseline",
                "_check_flags_empty_and_low_quality_exports",
            )
        )

    def _check_complete_inline_model_metadata_cannot_replace_explicit_report(self) -> None:
        summary = build_operations_summary(
            [_valid_prediction(1)],
            [{"league": "Premier League"}],
            models_loaded=1,
            competitions_requested=1,
            generated_at="2026-07-09T00:00:00+00:00",
        )

        self.assertEqual(summary["models_without_quality_metrics"], ["PL"])
        self.assertIn(
            "MODEL_QUALITY_UNAVAILABLE",
            {item["code"] for item in summary["alerts"]},
        )
        self.assertEqual(summary["status"], "critical")

    def _check_cold_start_lineage_tampering_is_critical(self) -> None:
        for mutation in ("flag", "history"):
            with self.subTest(mutation=mutation):
                prediction = _valid_prediction(1)
                prediction["data_quality"] = {
                    "is_cold_start": True,
                    "home_history_matches": 0,
                    "away_history_matches": 5,
                }
                prediction["forecast_id"] = forecast_id_for_prediction(prediction)
                if mutation == "flag":
                    prediction["data_quality"]["is_cold_start"] = False
                else:
                    prediction["data_quality"]["home_history_matches"] = 99
                summary = build_operations_summary(
                    [prediction],
                    [{"league": "Premier League"}],
                    model_reports=[_quality_evidence("PL")],
                    models_loaded=1,
                    competitions_requested=1,
                    generated_at="2026-07-09T00:00:00+00:00",
                )
                self.assertEqual(summary["invalid_prediction_quality_count"], 1)
                self.assertIn(
                    "PREDICTION_DATA_QUALITY_INVALID",
                    {item["code"] for item in summary["alerts"]},
                )
                self.assertEqual(summary["status"], "critical")

    def _check_forecast_id_is_recomputed_from_exported_lineage(self) -> None:
        for mutation in (
            "match",
            "kickoff",
            "home_team",
            "model",
            "probabilities",
            "feature_fingerprint",
        ):
            with self.subTest(mutation=mutation):
                prediction = _valid_prediction(1)
                if mutation == "match":
                    prediction["match_id"] = 99
                elif mutation == "kickoff":
                    prediction["utc_date"] = "2026-07-11T19:00:00+00:00"
                elif mutation == "home_team":
                    prediction["home_team_id"] = 999
                elif mutation == "model":
                    prediction["model"]["model_sha256"] = "b" * 64
                elif mutation == "probabilities":
                    prediction["probabilities"] = {
                        "home_win": 0.9,
                        "draw": 0.05,
                        "away_win": 0.05,
                    }
                    prediction["confidence"] = 0.9
                    prediction["prediction"] = "HOME_WIN"
                else:
                    prediction["feature_fingerprint"] = "e" * 64
                summary = build_operations_summary(
                    [prediction],
                    [{"league": "Premier League"}],
                    model_reports=[_quality_evidence("PL")],
                    models_loaded=1,
                    competitions_requested=1,
                    generated_at="2026-07-09T00:00:00+00:00",
                )
                self.assertEqual(summary["invalid_prediction_identity_count"], 1)
                self.assertIn(
                    "PREDICTION_IDENTITY_INVALID",
                    {item["code"] for item in summary["alerts"]},
                )
                self.assertEqual(summary["status"], "critical")

    def _check_forecast_digest_case_variants_are_one_conflicting_identity(self) -> None:
        digest = "ab" * 32
        predictions = [
            {**_valid_prediction(index), "forecast_id": value}
            for index, value in enumerate(
                (digest, digest.upper(), "Ab" * 32),
                start=1,
            )
        ]
        summary = build_operations_summary(
            predictions,
            [{"league": "Premier League"}],
            model_reports=[_quality_evidence("PL")],
            models_loaded=1,
            competitions_requested=1,
            generated_at="2026-07-09T00:00:00+00:00",
        )

        self.assertEqual(summary["prediction_count"], 1)
        self.assertEqual(summary["forecast_conflict_count"], 2)
        self.assertEqual(summary["status"], "critical")

    def _check_invalid_probability_contract_is_critical(self) -> None:
        for mutation in (
            "negative",
            "above_one",
            "two",
            "bool",
            "nan",
            "sum",
            "confidence_mismatch",
            "class_mismatch",
        ):
            with self.subTest(mutation=mutation):
                prediction = _valid_prediction(1)
                if mutation == "negative":
                    prediction["confidence"] = -0.1
                elif mutation == "above_one":
                    prediction["confidence"] = 1.0001
                elif mutation == "two":
                    prediction["confidence"] = 2.0
                elif mutation == "bool":
                    prediction["confidence"] = True
                elif mutation == "nan":
                    prediction["confidence"] = float("nan")
                elif mutation == "sum":
                    prediction["probabilities"]["draw"] = 0.5
                elif mutation == "confidence_mismatch":
                    prediction["confidence"] = 0.7
                else:
                    prediction["prediction"] = "AWAY_WIN"
                summary = build_operations_summary(
                    [prediction],
                    [{"league": "Premier League"}],
                    model_reports=[_quality_evidence("PL")],
                    models_loaded=1,
                    competitions_requested=1,
                    generated_at="2026-07-09T00:00:00+00:00",
                )
                self.assertEqual(summary["invalid_prediction_probability_count"], 1)
                self.assertIn(
                    "PREDICTION_PROBABILITIES_INVALID",
                    {item["code"] for item in summary["alerts"]},
                )
                self.assertEqual(summary["status"], "critical")

    def _check_quality_evidence_preserves_model_identity(self) -> None:
        report = _quality_evidence("PL")
        summary = build_operations_summary(
            [
                {
                    "forecast_id": "1" * 64,
                    "match_id": 1,
                    "competition": "PL",
                    "confidence": 0.72,
                    "model": {"schema_version": 3, "model_sha256": "a" * 64},
                }
            ],
            [{"league": "Premier League"}],
            model_reports=[report],
            models_loaded=1,
            competitions_requested=1,
            generated_at="2026-07-09T00:00:00+00:00",
        )

        evidence = summary["model_quality_by_competition"][0]
        self.assertEqual(evidence["schema_version"], 3)
        self.assertEqual(evidence["model_sha256"], "a" * 64)

    def _check_malformed_reliability_bins_fail_quality_closed(self) -> None:
        for mutation in ("count", "calibration"):
            with self.subTest(mutation=mutation):
                report = _quality_evidence("PL")
                if mutation == "count":
                    report["backtest_reliability"][0]["bins"][0]["count"] = 59
                else:
                    report["backtest_reliability"][0]["calibration_error"] = 0.01
                summary = build_operations_summary(
                    [{"competition": "PL", "confidence": 0.72}],
                    [{"league": "Premier League"}],
                    model_reports=[report],
                    models_loaded=1,
                    competitions_requested=1,
                    generated_at="2026-07-09T00:00:00+00:00",
                )
                self.assertEqual(summary["models_without_quality_metrics"], ["PL"])
                self.assertEqual(summary["status"], "critical")
    def _check_duplicate_forecasts_and_conflicts_are_critical(self) -> None:
        base = {
            "forecast_id": "forecast-1",
            "match_id": 1,
            "competition": "PL",
            "confidence": 0.72,
        }
        for duplicate in (dict(base), {**base, "confidence": 0.61}):
            with self.subTest(duplicate=duplicate):
                summary = build_operations_summary(
                    [base, duplicate],
                    [{"league": "Premier League"}],
                    model_reports=[_quality_evidence("PL")],
                    models_loaded=1,
                    competitions_requested=1,
                    generated_at="2026-07-09T00:00:00+00:00",
                )
                self.assertEqual(summary["prediction_count"], 1)
                self.assertEqual(summary["duplicate_forecast_count"], 1)
                self.assertEqual(summary["status"], "critical")
                self.assertIn(
                    "FORECAST_ID_CONFLICT" if duplicate["confidence"] != 0.72 else "DUPLICATE_FORECASTS",
                    {item["code"] for item in summary["alerts"]},
                )

    def _check_worst_class_calibration_blocks_otherwise_healthy_model(self) -> None:
        report = _quality_evidence("PL")
        report["backtest_reliability"][1]["calibration_error"] = 0.95
        report["backtest_reliability"][1]["bins"][0]["mean_probability"] = 0.025
        report["backtest_reliability"][1]["bins"][0]["observed_rate"] = 0.975
        report["backtest_classwise_calibration_error"] = (0.07 + 0.95 + 0.07) / 3
        report["backtest_max_class_calibration_error"] = 0.95
        summary = build_operations_summary(
            [{"competition": "PL", "confidence": 0.72}],
            [{"league": "Premier League"}],
            model_reports=[report],
            models_loaded=1,
            competitions_requested=1,
            generated_at="2026-07-09T00:00:00+00:00",
        )
        self.assertEqual(summary["models_with_poor_classwise_calibration"], ["PL"])
        self.assertIn(
            "MODEL_CLASSWISE_CALIBRATION_POOR",
            {item["code"] for item in summary["alerts"]},
        )
        self.assertEqual(summary["status"], "critical")
    def _check_temporal_instability_is_a_critical_model_signal(self) -> None:
        report = _quality_evidence("PL")
        report["backtest_worst_fold_accuracy_lift"] = -0.2

        summary = build_operations_summary(
            [{"competition": "PL", "confidence": 0.72}],
            [{"league": "Premier League"}],
            model_reports=[report],
            models_loaded=1,
            competitions_requested=1,
            generated_at="2026-07-09T00:00:00+00:00",
        )

        self.assertEqual(summary["models_temporally_unstable"], ["PL"])
        self.assertIn(
            "MODEL_TEMPORALLY_UNSTABLE",
            {item["code"] for item in summary["alerts"]},
        )
        self.assertEqual(summary["status"], "critical")

    def _check_validation_only_inline_model_metadata_fails_closed(self) -> None:
        summary = build_operations_summary(
            [
                {
                    "competition": "PL",
                    "confidence": 0.72,
                    "uncertainty": 0.4,
                    "model": {
                        "validation_accuracy": 0.62,
                        "baseline_accuracy": 0.5,
                    },
                }
            ],
            [{"league": "Premier League"}],
            models_loaded=1,
            competitions_requested=1,
            generated_at="2026-08-10T09:00:00+00:00",
        )

        self.assertEqual(summary["status"], "critical")
        self.assertEqual(summary["models_without_quality_metrics"], ["PL"])
        self.assertIn(
            "MODEL_QUALITY_UNAVAILABLE",
            {item["code"] for item in summary["alerts"]},
        )

    def _check_fallback_and_staleness_survive_incomplete_quality_evidence(self) -> None:
        report = {
            "competition": "PL",
            "trained_at": "2026-08-01T08:00:00+00:00",
            "validation_accuracy": 0.62,
            "validation_log_loss": 0.89,
            "validation_brier_score": 0.56,
            "baseline_accuracy": 0.5,
            "backtest_folds": 3,
            "backtest_rows": 60,
            "backtest_accuracy": 0.58,
            "backtest_baseline_accuracy": 0.48,
            "backtest_log_loss": 0.92,
            "backtest_calibration_error": 0.09,
            "recovered_from_previous": True,
        }

        summary = build_operations_summary(
            [{"competition": "PL", "confidence": 0.72}],
            [{"league": "Premier League"}],
            model_reports=[report],
            models_loaded=1,
            competitions_requested=1,
            generated_at="2026-08-10T08:00:00+00:00",
        )

        self.assertEqual(summary["models_without_quality_metrics"], ["PL"])
        self.assertEqual(summary["models_recovered_from_previous"], ["PL"])
        self.assertEqual(summary["models_stale"], ["PL"])
        codes = {item["code"] for item in summary["alerts"]}
        self.assertTrue(
            {"MODEL_QUALITY_UNAVAILABLE", "MODEL_FALLBACK_ACTIVE", "MODEL_STALE"}
            <= codes
        )
    def _check_model_reports_drive_quality_without_upcoming_fixtures(self) -> None:
        report = {
            **_quality_evidence("PL"),
            "trained_at": "2026-08-10T07:00:00+00:00",
            "backtest_accuracy": 0.42,
            "backtest_baseline_accuracy": 0.48,
            "backtest_log_loss": 1.02,
            "backtest_brier_score": 0.67,
            "backtest_calibration_error": 0.31,
            "recovered_from_previous": True,
        }

        summary = build_operations_summary(
            [],
            [{"league": "Premier League"}],
            model_reports=[report],
            models_loaded=1,
            competitions_requested=1,
            generated_at="2026-08-10T09:00:00+00:00",
        )

        self.assertEqual(summary["model_quality_by_competition"][0]["competition"], "PL")
        self.assertEqual(summary["model_quality_by_competition"][0]["backtest_folds"], 3)
        self.assertEqual(
            summary["model_quality_by_competition"][0]["backtest_elo_baseline_accuracy"],
            0.47,
        )
        self.assertEqual(summary["models_without_quality_metrics"], [])
        self.assertEqual(summary["models_below_backtest_baseline"], ["PL"])
        self.assertEqual(summary["models_with_poor_calibration"], ["PL"])
        self.assertEqual(summary["models_recovered_from_previous"], ["PL"])
        codes = {alert["code"] for alert in summary["alerts"]}
        self.assertIn("MODEL_BACKTEST_BELOW_BASELINE", codes)
        self.assertIn("MODEL_CALIBRATION_POOR", codes)
        self.assertIn("MODEL_FALLBACK_ACTIVE", codes)

    def _check_stale_model_is_a_critical_quality_signal(self) -> None:
        report = {
            **_quality_evidence("PL"),
            "trained_at": "2026-08-01T08:00:00+00:00",
            "recovered_from_previous": False,
        }

        summary = build_operations_summary(
            [{"competition": "PL", "confidence": 0.72}],
            [{"league": "Premier League"}],
            model_reports=[report],
            models_loaded=1,
            competitions_requested=1,
            generated_at="2026-08-10T08:00:00+00:00",
            maximum_model_age_hours=72,
        )

        self.assertEqual(summary["models_stale"], ["PL"])
        self.assertEqual(
            summary["model_quality_by_competition"][0]["model_age_hours"],
            216.0,
        )
        self.assertIn("MODEL_STALE", {item["code"] for item in summary["alerts"]})
        self.assertEqual(summary["status"], "critical")

    def _check_builds_healthy_summary_for_covered_quality_run(self) -> None:
        summary = build_operations_summary(
            [
                _valid_prediction(1, "PL", 0.72),
                {**_valid_prediction(2, "PL", 0.61), "uncertainty": 0.6},
                {
                    **_valid_prediction(3, "SA", 0.54),
                    "uncertainty": 0.5,
                    "model": _quality_evidence(
                        "SA", validation_accuracy=0.58, baseline_accuracy=0.51
                    ),
                },
            ],
            [{"league": "Premier League"}],
            models_loaded=2,
            competitions_requested=2,
            model_reports=[
                _quality_evidence("PL"),
                _quality_evidence(
                    "SA", validation_accuracy=0.58, baseline_accuracy=0.51
                ),
            ],
            generated_at="2026-07-09T00:00:00+00:00",
        )
        self.assertEqual(summary["schema"], "SoccerAnalytics.Operations.v3")
        self.assertEqual(summary["status"], "healthy")
        self.assertEqual(summary["average_confidence"], 0.6233)
        self.assertEqual(summary["average_uncertainty"], 0.5)
        self.assertEqual(summary["model_coverage"]["ratio"], 1.0)
        self.assertEqual(summary["cold_start_ratio"], 0.0)
        self.assertEqual(summary["models_below_baseline"], [])
        self.assertEqual(summary["alerts"], [])

    def _check_flags_model_coverage_cold_start_and_below_baseline(self) -> None:
        predictions = []
        for match_id, confidence, uncertainty in (
            (1, 0.7, 0.8),
            (2, 0.65, 0.7),
        ):
            prediction = _valid_prediction(match_id, confidence=confidence)
            prediction["uncertainty"] = uncertainty
            prediction["data_quality"] = {
                "is_cold_start": True,
                "home_history_matches": 0,
                "away_history_matches": 5,
            }
            prediction["model"] = _quality_evidence(
                "PL", validation_accuracy=0.4, baseline_accuracy=0.5
            )
            prediction["forecast_id"] = forecast_id_for_prediction(prediction)
            predictions.append(prediction)
        summary = build_operations_summary(
            predictions,
            [{"league": "Premier League"}],
            prediction_errors=[{"competition": "SA", "code": "MODEL_UNAVAILABLE", "message": "missing"}],
            model_reports=[
                _quality_evidence(
                    "PL", validation_accuracy=0.4, baseline_accuracy=0.5
                )
            ],
            models_loaded=1,
            competitions_requested=2,
            generated_at="2026-07-09T00:00:00+00:00",
        )
        codes = {alert["code"] for alert in summary["alerts"]}
        self.assertEqual(summary["status"], "critical")
        self.assertEqual(summary["model_coverage"]["ratio"], 0.5)
        self.assertEqual(summary["cold_start_ratio"], 1.0)
        self.assertEqual(summary["models_below_baseline"], ["PL"])
        self.assertIn("MODEL_COVERAGE_GAP", codes)
        self.assertIn("PREDICTION_ERRORS", codes)
        self.assertIn("COLD_START_CLUSTER", codes)
        self.assertIn("MODEL_BELOW_BASELINE", codes)

    def _check_flags_empty_and_low_quality_exports(self) -> None:
        empty = build_operations_summary(
            [],
            [],
            models_loaded=0,
            competitions_requested=0,
        )
        self.assertEqual(empty["status"], "critical")
        self.assertEqual({alert["code"] for alert in empty["alerts"]}, {"NO_PREDICTIONS", "NO_SCORES"})

    def _check_missing_model_quality_metrics_are_critical(self) -> None:
        summary = build_operations_summary(
            [
                {
                    "competition": "PL",
                    "confidence": 0.72,
                    "uncertainty": 0.4,
                    "data_quality": {"is_cold_start": False},
                    "model": {"validation_accuracy": None, "baseline_accuracy": None},
                }
            ],
            [{"league": "Premier League"}],
            models_loaded=1,
            competitions_requested=1,
        )
        self.assertEqual(summary["status"], "critical")
        self.assertEqual(summary["models_without_quality_metrics"], ["PL"])
        self.assertIn(
            "MODEL_QUALITY_UNAVAILABLE",
            {alert["code"] for alert in summary["alerts"]},
        )


MODEL_SHA = "a" * 64


def _quality_row(competition: str = "PL", model_sha256: str = MODEL_SHA) -> dict:
    return {
        "competition": competition,
        "schema_version": 3,
        "model_sha256": model_sha256,
        "trained_at": "2026-07-08T00:00:00+00:00",
        "validation_accuracy": 0.62,
        "baseline_accuracy": 0.48,
        "validation_log_loss": 0.89,
        "validation_brier_score": 0.56,
        "validation_ranked_probability_score": 0.21,
        "backtest_folds": 3,
        "backtest_rows": 60,
        "backtest_accuracy": 0.58,
        "backtest_baseline_accuracy": 0.47,
        "backtest_log_loss": 0.92,
        "backtest_brier_score": 0.59,
        "backtest_ranked_probability_score": 0.22,
        "backtest_calibration_error": 0.09,
        "backtest_classwise_calibration_error": 0.08,
        "backtest_max_class_calibration_error": 0.1,
        "backtest_sharpness": 0.61,
        "backtest_worst_fold_accuracy_lift": 0.02,
        "backtest_accuracy_lift_stddev": 0.03,
        "model_age_hours": 24.0,
        "backtest_reliability": [
            {
                "class_index": class_index,
                "calibration_error": error,
                "sample_count": 60,
            }
            for class_index, error in enumerate((0.07, 0.1, 0.07))
        ],
    }


def _valid_release_inputs() -> tuple[list[dict], dict]:
    predictions = []
    for index, value in enumerate((0.72, 0.61, 0.56), start=1):
        prediction = {
            "match_id": index,
            "competition": "PL",
            "utc_date": "2026-07-10T19:00:00+00:00",
            "home_team_id": 100 + index,
            "away_team_id": 200 + index,
            "feature_fingerprint": "f" * 64,
            "data_quality": {
                "is_cold_start": False,
                "home_history_matches": 5,
                "away_history_matches": 5,
            },
            "prediction": "HOME_WIN",
            "confidence": value,
            "probabilities": {
                "home_win": value,
                "draw": (1.0 - value) / 2.0,
                "away_win": (1.0 - value) / 2.0,
            },
            "model": {
                "schema_version": 3,
                "model_sha256": MODEL_SHA,
                "trained_at": "2026-07-08T00:00:00+00:00",
            },
        }
        prediction["forecast_id"] = forecast_id_for_prediction(prediction)
        predictions.append(prediction)
    operations = {
        "schema": "SoccerAnalytics.Operations.v3",
        "generated_at": "2026-07-09T00:00:00+00:00",
        "status": "healthy",
        "prediction_count": 3,
        "average_confidence": 0.63,
        "high_confidence_count": 2,
        "model_coverage": {"loaded": 1, "requested": 1, "ratio": 1.0},
        "model_quality_by_competition": [_quality_row()],
        "cold_start_count": 0,
        "cold_start_ratio": 0.0,
        "models_below_baseline": [],
        "models_without_quality_metrics": [],
        "models_below_backtest_baseline": [],
        "models_with_poor_calibration": [],
        "models_with_poor_classwise_calibration": [],
        "models_recovered_from_previous": [],
        "models_stale": [],
        "models_temporally_unstable": [],
        "thresholds": {
            "high_confidence": 0.6,
            "maximum_calibration_error": 0.2,
            "maximum_classwise_calibration_error": 0.2,
            "maximum_model_age_hours": 72.0,
            "minimum_worst_fold_accuracy_lift": -0.15,
        },
        "alerts": [],
    }
    return predictions, operations


class TestReleaseGovernance(ContractTestCase):
    def test_release_expires_within_a_day_and_before_the_first_kickoff(self):
        predictions, operations = _valid_release_inputs()
        release = build_release_governance(predictions, operations, generated_at=operations["generated_at"])
        self.assertTrue(release["can_publish_recommendations"])
        self.assertEqual(release["expires_at"], "2026-07-10T00:00:00+00:00")
        predictions[0]["utc_date"] = "2026-07-09T19:00:00+00:00"
        predictions[0]["forecast_id"] = forecast_id_for_prediction(predictions[0])
        release = build_release_governance(predictions, operations, generated_at=operations["generated_at"])
        self.assertTrue(release["can_publish_recommendations"])
        self.assertEqual(release["expires_at"], predictions[0]["utc_date"])

    def test_started_fixture_cannot_be_released_even_with_valid_identity(self):
        predictions, operations = _valid_release_inputs()
        predictions[0]["utc_date"] = operations["generated_at"]
        predictions[0]["forecast_id"] = forecast_id_for_prediction(predictions[0])
        release = build_release_governance(predictions, operations, generated_at=operations["generated_at"])
        self.assertFalse(release["can_publish_recommendations"])
        self.assertIn("PREDICTION_KICKOFF_INVALID", {item["code"] for item in release["blockers"]})


    def test_prediction_audit_contracts(self) -> None:
        self._run_contracts(
            (
                "_check_release_recomputes_cold_start_evidence",
                "_check_release_recomputes_forecast_id_from_exported_lineage",
                "_check_forecast_digest_case_variants_cannot_inflate_release",
                "_check_release_rejects_invalid_probability_contract",
                "_check_missing_cryptographic_prediction_identity_cannot_be_approved",
                "_check_prediction_model_must_match_competition_quality_evidence",
                "_check_duplicate_forecasts_cannot_inflate_release_coverage",
                "_check_same_match_with_different_forecast_ids_is_a_conflict",
            )
        )

    def test_operations_evidence_contracts(self) -> None:
        self._run_contracts(
            (
                "_check_release_recomputes_model_age_and_snapshot_identity",
                "_check_release_recomputes_quality_summaries_and_thresholds",
                "_check_inconsistent_model_coverage_is_rejected",
                "_check_bare_competition_names_cannot_forge_complete_v3_evidence",
                "_check_missing_model_evidence_coverage_blocks_claimed_healthy_summary",
                "_check_blocks_missing_or_non_finite_quality_evidence",
            )
        )

    def test_release_decision_contracts(self) -> None:
        self._run_contracts(
            (
                "_check_blocks_failed_backtest_poor_calibration_and_fallback_model",
                "_check_approves_complete_quality_checked_release",
                "_check_blocks_incomplete_models_bad_validation_and_cold_start",
            )
        )

    def _check_release_recomputes_cold_start_evidence(self) -> None:
        for mutation in ("flag", "history", "summary"):
            with self.subTest(mutation=mutation):
                predictions, operations = _valid_release_inputs()
                prediction = predictions[0]
                prediction["data_quality"] = {
                    "is_cold_start": True,
                    "home_history_matches": 0,
                    "away_history_matches": 5,
                }
                prediction["forecast_id"] = forecast_id_for_prediction(prediction)
                operations["cold_start_count"] = 1
                operations["cold_start_ratio"] = round(1 / 3, 4)
                if mutation == "flag":
                    prediction["data_quality"]["is_cold_start"] = False
                elif mutation == "history":
                    prediction["data_quality"]["home_history_matches"] = 99
                else:
                    operations["cold_start_count"] = 0
                    operations["cold_start_ratio"] = 0.0

                release = build_release_governance(
                    predictions,
                    operations,
                    generated_at="2026-07-09T00:00:00+00:00",
                )

                self.assertEqual(release["release_decision"], "blocked")
                expected = (
                    "COLD_START_AUDIT_MISMATCH"
                    if mutation == "summary"
                    else "FORECAST_IDENTITY_INCOMPLETE"
                )
                self.assertIn(expected, {item["code"] for item in release["blockers"]})

    def _check_release_recomputes_forecast_id_from_exported_lineage(self) -> None:
        for mutation in (
            "match",
            "kickoff",
            "home_team",
            "model",
            "probabilities",
            "feature_fingerprint",
        ):
            with self.subTest(mutation=mutation):
                predictions, operations = _valid_release_inputs()
                prediction = predictions[0]
                if mutation == "match":
                    prediction["match_id"] = 99
                elif mutation == "kickoff":
                    prediction["utc_date"] = "2026-07-11T19:00:00+00:00"
                elif mutation == "home_team":
                    prediction["home_team_id"] = 999
                elif mutation == "model":
                    prediction["model"]["model_sha256"] = "b" * 64
                elif mutation == "probabilities":
                    prediction["probabilities"] = {
                        "home_win": 0.9,
                        "draw": 0.05,
                        "away_win": 0.05,
                    }
                    prediction["confidence"] = 0.9
                    prediction["prediction"] = "HOME_WIN"
                else:
                    prediction["feature_fingerprint"] = "e" * 64

                release = build_release_governance(
                    predictions,
                    operations,
                    generated_at="2026-07-09T00:00:00+00:00",
                )

                self.assertEqual(release["release_decision"], "blocked")
                self.assertIn(
                    "FORECAST_IDENTITY_INCOMPLETE",
                    {item["code"] for item in release["blockers"]},
                )

    def _check_forecast_digest_case_variants_cannot_inflate_release(self) -> None:
        predictions, operations = _valid_release_inputs()
        digest = "ab" * 32
        for prediction, value in zip(
            predictions,
            (digest, digest.upper(), "Ab" * 32),
            strict=True,
        ):
            prediction["forecast_id"] = value

        release = build_release_governance(
            predictions,
            operations,
            generated_at="2026-07-09T00:00:00+00:00",
        )

        self.assertEqual(release["release_decision"], "blocked")
        self.assertEqual(release["audit"]["unique_prediction_count"], 1)
        self.assertIn(
            "FORECAST_ID_CONFLICT",
            {item["code"] for item in release["blockers"]},
        )

    def _check_release_recomputes_model_age_and_snapshot_identity(self) -> None:
        for mutation in ("stale", "future", "snapshot_mismatch"):
            with self.subTest(mutation=mutation):
                predictions, operations = _valid_release_inputs()
                evidence = operations["model_quality_by_competition"][0]
                if mutation == "stale":
                    evidence["trained_at"] = "2020-01-01T00:00:00+00:00"
                    evidence["model_age_hours"] = 1.0
                elif mutation == "future":
                    evidence["trained_at"] = "2027-01-01T00:00:00+00:00"
                    evidence["model_age_hours"] = 1.0
                else:
                    operations["generated_at"] = "2026-07-09T01:00:00+00:00"

                release = build_release_governance(
                    predictions,
                    operations,
                    generated_at="2026-07-09T00:00:00+00:00",
                )

                self.assertEqual(release["release_decision"], "blocked")
                expected = (
                    "OPERATIONS_SNAPSHOT_MISMATCH"
                    if mutation == "snapshot_mismatch"
                    else "MODEL_EVIDENCE_INCOMPLETE"
                )
                self.assertIn(expected, {item["code"] for item in release["blockers"]})

    def _check_release_rejects_invalid_probability_contract(self) -> None:
        for mutation in ("negative", "above_one", "two", "bool", "nan", "sum", "confidence", "class"):
            with self.subTest(mutation=mutation):
                predictions, operations = _valid_release_inputs()
                prediction = predictions[0]
                if mutation == "negative":
                    prediction["confidence"] = -0.1
                elif mutation == "above_one":
                    prediction["confidence"] = 1.0001
                elif mutation == "two":
                    prediction["confidence"] = 2.0
                elif mutation == "bool":
                    prediction["confidence"] = True
                elif mutation == "nan":
                    prediction["confidence"] = float("nan")
                elif mutation == "sum":
                    prediction["probabilities"]["draw"] = 0.5
                elif mutation == "confidence":
                    prediction["confidence"] = 0.7
                else:
                    prediction["prediction"] = "AWAY_WIN"

                release = build_release_governance(
                    predictions,
                    operations,
                    generated_at="2026-07-09T00:00:00+00:00",
                )

                self.assertIn(
                    "PREDICTION_PROBABILITIES_INVALID",
                    {item["code"] for item in release["blockers"]},
                )

    def _check_missing_cryptographic_prediction_identity_cannot_be_approved(self) -> None:
        predictions, operations = _valid_release_inputs()
        for item in predictions:
            item.pop("forecast_id")
            item.pop("model")

        release = build_release_governance(
            predictions, operations, generated_at="2026-07-09T00:00:00+00:00"
        )

        self.assertEqual(release["release_decision"], "blocked")
        self.assertIn(
            "FORECAST_IDENTITY_INCOMPLETE",
            {item["code"] for item in release["blockers"]},
        )

    def _check_prediction_model_must_match_competition_quality_evidence(self) -> None:
        predictions, operations = _valid_release_inputs()
        for item in predictions:
            item["model"]["model_sha256"] = "b" * 64
            item["forecast_id"] = forecast_id_for_prediction(item)

        release = build_release_governance(
            predictions, operations, generated_at="2026-07-09T00:00:00+00:00"
        )

        self.assertEqual(release["release_decision"], "blocked")
        self.assertIn(
            "MODEL_EVIDENCE_INCOMPLETE",
            {item["code"] for item in release["blockers"]},
        )

    def _check_release_recomputes_quality_summaries_and_thresholds(self) -> None:
        for mutation in ("reliability", "below_baseline"):
            with self.subTest(mutation=mutation):
                predictions, operations = _valid_release_inputs()
                evidence = operations["model_quality_by_competition"][0]
                if mutation == "reliability":
                    evidence["backtest_reliability"][1]["calibration_error"] = 0.95
                else:
                    evidence["validation_accuracy"] = 0.4

                release = build_release_governance(
                    predictions,
                    operations,
                    generated_at="2026-07-09T00:00:00+00:00",
                )

                self.assertEqual(release["release_decision"], "blocked")
                self.assertIn(
                    "MODEL_EVIDENCE_INCOMPLETE",
                    {item["code"] for item in release["blockers"]},
                )

    def _check_inconsistent_model_coverage_is_rejected(self) -> None:
        release = build_release_governance(
            [],
            {
                "schema": "SoccerAnalytics.Operations.v3",
                "status": "critical",
                "prediction_count": 0,
                "average_confidence": 0.0,
                "high_confidence_count": 0,
                "model_coverage": {"loaded": 2, "requested": 1, "ratio": 0.5},
                "model_quality_by_competition": [],
                "cold_start_ratio": 0.0,
                "alerts": [],
            },
        )

        self.assertIn(
            "MODEL_COVERAGE_INVALID",
            {item["code"] for item in release["blockers"]},
        )

    def _check_duplicate_forecasts_cannot_inflate_release_coverage(self) -> None:
        predictions = [
            {
                "forecast_id": "same",
                "match_id": 1,
                "competition": "PL",
                "confidence": 0.72,
            }
        ] * 3
        release = build_release_governance(
            predictions,
            {
                "schema": "SoccerAnalytics.Operations.v3",
                "status": "healthy",
                "average_confidence": 0.72,
                "high_confidence_count": 3,
                "model_coverage": {"loaded": 1, "requested": 1, "ratio": 1.0},
                "model_quality_by_competition": [],
                "cold_start_ratio": 0.0,
                "alerts": [],
            },
        )
        codes = {item["code"] for item in release["blockers"]}
        self.assertEqual(release["audit"]["unique_prediction_count"], 1)
        self.assertIn("DUPLICATE_FORECASTS", codes)
        self.assertIn("INSUFFICIENT_FIXTURE_COVERAGE", codes)

    def _check_same_match_with_different_forecast_ids_is_a_conflict(self) -> None:
        predictions = [
            {
                "forecast_id": forecast_id,
                "match_id": 1,
                "competition": "PL",
                "confidence": confidence,
            }
            for forecast_id, confidence in (("first", 0.72), ("second", 0.61))
        ]
        release = build_release_governance(
            predictions,
            {
                "schema": "SoccerAnalytics.Operations.v3",
                "status": "critical",
                "prediction_count": 1,
                "average_confidence": 0.72,
                "high_confidence_count": 1,
                "model_coverage": {"loaded": 0, "requested": 1, "ratio": 0.0},
                "cold_start_ratio": 0.0,
                "alerts": [],
            },
        )
        self.assertIn(
            "FORECAST_ID_CONFLICT",
            {item["code"] for item in release["blockers"]},
        )
    def _check_bare_competition_names_cannot_forge_complete_v3_evidence(self) -> None:
        release = build_release_governance(
            [{"competition": "PL", "confidence": 0.72}] * 3,
            {
                "status": "healthy",
                "average_confidence": 0.72,
                "high_confidence_count": 3,
                "model_coverage": {"loaded": 1, "requested": 1, "ratio": 1.0},
                "model_quality_by_competition": [{"competition": "PL"}],
                "cold_start_ratio": 0.0,
                "alerts": [],
            },
        )

        self.assertEqual(release["release_decision"], "blocked")
        self.assertIn(
            "MODEL_EVIDENCE_INCOMPLETE",
            {item["code"] for item in release["blockers"]},
        )

    def _check_missing_model_evidence_coverage_blocks_claimed_healthy_summary(self) -> None:
        release = build_release_governance(
            [{"competition": "PL", "confidence": 0.72}] * 3,
            {
                "status": "healthy",
                "average_confidence": 0.72,
                "high_confidence_count": 3,
                "model_coverage": {"loaded": 1, "requested": 1, "ratio": 1.0},
                "cold_start_ratio": 0.0,
                "models_below_baseline": [],
                "models_without_quality_metrics": [],
                "models_below_backtest_baseline": [],
                "models_with_poor_calibration": [],
                "models_recovered_from_previous": [],
                "models_stale": [],
                "alerts": [],
            },
        )

        self.assertEqual(release["release_decision"], "blocked")
        self.assertIn(
            "MODEL_EVIDENCE_INCOMPLETE",
            {item["code"] for item in release["blockers"]},
        )
    def _check_blocks_failed_backtest_poor_calibration_and_fallback_model(self) -> None:
        predictions = [{"competition": "PL", "confidence": 0.72}] * 3
        release = build_release_governance(
            predictions,
            {
                "status": "critical",
                "average_confidence": 0.72,
                "high_confidence_count": 3,
                "model_coverage": {"loaded": 1, "requested": 1, "ratio": 1.0},
                "cold_start_ratio": 0.0,
                "models_below_baseline": [],
                "models_without_quality_metrics": [],
                "models_below_backtest_baseline": ["PL"],
                "models_with_poor_calibration": ["PL"],
                "models_recovered_from_previous": ["PL"],
                "models_stale": ["PL"],
                "alerts": [],
            },
        )

        codes = {blocker["code"] for blocker in release["blockers"]}
        self.assertIn("MODEL_BACKTEST_BELOW_BASELINE", codes)
        self.assertIn("MODEL_CALIBRATION_TOO_HIGH", codes)
        self.assertIn("MODEL_FALLBACK_ACTIVE", codes)
        self.assertIn("MODEL_STALE", codes)

    def _check_approves_complete_quality_checked_release(self) -> None:
        predictions, operations = _valid_release_inputs()
        release = build_release_governance(
            predictions,
            operations,
            generated_at="2026-07-09T00:00:00+00:00",
        )
        self.assertEqual(release["schema"], "SoccerAnalytics.ReleaseGovernance.v3")
        self.assertEqual(release["release_decision"], "approved")
        self.assertTrue(release["can_publish_recommendations"])
        self.assertEqual(release["blockers"], [])

    def _check_blocks_incomplete_models_bad_validation_and_cold_start(self) -> None:
        predictions, operations = _valid_release_inputs()
        for prediction in predictions:
            prediction["data_quality"] = {
                "is_cold_start": True,
                "home_history_matches": 0,
                "away_history_matches": 5,
            }
            prediction["forecast_id"] = forecast_id_for_prediction(prediction)
        operations.update(
            {
                "status": "critical",
                "model_coverage": {"loaded": 1, "requested": 2, "ratio": 0.5},
                "cold_start_count": 3,
                "cold_start_ratio": 1.0,
                "models_below_baseline": ["PL"],
                "alerts": [
                    {"level": "critical", "code": "MODEL_COVERAGE_GAP", "message": "missing model"},
                    {"level": "critical", "code": "MODEL_BELOW_BASELINE", "message": "bad model"},
                ],
            }
        )
        release = build_release_governance(
            predictions,
            operations,
            generated_at="2026-07-09T00:00:00+00:00",
        )
        codes = {blocker["code"] for blocker in release["blockers"]}
        self.assertEqual(release["release_decision"], "blocked")
        self.assertIn("OPERATIONS_NOT_HEALTHY", codes)
        self.assertIn("MODEL_COVERAGE_INCOMPLETE", codes)
        self.assertIn("MODEL_QUALITY_BELOW_BASELINE", codes)
        self.assertIn("EXCESSIVE_COLD_START", codes)
        self.assertIn("CRITICAL_ALERT_MODEL_COVERAGE_GAP", codes)

    def _check_blocks_missing_or_non_finite_quality_evidence(self) -> None:
        predictions = [{"competition": "PL", "confidence": 0.72}] * 3
        release = build_release_governance(
            predictions,
            {
                "status": "healthy",
                "average_confidence": math.nan,
                "high_confidence_count": 3,
                "model_coverage": {"loaded": 1, "requested": 1, "ratio": 1.0},
                "cold_start_ratio": 0.0,
                "models_below_baseline": [],
                "models_without_quality_metrics": ["PL"],
                "alerts": [],
            },
        )
        codes = {blocker["code"] for blocker in release["blockers"]}
        self.assertEqual(release["release_decision"], "blocked")
        self.assertIn("OPERATIONS_PREDICTION_AUDIT_MISMATCH", codes)
        self.assertIn("MODEL_QUALITY_UNAVAILABLE", codes)
