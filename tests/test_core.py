from __future__ import annotations

import os
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from app.config import (
    DEFAULT_COMPETITIONS_MAP,
    DEFAULT_SEASONS,
    PipelineSettings,
    ConfigurationError,
    default_training_seasons,
    load_settings,
    resolve_competitions,
)


class TestConfig(unittest.TestCase):
    def _run_contracts(self, names: tuple[str, ...]) -> None:
        for name in names:
            with self.subTest(contract=name):
                getattr(self, name)()

    def test_settings_loading_contracts(self):
        self._run_contracts(
            (
                "_check_load_settings_uses_env_overrides",
                "_check_defaults_remain_available_without_env",
                "_check_default_training_seasons_tracks_latest_usable_european_season",
                "_check_pipeline_settings_requires_explicit_field_names",
            )
        )

    def test_competition_and_season_selection_contracts(self):
        self._run_contracts(
            (
                "_check_resolve_competitions_rejects_unknown_explicit_code",
                "_check_training_seasons_are_validated_and_deduplicated",
                "_check_invalid_explicit_competition_env_is_rejected",
                "_check_delimiter_only_competition_value_uses_defaults",
            )
        )

    def test_numeric_setting_contracts(self):
        self._check_non_positive_or_malformed_numeric_settings_are_rejected()

    def _check_load_settings_uses_env_overrides(self):
        env = {
            "SOCCER_ANALYTICS_COMPETITIONS": "PL,SA",
            "SOCCER_ANALYTICS_TRAINING_SEASONS": "2022,2024",
            "SOCCER_ANALYTICS_PREDICTION_DAYS": "5",
            "SOCCER_ANALYTICS_SITE_EXPORT_DAYS": "2",
            "SOCCER_ANALYTICS_REQUEST_TIMEOUT": "7.5",
            "SOCCER_ANALYTICS_REQUEST_RETRIES": "4",
            "SOCCER_ANALYTICS_MODEL_DIR": "/tmp/soccer-models",
            "SOCCER_ANALYTICS_SITE_DATA_DIR": "/tmp/soccer-site-data",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = load_settings()

        self.assertEqual(settings.competitions_map, {"PL": 2021, "SA": 2019})
        self.assertEqual(settings.training_seasons, ["2022", "2024"])
        self.assertEqual(settings.prediction_days, 5)
        self.assertEqual(settings.site_export_days, 2)
        self.assertEqual(settings.request_timeout_seconds, 7.5)
        self.assertEqual(settings.request_retries, 4)
        self.assertEqual(settings.model_dir, Path("/tmp/soccer-models"))
        self.assertEqual(settings.site_data_dir, Path("/tmp/soccer-site-data"))

    def _check_resolve_competitions_rejects_unknown_explicit_code(self):
        settings = load_settings()
        with self.assertRaises(ValueError) as context:
            resolve_competitions("UNKNOWN", settings)

        self.assertIn("UNKNOWN", str(context.exception))

    def _check_defaults_remain_available_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings()

        self.assertEqual(settings.competitions_map, DEFAULT_COMPETITIONS_MAP)
        self.assertEqual(settings.training_seasons, DEFAULT_SEASONS)
        self.assertEqual(settings.request_timeout_seconds, 15.0)
        self.assertEqual(settings.request_retries, 3)
        self.assertEqual(settings.model_dir, Path("models"))
        self.assertEqual(settings.site_data_dir, Path("docs/data"))

    def _check_default_training_seasons_tracks_latest_usable_european_season(self):
        self.assertEqual(
            default_training_seasons(date(2026, 8, 10)),
            ["2023", "2024", "2025", "2026"],
        )
        self.assertEqual(
            default_training_seasons(date(2026, 2, 1)),
            ["2022", "2023", "2024", "2025"],
        )

    def _check_training_seasons_are_validated_and_deduplicated(self):
        with patch.dict(
            os.environ,
            {"SOCCER_ANALYTICS_TRAINING_SEASONS": "2022,2024,2022"},
            clear=True,
        ):
            self.assertEqual(load_settings().training_seasons, ["2022", "2024"])

        with patch.dict(
            os.environ,
            {"SOCCER_ANALYTICS_TRAINING_SEASONS": "2022,season-23"},
            clear=True,
        ):
            with self.assertRaises(ConfigurationError) as context:
                load_settings()
        self.assertIn("season-23", str(context.exception))

    def _check_invalid_explicit_competition_env_is_rejected(self):
        with patch.dict(
            os.environ,
            {"SOCCER_ANALYTICS_COMPETITIONS": "PL,TYPO"},
            clear=True,
        ):
            with self.assertRaises(ConfigurationError) as context:
                load_settings()
        self.assertIn("TYPO", str(context.exception))

    def _check_pipeline_settings_requires_explicit_field_names(self):
        with self.assertRaises(TypeError):
            PipelineSettings({"PL": 2021}, ["2024"], 3, 1)
        settings = PipelineSettings(
            competitions_map={"PL": 2021},
            training_seasons=["2024"],
            prediction_days=3,
            site_export_days=1,
        )
        self.assertEqual(settings.request_timeout_seconds, 15.0)
        self.assertEqual(settings.request_retries, 3)
        self.assertEqual(settings.model_dir, Path("models"))
        self.assertEqual(settings.site_data_dir, Path("docs/data"))

    def _check_delimiter_only_competition_value_uses_defaults(self):
        with patch.dict(
            os.environ,
            {"SOCCER_ANALYTICS_COMPETITIONS": ", ,"},
            clear=True,
        ):
            settings = load_settings()
        self.assertEqual(settings.competitions_map, DEFAULT_COMPETITIONS_MAP)
        self.assertEqual(resolve_competitions(",,", settings), DEFAULT_COMPETITIONS_MAP)

    def _check_non_positive_or_malformed_numeric_settings_are_rejected(self):
        invalid_values = {
            "SOCCER_ANALYTICS_PREDICTION_DAYS": "0",
            "SOCCER_ANALYTICS_SITE_EXPORT_DAYS": "-1",
            "SOCCER_ANALYTICS_REQUEST_TIMEOUT": "soon",
            "SOCCER_ANALYTICS_REQUEST_RETRIES": "1.5",
        }
        for key, value in invalid_values.items():
            with self.subTest(key=key, value=value):
                with patch.dict(os.environ, {key: value}, clear=True):
                    with self.assertRaises(ConfigurationError) as context:
                        load_settings()
                self.assertIn(key, str(context.exception))


import unittest

from app.ml.evaluation import (
    EvaluationError,
    classwise_reliability,
    evaluate_expanding_window,
    top_label_calibration_error,
)


class _BoundaryEstimator:
    def __init__(self, boundaries: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> None:
        self.boundaries = boundaries
        self.train_ids: tuple[int, ...] = ()

    def fit(self, rows, targets) -> None:
        self.train_ids = tuple(int(row["sequence"]) for row in rows)

    def predict_proba(self, rows):
        validation_ids = tuple(int(row["sequence"]) for row in rows)
        self.boundaries.append((self.train_ids, validation_ids))
        return [[0.6, 0.2, 0.2] for _ in rows]

    def predict(self, rows):
        return [0 for _ in rows]


class _BoundaryCalibrationEstimator:
    def __init__(self, feature_names) -> None:
        self.feature_names = tuple(feature_names)

    def fit(self, rows, targets) -> None:
        return None

    def predict_proba(self, rows):
        return [[0.7999996, 0.1000002, 0.1000002] for _ in rows]

    def predict(self, rows):
        return [0 for _ in rows]


class TestTemporalEvaluation(unittest.TestCase):
    def _run_contracts(self, names: tuple[str, ...]) -> None:
        for name in names:
            with self.subTest(contract=name):
                getattr(self, name)()

    def test_metric_and_calibration_contracts(self) -> None:
        self._run_contracts(
            (
                "_check_fold_metrics_keep_raw_precision_for_stability_gate",
                "_check_classwise_reliability_exposes_sampled_probability_bins",
                "_check_aggregate_metrics_keep_raw_precision_for_release_gates",
                "_check_top_label_calibration_error_has_hand_checked_value",
            )
        )

    def test_expanding_window_contracts(self) -> None:
        self._run_contracts(
            (
                "_check_domain_baseline_uses_only_prior_elo_evidence",
                "_check_expanding_window_never_trains_on_future_rows",
                "_check_evaluation_rejects_too_few_rows_for_requested_folds",
            )
        )

    def _check_fold_metrics_keep_raw_precision_for_stability_gate(self) -> None:
        result = evaluate_expanding_window(
            [{"sequence": float(index)} for index in range(18)],
            [0, 1, 2] * 6,
            ("sequence",),
            lambda _: _BoundaryEstimator([]),
            folds=3,
        )

        self.assertEqual(result.fold_metrics[0]["accuracy"], 1 / 3)
        self.assertNotEqual(
            result.fold_metrics[0]["accuracy"],
            round(result.fold_metrics[0]["accuracy"], 6),
        )

    def _check_classwise_reliability_exposes_sampled_probability_bins(self) -> None:
        reliability = classwise_reliability(
            [0, 1],
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]],
            bins=2,
        )

        self.assertEqual([item["class_index"] for item in reliability], [0, 1, 2])
        self.assertAlmostEqual(reliability[0]["calibration_error"], 0.2)
        self.assertEqual(sum(item["count"] for item in reliability[0]["bins"]), 2)
        self.assertAlmostEqual(reliability[2]["calibration_error"], 0.1)

    def _check_aggregate_metrics_keep_raw_precision_for_release_gates(self) -> None:
        result = evaluate_expanding_window(
            [{"sequence": float(index)} for index in range(6)],
            [0] * 6,
            ("sequence",),
            _BoundaryCalibrationEstimator,
            folds=2,
        )

        self.assertGreater(result.calibration_error, 0.2)
        self.assertAlmostEqual(result.calibration_error, 0.2000004, places=7)

    def _check_top_label_calibration_error_has_hand_checked_value(self) -> None:
        error = top_label_calibration_error(
            [0, 1],
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]],
            bins=2,
        )
        self.assertAlmostEqual(error, 0.25)

    def _check_domain_baseline_uses_only_prior_elo_evidence(self) -> None:
        rows = [
            {"elo_diff": elo, "sequence": float(index)}
            for index, elo in enumerate(
                elo for _ in range(6) for elo in (-200.0, 0.0, 200.0)
            )
        ]
        targets = [outcome for _ in range(6) for outcome in (0, 1, 2)]

        result = evaluate_expanding_window(
            rows,
            targets,
            ("elo_diff", "sequence"),
            lambda _: _BoundaryEstimator([]),
            folds=3,
        )

        self.assertEqual(result.majority_baseline_accuracy, 1 / 3)
        self.assertEqual(result.elo_baseline_accuracy, 1.0)
        self.assertEqual(result.baseline_accuracy, 1.0)
        self.assertTrue(
            all(item["elo_baseline_accuracy"] == 1.0 for item in result.fold_metrics)
        )

    def _check_expanding_window_never_trains_on_future_rows(self) -> None:
        boundaries: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        rows = [{"sequence": float(index)} for index in range(12)]
        targets = [index % 3 for index in range(12)]

        result = evaluate_expanding_window(
            rows,
            targets,
            ("sequence",),
            lambda _: _BoundaryEstimator(boundaries),
            folds=3,
        )

        self.assertEqual(
            boundaries,
            [
                (tuple(range(6)), (6, 7)),
                (tuple(range(8)), (8, 9)),
                (tuple(range(10)), (10, 11)),
            ],
        )
        self.assertEqual(result.fold_count, 3)
        self.assertEqual(result.evaluated_rows, 6)
        self.assertGreaterEqual(result.ranked_probability_score, 0.0)
        self.assertGreaterEqual(result.classwise_calibration_error, 0.0)
        self.assertGreaterEqual(result.sharpness, 0.0)
        self.assertEqual(len(result.reliability), 3)
        self.assertEqual(
            [item["training_rows"] for item in result.fold_metrics],
            [6, 8, 10],
        )
        self.assertEqual(
            [item["validation_rows"] for item in result.fold_metrics],
            [2, 2, 2],
        )

    def _check_evaluation_rejects_too_few_rows_for_requested_folds(self) -> None:
        with self.assertRaises(EvaluationError):
            evaluate_expanding_window(
                [{"sequence": 0.0}, {"sequence": 1.0}],
                [0, 1],
                ("sequence",),
                lambda _: _BoundaryEstimator([]),
                folds=3,
            )


import json
import math
import pickle
import unittest
from datetime import UTC, datetime

from app.ml.feature_engineering import FeatureEngineer, FeatureEngineeringError


def api_match(
    match_id,
    kickoff,
    home_id,
    away_id,
    home_score,
    away_score,
    *,
    status="FINISHED",
):
    return {
        "id": match_id,
        "utcDate": kickoff,
        "status": status,
        "homeTeam": {"id": home_id, "name": f"Team {home_id}"},
        "awayTeam": {"id": away_id, "name": f"Team {away_id}"},
        "score": {
            "winner": (
                "HOME_TEAM"
                if home_score is not None and home_score > away_score
                else "AWAY_TEAM"
                if home_score is not None and home_score < away_score
                else "DRAW"
            ),
            "fullTime": {"home": home_score, "away": away_score},
        },
    }


class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.engineer = FeatureEngineer(window=5)

    def _run_contracts(self, names: tuple[str, ...]) -> None:
        for name in names:
            with self.subTest(contract=name):
                getattr(self, name)()

    def test_temporal_feature_contracts(self):
        self._run_contracts(
            (
                "_check_one_row_per_match_and_home_perspective_targets",
                "_check_features_are_created_before_current_result_updates_state",
                "_check_elo_updates_are_zero_sum_and_serialized",
                "_check_same_kickoff_bucket_cannot_see_another_result",
            )
        )

    def test_fixture_and_state_contracts(self):
        self._run_contracts(
            (
                "_check_fixture_features_use_saved_state_and_cold_start_defaults",
                "_check_serialized_states_round_trip_and_corruption_is_domain_error",
            )
        )

    def test_invalid_match_contracts(self):
        self._run_contracts(
            (
                "_check_identical_duplicates_are_deduplicated_but_conflicts_fail",
                "_check_incomplete_finished_match_is_rejected",
            )
        )

    def _check_one_row_per_match_and_home_perspective_targets(self):
        matches = [
            api_match(1, "2026-01-01T12:00:00Z", 10, 20, 2, 0),
            api_match(2, "2026-01-02T12:00:00Z", 30, 40, 1, 1),
            api_match(3, "2026-01-03T12:00:00Z", 50, 60, 0, 3),
        ]

        dataset = self.engineer.build_training_dataset(matches)

        self.assertEqual(dataset.match_ids, (1, 2, 3))
        self.assertEqual(dataset.targets, (2, 1, 0))
        self.assertEqual(len(dataset.rows), 3)
        self.assertEqual(tuple(dataset.rows[0]), self.engineer.features)

    def _check_features_are_created_before_current_result_updates_state(self):
        matches = [
            api_match(2, "2026-01-08T12:00:00Z", 10, 30, 0, 1),
            api_match(1, "2026-01-01T12:00:00Z", 10, 20, 2, 0),
        ]

        dataset = self.engineer.build_training_dataset(matches)
        first, second = dataset.rows

        self.assertEqual(dataset.match_ids, (1, 2))
        self.assertEqual(first["home_matches_seen"], 0.0)
        self.assertEqual(first["home_form_points"], 1.0)
        self.assertEqual(second["home_matches_seen"], 1.0)
        self.assertEqual(second["home_form_points"], 3.0)
        self.assertGreater(second["home_elo"], 1500.0)

    def _check_elo_updates_are_zero_sum_and_serialized(self):
        dataset = self.engineer.build_training_dataset(
            [api_match(1, "2026-01-01T12:00:00Z", 10, 20, 3, 1)]
        )

        home = dataset.team_states["10"]
        away = dataset.team_states["20"]
        self.assertAlmostEqual(home["elo"] + away["elo"], 3000.0, places=10)
        self.assertGreater(home["elo"], 1500.0)
        self.assertEqual(home["matches_played"], 1)
        self.assertEqual(home["last_match_at"], "2026-01-01T12:00:00+00:00")

    def _check_fixture_features_use_saved_state_and_cold_start_defaults(self):
        dataset = self.engineer.build_training_dataset(
            [api_match(1, "2026-01-01T12:00:00Z", 10, 20, 2, 0)]
        )

        features = self.engineer.build_fixture_features(
            10,
            999,
            datetime(2026, 1, 8, 12, tzinfo=UTC),
            dataset.team_states,
        )

        self.assertEqual(tuple(features), self.engineer.features)
        self.assertEqual(features["home_matches_seen"], 1.0)
        self.assertEqual(features["away_matches_seen"], 0.0)
        self.assertEqual(features["away_elo"], 1500.0)
        self.assertEqual(features["home_rest_days"], 7.0)
        self.assertTrue(all(math.isfinite(value) for value in features.values()))

    def _check_identical_duplicates_are_deduplicated_but_conflicts_fail(self):
        original = api_match(1, "2026-01-01T12:00:00Z", 10, 20, 2, 0)
        duplicate = dict(original)
        self.assertEqual(
            len(self.engineer.build_training_dataset([original, duplicate]).rows),
            1,
        )

        conflict = api_match(1, "2026-01-01T12:00:00Z", 10, 20, 0, 2)
        with self.assertRaises(FeatureEngineeringError) as context:
            self.engineer.build_training_dataset([original, conflict])
        self.assertIn("重复", str(context.exception))

    def _check_same_kickoff_bucket_cannot_see_another_result(self):
        dataset = self.engineer.build_training_dataset(
            [
                api_match(1, "2026-01-01T12:00:00Z", 10, 20, 3, 0),
                api_match(2, "2026-01-01T12:00:00Z", 10, 30, 0, 1),
            ]
        )
        self.assertEqual(dataset.rows[0]["home_matches_seen"], 0.0)
        self.assertEqual(dataset.rows[1]["home_matches_seen"], 0.0)
        self.assertEqual(dataset.rows[1]["home_form_points"], 1.0)
        self.assertEqual(dataset.rows[1]["home_elo"], 1500.0)

    def _check_serialized_states_round_trip_and_corruption_is_domain_error(self):
        dataset = self.engineer.build_training_dataset(
            [api_match(1, "2026-01-01T12:00:00Z", 10, 20, 2, 0)]
        )
        json_states = json.loads(json.dumps(dataset.team_states))
        pickle_states = pickle.loads(pickle.dumps(dataset.team_states))
        kickoff = datetime(2026, 1, 8, 12, tzinfo=UTC)
        self.assertEqual(
            self.engineer.build_fixture_features(10, 20, kickoff, json_states),
            self.engineer.build_fixture_features(10, 20, kickoff, pickle_states),
        )

        corrupt_states = {"10": {"elo": float("nan"), "matches_played": 1, "recent": []}}
        with self.assertRaises(FeatureEngineeringError):
            self.engineer.build_fixture_features(10, 20, kickoff, corrupt_states)
        with self.assertRaises(FeatureEngineeringError):
            self.engineer.build_fixture_features(10, 20, kickoff, {"bad-id": {}})
        with self.assertRaises(FeatureEngineeringError):
            self.engineer.build_fixture_features(
                10,
                20,
                kickoff,
                {"10": {"elo": 1500.0, "matches_played": 1, "recent": [{"points": 3.0}]}},
            )

    def _check_incomplete_finished_match_is_rejected(self):
        incomplete = api_match(1, "2026-01-01T12:00:00Z", 10, 20, None, None)
        with self.assertRaises(FeatureEngineeringError) as context:
            self.engineer.build_training_dataset([incomplete])
        self.assertIn("比分", str(context.exception))


import unittest

import requests

from app.data_service.fetch.fetcher import DataSourceError, FootballDataClient


class FakeResponse:
    def __init__(self, status_code, payload=None, *, headers=None, json_error=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise self._json_error
        return self._payload


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class NoopLimiter:
    def __init__(self):
        self.before_count = 0
        self.after_count = 0

    def wait_if_needed(self):
        self.before_count += 1

    def add_request(self):
        self.after_count += 1


class TestFootballDataClient(unittest.TestCase):
    def make_client(self, outcomes, **overrides):
        session = FakeSession(outcomes)
        limiter = NoopLimiter()
        sleeps = []
        options = {
            "api_key": "secret-token",
            "session": session,
            "limiter": limiter,
            "timeout": 8.5,
            "max_retries": 2,
            "backoff_factor": 0.25,
            "sleep": sleeps.append,
        }
        options.update(overrides)
        return FootballDataClient(**options), session, limiter, sleeps

    def _run_contracts(self, names: tuple[str, ...]) -> None:
        for name in names:
            with self.subTest(contract=name):
                getattr(self, name)()

    def test_request_and_retry_contracts(self):
        self._run_contracts(
            (
                "_check_successful_request_uses_auth_timeout_and_normalized_url",
                "_check_missing_api_key_fails_without_network_call",
                "_check_retryable_status_uses_retry_after_then_exponential_backoff",
                "_check_network_errors_stop_after_configured_retry_budget",
                "_check_retryable_http_failure_stops_after_budget",
                "_check_invalid_retry_after_falls_back_to_exponential_delay",
                "_check_non_retryable_4xx_is_attempted_once",
                "_check_invalid_json_shape_is_rejected",
            )
        )

    def test_match_endpoint_contracts(self):
        self._run_contracts(
            (
                "_check_fetch_matches_forwards_season_and_status",
                "_check_malformed_nested_matches_is_not_silently_treated_as_empty",
                "_check_scheduled_matches_builds_complete_query",
            )
        )

    def test_client_surface_has_no_obsolete_helpers(self):
        self._check_obsolete_client_helpers_are_not_public()

    def _check_successful_request_uses_auth_timeout_and_normalized_url(self):
        client, session, limiter, sleeps = self.make_client(
            [FakeResponse(200, {"matches": []})]
        )

        result = client.get("/competitions/PL/matches", {"status": "SCHEDULED"})

        self.assertEqual(result, {"matches": []})
        self.assertEqual(session.calls[0][0], "https://api.football-data.org/v4/competitions/PL/matches")
        self.assertEqual(session.calls[0][1]["headers"], {"X-Auth-Token": "secret-token"})
        self.assertEqual(session.calls[0][1]["timeout"], 8.5)
        self.assertEqual(session.calls[0][1]["params"], {"status": "SCHEDULED"})
        self.assertEqual((limiter.before_count, limiter.after_count), (1, 1))
        self.assertEqual(sleeps, [])

    def _check_missing_api_key_fails_without_network_call(self):
        client, session, _, _ = self.make_client([], api_key=None)

        with self.assertRaises(DataSourceError) as context:
            client.get("competitions/PL/matches")

        self.assertFalse(context.exception.retryable)
        self.assertEqual(context.exception.endpoint, "competitions/PL/matches")
        self.assertEqual(session.calls, [])

    def _check_retryable_status_uses_retry_after_then_exponential_backoff(self):
        client, session, _, sleeps = self.make_client(
            [
                FakeResponse(429, {"error": "slow down"}, headers={"Retry-After": "1.5"}),
                FakeResponse(503, {"error": "unavailable"}),
                FakeResponse(200, {"matches": [1]}),
            ]
        )

        self.assertEqual(client.get("competitions/PL/matches"), {"matches": [1]})
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(sleeps, [1.5, 0.5])

    def _check_network_errors_stop_after_configured_retry_budget(self):
        client, session, _, sleeps = self.make_client(
            [
                requests.ConnectionError("offline-1"),
                requests.ConnectionError("offline-2"),
                requests.ConnectionError("offline-3"),
            ]
        )

        with self.assertRaises(DataSourceError) as context:
            client.get("competitions/PL/matches")

        self.assertTrue(context.exception.retryable)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(sleeps, [0.25, 0.5])

    def _check_retryable_http_failure_stops_after_budget(self):
        client, session, _, sleeps = self.make_client(
            [FakeResponse(503, {}), FakeResponse(503, {}), FakeResponse(503, {})]
        )
        with self.assertRaises(DataSourceError) as context:
            client.get("competitions/PL/matches")
        self.assertTrue(context.exception.retryable)
        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(sleeps, [0.25, 0.5])

    def _check_invalid_retry_after_falls_back_to_exponential_delay(self):
        client, _, _, sleeps = self.make_client(
            [
                FakeResponse(429, {}, headers={"Retry-After": "-10"}),
                FakeResponse(200, {"matches": []}),
            ]
        )
        self.assertEqual(client.get("competitions/PL/matches"), {"matches": []})
        self.assertEqual(sleeps, [0.25])

    def _check_fetch_matches_forwards_season_and_status(self):
        client, session, _, _ = self.make_client([FakeResponse(200, {"matches": [{"id": 9}]})])
        self.assertEqual(
            client.fetch_matches("pl", season="2025", status="FINISHED"),
            [{"id": 9}],
        )
        self.assertEqual(session.calls[0][1]["params"], {"season": "2025", "status": "FINISHED"})

    def _check_malformed_nested_matches_is_not_silently_treated_as_empty(self):
        for payload in ({}, {"matches": {"id": 9}}, {"matches": ["bad-row"]}):
            with self.subTest(payload=payload):
                client, _, _, _ = self.make_client([FakeResponse(200, payload)])
                with self.assertRaises(DataSourceError) as context:
                    client.fetch_matches("PL", season="2025", status="FINISHED")
                self.assertIn("matches", str(context.exception))
                self.assertFalse(context.exception.retryable)

    def _check_non_retryable_4xx_is_attempted_once(self):
        client, session, _, sleeps = self.make_client(
            [FakeResponse(403, {"message": "forbidden"})]
        )

        with self.assertRaises(DataSourceError) as context:
            client.get("competitions/PL/matches")

        self.assertEqual(context.exception.status_code, 403)
        self.assertFalse(context.exception.retryable)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(sleeps, [])

    def _check_invalid_json_shape_is_rejected(self):
        client, _, _, _ = self.make_client([FakeResponse(200, ["not", "an", "object"])])

        with self.assertRaises(DataSourceError) as context:
            client.get("competitions/PL/matches")

        self.assertFalse(context.exception.retryable)

    def _check_scheduled_matches_builds_complete_query(self):
        matches = [{"id": index, "status": status} for index, status in enumerate(
            ("SCHEDULED", "TIMED", "FINISHED", "IN_PLAY", "POSTPONED", "CANCELLED")
        )]
        client, session, _, _ = self.make_client([FakeResponse(200, {"matches": matches})])

        result = client.fetch_scheduled_matches("pl", "2026-08-10", "2026-08-12")

        self.assertEqual(result, matches[:2])
        self.assertEqual(session.calls[0][1]["params"], {
            "dateFrom": "2026-08-10",
            "dateTo": "2026-08-12",
        })
        self.assertTrue(session.calls[0][0].endswith("competitions/PL/matches"))

    def _check_obsolete_client_helpers_are_not_public(self):
        client, _, _, _ = self.make_client([])
        for name in (
            "_get",
            "fetch_multiple_seasons",
            "fetch_competition_details",
            "fetch_team_squad",
            "fetch_standings",
            "fetch_top_scorers",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(client, name))


import math
import pickle
import unittest

from app.ml.model import (
    ModelTrainingError,
    XGBoostClassifier,
    accuracy_score,
    multiclass_brier_score,
    multiclass_log_loss,
    ranked_probability_score,
)


class TestXGBoostClassifier(unittest.TestCase):
    def _run_contracts(self, names: tuple[str, ...]) -> None:
        for name in names:
            with self.subTest(contract=name):
                getattr(self, name)()

    def test_fit_and_explanation_contracts(self):
        self._run_contracts(
            (
                "_check_tree_shap_exactly_decomposes_winner_margin",
                "_check_fit_learns_ordered_three_class_probabilities",
            )
        )

    def test_determinism_and_numerical_stability_contracts(self):
        self._run_contracts(
            (
                "_check_training_is_deterministic_and_pickle_round_trips",
                "_check_extreme_finite_features_never_produce_non_finite_model",
            )
        )

    def test_input_and_metric_contracts(self):
        self._run_contracts(
            (
                "_check_fit_rejects_missing_or_non_finite_features_and_bad_targets",
                "_check_invalid_metric_inputs_are_rejected",
                "_check_metric_functions_have_hand_checked_values",
            )
        )

    def _check_tree_shap_exactly_decomposes_winner_margin(self):
        rows, targets = self.training_data()
        model = XGBoostClassifier(("strength", "home"), rounds=80).fit(rows, targets)
        probe = {"strength": 25.0, "home": 1.0}
        explanation = model.explain_prediction(probe, top_n=2)

        self.assertEqual(model.technology, "xgboost")
        self.assertEqual(model.explanation_method, "tree_shap_margin")
        self.assertEqual(explanation["predicted_class_index"], 2)
        self.assertEqual(explanation["method"], "tree_shap_margin")
        self.assertAlmostEqual(
            explanation["bias_contribution"]
            + sum(item["contribution"] for item in explanation["drivers"]),
            explanation["logit_margin"],
            places=5,
        )
        self.assertEqual(explanation["top_drivers"][0]["feature"], "strength")
        self.assertLessEqual(model.rounds_used_, 80)

    def training_data(self):
        rows = []
        targets = []
        for value in range(-30, 31):
            rows.append({"strength": float(value), "home": 1.0})
            targets.append(0 if value < -5 else 1 if value <= 5 else 2)
        return rows, targets

    def _check_fit_learns_ordered_three_class_probabilities(self):
        rows, targets = self.training_data()
        model = XGBoostClassifier(("strength", "home"), rounds=80)

        uniform_loss = -math.log(1 / 3)
        model.fit(rows, targets)
        probabilities = model.predict_proba(
            [
                {"strength": -25.0, "home": 1.0},
                {"strength": 0.0, "home": 1.0},
                {"strength": 25.0, "home": 1.0},
            ]
        )

        self.assertEqual(model.predict([
            {"strength": -25.0, "home": 1.0},
            {"strength": 0.0, "home": 1.0},
            {"strength": 25.0, "home": 1.0},
        ]), [0, 1, 2])
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-6 for row in probabilities))
        self.assertTrue(all(0.0 <= value <= 1.0 for row in probabilities for value in row))
        self.assertLess(model.training_loss_, uniform_loss)

    def _check_training_is_deterministic_and_pickle_round_trips(self):
        rows, targets = self.training_data()
        first = XGBoostClassifier(("strength", "home"), rounds=60).fit(rows, targets)
        second = XGBoostClassifier(("strength", "home"), rounds=60).fit(rows, targets)

        probe = [{"strength": 12.0, "home": 1.0}]
        self.assertEqual(first.predict_proba(probe), second.predict_proba(probe))
        loaded = pickle.loads(pickle.dumps(first))
        self.assertEqual(first.predict_proba(probe), loaded.predict_proba(probe))

    def _check_fit_rejects_missing_or_non_finite_features_and_bad_targets(self):
        for kwargs in (
            {"rounds": "many"},
            {"learning_rate": "fast"},
            {"seed": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ModelTrainingError):
                    XGBoostClassifier(("strength", "home"), **kwargs)
        model = XGBoostClassifier(("strength", "home"), rounds=20)
        invalid_cases = [
            ([{"strength": 1.0}], [2]),
            ([{"strength": math.inf, "home": 1.0}], [2]),
            ([{"strength": 1.0, "home": 1.0}], [9]),
            ([], []),
        ]
        for rows, targets in invalid_cases:
            with self.subTest(rows=rows, targets=targets):
                with self.assertRaises(ModelTrainingError):
                    model.fit(rows, targets)

    def _check_extreme_finite_features_never_produce_non_finite_model(self):
        model = XGBoostClassifier(("x",), rounds=20)
        model.fit([{"x": -1e30}, {"x": 0.0}, {"x": 1e30}], [0, 1, 2])
        probabilities = model.predict_proba([{"x": 1e30}])[0]
        self.assertTrue(math.isfinite(model.training_loss_))
        self.assertTrue(all(math.isfinite(value) for value in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=6)

    def _check_invalid_metric_inputs_are_rejected(self):
        invalid_probability_rows = [
            [[0.5, 0.5, 0.5]],
            [[float("nan"), 0.0, 1.0]],
            [[-0.1, 0.1, 1.0]],
            [[True, 0.0, 0.0]],
        ]
        for probabilities in invalid_probability_rows:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ModelTrainingError):
                    multiclass_log_loss([0], probabilities)
                with self.assertRaises(ModelTrainingError):
                    multiclass_brier_score([0], probabilities)
        with self.assertRaises(ModelTrainingError):
            accuracy_score([0], [7])
        with self.assertRaises(ModelTrainingError):
            accuracy_score([True], [0])

    def _check_metric_functions_have_hand_checked_values(self):
        targets = [0, 2]
        probabilities = [[0.8, 0.1, 0.1], [0.2, 0.2, 0.6]]

        self.assertEqual(accuracy_score(targets, [0, 2]), 1.0)
        self.assertAlmostEqual(
            multiclass_log_loss(targets, probabilities),
            (-math.log(0.8) - math.log(0.6)) / 2,
            places=12,
        )
        self.assertAlmostEqual(
            multiclass_brier_score(targets, probabilities),
            ((0.2**2 + 0.1**2 + 0.1**2) + (0.2**2 + 0.2**2 + 0.4**2)) / 2,
            places=12,
        )
        self.assertAlmostEqual(
            ranked_probability_score(targets, probabilities),
            0.0625,
            places=12,
        )
