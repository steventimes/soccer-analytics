from __future__ import annotations


import pickle
import hashlib
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ml import artifacts
from app.ml.artifacts import (
    MODEL_SCHEMA_VERSION,
    ModelArtifactError,
    ModelBundle,
    load_model_bundle,
    load_model_bundle_with_fallback,
    previous_model_path,
    save_model_bundle,
    validate_model_bundle,
)
from app.ml.model import XGBoostClassifier


class ExplodingEstimator:
    def predict(self, rows):
        return []

    def predict_proba(self, rows):
        return []

    def __reduce__(self):
        raise RuntimeError("serialization exploded")


class FakeEstimator:
    def predict(self, rows):
        return [2 for _ in rows]

    def predict_proba(self, rows):
        return [[0.2, 0.3, 0.5] for _ in rows]


def _explode_while_unpickling():
    raise RuntimeError("unpickle hook exploded")


class ExplodingUnpicklePayload:
    def __reduce__(self):
        return (_explode_while_unpickling, ())


def valid_bundle(**overrides):
    values = {
        "competition_code": "PL",
        "competition_id": 2021,
        "estimator": FakeEstimator(),
        "feature_names": ("elo_diff", "form_points"),
        "classes": ("L", "D", "W"),
        "trained_at": "2026-08-10T07:00:00+00:00",
        "metrics": {"accuracy": 0.55, "rows": 120},
        "team_states": {
            "57": {
                "elo": 1532.5,
                "matches_played": 12,
                "last_match_at": "2026-08-08T15:00:00+00:00",
                "recent": [],
            }
        },
        "feature_defaults": {"elo_diff": 0.0, "form_points": 1.2},
        "training_rows": 120,
    }
    values.update(overrides)
    return ModelBundle(**values)


class TestModelArtifacts(unittest.TestCase):
    def _run_contracts(self, names: tuple[str, ...]) -> None:
        for name in names:
            with self.subTest(contract=name):
                getattr(self, name)()

    def test_fallback_and_versioning_contracts(self):
        self._run_contracts(
            (
                "_check_team_state_after_training_time_is_rejected_and_recovers",
                "_check_invalid_team_state_in_active_bundle_recovers_from_previous",
                "_check_inference_invalid_active_xgboost_recovers_from_previous",
                "_check_second_save_keeps_first_valid_bundle_as_previous_version",
                "_check_corrupt_active_bundle_recovers_explicitly_from_previous_version",
                "_check_semantically_malformed_active_bundle_also_recovers_from_previous",
                "_check_invalid_default_keys_in_active_bundle_trigger_previous_fallback",
                "_check_runtime_error_during_unpickle_triggers_previous_fallback",
            )
        )

    def test_artifact_storage_contracts(self):
        self._run_contracts(
            (
                "_check_round_trip_preserves_self_describing_bundle",
                "_check_save_atomically_replaces_existing_file_without_temp_residue",
                "_check_save_aligns_private_artifact_owner_with_model_directory",
                "_check_failed_serialization_removes_temporary_file",
            )
        )

    def test_artifact_validation_contracts(self):
        self._run_contracts(
            (
                "_check_validation_reports_all_independent_contract_failures",
                "_check_load_wraps_missing_fields_from_older_bundle",
                "_check_load_rejects_legacy_bare_estimator",
                "_check_load_wraps_missing_and_corrupt_artifacts",
            )
        )

    def _check_team_state_after_training_time_is_rejected_and_recovers(self):
        future_state = {
            "57": {
                "elo": 1532.5,
                "matches_played": 12,
                "last_match_at": "2099-01-01T00:00:00+00:00",
                "recent": [],
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pl.bundle"
            save_model_bundle(valid_bundle(training_rows=100), path)
            save_model_bundle(valid_bundle(training_rows=120), path)
            path.write_bytes(
                pickle.dumps(
                    valid_bundle(team_states=future_state, training_rows=140)
                )
            )

            loaded = load_model_bundle_with_fallback(path)

            self.assertTrue(loaded.recovered_from_previous)
            self.assertEqual(loaded.bundle.training_rows, 100)
            with self.assertRaises(ModelArtifactError):
                save_model_bundle(
                    valid_bundle(team_states=future_state),
                    Path(temp_dir) / "invalid.bundle",
                )

    def _check_invalid_team_state_in_active_bundle_recovers_from_previous(self):
        corruptions = (
            {"57": {"elo": "bad"}},
            {"57": {"recent": "bad"}},
            {"57": {"matches_played": True}},
            {
                "57": {
                    "elo": 1500.0,
                    "matches_played": 1,
                    "last_match_at": 0,
                    "recent": [],
                }
            },
            {"not-a-team-id": {}},
        )
        for team_states in corruptions:
            with self.subTest(team_states=team_states), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "pl.bundle"
                save_model_bundle(valid_bundle(training_rows=100), path)
                save_model_bundle(valid_bundle(training_rows=120), path)
                path.write_bytes(
                    pickle.dumps(
                        valid_bundle(team_states=team_states, training_rows=140)
                    )
                )

                loaded = load_model_bundle_with_fallback(path)

                self.assertTrue(loaded.recovered_from_previous)
                self.assertEqual(loaded.bundle.training_rows, 100)

    def _check_inference_invalid_active_xgboost_recovers_from_previous(self):
        rows = [
            {"elo_diff": float(value), "form_points": float(value % 3)}
            for value in range(-6, 7)
        ]
        targets = [0 if value < -2 else 1 if value <= 2 else 2 for value in range(-6, 7)]
        for corruption in ("unfitted", "missing_booster", "bad_rounds"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "pl.bundle"
                save_model_bundle(valid_bundle(training_rows=100), path)
                save_model_bundle(valid_bundle(training_rows=120), path)
                model = XGBoostClassifier(
                    ("elo_diff", "form_points"), rounds=12
                ).fit(rows, targets)
                if corruption == "unfitted":
                    model.is_fitted_ = False
                elif corruption == "missing_booster":
                    model.booster_ = None
                else:
                    model.rounds_used_ = 0
                path.write_bytes(
                    pickle.dumps(
                        valid_bundle(estimator=model, training_rows=140)
                    )
                )

                loaded = load_model_bundle_with_fallback(path)

                self.assertTrue(loaded.recovered_from_previous)
                self.assertEqual(loaded.bundle.training_rows, 100)

    def _check_second_save_keeps_first_valid_bundle_as_previous_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pl.bundle"
            first = valid_bundle(
                trained_at="2026-08-09T07:00:00+00:00",
                training_rows=100,
            )
            second = valid_bundle(
                trained_at="2026-08-10T07:00:00+00:00",
                training_rows=120,
            )
            save_model_bundle(first, path)
            save_model_bundle(second, path)

            active = load_model_bundle(path)
            previous = load_model_bundle(previous_model_path(path))

        self.assertEqual(active.training_rows, 120)
        self.assertEqual(previous.training_rows, 100)

    def _check_corrupt_active_bundle_recovers_explicitly_from_previous_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pl.bundle"
            save_model_bundle(
                valid_bundle(
                    trained_at="2026-08-09T07:00:00+00:00",
                    training_rows=100,
                ),
                path,
            )
            save_model_bundle(
                valid_bundle(
                    trained_at="2026-08-10T07:00:00+00:00",
                    training_rows=120,
                ),
                path,
            )
            path.write_bytes(b"corrupt-active")

            loaded = load_model_bundle_with_fallback(path)
            expected_digest = hashlib.sha256(
                loaded.source_path.read_bytes()
            ).hexdigest()

        self.assertTrue(loaded.recovered_from_previous)
        self.assertEqual(loaded.bundle.training_rows, 100)
        self.assertEqual(loaded.source_path.name, "pl.previous.bundle")
        self.assertEqual(
            loaded.artifact_sha256,
            expected_digest,
        )

    def _check_semantically_malformed_active_bundle_also_recovers_from_previous(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pl.bundle"
            save_model_bundle(
                valid_bundle(training_rows=100),
                path,
            )
            save_model_bundle(
                valid_bundle(training_rows=120),
                path,
            )
            path.write_bytes(
                pickle.dumps(
                    valid_bundle(
                        feature_names=42,
                        feature_defaults={},
                        training_rows=140,
                    )
                )
            )

            loaded = load_model_bundle_with_fallback(path)

        self.assertTrue(loaded.recovered_from_previous)
        self.assertEqual(loaded.bundle.training_rows, 100)

    def _check_invalid_default_keys_in_active_bundle_trigger_previous_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pl.bundle"
            save_model_bundle(valid_bundle(training_rows=100), path)
            save_model_bundle(valid_bundle(training_rows=120), path)
            malformed = valid_bundle(training_rows=140)
            malformed.feature_defaults[7] = 0.0
            path.write_bytes(pickle.dumps(malformed))

            loaded = load_model_bundle_with_fallback(path)

        self.assertTrue(loaded.recovered_from_previous)
        self.assertEqual(loaded.bundle.training_rows, 100)

    def _check_runtime_error_during_unpickle_triggers_previous_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pl.bundle"
            save_model_bundle(valid_bundle(training_rows=100), path)
            save_model_bundle(valid_bundle(training_rows=120), path)
            path.write_bytes(pickle.dumps(ExplodingUnpicklePayload()))

            loaded = load_model_bundle_with_fallback(path)

        self.assertTrue(loaded.recovered_from_previous)
        self.assertEqual(loaded.bundle.training_rows, 100)

    def _check_round_trip_preserves_self_describing_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pl.bundle"
            saved_path = save_model_bundle(valid_bundle(), path)
            # 模型包含训练状态，不按公开站点制品的权限发布。
            self.assertEqual(stat.S_IMODE(saved_path.stat().st_mode), 0o600)
            loaded = load_model_bundle(path)

        self.assertEqual(saved_path, path)
        self.assertEqual(loaded.schema_version, MODEL_SCHEMA_VERSION)
        self.assertEqual(loaded.competition_code, "PL")
        self.assertEqual(loaded.feature_names, ("elo_diff", "form_points"))
        self.assertEqual(loaded.classes, ("L", "D", "W"))
        self.assertEqual(loaded.team_states["57"]["matches_played"], 12)
        self.assertIsInstance(loaded.estimator, FakeEstimator)

    def _check_save_atomically_replaces_existing_file_without_temp_residue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = directory / "pl.bundle"
            path.write_bytes(b"old-content")

            save_model_bundle(valid_bundle(), path)

            self.assertEqual(load_model_bundle(path).competition_id, 2021)
            self.assertEqual(list(directory.glob(".pl.bundle.*.tmp")), [])

    def _check_save_aligns_private_artifact_owner_with_model_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pl.bundle"
            parent = path.parent.stat()
            with patch.object(artifacts.os, "fchown") as fchown:
                save_model_bundle(valid_bundle(), path)

        self.assertEqual(fchown.call_count, 1)
        self.assertEqual(fchown.call_args.args[1:], (parent.st_uid, parent.st_gid))

    def _check_validation_reports_all_independent_contract_failures(self):
        bundle = valid_bundle(
            competition_code=" pl ",
            competition_id=0,
            estimator=object(),
            feature_names=("elo_diff", "elo_diff"),
            classes=("W", "D", "L"),
            trained_at="2026-08-10T07:00:00",
            feature_defaults={"elo_diff": float("inf")},
            training_rows=0,
            schema_version=999,
        )

        with self.assertRaises(ModelArtifactError) as context:
            validate_model_bundle(bundle)

        message = str(context.exception)
        for fragment in (
            "schema_version",
            "competition_code",
            "competition_id",
            "feature_names",
            "classes",
            "trained_at",
            "training_rows",
            "feature_defaults",
            "predict_proba",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, message)

    def _check_load_wraps_missing_fields_from_older_bundle(self):
        bundle = valid_bundle()
        del bundle.feature_defaults
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "incomplete.bundle"
            path.write_bytes(pickle.dumps(bundle))
            with self.assertRaises(ModelArtifactError) as context:
                load_model_bundle(path)
        self.assertIn("不兼容", str(context.exception))

    def _check_failed_serialization_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = directory / "broken.bundle"
            with self.assertRaises(ModelArtifactError):
                save_model_bundle(valid_bundle(estimator=ExplodingEstimator()), path)
            self.assertFalse(path.exists())
            self.assertEqual(list(directory.glob(".broken.bundle.*.tmp")), [])

    def _check_load_rejects_legacy_bare_estimator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.bundle"
            path.write_bytes(pickle.dumps(FakeEstimator()))

            with self.assertRaises(ModelArtifactError) as context:
                load_model_bundle(path)

        self.assertIn("ModelBundle", str(context.exception))

    def _check_load_wraps_missing_and_corrupt_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.bundle"
            corrupt = Path(temp_dir) / "corrupt.bundle"
            corrupt.write_bytes(b"not-a-pickle")

            with self.assertRaises(ModelArtifactError) as missing_context:
                load_model_bundle(missing)
            with self.assertRaises(ModelArtifactError) as corrupt_context:
                load_model_bundle(corrupt)

        self.assertIn("不存在", str(missing_context.exception))
        self.assertIn("无法读取", str(corrupt_context.exception))


import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from app.ml.artifacts import ModelBundle, load_model_bundle, save_model_bundle
from app.ml.feature_engineering import FeatureEngineer
from app.ml.forecast_identity import forecast_id_for_prediction
from app.ml.prediction import PredictionEngine, PredictionError
from app.ml.training import ModelTrainer, TrainingError


def historical_match(match_id, kickoff, home_id, away_id, outcome):
    scores = {"W": (2, 0), "D": (1, 1), "L": (0, 2)}
    home_score, away_score = scores[outcome]
    return {
        "id": match_id,
        "utcDate": kickoff,
        "status": "FINISHED",
        "homeTeam": {"id": home_id, "name": f"Team {home_id}"},
        "awayTeam": {"id": away_id, "name": f"Team {away_id}"},
        "score": {"fullTime": {"home": home_score, "away": away_score}},
    }


def scheduled_match(match_id=900, home_id=1, away_id=99):
    return {
        "id": match_id,
        "utcDate": "2026-08-11T18:00:00Z",
        "status": "SCHEDULED",
        "homeTeam": {"id": home_id, "name": f"Home {home_id}"},
        "awayTeam": {"id": away_id, "name": f"Away {away_id}"},
        "score": {"fullTime": {"home": None, "away": None}},
    }


def season_matches(count=45):
    teams = (1, 2, 3, 4, 5, 6)
    outcomes = ("W", "D", "L")
    matches = []
    for index in range(count):
        home = teams[index % len(teams)]
        away = teams[(index * 2 + 1) % len(teams)]
        if home == away:
            away = teams[(teams.index(away) + 1) % len(teams)]
        matches.append(
            historical_match(
                index + 1,
                f"2025-{index // 27 + 1:02d}-{index % 27 + 1:02d}T12:00:00Z",
                home,
                away,
                outcomes[index % 3],
            )
        )
    return matches


class FakeClient:
    source_name = "football-data.org"

    def __init__(self, by_season=None, scheduled=None):
        self.by_season = by_season or {}
        self.scheduled = scheduled or []
        self.history_calls = []
        self.scheduled_calls = []

    def fetch_matches(self, code, *, season=None, status=None):
        self.history_calls.append((code, season, status))
        return list(self.by_season.get(season, []))

    def fetch_scheduled_matches(self, code, date_from, date_to):
        self.scheduled_calls.append((code, date_from, date_to))
        return list(self.scheduled)


class RecordingEstimator:
    fit_sizes = []

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, rows, targets):
        self.fit_sizes.append(len(rows))
        return self

    def predict_proba(self, rows):
        return [[0.34, 0.33, 0.33] for _ in rows]

    def predict(self, rows):
        return [0 for _ in rows]


class TrackingXGBoostEstimator(XGBoostClassifier):
    fit_sizes = []

    def __init__(self, feature_names):
        super().__init__(feature_names)

    def fit(self, rows, targets):
        type(self).fit_sizes.append(len(rows))
        return super().fit(rows, targets)


class FixedEstimator:
    def predict_proba(self, rows):
        return [[0.1, 0.2, 0.7] for _ in rows]

    def predict(self, rows):
        return [2 for _ in rows]


class MutableProbabilityEstimator(FixedEstimator):
    probabilities = [0.1, 0.2, 0.7]

    def predict_proba(self, rows):
        return [list(type(self).probabilities) for _ in rows]


class MisleadingEstimator(FixedEstimator):
    margin = 0.3

    def explain_prediction(self, row, *, top_n=3):
        return {
            "predicted_class_index": 0,
            "runner_up_class_index": 1,
            "logit_margin": type(self).margin,
            "top_drivers": [
                {
                    "feature": "elo_diff",
                    "raw_value": row["elo_diff"],
                    "contribution": 0.3,
                }
            ],
        }


class ExplodingExplanationEstimator(FixedEstimator):
    def explain_prediction(self, row, *, top_n=3):
        raise RuntimeError("explanation exploded")


class ExplodingProbabilityEstimator(FixedEstimator):
    calls = 0

    def predict_proba(self, rows):
        type(self).calls += 1
        if type(self).calls <= 2:
            return super().predict_proba(rows)
        raise RuntimeError("prediction exploded")


class ShiftedFeatureEngineer(FeatureEngineer):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def build_fixture_features(self, home_team_id, away_team_id, kickoff, team_states):
        values = super().build_fixture_features(
            home_team_id, away_team_id, kickoff, team_states
        )
        values["elo_diff"] += self.shift
        return values


class TestModelTrainer(unittest.TestCase):
    def test_training_lifecycle_and_quality_gates(self):
        for name in (
            "_check_training_rejects_worst_class_calibration",
            "_check_training_rejects_temporally_unstable_fold",
            "_check_training_uses_temporal_holdout_then_refits_all_rows",
            "_check_real_training_saves_loadable_self_describing_bundle",
            "_check_training_rejects_insufficient_history",
        ):
            with self.subTest(contract=name):
                getattr(self, name)()

    def _check_training_rejects_worst_class_calibration(self):
        trainer = ModelTrainer(
            client=FakeClient({"2025": season_matches(45)}),
            model_factory=RecordingEstimator,
            minimum_matches=12,
            maximum_calibration_error=1.0,
            maximum_classwise_calibration_error=0.0,
        )

        with self.assertRaises(TrainingError) as context:
            trainer.train_competition("PL", 2021, ["2025"], save=False)

        self.assertIn("最差类别校准误差", str(context.exception))

    def _check_training_rejects_temporally_unstable_fold(self):
        RecordingEstimator.fit_sizes = []
        trainer = ModelTrainer(
            client=FakeClient({"2025": season_matches(45)}),
            model_factory=RecordingEstimator,
            minimum_matches=12,
            minimum_worst_fold_accuracy_lift=-0.1,
        )

        with self.assertRaises(TrainingError) as context:
            trainer.train_competition("PL", 2021, ["2025"], save=False)

        self.assertIn("最差折", str(context.exception))

    def _check_training_uses_temporal_holdout_then_refits_all_rows(self):
        matches = season_matches(45)
        client = FakeClient({"2025": matches})
        TrackingXGBoostEstimator.fit_sizes = []
        trainer = ModelTrainer(
            client=client,
            model_factory=TrackingXGBoostEstimator,
            minimum_matches=12,
            holdout_fraction=0.2,
            clock=lambda: datetime(2026, 8, 10, 7, tzinfo=UTC),
        )

        bundle = trainer.train_competition("pl", 2021, ["2025"], save=False)

        self.assertEqual(TrackingXGBoostEstimator.fit_sizes, [22, 30, 38, 36, 45])
        self.assertEqual(bundle.competition_code, "PL")
        self.assertEqual(bundle.training_rows, 45)
        self.assertEqual(bundle.classes, ("L", "D", "W"))
        self.assertEqual(bundle.metrics["validation_rows"], 9)
        self.assertIn("validation_log_loss", bundle.metrics)
        self.assertIn("baseline_accuracy", bundle.metrics)
        self.assertIn("elo_baseline_accuracy", bundle.metrics)
        self.assertIn("majority_baseline_accuracy", bundle.metrics)
        self.assertEqual(bundle.metrics["backtest_folds"], 3)
        self.assertEqual(bundle.metrics["backtest_rows"], 23)
        self.assertIn("backtest_calibration_error", bundle.metrics)
        self.assertGreaterEqual(
            bundle.metrics["backtest_baseline_accuracy"],
            bundle.metrics["backtest_elo_baseline_accuracy"],
        )
        self.assertGreaterEqual(
            bundle.metrics["backtest_baseline_accuracy"],
            bundle.metrics["backtest_majority_baseline_accuracy"],
        )
        self.assertIn("validation_ranked_probability_score", bundle.metrics)
        self.assertIn("backtest_ranked_probability_score", bundle.metrics)
        self.assertIn("backtest_classwise_calibration_error", bundle.metrics)
        self.assertIn("backtest_sharpness", bundle.metrics)
        self.assertIn("backtest_worst_fold_accuracy_lift", bundle.metrics)
        self.assertIn("backtest_accuracy_lift_stddev", bundle.metrics)
        self.assertEqual(len(bundle.metrics["backtest_reliability"]), 3)
        self.assertEqual(len(bundle.metrics["backtest_fold_metrics"]), 3)
        self.assertEqual(client.history_calls, [("PL", "2025", "FINISHED")])

    def _check_real_training_saves_loadable_self_describing_bundle(self):
        client = FakeClient({"2025": season_matches(45)})
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(
                client=client,
                model_dir=Path(temp_dir),
                minimum_matches=12,
                clock=lambda: datetime(2026, 8, 10, 7, tzinfo=UTC),
            )
            bundle = trainer.train_competition("PL", 2021, ["2025"])
            loaded = load_model_bundle(Path(temp_dir) / "pl.bundle")

        self.assertEqual(loaded.training_rows, 45)
        self.assertEqual(loaded.feature_names, FeatureEngineer.FEATURES)
        self.assertEqual(len(loaded.team_states), 6)
        self.assertGreaterEqual(bundle.metrics["validation_log_loss"], 0.0)
        self.assertGreaterEqual(bundle.metrics["validation_brier_score"], 0.0)
        self.assertGreaterEqual(bundle.metrics["backtest_log_loss"], 0.0)
        self.assertGreaterEqual(bundle.metrics["backtest_brier_score"], 0.0)
        self.assertGreaterEqual(bundle.metrics["backtest_calibration_error"], 0.0)

    def _check_training_rejects_insufficient_history(self):
        trainer = ModelTrainer(
            client=FakeClient({"2025": season_matches(5)}),
            minimum_matches=12,
        )
        with self.assertRaises(TrainingError) as context:
            trainer.train_competition("PL", 2021, ["2025"], save=False)
        self.assertIn("至少", str(context.exception))


class TestPredictionEngine(unittest.TestCase):
    def _run_contracts(self, names: tuple[str, ...]) -> None:
        for name in names:
            with self.subTest(contract=name):
                getattr(self, name)()

    def test_prediction_integrity_and_explanation_contracts(self):
        self._run_contracts(
            (
                "_check_duplicate_fixtures_are_merged_and_conflicts_fail_closed",
                "_check_forecast_id_binds_features_and_final_probabilities",
                "_check_explanation_runtime_failure_does_not_remove_prediction",
                "_check_mismatched_or_non_finite_explanation_fails_closed",
                "_check_prediction_maps_model_classes_and_reports_cold_start_quality",
                "_check_trained_xgboost_prediction_exports_tree_shap_drivers",
            )
        )

    def test_prediction_recovery_and_run_contracts(self):
        self._run_contracts(
            (
                "_check_prediction_uses_previous_bundle_and_marks_recovery",
                "_check_prediction_run_reports_model_quality_without_fixtures",
                "_check_prediction_rejects_feature_schema_mismatch",
                "_check_prediction_run_keeps_other_competitions_when_one_model_is_missing",
            )
        )

    def _check_duplicate_fixtures_are_merged_and_conflicts_fail_closed(self):
        fixture = scheduled_match()
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            save_model_bundle(self.make_bundle(), models_dir / "pl.bundle")
            duplicate_predictions = PredictionEngine(
                models_dir=models_dir,
                client=FakeClient(scheduled=[fixture, dict(fixture)]),
            ).predict_competition(
                "PL", days=1, now=datetime(2026, 8, 10, 7, tzinfo=UTC)
            )
            conflict = scheduled_match(away_id=98)
            with self.assertRaises(PredictionError) as context:
                PredictionEngine(
                    models_dir=models_dir,
                    client=FakeClient(scheduled=[fixture, conflict]),
                ).predict_competition(
                    "PL", days=1, now=datetime(2026, 8, 10, 7, tzinfo=UTC)
                )

        self.assertEqual(len(duplicate_predictions), 1)
        self.assertEqual(context.exception.code, "DUPLICATE_FIXTURE_CONFLICT")

    def _check_forecast_id_binds_features_and_final_probabilities(self):
        now = datetime(2026, 8, 10, 7, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            save_model_bundle(
                self.make_bundle(estimator=MutableProbabilityEstimator()),
                models_dir / "pl.bundle",
            )
            base_engine = PredictionEngine(
                models_dir=models_dir,
                client=FakeClient(scheduled=[scheduled_match(home_id=1, away_id=2)]),
                feature_engineer=ShiftedFeatureEngineer(0.0),
            )
            base = base_engine.predict_competition("PL", days=1, now=now)[0]
            identical = base_engine.predict_competition("PL", days=1, now=now)[0]
            shifted = PredictionEngine(
                models_dir=models_dir,
                client=FakeClient(scheduled=[scheduled_match(home_id=1, away_id=2)]),
                feature_engineer=ShiftedFeatureEngineer(25.0),
            ).predict_competition("PL", days=1, now=now)[0]
            MutableProbabilityEstimator.probabilities = [0.2, 0.3, 0.5]
            changed_probability = base_engine.predict_competition(
                "PL", days=1, now=now
            )[0]
            MutableProbabilityEstimator.probabilities = [0.1, 0.2, 0.7]

        self.assertNotEqual(base["forecast_id"], shifted["forecast_id"])
        self.assertNotEqual(base["forecast_id"], changed_probability["forecast_id"])
        self.assertEqual(base["forecast_id"], identical["forecast_id"])

    def _check_explanation_runtime_failure_does_not_remove_prediction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            save_model_bundle(
                self.make_bundle(estimator=ExplodingExplanationEstimator()),
                models_dir / "pl.bundle",
            )
            prediction = PredictionEngine(
                models_dir=models_dir,
                client=FakeClient(scheduled=[scheduled_match()]),
            ).predict_competition(
                "PL", days=1, now=datetime(2026, 8, 10, 7, tzinfo=UTC)
            )[0]

        self.assertFalse(prediction["explanation"]["available"])
        self.assertEqual(
            prediction["explanation"]["reason"],
            "explanation_contract_failed",
        )

    def _check_mismatched_or_non_finite_explanation_fails_closed(self):
        for margin in (0.3, float("nan")):
            with self.subTest(margin=margin), tempfile.TemporaryDirectory() as temp_dir:
                models_dir = Path(temp_dir)
                MisleadingEstimator.margin = margin
                save_model_bundle(
                    self.make_bundle(estimator=MisleadingEstimator()),
                    models_dir / "pl.bundle",
                )
                prediction = PredictionEngine(
                    models_dir=models_dir,
                    client=FakeClient(scheduled=[scheduled_match()]),
                ).predict_competition(
                    "PL", days=1, now=datetime(2026, 8, 10, 7, tzinfo=UTC)
                )[0]

                self.assertFalse(prediction["explanation"]["available"])
    def make_bundle(self, **overrides):
        engineer = FeatureEngineer()
        state = engineer.build_training_dataset(
            [historical_match(1, "2026-08-01T18:00:00Z", 1, 2, "W")]
        )
        values = {
            "competition_code": "PL",
            "competition_id": 2021,
            "estimator": FixedEstimator(),
            "feature_names": engineer.features,
            "classes": ("L", "D", "W"),
            "trained_at": "2026-08-10T07:00:00+00:00",
            "metrics": {"validation_accuracy": 0.55, "validation_log_loss": 0.95},
            "team_states": state.team_states,
            "feature_defaults": state.feature_defaults,
            "training_rows": 100,
        }
        values.update(overrides)
        return ModelBundle(**values)

    def _check_prediction_uses_previous_bundle_and_marks_recovery(self):
        client = FakeClient(scheduled=[scheduled_match()])
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            path = models_dir / "pl.bundle"
            save_model_bundle(self.make_bundle(training_rows=100), path)
            save_model_bundle(self.make_bundle(training_rows=120), path)
            path.write_bytes(b"corrupt-active")

            predictions = PredictionEngine(
                models_dir=models_dir,
                client=client,
            ).predict_competition(
                "PL",
                days=1,
                now=datetime(2026, 8, 10, 7, tzinfo=UTC),
            )

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["model"]["training_rows"], 100)
        self.assertTrue(predictions[0]["model"]["recovered_from_previous"])

    def _check_prediction_run_reports_model_quality_without_fixtures(self):
        metrics = {
            "validation_accuracy": 0.61,
            "validation_log_loss": 0.88,
            "baseline_accuracy": 0.48,
            "backtest_folds": 3,
            "backtest_rows": 60,
            "backtest_accuracy": 0.57,
            "backtest_baseline_accuracy": 0.46,
            "backtest_log_loss": 0.91,
            "backtest_brier_score": 0.58,
            "backtest_calibration_error": 0.08,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            save_model_bundle(
                self.make_bundle(metrics=metrics),
                models_dir / "pl.bundle",
            )
            run = PredictionEngine(
                models_dir=models_dir,
                client=FakeClient(scheduled=[]),
            ).predict_all(
                {"PL": 2021},
                days=1,
                now=datetime(2026, 8, 10, 7, tzinfo=UTC),
            )

        self.assertEqual(run.predictions, ())
        self.assertEqual(len(run.model_reports), 1)
        report = run.model_reports[0]
        self.assertEqual(report["competition"], "PL")
        self.assertEqual(report["backtest_folds"], 3)
        self.assertEqual(report["backtest_calibration_error"], 0.08)
        self.assertEqual(len(report["backtest_fold_metrics"]), 0)
        self.assertFalse(report["recovered_from_previous"])

    def _check_prediction_maps_model_classes_and_reports_cold_start_quality(self):
        client = FakeClient(scheduled=[scheduled_match()])
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            save_model_bundle(self.make_bundle(), models_dir / "pl.bundle")
            engine = PredictionEngine(models_dir=models_dir, client=client)

            predictions = engine.predict_competition(
                "PL",
                days=3,
                now=datetime(2026, 8, 10, 7, tzinfo=UTC),
            )

        self.assertEqual(len(predictions), 1)
        prediction = predictions[0]
        self.assertEqual(prediction["prediction"], "HOME_WIN")
        self.assertEqual(prediction["probabilities"], {
            "home_win": 0.7,
            "draw": 0.2,
            "away_win": 0.1,
        })
        self.assertEqual(prediction["confidence"], 0.7)
        self.assertEqual(prediction["confidence_semantics"], "top_outcome_probability")
        self.assertEqual(prediction["probability_margin"], 0.5)
        self.assertGreater(prediction["uncertainty"], 0.0)
        self.assertFalse(prediction["explanation"]["available"])
        self.assertEqual(prediction["predicted_at"], "2026-08-10T07:00:00+00:00")
        self.assertEqual(len(prediction["forecast_id"]), 64)
        self.assertEqual(len(prediction["feature_fingerprint"]), 64)
        self.assertEqual(
            prediction["forecast_id"],
            forecast_id_for_prediction(prediction),
        )
        self.assertEqual(len(prediction["model"]["model_sha256"]), 64)
        self.assertNotIn("backtest_reliability", prediction["model"])
        self.assertNotIn("backtest_fold_metrics", prediction["model"])
        self.assertEqual(
            prediction["data_provenance"],
            {
                "source": "football-data.org",
                "source_match_id": 900,
                "source_updated_at": None,
                "snapshot_at": "2026-08-10T07:00:00+00:00",
            },
        )
        self.assertTrue(prediction["data_quality"]["is_cold_start"])
        self.assertEqual(prediction["data_quality"]["home_history_matches"], 1)
        self.assertEqual(prediction["data_quality"]["away_history_matches"], 0)
        self.assertEqual(prediction["model"]["training_rows"], 100)
        self.assertEqual(client.scheduled_calls, [("PL", "2026-08-10", "2026-08-13")])

    def _check_trained_xgboost_prediction_exports_tree_shap_drivers(self):
        engineer = FeatureEngineer()
        rows = []
        targets = []
        for _ in range(20):
            for signal, target in ((-1.0, 0), (0.0, 1), (1.0, 2)):
                row = {name: 0.0 for name in engineer.features}
                row["home_advantage"] = signal
                rows.append(row)
                targets.append(target)
        model = XGBoostClassifier(
            engineer.features,
            rounds=60,
            minimum_child_weight=1.0,
        ).fit(rows, targets)
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            save_model_bundle(self.make_bundle(estimator=model), models_dir / "pl.bundle")
            prediction = PredictionEngine(
                models_dir=models_dir,
                client=FakeClient(scheduled=[scheduled_match(home_id=1, away_id=2)]),
            ).predict_competition(
                "PL", days=1, now=datetime(2026, 8, 10, 7, tzinfo=UTC)
            )[0]

        explanation = prediction["explanation"]
        self.assertTrue(explanation["available"])
        self.assertEqual(explanation["method"], "tree_shap_margin")
        self.assertEqual(explanation["predicted_vs_runner_up"], "HOME_WIN vs DRAW")
        self.assertEqual(explanation["top_drivers"][0]["feature"], "home_advantage")
        self.assertEqual(explanation["top_drivers"][0]["direction"], "supports_prediction")

    def _check_prediction_rejects_feature_schema_mismatch(self):
        client = FakeClient(scheduled=[scheduled_match()])
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            save_model_bundle(
                self.make_bundle(
                    feature_names=("different",),
                    feature_defaults={"different": 0.0},
                ),
                models_dir / "pl.bundle",
            )
            engine = PredictionEngine(models_dir=models_dir, client=client)
            with self.assertRaises(PredictionError) as context:
                engine.predict_competition("PL", days=1)
        self.assertIn("特征版本", str(context.exception))

    def _check_prediction_run_keeps_other_competitions_when_one_model_is_missing(self):
        client = FakeClient(scheduled=[])
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            save_model_bundle(self.make_bundle(), models_dir / "pl.bundle")
            run = PredictionEngine(models_dir=models_dir, client=client).predict_all(
                {"PL": 2021, "SA": 2019},
                days=1,
                now=datetime(2026, 8, 10, 7, tzinfo=UTC),
            )

        self.assertEqual(run.predictions, ())
        self.assertEqual(run.models_loaded, 1)
        self.assertEqual(run.competitions_requested, 2)
        self.assertEqual(run.errors[0]["competition"], "SA")
        self.assertEqual(run.errors[0]["code"], "MODEL_UNAVAILABLE")

        ExplodingProbabilityEstimator.calls = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            save_model_bundle(
                self.make_bundle(estimator=ExplodingProbabilityEstimator()),
                models_dir / "pl.bundle",
            )
            save_model_bundle(
                self.make_bundle(competition_code="SA", competition_id=2019),
                models_dir / "sa.bundle",
            )
            run = PredictionEngine(
                models_dir=models_dir,
                client=FakeClient(scheduled=[scheduled_match()]),
            ).predict_all(
                {"PL": 2021, "SA": 2019},
                days=1,
                now=datetime(2026, 8, 10, 7, tzinfo=UTC),
            )

        self.assertEqual(len(run.predictions), 1)
        self.assertEqual(run.predictions[0]["competition"], "SA")
        self.assertEqual(run.errors[0]["competition"], "PL")
        self.assertEqual(run.errors[0]["code"], "UNEXPECTED_PREDICTION_ERROR")


import subprocess
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import pipeline


@dataclass
class FakeBundle:
    training_rows: int
    metrics: dict


class FakeTrainer:
    def __init__(self):
        self.calls = []

    def train_competition(self, code, competition_id, seasons):
        self.calls.append((code, competition_id, tuple(seasons)))
        if code == "SA":
            raise RuntimeError("source unavailable")
        return FakeBundle(
            120,
            {
                "validation_accuracy": 0.56,
                "validation_ranked_probability_score": 0.21,
                "backtest_accuracy": 0.54,
                "backtest_baseline_accuracy": 0.48,
                "backtest_majority_baseline_accuracy": 0.41,
                "backtest_elo_baseline_accuracy": 0.48,
                "backtest_calibration_error": 0.09,
                "backtest_ranked_probability_score": 0.22,
                "backtest_classwise_calibration_error": 0.08,
                "backtest_sharpness": 0.61,
                "backtest_worst_fold_accuracy_lift": 0.02,
                "backtest_accuracy_lift_stddev": 0.03,
            },
        )


class TestPipeline(unittest.TestCase):
    def test_training_pipeline_returns_structured_partial_result(self):
        trainer = FakeTrainer()

        result = pipeline.run_training_pipeline(
            competition_codes="PL,SA",
            seasons=["2025"],
            trainer=trainer,
        )

        self.assertEqual(result["trained_count"], 1)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["results"][0]["competition"], "PL")
        self.assertEqual(result["results"][0]["training_rows"], 120)
        self.assertEqual(result["results"][0]["backtest_accuracy"], 0.54)
        self.assertEqual(result["results"][0]["backtest_baseline_accuracy"], 0.48)
        self.assertEqual(result["results"][0]["backtest_elo_baseline_accuracy"], 0.48)
        self.assertEqual(result["results"][0]["backtest_calibration_error"], 0.09)
        self.assertEqual(result["results"][0]["validation_ranked_probability_score"], 0.21)
        self.assertEqual(result["results"][0]["backtest_ranked_probability_score"], 0.22)
        self.assertEqual(result["results"][0]["backtest_worst_fold_accuracy_lift"], 0.02)
        self.assertEqual(result["errors"][0]["competition"], "SA")
        self.assertEqual(trainer.calls, [
            ("PL", 2021, ("2025",)),
            ("SA", 2019, ("2025",)),
        ])

    def test_cli_exposes_only_current_commands_in_minimal_runtime(self):
        parser = pipeline.build_parser()
        help_text = parser.format_help()
        self.assertIn("train", help_text)
        self.assertIn("predict", help_text)
        self.assertIn("export-site", help_text)
        self.assertIn("all", help_text)
        self.assertNotIn("simulate", help_text)
        export_args = parser.parse_args(["export-site", "--competitions", "PL"])
        self.assertEqual(export_args.competitions, "PL")
        result = subprocess.run(
            [sys.executable, "-m", "app.pipeline", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Soccer Analytics", result.stdout)

    def test_pipeline_forwards_configuration_scope_and_refresh_errors(self):
        settings = SimpleNamespace(
            model_dir=Path("/custom/model-store"),
            site_data_dir=Path("/custom/site-data"),
            site_export_days=2,
            prediction_days=5,
            competitions_map={"PL": 2021, "SA": 2019},
        )
        with self.subTest("configured prediction range"):
            engine = Mock()
            engine.predict_all.return_value = SimpleNamespace(
                predictions=(),
                errors=(),
                models_loaded=1,
                competitions_requested=2,
            )
            with patch.object(pipeline, "load_settings", return_value=settings):
                pipeline.run_predictions_pipeline(engine=engine)
            engine.predict_all.assert_called_once_with(
                settings.competitions_map,
                days=5,
            )

        with self.subTest("configured artifact directories"):
            with patch.object(
                pipeline,
                "load_settings",
                return_value=settings,
            ), patch.object(pipeline, "export_site_data", return_value={}) as export:
                pipeline.run_export_site_pipeline()
            export.assert_called_once_with(
                days=2,
                models_dir=settings.model_dir,
                output_dir=settings.site_data_dir,
            )

        refresh_errors = [
            {
                "competition": "SA",
                "code": "TrainingError",
                "message": "refresh failed",
            }
        ]
        with self.subTest("full pipeline scope"):
            with patch.object(
                pipeline,
                "run_training_pipeline",
                return_value={"trained_count": 1, "errors": refresh_errors},
            ) as training, patch.object(
                pipeline,
                "run_export_site_pipeline",
                return_value={},
            ) as export:
                pipeline.run_full_pipeline(
                    days=2,
                    competition_codes="PL",
                    seasons=["2025"],
                    trainer="trainer",
                )
            training.assert_called_once_with(
                competition_codes="PL",
                seasons=["2025"],
                trainer="trainer",
            )
            export.assert_called_once_with(
                days=2,
                competition_codes="PL",
                refresh_errors=refresh_errors,
            )

        with self.subTest("selected competition export"):
            with patch.object(
                pipeline,
                "load_settings",
                return_value=settings,
            ), patch.object(pipeline, "export_site_data", return_value={}) as export:
                pipeline.run_export_site_pipeline(competition_codes="PL")
            export.assert_called_once_with(
                days=2,
                models_dir=settings.model_dir,
                output_dir=settings.site_data_dir,
                competitions={"PL": 2021},
            )
