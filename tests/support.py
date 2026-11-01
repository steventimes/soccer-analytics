"""Shared test contracts, optional-runtime gates, and forecast fixtures."""

import importlib.util
import unittest


class ContractTestCase(unittest.TestCase):
    def _run_contracts(self, names: tuple[str, ...]) -> None:
        for name in names:
            with self.subTest(contract=name):
                getattr(self, name)()


def requires_modules(*names):
    """Skip only checks that execute a specific optional runtime."""
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    return unittest.skipIf(bool(missing), "Optional dependencies missing: " + ", ".join(missing))


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
    from app.ml.forecast_identity import forecast_id_for_prediction

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
