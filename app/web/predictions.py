from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import COMPETITIONS_MAP
from app.data_service.fetch.fetcher import FootballDataClient
from app.ml.feature_engineering import FeatureEngineer


def _default_feature_payload() -> dict[str, float]:
    return {
        "rolling_xG": 1.5,
        "rolling_xGA": 1.2,
        "rolling_deep": 5.0,
        "rolling_ppda": 10.0,
        "rolling_goals": 1.2,
        "rolling_wins": 2.0,
        "is_home": 1.0,
        "xG_diff": 0.1,
        "ppda_diff": -2.0,
        "deep_diff": 1.0,
        "points_diff": 5.0,
    }


def _label_for_class(value: Any) -> str:
    label_map = {0: "Loss", 1: "Draw", 2: "Win"}
    try:
        return label_map.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def generate_predictions(models_dir: Path, days: int = 1) -> list[dict[str, Any]]:
    feature_engineer = FeatureEngineer()
    client = FootballDataClient()

    date_from = datetime.now().strftime("%Y-%m-%d")
    date_to = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

    predictions: list[dict[str, Any]] = []

    for code in COMPETITIONS_MAP.keys():
        model_path = models_dir / f"{code.lower()}_model.joblib"
        if not model_path.exists():
            continue

        model = joblib.load(model_path)
        matches_data = client._get(
            f"competitions/{code}/matches",
            {"status": "SCHEDULED", "dateFrom": date_from, "dateTo": date_to},
        )

        if not matches_data or "matches" not in matches_data:
            continue

        for match in matches_data["matches"]:
            payload = _default_feature_payload()
            features_df = pd.DataFrame([payload])
            X = features_df[feature_engineer.features]
            probs = model.predict_proba(X)[0]

            class_labels = [_label_for_class(label) for label in model.classes_]
            probability_table = {
                label: float(prob)
                for label, prob in zip(class_labels, probs)
            }
            best_label = max(probability_table, key=probability_table.get)

            predictions.append(
                {
                    "competition": code,
                    "utc_date": match.get("utcDate"),
                    "home_team": match.get("homeTeam", {}).get("name"),
                    "away_team": match.get("awayTeam", {}).get("name"),
                    "prediction": best_label,
                    "confidence": probability_table[best_label],
                    "probabilities": probability_table,
                }
            )

    return predictions