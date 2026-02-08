from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.web.predictions import generate_predictions
from app.web.preset_questions import build_preset_questions, load_cached_questions
from app.web.scores import fetch_daily_scores
from app.web.site_paths import data_dir, repo_root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def export_site_data(days: int = 1) -> dict[str, Path]:
    data_path = data_dir()
    data_path.mkdir(parents=True, exist_ok=True)

    models_dir = repo_root() / "models"
    predictions = generate_predictions(models_dir=models_dir, days=days)
    predictions_payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "predictions": predictions,
    }
    predictions_path = data_path / "predictions.json"
    _write_json(predictions_path, predictions_payload)

    preset_cache_path = data_path / "preset_questions.json"
    cached = load_cached_questions(preset_cache_path)
    if cached is None:
        preset_payload = build_preset_questions(predictions)
        _write_json(preset_cache_path, preset_payload)

    scores_payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "scores": fetch_daily_scores(),
    }
    scores_path = data_path / "scores.json"
    _write_json(scores_path, scores_payload)

    return {
        "predictions": predictions_path,
        "preset_questions": preset_cache_path,
        "scores": scores_path,
    }


if __name__ == "__main__":
    export_site_data()