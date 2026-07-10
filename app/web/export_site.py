from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.web.operations import build_operations_summary
from app.web.preset_questions import build_preset_questions, load_cached_questions
from app.web.release_governance import build_release_governance
from app.web.scores import fetch_daily_scores
from app.web.site_paths import data_dir, repo_root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _artifact_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _build_manifest(
    predictions_path: Path,
    preset_cache_path: Path,
    scores_path: Path,
    operations_path: Path,
    release_path: Path,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": _timestamp(),
        "prediction_count": len(predictions),
        "files": {
            "predictions": _artifact_path(predictions_path),
            "preset_questions": _artifact_path(preset_cache_path),
            "scores": _artifact_path(scores_path),
            "operations": _artifact_path(operations_path),
            "release_governance": _artifact_path(release_path),
        },
    }


def _generate_predictions(models_dir: Path, days: int) -> list[dict[str, Any]]:
    from app.web.predictions import generate_predictions

    return generate_predictions(models_dir=models_dir, days=days)


def export_site_data(days: int = 1) -> dict[str, Path]:
    data_path = data_dir()
    data_path.mkdir(parents=True, exist_ok=True)

    models_dir = repo_root() / "models"
    predictions = _generate_predictions(models_dir=models_dir, days=days)
    predictions_payload = {
        "generated_at": _timestamp(),
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
        "generated_at": _timestamp(),
        "scores": fetch_daily_scores(),
    }
    scores_path = data_path / "scores.json"
    _write_json(scores_path, scores_payload)

    operations_path = data_path / "operations.json"
    operations_summary = build_operations_summary(predictions, scores_payload["scores"])
    _write_json(operations_path, operations_summary)

    release_path = data_path / "release.json"
    _write_json(
        release_path,
        build_release_governance(predictions, operations_summary),
    )

    manifest_path = data_path / "manifest.json"
    _write_json(
        manifest_path,
        _build_manifest(predictions_path, preset_cache_path, scores_path, operations_path, release_path, predictions),
    )

    return {
        "manifest": manifest_path,
        "predictions": predictions_path,
        "preset_questions": preset_cache_path,
        "scores": scores_path,
        "operations": operations_path,
        "release_governance": release_path,
    }


if __name__ == "__main__":
    export_site_data()
