from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.config import load_settings
from app.ml.prediction import PredictionEngine, PredictionRun
from app.web.operations import build_operations_summary
from app.web.preset_questions import build_preset_questions
from app.web.release_governance import build_release_governance
from app.web.scores import fetch_daily_scores_with_diagnostics
from app.web.site_paths import repo_root


def _utc(now: datetime | None = None) -> datetime:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """同目录落临时文件再替换；失败时旧制品保持原样。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(_json_bytes(payload))
            handle.flush()
            # 容器以不同 UID 写挂载卷时，让制品归属于输出目录所有者。
            try:
                parent = path.parent.stat()
                os.fchown(handle.fileno(), parent.st_uid, parent.st_gid)
            except (AttributeError, OSError):
                pass
            # 站点制品需要由容器外的静态服务器和审计进程读取。
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        # 目录 fsync 让 rename 在突然断电后也具备持久性；不支持时文件原子性仍成立。
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _artifact_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _generate_prediction_run(
    models_dir: Path,
    days: int,
    *,
    now: datetime,
    competitions: Mapping[str, int] | None = None,
) -> PredictionRun:
    return PredictionEngine(models_dir=models_dir).predict_all(
        competitions,
        days=days,
        now=now,
    )


def _manifest(
    *,
    outputs: dict[str, Path],
    generated_at: str,
    predictions: list[dict[str, Any]],
    errors: list[dict[str, str]],
    refresh_errors: list[dict[str, Any]],
    score_errors: list[dict[str, Any]],
    models_loaded: int,
    competitions_requested: int,
) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    for name, path in outputs.items():
        raw = path.read_bytes()
        artifacts[name] = {
            "path": _artifact_path(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    ratio = None if competitions_requested == 0 else round(models_loaded / competitions_requested, 4)
    run_seed = generated_at + "|" + "|".join(
        artifacts[name]["sha256"] for name in sorted(artifacts)
    )
    return {
        "schema": "SoccerAnalytics.SiteManifest.v3",
        "generated_at": generated_at,
        "run_id": hashlib.sha256(run_seed.encode("utf-8")).hexdigest()[:20],
        "prediction_count": len(predictions),
        "error_count": len(errors) + len(refresh_errors) + len(score_errors),
        "prediction_error_count": len(errors),
        "model_refresh_error_count": len(refresh_errors),
        "score_error_count": len(score_errors),
        "model_coverage": {
            "loaded": models_loaded,
            "requested": competitions_requested,
            "ratio": ratio,
        },
        "files": {name: metadata["path"] for name, metadata in artifacts.items()},
        "artifacts": artifacts,
    }


def export_site_data(
    days: int = 1,
    *,
    prediction_run: PredictionRun | None = None,
    scores: list[dict[str, Any]] | None = None,
    models_dir: Path | None = None,
    output_dir: Path | None = None,
    competitions: Mapping[str, int] | None = None,
    refresh_errors: Sequence[Mapping[str, Any]] = (),
    now: datetime | None = None,
) -> dict[str, Path]:
    if days <= 0:
        raise ValueError("导出预测天数必须大于 0")
    settings = load_settings()
    active_now = _utc(now)
    active_models_dir = Path(models_dir or settings.model_dir)
    active_output_dir = Path(output_dir or settings.site_data_dir)
    active_output_dir.mkdir(parents=True, exist_ok=True)

    run = prediction_run or _generate_prediction_run(
        active_models_dir,
        days,
        now=active_now,
        competitions=competitions,
    )
    if prediction_run is not None and now is None:
        try:
            active_now = _utc(datetime.fromisoformat(run.generated_at.replace("Z", "+00:00")))
        except ValueError:
            pass
    generated_at = active_now.isoformat()
    predictions = list(run.predictions)
    errors = list(run.errors)
    model_refresh_errors = [dict(item) for item in refresh_errors]
    if scores is None:
        score_result = fetch_daily_scores_with_diagnostics(active_now)
        score_rows = list(score_result.scores)
        score_errors = list(score_result.errors)
    else:
        score_rows = list(scores)
        score_errors = []

    prediction_payload = {
        "schema": "SoccerAnalytics.Predictions.v3",
        "generated_at": generated_at,
        "predictions": predictions,
        "errors": errors,
        "refresh_errors": model_refresh_errors,
        "models_loaded": run.models_loaded,
        "competitions_requested": run.competitions_requested,
        "model_reports": list(run.model_reports),
    }
    scores_payload = {
        "schema": "SoccerAnalytics.Scores.v2",
        "generated_at": generated_at,
        "scores": score_rows,
        "errors": score_errors,
    }
    operations_payload = build_operations_summary(
        predictions,
        score_rows,
        prediction_errors=errors,
        score_errors=score_errors,
        model_refresh_errors=model_refresh_errors,
        model_reports=run.model_reports,
        models_loaded=run.models_loaded,
        competitions_requested=run.competitions_requested,
        generated_at=generated_at,
    )
    release_payload = build_release_governance(
        predictions,
        operations_payload,
        generated_at=generated_at,
    )
    preset_payload = build_preset_questions(
        predictions,
        now=active_now,
        can_publish_recommendations=(
            release_payload.get("can_publish_recommendations") is True
        ),
    )

    outputs = {
        "predictions": active_output_dir / "predictions.json",
        "preset_questions": active_output_dir / "preset_questions.json",
        "scores": active_output_dir / "scores.json",
        "operations": active_output_dir / "operations.json",
        "release_governance": active_output_dir / "release.json",
    }
    for name, payload in (
        ("predictions", prediction_payload),
        ("preset_questions", preset_payload),
        ("scores", scores_payload),
        ("operations", operations_payload),
        ("release_governance", release_payload),
    ):
        _atomic_write_json(outputs[name], payload)

    manifest_path = active_output_dir / "manifest.json"
    manifest_payload = _manifest(
        outputs=outputs,
        generated_at=generated_at,
        predictions=predictions,
        errors=errors,
        refresh_errors=model_refresh_errors,
        score_errors=score_errors,
        models_loaded=run.models_loaded,
        competitions_requested=run.competitions_requested,
    )
    # 清单是提交标记，必须在其余文件全部原子替换成功后最后写入。
    _atomic_write_json(manifest_path, manifest_payload)
    return {"manifest": manifest_path, **outputs}
