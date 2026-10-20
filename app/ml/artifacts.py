from __future__ import annotations

import hashlib
import math
import os
import pickle
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.ml.model import XGBoostClassifier
from app.ml.feature_engineering import FeatureEngineeringError, validate_team_states

MODEL_SCHEMA_VERSION = 3


class ModelArtifactError(RuntimeError):
    """模型资产不完整、损坏或与当前代码不兼容。"""


@dataclass
class ModelBundle:
    """把推理必需的模型、特征语义和历史状态绑定为单个版本化资产。"""

    competition_code: str
    competition_id: int
    estimator: Any
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    trained_at: str
    metrics: dict[str, Any]
    team_states: dict[str, dict[str, Any]]
    feature_defaults: dict[str, float]
    training_rows: int
    schema_version: int = MODEL_SCHEMA_VERSION


@dataclass(frozen=True)
class LoadedModelBundle:
    bundle: ModelBundle
    source_path: Path
    recovered_from_previous: bool
    artifact_sha256: str


def previous_model_path(path: Path) -> Path:
    """把当前模型路径稳定映射为唯一的上一版模型路径。"""
    target = Path(path)
    if target.suffix:
        return target.with_name(f"{target.stem}.previous{target.suffix}")
    return target.with_name(f"{target.name}.previous")


def validate_model_bundle(bundle: ModelBundle) -> None:
    """一次收集全部契约问题，避免修复一个字段后才发现下一个。"""
    if not isinstance(bundle, ModelBundle):
        raise ModelArtifactError("模型文件不是 ModelBundle，可能是旧版裸模型")

    required_fields = (
        "competition_code",
        "competition_id",
        "estimator",
        "feature_names",
        "classes",
        "trained_at",
        "metrics",
        "team_states",
        "feature_defaults",
        "training_rows",
        "schema_version",
    )
    missing_fields = [name for name in required_fields if not hasattr(bundle, name)]
    if missing_fields:
        raise ModelArtifactError(
            "模型包不兼容，缺少字段: " + ", ".join(missing_fields)
        )

    issues: list[str] = []
    if bundle.schema_version != MODEL_SCHEMA_VERSION:
        issues.append(
            f"schema_version 应为 {MODEL_SCHEMA_VERSION}，实际为 {bundle.schema_version}"
        )

    if (
        not isinstance(bundle.competition_code, str)
        or not bundle.competition_code
        or bundle.competition_code != bundle.competition_code.strip().upper()
    ):
        issues.append("competition_code 必须是非空、无空白的大写代码")
    if (
        isinstance(bundle.competition_id, bool)
        or not isinstance(bundle.competition_id, int)
        or bundle.competition_id <= 0
    ):
        issues.append("competition_id 必须是正整数")

    feature_names = bundle.feature_names
    validated_feature_names: tuple[str, ...] = ()
    if (
        not isinstance(feature_names, tuple)
        or not feature_names
        or any(not isinstance(name, str) or not name.strip() for name in feature_names)
    ):
        issues.append("feature_names 必须是非空字符串元组")
    elif len(set(feature_names)) != len(feature_names):
        issues.append("feature_names 不能包含重复字段")
    else:
        validated_feature_names = feature_names

    if bundle.classes != ("L", "D", "W"):
        issues.append('classes 必须严格为 ("L", "D", "W")')

    trained_at: datetime | None = None
    try:
        trained_at = datetime.fromisoformat(bundle.trained_at)
        if trained_at.tzinfo is None or trained_at.utcoffset() is None:
            raise ValueError("缺少时区")
    except (TypeError, ValueError):
        trained_at = None
        issues.append("trained_at 必须是带时区的 ISO8601 时间")

    if (
        isinstance(bundle.training_rows, bool)
        or not isinstance(bundle.training_rows, int)
        or bundle.training_rows <= 0
    ):
        issues.append("training_rows 必须是正整数")

    if not isinstance(bundle.feature_defaults, dict):
        issues.append("feature_defaults 必须是字典")
    else:
        missing_defaults = [
            name
            for name in validated_feature_names
            if name not in bundle.feature_defaults
        ]
        unexpected_defaults = [
            key
            for key in bundle.feature_defaults
            if not isinstance(key, str) or key not in validated_feature_names
        ]
        invalid_defaults = [
            name
            for name, value in bundle.feature_defaults.items()
            if isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ]
        if missing_defaults or unexpected_defaults or invalid_defaults:
            details = []
            if missing_defaults:
                details.append("缺失 " + ", ".join(missing_defaults))
            if invalid_defaults:
                details.append(
                    "非有限数值 "
                    + ", ".join(str(name) for name in invalid_defaults)
                )
            if unexpected_defaults:
                details.append(
                    "非特征字段 "
                    + ", ".join(str(name) for name in unexpected_defaults)
                )
            issues.append("feature_defaults 无效: " + "；".join(details))

    if not isinstance(bundle.team_states, dict):
        issues.append("team_states 必须是字典")
    else:
        try:
            validated_states = validate_team_states(bundle.team_states)
            if trained_at is not None and any(
                state.last_match_at is not None
                and state.last_match_at > trained_at
                for state in validated_states.values()
            ):
                raise FeatureEngineeringError("球队状态时间不能晚于模型训练时间")
        except FeatureEngineeringError as exc:
            issues.append(f"team_states 无效: {exc}")
    if not isinstance(bundle.metrics, dict):
        issues.append("metrics 必须是字典")

    if not callable(getattr(bundle.estimator, "predict", None)):
        issues.append("estimator 缺少 predict")
    if not callable(getattr(bundle.estimator, "predict_proba", None)):
        issues.append("estimator 缺少 predict_proba")
    if isinstance(bundle.estimator, XGBoostClassifier):
        estimator = bundle.estimator
        feature_count = len(validated_feature_names)
        structure_valid = (
            estimator.is_fitted_ is True
            and estimator.feature_names == validated_feature_names
            and estimator.booster_ is not None
            and estimator.rounds_used_ > 0
            and estimator.rounds_used_ <= estimator.rounds
        )
        if structure_valid:
            try:
                structure_valid = estimator.booster_.num_features() == feature_count
            except Exception:
                structure_valid = False
        if not structure_valid:
            issues.append("XGBoostClassifier 训练状态、轮数或特征维度非法")

    if (
        callable(getattr(bundle.estimator, "predict_proba", None))
        and validated_feature_names
        and isinstance(bundle.feature_defaults, dict)
        and not ({*validated_feature_names} - set(bundle.feature_defaults))
    ):
        try:
            smoke_rows = bundle.estimator.predict_proba(
                [{name: bundle.feature_defaults[name] for name in validated_feature_names}]
            )
            if not isinstance(smoke_rows, (list, tuple)) or len(smoke_rows) != 1:
                raise ValueError("概率行数量非法")
            smoke_probabilities = smoke_rows[0]
            if not isinstance(smoke_probabilities, (list, tuple)) or len(smoke_probabilities) != 3:
                raise ValueError("概率类别数量非法")
            normalized = [float(value) for value in smoke_probabilities]
            if (
                any(not math.isfinite(value) or value < 0.0 for value in normalized)
                or math.fsum(normalized) <= 0.0
            ):
                raise ValueError("概率包含非法数值")
        except Exception as exc:
            issues.append(
                f"estimator 无法使用默认特征完成推理: {type(exc).__name__}: {exc}"
            )

    if issues:
        raise ModelArtifactError("模型包校验失败: " + "；".join(issues))


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            # 模型仍保持 0600，但容器写挂载卷时归属于模型目录所有者。
            try:
                parent = target.parent.stat()
                os.fchown(handle.fileno(), parent.st_uid, parent.st_gid)
            except (AttributeError, OSError):
                pass
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_model_bundle(bundle: ModelBundle, path: Path) -> Path:
    """原子保存当前模型；仅用已验证的当前模型刷新上一版。"""
    validate_model_bundle(bundle)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
        if target.exists():
            try:
                load_model_bundle(target)
            except ModelArtifactError:
                # 损坏的当前文件不能污染仍可恢复的上一版。
                pass
            else:
                _atomic_write_bytes(previous_model_path(target), target.read_bytes())
        _atomic_write_bytes(target, payload)
    except Exception as exc:
        raise ModelArtifactError(f"模型包写入失败: {target}") from exc

    return target


def _load_model_bundle_with_digest(path: Path) -> tuple[ModelBundle, str]:
    target = Path(path)
    if not target.exists():
        raise ModelArtifactError(f"模型包不存在: {target}")

    try:
        # This remains an executable, trusted-local format. Check storage before
        # unpickling; a digest/schema check cannot authenticate pickle contents.
        parent_status = target.parent.stat()
        if parent_status.st_mode & 0o022:
            raise ModelArtifactError("模型目录不能允许其他用户写入")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        if target.is_symlink():
            raise ModelArtifactError("模型文件不能是符号链接")
        with os.fdopen(os.open(target, flags), "rb") as handle:
            status = os.fstat(handle.fileno())
            if not stat.S_ISREG(status.st_mode) or status.st_mode & 0o022:
                raise ModelArtifactError("模型必须是其他用户不可写的普通文件")
            if status.st_uid != parent_status.st_uid:
                raise ModelArtifactError("模型文件所有者必须与可信模型目录一致")
            payload = handle.read()
        loaded = pickle.loads(payload)
    except ModelArtifactError:
        raise
    except Exception as exc:
        # pickle 钩子可能抛出任意运行时异常；统一包装后才能安全尝试上一版。
        raise ModelArtifactError(f"模型包无法读取: {target}") from exc

    if not isinstance(loaded, ModelBundle):
        raise ModelArtifactError("模型文件不是 ModelBundle，旧版裸模型必须重新训练")
    try:
        validate_model_bundle(loaded)
    except ModelArtifactError:
        raise
    except Exception as exc:
        # 任意语义损坏都统一为领域错误，确保上一版回退一定会执行。
        raise ModelArtifactError(f"模型包校验异常: {target}") from exc
    return loaded, hashlib.sha256(payload).hexdigest()


def load_model_bundle(path: Path) -> ModelBundle:
    """从一次不可变字节快照反序列化并验证模型包。"""
    return _load_model_bundle_with_digest(path)[0]


def load_model_bundle_with_fallback(path: Path) -> LoadedModelBundle:
    """优先读取当前模型，损坏或缺失时显式回退到上一版。"""
    target = Path(path)
    try:
        bundle, artifact_sha256 = _load_model_bundle_with_digest(target)
    except ModelArtifactError as active_error:
        previous = previous_model_path(target)
        try:
            bundle, artifact_sha256 = _load_model_bundle_with_digest(previous)
        except ModelArtifactError as previous_error:
            raise ModelArtifactError(
                f"当前模型不可用 ({active_error})；上一版也不可用 ({previous_error})"
            ) from previous_error
        return LoadedModelBundle(
            bundle=bundle,
            source_path=previous,
            recovered_from_previous=True,
            artifact_sha256=artifact_sha256,
        )
    return LoadedModelBundle(
        bundle=bundle,
        source_path=target,
        recovered_from_previous=False,
        artifact_sha256=artifact_sha256,
    )
