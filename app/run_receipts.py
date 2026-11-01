"""Small local run identities shared by training and match-study commands."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def content_digest(value: Any) -> str:
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":"))
    digest = hashlib.sha256()
    for part in encoder.iterencode(value):
        digest.update(part.encode())
    return digest.hexdigest()


def estimator_parameters(estimator):
    parameters = {name: value for name, value in getattr(estimator, "__dict__", {}).items()
                  if not name.endswith("_") and name != "feature_names"
                  and isinstance(value, (str, int, float, bool, type(None)))}
    pipeline = getattr(estimator, "pipeline", None)
    if callable(getattr(pipeline, "get_params", None)):
        for name, value in pipeline.get_params().items():
            try:
                content_digest(value)
            except (TypeError, ValueError):
                continue  # Pipeline objects themselves are not configuration JSON.
            parameters[name] = value
    return parameters


def build_run_receipt(command, *, configuration, inputs, stages):
    """Hash only application source and supplied identities, never the warehouse."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / "app").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              text=True, capture_output=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = None
    packages = {}
    for name in ("numpy", "scipy", "xgboost-cpu", "scikit-learn", "torch", "optuna",
                 "lightgbm", "catboost", "scikit-optimize", "fpstreams", "interpret-core"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    body = {"schema": "SoccerAnalytics.RunReceipt.v1", "command": command,
            "code": {"git_head": head, "application_sha256": digest.hexdigest()},
            "environment": {"python": platform.python_version(), "packages": packages},
            "configuration": configuration, "inputs": inputs, "stages": stages}
    return {**body, "run_id": content_digest(body)}
