"""Atomic receipts and public artifacts shared by collectors and exports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    write_bytes_atomic(path, json_bytes(payload))


def write_bytes_atomic(path: Path, content: bytes) -> None:
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
            handle.write(content)
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
