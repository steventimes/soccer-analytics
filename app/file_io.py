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


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def write_bytes_atomic(path: Path, content: bytes, *, file_mode: int = 0o644) -> None:
    """同目录落临时文件再替换；失败时旧制品保持原样。"""
    if type(file_mode) is not int or not 0 <= file_mode <= 0o777:
        raise ValueError("file_mode must contain only ordinary permission bits")
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
            # Public exports use 0644; trusted model bundles explicitly use 0600.
            os.fchmod(handle.fileno(), file_mode)
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        # 目录 fsync 让 rename 在突然断电后也具备持久性；不支持时文件原子性仍成立。
        _sync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_files_recoverable(root: Path, files: Mapping[str, bytes]) -> None:
    """Stage a batch and restore prior files after an ordinary publish exception.

    Individual replacements are atomic. Concurrent readers and process/power
    loss still require a separate generation protocol at the hosting boundary.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Batch paths must stay below the output directory")
    with tempfile.TemporaryDirectory(prefix=".research-export-", dir=root.parent) as temporary:
        workspace = Path(temporary)
        backups, published = {}, []
        for index, (name, content) in enumerate(files.items()):
            staged = workspace / "new" / name
            write_bytes_atomic(staged, content)
            target = root / name
            backup = workspace / f"previous-{index}"
            if target.is_symlink():
                backup.symlink_to(os.readlink(target))
                backups[name] = backup
            elif target.exists():
                os.link(target, backup)
                backups[name] = backup
        try:
            for name in files:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(workspace / "new" / name, target)
                published.append(name)
            for directory in {(root / name).parent for name in published}:
                _sync_directory(directory)
        except BaseException as original:
            failures = []
            for name in reversed(published):
                try:
                    if name in backups:
                        os.replace(backups[name], root / name)
                    else:
                        (root / name).unlink(missing_ok=True)
                    _sync_directory((root / name).parent)
                except OSError as error:
                    failures.append(f"{name}: {error}")
            if failures:
                recovery = workspace.with_name(workspace.name + "-recovery")
                os.replace(workspace, recovery)
                raise OSError(f"Publication rollback incomplete; backups retained at {recovery}: "
                              + "; ".join(failures)) from original
            raise
