"""Pinned, checksummed public data archives, including Git LFS payloads."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import requests

from app.file_io import write_json_atomic as _atomic_write_json
from .fetch.retry import RETRYABLE_STATUS_CODES, RetryPolicy, close_response

logger = logging.getLogger(__name__)
REPOSITORIES = {
    "statsbomb": "hudl/open-data",
    "metrica": "metrica-sports/sample-data",
    "skillcorner": "SkillCorner/opendata",
    "openfootball": "openfootball/football.json",
    "jokecamp": "jokecamp/FootballData",
    "eddwebster": "eddwebster/football_analytics",
    "international-results": "martj42/international_results",
}


def _selected(dataset: str, path: str) -> bool:
    if path in {"README.md", "LICENSE", "LICENSE.md", "LICENSE.pdf"}:
        return True
    if dataset == "openfootball":
        return bool(re.fullmatch(r"\d{4}(?:-\d{2})?/[^/]+\.json", path))
    if dataset == "international-results":
        return path in {
            "results.csv",
            "goalscorers.csv",
            "shootouts.csv",
            "former_names.csv",
        }
    if dataset in {"jokecamp", "eddwebster"}:
        return (dataset == "jokecamp" or path.startswith("data/")) and PurePosixPath(
            path
        ).suffix.lower() in {".csv", ".json", ".xls", ".xlsx"}
    return path.startswith("data/")


def _request(session, url: str, *, stream=False, before_request=None, retry_policy=None, sleep=None):
    policy = retry_policy if retry_policy is not None else RetryPolicy(backoff_factor=1.)
    sleep = sleep if sleep is not None else time.sleep
    for attempt in range(4):
        try:
            if before_request is not None:
                before_request()
            response = session.get(url, timeout=(20, 120), stream=stream)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            response = exc.response
            close_response(response)
            if response is not None and response.status_code not in RETRYABLE_STATUS_CODES:
                raise
            if attempt == 3:
                # Exhaustion preserves requests' exception type; expose the hint without waiting.
                exc.retry_after_seconds = policy.retry_after(response)
                raise
            sleep(policy.delay(response, attempt))
    raise AssertionError("unreachable")


def extract_archive(
    stream, root: Path, tree: dict, *, lfs_fetch=None, progress=None
) -> list[dict]:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = []
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if ".." in parts or member.name.startswith("/"):
                raise ValueError("Archive path traversal")
            if len(parts) < 2 or not member.isfile():
                continue
            relative = "/".join(parts[1:])
            if relative not in tree:
                continue
            entry = tree[relative]
            if member.size != entry["size"]:
                raise ValueError(f"Archive size mismatch: {relative}")
            target = (root / relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Archive target escaped root")
            target.parent.mkdir(parents=True, exist_ok=True)
            sha256 = hashlib.sha256()
            git_digest = hashlib.sha1(
                f"blob {member.size}\0".encode(), usedforsecurity=False
            )
            source = archive.extractfile(member)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    while chunk := source.read(1024 * 1024):
                        handle.write(chunk)
                        sha256.update(chunk)
                        git_digest.update(chunk)
                if git_digest.hexdigest() != entry["sha"]:
                    raise ValueError(f"Git blob checksum mismatch: {relative}")
                size = member.size
                lfs = None
                if size < 256:
                    text = temporary.read_text(encoding="utf-8", errors="ignore")
                    if text.startswith("version https://git-lfs.github.com/spec/v1"):
                        oid = re.search(r"oid sha256:([0-9a-f]{64})", text)
                        length = re.search(r"size (\d+)", text)
                        if oid is None or length is None or lfs_fetch is None:
                            raise ValueError(f"Unresolved Git LFS pointer: {relative}")
                        lfs = {"sha256": oid[1], "size": int(length[1])}
                        if lfs["size"] > shutil.disk_usage(root).free - 2 * 1024**3:
                            raise ValueError(
                                f"Insufficient disk for LFS payload: {relative}"
                            )
                        sha256 = hashlib.sha256()
                        size = 0
                        with temporary.open("wb") as handle:
                            for chunk in lfs_fetch(relative):
                                handle.write(chunk)
                                sha256.update(chunk)
                                size += len(chunk)
                        if size != lfs["size"] or sha256.hexdigest() != lfs["sha256"]:
                            raise ValueError(f"LFS checksum mismatch: {relative}")
                os.replace(temporary, target)
                temporary = None
                files.append(
                    {
                        "path": relative,
                        "bytes": size,
                        "sha256": sha256.hexdigest(),
                        "git_blob_sha": entry["sha"],
                        "lfs": lfs,
                    }
                )
                if progress and (len(files) % 100 == 0 or lfs is not None):
                    progress(files)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    if set(tree) != {entry["path"] for entry in files}:
        raise ValueError("Archive missing selected tree entries")
    return files


def collect_repository(
    dataset: str, root: Path, *, tree_payload=None, refresh=False
) -> dict:
    if dataset not in REPOSITORIES:
        raise ValueError(f"Unknown public dataset: {dataset}")
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not refresh:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete") and all(
            (root / item["path"]).is_file()
            and (root / item["path"]).stat().st_size == item["bytes"]
            for item in manifest["files"]
        ):
            return manifest
    session = requests.Session()
    repo = REPOSITORIES[dataset]
    if tree_payload is None:
        response = _request(
            session, f"https://api.github.com/repos/{repo}/git/trees/master?recursive=1"
        )
        tree_payload = response.json()
    if tree_payload.get("truncated"):
        raise ValueError("GitHub tree is truncated; refusing an incomplete inventory")
    tree = {
        item["path"]: item
        for item in tree_payload["tree"]
        if item["type"] == "blob" and _selected(dataset, item["path"])
    }
    revision = tree_payload["sha"]
    url = f"https://codeload.github.com/{repo}/tar.gz/{revision}"
    metadata = {
        "dataset": dataset,
        "repository": f"https://github.com/{repo}",
        "revision": revision,
        "archive_url": url,
        "fetched_at": datetime.now(UTC).isoformat(),
        "attribution": {
            "statsbomb": "StatsBomb Open Data",
            "metrica": "Metrica Sports",
            "skillcorner": "SkillCorner",
            "openfootball": "OpenFootball",
            "jokecamp": "jokecamp FootballData community collection",
            "eddwebster": "Edd Webster football_analytics community snapshots",
            "international-results": "Mart Jürisoo international football results",
        }[dataset],
    }
    root.mkdir(parents=True, exist_ok=True)

    def lfs_fetch(path):
        media = f"https://media.githubusercontent.com/media/{repo}/{revision}/{path}"
        logger.info("%s LFS: %s", dataset, path)
        response = _request(session, media, stream=True)
        try:
            yield from response.iter_content(1024 * 1024)
        finally:
            response.close()

    def progress(files):
        logger.info(
            "%s: %d/%d assets, %.2f GB",
            dataset,
            len(files),
            len(tree),
            sum(item["bytes"] for item in files) / 1e9,
        )
        _atomic_write_json(
            root / "manifest.partial.json",
            {**metadata, "complete": False, "files": files},
        )

    response = _request(session, url, stream=True)
    try:
        files = extract_archive(
            response.raw, root, tree, lfs_fetch=lfs_fetch, progress=progress
        )
    finally:
        response.close()
    manifest = {
        **metadata,
        "complete": True,
        "files": files,
        "bytes": sum(item["bytes"] for item in files),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def read_asset_json(root: Path, relative: str, index: dict | None = None):
    root = Path(root)
    if index is None:
        manifest = json.loads((root / "manifest.json").read_text())
        index = {entry["path"]: entry for entry in manifest["files"]}
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or relative not in index:
        raise ValueError("Asset is outside the verified inventory")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != index[relative]["sha256"]:
        raise ValueError(f"Asset checksum mismatch: {relative}")
    try:
        import orjson
    except ImportError:
        return json.loads(payload)
    return orjson.loads(payload)


def verify_asset_file(root: Path, relative: str, index: dict) -> Path:
    path = (Path(root) / relative).resolve()
    if relative not in index or not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Asset is outside inventory")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != index[relative]["sha256"]:
        raise ValueError(f"Asset checksum mismatch: {relative}")
    return path
