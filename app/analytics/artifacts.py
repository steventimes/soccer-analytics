"""Shared receipts, verified JSON caches and warehouse artifact storage."""

import hashlib
import json
from pathlib import Path


def file_digest(path):
    """Hash original logical bytes, including a packed fallback representation."""
    from app.data_service.archive import open_asset_file

    digest = hashlib.sha256()
    with open_asset_file(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_signature(connection, providers):
    placeholders = ",".join("?" for _ in providers)
    rows = connection.execute(
        f"SELECT d.name,a.path,a.sha256,a.bytes FROM assets a JOIN datasets d ON d.id=a.dataset_id "
        f"WHERE d.provider IN ({placeholders}) AND a.complete=1 ORDER BY d.name,a.path",
        providers,
    )
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(json.dumps(list(row), ensure_ascii=False).encode())
        count += 1
    return {
        "providers": list(providers),
        "complete_assets": count,
        "sha256": digest.hexdigest(),
    }


def cached_json(path, signature, build):
    from app.data_service.archive import open_asset_file, resolve_asset_path
    from app.file_io import write_bytes_atomic

    path = Path(path)
    receipt = path.with_suffix(".receipt.json")
    if resolve_asset_path(path).exists() and receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["inputs"] == signature and previous[
            "content_sha256"
        ] == file_digest(path):
            with open_asset_file(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
    value = build()
    write_bytes_atomic(path, json.dumps(value).encode("utf-8"))
    write_bytes_atomic(
        receipt,
        json.dumps({"inputs": signature, "content_sha256": file_digest(path)}, indent=2).encode("utf-8")
    )
    return value


def archive_outputs(db, dataset_name, output, kind, edition, *, filenames=None):
    """Archive all supported outputs, or a fully validated explicit selection.

    Explicit names are relative to output. Validate the whole selection before
    creating a dataset or writing any artifact so partial selections fail closed.
    """
    from app.data_service.build_warehouse import document
    from app.data_service.archive import resolve_asset_path

    output = Path(output)
    supported = {".json", ".jsonl", ".joblib", ".npz", ".png", ".svg"}
    root = output.resolve()
    if filenames is None:
        candidates = set()
        for physical in output.rglob("*"):
            if not physical.is_file():
                continue
            logical = physical.with_suffix("") if physical.suffix == ".gz" else physical
            if logical.suffix in supported:
                candidates.add(logical)
        files = sorted(candidates)
    else:
        if isinstance(filenames, (str, bytes, Path)):
            raise ValueError("filenames must be a collection of relative file names")
        files, seen = [], set()
        for name in filenames:
            if not isinstance(name, (str, Path)):
                raise ValueError("Artifact names must be relative paths")
            relative = Path(name)
            if (relative.is_absolute() or ".." in relative.parts or relative in seen
                    or relative.suffix not in supported):
                raise ValueError(f"Missing, duplicate or unsafe artifact selection: {name}")
            seen.add(relative)
            files.append(output / relative)
        files.sort()
    # Validate the physical representation before creating any archive records.
    # Explicit .gz names remain unsupported; only logical output names use fallback.
    for file in files:
        physical = resolve_asset_path(file)
        if (not physical.is_file() or not physical.resolve().is_relative_to(root)
                or (physical != file and physical.stat().st_nlink != 1)):
            raise ValueError(f"Missing, duplicate or unsafe artifact selection: {file.relative_to(output)}")

    dataset = db.dataset(dataset_name, "derived", "soccer_analytics", edition)
    db.connection.execute(
        "CREATE TABLE IF NOT EXISTS analysis_run_assets(run_kind TEXT,run_id TEXT,asset_id INTEGER REFERENCES assets(id),PRIMARY KEY(run_kind,run_id,asset_id))"
    )
    for file in files:
        aid = db.asset(dataset, file, str(file.relative_to(output)))
        db.connection.execute(
            "INSERT OR IGNORE INTO analysis_run_assets VALUES(?,?,?)",
            (kind, edition, aid),
        )
        if file.suffix == ".json":
            document(db, aid, file, kind)
