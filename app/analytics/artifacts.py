"""Shared receipts, verified JSON caches and warehouse artifact storage."""

import hashlib
import json
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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
    path = Path(path)
    receipt = path.with_suffix(".receipt.json")
    if path.exists() and receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["inputs"] == signature and previous[
            "content_sha256"
        ] == file_digest(path):
            return json.loads(path.read_text())
    value = build()
    path.write_text(json.dumps(value))
    receipt.write_text(
        json.dumps({"inputs": signature, "content_sha256": file_digest(path)}, indent=2)
    )
    return value


def archive_outputs(db, dataset_name, output, kind, edition, *, filenames=None):
    """Archive all supported outputs, or a fully validated explicit selection.

    Explicit names are relative to output. Validate the whole selection before
    creating a dataset or writing any artifact so partial selections fail closed.
    """
    from app.data_service.build_warehouse import document

    output = Path(output)
    supported = {".json", ".jsonl", ".joblib", ".npz", ".png", ".svg"}
    if filenames is None:
        files = [
            file for file in sorted(output.rglob("*"))
            if file.is_file() and file.suffix in supported
        ]
    else:
        if isinstance(filenames, (str, bytes, Path)):
            raise ValueError("filenames must be a collection of relative file names")
        files, seen = [], set()
        root = output.resolve()
        for name in filenames:
            if not isinstance(name, (str, Path)):
                raise ValueError("Artifact names must be relative paths")
            relative = Path(name)
            file = output / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative in seen
                or relative.suffix not in supported
                or not file.is_file()
                or not file.resolve().is_relative_to(root)
            ):
                raise ValueError(f"Missing, duplicate or unsafe artifact selection: {name}")
            seen.add(relative)
            files.append(file)
        files.sort()

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
