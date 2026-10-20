"""Official public Figshare soccer-log archive and bounded JSON-array ingestion."""

import argparse
import hashlib
import json
import re
from pathlib import Path, PureWindowsPath
import tempfile
import zipfile
import requests
from .warehouse import Warehouse


def json_items(stream):
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    finished = False
    needs_value = True
    had_items = False
    while True:
        chunk = stream.read(65536)
        eof = not chunk
        buffer = buffer[position:] + chunk
        position = 0
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    break
                if buffer[position] != "[":
                    raise ValueError("Expected a JSON array")
                started = True
                position += 1
                continue
            if finished:
                if buffer[position:].strip():
                    raise ValueError("Trailing JSON content")
                break
            if position >= len(buffer):
                break
            if buffer[position] == "]":
                if needs_value and had_items:
                    raise ValueError("Trailing array comma")
                finished = True
                position += 1
                continue
            try:
                item, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                break
            # Require a full separator before yielding an item at a buffer boundary.
            after = end
            while after < len(buffer) and buffer[after].isspace():
                after += 1
            if after == len(buffer) and not eof:
                break
            if after >= len(buffer) or buffer[after] not in ",]":
                partial_number = r"-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+-]?\d*)?"
                if (
                    not eof
                    and after == end
                    and isinstance(item, (int, float))
                    and re.fullmatch(partial_number, buffer[position:])
                ):
                    break
                raise ValueError("Missing array separator")
            yield item
            had_items = True
            if buffer[after] == ",":
                position = after + 1
                needs_value = True
            else:
                position = after
                needs_value = False
                continue
        if eof:
            if not finished:
                raise ValueError("Truncated JSON array")
            return


def download_path(root, name):
    windows = PureWindowsPath(name) if isinstance(name, str) else None
    if (
        not isinstance(name, str)
        or not name
        or name in {".", "..", "manifest.json"}
        or any(c in name for c in "/\\\x00:")
        or name.endswith((".", " "))
        or windows.is_reserved()
    ):
        raise ValueError("Figshare filename must be a plain filename")
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Figshare destination must remain inside the collection root")
    if path.exists() and not path.is_file():
        raise ValueError("Figshare destination is not a regular file")
    return path


def write_download(path, chunks, expected=None):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            digest = hashlib.md5()
            for chunk in chunks:
                output.write(chunk)
                digest.update(chunk)
        if expected and digest.hexdigest() != expected:
            raise ValueError(f"Figshare digest mismatch: {path}")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def collect(root=".tmp/open-data/wyscout"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    manifest = root / "manifest.json"
    if manifest.is_symlink():
        raise ValueError("Figshare manifest cannot be a symlink")
    metadata = []
    with requests.Session() as session:
        page = 1
        while True:
            response = session.get(
                "https://api.figshare.com/v2/collections/4415000/articles",
                params={"page": page, "page_size": 100},
                timeout=60,
            )
            response.raise_for_status()
            articles = response.json()
            if not articles:
                break
            for article in articles:
                response = session.get(
                    f'https://api.figshare.com/v2/articles/{article["id"]}', timeout=60
                )
                response.raise_for_status()
                details = response.json()
                metadata.append(details)
                for file in details.get("files", []):
                    path = download_path(root, file["name"])
                    expected = file.get("computed_md5")
                    if not path.exists():
                        url = (
                            file.get("download_url")
                            or f'https://figshare.com/ndownloader/files/{file["id"]}'
                        )
                        with session.get(
                            url, stream=True, timeout=(20, 180)
                        ) as download:
                            download.raise_for_status()
                            write_download(
                                path, download.iter_content(1024 * 1024), expected
                            )
                    digest = hashlib.md5()
                    with path.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                    if expected and digest.hexdigest() != expected:
                        raise ValueError(f"Figshare digest mismatch: {path}")
                    print("Collected", path.name, path.stat().st_size, flush=True)
            page += 1
    write_download(manifest, [json.dumps(metadata, indent=2).encode("utf-8")])


def ingest(database="data/football.sqlite3", root=".tmp/open-data/wyscout"):
    root = Path(root)
    with Warehouse(database) as db:
        dataset = db.dataset(
            "wyscout",
            "real_events",
            "wyscout",
            edition="Public 2017/18 leagues, Euro 2016, World Cup 2018",
        )
        db.connection.executescript("""
        CREATE TABLE IF NOT EXISTS wyscout_events(asset_id INTEGER REFERENCES assets(id),row_number INTEGER,event_id INTEGER,match_id INTEGER,player_id INTEGER,team_id INTEGER,period TEXT,seconds REAL,event_type TEXT,payload TEXT,PRIMARY KEY(asset_id,row_number));
        CREATE TABLE IF NOT EXISTS wyscout_entities(asset_id INTEGER REFERENCES assets(id),row_number INTEGER,entity_type TEXT,entity_id INTEGER,payload TEXT,PRIMARY KEY(asset_id,row_number));
        CREATE INDEX IF NOT EXISTS wyscout_event_match ON wyscout_events(match_id,player_id);
        """)
        from .build_warehouse import document
        import io

        for source in sorted(
            p for p in root.iterdir() if p.is_file() and not p.name.endswith(".part")
        ):
            aid = db.asset(dataset, source, source.name)
            if source.suffix == ".zip":
                with zipfile.ZipFile(source) as archive:
                    for member in archive.namelist():
                        if not member.endswith(".json") or "__MACOSX" in member:
                            continue
                        with archive.open(member) as stream:
                            member_id = db.asset_stream(
                                dataset, stream, source.name + "/" + member
                            )
                        with archive.open(member) as stream:
                            ingest_array(
                                db,
                                member_id,
                                io.TextIOWrapper(stream, encoding="utf-8"),
                                Path(member).stem,
                            )
            elif source.suffix == ".json" and source.name != "manifest.json":
                try:
                    with source.open(encoding="utf-8") as stream:
                        ingest_array(db, aid, stream, source.stem)
                except (ValueError, json.JSONDecodeError) as exc:
                    if source.stem != "referees":
                        raise
                    db.connection.execute(
                        "DELETE FROM wyscout_entities WHERE asset_id=?", (aid,)
                    )
                    db.connection.execute(
                        "INSERT OR REPLACE INTO quarantine VALUES(?,?)", (aid, str(exc))
                    )
                    db.finish(aid, "wyscout_referees", 0)
                    print("Quarantined optional referees:", exc, flush=True)
            elif source.suffix == ".csv":
                db.csv_table(aid, source, "wyscout_" + source.stem)
            elif source.name == "manifest.json":
                document(db, aid, source, "wyscout_provenance")
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def ingest_array(db, aid, stream, kind):
    if db.done(aid, "wyscout_" + kind):
        return
    events = kind.startswith("events")
    batch = []
    count = 0
    for count, item in enumerate(json_items(stream), 1):
        text = json.dumps(item, separators=(",", ":"))
        if events:
            batch.append(
                (
                    aid,
                    count,
                    item.get("id"),
                    item.get("matchId"),
                    item.get("playerId"),
                    item.get("teamId"),
                    item.get("matchPeriod"),
                    item.get("eventSec"),
                    item.get("eventName"),
                    text,
                )
            )
            sql = "INSERT OR REPLACE INTO wyscout_events VALUES(?,?,?,?,?,?,?,?,?,?)"
        else:
            batch.append(
                (
                    aid,
                    count,
                    kind,
                    item.get("wyId") if isinstance(item, dict) else None,
                    text,
                )
            )
            sql = "INSERT OR REPLACE INTO wyscout_entities VALUES(?,?,?,?,?)"
        if len(batch) >= 5000:
            db.connection.executemany(sql, batch)
            db.connection.commit()
            batch = []
    if batch:
        db.connection.executemany(sql, batch)
    db.finish(aid, "wyscout_" + kind, count)
    print("Imported", kind, count, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["collect", "ingest"])
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument("--sources", default=".tmp/open-data/wyscout")
    args = parser.parse_args()
    (
        collect(args.sources)
        if args.command == "collect"
        else ingest(args.database, args.sources)
    )
