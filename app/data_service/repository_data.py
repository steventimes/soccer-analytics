"""Import community snapshots with checksum lineage and exact mirror deduplication.

Repository visibility does not establish original publisher, measurement grain or
redistribution rights. Raw and engineered tables keep separate provenance.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
from pathlib import Path

from .archive import verify_asset_file
from .build_warehouse import document
from .warehouse import Warehouse

logger = logging.getLogger(__name__)


def csv_schema(path):
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with path.open(encoding=encoding, newline="") as stream:
                sample = stream.read(8192)
                stream.seek(0)
                try:
                    delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
                except csv.Error:
                    delimiter = ","
                headers = next(csv.reader(stream, delimiter=delimiter))
            return (
                headers,
                hashlib.sha256(
                    json.dumps(headers, ensure_ascii=False).encode()
                ).hexdigest()[:16],
            )
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV encoding could not be read")


def source_labels(repo, path):
    parts = Path(path).parts
    if (
        repo == "eddwebster/football_analytics"
        and len(parts) > 1
        and parts[0] == "data"
    ):
        family = parts[1]
        return (
            family,
            {"sb": "statsbomb", "tm": "transfermarkt"}.get(family, family),
            "engineered" if "engineered" in parts else "archived_snapshot",
        )
    family = parts[0].lower().replace(" ", "_") if len(parts) > 1 else "other"
    upstream = (
        "football-data.co.uk"
        if parts[0] == "football-data.co.uk"
        else "mixed_or_unverified"
    )
    return family, upstream, "community_collection"


def ingest(database, roots):
    seen_csv = set()
    mirrors = 0
    payloads = 0
    catalogued = 0
    issues = []
    families = Counter()
    with Warehouse(database) as db:
        db.connection.execute(
            """CREATE TABLE IF NOT EXISTS repository_files(repo TEXT,revision TEXT,path TEXT,asset_id INTEGER REFERENCES assets(id),family TEXT,upstream_provider TEXT,stage TEXT,table_name TEXT,row_count INTEGER,status TEXT,quality_note TEXT,PRIMARY KEY(repo,revision,path))"""
        )
        existing = {
            r["sha256"]: r["id"]
            for r in db.connection.execute(
                "SELECT id,sha256 FROM assets WHERE complete=1 ORDER BY id DESC"
            )
        }
        for root in map(Path, roots):
            manifest = json.loads((root / "manifest.json").read_text())
            if not manifest.get("complete"):
                raise ValueError("Repository collection is incomplete: " + str(root))
            repo = manifest["repository"].removeprefix("https://github.com/")
            revision = manifest["revision"]
            dataset = db.dataset(
                "community:" + repo,
                "mixed_public_snapshots",
                "github_community",
                revision,
            )
            index = {entry["path"]: entry for entry in manifest["files"]}
            for relative, entry in index.items():
                path = verify_asset_file(root, relative, index)
                aid = existing.get(entry["sha256"])
                mirror = aid is not None
                if aid is None:
                    aid = db.asset(dataset, path, relative)
                    existing[entry["sha256"]] = aid
                    payloads += 1
                else:
                    mirrors += 1
                family, upstream, stage = source_labels(repo, relative)
                table = None
                rows = None
                status = "raw_only"
                note = "Original source/coverage must be checked before modeling; no automatic identity merge."
                if path.suffix.lower() == ".csv":
                    headers, schema = csv_schema(path)
                    previous = db.connection.execute(
                        "SELECT kind,rows FROM imports WHERE asset_id=? AND kind!='document' ORDER BY kind LIMIT 1",
                        (aid,),
                    ).fetchone()
                    if (
                        previous
                        and db.connection.execute(
                            "SELECT count(*) FROM csv_columns WHERE asset_id=?", (aid,)
                        ).fetchone()[0]
                        > 0
                    ):
                        table, rows = previous["kind"], previous["rows"]
                    elif len(headers) <= 1900:
                        table = "community_csv_" + schema
                        try:
                            db.csv_table(aid, path, table)
                            rows = db.connection.execute(
                                "SELECT rows FROM imports WHERE asset_id=? AND kind=?",
                                (aid, table),
                            ).fetchone()[0]
                        except (ValueError, csv.Error) as exc:
                            table = None
                            issues.append(
                                {
                                    "repo": repo,
                                    "path": relative,
                                    "error": type(exc).__name__,
                                    "message": str(exc),
                                }
                            )
                    else:
                        note = "More than1900source columns; preserved raw to respect SQLite column limit."
                    if table:
                        seen_csv.add(aid)
                        status = "csv"
                elif path.suffix.lower() == ".json":
                    try:
                        document(db, aid, path, "community_repository_json")
                        status = "json"
                        quarantined = db.connection.execute(
                            "SELECT reason FROM quarantine WHERE asset_id=?", (aid,)
                        ).fetchone()
                        if quarantined:
                            status = "quarantined_json"
                            issues.append(
                                {
                                    "repo": repo,
                                    "path": relative,
                                    "error": quarantined[0],
                                }
                            )
                    except (ValueError, json.JSONDecodeError) as exc:
                        db.connection.execute(
                            "INSERT OR REPLACE INTO quarantine VALUES(?,?)",
                            (aid, str(exc)),
                        )
                        status = "quarantined_json"
                        issues.append(
                            {"repo": repo, "path": relative, "error": str(exc)}
                        )
                elif path.suffix.lower() in {".xls", ".xlsx"}:
                    status = "raw_spreadsheet"
                    note = "Original spreadsheet preserved in compressed database chunks; not treated as parsed CSV or model-ready data."
                if mirror:
                    status = "exact_mirror_" + status
                db.connection.execute(
                    "INSERT INTO repository_files VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(repo,revision,path) DO UPDATE SET asset_id=excluded.asset_id,table_name=excluded.table_name,row_count=excluded.row_count,status=excluded.status,quality_note=excluded.quality_note",
                    (
                        repo,
                        revision,
                        relative,
                        aid,
                        family,
                        upstream,
                        stage,
                        table,
                        rows,
                        status,
                        note,
                    ),
                )
                catalogued += 1
                families[repo + "/" + family] += 1
                if catalogued % 250 == 0:
                    db.connection.commit()
                    logger.info(
                        "Repository files catalogued: %d, new payloads:%d, mirrors:%d",
                        catalogued,
                        payloads,
                        mirrors,
                    )
            aid = db.asset(dataset, root / "manifest.json", "manifest.json")
            document(db, aid, root / "manifest.json", "community_repository_manifest")
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return {
        "catalogued_files": catalogued,
        "new_payloads": payloads,
        "mirrored_files": mirrors,
        "unique_csv_assets": len(seen_csv),
        "files_by_family": dict(families),
        "issues": issues,
        "interpretation": "File/CSV population counts, not unique matches or independent sources. Exact byte mirrors use one canonical asset. Reordered or transformed copies still need semantic review.",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[".tmp/open-data/jokecamp", ".tmp/open-data/eddwebster"],
    )
    parser.add_argument("--output", default=".tmp/source-review/repository-ingest.json")
    args = parser.parse_args()
    report = ingest(args.database, args.sources)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
