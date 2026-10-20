"""Store SoccerData frames, column schemas, access reports and original caches."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .warehouse import Warehouse
from .build_warehouse import document
from .repository_data import csv_schema


def public_path(path, expected):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(expected):
        raise ValueError("SDK asset is outside its public source subtree")
    if resolved.is_file() and resolved.stat().st_nlink != 1:
        raise ValueError("SDK public assets cannot be hard-link aliases")
    return resolved


def ingest(database, root):
    with Warehouse(database) as db:
        result = ingest_into(db, root)
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return result


def ingest_into(db, root):
    root = Path(root).resolve()
    counts = Counter()
    statuses = []
    db.connection.execute(
        """CREATE TABLE IF NOT EXISTS soccerdata_imports(provider TEXT,run_directory TEXT,method TEXT,asset_id INTEGER REFERENCES assets(id),schema_asset_id INTEGER REFERENCES assets(id),table_name TEXT,rows INTEGER,observed_at TEXT,scope TEXT,PRIMARY KEY(provider,run_directory,method,asset_id))"""
    )
    for report_path in sorted(root.glob("*/report.json")):
        folder = report_path.parent
        if folder.resolve() != folder or report_path.resolve() != report_path:
            raise ValueError("SDK run and report cannot be directory/file aliases")
        public_path(report_path, folder)
        report = json.loads(report_path.read_text())
        provider = report["provider"]
        dataset = db.dataset(
            "soccerdata:" + provider,
            "real_source_adapter",
            "soccerdata",
            report["soccerdata_version"],
        )
        statuses.append(
            {
                "run_directory": folder.name,
                "provider": provider,
                "status": report["status"],
                "tables": len(report["tables"]),
                "rows": sum(t["rows"] for t in report["tables"]),
                "errors": report["errors"],
            }
        )
        aid = db.asset(dataset, report_path, folder.name + "/report.json")
        document(db, aid, report_path, "soccerdata_access_report")
        for table in report["tables"]:
            path = public_path(table["path"], folder / "tables")
            schema_path = public_path(
                path.with_suffix(".schema.json"), folder / "tables"
            )
            aid = db.asset(dataset, path, str(path.relative_to(root)))
            sid = db.asset(dataset, schema_path, str(schema_path.relative_to(root)))
            document(db, sid, schema_path, "soccerdata_frame_schema")
            headers, signature = csv_schema(path)
            name = "soccerdata_csv_" + provider.lower() + "_" + signature
            db.csv_table(aid, path, name)
            imported = db.connection.execute(
                "SELECT rows FROM imports WHERE asset_id=? AND kind=?", (aid, name)
            ).fetchone()[0]
            if imported != table["rows"]:
                raise ValueError("SDK export row count differs from its frame receipt")
            db.connection.execute(
                "INSERT OR REPLACE INTO soccerdata_imports VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    provider,
                    folder.name,
                    table["method"],
                    aid,
                    sid,
                    name,
                    imported,
                    report["observed_at"],
                    json.dumps(
                        {
                            "leagues": report["leagues"],
                            "seasons": report["seasons"],
                            "resolved_seasons": report.get("resolved_seasons"),
                            "controls": report["controls"],
                        }
                    ),
                ),
            )
            counts[provider] += imported
        cache = folder / "cache"
        public_path(cache, cache)
        for cached in sorted(cache.rglob("*")):
            if not cached.is_file():
                continue
            public_path(cached, folder / "cache")
            # Public downloaded payloads only. Browser profiles and runtime logs
            # are deliberately outside this cache subtree.
            aid = db.asset(dataset, cached, str(cached.relative_to(root)))
            if cached.suffix == ".json":
                document(db, aid, cached, "soccerdata_original_json")
    db.connection.execute("""UPDATE datasets SET family=CASE WHEN EXISTS (
            SELECT 1 FROM soccerdata_imports s JOIN assets a ON a.id=s.asset_id
            WHERE a.dataset_id=datasets.id AND s.rows>0
        ) THEN 'real_source_adapter' ELSE 'access_diagnostics' END
        WHERE name LIKE 'soccerdata:%'""")
    return {
        "frame_rows_by_provider": dict(counts),
        "runs": statuses,
        "note": "Frame rows can include overlapping scopes/partial seasons. Reports for blocked sources are access evidence, not successful data collections.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", default="data/football.sqlite3")
    p.add_argument("--sources", default=".tmp/open-data/soccerdata")
    p.add_argument("--output", default=".tmp/source-review/soccerdata-ingest.json")
    a = p.parse_args()
    r = ingest(a.database, a.sources)
    Path(a.output).write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
