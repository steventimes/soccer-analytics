"""Store SoccerData frames, column schemas, access reports and original caches."""

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re

from .warehouse import Warehouse, quote
from .build_warehouse import document
from .repository_data import csv_schema
from .soccerdata_collect import RESPONSE_CAPTURE_SCHEMA


def public_path(path, expected):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(expected):
        raise ValueError("SDK asset is outside its public source subtree")
    if resolved.is_file() and resolved.stat().st_nlink != 1:
        raise ValueError("SDK public assets cannot be hard-link aliases")
    return resolved


def response_receipts(report, folder):
    """Validate new capture receipts; archived asset hashes verify their bytes."""
    if "response_capture" not in report:
        return {}
    capture = report["response_capture"]
    if not isinstance(capture, dict) or capture.get("schema") != RESPONSE_CAPTURE_SCHEMA:
        raise ValueError("Unrecognized SDK response capture schema")
    if "body_directory" in capture and capture["body_directory"] != "cache/_responses":
        raise ValueError("Invalid SDK response body directory")
    requests = report.get("requests")
    if not isinstance(requests, list):
        raise ValueError("SDK response capture requires a request list")
    receipts = {}

    def timestamp(value):
        if not isinstance(value, str):
            raise ValueError("SDK response timestamps must include a timezone")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Invalid SDK response timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("SDK response timestamps must include a timezone")

    def body(metadata, kinds):
        if (not isinstance(metadata, dict) or not isinstance(metadata.get("kind"), str)
                or metadata["kind"] not in kinds):
            raise ValueError("Invalid SDK response body kind")
        digest, size = metadata.get("sha256"), metadata.get("bytes")
        if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or type(size) is not int or size < 0):
            raise ValueError("Invalid SDK response body fingerprint")
        relative = f"cache/_responses/{digest}.bin"
        if metadata.get("path") != relative:
            raise ValueError("SDK response body path must match its fingerprint")
        expected = folder / relative
        path = public_path(expected, folder / "cache" / "_responses")
        if path != expected or not path.is_file():
            raise ValueError("SDK response body is missing or is a file alias")
        receipt = (digest, size)
        if path in receipts and receipts[path] != receipt:
            raise ValueError("Conflicting SDK response body receipts")
        receipts[path] = receipt

    for ordinal, entry in enumerate(requests, 1):
        if (not isinstance(entry, dict) or type(entry.get("ordinal")) is not int
                or entry["ordinal"] != ordinal):
            raise ValueError("SDK request ordinals must be consecutive integers")
        timestamp(entry.get("started_at"))
        timestamp(entry.get("completed_at"))
        elapsed = entry.get("elapsed_seconds")
        try:
            valid_elapsed = type(elapsed) in (int, float) and elapsed >= 0 and math.isfinite(elapsed)
        except OverflowError:
            valid_elapsed = False
        if not valid_elapsed:
            raise ValueError("SDK request duration must be finite and nonnegative")
        for key in ("error", "capture_error"):
            if key in entry and (not isinstance(entry[key], str) or not entry[key].strip()):
                raise ValueError("Invalid SDK response failure description")
        failed = "error" in entry or "capture_error" in entry
        if "response_observed_at" in entry:
            timestamp(entry["response_observed_at"])
        if "response" in entry:
            timestamp(entry.get("response_observed_at"))
            body(entry["response"], {"sdk_response_content", "browser_page_source_utf8"})
        elif not failed:
            raise ValueError("A completed SDK request requires its response body")
        if "parser_payload" in entry:
            if entry.get("response", {}).get("kind") != "browser_page_source_utf8":
                raise ValueError("SDK browser parser payload requires its page source")
            body(entry["parser_payload"], {"browser_extracted_payload_utf8"})
        elif (not failed and entry.get("response", {}).get("kind") == "browser_page_source_utf8"):
            raise ValueError("A completed SDK browser request requires its parser payload")
    return receipts


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
        with db.atomic():
            folder = report_path.parent
            if folder.resolve() != folder or report_path.resolve() != report_path:
                raise ValueError("SDK run and report cannot be directory/file aliases")
            public_path(report_path, folder)
            report = json.loads(report_path.read_text())
            receipts = response_receipts(report, folder)
            for table in report["tables"]:
                if type(table["rows"]) is not int or table["rows"] < 0:
                    raise ValueError("SDK frame receipt rows must be a nonnegative integer")
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
                    **{key: report[key] for key in ("normalization", "quality_status", "source_comparison") if key in report},
                }
            )
            aid = db.asset(dataset, report_path, folder.name + "/report.json")
            document(db, aid, report_path, "soccerdata_access_report")
            for table in report["tables"]:
                expected_rows = table["rows"]
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
                if imported != expected_rows:
                    raise ValueError("SDK export row count differs from its frame receipt")
                if db.connection.execute(
                    "SELECT 1 FROM csv_irregular_rows WHERE asset_id=? LIMIT 1", (aid,)
                ).fetchone() is not None:
                    raise ValueError("SDK export contains rows with inconsistent column counts")
                actual_rows = db.connection.execute(
                    f"SELECT COUNT(*) FROM {quote(name)} WHERE asset_id=?", (aid,)
                ).fetchone()[0]
                if actual_rows != expected_rows:
                    raise ValueError("SDK SQL row count differs from its frame receipt")
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
                                **{key: report[key] for key in ("normalization", "quality_status", "source_comparison") if key in report},
                            }
                        ),
                    ),
                )
                counts[provider] += imported
            cache = folder / "cache"
            public_path(cache, cache)
            pending_receipts = set(receipts)
            for cached in sorted(cache.rglob("*")):
                if not cached.is_file():
                    continue
                public_path(cached, folder / "cache")
                # Public downloaded payloads only. Browser profiles and runtime logs
                # are deliberately outside this cache subtree.
                aid = db.asset(dataset, cached, str(cached.relative_to(root)))
                if cached in receipts:
                    actual = db.connection.execute(
                        "SELECT sha256,bytes FROM assets WHERE id=?", (aid,)
                    ).fetchone()
                    if tuple(actual) != receipts[cached]:
                        raise ValueError("Archived SDK response differs from its capture receipt")
                    pending_receipts.remove(cached)
                if cached.suffix == ".json":
                    document(db, aid, cached, "soccerdata_original_json")
            if pending_receipts:
                raise ValueError("SDK response receipts were not all archived")
            db.connection.execute("""UPDATE datasets SET family=CASE WHEN EXISTS (
                    SELECT 1 FROM soccerdata_imports s JOIN assets a ON a.id=s.asset_id
                    WHERE a.dataset_id=datasets.id AND s.rows>0
                ) THEN 'real_source_adapter' ELSE 'access_diagnostics' END
                WHERE id=?""", (dataset,))
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
