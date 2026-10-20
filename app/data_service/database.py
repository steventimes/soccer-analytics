"""Inspect or export source assets without loading large files into memory."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import zlib
from pathlib import Path


@contextmanager
def _read_snapshot(database):
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.close()


def export_asset(database, asset_id, output):
    with _read_snapshot(database) as connection:
        row = connection.execute(
            "SELECT sha256,bytes,complete FROM assets WHERE id=?", (asset_id,)
        ).fetchone()
        if row is None or not row[2]:
            raise ValueError("Unknown or incomplete asset")
        output = Path(output)
        if output.exists():
            raise FileExistsError(output)
        digest = hashlib.sha256()
        size = 0
        fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                for (chunk,) in connection.execute(
                    "SELECT content FROM raw_chunks WHERE asset_id=? ORDER BY ordinal",
                    (asset_id,),
                ):
                    raw = zlib.decompress(chunk)
                    digest.update(raw)
                    size += len(raw)
                    stream.write(raw)
                if digest.hexdigest() != row[0] or size != row[1]:
                    raise ValueError("Stored asset digest mismatch")
                stream.flush()
                os.fsync(stream.fileno())
            # Publish complete, verified bytes without replacing any existing path.
            os.link(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)


def analysis_lineage(database, run_id, *, run_kind=None, limit=50):
    """Read explicit input/run/output relations; never infer missing ancestry."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer between 1 and 1000")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a nonempty string")
    with _read_snapshot(database) as connection:
        connection.row_factory = sqlite3.Row
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        relations = {key: name for key, name in (("inputs", "analysis_run_inputs"), ("outputs", "analysis_run_assets")) if name in tables}
        kinds = set()
        for table in relations.values():
            kinds.update(r[0] for r in connection.execute(f"SELECT DISTINCT run_kind FROM {table} WHERE run_id=?", (run_id,)))
        if run_kind is None:
            if len(kinds) > 1:
                raise ValueError("Run ID is ambiguous; provide run_kind")
            run_kind = next(iter(kinds), None)
        if run_kind is None or run_kind not in kinds:
            raise ValueError("Unknown analysis run")
        result = {"schema": "SoccerAnalytics.AnalysisLineage.v1", "run_kind": run_kind, "run_id": run_id,
                  "lineage_scope": "recorded_assets_only"}
        for key in ("inputs", "outputs"):
            rows, count = [], 0
            if key in relations:
                table = relations[key]
                count = connection.execute(f"SELECT COUNT(*) FROM {table} WHERE run_kind=? AND run_id=?", (run_kind, run_id)).fetchone()[0]
                rows = [dict(r) for r in connection.execute(
                    f"SELECT r.asset_id,a.path,a.sha256,a.bytes,a.complete,d.name dataset,d.provider,d.edition "
                    f"FROM {table} r LEFT JOIN assets a ON a.id=r.asset_id LEFT JOIN datasets d ON d.id=a.dataset_id "
                    "WHERE r.run_kind=? AND r.run_id=? ORDER BY r.asset_id LIMIT ?", (run_kind, run_id, limit))]
            result[key] = {"total": count, "assets": rows, "truncated": count > len(rows)}
        result["inputs_recorded"] = result["inputs"]["total"] > 0
        return result


def match_xg_coverage(database):
    """Audit provider-labelled match xG per source asset, without pooling revisions."""
    def valid_value(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(value) and value >= 0
        except OverflowError:
            return False

    snapshots = []
    with _read_snapshot(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT a.id asset_id,a.path,a.sha256,d.name dataset,j.payload "
            "FROM documents j JOIN assets a ON a.id=j.asset_id "
            "JOIN datasets d ON d.id=a.dataset_id WHERE a.complete=1 "
            "AND json_extract(j.payload,'$.source')='football-csv' "
            "AND json_extract(j.payload,'$.provider')='football-data.co.uk' "
            "AND json_type(j.payload,'$.matches')='array' ORDER BY a.id"
        )
        for row in rows:
            snapshot = json.loads(row["payload"])
            counts = {side: dict.fromkeys(
                ("valid", "missing", "invalid", "column_absent", "untracked"), 0
            ) for side in ("home", "away")}
            pairs = []
            inconsistent = numeric_pairs = 0
            for match in snapshot["matches"]:
                if not isinstance(match, dict):
                    raise ValueError(f"Asset {row['asset_id']} contains a non-object match")
                provenance = match.get("statistics_metadata")
                metadata = provenance.get("xg") if isinstance(provenance, dict) else None
                statistics = match.get("statistics")
                statistics = statistics if isinstance(statistics, dict) else {}
                values = [statistics.get(f"{side}_xg") for side in ("home", "away")]
                valid = [valid_value(value) for value in values]
                numeric_pairs += all(valid)
                if metadata is None:
                    for side in counts:
                        counts[side]["untracked"] += 1
                    # Legacy values without provenance are never aggregated.
                    inconsistent += any(value is not None for value in values)
                    continue
                consistent = (
                    isinstance(metadata, dict)
                    and metadata.get("source_fields") == {"home": "HxG", "away": "AxG"}
                    and metadata.get("definition_status") == "unverified"
                    and metadata.get("usage_policy") == "post_match_only"
                    and type(metadata.get("complete_pair")) is bool
                    and metadata["complete_pair"] == all(valid)
                )
                for side, value, good in zip(("home", "away"), values, valid):
                    state = metadata.get(f"{side}_status") if isinstance(metadata, dict) else None
                    if not isinstance(state, str) or state not in counts[side] or state == "untracked":
                        counts[side]["untracked"] += 1
                        consistent = False
                    else:
                        counts[side][state] += 1
                        consistent = consistent and (
                            good if state == "valid" else value is None
                        )
                inconsistent += not consistent
                if consistent and all(valid) and match.get("status") == "FINISHED":
                    pairs.append(values)
            summary = {}
            for index, side in enumerate(("home", "away")):
                values = [pair[index] for pair in pairs]
                summary[side] = {
                    "mean": math.fsum(value / len(values) for value in values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }
            snapshots.append({
                **{key: row[key] for key in ("asset_id", "path", "sha256", "dataset")},
                **{key: snapshot.get(key) for key in ("source", "provider", "competition", "season", "fetched_at", "normalized_at")},
                "matches": len(snapshot["matches"]),
                "field_status_counts": counts,
                "complete_numeric_pairs": numeric_pairs,
                "inconsistent_metadata_matches": inconsistent,
                "complete_finished_pairs": len(pairs),
                "paired_xg_summary": summary,
            })
    return {
        "schema": "SoccerAnalytics.MatchXGCoverage.v1",
        "scope": "Per-asset football-csv snapshots; revisions are not deduplicated or pooled.",
        "definition_status": "unverified",
        "usage_policy": "post_match_only",
        "availability_timestamp": None,
        "snapshots": snapshots,
    }


def verify_database(database):
    with _read_snapshot(database) as connection:
        integrity = [r[0] for r in connection.execute("PRAGMA quick_check")]
        foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
        incomplete = connection.execute(
            "SELECT COUNT(*) FROM assets WHERE complete!=1"
        ).fetchone()[0]
        tested = []
        for (dataset_id,) in connection.execute("SELECT id FROM datasets"):
            sample = connection.execute(
                "SELECT id,sha256,bytes FROM assets WHERE dataset_id=? AND complete=1 ORDER BY bytes LIMIT 3",
                (dataset_id,),
            ).fetchall()
            for aid, expected, expected_bytes in sample:
                digest = hashlib.sha256()
                size = 0
                for (chunk,) in connection.execute(
                    "SELECT content FROM raw_chunks WHERE asset_id=? ORDER BY ordinal",
                    (aid,),
                ):
                    raw = zlib.decompress(chunk)
                    digest.update(raw)
                    size += len(raw)
                if digest.hexdigest() != expected or size != expected_bytes:
                    raise ValueError(f"Asset {aid} digest mismatch")
                tested.append(aid)
        result = {
            "quick_check": integrity,
            "foreign_key_errors": foreign_keys,
            "incomplete_assets": incomplete,
            "verified_source_assets": tested,
        }
        if integrity != ["ok"] or foreign_keys or incomplete:
            raise ValueError(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument("--export-asset", type=int)
    parser.add_argument("--output")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--xg-coverage", action="store_true", help="Read-only per-snapshot audit of provider-labelled HxG/AxG values")
    parser.add_argument("--lineage-run", help="Analysis run ID whose explicit sources and outputs should be inspected")
    parser.add_argument("--run-kind", help="Required when multiple analysis kinds share a run ID")
    parser.add_argument("--limit", type=int, default=50, help="Maximum assets per side of the lineage (1–1000)")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_database(args.database), indent=2))
    elif args.xg_coverage:
        print(json.dumps(match_xg_coverage(args.database), allow_nan=False, indent=2))
    elif args.lineage_run is not None:
        print(json.dumps(analysis_lineage(args.database, args.lineage_run, run_kind=args.run_kind, limit=args.limit), indent=2))
    elif args.export_asset is not None:
        if not args.output:
            parser.error("--output is required for asset export")
        export_asset(args.database, args.export_asset, args.output)
    else:
        with _read_snapshot(args.database) as connection:
            connection.row_factory = sqlite3.Row
            print(
                json.dumps(
                    {
                        "datasets": [
                            dict(r)
                            for r in connection.execute("SELECT * FROM datasets")
                        ],
                        "imports": [
                            dict(r)
                            for r in connection.execute(
                                "SELECT kind,SUM(rows) rows FROM imports GROUP BY kind"
                            )
                        ],
                        "quarantine": [
                            dict(r)
                            for r in connection.execute("SELECT * FROM quarantine")
                        ],
                    },
                    indent=2,
                )
            )
