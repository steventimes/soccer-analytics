"""Bounded, offline audits of exact provider IDs against a pinned Reep release.

Mappings are third-party assertions, never name matches or historical availability
claims. No warehouse, network, SDK, overlay or cross-run cache is used.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

PROVIDERS = frozenset({"espn", "understat", "statsbomb"})
TABLES = {
    "redirects": ("from_id", "to_id", "reason", "created_at"),
    "bridges": ("reep_id", "provider", "namespace", "external_id", "entity_type"),
    "entities": ("reep_id", "entity_type", "canonical_name", "metadata_json"),
}
QUERY_COLUMNS = ("provider", "namespace", "external_id", "entity_type")


def _sha(value):
    if not isinstance(value, str) or re.fullmatch("[a-f0-9]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


def _file(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Identity inputs cannot be file or directory aliases")
    info = path.stat()
    if not path.is_file() or info.st_nlink != 1:
        raise ValueError("Identity input must be a regular unshared file")
    return info


def _identity(path):
    info = _file(path)
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _small_json(path, limit):
    if _file(path).st_size > limit:
        raise ValueError("Metadata/query byte budget exceeded")
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _rows(path, required, budget):
    """Count decoded bytes and CSV records, including unselected rows."""

    def lines():
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            first = True
            while True:
                block = stream.readline(min(1024 * 1024, budget["bytes"] + 1))
                if not block:
                    break
                budget["bytes"] -= len(block)
                if budget["bytes"] < 0:
                    raise ValueError("Decoded byte budget exceeded")
                if len(block) == 1024 * 1024 and not block.endswith(b"\n"):
                    raise ValueError("CSV line budget exceeded")
                yield block.decode("utf-8-sig" if first else "utf-8")
                first = False

    reader = csv.DictReader(lines(), strict=True)
    headers = reader.fieldnames or []
    if len(headers) != len(set(headers)) or not set(required) <= set(headers):
        raise ValueError(f"Unexpected CSV schema: {path.name}")
    for number, row in enumerate(reader, 1):
        budget["rows"] -= 1
        if budget["rows"] < 0:
            raise ValueError("CSV row budget exceeded")
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"Ragged CSV record in {path.name}: {number}")
        yield number, row


def audit_identity_bridges(
    root,
    queries,
    *,
    release_sha256,
    max_bytes=384 * 1024 * 1024,
    max_decoded_bytes=4 * 1024**3,
    max_rows=21_000_000,
    max_queries=10_000,
):
    """Audit supplied query CSV (exact, case-sensitive IDs) without modifying inputs.

    All three complete source tables must match release.json before any mapping is
    returned. Limits cover full scans, not only matching rows. Row positions are
    one-based CSV data-record ordinals, excluding the header (not physical lines).
    """
    limits = (max_bytes, max_decoded_bytes, max_rows, max_queries)
    if any(type(v) is not int or v <= 0 for v in limits):
        raise ValueError("Identity audit budgets must be positive integers")
    root, queries = Path(root).absolute(), Path(queries).absolute()
    release_path = root / "release.json"
    snapshots = {release_path: _identity(release_path), queries: _identity(queries)}
    release, release_digest = _small_json(release_path, 1024 * 1024)
    if release_digest != _sha(release_sha256):
        raise ValueError("Release checksum mismatch")
    if (
        release.get("schema_version") != "bridge-register-v1"
        or release.get("publishable") is not True
        or release.get("licence", {}).get("spdx") != "CC0-1.0"
        or not isinstance(release.get("stamp"), str)
        or not release["stamp"]
        or release.get("release_tag") != release["stamp"]
    ):
        raise ValueError("Unsupported release identity, publication state or licence")
    qstat = _file(queries)
    if queries.suffix != ".csv" or qstat.st_size > 8 * 1024 * 1024:
        raise ValueError("Supply a query CSV no larger than 8 MiB")
    with queries.open("rb") as stream:
        query_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    query_rows = list(
        _rows(queries, QUERY_COLUMNS, {"bytes": 8 * 1024 * 1024, "rows": max_queries})
    )
    if not query_rows:
        raise ValueError("At least one explicit query ID is required")
    for _, row in query_rows:
        if row["provider"] not in PROVIDERS or any(
            not row[c] or row[c] != row[c].strip() for c in QUERY_COLUMNS
        ):
            raise ValueError(
                "Queries require supported provider and exact nonempty ID/type/namespace"
            )
    wanted = {tuple(row[k] for k in QUERY_COLUMNS[:3]) for _, row in query_rows}
    sources = {}
    total = 0
    # Authenticate every full compressed file before parsing. No partial-download
    # or nearest-name fallback is allowed to impersonate a complete source table.
    for table in TABLES:
        name = f"csv/{table}.csv.gz"
        entry = release["files"][name]
        path = root / name
        info = _file(path)
        if type(entry["bytes"]) is not int or info.st_size != entry["bytes"]:
            raise ValueError(f"Source byte size mismatch: {name}")
        total += info.st_size
        if total > max_bytes:
            raise ValueError("Compressed byte budget exceeded")
        snapshots[path] = _identity(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != _sha(entry["sha256"]):
            raise ValueError(f"Source checksum mismatch: {name}")
        if type(entry.get("rows")) is not int or entry["rows"] < 0:
            raise ValueError("Source table requires an exact nonnegative row count")
        sources[table] = {
            "path": name,
            "sha256": digest,
            "bytes": info.st_size,
            "expected_rows": entry["rows"],
        }
    budget = {"bytes": max_decoded_bytes, "rows": max_rows}
    redirects = defaultdict(list)
    bridges = defaultdict(list)
    entities = defaultdict(list)
    targets = set()
    for table in TABLES:
        count = 0
        for number, row in _rows(root / sources[table]["path"], TABLES[table], budget):
            count += 1
            if table == "redirects":
                if not row["from_id"]:
                    raise ValueError("Redirect is missing its original ID")
                redirects[row["from_id"]].append({"record": number, **row})
                targets.update((row["from_id"], row["to_id"]))
            elif table == "bridges":
                key = tuple(row[k] for k in QUERY_COLUMNS[:3])
                if key in wanted:
                    bridges[key].append({"record": number, **row})
                    targets.add(row["reep_id"])
            elif row["reep_id"] in targets:
                entities[row["reep_id"]].append({"record": number, **row})
        if count != sources[table]["expected_rows"]:
            raise ValueError(f"Source row count mismatch: {table}")
        sources[table]["rows_scanned"] = count
    # Recheck identities after scans so concurrent source changes cannot produce
    # a report authenticated against pre-change checksums.
    if any(_identity(path) != identity for path, identity in snapshots.items()):
        raise ValueError("Identity input changed during audit")

    def resolve(original, expected_type):
        current = original
        chain = []
        seen = set()
        while current in redirects:
            if current in seen or len(seen) >= 64:
                return {
                    "status": "invalid_redirect",
                    "chain": chain,
                    "resolved_reep_id": None,
                }
            seen.add(current)
            edges = redirects[current]
            destinations = {e["to_id"] for e in edges}
            chain.extend(edges)
            if len(destinations) != 1:
                return {"status": "ambiguous", "chain": chain, "resolved_reep_id": None}
            current = next(iter(destinations))
            if not current:
                return {"status": "retired", "chain": chain, "resolved_reep_id": None}
        records = entities.get(current, [])
        details = {"chain": chain, "resolved_reep_id": current, "entities": records}
        if not records:
            return {**details, "status": "missing_entity"}
        for rid in seen | {current}:
            if any(e["entity_type"] != expected_type for e in entities.get(rid, [])):
                return {**details, "status": "type_mismatch"}
        if len(records) != 1:
            return {**details, "status": "ambiguous"}
        metadata = json.loads(records[0]["metadata_json"] or "{}")
        if not isinstance(metadata, dict):
            return {**details, "status": "invalid_entity"}
        if metadata.get("status") in ("retired", "withdrawn", "tombstoned"):
            return {**details, "status": "retired"}
        return {**details, "status": "redirected" if chain else "unique"}

    output = []
    for number, query in query_rows:
        key = tuple(query[k] for k in QUERY_COLUMNS[:3])
        matches = []
        for row in bridges.get(key, []):
            if not row["reep_id"]:
                raise ValueError("Matched bridge is missing its Reep ID")
            result = resolve(row["reep_id"], query["entity_type"])
            if row["entity_type"] != query["entity_type"]:
                result["status"] = "type_mismatch"
            matches.append(
                {"original_reep_id": row["reep_id"], "bridge": row, **result}
            )
        statuses = {m["status"] for m in matches}
        resolved = {
            m["resolved_reep_id"]
            for m in matches
            if m["status"] in ("unique", "redirected")
        }
        if not matches:
            status = "uncovered"
        elif statuses <= {"unique", "redirected"}:
            status = (
                "ambiguous"
                if len(resolved) != 1
                else ("redirected" if "redirected" in statuses else "unique")
            )
        elif len(statuses) == 1:
            status = next(iter(statuses))
        else:
            status = "ambiguous"
        output.append(
            {
                "query_record": number,
                "query": query,
                "status": status,
                "matches": matches,
                "resolved_reep_id": next(iter(resolved))
                if status in ("unique", "redirected")
                else None,
                "duplicate_bridge_rows": len(matches)
                - len(
                    {
                        tuple((k, v) for k, v in m["bridge"].items() if k != "record")
                        for m in matches
                    }
                ),
            }
        )
    counts = dict(sorted(Counter(r["status"] for r in output).items()))
    group_counts = defaultdict(Counter)
    for row in output:
        group_counts[
            tuple(row["query"][k] for k in ("provider", "namespace", "entity_type"))
        ][row["status"]] += 1
    matched = sum(counts.get(k, 0) for k in ("unique", "redirected"))
    return {
        "schema": "SoccerAnalytics.IdentityBridgeAudit.v1",
        "status": "audited",
        "adoption": "not_adopted",
        "historical_availability_verified": False,
        "mapping_status": (
            "no_intersection"
            if not any(r["matches"] for r in output)
            else "mapped_queries_only"
            if matched == len(output)
            else "review_required"
        ),
        "release": {
            "stamp": release["stamp"],
            "sha256": release_digest,
            "licence": "CC0-1.0",
        },
        "queries": {"sha256": query_digest, "rows": len(query_rows)},
        "sources": sources,
        "limits": dict(
            zip(("compressed_bytes", "decoded_bytes", "rows", "queries"), limits)
        ),
        "usage": {
            "compressed_bytes": total,
            "decoded_bytes": max_decoded_bytes - budget["bytes"],
            "rows": max_rows - budget["rows"],
        },
        "counts": counts,
        "groups": [
            {
                "provider": k[0],
                "namespace": k[1],
                "entity_type": k[2],
                "queries": sum(v.values()),
                "counts": dict(sorted(v.items())),
            }
            for k, v in sorted(group_counts.items())
        ],
        "results": output,
    }
