"""Verified local observations for score-only, point-in-time research cohorts.

Original snapshots and receipts stay unchanged. Data captured after selection
cannot revise the selected development history. No model or warehouse is loaded.
"""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re

from app.data_service.fetch.public import FootballCSVClient, LEAGUES
from app.data_service.match_records import normalize_match
from app.run_receipts import content_digest as digest


def prepare_observed_history(
    snapshot_paths,
    csv_paths,
    *,
    competition,
    season,
    as_of,
    selection_as_of,
    max_input_bytes=32 * 1024 * 1024,
    max_matches=100_000,
):
    """Reconcile verified local observations without backdating or copying late stats.

    Conflicting kickoff/team/score versions are excluded, not silently overwritten.
    Only score-based canonical fields leave this preparation boundary. Original
    snapshots/CSV receipts remain immutable; evidence identifies each exact input.
    """

    def timestamp(value):
        parsed = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, str)
            else value
        )
        if (
            not isinstance(parsed, datetime)
            or parsed.tzinfo is None
            or parsed.utcoffset() is None
        ):
            raise ValueError("Observation times require an explicit timezone")
        return parsed.astimezone(UTC)

    cutoff, selection = timestamp(as_of), timestamp(selection_as_of)
    if selection > cutoff:
        raise ValueError("Selection cutoff cannot be after observation cutoff")
    if (
        competition not in LEAGUES
        or not isinstance(season, str)
        or re.fullmatch(r"\d{4}", season) is None
        or not 1900 <= int(season) <= 2100
    ):
        raise ValueError("Invalid observed-history league/season")
    if any(type(v) is not int or v <= 0 for v in (max_input_bytes, max_matches)):
        raise ValueError("Observation input budgets must be positive integers")
    paths = [(Path(p).resolve(), "snapshot") for p in snapshot_paths] + [
        (Path(p).resolve(), "csv") for p in csv_paths
    ]
    if not paths or len({p for p, _ in paths}) != len(paths):
        raise ValueError("Supply distinct, explicit observation inputs")
    budget = 0
    observations = {}
    evidence = []
    scanned = 0
    original_files = {}

    def read(path):
        nonlocal budget
        info = path.stat()
        if not path.is_file() or info.st_nlink != 1:
            raise ValueError("Observation input requires a regular unshared file")
        budget += info.st_size
        if budget > max_input_bytes:
            raise ValueError("Observation byte budget exceeded")
        raw = path.read_bytes()
        original_files[path] = hashlib.sha256(raw).hexdigest()
        return raw

    for path, kind in paths:
        raw = read(path)
        if kind == "snapshot":
            payload = json.loads(raw)
            if (
                payload.get("schema") != 1
                or payload.get("sha256")
                != digest({k: v for k, v in payload.items() if k != "sha256"})
                or (
                    payload.get("source"),
                    payload.get("provider"),
                    payload.get("competition"),
                    payload.get("season"),
                )
                != ("football-csv", "football-data.co.uk", competition, season)
            ):
                raise ValueError("Observation snapshot checksum or scope mismatch")
            observed = timestamp(payload["fetched_at"])
            # Derived cohorts are not independent observations and cannot be fed
            # back as new evidence with a different availability interpretation.
            if payload.get("snapshot_kind") is not None:
                raise ValueError("Supply original observations, not a derived cohort")
            rows = payload["matches"]
            input_digest = payload["sha256"]
            receipt_digest = None
        else:
            receipt_path = path.with_suffix(path.suffix + ".receipt.json")
            receipt_raw = read(receipt_path)
            receipt = json.loads(receipt_raw)
            if (
                receipt.get("sha256") != hashlib.sha256(raw).hexdigest()
                or type(receipt.get("bytes")) is not int
                or receipt["bytes"] != len(raw)
            ):
                raise ValueError("Observation CSV checksum or size mismatch")
            observed = timestamp(receipt["observed_at"])
            division = LEAGUES[competition][0]
            period = season[-2:] + str(int(season) + 1)[-2:]
            prefix = f"https://www.football-data.co.uk/mmz4281/{period}/"
            url = receipt["url"]
            if url not in (prefix + division + ".csv", prefix + "data.zip"):
                raise ValueError("Observation CSV URL does not match league/season")
            if (
                url.endswith("data.zip")
                and receipt.get("archive_member") != division + ".csv"
            ):
                raise ValueError("Observation ZIP member does not match league")
            rows = FootballCSVClient().parse_csv_text(
                competition,
                raw.decode("utf-8-sig"),
                url,
                include_record_number=True,
                division_filter=True,
            )
            if not rows:
                raise ValueError("Observation CSV contains no match records")
            input_digest = hashlib.sha256(raw).hexdigest()
            receipt_digest = hashlib.sha256(receipt_raw).hexdigest()
        if observed > cutoff:
            raise ValueError("Observation was recorded after the requested cutoff")
        source_index = len(evidence)
        evidence.append(
            {
                "path": str(path),
                "kind": kind,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "input_sha256": input_digest,
                "receipt_sha256": receipt_digest,
                "observed_at": observed.isoformat(),
                "rows": len(rows),
            }
        )
        for ordinal, row in enumerate(rows, 1):
            if kind == "csv":
                ordinal = row["_source_record"]
            scanned += 1
            if scanned > max_matches:
                raise ValueError("Observation match budget exceeded")
            if row.get("status", "FINISHED") != "FINISHED":
                continue
            if row.get("time_precision") != "minute":
                raise ValueError(
                    "Observed history requires precise source kickoff times"
                )
            record = normalize_match(row)
            kickoff = timestamp(row["utcDate"])
            season_start = datetime(int(season), 7, 1, tzinfo=UTC)
            season_end = datetime(int(season) + 1, 7, 1, tzinfo=UTC)
            if not season_start <= kickoff < season_end:
                raise ValueError("Observation match is outside the requested season")
            if observed <= kickoff:
                raise ValueError("Finished result observation must be after kickoff")
            if row.get("source") != "football-data.co.uk":
                raise ValueError("Observation contains a different provider")
            if kickoff < selection and observed >= selection:
                continue  # Later observations cannot revise or exclude development rows.
            # Ignore any claimed earlier result timestamp; the verified complete
            # input observation is the conservative upper bound used here.
            core = (
                record.match_id,
                kickoff.isoformat(),
                record.home_team_id,
                record.away_team_id,
                record.home_score,
                record.away_score,
            )
            observations.setdefault(record.match_id, []).append(
                (observed, core, source_index, ordinal, row)
            )
    matches = []
    excluded = []
    for mid, versions in sorted(observations.items()):
        if len({v[1] for v in versions}) != 1:
            excluded.append(
                {
                    "match_id": mid,
                    "reason": "conflicting_core_versions",
                    "evidence": [{"input": v[2], "record": v[3]} for v in versions],
                }
            )
            continue
        observed, core, source_index, ordinal, row = min(
            versions, key=lambda v: (v[0], v[2], v[3])
        )
        # Never expose late xG, odds, names revised later, or unverified timings
        # as though they were available at the earliest observed score time.
        match = {
            "id": mid,
            "utcDate": core[1],
            "status": "FINISHED",
            "homeTeam": {"id": core[2], "name": row["homeTeam"]["name"]},
            "awayTeam": {"id": core[3], "name": row["awayTeam"]["name"]},
            "score": {"fullTime": {"home": core[4], "away": core[5]}},
            "source": "football-data.co.uk",
            "source_url": row["source_url"],
            "time_precision": "minute",
            "result_observed_at": observed.isoformat(),
            "observation_evidence": {"input": source_index, "record": ordinal},
            "result_availability_basis": "verified_local_snapshot_observation",
        }
        matches.append(match)
    if not matches:
        raise ValueError("No conflict-free observed results remain")
    for path, sha in original_files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("Observation input changed during preparation")
    body = {
        "schema": 1,
        "source": "football-csv",
        "provider": "football-data.co.uk",
        "competition": competition,
        "season": season,
        "fetched_at": max(e["observed_at"] for e in evidence),
        "snapshot_kind": "observed_result_cohort",
        "assembled_as_of": cutoff.isoformat(),
        "selection_as_of": selection.isoformat(),
        "matches": sorted(matches, key=lambda m: (m["utcDate"], m["id"])),
        "observation_inputs": evidence,
        "excluded_records": excluded,
        "preparation": {
            "scanned_rows": scanned,
            "retained_matches": len(matches),
            "excluded_conflicting_ids": len(excluded),
            "input_bytes": budget,
            "availability": "earliest verified local observation of unchanged score/kickoff/team IDs; not match-end time",
            "features": "score-derived only; post-match statistics and odds omitted",
        },
    }
    return {**body, "sha256": digest(body)}
