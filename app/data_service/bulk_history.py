"""Discover every published European season ZIP and normalize its divisions."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests

from app.data_service.archive import _request
from app.data_service.fetch.public import LEAGUES, FootballCSVClient
from app.data_service.history import digest, read_snapshot
from app.file_io import write_json_atomic as _atomic_write_json
from app.file_io import write_bytes_atomic
from app.data_service.freshness import SnapshotCachePolicy, utc_now
from app.data_service.fetch.retry import close_response

logger = logging.getLogger(__name__)
INDEX_URL = "https://www.football-data.co.uk/downloadm.php"


def discover_season_archives(html: str) -> dict[str, str]:
    archives = {}
    for href in re.findall(r'href=["\x27]([^"\x27]+)', html, re.I):
        if not href.lower().endswith(".zip"):
            continue
        match = re.search(r"mmz4281/(\d{4})/", href)
        if match is None:
            continue
        first = int(match[1][:2])
        year = 1900 + first if first >= 90 else 2000 + first
        if int(match[1][2:]) != (year + 1) % 100:
            continue
        # Prefer the CSV ZIP, not any alternate spreadsheet archive.
        if "all-euro-data" not in href and not href.endswith("data.zip"):
            continue
        archives[str(year)] = urljoin(INDEX_URL, href)
    return dict(sorted(archives.items()))


def collect_all_csv(root: Path, *, refresh=False, max_age_hours=None, active_max_age_hours=None, clock=None) -> dict:
    policy = SnapshotCachePolicy(max_age_hours, active_max_age_hours)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = _request(session, INDEX_URL)
    try:
        html = response.text
    finally:
        close_response(response)
    (root / "source-index.html").write_text(html, encoding="utf-8")
    archives = discover_season_archives(html)
    if not archives:
        raise ValueError("Official page contains no recognized season archives")
    inverse = {details[0]: code for code, details in LEAGUES.items()}
    client = FootballCSVClient()
    collections, files, errors = [], [], []
    history = root / "history"
    for season, url in archives.items():
        archive_path = root / "archives" / f"{season}.zip"
        receipt_path = archive_path.with_suffix('.receipt.json')
        decision = None
        try:
            archive_bytes = archive_path.read_bytes() if archive_path.exists() else None
            receipt = None
            if archive_bytes is not None and receipt_path.exists():
                try:
                    candidate = json.loads(receipt_path.read_text())
                    if (candidate.get('schema') == 1 and candidate.get('source_url') == url
                            and candidate.get('sha256') == hashlib.sha256(archive_bytes).hexdigest()):
                        receipt = candidate
                except (OSError, ValueError, AttributeError):
                    pass
            decision = policy.decide(receipt.get('fetched_at') if receipt else None, season,
                                     now=utc_now(clock), exists=archive_bytes is not None, refresh=refresh)
            downloaded = decision['action'] == 'fetch'
            if downloaded:
                response = _request(session, url)
                try:
                    archive_bytes = response.content
                finally:
                    close_response(response)
                fetched_at = utc_now(clock).isoformat()
            else:
                fetched_at = receipt.get('fetched_at') if receipt else None
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                if archive.testzip() is not None:
                    raise ValueError("Archive CRC check failed")
                archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
                if downloaded:
                    # Verify the whole replacement before touching a valid existing ZIP.
                    write_bytes_atomic(archive_path, archive_bytes)
                    _atomic_write_json(receipt_path, {'schema': 1, 'source_url': url,
                                       'fetched_at': fetched_at, 'sha256': archive_sha256})
                for name in sorted(archive.namelist()):
                    division = Path(name).stem.upper()
                    if not name.lower().endswith(".csv") or division not in inverse:
                        continue
                    code = inverse[division]
                    raw = archive.read(name)
                    raw_path = root / "data" / season / f"{division}.csv"
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    if not raw_path.exists() or raw_path.read_bytes() != raw:
                        write_bytes_atomic(raw_path, raw)
                    files.append(
                        {
                            "path": str(raw_path.relative_to(root)),
                            "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "source_url": url,
                            "archive_member": name,
                        }
                    )
                    snapshot_path = history / "football-csv" / code / f"{season}.json"
                    try:
                        snapshot = None
                        if snapshot_path.exists() and not downloaded:
                            snapshot = read_snapshot(snapshot_path)
                            if (snapshot['source'], snapshot['competition'], snapshot['season']) != ('football-csv', code, season):
                                raise ValueError('Snapshot scope differs from archive member')
                            snapshot_archive = snapshot.get('archive_sha256')
                            if snapshot_archive != archive_sha256 and (receipt is not None or snapshot_archive is not None):
                                snapshot = None
                        if snapshot is None:
                            try:
                                text = raw.decode("utf-8-sig")
                            except UnicodeDecodeError:
                                text = raw.decode("cp1252")
                            matches = [
                                m
                                for m in client.parse_csv_text(code, text, url)
                                if m["status"] == "FINISHED"
                            ]
                            if not matches:
                                raise ValueError("No completed matches")
                            unique = {}
                            for item in matches:
                                if item["id"] in unique and unique[item["id"]] != item:
                                    raise ValueError("Conflicting duplicate fixture")
                                unique[item["id"]] = item
                            snapshot = {
                                "schema": 1,
                                "source": "football-csv",
                                "provider": client.source_name,
                                "competition": code,
                                "season": season,
                                "fetched_at": fetched_at,
                                "normalized_at": utc_now(clock).isoformat(),
                                "archive_sha256": archive_sha256,
                                "matches": sorted(
                                    unique.values(),
                                    key=lambda m: (m["utcDate"], m["id"]),
                                ),
                            }
                            snapshot["sha256"] = digest(snapshot)
                            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                            _atomic_write_json(snapshot_path, snapshot)
                        collections.append(
                            {
                                "source": "football-csv",
                                "competition": code,
                                "season": season,
                                "path": str(snapshot_path.relative_to(history)),
                                "sha256": snapshot["sha256"],
                                "rows": len(snapshot["matches"]),
                                "fetched_at": snapshot["fetched_at"],
                                "cache": decision,
                                "source_urls": [url],
                            }
                        )
                    except Exception as exc:
                        errors.append(
                            {
                                "source": "football-csv",
                                "competition": code,
                                "season": season,
                                "stage": "normalize",
                                "message": str(exc),
                            }
                        )
            logger.info(
                "CSV archive %s: %d league-seasons, %d matches",
                season,
                len(collections),
                sum(entry["rows"] for entry in collections),
            )
            _atomic_write_json(
                root / "manifest.partial.json",
                {"collections": collections, "errors": errors, "files": files},
            )
        except Exception as exc:
            errors.append(
                {
                    "source": "football-csv",
                    "season": season,
                    "stage": "archive",
                    "message": str(exc),
                    "cache": decision,
                }
            )
    inventory = {
        "schema": 1,
        "generated_at": utc_now(clock).isoformat(),
        "cache_policy": policy.summary(),
        "collections": collections,
        "errors": errors,
        "rows": sum(entry["rows"] for entry in collections),
    }
    history.mkdir(exist_ok=True)
    _atomic_write_json(history / "manifest.json", inventory)
    report = {
        **inventory,
        "index_url": INDEX_URL,
        "published_seasons": list(archives),
        "files": files,
        "bytes": sum(entry["bytes"] for entry in files),
        "complete": not errors,
        "note": "Archive presence is not a complete season guarantee; latest season can be partial.",
    }
    _atomic_write_json(root / "manifest.json", report)
    return report
