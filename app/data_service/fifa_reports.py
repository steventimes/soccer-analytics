"""Anonymous official FIFA match-report collection and source-located metrics.

Reports contain tracking-derived aggregates and diagrams, not raw trajectories.
Poppler's pdftotext is required; downloaded code is never executed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from threading import Lock
from urllib.parse import unquote, urljoin, urlparse
import xml.etree.ElementTree as ET

import requests
from .warehouse import Warehouse
from .wyscout import write_download
from .fetch.fetcher import SlidingWindowRateLimiter

SERIES = {
    "fcwc2025": {
        "edition": 2025,
        "matches": 63,
        "competition": "FIFA Club World Cup",
        "indices": [
            "https://www.fifatrainingcentre.com/en/game/tournaments/fcwc/2025/post-match-summary-reports.php"
        ],
    },
    "fwc2026": {
        "edition": 2026,
        "matches": 104,
        "competition": "FIFA World Cup",
        "indices": [
            "https://www.fifatrainingcentre.com/en/fifa-world-cup-2026/match-report-hub.php",
            "https://www.fifatrainingcentre.com/en/fifa-world-cup-2026/match-report-hub-knockout-stage.php",
        ],
    },
}
PHASES = {
    "in_possession": [
        "Build Up Unopposed",
        "Build Up Opposed",
        "Progression",
        "Final Third",
        "Long Ball",
        "Attacking Transition",
        "Counter Attack",
        "Set Piece",
    ],
    "out_of_possession": [
        "High Press",
        "Mid Press",
        "Low Press",
        "High Block",
        "Mid Block",
        "Low Block",
        "Recovery",
        "Defensive Transition",
        "Counter-press",
    ],
}


class ReportLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v.strip() for k, v in attrs if k == "href" and v)


def report_links(html, index_url, maximum):
    parser = ReportLinks()
    parser.feed(html)
    result = {}
    for href in parser.links:
        url = urljoin(index_url, href)
        parts = urlparse(url)
        if (
            parts.scheme != "https"
            or parts.netloc != "www.fifatrainingcentre.com"
            or not parts.path.startswith("/media/native/")
        ):
            continue
        name = unquote(Path(parts.path).name)
        match = re.search(r"\bM(\d{1,3})(?:[ ._-])", name)
        if not match or not name.lower().endswith(".pdf"):
            continue
        number = int(match[1])
        if not 1 <= number <= maximum:
            raise ValueError("Indexed FIFA match number outside expected tournament")
        if number in result and result[number] != url:
            raise ValueError("Multiple indexed report versions require review")
        result[number] = url
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(root, series):
    settings = SERIES[series]
    root = Path(root).resolve() / series
    root.mkdir(parents=True, exist_ok=True)
    limiter = SlidingWindowRateLimiter(30, 60)
    request_lock = Lock()
    urls = {}
    files = []
    errors = []

    def download(url, path, pdf=False):
        if path.is_symlink():
            raise ValueError("FIFA cache target cannot be an alias")
        receipt = path.with_suffix(path.suffix + ".receipt.json")
        if receipt.is_symlink():
            raise ValueError("FIFA cache receipt cannot be an alias")
        if path.exists() and receipt.exists():
            old = json.loads(receipt.read_text())
            if old["url"] == url and digest(path) == old["sha256"]:
                return old
        with request_lock:
            limiter.wait_if_needed()
            limiter.add_request()
        with requests.Session() as session:
            session.trust_env = False
            with session.get(url, stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()

                def chunks():
                    size = 0
                    first = True
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        if first and pdf and not chunk.startswith(b"%PDF-"):
                            raise ValueError("Official report response is not PDF")
                        first = False
                        size += len(chunk)
                        if size > 50_000_000:
                            raise ValueError("Report exceeds bounded download size")
                        yield chunk

                write_download(path, chunks())
        record = {
            "url": url,
            "path": path.name,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        write_download(receipt, [json.dumps(record).encode()])
        return record

    for i, url in enumerate(settings["indices"]):
        path = root / f"index-{i}.html"
        record = download(url, path)
        files.append(record)
        for number, target in report_links(
            path.read_text(), url, settings["matches"]
        ).items():
            if number in urls and urls[number] != target:
                raise ValueError("Conflicting report across official hubs")
            urls[number] = target
    if set(urls) != set(range(1, settings["matches"] + 1)):
        raise ValueError(
            "Official report hub does not cover the expected complete tournament"
        )
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {
            pool.submit(download, url, root / f"M{number:03}.pdf", True): number
            for number, url in urls.items()
        }
        for future in as_completed(jobs):
            number = jobs[future]
            try:
                files.append({"match_number": number, **future.result()})
                print("Collected", series, number, flush=True)
            except Exception as exc:
                errors.append({"match_number": number, "error": str(exc)})
    manifest = {
        "series": series,
        **settings,
        "observed_at": datetime.now().astimezone().isoformat(),
        "files": sorted(files, key=lambda r: r["path"]),
        "errors": errors,
        "scope": "Anonymous official reports; aggregate metrics and diagrams, not acquired raw player/ball trajectories.",
    }
    write_download(root / "manifest.json", [json.dumps(manifest, indent=2).encode()])
    return manifest


def parse_report(pages, expected_number, edition):
    if len(pages) < 4:
        raise ValueError("Report lacks summary/phase pages")
    cover = pages[0]
    number = re.search(r"Match\s+(\d+)", cover, re.I)
    score = re.search(r"^(.+?)\s+(\d+)\s*[-–]\s*(\d+)\s+(.+?)\s*$", cover, re.M)
    if not score:
        # Two official covers wrap both club names and the score over two lines.
        # The team-summary header keeps explicit left/right scores on one line.
        header = pages[1].split("STARTING", 1)[0]
        score = re.search(
            r"^\s*(.+?)\s{2,}(\d+)\s{2,}(\d+)\s{2,}(.+?)\s*$", header, re.M
        )
    date = re.search(r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b", cover)
    if not number or int(number[1]) != expected_number or not score or not date:
        raise ValueError("Report identity/score/date cannot be verified")
    day = datetime.strptime(date[1], "%d %B %Y").date()
    if day.year != edition:
        raise ValueError("Report belongs to another edition")
    metadata = {
        "match_number": expected_number,
        "date": day.isoformat(),
        "teams": [score[1].strip(), score[4].strip()],
        "goals": [int(score[2]), int(score[3])],
        "score_duration": "as_reported; extra_time/shootouts not inferred from score alone",
    }
    stage = re.search(r"^\s*(.+?)\s*-\s*Match\s+\d+\s*$", cover, re.M)
    metadata["stage"] = stage[1].strip() if stage else None
    metrics = []

    def add(name, values, unit, page, label):
        for side, value in enumerate(values):
            metrics.append(
                {
                    "team_slot": side,
                    "metric": name,
                    "value": float(value),
                    "unit": unit,
                    "page": page,
                    "source_label": label,
                }
            )

    def pair(page, label, value=r"(\d+(?:\.\d+)?)", unit="count", name=None):
        found = re.search(
            r"^\s*" + value + r"\s+" + re.escape(label) + r"\s+" + value + r"\s*$",
            pages[page - 1],
            re.M,
        )
        if found:
            add(
                name or re.sub(r"\W+", "_", label.lower()).strip("_"),
                found.groups(),
                unit,
                page,
                label,
            )
        return found

    basic = pages[2]
    possession = re.search(
        r"Total\s+(\d+(?:\.\d+)?)%\s+(\d+(?:\.\d+)?)%\s+(\d+(?:\.\d+)?)%\s+Total", basic
    )
    if possession:
        values = [float(v) for v in possession.groups()]
        if any(not 0 <= v <= 100 for v in values) or abs(sum(values) - 100) > 0.3:
            raise ValueError("Invalid three-state possession shares")
        add(
            "controlled_possession_pct",
            [values[0], values[2]],
            "percent",
            3,
            "Possession",
        )
        metadata["in_contest_pct"] = values[1]
    for label in [
        "Goals",
        "Completed Line Breaks",
        "Defensive Line Breaks",
        "Receptions in the Final Third",
        "Crosses",
        "Ball Progressions",
        "Forced Turnovers",
        "Second Balls",
    ]:
        pair(3, label)
    for label, names in [
        ("Attempts at Goal (On Target)", ["attempts", "on_target_attempts"]),
        ("Total Passes (Complete)", ["passes", "completed_passes"]),
        (
            "Defensive Pressures Applied (Direct Pressures)",
            ["total_pressures", "direct_pressures"],
        ),
    ]:
        found = re.search(
            r"^\s*(\d+)\s*\((\d+)\)\s+"
            + re.escape(label)
            + r"\s+(\d+)\s*\((\d+)\)\s*$",
            basic,
            re.M,
        )
        if found:
            a, b, c, d = map(int, found.groups())
            add(names[0], [a, c], "count", 3, label)
            add(names[1], [b, d], "count", 3, label)
    pair(3, "Pass Completion %", r"(\d+(?:\.\d+)?)\s*%", "percent")
    pair(3, "Total Distance Covered", r"(\d+(?:\.\d+)?)\s*km", "km")
    for state, labels in PHASES.items():
        for label in labels:
            pair(
                4,
                label,
                r"(\d+(?:\.\d+)?)%",
                "percent",
                state + "_" + re.sub(r"\W+", "_", label.lower()).strip("_"),
            )
    for i, text in enumerate(pages, 1):
        if "Defensive Pressure" in text and "Avg Pressure Duration" in text:
            for label, unit, value in [
                ("Avg Pressure Duration", "seconds", r"(\d+(?:\.\d+)?)s"),
                ("Ball Recovery Time", "seconds", r"(\d+(?:\.\d+)?)s"),
                ("Pushing on into Pressing", "count", r"(\d+)"),
                ("Pressing Direction Inside", "count", r"(\d+)"),
                ("Pressing Direction Outside", "count", r"(\d+)"),
            ]:
                pair(i, label, value, unit)
    names = {r["metric"] for r in metrics}
    if (
        not {
            "controlled_possession_pct",
            "passes",
            "completed_passes",
            "attempts",
            "in_possession_long_ball",
            "out_of_possession_high_press",
        }
        <= names
    ):
        raise ValueError(
            "Core report fields missing; retain raw report instead of zero filling"
        )
    return metadata, metrics


def shape_metrics(xml, layout_pages, teams):
    tree = ET.fromstring(xml)
    result = []
    counts = {"attack": 0, "defense": 0}
    for page_number, page in enumerate(
        tree.iter("{http://www.w3.org/1999/xhtml}page"), 1
    ):
        text = layout_pages[page_number - 1]
        kind = (
            "attack"
            if "In Possession Line Height & Team Length" in text
            else "defense" if "Defensive Line Height & Team Length" in text else None
        )
        if kind is None:
            continue
        side = counts[kind]
        counts[kind] += 1
        if side > 1 or re.sub(r"\W+", "", teams[side].lower()) not in re.sub(
            r"\W+", "", text.lower()
        ):
            continue
        columns = [[], [], []]
        width = float(page.attrib["width"])
        for word in page.iter("{http://www.w3.org/1999/xhtml}word"):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)m", word.text or "")
            if match:
                x = (float(word.attrib["xMin"]) + float(word.attrib["xMax"])) / 2
                if 0 <= x < width:
                    columns[min(2, int(3 * x / width))].append((x, float(match[1])))
        states = (
            ["build_up_low", "build_up_mid", "final_third"]
            if kind == "attack"
            else ["high_block_press", "mid_block", "low_block"]
        )
        for state, values in zip(states, columns):
            if len(values) != 3 or any(not 0 < v <= 120 for _, v in values):
                continue
            for name, (_, value) in zip(
                ["team_length", "team_width", "line_height"], sorted(values)
            ):
                result.append(
                    {
                        "team_slot": side,
                        "metric": kind + "_" + state + "_" + name,
                        "value": value,
                        "unit": "m",
                        "page": page_number,
                        "source_label": "dimension annotation; within-panel x order verified against source diagrams",
                    }
                )
    return result


def physical_rows(pages, teams):
    """Keep report shirt/name identities and the provider's speed-zone definitions."""
    fields = [
        "total_distance_m",
        "zone_1_0_7_kmh_m",
        "zone_2_7_15_kmh_m",
        "zone_3_15_20_kmh_m",
        "zone_4_20_25_kmh_m",
        "zone_5_above_25_kmh_m",
    ]
    result = []
    seen = set()
    for page, text in enumerate(pages, 1):
        title = re.search(r"^\s*Physical Data\s+(.+?)\s*$", text, re.M)
        if not title:
            continue
        if title[1].strip() not in teams:
            raise ValueError("Physical data team identity mismatch")
        side = teams.index(title[1].strip())
        if not all(
            term in text
            for term in [
                "Zone 1: 0-7",
                "Zone 2: 7-15",
                "Zone 3: 15-20",
                "Zone 4: 20-25",
                "Zone 5: 25+",
                "Top Speed",
            ]
        ):
            raise ValueError("Physical speed zone definitions changed")
        definition = re.search(r"\(Zone (\d)\)\s+\(Zone (\d(?: & \d)?)\)", text)
        if not definition or definition.groups() not in {("3", "4 & 5"), ("4", "5")}:
            raise ValueError("Physical run-count labels cannot be verified")
        count_fields = [
            "high_speed_runs_zone_" + definition[1] + "_count",
            "sprints_zone_" + definition[2].replace(" & ", "_") + "_count",
        ]
        found = 0
        for line in text.splitlines():
            row = re.fullmatch(
                r"\s*(\d{1,2})\s+(.+?)\s+((?:\d+(?:\.\d+)?\s+){8}\d+(?:\.\d+)?)\s*",
                line,
            )
            if not row:
                continue
            number = int(row[1])
            values = [float(v) for v in row[3].split()]
            if (side, number) in seen or abs(values[0] - sum(values[1:6])) > 0.6:
                raise ValueError(
                    "Physical player row duplicate or distance zones inconsistent"
                )
            seen.add((side, number))
            found += 1
            result.append(
                {
                    "team_slot": side,
                    "shirt_number": number,
                    "player_name": row[2].strip(),
                    "page": page,
                    "count_labels_as_reported": {
                        "high_speed_runs": "Zone " + definition[1],
                        "sprints": "Zone " + definition[2],
                    },
                    **dict(zip(fields + count_fields + ["top_speed_kmh"], values)),
                }
            )
        if not found:
            raise ValueError("Physical page has no readable player rows")
    return result


def ingest(database, root, series):
    root = Path(root).resolve() / series
    manifest = json.loads((root / "manifest.json").read_text())
    settings = SERIES[series]
    parsed = []
    errors = []
    with Warehouse(database) as db:
        ds = db.dataset(
            "fifa-reports:" + series,
            "tracking_derived_report",
            "fifa_training_centre",
            str(settings["edition"]),
        )
        db.connection.executescript(
            "CREATE TABLE IF NOT EXISTS fifa_report_matches(series TEXT,match_number INTEGER,asset_id INTEGER REFERENCES assets(id),payload TEXT,PRIMARY KEY(series,match_number));CREATE TABLE IF NOT EXISTS fifa_report_metrics(asset_id INTEGER REFERENCES assets(id),team_slot INTEGER,metric TEXT,value REAL,unit TEXT,page INTEGER,source_label TEXT,PRIMARY KEY(asset_id,team_slot,metric,page));CREATE TABLE IF NOT EXISTS fifa_report_pages(asset_id INTEGER REFERENCES assets(id),page INTEGER,layout_text TEXT,PRIMARY KEY(asset_id,page));CREATE TABLE IF NOT EXISTS fifa_report_physical(asset_id INTEGER REFERENCES assets(id),team_slot INTEGER,shirt_number INTEGER,player_name TEXT,page INTEGER,payload TEXT,PRIMARY KEY(asset_id,team_slot,shirt_number));"
        )
        for record in manifest["files"]:
            candidate = root / record["path"]
            path = candidate.resolve()
            if (
                candidate.is_symlink()
                or not path.is_relative_to(root)
                or path.stat().st_nlink != 1
                or digest(path) != record["sha256"]
            ):
                raise ValueError("Report source path/hash mismatch")
            aid = db.asset(ds, path, record["path"])
            if "match_number" not in record:
                continue
            try:
                layout = subprocess.run(
                    ["pdftotext", "-layout", str(path), "-"],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=45,
                ).stdout
                pages = layout.split("\f")
                metadata, metrics = parse_report(
                    pages, record["match_number"], settings["edition"]
                )
                bbox = subprocess.run(
                    ["pdftotext", "-bbox", str(path), "-"],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=45,
                ).stdout
                metrics.extend(shape_metrics(bbox, pages, metadata["teams"]))
                physical = physical_rows(pages, metadata["teams"])
                metadata.update(
                    {
                        "source_url": record["url"],
                        "source_sha256": record["sha256"],
                        "report_pages": len(pages) - (not pages[-1].strip()),
                        "metrics": len(metrics),
                        "physical_player_rows": len(physical),
                        "data_grain": "whole-match tracking-derived aggregates, not raw event/trajectory coordinates",
                    }
                )
                db.connection.execute(
                    "INSERT OR REPLACE INTO fifa_report_matches VALUES(?,?,?,?)",
                    (series, record["match_number"], aid, json.dumps(metadata)),
                )
                db.connection.executemany(
                    "INSERT OR REPLACE INTO fifa_report_metrics VALUES(?,?,?,?,?,?,?)",
                    [
                        (
                            aid,
                            r["team_slot"],
                            r["metric"],
                            r["value"],
                            r["unit"],
                            r["page"],
                            r["source_label"],
                        )
                        for r in metrics
                    ],
                )
                db.connection.executemany(
                    "INSERT OR REPLACE INTO fifa_report_pages VALUES(?,?,?)",
                    [(aid, i, p) for i, p in enumerate(pages, 1) if p.strip()],
                )
                db.connection.executemany(
                    "INSERT OR REPLACE INTO fifa_report_physical VALUES(?,?,?,?,?,?)",
                    [
                        (
                            aid,
                            r["team_slot"],
                            r["shirt_number"],
                            r["player_name"],
                            r["page"],
                            json.dumps(r),
                        )
                        for r in physical
                    ],
                )
                db.connection.execute("DELETE FROM quarantine WHERE asset_id=?", (aid,))
                db.finish(aid, "fifa_report", len(metrics))
                parsed.append(metadata)
                print(
                    "Imported", series, record["match_number"], len(metrics), flush=True
                )
            except (ValueError, subprocess.SubprocessError, ET.ParseError) as exc:
                errors.append(
                    {"match_number": record["match_number"], "error": str(exc)}
                )
                db.connection.execute(
                    "INSERT OR REPLACE INTO quarantine VALUES(?,?)", (aid, str(exc))
                )
        db.connection.commit()
        db.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    report = {
        "series": series,
        "parsed_matches": len(parsed),
        "expected_matches": settings["matches"],
        "metric_rows": sum(m["metrics"] for m in parsed),
        "physical_player_rows": sum(m["physical_player_rows"] for m in parsed),
        "errors": errors,
        "matches": parsed,
        "scope": "Official report-level metrics. Missing fields are not zero; phase shares are retained without forcing their sum to100. Controlled possession retains the third in-contest state.",
    }
    write_download(root / "ingest.json", [json.dumps(report, indent=2).encode()])
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["collect", "ingest"])
    p.add_argument("--series", choices=SERIES, default="fcwc2025")
    p.add_argument("--root", default=".tmp/open-data/fifa-reports")
    p.add_argument("--database", default="data/football.sqlite3")
    a = p.parse_args()
    r = (
        collect(a.root, a.series)
        if a.command == "collect"
        else ingest(a.database, a.root, a.series)
    )
    print(
        json.dumps(
            {k: v for k, v in r.items() if k not in {"files", "matches"}}, indent=2
        )
    )
