"""Chronological, provider-consistent event sequences from the local warehouse."""

from collections import Counter, deque
import json
from pathlib import Path
import sqlite3

import numpy as np

from app.analytics.artifacts import source_signature

LEAGUES = ("England", "France", "Germany", "Italy", "Spain")
VALID_START = "2018-01-01"
TEST_START = "2018-03-01"
SEQUENCE_LENGTH = 8
SEASON_START = "2017-07-01"
SEASON_END = "2018-07-01"
PASS_LABEL_AUDIT_FIELDS = (
    "event_rows", "pass_events", "eligible_passes", "excluded_invalid_start",
    "excluded_endpoint_count", "excluded_outcome_tags", "excluded_invalid_endpoint",
    "known_outcome_valid_start_passes", "known_outcome_valid_start_completed",
    "eligible_completed",
)


def split_matches(records, target):
    if target not in LEAGUES:
        raise ValueError("Unknown target league")
    groups = {key: [] for key in ("source", "train", "validation", "test")}
    for row in sorted(records, key=lambda r: (r["date"], r["id"])):
        if not SEASON_START <= row["date"] < SEASON_END:
            continue
        if row["league"] != target:
            if row["date"] < VALID_START:
                groups["source"].append(row)
        else:
            key = (
                "train"
                if row["date"] < VALID_START
                else "validation" if row["date"] < TEST_START else "test"
            )
            groups[key].append(row)
    return groups


def pass_sequences(events, *, audit=None):
    """Only event starts enter inputs. Current endpoints/accuracy are labels.

    Earlier opponent starts are rotated into the current team's frame. Context
    resets at match/period boundaries and contains no player/team identity.
    Clocks must be nondecreasing within each match/period; tied clocks retain
    source row order and do not establish physical or feed availability order.
    Optional audit receives fresh per-call counts after successful encoding.
    Pass population means eventName == 'Pass', before geometry/label filtering;
    exclusions are disjoint and recorded in the encoder's decision order.
    """
    history = deque(maxlen=SEQUENCE_LENGTH)
    boundary = None
    tokens, values, labels = [], [], []
    counts = Counter({key: 0 for key in PASS_LABEL_AUDIT_FIELDS})
    for event in events:
        counts["event_rows"] += 1
        is_pass = event.get("eventName") == "Pass"
        counts["pass_events"] += int(is_pass)
        key = (event["matchId"], event["matchPeriod"])
        if key != boundary:
            history.clear()
            boundary = key
        points = event.get("positions") or []
        try:
            x, y = (float(points[0][axis]) / 100 for axis in ("x", "y"))
            second = float(event["eventSec"])
            kind = int(event["eventId"])
        except (IndexError, KeyError, TypeError, ValueError):
            counts["excluded_invalid_start"] += int(is_pass)
            continue
        if not (
            np.isfinite([x, y, second]).all()
            and 0 <= x <= 1
            and 0 <= y <= 1
            and 1 <= kind <= 10
        ):
            counts["excluded_invalid_start"] += int(is_pass)
            continue
        if history and second < history[-1][3]:
            raise ValueError("Event clocks must be nondecreasing within a match period")
        history.append((kind, x, y, second, event["teamId"]))
        tags = {t.get("id") for t in event.get("tags", [])}
        if not is_pass:
            continue
        known_outcome = (1801 in tags) != (1802 in tags)
        if known_outcome:
            counts["known_outcome_valid_start_passes"] += 1
            counts["known_outcome_valid_start_completed"] += int(1801 in tags)
        if len(points) != 2:
            counts["excluded_endpoint_count"] += 1
            continue
        if not known_outcome:
            counts["excluded_outcome_tags"] += 1
            continue
        try:
            end = [float(points[1][axis]) / 100 for axis in ("x", "y")]
        except (KeyError, TypeError, ValueError):
            counts["excluded_invalid_endpoint"] += 1
            continue
        if not (np.isfinite(end).all() and all(0 <= v <= 1 for v in end)):
            counts["excluded_invalid_endpoint"] += 1
            continue
        token = np.zeros(SEQUENCE_LENGTH, dtype=np.int64)
        value = np.zeros((SEQUENCE_LENGTH, 4), dtype=np.float32)
        for j, (t, hx, hy, hs, team) in enumerate(
            history, SEQUENCE_LENGTH - len(history)
        ):
            same = team == event["teamId"]
            token[j] = t
            value[j] = [
                hx if same else 1 - hx,
                hy if same else 1 - hy,
                min(120, max(0, second - hs)) / 120,
                float(same),
            ]
        tokens.append(token)
        values.append(value)
        labels.append([float(1801 in tags), end[0] - x])
        counts["eligible_passes"] += 1
        counts["eligible_completed"] += int(1801 in tags)
    if audit is not None:
        audit.update(counts)
    return (
        np.asarray(tokens, dtype=np.int64).reshape(-1, SEQUENCE_LENGTH),
        np.asarray(values, dtype=np.float32).reshape(-1, SEQUENCE_LENGTH, 4),
        np.asarray(labels, dtype=np.float32).reshape(-1, 2),
    )


def load_dataset(database, target, train_cap):
    if train_cap <= 0:
        raise ValueError("train_cap must be positive")
    with sqlite3.connect(
        Path(database).resolve().as_uri() + "?mode=ro", uri=True
    ) as db:
        records = []
        for kind, payload in db.execute(
            "SELECT entity_type,payload FROM wyscout_entities WHERE entity_type LIKE 'matches_%'"
        ):
            league = kind.removeprefix("matches_")
            if league not in LEAGUES:
                continue
            match = json.loads(payload)
            if match.get("status") == "Played" and match.get("duration") == "Regular":
                records.append(
                    {
                        "id": match["wyId"],
                        "date": match["dateutc"][:10],
                        "league": league,
                    }
                )
        # Reject duplicated fixtures rather than duplicate their training weight.
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("Duplicate Wyscout match identifiers")
        groups = split_matches(records, target)
        data, receipts = {}, {}
        rng = np.random.default_rng(2026)
        for key, matches in groups.items():
            pieces, match_ids = [], []
            eligibility = Counter()
            for match in matches:
                events = [
                    json.loads(row[0])
                    for row in db.execute(
                        "SELECT payload FROM wyscout_events WHERE match_id=? AND period IN ('1H','2H') ORDER BY period,seconds,row_number",
                        (match["id"],),
                    )
                ]
                match_audit = {}
                piece = pass_sequences(events, audit=match_audit)
                eligibility.update(match_audit)
                if len(piece[0]):
                    pieces.append(piece)
                    match_ids.extend([match["id"]] * len(piece[0]))
            if not pieces:
                raise ValueError(f"No usable event sequences in {key}")
            combined = tuple(np.concatenate([p[i] for p in pieces]) for i in range(3))
            cap = train_cap * 2 if key == "source" else train_cap
            chosen = np.sort(
                rng.choice(len(combined[0]), min(cap, len(combined[0])), replace=False)
            )
            data[key] = tuple(array[chosen] for array in combined)
            selected_ids = {match_ids[i] for i in chosen}
            selected_matches = [r for r in matches if r["id"] in selected_ids]
            receipts[key] = {
                "sequences_available": len(combined[0]),
                "sequences_used": len(chosen),
                "matches_used": len(selected_ids),
                "date_start": min(r["date"] for r in selected_matches),
                "date_end": max(r["date"] for r in selected_matches),
                "leagues": dict(Counter(r["league"] for r in selected_matches)),
                "completion_rate": float(data[key][2][:, 0].mean()),
                "label_eligibility": {"matches_considered": len(matches), **eligibility},
                "match_ids": sorted(selected_ids),
                "sample_match_ids": [match_ids[i] for i in chosen],
            }
        return data, {
            "provider": "wyscout",
            "season": "2017/18",
            "target_league": target,
            "validation_start": VALID_START,
            "test_start": TEST_START,
            "splits": receipts,
            "inputs": source_signature(db, ("wyscout",)),
        }
