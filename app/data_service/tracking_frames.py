"""Provider tracking-frame readers with native clock and coordinate contracts."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def _seconds(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    fields = str(value).split(":")
    return sum(float(part) * 60**index for index, part in enumerate(reversed(fields)))


def _coordinate_contract(dimensions_source, *, provider):
    return {
        "units": "meters",
        "origin": "pitch_center",
        "x_axis": "length, positive in supplier increasing-x direction",
        "y_axis": (
            "width, positive down in Metrica normalized coordinates"
            if provider == "metrica"
            else "width, supplier native signed y; no axis flip"
        ),
        "pitch_dimensions_source": dimensions_source,
        "outside_pitch_policy": "preserved, not clipped",
        "orientation_policy": "supplier directions when available; otherwise unknown",
        "detection_provenance": (
            "unknown; legacy detected mask indicates finite coordinates only"
            if provider == "metrica"
            else "supplier is_detected; false and unknown preserved"
        ),
    }


def _metrica_point(x, y, length, width):
    return {
        "x": (x - 0.5) * length,
        "y": (y - 0.5) * width,
        "supplier_detected": None,
        "extrapolated": None,
    }


def metrica_csv_frames(home_path: Path, away_path: Path, *, match_id=None,
                       pitch_length=105.0, pitch_width=68.0,
                       attacking_directions=None):
    """Stream aligned native frames; legacy detected means finite, not supplier detection.

    Default dimensions are nominal assumptions for the normalized sample CSV.
    Optional attacking_directions maps period IDs to team-to-x-direction maps.
    """
    match_id = match_id if match_id is not None else Path(home_path).parent.name
    coordinates = _coordinate_contract(
        "assumed nominal 105x68m" if (pitch_length, pitch_width) == (105.0, 68.0)
        else "caller-supplied dimensions", provider="metrica"
    )
    with Path(home_path).open(newline="") as home, Path(away_path).open(
        newline=""
    ) as away:
        readers = [csv.reader(home), csv.reader(away)]
        headers = []
        for reader in readers:
            next(reader)
            next(reader)
            headers.append(next(reader))
        from itertools import zip_longest

        for rows in zip_longest(*readers):
            if any(row is None for row in rows) or rows[0][:3] != rows[1][:3]:
                raise ValueError("Metrica home/away tracking clocks do not align")
            players = []
            for team, row, header in zip(("home", "away"), rows, headers):
                for index in range(3, len(header) - 1, 2):
                    if not header[index].startswith("Player"):
                        continue
                    try:
                        x, y = float(row[index]), float(row[index + 1])
                    except (ValueError, IndexError):
                        x, y = math.nan, math.nan
                    players.append(
                        {
                            "id": f"{team}:{header[index]}",
                            "name": header[index],
                            "team": team,
                            **_metrica_point(x, y, pitch_length, pitch_width),
                            "detected": (
                                True if math.isfinite(x) and math.isfinite(y) else None
                            ),
                        }
                    )
            period = int(rows[0][0])
            ball_index = headers[0].index("Ball")
            try:
                bx, by = float(rows[0][ball_index]), float(rows[0][ball_index + 1])
            except (ValueError, IndexError):
                bx, by = math.nan, math.nan
            yield {
                "provider": "metrica", "match_id": match_id,
                "frame": int(rows[0][1]), "native_timestamp": rows[0][2],
                "period": period, "time": float(rows[0][2]),
                "players": players,
                "ball": {**_metrica_point(bx, by, pitch_length, pitch_width), "z": None},
                "pitch_length": pitch_length, "pitch_width": pitch_width,
                "coordinate_system": coordinates,
                "attacking_direction": (attacking_directions or {}).get(period),
                "clock_basis": "native CSV Time [s], native Period and Frame retained",
            }


def metrica_epts_frames(metadata: Path, raw: Path, *, match_id=None,
                        attacking_directions=None):
    from kloppy import metrica

    dataset = metrica.load_tracking_epts(
        meta_data=metadata, raw_data=raw, sample_rate=1.0, coordinates="metrica"
    )
    pitch = dataset.metadata.pitch_dimensions
    for frame in dataset.records:
        players = [
            {
                "id": f"{player.team.team_id}:{player.player_id}",
                "name": player.full_name,
                "team": player.team.name,
                **_metrica_point(data.coordinates.x, data.coordinates.y,
                                 pitch.pitch_length, pitch.pitch_width),
                "detected": (True if math.isfinite(data.coordinates.x)
                             and math.isfinite(data.coordinates.y) else None),
            }
            for player, data in frame.players_data.items()
            if data.coordinates is not None
        ]
        ball = frame.ball_coordinates
        direction = {"LTR": 1, "RTL": -1}.get(getattr(frame.attacking_direction, "name", None))
        directions = (attacking_directions or {}).get(frame.period.id)
        if directions is None and direction is not None:
            directions = {
                team.name: direction if getattr(team.ground, "name", None) == "HOME" else -direction
                for team in dataset.metadata.teams
                if getattr(team.ground, "name", None) in {"HOME", "AWAY"}
            } or None
        yield {
            "provider": "metrica", "match_id": match_id if match_id is not None else Path(raw).parent.name,
            "frame": frame.frame_id,
            "native_timestamp": frame.frame_id / dataset.metadata.frame_rate,
            "period_relative_time": frame.timestamp.total_seconds(),
            "period": frame.period.id,
            "time": frame.timestamp.total_seconds(),
            "players": players,
            "ball": {**_metrica_point(ball.x if ball else math.nan,
                                      ball.y if ball else math.nan,
                                      pitch.pitch_length, pitch.pitch_width),
                     "z": getattr(ball, "z", None)},
            "pitch_length": pitch.pitch_length, "pitch_width": pitch.pitch_width,
            "coordinate_system": _coordinate_contract("EPTS metadata via kloppy", provider="metrica"),
            "attacking_direction": directions,
            "clock_basis": "legacy time is kloppy period-relative seconds; native_timestamp is raw frame/frame_rate seconds",
            "orientation_source": "caller mapping" if attacking_directions else "EPTS metadata via kloppy, unknown if not supplied",
        }


def skillcorner_frames(path: Path, metadata: dict):
    lookup = {player["id"]: player for player in metadata["players"]}
    try:
        import orjson

        loads = orjson.loads
    except ImportError:
        import json

        loads = json.loads
    with Path(path).open("rb") as handle:
        for line in handle:
            frame = loads(line)
            if frame.get("period") is None or frame.get("timestamp") is None:
                continue
            players = []
            for point in frame.get("player_data", []):
                player = lookup.get(point.get("player_id"))
                if player is None:
                    continue
                players.append(
                    {
                        "id": player["id"],
                        "name": player.get("short_name"),
                        "team": player["team_id"],
                        "x": point.get("x"),
                        "y": point.get("y"),
                        "detected": point.get("is_detected"),
                        "supplier_detected": point.get("is_detected"),
                        "extrapolated": (not point["is_detected"]
                                         if isinstance(point.get("is_detected"), bool) else None),
                        "goalkeeper": (player.get("player_role") or {}).get("acronym")
                        == "GK",
                    }
                )
            sides = metadata.get("home_team_side") or []
            side = sides[frame["period"] - 1] if 0 < frame["period"] <= len(sides) else None
            direction = {"left_to_right": 1, "right_to_left": -1}.get(side)
            directions = None
            if direction is not None and metadata.get("home_team") and metadata.get("away_team"):
                directions = {str(metadata["home_team"]["id"]): direction,
                              str(metadata["away_team"]["id"]): -direction}
            ball = frame.get("ball_data") or {}
            yield {
                "provider": "skillcorner", "match_id": metadata.get("id"),
                "frame": frame.get("frame"), "native_timestamp": frame["timestamp"],
                "period": frame["period"], "time": _seconds(frame["timestamp"]),
                "players": players,
                "ball": {"x": ball.get("x"), "y": ball.get("y"), "z": ball.get("z"),
                         "supplier_detected": ball.get("is_detected"),
                         "extrapolated": (not ball["is_detected"]
                                          if isinstance(ball.get("is_detected"), bool) else None)},
                "pitch_length": metadata["pitch_length"], "pitch_width": metadata["pitch_width"],
                "coordinate_system": _coordinate_contract("supplier match metadata", provider="skillcorner"),
                "home_team_side": side, "attacking_direction": directions,
                "clock_basis": "native timestamp parsed as seconds; native period/frame retained",
            }
