"""Physical and shape summaries over normalized provider tracking frames."""

from __future__ import annotations

import math
from collections import defaultdict

from app.data_service.tracking_frames import (
    _seconds,
    _coordinate_contract,
    _metrica_point,
    metrica_csv_frames,
    metrica_epts_frames,
    skillcorner_frames,
)


def summarize_tracking(
    frames,
    *,
    maximum_gap_seconds=0.25,
    maximum_speed_mps=12.0,
    high_intensity_mps=5.5,
    sprint_mps=7.0,
    minimum_shape_players=8,
) -> dict:
    previous, players, shapes, last_shape = {}, {}, defaultdict(list), {}
    frame_count = invalid_frames = 0
    periods = {}
    for frame in frames:
        frame_count += 1
        period, timestamp = frame.get("period"), frame.get("time")
        if period is None or timestamp is None or not math.isfinite(timestamp):
            invalid_frames += 1
            continue
        old_range = periods.get(period, (timestamp, timestamp))
        periods[period] = (min(old_range[0], timestamp), max(old_range[1], timestamp))
        by_team = defaultdict(list)
        for point in frame.get("players", []):
            key = (str(point["team"]), str(point["id"]))
            if key not in players:
                players[key] = {
                    "player_id": point["id"],
                    "team": point["team"],
                    "name": point.get("name"),
                    "detected_samples": 0,
                    "extrapolated_samples": 0,
                    "missing_samples": 0,
                    "distance_m": 0.0,
                    "observed_segment_seconds": 0.0,
                    "high_intensity_distance_m": 0.0,
                    "sprint_distance_m": 0.0,
                    "sampled_peak_speed_mps": 0.0,
                    "excluded_speed_spikes": 0,
                    "excluded_gaps": 0,
                    "speed_histogram": [0] * 121,
                    "occupancy_grid": [[0] * 20 for _ in range(12)],
                    "trajectory_example": [],
                    "sprint_bouts_ge_1s": 0,
                    "_sprint_seconds": 0.0,
                }
            row = players[key]
            x, y = point.get("x"), point.get("y")
            detected = point.get("detected")
            valid = (
                isinstance(x, (int, float))
                and isinstance(y, (int, float))
                and math.isfinite(x)
                and math.isfinite(y)
            )
            if not valid or detected is not True:
                row[
                    (
                        "extrapolated_samples"
                        if valid and detected is False
                        else "missing_samples"
                    )
                ] += 1
                previous.pop(key, None)
                if row["_sprint_seconds"] >= 1.0:
                    row["sprint_bouts_ge_1s"] += 1
                row["_sprint_seconds"] = 0.0
                continue
            row["detected_samples"] += 1
            if not point.get("goalkeeper", False):
                by_team[point["team"]].append((x, y))
            length, width = frame.get("pitch_length", 105.0), frame.get(
                "pitch_width", 68.0
            )
            ix = min(19, max(0, int((x / length + 0.5) * 20)))
            iy = min(11, max(0, int((y / width + 0.5) * 12)))
            row["occupancy_grid"][iy][ix] += 1
            if (
                period == 1
                and timestamp <= 120
                and (
                    not row["trajectory_example"]
                    or timestamp - row["trajectory_example"][-1][0] >= 0.5
                )
            ):
                row["trajectory_example"].append([timestamp, x, y])
            prior = previous.get(key)
            previous[key] = (period, timestamp, x, y)
            if prior is None:
                continue
            dt = timestamp - prior[1]
            if prior[0] != period or not 0 < dt <= maximum_gap_seconds:
                row["excluded_gaps"] += 1
                if row["_sprint_seconds"] >= 1.0:
                    row["sprint_bouts_ge_1s"] += 1
                row["_sprint_seconds"] = 0.0
                continue
            distance = math.hypot(x - prior[2], y - prior[3])
            speed = distance / dt
            if speed > maximum_speed_mps:
                row["excluded_speed_spikes"] += 1
                if row["_sprint_seconds"] >= 1.0:
                    row["sprint_bouts_ge_1s"] += 1
                row["_sprint_seconds"] = 0.0
                continue
            row["distance_m"] += distance
            row["observed_segment_seconds"] += dt
            row["high_intensity_distance_m"] += (
                distance if speed >= high_intensity_mps else 0.0
            )
            row["sprint_distance_m"] += distance if speed >= sprint_mps else 0.0
            row["sampled_peak_speed_mps"] = max(row["sampled_peak_speed_mps"], speed)
            row["speed_histogram"][min(120, int(speed * 10))] += 1
            if speed >= sprint_mps:
                row["_sprint_seconds"] += dt
            else:
                if row["_sprint_seconds"] >= 1.0:
                    row["sprint_bouts_ge_1s"] += 1
                row["_sprint_seconds"] = 0.0
        for team, points in by_team.items():
            if (
                len(points) < minimum_shape_players
                or timestamp - last_shape.get((period, team), -1.0) < 1.0
            ):
                continue
            last_shape[(period, team)] = timestamp
            xs, ys = zip(*points)
            shapes[team].append(
                {
                    "depth": max(xs) - min(xs),
                    "width": max(ys) - min(ys),
                    "centroid_x": sum(xs) / len(xs),
                    "centroid_y": sum(ys) / len(ys),
                    "players": len(points),
                }
            )
    duration = sum(end - start for start, end in periods.values())
    for row in players.values():
        if row.pop("_sprint_seconds") >= 1.0:
            row["sprint_bouts_ge_1s"] += 1
        row["valid_segment_coverage_fraction"] = (
            row["observed_segment_seconds"] / duration if duration else None
        )
        row["mean_observed_speed_mps"] = (
            row["distance_m"] / row["observed_segment_seconds"]
            if row["observed_segment_seconds"]
            else None
        )
    team_shape = [
        {
            "team": team,
            "qualified_shape_samples": len(rows),
            "mean_depth_m": sum(r["depth"] for r in rows) / len(rows),
            "mean_width_m": sum(r["width"] for r in rows) / len(rows),
            "mean_visible_players": sum(r["players"] for r in rows) / len(rows),
        }
        for team, rows in sorted(shapes.items())
    ]
    return {
        "frames": frame_count,
        "invalid_frames": invalid_frames,
        "tracked_clock_seconds": duration,
        "players": sorted(
            players.values(), key=lambda row: (str(row["team"]), str(row["player_id"]))
        ),
        "team_shape": team_shape,
        "protocol": {
            "coordinate_units": "meters, centered pitch",
            "maximum_gap_seconds": maximum_gap_seconds,
            "maximum_speed_mps": maximum_speed_mps,
            "high_intensity_mps": high_intensity_mps,
            "sprint_mps": sprint_mps,
            "minimum_shape_players": minimum_shape_players,
            "shape_sampling": "At most 1Hz, detected non-goalkeepers when role is known",
            "distance": "Sum of valid consecutive detected-position segments; no extrapolated segments",
            "interpretation": "Observed-distance estimates; visibility gaps and position noise affect coverage. Not full-match workload when coverage is incomplete.",
        },
    }
