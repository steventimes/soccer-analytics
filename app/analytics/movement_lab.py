"""Observed spatial exchanges and external load, with explicit exposure gates.

Coordinates are metres in each team's attacking orientation. These detectors
produce review candidates and match workload, not tactical intent or fitness tests.
"""

from collections import defaultdict
from itertools import combinations
import math


def smooth_segments(points):
    """Half-second chord velocities from consecutive detected 10 Hz positions.

    Missing detections, period boundaries and implausible raw speeds reset the
    filter. Acceleration is the change in smoothed scalar speed, not vector load.
    """
    window = []
    previous_velocity = None
    segments = []
    for point in points:
        period, frame, seconds, x, y = point
        if window:
            prior = window[-1]
            dt = seconds - prior[2]
            valid = (
                period == prior[0]
                and frame == prior[1] + 1
                and 0 < dt <= 0.25
                and math.hypot(x - prior[3], y - prior[4]) / dt <= 12
            )
            if not valid:
                window = []
                previous_velocity = None
        window.append(point)
        window = window[-6:]
        if len(window) != 6 or frame % 5:
            continue
        start = window[0]
        duration = seconds - start[2]
        if not 0.45 <= duration <= 0.55:
            continue
        speed = math.hypot(x - start[3], y - start[4]) / duration
        acceleration = None
        if previous_velocity and frame - previous_velocity[0] == 5:
            acceleration = (speed - previous_velocity[2]) / (
                seconds - previous_velocity[1]
            )
        segments.append(
            {
                "period": period,
                "frame": frame,
                "seconds": seconds,
                "duration": duration,
                "speed_mps": speed,
                "acceleration_mps2": acceleration,
            }
        )
        previous_velocity = (frame, seconds, speed)
    return segments


def peak_windows(segments, durations=(60, 180, 300, 600), coverage=0.8):
    """Rolling clock windows, sampled each second, within a single period.

    Distances are divided by clock duration; missing time is never scaled up.
    Overlapping near-peak windows are reported as anchor counts, not independent
    efforts or total minutes near peak.
    """
    import numpy as np

    periods = defaultdict(list)
    for row in segments:
        periods[row["period"]].append(row)
    results = {
        str(d): {
            "distance_m_per_minute": None,
            "sprint_m_per_minute": None,
            "eligible_windows": 0,
            "near_90pct_peak_anchor_count": 0,
        }
        for d in durations
    }
    values = defaultdict(list)
    for period, rows in periods.items():
        first = math.floor(min(r["seconds"] - r["duration"] for r in rows))
        last = math.floor(max(r["seconds"] for r in rows))
        buckets = np.zeros((last - first + 1, 3))
        for row in rows:
            # Split a segment at second boundaries before accumulating exposure.
            start, end = row["seconds"] - row["duration"], row["seconds"]
            for second in range(math.floor(start), math.ceil(end)):
                dt = max(0.0, min(end, second + 1) - max(start, second))
                speed = row["speed_mps"]
                buckets[second - first] += [
                    dt,
                    dt * speed,
                    dt * speed if speed > 25 / 3.6 else 0,
                ]
        cumulative = np.vstack([np.zeros(3), np.cumsum(buckets, axis=0)])
        for duration in durations:
            for end in range(first + duration, last + 1):
                totals = cumulative[end - first] - cumulative[end - duration - first]
                if totals[0] + 1e-6 >= coverage * duration:
                    values[str(duration)].append(
                        (
                            float(totals[1] * 60 / duration),
                            float(totals[2] * 60 / duration),
                            float(totals[0] / duration),
                            period,
                            end,
                        )
                    )
    for duration, windows in values.items():
        peak = max(windows, key=lambda w: w[0])
        results[duration] = {
            "distance_m_per_minute": peak[0],
            "sprint_m_per_minute": max(w[1] for w in windows),
            "coverage_fraction_at_distance_peak": peak[2],
            "period": peak[3],
            "end_seconds": peak[4],
            "eligible_windows": len(windows),
            "near_90pct_peak_anchor_count": sum(w[0] >= 0.9 * peak[0] for w in windows),
        }
    return results


def rotation_candidates(snapshots, cooldown_seconds=20):
    """Persistent changes of pair order during five observed possession samples.

    Both players must move, stay nearby, remain detected throughout, and swap
    stable order before/after the crossing. Collective movement alone cannot
    trigger a candidate. The decision clock is the fifth sample, after confirmation.
    """
    candidates, last = [], {}
    for index in range(4, len(snapshots)):
        window = snapshots[index - 4 : index + 1]
        period = window[0]["period"]
        if any(s["period"] != period for s in window):
            continue
        if any(b["frame"] - a["frame"] != 10 for a, b in zip(window, window[1:])):
            continue
        team = window[0]["owner"]
        if team is None or any(s["owner"] != team for s in window):
            continue
        shared = set.intersection(*(set(s["teams"].get(team, {})) for s in window))
        for a, b in combinations(sorted(shared), 2):
            key = (period, team, a, b)
            frame = window[-1]["frame"]
            if frame - last.get(key, -10000) < cooldown_seconds * 10:
                continue
            first_a, first_b = window[0]["teams"][team][a], window[0]["teams"][team][b]
            end_a, end_b = window[-1]["teams"][team][a], window[-1]["teams"][team][b]
            if any(
                math.dist(pa, pb) > 25
                for pa, pb in [(first_a, first_b), (end_a, end_b)]
            ):
                continue
            if min(math.dist(first_a, end_a), math.dist(first_b, end_b)) < 3:
                continue
            for axis, label in [(0, "depth"), (1, "lateral")]:
                differences = [
                    s["teams"][team][a][axis] - s["teams"][team][b][axis]
                    for s in window
                ]
                before, after = differences[:2], differences[-2:]
                stable = (
                    min(abs(v) for v in before + after) >= 3
                    and before[0] * before[1] > 0
                    and after[0] * after[1] > 0
                    and before[0] * after[0] < 0
                )
                if (
                    not stable
                    or max(
                        abs(first_a[1 - axis] - first_b[1 - axis]),
                        abs(end_a[1 - axis] - end_b[1 - axis]),
                    )
                    > 15
                ):
                    continue
                candidates.append(
                    {
                        "period": period,
                        "team_id": team,
                        "player_ids": [a, b],
                        "axis": label,
                        "start_frame": window[0]["frame"],
                        "decision_frame": frame,
                        "start_positions": [first_a, first_b],
                        "end_positions": [end_a, end_b],
                        "kind": "observed pair-order exchange candidate",
                    }
                )
                last[key] = frame
                break
    return candidates


def bout_counts(segments, threshold, acceleration=False):
    """Count contiguous efforts of at least one second, resetting at every gap."""
    bouts = []
    current = []
    previous = None
    for row in [*segments, None]:
        value = (
            row["acceleration_mps2"]
            if row and acceleration
            else row["speed_mps"] if row else None
        )
        qualifies = value is not None and (
            value >= threshold if threshold >= 0 else value <= threshold
        )
        consecutive = not previous or (
            row
            and row["period"] == previous["period"]
            and row["frame"] - previous["frame"] == 5
        )
        if current and (not qualifies or not consecutive):
            if sum(r["duration"] for r in current) >= 1 - 1e-6:
                bouts.append(
                    {
                        "period": current[0]["period"],
                        "start_frame": current[0]["frame"] - 5,
                        "end_frame": current[-1]["frame"],
                    }
                )
            current = []
        if qualifies:
            current.append(row)
        previous = row
    return bouts


def external_load(segments, expected_minutes):
    observed = sum(r["duration"] for r in segments)
    sprint_bouts = bout_counts(segments, 25 / 3.6)
    repeated = []
    observed_frames = {(r["period"], r["frame"]) for r in segments}
    for bout in sprint_bouts:
        prior = repeated[-1][-1] if repeated else None
        observed_recovery = prior and all(
            (bout["period"], frame) in observed_frames
            for frame in range(prior["end_frame"] + 5, bout["start_frame"] + 1, 5)
        )
        if (
            prior
            and bout["period"] == prior["period"]
            and bout["start_frame"] - prior["end_frame"] <= 200
            and observed_recovery
        ):
            repeated[-1].append(bout)
        else:
            repeated.append([bout])
    distances = {
        str(kmh): sum(
            r["speed_mps"] * r["duration"]
            for r in segments
            if r["speed_mps"] > kmh / 3.6
        )
        for kmh in (19.8, 20, 25, 25.2)
    }
    distance = sum(r["speed_mps"] * r["duration"] for r in segments)
    return {
        "observed_seconds": observed,
        "nominal_minutes": expected_minutes,
        "detected_segment_coverage_fraction": (
            observed / (expected_minutes * 60) if expected_minutes else None
        ),
        "smoothed_observed_distance_m": distance,
        "observed_distance_m_per_minute": (
            distance * 60 / observed if observed else None
        ),
        "distance_above_kmh": distances,
        "sprint_bouts_ge1s": len(sprint_bouts),
        "repeated_sprint_clusters_ge3": sum(len(c) >= 3 for c in repeated),
        "high_acceleration_bouts_ge1s": len(bout_counts(segments, 3, True)),
        "high_deceleration_bouts_ge1s": len(bout_counts(segments, -3, True)),
        "acceleration_observed_seconds": sum(
            r["duration"] for r in segments if r["acceleration_mps2"] is not None
        ),
        "rolling_peaks": peak_windows(segments),
    }
