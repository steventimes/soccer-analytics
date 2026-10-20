"""Build bounded, source-backed public exploration summaries from local inputs.

Run with python -m app.analytics.exploration. No downloads or active model writes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import re

from app.file_io import write_json_atomic

CORE_LEAGUES = {"PL": "英超", "PD": "西甲", "BL1": "德甲", "SA": "意甲", "FL1": "法甲"}
PLAYER_METRICS = ("passes_per90", "progressive_passes_per90", "shots_per90",
                  "goals_per90", "defensive_duels_per90", "interceptions_per90",
                  "pass_completion", "duel_win_rate")


def summarize_shot_sequences(match, events):
    """Retain ordered attacking actions through the last shot in each possession.

    StatsBomb possession IDs describe control, including opposition events.
    These are not SoccerStories' stricter phases separated by every opposition touch.
    """
    teams = (match["home_team"], match["away_team"])
    groups, unassigned = {}, dict.fromkeys(teams, 0)

    def point(value):
        if (isinstance(value, list) and len(value) >= 2
                and all(type(v) in (int, float) and math.isfinite(v) for v in value[:2])
                and 0 <= value[0] <= 120 and 0 <= value[1] <= 80):
            return value[:2]
        return None

    for event in events:
        period, team = event.get("period"), event.get("team", {}).get("name")
        if type(period) is not int or period not in (1, 2, 3, 4) or team not in teams:
            continue
        kind = event.get("type", {}).get("name")
        possession, owner, index = event.get("possession"), event.get("possession_team", {}).get("name"), event.get("index")
        valid = (type(possession) is int and possession > 0 and owner in teams
                 and type(index) is int and index > 0
                 and type(event.get("minute")) is int and event["minute"] >= 0
                 and type(event.get("second")) is int and 0 <= event["second"] < 60)
        if not valid or kind == "Shot" and owner != team:
            if kind == "Shot":
                unassigned[team] += 1
            continue
        groups.setdefault((period, possession, owner), []).append(event)
    phases = []
    for (period, possession, team), recorded in groups.items():
        recorded.sort(key=lambda e: e["index"])
        shots = [e for e in recorded if e["team"]["name"] == team and e["type"]["name"] == "Shot"]
        if not shots:
            continue
        last = shots[-1]["index"]
        recorded = [e for e in recorded if e["index"] <= last]
        actions = []
        for event in recorded:
            kind = event["type"]["name"]
            if event["team"]["name"] != team or kind not in ("Pass", "Carry", "Dribble", "Shot"):
                continue
            detail = event.get(kind.lower(), {})
            xg = detail.get("statsbomb_xg", 0) if kind == "Shot" else None
            if kind == "Shot" and (type(xg) not in (int, float) or not math.isfinite(xg) or not 0 <= xg <= 1):
                raise ValueError("Invalid shot sequence xG")
            actions.append({"index": event["index"], "minute": event["minute"], "second": event["second"],
                            "kind": kind, "player": event.get("player", {}).get("name", "未知球员"),
                            "recipient": detail.get("recipient", {}).get("name", ""),
                            "start": point(event.get("location")), "end": point(detail.get("end_location")),
                            "outcome": detail.get("outcome", {}).get("name", "Complete" if kind == "Pass" else "Recorded" if kind == "Carry" else "Unknown"),
                            "xg": xg})
        phases.append({"id": f"{period}-{possession}", "period": period, "possession": possession, "team": team,
                       "shots": len(shots), "goals": sum(a["outcome"] == "Goal" for a in actions if a["kind"] == "Shot"),
                       "xg": sum(a["xg"] for a in actions if a["kind"] == "Shot"),
                       "opponent_events": sum(e["team"]["name"] != team for e in recorded), "actions": actions})
    phases.sort(key=lambda p: (p["period"], p["actions"][0]["index"]))
    return {"schema": "Touchline.ShotSequences.v1", "phases": phases, "unassigned_shots": unassigned}


def summarize_match(match, events):
    """Aggregate locations into 10x10 provider-unit zones; preserve directed passes.

    Stoppage stays in its half's last segment. Shootouts are excluded.
    Coordinates are native StatsBomb 120x80, attacking left to right.
    """
    events = list(events)
    groups = {}
    names = {}
    for e in events:
        period = e.get("period")
        team = e.get("team", {}).get("name")
        if period not in (1, 2, 3, 4) or team not in (match["home_team"], match["away_team"]):
            continue
        start, duration = {1: (0, 45), 2: (45, 45), 3: (90, 15), 4: (105, 15)}[period]
        segment = start // 15 + min(duration // 15 - 1, max(0, (e.get("minute", start) - start) // 15))
        key = (segment, team)
        if key not in groups:
            groups[key] = {"segment": segment, "period": period, "team": team,
                           "shots": 0, "goals": 0, "xg": 0., "unlocated_shots": 0,
                           "passes": 0, "completed_passes": 0, "cells": {}, "nodes": {}, "edges": Counter()}
        row = groups[key]
        kind = e.get("type", {}).get("name")
        point = e.get("location", [])
        located = (len(point) >= 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in point[:2])
                   and 0 <= point[0] <= 120 and 0 <= point[1] <= 80)
        if kind == "Shot":
            shot = e["shot"]
            xg = float(shot.get("statsbomb_xg", 0))
            goal = int(shot.get("outcome", {}).get("name") == "Goal")
            row["shots"] += 1; row["goals"] += goal; row["xg"] += xg
            if located:
                cell = min(7, int(point[1] // 10)) * 12 + min(11, int(point[0] // 10))
                zone = row["cells"].setdefault(cell, {"cell": cell, "shots": 0, "goals": 0, "xg": 0.})
                zone["shots"] += 1; zone["goals"] += goal; zone["xg"] += xg
            else:
                row["unlocated_shots"] += 1
        elif kind == "Pass":
            row["passes"] += 1
            player, recipient = e.get("player", {}), e.get("pass", {}).get("recipient", {})
            if player.get("id") and located:
                pid = str(player["id"]); names[pid] = player["name"]
                node = row["nodes"].setdefault(pid, [0., 0., 0])
                node[0] += point[0]; node[1] += point[1]; node[2] += 1
            if not e.get("pass", {}).get("outcome"):
                row["completed_passes"] += 1
                if player.get("id") and recipient.get("id"):
                    names[str(recipient["id"])] = recipient["name"]
                    row["edges"][(str(player["id"]), str(recipient["id"]))] += 1
    segments = []
    for key, row in sorted(groups.items()):
        row["cells"] = sorted(row["cells"].values(), key=lambda c: c["cell"])
        row["edges"] = [[a, b, n] for (a, b), n in sorted(row["edges"].items())]
        segments.append(row)
    result = {"id": f"statsbomb-{match['match_id']}", "date": match["date"],
            "teams": [match["home_team"], match["away_team"]], "players": names,
            "segments": segments, "shot_sequences": summarize_shot_sequences(match, events)}
    if any('index' in event for event in events):
        from .passing_motifs import summarize
        result['passing_motifs'] = summarize(match, events)
    return result


def team_comparisons(matches):
    groups = {}; seen = set()
    for m in matches:
        league, season = m["competition"], str(m["season"])
        identity = (league, season, m["id"])
        if league not in CORE_LEAGUES or identity in seen:
            continue
        seen.add(identity)
        for side, other in (("home", "away"), ("away", "home")):
            key = (league, season, m[f"{side}_id"])
            row = groups.setdefault(key, {"id": f"football-csv:{key[2]}", "name": m[side], "league": league,
                                          "season": season, "matches": 0, "goals_for": 0, "goals_against": 0,
                                          "points": 0, "wins": 0, "clean_sheets": 0,
                                          "date_start": m["date"], "date_end": m["date"]})
            gf, ga = m[f"{side}_goals"], m[f"{other}_goals"]
            row["matches"] += 1; row["goals_for"] += gf; row["goals_against"] += ga
            row["points"] += 3 if gf > ga else int(gf == ga)
            row["wins"] += int(gf > ga); row["clean_sheets"] += int(ga == 0)
            row["date_start"] = min(row["date_start"], m["date"])
            row["date_end"] = max(row["date_end"], m["date"])
    for row in groups.values():
        n = row["matches"]
        row["metrics"] = {"points_per_match": row["points"] / n, "goals_for_per_match": row["goals_for"] / n,
                          "goals_against_per_match": row["goals_against"] / n,
                          "win_rate": row["wins"] / n, "clean_sheet_rate": row["clean_sheets"] / n}
    return sorted(groups.values(), key=lambda r: (r["league"], r["season"], r["name"]))


def validation_periods(predictions, prior):
    groups = {}
    for row in predictions:
        month = row["date"][:7]
        target, probs = row["target"], row["probabilities"]
        if target not in (0, 1, 2) or len(probs) != 3 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs) or not math.isclose(sum(probs), 1, abs_tol=1e-5):
            raise ValueError("Invalid held-out match probabilities")
        g = groups.setdefault(month, {"month": month, "count": 0, "correct": 0, "baseline_correct": 0,
                                     "log_loss_sum": 0., "brier_sum": 0., "baseline_brier_sum": 0.,
                                     "calibration": {k: [{"count": 0, "probability_sum": 0., "hits": 0} for _ in range(10)] for k in ("top", "away", "draw", "home")}})
        predicted = max(range(3), key=probs.__getitem__)
        confidence = probs[predicted]
        g["count"] += 1; g["correct"] += int(predicted == target)
        g["baseline_correct"] += int(max(range(3), key=prior.__getitem__) == target)
        g["log_loss_sum"] -= math.log(max(1e-15, probs[target]))
        g["brier_sum"] += sum((p - int(i == target)) ** 2 for i, p in enumerate(probs))
        g["baseline_brier_sum"] += sum((p - int(i == target)) ** 2 for i, p in enumerate(prior))
        for key, probability, hit in [("top", confidence, predicted == target)] + [(k, probs[i], target == i) for i, k in enumerate(("away", "draw", "home"))]:
            b = g["calibration"][key][min(9, int(probability * 10))]
            b["count"] += 1; b["probability_sum"] += probability; b["hits"] += int(hit)
    for g in groups.values():
        g["accuracy"] = g["correct"] / g["count"]
    return [groups[k] for k in sorted(groups)]


def historical_validation(matches):
    """Frozen XGBoost per league, 2018–2023 training, 2024–2025 holdout.

    Date-only kickoffs share a bucket. Rolling features may use earlier test
    results; fitted parameters never see test labels. This is retrospective.
    """
    from app.ml.feature_engineering import FeatureEngineer, MatchRecord
    from app.ml.model import XGBoostClassifier

    leagues = []
    for league, name in CORE_LEAGUES.items():
        selected = [m for m in matches if m["competition"] == league and 2018 <= int(m["season"]) <= 2025]
        engineer = FeatureEngineer()
        dataset = engineer.build_training_dataset(MatchRecord(int(m["id"]), datetime.fromisoformat(m["date"]).replace(tzinfo=UTC), "FINISHED", int(m["home_id"]), int(m["away_id"]), m["home_goals"], m["away_goals"]) for m in selected)
        by_id = {m["id"]: m for m in selected}
        train = [i for i, mid in enumerate(dataset.match_ids) if int(by_id[mid]["season"]) < 2024]
        test = [i for i, mid in enumerate(dataset.match_ids) if int(by_id[mid]["season"]) >= 2024]
        if len(train) < 100 or not test:
            continue
        training_end = max(dataset.kickoffs[i] for i in train)
        test_start = min(dataset.kickoffs[i] for i in test)
        if training_end >= test_start:
            raise ValueError("Training dates overlap holdout")
        estimator = XGBoostClassifier(engineer.features)
        targets = [dataset.targets[i] for i in train]
        estimator.fit([dataset.rows[i] for i in train], targets)
        probabilities = estimator.predict_proba([dataset.rows[i] for i in test])
        counts = Counter(targets); prior = [counts[i] / len(train) for i in range(3)]
        periods = validation_periods([{"date": dataset.kickoffs[i][:10], "target": dataset.targets[i], "probabilities": p} for i, p in zip(test, probabilities)], prior)
        leagues.append({"id": league, "name": name, "training_rows": len(train), "training_end": training_end[:10],
                        "test_start": test_start[:10], "test_end": max(dataset.kickoffs[i] for i in test)[:10],
                        "model": "XGBoost · 固定默认参数", "seed": estimator.seed, "periods": periods})
    code_paths = (Path("app/analytics/exploration.py"), Path("app/ml/feature_engineering.py"), Path("app/ml/model.py"))
    source = {"provider": "Football-Data", "source_url": "https://www.football-data.co.uk/data.php",
              "model_family": "app.ml.model.XGBoostClassifier", "parameters": {"rounds": 160, "max_depth": 3, "learning_rate": .04, "seed": 20260817},
              "code_sha256": {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}}
    return {"schema": "Touchline.HistoricalValidation.v1", "leagues": leagues, "source": source,
            "scope": "Football-Data 历史留出评估。各联赛用 2018–2023 起始赛季训练，固定权重后按日期评估 2024–2025 起始赛季。滚动特征仅读取此前已结束的比赛，同日比赛共享赛前状态。",
            "limits": "这是回顾性评估，不是当时发布的赛前预测，也不代表当前线上模型。基线始终预测训练集中最多的赛果。校准点没有置信区间；小样本分箱需谨慎解释。"}


def build(data_dir, scouting_dir, assessment_dir, output_dir):
    from app.data_service.archive import read_asset_json

    data_dir, scouting_dir, assessment_dir, output_dir = map(Path, (data_dir, scouting_dir, assessment_dir, output_dir))
    report_path = scouting_dir / "statsbomb/report.json"
    manifest_path = data_dir / "statsbomb/manifest.json"
    player_path, team_path = assessment_dir / "player-profiles.json", assessment_dir / "team-data.json"
    report, manifest, profiles, teams = [json.loads(p.read_text()) for p in (report_path, manifest_path, player_path, team_path)]
    index = {r["path"]: r for r in manifest["files"]}
    chosen = [m for m in report["matches"] if m["competition_id"] == 55 and m["season"] == "2024" and m["gender"] == "male"]
    matches = []
    for m in sorted(chosen, key=lambda m: (m["date"], m["match_id"]), reverse=True):
        summary = summarize_match(m, read_asset_json(data_dir / "statsbomb", f"data/events/{m['match_id']}.json", index))
        write_json_atomic(output_dir / "matches" / f"{summary['id']}.json", summary)
        matches.append({k: summary[k] for k in ("id", "date", "teams") } | {"score": [m["home_score"], m["away_score"]], "competition": "2024 男足欧洲杯"})
    catalog = {"schema": "Touchline.MatchExplorer.v1", "matches": matches,
               "source": "StatsBomb Open Data", "revision": manifest["revision"],
               "source_url": "https://github.com/hudl/open-data", "pitch": [120, 80],
               "scope": "2024 男足欧洲杯，本地已采集的比赛。射门按 10×10 StatsBomb 单位分区汇总，圆心是分区中心，不是精确射门位置。传球节点是所选时段的传球起点均值，不是真实站位。"}
    players = [{"id": f"wyscout:{p['entity_id']}", "name": re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), p["name"]), "league": p["league"], "season": "2017/18", "position": p["position"],
                "minutes": p["minutes"], "matches": p["matches"], "date_end": p["as_of"],
                "metrics": {k: p["observed_metrics"].get(k) for k in PLAYER_METRICS}} for p in profiles]
    comparisons = {"schema": "Touchline.Comparisons.v1", "teams": team_comparisons([m for m in teams["matches"] if int(m["season"]) >= 2022]), "players": players,
                   "team_source": "Football-Data · 2022–2026 起始赛季，2026 赛季仍在进行",
                   "player_source": "Wyscout · 2017/18 五大联赛 · 观察指标，不是能力或潜力评分"}
    validation = historical_validation(teams["matches"])
    validation["source"]["input_sha256"] = hashlib.sha256(team_path.read_bytes()).hexdigest()
    for name, payload in (("index", catalog), ("comparisons", comparisons), ("validation", validation)):
        write_json_atomic(output_dir / f"{name}.json", payload)
    receipt = {"schema": "Touchline.ExplorationBuild.v1", "built_at": datetime.now(UTC).isoformat(),
               "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (report_path, manifest_path, player_path, team_path)},
               "matches": len(matches), "team_profiles": len(comparisons["teams"]), "player_profiles": len(players),
               "validation_matches": {r["id"]: sum(p["count"] for p in r["periods"]) for r in validation["leagues"]}}
    write_json_atomic(output_dir / "build-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(".tmp/open-data"))
    parser.add_argument("--scouting-dir", type=Path, default=Path(".tmp/scouting"))
    parser.add_argument("--assessment-dir", type=Path, default=Path(".tmp/assessments"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/exploration"))
    args = parser.parse_args()
    print(json.dumps(build(args.data_dir, args.scouting_dir, args.assessment_dir, args.output_dir), ensure_ascii=False))
