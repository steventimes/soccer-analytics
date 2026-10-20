"""Offline football 1X2 reference; stage labels do not establish availability at T."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import date
import json
import math
from pathlib import Path
import sqlite3

from app.analytics.artifacts import archive_outputs, file_digest
from app.data_service.history import digest, read_snapshot
from app.data_service.warehouse import Warehouse
from app.file_io import write_json_atomic
from app.ml.model import accuracy_score, multiclass_brier_score, multiclass_log_loss

SCHEMA = "SoccerAnalytics.MarketReference.v1"
LEAGUES = {"PL": "英超", "PD": "西甲", "BL1": "德甲", "SA": "意甲", "FL1": "法甲"}
BOOKS = ("bet365", "average", "pinnacle")
STAGES = ("pre_close", "closing")
OUTCOMES = ("away", "draw", "home")  # Same order as the existing model metrics.
PINNACLE_WARNING_FROM = "2025-07-23"
SOURCE_NOTES = "https://www.football-data.co.uk/notes.txt"
SOURCE_WARNING = "https://www.football-data.co.uk/data.php"


def implied_probabilities(quote):
    """Multiplicative normalization of a complete decimal 1X2 quote, plus overround."""
    values = [quote.get(key) for key in OUTCOMES]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 1
           for value in values):
        raise ValueError("All three decimal odds must be finite numbers greater than one")
    inverses = [1 / value for value in values]
    total = math.fsum(inverses)
    return [value / total for value in inverses], total - 1


def score(targets, probabilities, overrounds=None):
    if not targets:
        return None
    return {"log_loss": multiclass_log_loss(targets, probabilities),
            "brier": multiclass_brier_score(targets, probabilities),
            "accuracy": accuracy_score(targets, [max(range(3), key=row.__getitem__) for row in probabilities]),
            "mean_overround": None if overrounds is None else math.fsum(overrounds) / len(targets)}


def evaluate(snapshots, test_seasons=(2024, 2025, 2026)):
    """Use one matched cohort per book/league/season, with a frozen earlier-season prior."""
    groups, seen, splits = {}, set(), set()
    for snapshot in snapshots:
        league, season = snapshot["competition"], int(snapshot["season"])
        if snapshot["provider"] != "football-data.co.uk" or league not in LEAGUES:
            raise ValueError("Market reference requires identified Football-Data league snapshots")
        if (league, season) in splits:
            raise ValueError("Duplicate league-season snapshot")
        splits.add((league, season))
        rows = []
        for match in snapshot["matches"]:
            day, identity = match["source_date"], (league, match["id"])
            date.fromisoformat(day)
            goals = match["score"]["fullTime"]
            if (identity in seen or match["source"] != "football-data.co.uk" or match["status"] != "FINISHED"
                    or any(type(goals.get(key)) is not int or goals[key] < 0 for key in ("home", "away"))):
                raise ValueError("Duplicate, unfinished or invalid market match")
            seen.add(identity)
            target = 2 if goals["home"] > goals["away"] else 0 if goals["home"] < goals["away"] else 1
            rows.append({"id": match["id"], "date": day, "target": target, "odds": match.get("odds", {})})
        groups[league, season] = sorted(rows, key=lambda row: (row["date"], row["id"]))
    results, predictions = [], []
    for league in LEAGUES:
        for season in test_seasons:
            test = groups.get((league, season), [])
            if not test:
                raise ValueError(f"Missing test population: {league} {season}")
            # Postponed prior-season games after the test starts cannot enter the prior.
            train = [row for (code, year), rows in groups.items() if code == league and year < season
                     for row in rows if row["date"] < test[0]["date"]]
            if not train:
                raise ValueError("No strictly earlier results for market prior")
            counts = Counter(row["target"] for row in train)
            prior = [(counts[i] + 1) / (len(train) + 3) for i in range(3)]
            for book in BOOKS:
                excluded = dict.fromkeys(("provider_stale_warning", "incomplete_pre_close",
                                          "incomplete_closing", "incomplete_both"), 0)
                paired = []
                for row in test:
                    if book == "pinnacle" and row["date"] >= PINNACLE_WARNING_FROM:
                        excluded["provider_stale_warning"] += 1
                        continue
                    quotes = {}
                    for stage in STAGES:
                        try:
                            quotes[stage] = implied_probabilities(row["odds"].get(stage, {}).get(book, {}))
                        except ValueError:
                            pass
                    if len(quotes) != 2:
                        missing = "both" if not quotes else "closing" if "pre_close" in quotes else "pre_close"
                        excluded["incomplete_" + missing] += 1
                        continue
                    paired.append({"competition": league, "season": season, "book": book,
                                   **{k: row[k] for k in ("id", "date", "target")},
                                   "prior": prior, **{stage: quotes[stage][0] for stage in STAGES},
                                   "overrounds": {stage: quotes[stage][1] for stage in STAGES}})
                targets = [row["target"] for row in paired]
                metrics = {stage: score(targets, [row[stage] for row in paired],
                                       [row["overrounds"][stage] for row in paired]) for stage in STAGES}
                metrics["prior"] = score(targets, [prior] * len(paired))
                results.append({"competition": league, "season": season, "book": book,
                                "population": len(test), "paired_matches": len(paired), "excluded": excluded,
                                "test_start": test[0]["date"], "test_end": test[-1]["date"],
                                "training_rows": len(train), "training_end": max(row["date"] for row in train),
                                "prior_probabilities": prior, "cohort_sha256": digest([row["id"] for row in paired]),
                                "metrics": metrics})
                predictions.extend(paired)
    return results, predictions


def build(database="data/football.sqlite3", source_root=".tmp/open-data/football-csv",
          output="results/market-reference.json", private_dir=".tmp/market-reference"):
    root, private, output = Path(source_root).resolve(), Path(private_dir), Path(output)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["collections"]
    snapshots, sources, inputs = [], [], {}
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        assets = connection.execute(
            "SELECT a.id,a.path,a.sha256,a.bytes FROM assets a JOIN datasets d ON d.id=a.dataset_id "
            "WHERE d.name='football-csv' AND d.provider='football-csv' AND a.complete=1"
        ).fetchall()
    for league in LEAGUES:
        for season in range(2019, 2027):
            selected = [e for e in entries if e["competition"] == league and e["season"] == str(season)]
            relative = f"history/football-csv/{league}/{season}.json"
            path = root / relative
            if len(selected) != 1 or selected[0]["path"] != relative.removeprefix("history/"):
                raise ValueError("Missing or ambiguous snapshot in source manifest")
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Unsafe market source path")
            payload, sha = read_snapshot(path), file_digest(path)
            entry = selected[0]
            if (payload["competition"] != league or payload["season"] != str(season)
                    or payload["source"] != "football-csv" or payload["sha256"] != entry["sha256"]
                    or len(payload["matches"]) != entry["rows"]):
                raise ValueError("Market snapshot disagrees with manifest")
            found = [a for a in assets if a[1].split("@sha256:")[0] == relative and a[2] == sha and a[3] == path.stat().st_size]
            if len(found) != 1:
                raise ValueError("Market input absent or ambiguous in verified archive")
            inputs[f"{league}:{season}"] = found[0][0]
            sources.append({"competition": league, "season": season, "path": relative,
                            "sha256": sha, "snapshot_sha256": payload["sha256"],
                            "fetched_at": payload["fetched_at"], "asset_id": found[0][0]})
            snapshots.append(payload)
    results, predictions = evaluate(snapshots)
    code = (Path("app/analytics/market_reference.py"), Path("app/data_service/history.py"), Path("app/ml/model.py"))
    receipt = {"sources": sources, "manifest_sha256": file_digest(manifest_path),
               "code_sha256": {str(path): file_digest(path) for path in code},
               "policy": {"pinnacle_warning_from": PINNACLE_WARNING_FROM, "test_seasons": [2024, 2025, 2026]}}
    run_id = digest(receipt)[:20]
    report = {"schema": SCHEMA, "run_id": run_id, "class_order": list(OUTCOMES),
              "source": {"provider": "football-data.co.uk", "notes_url": SOURCE_NOTES,
                         "warning_url": SOURCE_WARNING, "input_sha256": digest(sources),
                         "snapshot_retrieved_at": max(s["fetched_at"] for s in sources)},
              "scope": "五大联赛 2024/25、2025/26 与归档中已有的 2026/27 比赛。每个联赛、赛季、报价来源使用同时具备两个阶段完整主平客赔率的相同比赛。",
              "method": "三项赔率的倒数按总和归一化。频率基线逐赛季冻结，只用更早赛季且早于测试首场的赛果，每类加 1。Brier 为三类误差平方之和，范围 0–2。",
              "limits": "普通列是收盘前报价，不代表开盘价。没有逐场报价发布时间，两个阶段均只作回顾性参考。归一化不保证真实概率。平均赔率不能视为可成交报价，超额倒数和不等于实际利润。各来源样本可能不同，不作跨来源排名，也不与其他样本上的模型直接排名。当前赛季只是归档覆盖的部分比赛。",
              "pinnacle_warning_from": PINNACLE_WARNING_FROM, "results": results}
    write_json_atomic(private / "report.json", report)
    write_json_atomic(private / "predictions.json", {"run_id": run_id, "rows": predictions})
    write_json_atomic(private / "receipt.json", receipt)
    with Warehouse(database) as db:
        try:
            archive_outputs(db, "market-reference", private, "market_reference", run_id,
                            filenames=["report.json", "predictions.json", "receipt.json"])
            db.record_analysis_inputs("market_reference", run_id, inputs.values())
            db.connection.commit()
        except BaseException:
            db.connection.rollback()
            raise
    write_json_atomic(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/football.sqlite3")
    parser.add_argument("--source-root", default=".tmp/open-data/football-csv")
    parser.add_argument("--output", default="results/market-reference.json")
    parser.add_argument("--private-dir", default=".tmp/market-reference")
    args = parser.parse_args()
    report = build(args.database, args.source_root, args.output, args.private_dir)
    print(json.dumps({"run_id": report["run_id"], "groups": len(report["results"])}))
