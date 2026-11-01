"""Strict observed-result replay for offline studies; legacy production recipes stay intact."""
from datetime import UTC, datetime
from dataclasses import replace
from statistics import median
from collections.abc import Mapping

from app.data_service.match_records import FeatureEngineeringError, normalize_match
from app.ml.feature_engineering import FeatureDataset, FeatureEngineer


def aware_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FeatureEngineeringError("严格时点必须显式包含时区")
    return parsed.astimezone(UTC)


def available_records(matches):
    unique = {}
    for item in matches:
        if isinstance(item, Mapping):
            if item.get("time_precision") in {"day", "date"}:
                raise FeatureEngineeringError("日期级开球时间不能用于严格时点重放")
            aware_time(item.get("utcDate", item.get("kickoff")))
        record = normalize_match(item)
        if record.status != "FINISHED":
            continue
        if record.result_available_at is None:
            raise FeatureEngineeringError(f"比赛 {record.match_id} 缺少结果实际可用时间")
        record = replace(record, kickoff=aware_time(record.kickoff),
                         result_available_at=aware_time(record.result_available_at))
        if record.match_id in unique and unique[record.match_id] != record:
            raise FeatureEngineeringError("严格重放存在冲突比赛 ID")
        unique[record.match_id] = record
    return sorted(unique.values(), key=lambda r: (r.kickoff, r.match_id))


def build_available_dataset(matches, *, as_of, engineer=None):
    """Only observed labels enter the sample; each row sees results strictly before kickoff."""
    engineer = engineer or FeatureEngineer()
    cutoff = aware_time(as_of)
    records = [r for r in available_records(matches) if r.result_available_at <= cutoff]
    updates = sorted(records, key=lambda r: (r.result_available_at, r.kickoff, r.match_id))
    states, rows, index, recent = {}, [], 0, {}

    def update(record):
        previous = {team: states[team].last_match_at for team in (record.home_team_id, record.away_team_id) if team in states}
        engineer._update_states(record, states)
        for team in (record.home_team_id, record.away_team_id):
            state = states[team]
            history = recent.setdefault(team, [])
            history.append((record.kickoff, record.match_id, state.recent[-1]))
            history.sort(key=lambda item: item[:2])
            del history[:-engineer.window]
            state.recent = [item[2] for item in history]
            state.last_match_at = max(previous.get(team) or record.kickoff, record.kickoff)

    for record in records:
        while index < len(updates) and updates[index].result_available_at < record.kickoff:
            update(updates[index])
            index += 1
        rows.append(engineer._build_features(record.home_team_id, record.away_team_id, record.kickoff, states))
    while index < len(updates):
        update(updates[index])
        index += 1
    defaults = {name: float(median([r[name] for r in rows])) if rows else engineer._neutral_default(name)
                for name in engineer.features}
    return FeatureDataset(tuple(rows), tuple(engineer._target(r) for r in records),
                          tuple(r.match_id for r in records),
                          {str(k): value.to_dict() for k, value in states.items()}, defaults,
                          tuple(r.kickoff.isoformat() for r in records),
                          tuple(r.result_available_at.isoformat() for r in records))


def training_indices(dataset, candidates, cutoff):
    """Purge labels unavailable at fit time; omitted metadata is explicit legacy behavior."""
    if not dataset.result_available_at:
        return list(candidates)
    if len(dataset.result_available_at) != len(dataset.rows):
        raise FeatureEngineeringError("结果可用时间与训练行长度不一致")
    limit = aware_time(cutoff)
    return [i for i in candidates if aware_time(dataset.result_available_at[i]) < limit]
