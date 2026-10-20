"""Explicit collection TTLs; pinned research snapshots never claim live freshness."""

from dataclasses import dataclass
from datetime import UTC, datetime
import math

from app.config import default_training_seasons


def utc_now(clock=None):
    now = clock() if clock is not None else datetime.now(UTC)
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('缓存时钟必须提供带时区的时间')
    return now.astimezone(UTC)


@dataclass(frozen=True)
class SnapshotCachePolicy:
    max_age_hours: float | None = None
    active_max_age_hours: float | None = None

    def __post_init__(self):
        for value in (self.max_age_hours, self.active_max_age_hours):
            if value is None:
                continue
            try:
                valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                         and math.isfinite(float(value) * 3600) and value >= 0)
            except (OverflowError, ValueError):
                valid = False
            if not valid:
                raise ValueError('缓存寿命须为有限非负小时数；未设置时保留固定快照')

    def summary(self):
        return {'max_age_hours': self.max_age_hours, 'active_max_age_hours': self.active_max_age_hours,
                'active_season_rule': 'UTC European July rollover; active TTL overrides global TTL for current/future starting years.',
                'stale_on_error': False, 'http_revalidation': False}

    def decide(self, fetched_at, season, *, now=None, exists=True, refresh=False):
        now = utc_now(lambda: now) if now is not None else utc_now()
        active_start = int(default_training_seasons(now.date())[-1])
        hours = (self.active_max_age_hours if int(season) >= active_start and self.active_max_age_hours is not None
                 else self.max_age_hours)
        timestamp_state, age = 'missing', None
        if fetched_at is not None:
            try:
                fetched = datetime.fromisoformat(fetched_at.replace('Z', '+00:00'))
                if fetched.tzinfo is None or fetched.utcoffset() is None:
                    raise ValueError('timezone required')
                age = (now - fetched.astimezone(UTC)).total_seconds()
                timestamp_state = 'valid' if age >= 0 else 'future'
            except (AttributeError, TypeError, ValueError, OverflowError):
                timestamp_state = 'invalid'
        if refresh:
            state = 'forced'
        elif not exists:
            state = 'missing'
        elif hours is None:
            state = 'pinned'
        elif timestamp_state != 'valid' or age >= hours * 3600:
            state = 'expired'
        else:
            state = 'fresh'
        return {'state': state, 'action': 'reuse' if state in ['pinned', 'fresh'] else 'fetch',
                'checked_at': now.isoformat(), 'age_seconds': age,
                'timestamp_state': timestamp_state, 'max_age_seconds': None if hours is None else hours * 3600}
