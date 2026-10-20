"""Shared retry waits for the existing read-only football download paths."""

from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import math
import random
import re
import time

import requests

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class RetryAfterLimitError(requests.RequestException):
    """The server requests a wait beyond this caller's inline retry budget."""

    def __init__(self, delay, limit, *, response=None):
        super().__init__(f"Retry-After requires {delay:g}s; inline wait limit is {limit:g}s", response=response)
        self.retry_after_seconds = delay
        self.max_retry_delay = limit


class RetryPolicy:
    def __init__(self, *, backoff_factor=.25, max_delay=60., jitter=True,
                 wall_clock=None, random_value=None):
        for value, label in ((backoff_factor, 'backoff_factor'), (max_delay, 'max_delay')):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{label} must be finite and nonnegative")
        if type(jitter) is not bool:
            raise ValueError("jitter must be a boolean")
        self.backoff_factor, self.max_delay, self.jitter = backoff_factor, max_delay, jitter
        self.wall_clock = wall_clock if wall_clock is not None else time.time
        self.random_value = random_value if random_value is not None else random.random
        if not callable(self.wall_clock) or not callable(self.random_value):
            raise ValueError("Clock and random source must be callable")

    def retry_after(self, response):
        headers = getattr(response, 'headers', None)
        raw = headers.get('Retry-After') if isinstance(headers, Mapping) else None
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            seconds = float(raw)
        except ValueError:
            now = float(self.wall_clock())
            if not math.isfinite(now):
                raise ValueError("Wall clock must return finite UTC epoch seconds")
            try:
                date = parsedate_to_datetime(raw)
                # RFC9110's obsolete asctime format has no explicit zone; it is UTC.
                if date.tzinfo is None:
                    date = date.replace(tzinfo=UTC)
                rfc850 = re.fullmatch(
                    r"[A-Za-z]+,\s+\d{2}-[A-Za-z]{3}-(\d{2})\s+\d{2}:\d{2}:\d{2}\s+GMT",
                    raw.strip(), re.IGNORECASE,
                )
                if rfc850:
                    # email.utils uses a fixed pivot; HTTP uses a rolling 50-year limit.
                    current = datetime.fromtimestamp(now, UTC)
                    try:
                        cutoff = current.replace(year=current.year + 50)
                    except ValueError:  # A leap-day clock has no matching day 50 years later.
                        cutoff = current.replace(year=current.year + 50, day=28)
                    year = cutoff.year // 100 * 100 + int(rfc850[1])
                    # Compare first: the intermediate century may have no February 29.
                    candidate = (year, date.month, date.day, date.hour, date.minute, date.second)
                    if candidate > cutoff.timetuple()[:6]:
                        year -= 100
                    date = date.replace(year=year)
                target = date.timestamp()
            except (ValueError, TypeError, OverflowError):
                return None
            seconds = max(0., target - now)
        return seconds if math.isfinite(seconds) and seconds >= 0 else None

    def delay(self, response, retry_index):
        if type(retry_index) is not int or retry_index < 0:
            raise ValueError("retry_index must be a nonnegative integer")
        hint = self.retry_after(response)
        if hint is not None:
            if hint > self.max_delay:
                raise RetryAfterLimitError(hint, self.max_delay, response=response)
            return hint  # Never shorten a valid server wait with local jitter or a cap.
        if self.backoff_factor == 0 or self.max_delay == 0:
            return 0.
        cutoff = math.log2(self.max_delay) - math.log2(self.backoff_factor)
        delay = self.max_delay if retry_index >= cutoff else math.ldexp(self.backoff_factor, retry_index)
        if self.jitter:
            draw = self.random_value()
            if isinstance(draw, bool) or not isinstance(draw, (int, float)) or not math.isfinite(draw) or not 0 <= draw <= 1:
                raise ValueError("Random source must return a finite number in[0,1]")
            delay *= draw
        return delay


def close_response(response):
    close = getattr(response, 'close', None)
    if callable(close):
        close()
