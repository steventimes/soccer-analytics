from __future__ import annotations

import logging
from collections import deque
from threading import Lock
import math
import os
import time
from typing import Any, Mapping

import requests

from app.config import load_settings

logger = logging.getLogger(__name__)


class DataSourceError(RuntimeError):
    """外部数据源失败，携带端点和可重试语义供上层决策。"""

    def __init__(
        self,
        message: str,
        *,
        endpoint: str,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.endpoint = endpoint
        self.status_code = status_code
        self.retryable = retryable


class SlidingWindowRateLimiter:
    """进程内滑动窗口限流；单任务部署不再需要 Redis 基础设施。"""

    def __init__(
        self,
        limit: int = 10,
        window: float = 60.0,
        *,
        clock: Any = None,
        sleep: Any = None,
    ) -> None:
        if limit <= 0 or window <= 0:
            raise ValueError("限流参数必须大于 0")
        self.limit = limit
        self.window = float(window)
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def _purge(self, now: float) -> None:
        boundary = now - self.window
        while self._timestamps and self._timestamps[0] <= boundary:
            self._timestamps.popleft()

    def wait_if_needed(self) -> None:
        while True:
            now = self._clock()
            with self._lock:
                self._purge(now)
                if len(self._timestamps) < self.limit:
                    return
                delay = max(0.001, self.window - (now - self._timestamps[0]))
            self._sleep(delay)

    def add_request(self) -> None:
        now = self._clock()
        with self._lock:
            self._purge(now)
            self._timestamps.append(now)


class FootballDataClient:
    source_name = "football-data.org"
    RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        api_key: str | None = None,
        *,
        session: Any = None,
        limiter: Any = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        backoff_factor: float = 0.25,
        sleep: Any = None,
    ) -> None:
        settings = load_settings()
        self.api_key = api_key if api_key is not None else os.getenv("FOOTBALL_DATA_API_KEY")
        self.base_url = "https://api.football-data.org/v4"
        self.session = session or requests.Session()
        self.limiter = limiter or SlidingWindowRateLimiter()
        self.timeout = timeout if timeout is not None else settings.request_timeout_seconds
        self.max_retries = max_retries if max_retries is not None else settings.request_retries
        self.backoff_factor = backoff_factor
        self._sleep = sleep or time.sleep

        if self.timeout <= 0:
            raise ValueError("timeout 必须大于 0")
        if self.max_retries < 0:
            raise ValueError("max_retries 不能小于 0")
        if self.backoff_factor < 0:
            raise ValueError("backoff_factor 不能小于 0")

    def _retry_delay(self, response: Any, retry_index: int) -> float:
        retry_after = getattr(response, "headers", {}).get("Retry-After")
        if retry_after is not None:
            try:
                parsed = float(retry_after)
                if math.isfinite(parsed) and parsed >= 0:
                    return parsed
            except (TypeError, ValueError):
                pass
        return self.backoff_factor * (2**retry_index)

    def _record_request(self) -> None:
        try:
            self.limiter.add_request()
        except Exception as exc:
            logger.warning("限流计数器失败，继续处理响应: %s", exc)

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized_endpoint = endpoint.strip("/")
        if not self.api_key:
            raise DataSourceError(
                "FOOTBALL_DATA_API_KEY 未配置",
                endpoint=normalized_endpoint,
                retryable=False,
            )

        url = f"{self.base_url}/{normalized_endpoint}"
        headers = {"X-Auth-Token": self.api_key}

        for attempt in range(self.max_retries + 1):
            try:
                self.limiter.wait_if_needed()
            except Exception as exc:
                logger.warning("限流等待失败，继续请求: %s", exc)

            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                self._record_request()
                if attempt < self.max_retries:
                    self._sleep(self.backoff_factor * (2**attempt))
                    continue
                raise DataSourceError(
                    f"请求失败: {type(exc).__name__}",
                    endpoint=normalized_endpoint,
                    retryable=True,
                ) from exc

            self._record_request()
            status_code = int(response.status_code)
            if 200 <= status_code < 300:
                try:
                    payload = response.json()
                except (ValueError, TypeError) as exc:
                    raise DataSourceError(
                        "数据源返回了无效 JSON",
                        endpoint=normalized_endpoint,
                        status_code=status_code,
                        retryable=False,
                    ) from exc
                if not isinstance(payload, dict):
                    raise DataSourceError(
                        "数据源 JSON 顶层必须是对象",
                        endpoint=normalized_endpoint,
                        status_code=status_code,
                        retryable=False,
                    )
                return payload

            retryable = status_code in self.RETRYABLE_STATUS_CODES
            if retryable and attempt < self.max_retries:
                self._sleep(self._retry_delay(response, attempt))
                continue
            raise DataSourceError(
                f"football-data 返回 HTTP {status_code}",
                endpoint=normalized_endpoint,
                status_code=status_code,
                retryable=retryable,
            )

        raise AssertionError("重试循环不应到达此处")

    @staticmethod
    def _matches(payload: Mapping[str, Any], *, endpoint: str) -> list[dict[str, Any]]:
        matches = payload.get("matches")
        if not isinstance(matches, list) or any(
            not isinstance(match, dict) for match in matches
        ):
            raise DataSourceError(
                "数据源 matches 字段必须是对象数组",
                endpoint=endpoint,
                retryable=False,
            )
        return matches

    def fetch_matches(
        self,
        competition_code: str,
        *,
        season: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if season:
            params["season"] = season
        if status:
            params["status"] = status
        endpoint = f"competitions/{competition_code.upper()}/matches"
        payload = self.get(
            endpoint,
            params or None,
        )
        return self._matches(payload, endpoint=endpoint)

    def fetch_scheduled_matches(
        self,
        competition_code: str,
        date_from: str,
        date_to: str,
    ) -> list[dict[str, Any]]:
        endpoint = f"competitions/{competition_code.upper()}/matches"
        payload = self.get(
            endpoint,
            {
                "dateFrom": date_from,
                "dateTo": date_to,
            },
        )
        # v4 将已确定开球时间的比赛标记为 TIMED；一次按日期抓取后筛选。
        return [
            match for match in self._matches(payload, endpoint=endpoint)
            if match.get("status") in {"SCHEDULED", "TIMED"}
        ]
