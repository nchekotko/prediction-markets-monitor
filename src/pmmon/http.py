"""HTTP plumbing shared by venue adapters: token-bucket rate limiting and retries.

Only GET requests against public, unauthenticated market-data endpoints are issued.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

log = logging.getLogger(__name__)

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class TokenBucket:
    """Async token bucket: ``rate`` requests/second with bursts up to ``capacity``."""

    def __init__(
        self,
        rate: float,
        capacity: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(1.0, rate)
        self._tokens = self.capacity
        self._clock = clock
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    async def acquire(self, tokens: float = 1.0) -> None:
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                await asyncio.sleep((tokens - self._tokens) / self.rate)


class ApiError(RuntimeError):
    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"HTTP {status} for {url}: {body[:200]}")
        self.status = status


class RestClient:
    """GET-only JSON client with rate limiting and exponential backoff with jitter.

    ``transport`` lets tests inject :class:`httpx.MockTransport` so no network is needed.
    """

    def __init__(
        self,
        base_url: str,
        *,
        rate_per_sec: float = 5.0,
        max_retries: int = 5,
        backoff_base: float = 0.5,
        backoff_cap: float = 20.0,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": "pmmon/0.1 (read-only research tool)"},
        )
        self.bucket = TokenBucket(rate_per_sec)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._sleep = sleep
        self.request_count = 0
        self.retry_count = 0

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.backoff_cap)
            except ValueError:
                pass
        exp = min(self.backoff_cap, self.backoff_base * 2**attempt)
        return exp * (0.5 + random.random() / 2)

    async def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        attempt = 0
        while True:
            await self.bucket.acquire()
            self.request_count += 1
            try:
                resp = await self._client.get(path, params=clean)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt >= self.max_retries:
                    raise
                delay = self._backoff(attempt, None)
                log.warning("transport error %s on %s, retry in %.2fs", exc, path, delay)
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code not in RETRY_STATUS or attempt >= self.max_retries:
                    raise ApiError(resp.status_code, str(resp.url), resp.text)
                delay = self._backoff(attempt, resp.headers.get("Retry-After"))
                log.warning("HTTP %s on %s, retry in %.2fs", resp.status_code, path, delay)
            attempt += 1
            self.retry_count += 1
            await self._sleep(delay)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> RestClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
