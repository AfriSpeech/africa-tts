"""Abuse protection for a public (key-less) API: per-client rate limits and a daily budget.

A browser page (such as the demo Space) cannot keep a secret, so a public deployment lets
anonymous clients in under limits instead of relying on a key:

    AFRICA_TTS_RATE_PER_MIN       requests per minute per client IP            (default 6)
    AFRICA_TTS_CHARS_PER_HOUR     characters of text per hour per client IP    (default 3000)
    AFRICA_TTS_CHARS_PER_DAY      characters per day for all anonymous clients (default 200000)
    AFRICA_TTS_PROXY_HOPS         trusted proxies in front of the server (default 0)

Requests carrying the API key (AFRICA_TTS_API_KEY) are not limited. The counters live in the
process: with several containers each enforces its own budget, so cap the number of containers
to bound the total.
"""

import os
import threading
import time
from collections import defaultdict, deque


class LimitExceeded(Exception):
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


class Limiter:
    def __init__(self):
        self.per_min = int(os.environ.get("AFRICA_TTS_RATE_PER_MIN", "6"))
        self.chars_hour = int(os.environ.get("AFRICA_TTS_CHARS_PER_HOUR", "3000"))
        self.chars_day = int(os.environ.get("AFRICA_TTS_CHARS_PER_DAY", "200000"))
        self._reqs: dict[str, deque] = defaultdict(deque)  # ip -> request times (last 60 s)
        self._chars: dict[str, deque] = defaultdict(deque)  # ip -> (time, chars) (last hour)
        self._day, self._day_chars = time.strftime("%Y-%m-%d", time.gmtime()), 0
        self._lock = threading.Lock()

    def check(self, ip: str, n_chars: int) -> None:
        """Count one request of n_chars from ip, or raise LimitExceeded."""
        now = time.time()
        with self._lock:
            day = time.strftime("%Y-%m-%d", time.gmtime(now))
            if day != self._day:
                self._day, self._day_chars = day, 0
            if self._day_chars + n_chars > self.chars_day:
                midnight = 86400 - int(now) % 86400
                raise LimitExceeded("the free demo's daily quota is used up; try again tomorrow",
                                    midnight)
            reqs = self._reqs[ip]
            while reqs and reqs[0] < now - 60:
                reqs.popleft()
            if len(reqs) >= self.per_min:
                raise LimitExceeded(f"rate limit: {self.per_min} requests per minute",
                                    int(reqs[0] + 60 - now) + 1)
            chars = self._chars[ip]
            while chars and chars[0][0] < now - 3600:
                chars.popleft()
            used = sum(c for _, c in chars)
            if used + n_chars > self.chars_hour:
                wait = int(chars[0][0] + 3600 - now) + 1 if chars else 3600
                raise LimitExceeded(f"rate limit: {self.chars_hour} characters per hour", wait)
            reqs.append(now)
            chars.append((now, n_chars))
            self._day_chars += n_chars
            if len(self._reqs) > 10000:  # forget idle clients
                for k in [k for k, v in self._reqs.items() if not v]:
                    self._reqs.pop(k, None)
                    self._chars.pop(k, None)


def client_ip(headers, peer: str | None) -> str:
    """The caller's IP. Behind N trusted proxies (AFRICA_TTS_PROXY_HOPS, e.g. 1 on Modal or behind
    one load balancer), it is the N-th X-Forwarded-For entry from the right: entries further left
    are written by the client and can be forged. With 0 hops, the socket peer."""
    hops = int(os.environ.get("AFRICA_TTS_PROXY_HOPS", "0"))
    xff = [h.strip() for h in headers.get("x-forwarded-for", "").split(",") if h.strip()]
    if hops > 0 and len(xff) >= hops:
        return xff[-hops]
    return peer or "unknown"
