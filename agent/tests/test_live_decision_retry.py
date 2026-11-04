"""Live-decision retry policy tests (free-tier model chains rate-limit).

Covers the bounded fresh-session retry in ``live_routes.py``:
- classification: provider 429/5xx/timeouts retry; ``no_progress``, budget
  trips, and deterministic 4xx stay fail-closed;
- attempts: a transient failure retries once (default), a persistent one
  stops at the configured cap, and ``max_attempts=1`` disables retrying;
- HALT short-circuit: a kill switch tripped mid-attempt wins over retrying.

Tests drive the coroutine with ``asyncio.run`` (no pytest-asyncio plugin).
"""

from __future__ import annotations

import asyncio
import os

import pytest

from src.api import live_routes


def _env_overrides(monkeypatch, *, max_attempts: str = "2", max_delay: str = "0") -> None:
    monkeypatch.setenv("VIBE_TRADING_LIVE_DECISION_MAX_ATTEMPTS", max_attempts)
    monkeypatch.setenv("VIBE_TRADING_LIVE_DECISION_RETRY_MAX_DELAY_S", max_delay)


def _stream_error(code: int | None, message: str = "upstream") -> RuntimeError:
    """Mimic the attempt-failure text the session service delivers upward."""
    tail = f"{message} (code: {code})" if code is not None else message
    return RuntimeError(f"provider_stream_error provider=openai model=auto-free: {tail}")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("provider_stream_error provider=x model=y: Error (code: 429)", True),
        ("provider_stream_error provider=x model=y: Error (code: 502)", True),
        ("provider_stream_error provider=x model=y: Error (code: 400)", False),
        ("provider_stream_error provider=x model=y: Error (code: 413)", False),
        ("provider_stream_error provider=x model=y: HTML body", True),
        ("rate limit exceeded", True),
        ("quota exhausted for all fallback models", True),
        ("Request timed out after 120s", True),
        ("connection reset by peer", True),
        ("no_progress: 8 tool rounds produced no new information", False),
        ("live model spending limit or usage accounting failure", False),
        ("decision budget exceeded", False),
    ],
)
def test_decision_error_retryable_classification(text: str, expected: bool) -> None:
    assert live_routes._decision_error_retryable(text) is expected


def test_rate_limit_failure_retries_and_succeeds(monkeypatch) -> None:
    _env_overrides(monkeypatch)
    delays: list[float] = []
    attempts: dict[str, int] = {"n": 0}

    async def _sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(live_routes.asyncio, "sleep", _sleep)

    async def attempt() -> dict[str, str]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _stream_error(429, "Retry-After: 30")
        return {"status": "success"}

    result = asyncio.run(live_routes._retry_live_decision_attempt(attempt, broker="bybit-demo"))
    assert result["status"] == "success"
    assert attempts["n"] == 2
    # Retry-After (30s) is honored, then capped by the test's 0s ceiling.
    assert delays == [0.0]


def test_persistent_rate_limit_stops_at_attempt_cap(monkeypatch) -> None:
    _env_overrides(monkeypatch, max_attempts="3")
    calls: dict[str, int] = {"n": 0}

    async def attempt() -> dict[str, str]:
        calls["n"] += 1
        raise _stream_error(429, "all fallback models exhausted")

    with pytest.raises(RuntimeError):
        asyncio.run(live_routes._retry_live_decision_attempt(attempt, broker="bybit-demo"))
    assert calls["n"] == 3


def test_max_attempts_one_disables_retrying(monkeypatch) -> None:
    _env_overrides(monkeypatch, max_attempts="1")
    calls: dict[str, int] = {"n": 0}

    async def attempt() -> dict[str, str]:
        calls["n"] += 1
        raise _stream_error(429, "rate limit")

    with pytest.raises(RuntimeError):
        asyncio.run(live_routes._retry_live_decision_attempt(attempt, broker="bybit-demo"))
    assert calls["n"] == 1


def test_no_progress_failure_is_not_retried(monkeypatch) -> None:
    _env_overrides(monkeypatch)
    calls: dict[str, int] = {"n": 0}

    async def attempt() -> dict[str, str]:
        calls["n"] += 1
        raise RuntimeError(
            "no_progress: I stopped because 8 tool rounds in a row produced no new information"
        )

    with pytest.raises(RuntimeError):
        asyncio.run(live_routes._retry_live_decision_attempt(attempt, broker="bybit-demo"))
    assert calls["n"] == 1


def test_cancelled_error_never_retries(monkeypatch) -> None:
    _env_overrides(monkeypatch, max_attempts="5")

    async def attempt() -> dict[str, str]:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(live_routes._retry_live_decision_attempt(attempt, broker="bybit-demo"))


def test_halt_short_circuits_retry(monkeypatch) -> None:
    _env_overrides(monkeypatch)
    monkeypatch.setattr("src.live.halt.halt_flag_set", lambda broker=None: True)
    calls: dict[str, int] = {"n": 0}

    async def attempt() -> dict[str, str]:
        calls["n"] += 1
        raise _stream_error(429, "rate limit")

    with pytest.raises(RuntimeError):
        asyncio.run(live_routes._retry_live_decision_attempt(attempt, broker="bybit-demo"))
    assert calls["n"] == 1


def test_env_knobs_clamp() -> None:
    old = os.environ.get("VIBE_TRADING_LIVE_DECISION_MAX_ATTEMPTS")
    try:
        os.environ["VIBE_TRADING_LIVE_DECISION_MAX_ATTEMPTS"] = "99"
        assert live_routes._live_decision_max_attempts() == 5
        os.environ["VIBE_TRADING_LIVE_DECISION_MAX_ATTEMPTS"] = "junk"
        assert live_routes._live_decision_max_attempts() == 2
        os.environ["VIBE_TRADING_LIVE_DECISION_RETRY_MAX_DELAY_S"] = "9999"
        assert live_routes._live_decision_retry_max_delay_s() == 300.0
    finally:
        if old is None:
            os.environ.pop("VIBE_TRADING_LIVE_DECISION_MAX_ATTEMPTS", None)
        else:
            os.environ["VIBE_TRADING_LIVE_DECISION_MAX_ATTEMPTS"] = old
