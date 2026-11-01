"""Offline regression tests for the Bybit direct-SDK runner path.

Covers the four surfaces added for the Bybit runner, with no broker, agent or
API server involved:

* the connector's crypto classification and short-position signing (the mandate
  gate and the exposure caps both key on them);
* the ``sdk_runner`` seam's READ normalization and WRITE dispatch;
* ``_build_live_runner``'s sdk branch: runner key namespacing, crypto trigger and
  the pinned trading connection;
* ``POST /live/runner/start``'s ``environment`` keying.

The trigger-market test exists because a runner-key alias once reached
``live_runner_market`` — ``"bybit-demo"`` is not a connector, so it fell back to
``us_equity`` and every 24/7 crypto tick was skipped as "market closed".
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api_server
from src.api import live_routes
from src.live.mandate.model import AssetClass, InstrumentType
from src.live.runtime import sdk_runner
from src.live.runtime.triggers import due_now
from src.trading import service
from src.trading.connectors.bybit import shaping
from src.trading.profiles import profile_by_id


# --------------------------------------------------------------------------- #
# Connector classification + position sign
# --------------------------------------------------------------------------- #


def test_bybit_orders_classify_as_crypto() -> None:
    assert service._order_classification("bybit", "BTC-USDT") == (
        InstrumentType.CRYPTO,
        AssetClass.CRYPTO,
    )


def test_short_position_quantity_is_signed_negative() -> None:
    row = shaping.position_to_dict({"symbol": "BTC/USDT:USDT", "side": "short", "contracts": 2})
    assert row["quantity"] == -2
    assert row["side"] == "short"


def test_long_and_spot_position_quantities_stay_positive() -> None:
    long_row = shaping.position_to_dict({"symbol": "BTC/USDT:USDT", "side": "buy", "contracts": 1})
    spot_row = shaping.position_to_dict({"symbol": "BTC", "amount": 3})
    assert long_row["quantity"] == 1
    assert spot_row["quantity"] == 3


# --------------------------------------------------------------------------- #
# sdk_runner seam
# --------------------------------------------------------------------------- #


def test_reads_raise_on_an_error_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = sdk_runner.bind(profile_by_id("bybit-paper-trade"))
    monkeypatch.setattr(
        service, "get_positions", lambda profile_id: {"status": "error", "error": "boom"}
    )
    with pytest.raises(RuntimeError, match="boom"):
        binding.read_positions()


def test_reads_unwrap_records_and_pass_the_balance_through(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = sdk_runner.bind(profile_by_id("bybit-paper-trade"))
    monkeypatch.setattr(
        service, "get_positions", lambda profile_id: {"status": "ok", "positions": [{"symbol": "BTC-USDT"}]}
    )
    monkeypatch.setattr(
        service, "get_open_orders", lambda profile_id: {"status": "ok", "open_orders": [{"order_id": "o1"}]}
    )
    monkeypatch.setattr(service, "get_account", lambda profile_id: {"status": "ok", "equity": 5.0})
    assert binding.read_positions() == [{"symbol": "BTC-USDT"}]
    assert binding.read_open_orders() == [{"order_id": "o1"}]
    assert binding.read_balance() == {"status": "ok", "equity": 5.0}


def test_close_submits_a_reduce_only_market_order(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = sdk_runner.bind(profile_by_id("bybit-paper-trade"))
    calls: list[tuple] = []
    monkeypatch.setattr(
        service,
        "place_order",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "ok"},
    )
    binding.submit_fn({"action": "close", "symbol": "BTC-USDT", "side": "sell", "qty": 1.0})
    assert calls == [
        (("BTC-USDT", "bybit-paper-trade"), {"side": "sell", "quantity": 1.0, "order_type": "market", "reduce_only": True})
    ]


def test_cancel_passes_the_symbol_the_connector_requires(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = sdk_runner.bind(profile_by_id("bybit-paper-trade"))
    calls: list[tuple] = []
    monkeypatch.setattr(
        service,
        "cancel_order",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "ok"},
    )
    binding.submit_fn({"action": "cancel", "order_id": "o1", "symbol": "BTC-USDT"})
    assert calls == [(("o1", "bybit-paper-trade"), {"symbol": "BTC-USDT"})]


# --------------------------------------------------------------------------- #
# _build_live_runner sdk branch
# --------------------------------------------------------------------------- #


class _StubSession:
    session_id = "live-sess-1"


class _StubSessionService:
    event_bus = SimpleNamespace(emit=lambda *args, **kwargs: None)

    def create_session(self, title: str = "") -> _StubSession:
        return _StubSession()


class _StubBinding:
    read_positions = staticmethod(lambda: [])
    read_balance = staticmethod(lambda: {"status": "ok"})
    read_open_orders = staticmethod(lambda: [])

    @staticmethod
    def submit_fn(order: dict) -> dict:
        return {"status": "ok"}


@pytest.fixture
def _host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the runtime root at ``tmp_path`` and stub every host seam."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    monkeypatch.setattr(api_server, "_runner_factory", None, raising=False)
    monkeypatch.setattr(api_server, "_get_session_service", lambda: _StubSessionService())
    monkeypatch.setattr(sdk_runner, "bind", lambda profile: _StubBinding())

    def _no_adapter(broker: str):
        raise AssertionError("a broker_sdk profile must never reach the MCP adapter")

    monkeypatch.setattr(api_server, "_live_broker_adapter", _no_adapter)
    return tmp_path


@pytest.mark.parametrize(
    ("environment", "runner_key", "profile_id"),
    [("live", "bybit", "bybit-live-trade"), ("paper", "bybit-demo", "bybit-paper-trade")],
)
def test_sdk_runner_uses_crypto_trigger_and_pins_the_connection(
    _host: Path, environment: str, runner_key: str, profile_id: str
) -> None:
    runner = api_server._build_live_runner("bybit", environment=environment)
    assert runner.broker == runner_key
    assert runner._profile_id == profile_id
    assert [trigger.market for trigger in runner._triggers] == ["crypto"]
    assert runner._market_watch_ms == 900_000
    # The real point of the assertion above: a crypto trigger is due 24/7, so a
    # demo key namespaced as "bybit-demo" can never be silently skipped.
    assert due_now(runner._triggers[0], 1793551723141)


def test_runner_key_namespaces_demo_state_apart_from_live() -> None:
    assert live_routes._runner_key("bybit", "live") == "bybit"
    assert live_routes._runner_key("bybit", "paper") == "bybit-demo"


# --------------------------------------------------------------------------- #
# POST /live/runner/start environment keying
# --------------------------------------------------------------------------- #


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    monkeypatch.setattr(api_server, "_runner_tasks", {}, raising=False)
    monkeypatch.setattr(api_server, "_runner_factory", None, raising=False)
    return TestClient(api_server.app, client=("127.0.0.1", 50000))


def _valid_mandate_state(broker: str) -> api_server.ActiveMandateState:
    return api_server.ActiveMandateState(
        broker=broker,
        account_ref="demo",
        created_at="2026-11-01T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
        expires_in_seconds=10_000_000,
        expired=False,
        limits=api_server.MandateLimits(
            max_order_notional_usd=300.0,
            max_total_exposure_usd=1500.0,
            max_leverage=3.0,
            max_trades_per_day=10,
            allowed_instruments=["crypto"],
            account_funding_usd=5000.0,
        ),
    )


def test_paper_start_requires_a_demo_mandate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(api_server, "_active_mandate_state", lambda broker: None)

    response = client.post("/live/runner/start", json={"broker": "bybit", "environment": "paper"})
    assert response.status_code == 409
    assert "bybit-demo" in response.json()["detail"]


def test_paper_start_registers_the_task_under_the_demo_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    mandate_keys: list[str] = []

    def _mandate(broker: str) -> api_server.ActiveMandateState:
        mandate_keys.append(broker)
        return _valid_mandate_state(broker)

    monkeypatch.setattr(api_server, "_active_mandate_state", _mandate)
    started: list[tuple[str, str]] = []
    monkeypatch.setattr(
        api_server,
        "_build_live_runner",
        lambda broker, environment="live": started.append((broker, environment))
        or SimpleNamespace(broker=broker),
    )

    async def _noop_drive(runner) -> None:  # never spawns a real agent/broker loop
        return None

    monkeypatch.setattr(api_server, "_drive_runner", _noop_drive)

    response = client.post("/live/runner/start", json={"broker": "bybit", "environment": "paper"})
    assert response.status_code == 200
    assert response.json() == {"broker": "bybit", "started": True, "already_running": False}
    # Both the mandate lookup and the build/run call go through the demo
    # namespace; the driver task is registered there too (a noop drive pops it
    # via the done callback, hence the absent-task assertion).
    assert mandate_keys == ["bybit-demo"]
    assert started == [("bybit", "paper")]
    assert api_server._runner_tasks == {}


def test_stop_keys_off_the_demo_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)

    class _DoneTask:
        def done(self) -> bool:
            return True

    monkeypatch.setattr(api_server, "_runner_tasks", {"bybit-demo": _DoneTask()})
    response = client.post("/live/runner/stop", json={"broker": "bybit", "environment": "paper"})
    assert response.status_code == 200
    assert response.json()["stopped"] is False
