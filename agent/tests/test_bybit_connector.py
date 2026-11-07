"""Offline contract tests for the Bybit connector.

Offline contract tests only — no authorized demo or live run was performed:
signed reads and signed order placement remain unverified against the broker.
Every exchange is a fake ccxt object (no network, no keyring, no credentials).
"""

import ccxt
import pytest

from src.trading.connectors.bybit import sdk
from src.trading.connectors.bybit.shaping import normalize_symbol


def _demo_urls():
    return {"api": {"public": "https://api-demo.{hostname}", "v3": "https://api-demo.{hostname}"}}


def _live_urls():
    return {"api": {"public": "https://api.{hostname}", "v3": "https://api.{hostname}"}}


def _testnet_urls():
    return {"api": {"public": "https://api-testnet.{hostname}", "v3": "https://api-testnet.{hostname}"}}


class FakeClient:
    """Fake standalone ccxt exchange used only for offline shaping tests."""

    def __init__(self, urls=None, hostname="bybit.com"):
        self.urls = urls if urls is not None else _demo_urls()
        self.hostname = hostname
        self.calls: list[tuple] = []


# --- config -----------------------------------------------------------------


def test_config_defaults_to_linear_paper_and_demo_host():
    cfg = sdk.BybitConfig()
    assert cfg.profile == "paper"
    assert cfg.market_type == "linear"
    assert cfg.readonly is True
    assert cfg.environment == "paper"
    assert cfg.is_demo is True
    assert cfg.host == "https://api-demo.bybit.com"


def test_live_profile_resolves_live_host_and_not_demo():
    cfg = sdk.BybitConfig(profile="live")
    assert cfg.is_demo is False
    assert cfg.host == "https://api.bybit.com"


def test_market_type_is_spot_or_linear_only():
    assert sdk.BybitConfig.from_mapping({"market_type": "spot"}).market_type == "spot"
    assert sdk.BybitConfig.from_mapping({"market_type": "linear"}).market_type == "linear"
    with pytest.raises(sdk.BybitConfigError):
        sdk.BybitConfig.from_mapping({"market_type": "usdm"})
    with pytest.raises(sdk.BybitConfigError):
        sdk.BybitConfig.from_mapping({"market_type": "inverse"})


def test_profile_acknowledges_only_three_intents():
    with pytest.raises(sdk.BybitConfigError):
        sdk.BybitConfig.from_mapping({"profile": "testnet"})
    with pytest.raises(sdk.BybitConfigError):
        sdk.BybitConfig.from_mapping({"profile": "nope"})


def test_timeout_must_be_finite_positive():
    for value in (0, -1, float("nan"), float("inf"), "abc"):
        with pytest.raises(sdk.BybitConfigError):
            sdk.BybitConfig.from_mapping({"timeout": value})


def test_override_keys_limit_caller_supplied_overrides():
    assert set(sdk._OVERRIDE_KEYS) == {"api_key", "api_secret", "profile", "market_type", "timeout"}


def test_with_overrides_roundtrips():
    cfg = sdk.BybitConfig()
    merged = cfg.with_overrides(profile="live", market_type="spot", timeout=5.0)
    assert merged.profile == "live"
    assert merged.market_type == "spot"
    assert merged.timeout == 5.0
    assert merged.host == "https://api.bybit.com"


def test_public_config_redacts_secrets():
    cfg = sdk.BybitConfig(api_key="abcd1234", api_secret="s3cret")
    public = sdk._public_config(cfg)
    assert public["api_secret"] == "***redacted***"
    assert public["api_key"] == "abcd***"
    assert "s3cret" not in str(public)


# --- host guard -------------------------------------------------------------


def test_assert_host_accepts_resolved_demo_and_live_clients():
    sdk._assert_host(sdk.BybitConfig(), FakeClient())  # demo urls on paper config
    live_cfg = sdk.BybitConfig(profile="live")
    sdk._assert_host(live_cfg, FakeClient(urls=_live_urls()))
    sdk._assert_host(live_cfg)  # config-only path compares the URL host


def test_assert_host_rejects_testnet_resolved_client_fail_closed():
    testnet = FakeClient(urls=_testnet_urls())
    with pytest.raises(sdk.BybitConfigError, match="api-demo.bybit.com"):
        sdk._assert_host(sdk.BybitConfig(), testnet)


def test_assert_host_rejects_live_client_for_paper_profile():
    with pytest.raises(sdk.BybitConfigError, match="api-demo.bybit.com"):
        sdk._assert_host(sdk.BybitConfig(), FakeClient(urls=_live_urls()))


def test_assert_host_rejects_demo_client_for_live_profile():
    with pytest.raises(sdk.BybitConfigError, match="api.bybit.com"):
        sdk._assert_host(sdk.BybitConfig(profile="live"), FakeClient())


# --- symbol / interval shaping ----------------------------------------------


@pytest.mark.parametrize(
    ("raw", "market_type", "expected"),
    [
        ("BTC-USDT", "spot", "BTC/USDT"),
        ("btcusdt", "spot", "BTC/USDT"),
        ("BTC/USDT", "spot", "BTC/USDT"),
        ("BTC/USDT:USDT", "spot", "BTC/USDT"),
        ("BTC-USDT", "linear", "BTC/USDT:USDT"),
        ("BTCUSDT", "linear", "BTC/USDT:USDT"),
        ("BTC/USDT:USDT", "linear", "BTC/USDT:USDT"),
        ("eth-usdc", "linear", "ETH/USDC:USDC"),
    ],
)
def test_normalize_symbol_table(raw, market_type, expected):
    assert normalize_symbol(raw, market_type=market_type) == expected


def test_normalize_symbol_rejects_unparseable_naming_input():
    with pytest.raises(ValueError, match="WEIRD"):
        normalize_symbol("WEIRD")


def test_normalize_symbol_rejects_unknown_market_type():
    with pytest.raises(ValueError, match="market_type"):
        normalize_symbol("BTC-USDT", market_type="usdm")


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("1m", "1m"), ("1H", "1h"), ("4H", "4h"), ("4h", "4h"),
        ("1D", "1d"), ("1w", "1w"), ("1W", "1w"), ("1M", "1M"),
    ],
)
def test_timeframe_map_table(token, expected):
    assert sdk._TIMEFRAME_MAP[token] == expected


def test_timeframe_map_covers_documented_tokens():
    documented = {"1m", "5m", "15m", "30m", "1h", "1H", "4h", "4H", "1d", "1D", "1w", "1W", "1M"}
    assert set(documented) == set(sdk._TIMEFRAME_MAP)


def test_unknown_period_defaults_to_1d(monkeypatch):
    client = FakeClient()
    got: dict = {}

    def ohlcv(symbol, timeframe=None, limit=None):
        got.update(symbol=symbol, timeframe=timeframe, limit=limit)
        return []

    client.fetch_ohlcv = ohlcv
    monkeypatch.setattr(sdk, "_public_exchange", lambda _cfg: client)
    result = sdk.get_historical_bars("BTC-USDT", config=sdk.BybitConfig(), period="3y")
    assert got == {"symbol": "BTC/USDT:USDT", "timeframe": "1d", "limit": 90}
    assert result["period"] == "3y"


# --- read path with fake exchange -------------------------------------------


@pytest.fixture
def demo_client(monkeypatch):
    client = FakeClient(_demo_urls())
    monkeypatch.setattr(sdk, "_exchange", lambda _cfg: client)
    monkeypatch.setattr(sdk, "_public_exchange", lambda _cfg: client)
    return client


def test_get_account_snapshot_shapes_nonzero_balances(demo_client):
    demo_client.fetch_balance = lambda: {
        "info": {},
        "BTC": {"free": 0.5, "used": 0.1, "total": 0.6},
        "USDT": {"free": 0, "used": 0, "total": 0},
    }
    result = sdk.get_account_snapshot(sdk.BybitConfig())
    assert result["status"] == "ok"
    assert result["paper_guard"] == "host_separated"
    assert result["profile"] == "paper"
    assert result["is_demo"] is True
    assert result["host"] == "https://api-demo.bybit.com"
    assert result["balances"] == [
        {"asset": "BTC", "free": 0.5, "used": 0.1, "total": 0.6}
    ]


def test_get_positions_linear_uses_fetch_positions(demo_client):
    demo_client.fetch_positions = lambda: [
        {
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "contracts": 1,
            "entryPrice": 65000.0,
            "markPrice": 66000.0,
            "notional": 66000.0,
            "unrealizedPnl": 1000.0,
            "status": "open",
        }
    ]
    result = sdk.get_positions(sdk.BybitConfig())
    assert result["status"] == "ok"
    assert result["positions"][0]["symbol"] == "BTC/USDT:USDT"
    assert result["positions"][0]["quantity"] == 1
    assert result["positions"][0]["entry_price"] == 65000.0


def test_get_positions_spot_reshapes_balances(demo_client):
    demo_client.fetch_balance = lambda: {
        "BTC": {"free": 2.0, "used": 0.0, "total": 2.0},
    }
    result = sdk.get_positions(sdk.BybitConfig(market_type="spot"))
    assert result["positions"] == [
        {"symbol": "BTC", "quantity": 2.0, "free": 2.0, "used": 0.0, "source": "spot"}
    ]


def test_get_open_orders_calls_without_symbol(demo_client):
    got: list[tuple] = []

    def open_orders(*args):
        got.append(args)
        return [
            {"id": "7", "symbol": "BTC/USDT:USDT", "side": "buy", "type": "limit",
             "price": 65000, "amount": 1, "filled": 0, "remaining": 1,
             "status": "open", "timestamp": 1700000000000}
        ]

    demo_client.fetch_open_orders = open_orders
    result = sdk.get_open_orders(sdk.BybitConfig())
    assert got == [()]  # all-symbol read: no symbol argument is passed
    assert result["open_orders"][0]["order_id"] == "7"
    assert result["open_orders"][0]["symbol"] == "BTC/USDT:USDT"
    assert "open_orders_note" not in result


def test_get_open_orders_degrades_on_symbol_required_only(demo_client):
    def fail_without_symbol():
        raise ccxt.ArgumentsRequired("fetchOpenOrders requires a symbol parameter for this exchange")

    demo_client.fetch_open_orders = fail_without_symbol
    result = sdk.get_open_orders(sdk.BybitConfig())
    assert result["status"] == "ok"
    assert result["open_orders"] == []
    assert "fetch_open_orders without a symbol failed" in result["open_orders_note"]


@pytest.mark.parametrize("error", [ccxt.AuthenticationError, ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.ExchangeError])
def test_get_open_orders_propagates_real_failures_unmasked(demo_client, error):
    def fail():
        raise error("synthetic broker failure")

    demo_client.fetch_open_orders = fail
    with pytest.raises(error, match="synthetic broker failure"):
        sdk.get_open_orders(sdk.BybitConfig())


def test_get_open_orders_executions_note_when_requested(demo_client):
    demo_client.fetch_open_orders = lambda: []
    result = sdk.get_open_orders(sdk.BybitConfig(), include_executions=True)
    assert result["executions"] == []
    assert "fetch_my_trades" in result["executions_note"]


def test_get_quote_normalizes_for_linear(demo_client):
    got: list[str] = []

    def ticker(symbol):
        got.append(symbol)
        return {"bid": 64999, "ask": 65001, "last": 65000, "timestamp": 1700000000000}

    demo_client.fetch_ticker = ticker
    result = sdk.get_quote("BTC-USDT", config=sdk.BybitConfig())
    assert got == ["BTC/USDT:USDT"]
    assert result["symbol"] == "BTC/USDT:USDT"
    assert result["quote"]["last"] == 65000


def test_get_quote_spot_symbol_form(demo_client):
    got: list[str] = []

    def ticker(symbol):
        got.append(symbol)
        return {"last": 65000}

    demo_client.fetch_ticker = ticker
    result = sdk.get_quote("btcusdt", config=sdk.BybitConfig(market_type="spot"))
    assert got == ["BTC/USDT"]
    assert result["symbol"] == "BTC/USDT"


def test_get_historical_bars_passes_unified_timeframe(demo_client):
    got: dict = {}

    def ohlcv(symbol, timeframe=None, limit=None):
        got.update(symbol=symbol, timeframe=timeframe, limit=limit)
        return [[1700000000000, 1, 2, 0, 1, 10]]

    demo_client.fetch_ohlcv = ohlcv
    result = sdk.get_historical_bars("BTC-USDT", config=sdk.BybitConfig(), period="4H", limit=3)
    assert got == {"symbol": "BTC/USDT:USDT", "timeframe": "4h", "limit": 3}
    assert result["bars"][0]["close"] == 1


def test_search_instruments_matches_linear_market_only(demo_client):
    demo_client.load_markets = lambda: {
        "BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "id": "BTCUSDT", "settle": "USDT",
                          "base": "BTC", "quote": "USDT", "linear": True, "active": True,
                          "swap": True, "spot": False},
        "BTC/USDT": {"symbol": "BTC/USDT", "id": "BTCUSDT", "base": "BTC", "quote": "USDT",
                     "spot": True, "linear": None, "active": True,
                     "swap": False, "future": False, "option": False},
    }
    result = sdk.search_instruments("BTC-USDT", config=sdk.BybitConfig())
    assert len(result["instruments"]) == 1
    row = result["instruments"][0]
    assert row["native_symbol"] == "BTC/USDT:USDT"
    assert row["exchange_symbol"] == "BTCUSDT"
    assert row["exchange"] == "BYBIT"


def test_search_instruments_spot_config_excludes_linear(demo_client):
    demo_client.load_markets = lambda: {
        "BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "id": "BTCUSDT", "linear": True, "active": True},
        "BTC/USDT": {"symbol": "BTC/USDT", "id": "BTCUSDT", "spot": True, "linear": None, "active": True},
    }
    result = sdk.search_instruments("BTC-USDT", config=sdk.BybitConfig(market_type="spot"))
    assert [r["native_symbol"] for r in result["instruments"]] == ["BTC/USDT"]


def test_search_instruments_skips_inactive_markets(demo_client):
    demo_client.load_markets = lambda: {
        "BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "id": "BTCUSDT", "linear": True, "active": False},
    }
    result = sdk.search_instruments("BTC-USDT", config=sdk.BybitConfig())
    assert result["instruments"] == []


# --- guard before every broker call -----------------------------------------


def test_guard_fires_before_any_broker_call_in_reads(monkeypatch):
    broker_touched: list[str] = []

    client = FakeClient(_live_urls())
    for name in ("fetch_balance", "fetch_positions", "fetch_open_orders", "fetch_ticker"):
        setattr(client, name, lambda _name=name: broker_touched.append(_name))

    def ohlcv(symbol, timeframe=None, limit=None):
        broker_touched.append("fetch_ohlcv")
        return []

    client.fetch_ohlcv = ohlcv
    monkeypatch.setattr(sdk, "_exchange", lambda _cfg: client)
    monkeypatch.setattr(sdk, "_public_exchange", lambda _cfg: client)
    # LIVE URLs on the fake client must fail a paper profile's guard before any call.
    for call in (lambda: sdk.get_account_snapshot(), lambda: sdk.get_positions(),
                 lambda: sdk.get_open_orders(), lambda: sdk.get_quote("BTC-USDT"),
                 lambda: sdk.get_historical_bars("BTC-USDT")):
        with pytest.raises(sdk.BybitConfigError, match="api-demo.bybit.com"):
            call()
    assert broker_touched == []


def test_check_status_reports_missing_credentials(monkeypatch):
    monkeypatch.setattr(sdk, "load_config", lambda: sdk.BybitConfig())
    report = sdk.check_status(sdk.BybitConfig())
    assert report["status"] == "error"
    assert "api_key" in report["error"]


def test_check_status_ok_on_paper_demo_client(monkeypatch, demo_client):
    demo_client.fetch_balance = lambda: {"USDT": {"free": 1.0, "used": 0.0, "total": 1.0}}
    report = sdk.check_status(sdk.BybitConfig(api_key="demo-key", api_secret="demo-secret"))
    assert report["status"] == "ok"
    assert report["paper_guard"] == "host_separated"
    assert report["account"]["is_demo"] is True
    assert report["config"]["api_secret"] == "***redacted***"


def test_check_status_counts_balances_on_linear(monkeypatch, demo_client):
    demo_client.fetch_balance = lambda: {"USDT": {"free": 1.0, "used": 0.0, "total": 1.0}}
    report = sdk.check_status(sdk.BybitConfig(api_key="k", api_secret="s"))
    assert report["account"]["balances"] == 1


# --- write path with fake exchange ------------------------------------------


@pytest.fixture
def order_client(monkeypatch):
    client = FakeClient(_demo_urls())
    monkeypatch.setattr(sdk, "_exchange", lambda _cfg: client)
    monkeypatch.setattr(sdk, "_public_exchange", lambda _cfg: client)
    return client


def test_place_order_market_quantity_forwards_exact_arguments(order_client):
    got: dict = {}

    def create_order(symbol, order_type, side, amount, price=None, params=None):
        params = {} if params is None else params
        got.update(symbol=symbol, side=side, amount=amount, price=price, params=params)
        return {"id": "ord-1", "symbol": symbol, "side": side, "type": "market",
                "status": "open", "filled": 0, "amount": amount, "price": price}

    order_client.create_order = create_order
    result = sdk.place_order(sdk.BybitConfig(), symbol="BTC-USDT", side="buy", quantity=0.01)
    assert got == {"symbol": "BTC/USDT:USDT", "side": "buy", "amount": 0.01,
                   "price": None, "params": {}}
    assert result["status"] == "ok"
    assert result["order_id"] == "ord-1"
    assert result["is_demo"] is True
    assert result["paper_guard"] == "host_separated"
    assert result["order_status"] == "open"  # accepted ACK, not a fill claim


def test_place_order_notional_forwards_cost_to_ccxt(order_client):
    got: dict = {}

    def create_order(symbol, order_type, side, amount, price=None, params=None):
        params = {} if params is None else params
        got.update(params=params, price=price)
        return {"id": "i", "status": "open", "filled": 0}

    order_client.create_order = create_order
    sdk.place_order(sdk.BybitConfig(), symbol="BTC-USDT", side="sell", notional=50)
    assert got["params"] == {"cost": 50.0}
    assert got["price"] is None


def test_place_order_limit_sets_gtc_and_forwards_price(order_client):
    got: dict = {}

    def create_order(symbol, order_type, side, amount, price=None, params=None):
        params = {} if params is None else params
        got.update(amount=amount, price=price, params=params)
        return {"id": "i", "status": "open", "filled": 0}

    order_client.create_order = create_order
    sdk.place_order(sdk.BybitConfig(), symbol="BTC-USDT", side="buy",
                    quantity=0.01, order_type="limit", limit_price=65000, time_in_force="ioc")
    assert got == {"amount": 0.01, "price": 65000.0, "params": {"timeInForce": "IOC"}}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(symbol="BTC-USDT", side="sideways", quantity=1), "side must be"),
        (dict(symbol="BTC-USDT", side="buy", quantity=1, order_type="stop"), "order_type must be"),
        (dict(symbol="BTC-USDT", side="buy"), "exactly one of"),
        (dict(symbol="BTC-USDT", side="buy", quantity=1, notional=1), "exactly one of"),
        (dict(symbol="BTC-USDT", side="buy", quantity=-1), "quantity must be"),
        (dict(symbol="BTC-USDT", side="buy", quantity="bogus"), "quantity must be"),
        (dict(symbol="BTC-USDT", side="buy", notional=0), "notional must be"),
        (dict(symbol="BTC-USDT", side="buy", quantity=1, order_type="limit"), "positive 'limit_price'"),
        (dict(symbol="BTC-USDT", side="buy", notional=1, order_type="limit", limit_price=1),
         "limit orders require 'quantity'"),
        (dict(symbol="BTC-USDT", side="buy", quantity=1, order_type="limit",
              limit_price=1, time_in_force="hour"), "time_in_force must be"),
        (dict(symbol="WEIRD", side="buy", quantity=1), "could not resolve a trading pair"),
    ],
)
def test_place_order_fail_closed_validation_matrix(order_client, kwargs, match):
    broker_touched: list[str] = []
    order_client.create_order = lambda *a: broker_touched.append("create_order")
    result = sdk.place_order(sdk.BybitConfig(), **kwargs)
    assert result["status"] == "error"
    assert match in result["error"]
    assert broker_touched == []


def test_place_order_rejects_unknown_broker_param_names(order_client):
    broker_touched: list[str] = []
    order_client.create_order = lambda *a: broker_touched.append("create_order")
    result = sdk.place_order(sdk.BybitConfig(), symbol="BTC-USDT", side="buy",
                             quantity=1, tp="99999")
    assert result["status"] == "error"
    assert "unsupported Bybit order parameter 'tp'" in result["error"]
    assert broker_touched == []


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (dict(orderLinkId="desk-ref-123"), {"orderLinkId": "desk-ref-123"}),
        (dict(reduceOnly=True, positionIdx=1),
         {"reduceOnly": True, "positionIdx": 1}),
        (dict(takeProfit={"triggerPrice": 70000}, stopLoss={"triggerPrice": 60000}),
         {"takeProfit": {"triggerPrice": 70000}, "stopLoss": {"triggerPrice": 60000}}),
    ],
)
def test_place_order_forwards_validated_broker_params(order_client, extra, expected):
    got: dict = {}

    def create_order(symbol, order_type, side, amount, price=None, params=None):
        params = {} if params is None else params
        got.update(params=params)
        return {"id": "i", "status": "open", "filled": 0}

    order_client.create_order = create_order
    result = sdk.place_order(sdk.BybitConfig(), symbol="BTC-USDT", side="sell",
                             quantity=1, **extra)
    assert result["status"] == "ok"
    assert got["params"] == expected


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (dict(orderLinkId=""), "orderLinkId must be"),
        (dict(orderLinkId="x" * 37), "at most 36"),
        (dict(reduceOnly="yes"), "reduceOnly must be"),
        (dict(positionIdx=3), "positionIdx must be"),
        (dict(positionIdx=True), "positionIdx must be"),
        (dict(takeProfit={"price": 1}), "triggerPrice"),
        (dict(stopLoss="60000"), "must be a mapping"),
    ],
)
def test_place_order_rejects_malformed_broker_params(order_client, extra, match):
    broker_touched: list[str] = []
    order_client.create_order = lambda *a: broker_touched.append("create_order")
    result = sdk.place_order(sdk.BybitConfig(), symbol="BTC-USDT", side="buy",
                             quantity=1, **extra)
    assert result["status"] == "error"
    assert match in result["error"]
    assert broker_touched == []


def test_place_order_guard_fires_before_submit(order_client, monkeypatch):
    order_client.urls = _live_urls()  # LIVE URLs vs paper config
    order_client.create_order = lambda *a: (_ for _ in ()).throw(AssertionError("must not reach broker"))
    result = sdk.place_order(sdk.BybitConfig(), symbol="BTC-USDT", side="buy", quantity=1)
    assert result["status"] == "error"
    assert "api-demo.bybit.com" in result["error"]


def test_place_order_propagates_broker_error_unmasked(order_client):
    def fail(*a):
        raise ccxt.NetworkError("synthetic submit failure")

    order_client.create_order = fail
    result = sdk.place_order(sdk.BybitConfig(), symbol="BTC-USDT", side="buy", quantity=1)
    assert result["status"] == "error"
    assert "synthetic submit failure" in result["error"]


def test_cancel_order_requires_symbol(order_client):
    result = sdk.cancel_order(sdk.BybitConfig(), "9", symbol=None)
    assert result["status"] == "error"
    assert result["error"] == "Bybit cancel requires symbol."
    result = sdk.cancel_order(sdk.BybitConfig(), "9", symbol="  ")
    assert result["status"] == "error"
    assert result["error"] == "Bybit cancel requires symbol."


def test_cancel_order_requires_order_id(order_client):
    result = sdk.cancel_order(sdk.BybitConfig(), "", symbol="BTC-USDT")
    assert result["status"] == "error"
    assert "order_id is required" in result["error"]


def test_cancel_order_normalizes_symbol_and_confirms(order_client):
    got: dict = {}

    def cancel(order_id, symbol):
        got.update(order_id=order_id, symbol=symbol)
        return {"id": order_id, "symbol": symbol, "side": "buy", "status": "canceled"}

    order_client.cancel_order = cancel
    result = sdk.cancel_order(sdk.BybitConfig(), "9", symbol="BTC-USDT")
    assert got == {"order_id": "9", "symbol": "BTC/USDT:USDT"}
    assert result["status"] == "ok"
    assert result["order_status"] == "canceled"
    assert result["is_demo"] is True


def test_cancel_order_guard_fires_before_broker_call(order_client):
    order_client.urls = _live_urls()
    order_client.cancel_order = lambda *a: (_ for _ in ()).throw(AssertionError("must not reach broker"))
    result = sdk.cancel_order(sdk.BybitConfig(), "9", symbol="BTC-USDT")
    assert result["status"] == "error"
    assert "api-demo.bybit.com" in result["error"]


def test_cancel_order_propagates_broker_error_unmasked(order_client):
    def fail(*a):
        raise ccxt.AuthenticationError("synthetic cancel failure")

    order_client.cancel_order = fail
    result = sdk.cancel_order(sdk.BybitConfig(), "9", symbol="BTC-USDT")
    assert result["status"] == "error"
    assert "synthetic cancel failure" in result["error"]


def _paper_service_profile(monkeypatch, name: str = "bybit-paper-trade"):
    from src.trading import service

    for profile in service.list_profiles():
        if profile.id == name:
            return profile
    raise AssertionError(f"profile missing in registry: {name}")


def test_service_place_attaches_takeprofit_stoploss_to_bybit(monkeypatch, order_client):
    from src.trading import service

    got: dict = {}

    def create_order(symbol, order_type, side, amount, price=None, params=None):
        got.update(params={} if params is None else params)
        return {"id": "ord-tp", "symbol": symbol, "side": side, "type": order_type,
                "status": "open", "filled": 0, "amount": amount, "price": price}

    order_client.create_order = create_order
    result = service.place_order(
        "BTC-USDT", "bybit-paper-trade", side="buy", quantity=0.01,
        take_profit_price=85000.0, stop_loss_price=72000.0,
    )
    assert result["status"] == "ok"
    assert got["params"] == {
        "takeProfit": {"triggerPrice": 85000.0},
        "stopLoss": {"triggerPrice": 72000.0},
    }
    assert result["order_id"] == "ord-tp"


def test_service_paper_placement_writes_order_shaped_ledger_evidence(monkeypatch, order_client):
    """A demo fill must leave evidence reconcile can adopt (symbol + tool)."""
    import json as _json

    import src.live.audit as audit_mod
    from src.trading import service

    ledger = audit_mod.audit_ledger_path()
    before = len(ledger.read_text(encoding="utf-8").splitlines()) if ledger.is_file() else 0

    def create_order(symbol, order_type, side, amount, price=None, params=None):
        return {"id": "ord-ev", "symbol": symbol, "side": side, "type": order_type,
                "status": "open", "filled": 0, "amount": amount, "price": price}

    order_client.create_order = create_order
    result = service.place_order("ETH-USDT", "bybit-paper-trade", side="buy", quantity=0.11)
    assert result["status"] == "ok"

    new_rows = [
        _json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()[before:]
        if line.strip()
    ]
    assert len(new_rows) == 1
    row = new_rows[0]
    assert row["server"] == "bybit-demo"
    assert row["kind"] == "order_placed"
    assert row["remote_tool"] == "place_order"
    assert row["outcome"] == "accepted"
    assert row["broker_request"]["symbol"] == "ETH-USDT"
    assert row["broker_request"]["quantity"] == 0.11


def test_service_place_refuses_attached_params_on_binance(monkeypatch):
    from src.trading import service

    result = service.place_order(
        "BTCUSDT", "binance-paper-trade", side="buy", quantity=0.001,
        stop_loss_price=70000.0,
    )
    assert result["status"] == "error"
    assert "stopLoss" in result["error"]
    assert "binance" in result["error"]


def test_tool_place_rejects_non_boolean_reduce_only(monkeypatch):
    import json

    from src.tools.trading_connector_tool import TradingPlaceOrderTool

    payload = json.loads(
        TradingPlaceOrderTool().execute(
            symbol="BTC-USDT", connection="bybit-paper-trade", side="buy",
            quantity=0.01, reduce_only="yes",
        )
    )
    assert payload["status"] == "error"
    assert "reduce_only" in payload["error"]


def test_tool_place_forwards_protection_to_service(monkeypatch):
    import json

    from src.tools import trading_connector_tool

    calls: dict = {}

    def fake_place(symbol, profile_id, **kwargs):
        calls.update(kwargs)
        return {"status": "ok", "order_id": "t-1"}

    monkeypatch.setattr(trading_connector_tool, "place_order", fake_place)
    payload = json.loads(
        trading_connector_tool.TradingPlaceOrderTool().execute(
            symbol="BTC-USDT", connection="bybit-paper-trade", side="sell",
            quantity=0.01, take_profit_price=80000.0, stop_loss_price=76000.0,
            reduce_only=True, order_link_id="close-1",
        )
    )
    assert payload["status"] == "ok"
    assert calls["take_profit_price"] == 80000.0
    assert calls["stop_loss_price"] == 76000.0
    assert calls["reduce_only"] is True
    assert calls["order_link_id"] == "close-1"


# --- leverage ----------------------------------------------------------------


def test_set_leverage_int_forwards_to_ccxt(order_client):
    got: dict = {}

    def set_leverage(leverage, symbol, params=None):
        got.update(leverage=leverage, symbol=symbol, params=params)
        return {"retMsg": "OK"}

    order_client.set_leverage = set_leverage
    result = sdk.set_leverage(sdk.BybitConfig(), "BTC-USDT", leverage=5)
    assert result["status"] == "ok"
    assert got == {"leverage": 5, "symbol": "BTC/USDT:USDT", "params": None}
    assert result["leverage"] == 5


def test_set_leverage_rejects_spot():
    result = sdk.set_leverage(
        sdk.BybitConfig(market_type="spot"), "BTC-USDT", leverage=5,
    )
    assert result["status"] == "error"
    assert "linear" in result["error"]


@pytest.mark.parametrize("junk", [0, 0.5, -3, None, "ten", float("inf"), float("nan")])
def test_set_leverage_rejects_bad_leverage(junk):
    result = sdk.set_leverage(sdk.BybitConfig(), "BTC-USDT", leverage=junk)
    assert result["status"] == "error"
    assert "leverage" in result["error"]


def test_set_leverage_guard_fires_before_broker_call(order_client):
    order_client.urls = _live_urls()
    order_client.set_leverage = lambda *a: (_ for _ in ()).throw(
        AssertionError("must not reach broker")
    )
    result = sdk.set_leverage(sdk.BybitConfig(), "BTC-USDT", leverage=5)
    assert result["status"] == "error"
    assert "api-demo.bybit.com" in result["error"]


def test_service_set_leverage_paper_routes_to_module(monkeypatch, order_client):
    from src.trading import service

    got: dict = {}

    def set_leverage(leverage, symbol, params=None):
        got.update(leverage=leverage, symbol=symbol)
        return {"retMsg": "OK"}

    order_client.set_leverage = set_leverage
    result = service.set_leverage("BTC-USDT", "bybit-paper-trade", leverage=5)
    assert result["status"] == "ok"
    assert got == {"leverage": 5, "symbol": "BTC/USDT:USDT"}


def _bybit_demo_mandate(max_leverage: float):
    """Same shape as the live fixture, keyed under the ``bybit-demo`` runner."""
    return _bybit_live_mandate(max_leverage)


def test_service_set_leverage_paper_over_demo_cap_denied(monkeypatch, order_client):
    from src.trading import service

    order_client.set_leverage = lambda *a: (_ for _ in ()).throw(
        AssertionError("must not reach broker")
    )
    monkeypatch.setattr(
        "src.live.mandate.store.load_mandate",
        lambda broker: _bybit_demo_mandate(3) if broker == "bybit-demo" else None,
    )
    result = service.set_leverage("BTC-USDT", "bybit-paper-trade", leverage=10)
    assert result["status"] == "error"
    assert "max_leverage" in result["error"]


def test_service_set_leverage_paper_within_demo_cap_executes(monkeypatch, order_client):
    from src.trading import service

    got: dict = {}

    def set_leverage(leverage, symbol, params=None):
        got.update(leverage=leverage, symbol=symbol)
        return {"retMsg": "OK"}

    order_client.set_leverage = set_leverage
    monkeypatch.setattr(
        "src.live.mandate.store.load_mandate",
        lambda broker: _bybit_demo_mandate(3) if broker == "bybit-demo" else None,
    )
    result = service.set_leverage("BTC-USDT", "bybit-paper-trade", leverage=2)
    assert result["status"] == "ok"
    assert got == {"leverage": 2, "symbol": "BTC/USDT:USDT"}


def test_service_set_leverage_paper_without_demo_mandate_unconstrained(monkeypatch, order_client):
    from src.trading import service

    order_client.set_leverage = lambda leverage, symbol, params=None: {"retMsg": "OK"}
    monkeypatch.setattr("src.live.mandate.store.load_mandate", lambda broker: None)
    result = service.set_leverage("BTC-USDT", "bybit-paper-trade", leverage=10)
    assert result["status"] == "ok"


def test_service_set_leverage_refused_on_binance():
    from src.trading import service

    result = service.set_leverage("BTCUSDT", "binance-paper-trade", leverage=5)
    assert result["status"] == "error"
    assert "leverage" in result["error"]
    assert "binance" in result["error"]


def _bybit_live_mandate(max_leverage: float):
    from datetime import datetime, timezone

    from src.live.mandate.model import (
        MANDATE_SCHEMA_VERSION,
        ConsentMeta,
        HardCaps,
        Mandate,
        UniverseConstraint,
    )

    return Mandate(
        schema_version=MANDATE_SCHEMA_VERSION,
        hard_caps=HardCaps(
            account_funding_usd=10_000.0,
            max_order_notional_usd=10_000.0,
            max_total_exposure_usd=10_000.0,
            max_leverage=max_leverage,
            allowed_instruments=(),
            max_trades_per_day=100,
        ),
        universe=UniverseConstraint(
            asset_classes=(), min_market_cap_usd=None,
            min_avg_daily_volume_usd=None, exclude_symbols=(),
        ),
        consent=ConsentMeta(
            created_at=datetime.now(timezone.utc).isoformat(),
            consent_token_sha256="deadbeef",
            broker="bybit",
            account_ref="demo-acct",
            expires_at="2999-01-01T00:00:00+00:00",
        ),
    )


def test_service_set_leverage_live_requires_mandate(monkeypatch, order_client):
    from src.trading import service

    def refuse(*a, **k):
        raise AssertionError("must not reach broker")

    order_client.urls = _live_urls()
    order_client.set_leverage = refuse
    monkeypatch.setattr("src.live.mandate.store.load_mandate", lambda broker: None)
    result = service.set_leverage("BTC-USDT", "bybit-live-trade", leverage=5)
    assert result["status"] == "error"
    assert "mandate" in result["error"]


def test_service_set_leverage_live_halted_denies(monkeypatch, order_client):
    from src.trading import service

    order_client.urls = _live_urls()
    order_client.set_leverage = lambda *a: (_ for _ in ()).throw(
        AssertionError("must not reach broker")
    )
    monkeypatch.setattr("src.live.mandate.store.load_mandate", lambda broker: _bybit_live_mandate(10))
    monkeypatch.setattr("src.live.halt.halt_flag_set", lambda broker: True)
    result = service.set_leverage("BTC-USDT", "bybit-live-trade", leverage=5)
    assert result["status"] == "error"
    assert "halted" in result["error"]


def test_service_set_leverage_live_cap_enforced(monkeypatch, order_client):
    from src.trading import service

    order_client.urls = _live_urls()
    order_client.set_leverage = lambda *a: (_ for _ in ()).throw(
        AssertionError("must not reach broker")
    )
    monkeypatch.setattr("src.live.mandate.store.load_mandate", lambda broker: _bybit_live_mandate(10))
    monkeypatch.setattr("src.live.halt.halt_flag_set", lambda broker: False)
    result = service.set_leverage("BTC-USDT", "bybit-live-trade", leverage=20)
    assert result["status"] == "error"
    assert "max_leverage" in result["error"]


def test_service_set_leverage_live_within_cap_executes_and_audits(monkeypatch, order_client):
    from src.trading import service

    got: dict = {}

    def set_leverage(leverage, symbol, params=None):
        got.update(leverage=leverage, symbol=symbol)
        return {"retMsg": "OK"}

    order_client.urls = _live_urls()
    order_client.set_leverage = set_leverage
    written: list = []
    monkeypatch.setattr("src.live.mandate.store.load_mandate", lambda broker: _bybit_live_mandate(10))
    monkeypatch.setattr("src.live.halt.halt_flag_set", lambda broker: False)
    import src.live.audit as audit

    def fake_write(event, *a, **k):
        written.append(event)
        return {"audited": True}

    monkeypatch.setattr(audit, "write_live_action", fake_write)
    result = service.set_leverage("BTC-USDT", "bybit-live-trade", leverage=10, session_id="sess-9")
    assert result["status"] == "ok"
    assert got == {"leverage": 10, "symbol": "BTC/USDT:USDT"}
    assert len(written) == 1
    assert written[0].kind == "leverage_set"
    assert written[0].outcome == "accepted"
    assert written[0].session_id == "sess-9"


def test_tool_set_leverage_forwards_and_validates(monkeypatch):
    import json

    from src.tools import trading_connector_tool

    calls: dict = {}

    def fake_set_leverage(symbol, profile_id, **kwargs):
        calls.update(kwargs)
        return {"status": "ok", "leverage": kwargs["leverage"]}

    monkeypatch.setattr(trading_connector_tool, "set_leverage", fake_set_leverage)
    payload = json.loads(
        trading_connector_tool.TradingSetLeverageTool().execute(
            symbol="BTC-USDT", connection="bybit-paper-trade", leverage=5,
        )
    )
    assert payload["status"] == "ok"
    assert calls["leverage"] == 5.0

    bad = json.loads(
        trading_connector_tool.TradingSetLeverageTool().execute(
            symbol="BTC-USDT", connection="bybit-paper-trade", leverage="five",
        )
    )
    assert bad["status"] == "error"
