"""Bybit ccxt connector with host-separated demo/live profiles.

Transport is ccxt's ``bybit`` class. ``paper`` profiles select the demo
account system on ``api-demo.bybit.com`` via ``enable_demo_trading(True)``;
live profiles select ``api.bybit.com``. ``set_sandbox_mode`` is never used —
it would select ``api-testnet.bybit.com``, a different system where demo keys
are meaningless. Every call asserts the effective REST host against the
declared environment before it reaches the broker.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from src.config.paths import get_runtime_root
from src.trading.connectors.bybit.shaping import (
    as_iter as _as_iter,
    nonzero_balances as _nonzero_balances,
    normalize_symbol,
    obj_get as _obj_get,
    ohlcv_to_dict as _ohlcv_to_dict,
    order_to_dict as _order_to_dict,
    position_to_dict as _position_to_dict,
    to_float as _to_float,
)

CONFIG_FILENAME = "bybit.json"

#: Profiles this connector understands and their default account environment.
PROFILE_ENVIRONMENTS = {
    "paper": "paper",
    "live-readonly": "live",
    "live": "live",
}

#: Allowed ``market_type`` values. Bybit perps are the primary surface; spot
#: is secondary. Inverse/options/portfolio margin are out of scope.
MARKET_TYPES = ("spot", "linear")

DEFAULT_DEMO_HOST = "https://api-demo.bybit.com"
LIVE_HOST = "https://api.bybit.com"

#: ``demo``/``live`` → the only REST subdomain the resolved client may use.
_EXPECTED_HOSTNAMES = {"demo": "api-demo.bybit.com", "live": "api.bybit.com"}


class BybitDependencyError(RuntimeError):
    """Raised when the optional ``ccxt`` package is not installed."""


class BybitConfigError(RuntimeError):
    """Raised when the connector configuration is missing or invalid."""


@dataclass(frozen=True)
class BybitConfig:
    """Bybit connector connection settings."""

    api_key: str = ""
    api_secret: str = ""
    profile: str = "paper"
    market_type: str = "linear"
    timeout: float = 15.0
    readonly: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "BybitConfig":
        """Build a config from a JSON-like mapping, normalizing the profile."""
        payload = dict(data or {})
        profile = str(payload.get("profile") or "paper").strip().lower()
        if profile not in PROFILE_ENVIRONMENTS:
            raise BybitConfigError("profile must be 'paper', 'live-readonly' or 'live'")
        market_type = str(payload.get("market_type") or "linear").strip().lower()
        if market_type not in MARKET_TYPES:
            raise BybitConfigError("market_type must be 'spot' or 'linear'")
        raw_timeout = payload.get("timeout")
        try:
            timeout = float(15.0 if raw_timeout is None else raw_timeout)
        except (TypeError, ValueError) as exc:
            raise BybitConfigError("timeout must be a positive number") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise BybitConfigError("timeout must be a positive number")
        return cls(
            api_key=str(payload.get("api_key") or "").strip(),
            api_secret=str(payload.get("api_secret") or "").strip(),
            profile=profile,
            market_type=market_type,
            timeout=timeout,
            readonly=bool(payload.get("readonly", True)),
        )

    def with_overrides(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        profile: str | None = None,
        market_type: str | None = None,
        timeout: float | None = None,
    ) -> "BybitConfig":
        """Return a copy with CLI/tool overrides applied."""
        payload = asdict(self)
        if api_key is not None:
            payload["api_key"] = api_key
        if api_secret is not None:
            payload["api_secret"] = api_secret
        if profile is not None:
            payload["profile"] = profile
        if market_type is not None:
            payload["market_type"] = market_type
        if timeout is not None:
            payload["timeout"] = timeout
        return BybitConfig.from_mapping(payload)

    @property
    def environment(self) -> str:
        """Return ``paper`` or ``live`` for this profile."""
        return PROFILE_ENVIRONMENTS.get(self.profile, "paper")

    @property
    def is_demo(self) -> bool:
        """Return whether this profile targets the demo account system."""
        return self.environment == "paper"

    @property
    def host(self) -> str:
        """Return the REST host this profile connects to."""
        return DEFAULT_DEMO_HOST if self.is_demo else LIVE_HOST


_OVERRIDE_KEYS = ("api_key", "api_secret", "profile", "market_type", "timeout")


def build_config(
    profile_config: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> "BybitConfig":
    """Resolve the effective config: saved file ← profile defaults ← CLI overrides.

    Credentials (``api_key`` / ``api_secret``) come from the saved
    ``~/.vibe-trading/bybit.json``; the selected connector profile supplies the
    ``profile`` intent; CLI/tool overrides win last.
    """
    base = asdict(load_config())
    for key, value in dict(profile_config or {}).items():
        if value is not None:
            base[key] = value
    cfg = BybitConfig.from_mapping(base)
    clean = {
        k: v for k, v in dict(overrides or {}).items() if k in _OVERRIDE_KEYS and v not in (None, "")
    }
    return cfg.with_overrides(**clean) if clean else cfg


def config_path() -> Path:
    """Return the user-level Bybit config path."""
    return get_runtime_root() / CONFIG_FILENAME


def load_config() -> BybitConfig:
    """Load Bybit settings from ``~/.vibe-trading/bybit.json``."""
    path = config_path()
    if not path.exists():
        return BybitConfig()
    try:
        return BybitConfig.from_mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BybitConfigError(f"invalid Bybit config at {path}: {exc}") from exc


def save_config(config: BybitConfig) -> Path:
    """Persist Bybit settings with owner-only permissions."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def ccxt_available() -> bool:
    """Return whether the optional ``ccxt`` library can be imported."""
    try:
        _require_ccxt()
        return True
    except BybitDependencyError:
        return False


def check_status(config: BybitConfig | None = None) -> dict[str, Any]:
    """Check SDK readiness, config completeness, and host separation.

    Returns a JSON-serializable health report. Does not place or mutate any
    broker state. The host guard asserts that the effective client resolves to
    ``api-demo.bybit.com`` for a paper profile, or ``api.bybit.com`` for a live
    profile, so a key/host mismatch fails closed before any read.
    """
    cfg = config or load_config()
    report: dict[str, Any] = {
        "status": "ok",
        "config": _public_config(cfg),
        "sdk": {"package": "ccxt", "installed": ccxt_available()},
        "paper_guard": "host_separated",
        "host": cfg.host,
    }

    missing = _missing_fields(cfg)
    if missing:
        report["status"] = "error"
        report["error"] = f"Bybit connector not configured: missing {', '.join(missing)}."
        return report

    if not report["sdk"]["installed"]:
        report["status"] = "error"
        report["error"] = "Optional dependency missing: install with `pip install ccxt`."
        return report

    try:
        _assert_host(cfg)
    except BybitConfigError as exc:
        report["status"] = "error"
        report["error"] = str(exc)
        return report

    try:
        snapshot = get_account_snapshot(cfg)
    except Exception as exc:  # noqa: BLE001 - health endpoint reports cleanly
        report["status"] = "error"
        report["error"] = str(exc)
        return report

    report["account"] = {
        "profile": cfg.profile,
        "is_demo": cfg.is_demo,
        "balances": len(snapshot.get("balances", [])),
    }
    return report


def get_account_snapshot(config: BybitConfig | None = None) -> dict[str, Any]:
    """Fetch account balances on the selected host via ccxt ``fetch_balance``.

    Returns the non-zero balances (each with ``free`` / ``used`` / ``total``).
    """
    cfg = config or load_config()
    ex = _client(cfg)
    balance = ex.fetch_balance()
    return {
        "status": "ok",
        "profile": cfg.profile,
        "is_demo": cfg.is_demo,
        "host": cfg.host,
        "paper_guard": "host_separated",
        "balances": _nonzero_balances(balance),
    }


def get_positions(config: BybitConfig | None = None) -> dict[str, Any]:
    """Fetch linear positions, or spot holdings as the position surface.

    Spot markets hold balances, not positions, so spot reads reshape the
    non-zero balance rows; linear (USDT/USDC perpetual) reads return the
    ccxt unified ``fetch_positions`` rows.
    """
    cfg = config or load_config()
    ex = _client(cfg)
    if cfg.market_type == "spot":
        balance = ex.fetch_balance()
        rows = [
            {
                "symbol": _obj_get(row, "asset"),
                "quantity": _obj_get(row, "total"),
                "free": _obj_get(row, "free"),
                "used": _obj_get(row, "used"),
                "source": "spot",
            }
            for row in _nonzero_balances(balance)
        ]
    else:
        rows = [_position_to_dict(item) for item in _as_iter(ex.fetch_positions())]
    return {
        "status": "ok",
        "profile": cfg.profile,
        "is_demo": cfg.is_demo,
        "host": cfg.host,
        "paper_guard": "host_separated",
        "positions": rows,
    }


def get_open_orders(
    config: BybitConfig | None = None,
    *,
    include_executions: bool = False,
) -> dict[str, Any]:
    """Fetch all open orders (no symbol filter).

    ``fetch_open_orders`` is called without a symbol to retrieve all open
    orders; some Bybit setups require a symbol, so the call is wrapped and a
    note is returned on failure rather than failing the whole call. Auth /
    network / rate-limit failures are deliberately excluded from that note and
    propagate so the caller sees a real failure, not a misleading ``ok`` with
    an empty list. This connector does not expose ``fetch_my_trades``, so
    ``include_executions=True`` degrades to an empty list with a note.
    """
    cfg = config or load_config()
    ex = _client(cfg)
    symbol_required = _symbol_required_errors()
    result: dict[str, Any] = {
        "status": "ok",
        "profile": cfg.profile,
        "is_demo": cfg.is_demo,
        "host": cfg.host,
        "paper_guard": "host_separated",
    }
    try:
        open_orders = ex.fetch_open_orders()
        result["open_orders"] = [_order_to_dict(item) for item in _as_iter(open_orders)]
    except symbol_required as exc:
        result["open_orders"] = []
        result["open_orders_note"] = f"fetch_open_orders without a symbol failed: {exc}"
    if include_executions:
        result["executions"] = []
        result["executions_note"] = "Bybit executions (fetch_my_trades) is not exposed by this connector."
    return result


def get_quote(symbol: str, *, config: BybitConfig | None = None, **_: Any) -> dict[str, Any]:
    """Fetch a latest ticker snapshot for ``symbol`` (ccxt unified format)."""
    cfg = config or load_config()
    clean = normalize_symbol(symbol, market_type=cfg.market_type)
    ex = _client(cfg, public=True)
    ticker = ex.fetch_ticker(clean)
    return {
        "status": "ok",
        "symbol": clean,
        "quote": {
            "bid": _obj_get(ticker, "bid"),
            "ask": _obj_get(ticker, "ask"),
            "last": _obj_get(ticker, "last"),
            "high": _obj_get(ticker, "high"),
            "low": _obj_get(ticker, "low"),
            "volume": _obj_get(ticker, "baseVolume"),
            "time": str(_obj_get(ticker, "timestamp", "")),
        },
    }


def search_instruments(
    query: str,
    *,
    config: BybitConfig | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Resolve an exact pair against the selected Bybit market catalog.

    The connector resolves only explicit pair spellings (``BTC-USDT``,
    ``BTCUSDT``, ``BTC/USDT`` or the ccxt linear ``BTC/USDT:USDT`` form) and
    never guesses from prose such as ``Bitcoin``. Only markets matching the
    configured ``market_type`` and an active status are returned.
    """
    cfg = config or load_config()
    requested = normalize_symbol(query, market_type=cfg.market_type)

    try:
        bounded_limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError, OverflowError):
        bounded_limit = 10

    markets = _client(cfg, public=True).load_markets()
    if not isinstance(markets, Mapping):
        raise BybitConfigError("Bybit load_markets returned a non-mapping payload")

    instruments: list[dict[str, Any]] = []
    for key, raw_market in markets.items():
        if not isinstance(raw_market, Mapping):
            continue
        native_symbol = str(raw_market.get("symbol") or key)
        if native_symbol != requested:
            continue
        if cfg.market_type == "spot" and raw_market.get("spot") is not True:
            continue
        if cfg.market_type == "linear" and raw_market.get("linear") is not True:
            continue
        if raw_market.get("active") is False:
            continue
        base, quote = native_symbol.split("/", 1)
        instruments.append(
            {
                "symbol": f"{base}-{quote}",
                "native_symbol": native_symbol,
                "exchange_symbol": str(raw_market.get("id") or "").strip() or None,
                "base": base,
                "quote": quote.split(":", 1)[0],
                "market": "crypto",
                "type": "cryptocurrency",
                "exchange": "BYBIT",
                "active": raw_market.get("active"),
            }
        )
        if len(instruments) >= bounded_limit:
            break

    return {"status": "ok", "query": query, "instruments": instruments}


#: Project/canonical period token → ccxt unified timeframe. ccxt accepts
#: unified timeframes on Bybit (its adapter translates to the V5 interval
#: minutes internally); the month token is ccxt's ``1M``.
_TIMEFRAME_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "1H": "1h", "4h": "4h", "4H": "4h",
    "1d": "1d", "1D": "1d", "1w": "1w", "1W": "1w", "1M": "1M",
}


def get_historical_bars(
    symbol: str,
    *,
    config: BybitConfig | None = None,
    period: str = "1d",
    limit: int = 90,
    **_: Any,
) -> dict[str, Any]:
    """Fetch historical OHLCV bars for ``symbol`` (ccxt unified format)."""
    cfg = config or load_config()
    clean = normalize_symbol(symbol, market_type=cfg.market_type)
    timeframe = _TIMEFRAME_MAP.get(period.strip(), "1d")
    ex = _client(cfg, public=True)
    bars = ex.fetch_ohlcv(clean, timeframe=timeframe, limit=int(limit))
    return {
        "status": "ok",
        "symbol": clean,
        "period": period,
        "bars": [_ohlcv_to_dict(item) for item in _as_iter(bars)],
    }


# ---------------------------------------------------------------------------
# Order placement (write path; guarded by host separation + profile readonly)
# ---------------------------------------------------------------------------

#: Project time-in-force intent → ccxt ``timeInForce``. Bybit has no DAY
#: policy; the unified ``"day"`` intent maps to GTC, the closest equivalent.
_TIME_IN_FORCE_MAP = {
    "day": "GTC",
    "gtc": "GTC",
    "ioc": "IOC",
    "fok": "FOK",
}


def _order_link_id_error(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "orderLinkId must be a non-empty string."
    if len(value) > 36:
        return "orderLinkId must be at most 36 characters."
    return None


def _trigger_price_error(value: Any) -> str | None:
    """Validate a Bybit ``takeProfit``/``stopLoss`` attachment shape."""
    if not isinstance(value, Mapping):
        return "takeProfit/stopLoss must be a mapping such as {'triggerPrice': 30000}."
    if not isinstance(value.get("triggerPrice"), (str, int, float)):
        return "takeProfit/stopLoss requires a string or numeric 'triggerPrice'."
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, (str, int, float)):
            return "takeProfit/stopLoss values must be strings or numbers."
    return None


def _bool_error(value: Any) -> str | None:
    return None if isinstance(value, bool) else "reduceOnly must be a bool."


def _position_idx_error(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1, 2):
        return "positionIdx must be 0, 1 or 2."
    return None


#: Bybit order extras accepted through the trailing ``**broker_params``:
#: name → shape validator. Every accepted key is forwarded to ccxt; anything
#: else fails closed instead of being forwarded blind (an unknown key could
#: silently re-size or re-price the order).
_BROKER_PARAM_RULES: dict[str, Callable[[Any], str | None]] = {
    "orderLinkId": _order_link_id_error,
    "reduceOnly": _bool_error,
    "positionIdx": _position_idx_error,
    "takeProfit": _trigger_price_error,
    "stopLoss": _trigger_price_error,
}


def _broker_params_error(params: Mapping[str, Any]) -> str | None:
    for name, value in params.items():
        rule = _BROKER_PARAM_RULES.get(name)
        if rule is None:
            return f"unsupported Bybit order parameter '{name}'."
        problem = rule(value)
        if problem:
            return problem
    return None


def place_order(
    config: BybitConfig | None = None,
    *,
    symbol: str,
    side: str,
    quantity: float | None = None,
    notional: float | None = None,
    order_type: str = "market",
    limit_price: float | None = None,
    time_in_force: str = "day",
    **broker_params: Any,
) -> dict[str, Any]:
    """Place an order on Bybit via ccxt's unified ``create_order``.

    The configured profile's host (demo vs live) is the authoritative
    paper/live discriminator and is asserted before anything is submitted, so a
    demo key can never reach the live host. The connector executes the intent
    the caller has already authorized; mandate/limit/kill-switch enforcement
    lives in a higher layer (service + ``sdk_order_gate``).

    Bybit acknowledges asynchronously: ``status: "ok"`` means the order was
    ACCEPTED, not filled. ``order_status`` carries Bybit's acknowledgement
    state; nothing here waits for or claims a fill.

    Either ``quantity`` (base-asset amount) or ``notional`` (quote-asset spend)
    must be given, never both. ``notional`` is forwarded as ccxt's ``cost``
    parameter, which ccxt's Bybit adapter turns into a quote-coin sized market
    order (``marketUnit="quoteCoin"``). Limit orders require ``quantity`` and
    ``limit_price``.

    ``broker_params`` accepts Bybit's documented order extras — ``orderLinkId``
    (≤36 characters), ``reduceOnly``, ``positionIdx``, ``takeProfit`` and
    ``stopLoss`` — validated for shape and forwarded to ccxt. They are
    validated BEFORE any broker call; an unknown key fails closed rather than
    being silently dropped.

    Args:
        config: Connector config; falls back to the saved config when ``None``.
        symbol: Pair in any accepted form (normalized for ``market_type``).
        side: ``"buy"`` or ``"sell"``.
        quantity: Base-asset amount. Mutually exclusive with ``notional``.
        notional: Quote-asset spend. Mutually exclusive with ``quantity``.
        order_type: ``"market"`` or ``"limit"``.
        limit_price: Required when ``order_type`` is ``"limit"``.
        time_in_force: Limit policy; ``"day"`` maps to Bybit GTC.
        **broker_params: Validated Bybit order extras.

    Returns:
        On acceptance: ``{"status": "ok", "order_id", "symbol", "side",
        "profile", "is_demo", "host", "paper_guard", "order_type",
        "order_status", "filled", "amount", "price"}``. On any validation or
        execution failure: ``{"status": "error", "error": str}`` (fail-closed;
        nothing is submitted on a validation error).
    """
    cfg = config or load_config()

    side_clean = str(side or "").strip().lower()
    if side_clean not in ("buy", "sell"):
        return {"status": "error", "error": "side must be 'buy' or 'sell'."}

    type_clean = str(order_type or "").strip().lower()
    if type_clean not in ("market", "limit"):
        return {"status": "error", "error": "order_type must be 'market' or 'limit'."}

    qty_given = quantity is not None
    notional_given = notional is not None
    if qty_given == notional_given:
        return {"status": "error", "error": "provide exactly one of 'quantity' or 'notional'."}

    qty_value = _to_float(quantity) if qty_given else None
    notional_value = _to_float(notional) if notional_given else None
    if qty_given and (qty_value is None or qty_value <= 0):
        return {"status": "error", "error": "quantity must be a positive number."}
    if notional_given and (notional_value is None or notional_value <= 0):
        return {"status": "error", "error": "notional must be a positive number."}

    if type_clean == "limit":
        if notional_given:
            return {"status": "error", "error": "limit orders require 'quantity', not 'notional'."}
        price_value = _to_float(limit_price)
        if price_value is None or price_value <= 0:
            return {"status": "error", "error": "limit orders require a positive 'limit_price'."}
    else:
        price_value = None

    params_error = _broker_params_error(broker_params)
    if params_error:
        return {"status": "error", "error": params_error}

    try:
        clean_symbol = normalize_symbol(symbol, market_type=cfg.market_type)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}

    params: dict[str, Any] = dict(broker_params)
    if type_clean == "limit":
        tif = _TIME_IN_FORCE_MAP.get(str(time_in_force or "").strip().lower())
        if tif is None:
            return {"status": "error", "error": "time_in_force must be one of 'day', 'gtc', 'ioc', 'fok'."}
        params["timeInForce"] = tif
        amount = qty_value
        price: float | None = price_value
    elif notional_given:
        params["cost"] = notional_value
        amount = notional_value
        price = None
    else:
        amount = qty_value
        price = None

    try:
        ex = _client(cfg)
    except BybitConfigError as exc:
        return {"status": "error", "error": str(exc)}

    try:
        order = ex.create_order(clean_symbol, type_clean, side_clean, amount, price, params)
    except Exception as exc:  # noqa: BLE001 - surface any ccxt/auth/network error as fail-closed
        return {"status": "error", "error": str(exc)}

    return {
        "status": "ok",
        "order_id": str(_obj_get(order, "id", "")),
        "symbol": _obj_get(order, "symbol", clean_symbol),
        "side": str(_obj_get(order, "side", side_clean)),
        "profile": cfg.profile,
        "is_demo": cfg.is_demo,
        "host": cfg.host,
        "paper_guard": "host_separated",
        "order_type": str(_obj_get(order, "type", type_clean)),
        "order_status": str(_obj_get(order, "status", "")),
        "filled": _obj_get(order, "filled"),
        "amount": _obj_get(order, "amount"),
        "price": _obj_get(order, "price"),
    }


def cancel_order(
    config: BybitConfig | None = None,
    order_id: str = "",
    *,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Cancel an open order by id. Bybit REQUIRES the order's symbol.

    Args:
        config: Connector config; falls back to the saved config when ``None``.
        order_id: The exchange order id to cancel.
        symbol: The order's pair. Bybit cannot cancel without it, so a
            missing/blank symbol fails closed before any broker call.

    Returns:
        On success: ``{"status": "ok", "order_id", "symbol", "side", "profile",
        "is_demo", "host", "paper_guard", "order_status"}``. On any validation
        or execution failure: ``{"status": "error", "error": str}``.
    """
    cfg = config or load_config()

    order_id_clean = str(order_id or "").strip()
    if not order_id_clean:
        return {"status": "error", "error": "order_id is required to cancel an order."}
    if symbol is None or not str(symbol).strip():
        return {"status": "error", "error": "Bybit cancel requires symbol."}

    try:
        clean_symbol = normalize_symbol(symbol, market_type=cfg.market_type)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}

    try:
        ex = _client(cfg)
    except BybitConfigError as exc:
        return {"status": "error", "error": str(exc)}

    try:
        order = ex.cancel_order(order_id_clean, clean_symbol)
    except Exception as exc:  # noqa: BLE001 - surface any ccxt/auth/network error as fail-closed
        return {"status": "error", "error": str(exc)}

    return {
        "status": "ok",
        "order_id": str(_obj_get(order, "id", order_id_clean)),
        "symbol": _obj_get(order, "symbol", clean_symbol),
        "side": str(_obj_get(order, "side", "")),
        "profile": cfg.profile,
        "is_demo": cfg.is_demo,
        "host": cfg.host,
        "paper_guard": "host_separated",
        "order_status": str(_obj_get(order, "status", "")),
    }


def set_leverage(
    config: BybitConfig | None = None,
    symbol: str = "",
    *,
    leverage: float = 1.0,
) -> dict[str, Any]:
    """Set per-symbol leverage on Bybit via ccxt's ``set_leverage``.

    Only the ``linear`` (perp) surface supports this; spot symbols refuse
    before any broker call. ``leverage`` must be a finite number ≥ 1. The
    broker enforces its own ceiling (Bybit's max per pair); a broker rejection
    surfaces unmasked so the caller sees the real limit.

    Args:
        config: Connector config; falls back to the saved config when ``None``.
        symbol: Pair in any accepted form (normalized for ``market_type``).
        leverage: Desired leverage multiple, ≥ 1.

    Returns:
        On success: ``{"status": "ok", "symbol", "leverage", "profile",
        "is_demo", "host", "paper_guard"}``. On any validation or execution
        failure: ``{"status": "error", "error": str}``.
    """
    cfg = config or load_config()

    if cfg.market_type != "linear":
        return {"status": "error", "error": "Bybit leverage requires a linear (perp) market_type."}

    try:
        lev = float(leverage)
    except (TypeError, ValueError, OverflowError):
        return {"status": "error", "error": "leverage must be a finite number >= 1."}
    if not math.isfinite(lev) or lev < 1:
        return {"status": "error", "error": "leverage must be a finite number >= 1."}

    try:
        clean_symbol = normalize_symbol(symbol, market_type=cfg.market_type)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}

    try:
        ex = _client(cfg)
    except BybitConfigError as exc:
        return {"status": "error", "error": str(exc)}

    try:
        ex.set_leverage(int(lev), clean_symbol)
    except Exception as exc:  # noqa: BLE001 - surface any ccxt/auth/network error as fail-closed
        return {"status": "error", "error": str(exc)}

    return {
        "status": "ok",
        "symbol": clean_symbol,
        "leverage": int(lev),
        "profile": cfg.profile,
        "is_demo": cfg.is_demo,
        "host": cfg.host,
        "paper_guard": "host_separated",
    }


# ---------------------------------------------------------------------------
# SDK plumbing
# ---------------------------------------------------------------------------


def _require_ccxt() -> ModuleType:
    try:
        import ccxt  # type: ignore
    except ModuleNotFoundError as exc:
        raise BybitDependencyError("ccxt is not installed; run `pip install ccxt`.") from exc
    return ccxt


def _symbol_required_errors() -> tuple[type[BaseException], ...]:
    """ccxt exception classes that mean "this call needs a symbol" (degrade-only).

    Auth/network/rate-limit errors are deliberately excluded so they propagate
    rather than being masked as a status:ok note.
    """
    ccxt = _require_ccxt()
    names = ("ArgumentsRequired", "BadSymbol", "NotSupported")
    classes = tuple(getattr(ccxt, n) for n in names if hasattr(ccxt, n))
    return classes or (ValueError,)


def _exchange(cfg: BybitConfig):
    """Build a ccxt Bybit client bound to the configured market/environment.

    Demo mode is selected through ``enable_demo_trading(True)`` — the demo
    key namespace on ``api-demo.bybit.com`` — never ``set_sandbox_mode``.
    """
    ccxt = _require_ccxt()
    client_config: dict[str, Any] = {
        "apiKey": cfg.api_key,
        "secret": cfg.api_secret,
        "enableRateLimit": True,
        "timeout": int(cfg.timeout * 1000),
        "options": {"defaultType": cfg.market_type},
    }
    ex = ccxt.bybit(client_config)
    if cfg.is_demo:
        ex.enable_demo_trading(True)
    return ex


def _public_exchange(cfg: BybitConfig):
    """Use the selected Bybit host without credentials for public market data."""
    return _exchange(replace(cfg, api_key="", api_secret=""))


def _client(cfg: BybitConfig, *, public: bool = False):
    """Build the client with the host guard asserted before any broker call."""
    _assert_host(cfg)
    if public:
        ex = _public_exchange(cfg)
    else:
        ex = _exchange(cfg)
    _assert_host(cfg, ex)
    return ex


def _resolved_hostname(exchange: Any) -> str:
    """Compose the effective REST host from a ccxt bybit URL template.

    ccxt's Bybit adapter stores URL templates such as ``https://api.{hostname}``
    plus a bare ``hostname`` (``bybit.com``). Composing them yields the host the
    next request would actually hit: ``api.bybit.com`` (live),
    ``api-demo.bybit.com`` (after ``enable_demo_trading``), or
    ``api-testnet.bybit.com`` (sandbox — which the guard must reject).
    """
    urls = getattr(exchange, "urls", None)
    api = urls.get("api") if isinstance(urls, Mapping) else None
    template = None
    if isinstance(api, Mapping):
        template = next((v for v in api.values() if isinstance(v, str)), None)
    elif isinstance(api, str):
        template = api
    if not template:
        return ""
    host = template.replace("{hostname}", str(getattr(exchange, "hostname", "") or ""))
    parsed = urlparse(host)
    return (parsed.hostname or host).lower()


def _assert_host(cfg: BybitConfig, exchange: Any = None) -> None:
    """Fail closed when the resolved host does not match the declared environment.

    The host is the authoritative discriminator: demo keys cannot reach the
    live host. A paper profile must resolve to ``api-demo.bybit.com``; a live
    profile must resolve to ``api.bybit.com``. Anything else — including the
    ``api-testnet.bybit.com`` system ``set_sandbox_mode`` would select — fails.
    """
    environment = "demo" if cfg.is_demo else "live"
    expected = _EXPECTED_HOSTNAMES[environment]
    if exchange is not None:
        host = _resolved_hostname(exchange)
    else:
        host = (urlparse(cfg.host).hostname or "").lower()
    if host != expected:
        raise BybitConfigError(
            f"Configured profile is '{cfg.profile}' ({environment}), but the resolved host "
            f"'{host or 'unknown'}' is not '{expected}'."
        )


def _missing_fields(cfg: BybitConfig) -> list[str]:
    missing = []
    if not cfg.api_key:
        missing.append("api_key")
    if not cfg.api_secret:
        missing.append("api_secret")
    return missing


def _public_config(cfg: BybitConfig) -> dict[str, Any]:
    """Config snapshot with secrets redacted."""
    data = asdict(cfg)
    if data.get("api_secret"):
        data["api_secret"] = "***redacted***"
    if data.get("api_key"):
        data["api_key"] = data["api_key"][:4] + "***"
    data["host"] = cfg.host
    return data
