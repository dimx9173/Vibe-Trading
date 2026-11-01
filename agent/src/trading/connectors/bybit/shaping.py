"""Pure symbol and response shaping helpers for the Bybit connector."""

from __future__ import annotations

from typing import Any, Mapping

#: Quote assets that terminate a concatenated Bybit pair spelling.
_QUOTE_ASSETS = ("USDT", "USDC", "DAI")

#: ``market_type`` → ccxt unified symbol template. Bybit linear (USDT/USDC
#: perpetual) unified symbols carry the settle currency as the ccxt
#: ``:SETTLE`` suffix; spot pairs do not.
_UNIFIED_SYMBOL_TEMPLATE = {
    "spot": "{base}/{quote}",
    "linear": "{base}/{quote}:{quote}",
}


def normalize_symbol(symbol: str, *, market_type: str = "linear") -> str:
    """Normalize a pair spelling to the ccxt unified symbol for ``market_type``.

    Accepts ``BTC-USDT`` (project form), ``BTCUSDT`` (exchange id), ``BTC/USDT``
    (ccxt spot) and ``BTC/USDT:USDT`` (ccxt linear). The shapes are derived from
    ``_UNIFIED_SYMBOL_TEMPLATE``; anything whose quote asset cannot be
    identified is rejected — Bybit cannot guess a pair.

    Raises:
        ValueError: On an unsupported ``market_type`` or an unparseable symbol
            (the message names the offending input).
    """
    if market_type not in _UNIFIED_SYMBOL_TEMPLATE:
        raise ValueError(f"unsupported Bybit market_type '{market_type}'")
    raw = str(symbol or "").strip().upper().replace("-", "/")
    core = raw.split(":", 1)[0]
    if "/" in core:
        base, _, quote = core.partition("/")
        if base and quote:
            return _UNIFIED_SYMBOL_TEMPLATE[market_type].format(base=base, quote=quote)
    clean = core.replace("/", "")
    for quote in _QUOTE_ASSETS:
        if clean.endswith(quote) and len(clean) > len(quote):
            return _UNIFIED_SYMBOL_TEMPLATE[market_type].format(
                base=clean[: -len(quote)], quote=quote
            )
    raise ValueError(f"could not resolve a trading pair from symbol '{symbol}'")


def as_iter(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def obj_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nonzero_balances(balance: Any) -> list[dict[str, Any]]:
    """Extract non-zero per-asset balances from a ccxt balance response."""
    rows: list[dict[str, Any]] = []
    if not isinstance(balance, Mapping):
        return rows
    skip = {"info", "timestamp", "datetime", "free", "used", "total"}
    for asset, detail in balance.items():
        if asset in skip or not isinstance(detail, Mapping):
            continue
        total = to_float(detail.get("total"))
        if not total:
            continue
        rows.append(
            {
                "asset": asset,
                "free": to_float(detail.get("free")),
                "used": to_float(detail.get("used")),
                "total": total,
            }
        )
    return rows


def position_to_dict(item: Any) -> dict[str, Any]:
    """Shape a ccxt unified position row (linear markets).

    Short rows carry a signed quantity: ccxt ``contracts`` is unsigned with
    ``side`` separate, so a ``"short"`` row is negated. Downstream
    (``enforcement._position_signed_market_value``, reconcile diffs) keys on
    ``quantity`` and would otherwise treat a short as a long.
    """
    side = str(obj_get(item, "side", ""))
    qty = obj_get(item, "contracts", obj_get(item, "amount"))
    if side == "short" and isinstance(qty, (int, float)) and qty:
        qty = -qty
    return {
        "symbol": obj_get(item, "symbol"),
        "side": side,
        "quantity": qty,
        "entry_price": obj_get(item, "entryPrice"),
        "mark_price": obj_get(item, "markPrice"),
        "notional": obj_get(item, "notional"),
        "unrealized_pnl": obj_get(item, "unrealizedPnl"),
        "status": str(obj_get(item, "status", "")),
    }


def order_to_dict(item: Any) -> dict[str, Any]:
    return {
        "order_id": str(obj_get(item, "id", "")),
        "symbol": obj_get(item, "symbol"),
        "side": str(obj_get(item, "side", "")),
        "order_type": str(obj_get(item, "type", "")),
        "price": obj_get(item, "price"),
        "quantity": obj_get(item, "amount"),
        "filled": obj_get(item, "filled"),
        "remaining": obj_get(item, "remaining"),
        "status": str(obj_get(item, "status", "")),
        "time": str(obj_get(item, "timestamp", "")),
    }


def ohlcv_to_dict(item: Any) -> dict[str, Any]:
    """Shape a ccxt OHLCV row into a named dictionary."""
    row = list(item) if isinstance(item, (list, tuple)) else []
    row += [None] * (6 - len(row))
    return {
        "time": str(row[0] if row[0] is not None else ""),
        "open": row[1],
        "high": row[2],
        "low": row[3],
        "close": row[4],
        "volume": row[5],
    }
