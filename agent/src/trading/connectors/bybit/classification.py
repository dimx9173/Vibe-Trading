"""Curated read/write classification for supported Bybit ccxt operations.

Keys are the ccxt unified method names this connector uses. Order-mutating
ccxt calls are pinned WRITE so the live gate never treats them as plain reads;
anything unlisted and not a known read is treated as WRITE (fail-closed) by
the gate.
"""

from __future__ import annotations

from src.live.classification import ToolClass

#: Bybit ccxt operation read/write catalog.
BYBIT_TOOL_CLASS: dict[str, ToolClass] = {
    # READ
    "fetch_balance": ToolClass.READ,
    "fetch_positions": ToolClass.READ,
    "fetch_open_orders": ToolClass.READ,
    "fetch_ticker": ToolClass.READ,
    "fetch_ohlcv": ToolClass.READ,
    "load_markets": ToolClass.READ,
}
