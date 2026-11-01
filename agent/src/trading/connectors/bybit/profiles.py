"""Built-in Bybit connector profiles.

Read-only demo and live profiles. Paper (demo) and live use different key
namespaces and different REST hosts (``api-demo.bybit.com`` vs
``api.bybit.com``), so they are distinct profiles. Order-placing paper and
live profiles are added by the write path; the live one carries
``orders.place.requires_mandate`` — the higher trading layer must gate it
behind the user's mandate before any order reaches the connector.
"""

from __future__ import annotations

from src.trading.types import READ_CAPABILITIES, TradingProfile

BYBIT_PROFILES: tuple[TradingProfile, ...] = (
    TradingProfile(
        id="bybit-paper-sdk",
        connector="bybit",
        label="Bybit Demo · ccxt Read-Only",
        environment="paper",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES,
        readonly=True,
        config={"profile": "paper"},
        notes=(
            "Reads a Bybit demo account (api-demo.bybit.com) via ccxt. Bybit demo "
            "accounts use a separate key namespace with their own user id, so demo "
            "keys cannot reach the live host."
        ),
    ),
    TradingProfile(
        id="bybit-live-sdk-readonly",
        connector="bybit",
        label="Bybit Live · ccxt Read-Only",
        environment="live",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES,
        readonly=True,
        config={"profile": "live-readonly"},
        notes=(
            "Reads a Bybit live account (api.bybit.com) via ccxt. The host guard "
            "fails closed unless the resolved REST host is api.bybit.com; the demo "
            "and testnet systems are never selected."
        ),
    ),
)
