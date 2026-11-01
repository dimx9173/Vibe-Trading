"""Offline tests for the Bybit profile rows and their registration.

Declarations only: these assert what the registry publishes, not that any
broker call succeeded. No sandbox or live run is performed here.
"""

import pytest

from src.live import registry
from src.trading import profiles as profile_registry
from src.trading import service
from src.trading.connectors.bybit.profiles import BYBIT_PROFILES
from src.trading.types import READ_CAPABILITIES

#: The declared rows for Commit A: (id, environment, readonly, capabilities, config.profile).
EXPECTED_ROWS = {
    "bybit-paper-sdk": ("paper", True, READ_CAPABILITIES, "paper"),
    "bybit-live-sdk-readonly": ("live", True, READ_CAPABILITIES, "live-readonly"),
}


@pytest.mark.parametrize("profile_id", sorted(EXPECTED_ROWS))
def test_row_declares_its_own_environment_and_permissions(profile_id):
    environment, readonly, capabilities, config_profile = EXPECTED_ROWS[profile_id]
    profile = next(p for p in BYBIT_PROFILES if p.id == profile_id)
    assert profile.connector == "bybit"
    assert profile.transport == "broker_sdk"
    assert profile.environment == environment
    assert profile.readonly is readonly
    assert profile.capabilities == capabilities
    assert profile.config == {"profile": config_profile}


def test_readonly_rows_carry_read_only_capabilities():
    for profile in BYBIT_PROFILES:
        assert profile.readonly is True
        assert profile.capabilities == READ_CAPABILITIES
        assert not any(cap.startswith("orders.place") for cap in profile.capabilities)


def test_demo_profile_note_names_the_demo_host():
    paper = next(p for p in BYBIT_PROFILES if p.id == "bybit-paper-sdk")
    assert "api-demo.bybit.com" in paper.notes
    assert "ccxt" in paper.notes
    assert "cannot reach the live host" in paper.notes


def test_profiles_are_registered_in_the_builtin_registry():
    builtin_ids = {profile.id for profile in profile_registry.BUILTIN_PROFILES}
    assert set(EXPECTED_ROWS) <= builtin_ids


@pytest.mark.parametrize("profile_id", sorted(EXPECTED_ROWS))
def test_profile_by_id_resolves_each_bybit_row(profile_id):
    assert profile_registry.profile_by_id(profile_id).id == profile_id


def test_service_dispatch_routes_bybit_to_its_sdk_module():
    assert service._SDK_CONNECTOR_MODULES["bybit"] == "src.trading.connectors.bybit.sdk"
    module = service._sdk_module("bybit")
    assert module.__name__ == "src.trading.connectors.bybit.sdk"
    assert set(service._allowed_override_keys(module)) == {
        "api_key", "api_secret", "profile", "market_type", "timeout"
    }


def test_live_registry_curates_the_bybit_tool_map():
    curated = registry._BROKER_CURATED_MAPS["bybit"]
    for read_key in ("fetch_balance", "fetch_positions", "fetch_open_orders",
                     "fetch_ticker", "fetch_ohlcv", "load_markets"):
        assert curated[read_key].value == "read"
