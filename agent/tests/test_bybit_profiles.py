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

#: The declared rows: (id, environment, readonly, capabilities, config.profile).
EXPECTED_ROWS = {
    "bybit-paper-sdk": ("paper", True, READ_CAPABILITIES, "paper"),
    "bybit-live-sdk-readonly": ("live", True, READ_CAPABILITIES, "live-readonly"),
    "bybit-paper-trade": ("paper", False, READ_CAPABILITIES + ("orders.place", "leverage.set", "runner.manage"), "paper"),
    "bybit-live-trade": (
        "live",
        False,
        READ_CAPABILITIES
        + ("orders.place.requires_mandate", "leverage.set.requires_mandate", "runner.manage.requires_mandate"),
        "live",
    ),
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


def test_paper_order_capability_never_leaks_into_a_live_row():
    for profile in BYBIT_PROFILES:
        if profile.environment == "live" and "orders.place" in profile.capabilities:
            assert profile.id == "bybit-live-trade"
            assert "orders.place.requires_mandate" in profile.capabilities
    live_trade = next(p for p in BYBIT_PROFILES if p.id == "bybit-live-trade")
    assert live_trade.config["profile"] == "live"


def test_demo_profiles_read_the_demo_host_and_say_so():
    paper = next(p for p in BYBIT_PROFILES if p.id == "bybit-paper-sdk")
    trade = next(p for p in BYBIT_PROFILES if p.id == "bybit-paper-trade")
    for profile in (paper, trade):
        assert "api-demo.bybit.com" in profile.notes
        assert "ccxt" in profile.notes
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
    assert curated["create_order"].value == "write"
    assert curated["cancel_order"].value == "write"
    assert curated["fetch_balance"].value == "read"
