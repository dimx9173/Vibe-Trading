"""Runner READ/WRITE seam for a direct-SDK (``broker_sdk``) connector.

The managed :class:`~src.live.runtime.runner.LiveRunner` was written against the
Robinhood remote-MCP adapter. A ``broker_sdk`` broker (Bybit) has no adapter, so
this module binds the same four callables the runner needs — positions, balance,
open orders, submit — to the **service layer** instead of a raw SDK:

* reads go through :func:`src.trading.service.get_positions` /
  ``get_account`` / ``get_open_orders``, so paper-vs-live routing, the host
  guard and profile selection stay single-sourced there;
* the WRITE callable (:attr:`SdkRunnerBinding.submit_fn`) exists solely for the
  halt sweep in :mod:`src.live.runtime.flatten`.

**Structural no-resend guarantee.** The reconciler receives READs only — it has
no write callable at all, so a reconciliation mismatch can never reach the
broker. Agent orders never pass through :attr:`SdkRunnerBinding.submit_fn`
either: the agent's own ``trading_place_order`` goes through
``service.place_order``, whose live path wraps ``execute_live_order`` and its
full mandate/kill-switch gate.

**Halt-flatten scope.** A halt flatten (mandate ``flatten_on_halt=True``) cannot
succeed for a live ``broker_sdk`` broker: the closing order goes through
``execute_live_order``, whose halt check denies every gated submit while the
halt sentinel is set. Only cancel-during-halt works. Live mandates keep the
default ``flatten_on_halt=False``; there is deliberately no gate bypass here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from src.trading.types import TradingProfile

__all__ = ["SdkRunnerBinding", "bind"]


@dataclass(frozen=True)
class SdkRunnerBinding:
    """The four callables a runner needs from one ``broker_sdk`` profile."""

    read_positions: Callable[[], Sequence[Mapping[str, Any]]]
    read_balance: Callable[[], Mapping[str, Any]]
    read_open_orders: Callable[[], Sequence[Mapping[str, Any]]]
    submit_fn: Callable[[Mapping[str, Any]], dict[str, Any]]


def _unwrap(payload: Mapping[str, Any], key: str) -> list[Any]:
    """Return ``payload[key]`` for an ok envelope, raising on an error envelope.

    Args:
        payload: A service read envelope.
        key: The per-operation records key (``positions`` / ``open_orders``).

    Returns:
        The records list (empty when the key is absent).

    Raises:
        RuntimeError: When the service reported ``status: "error"``.
    """
    if str(payload.get("status", "")).lower() == "error":
        raise RuntimeError(str(payload.get("error") or "service read failed"))
    return list(payload.get(key) or [])


def bind(profile: TradingProfile) -> SdkRunnerBinding:
    """Bind runner READ/WRITE callables to one ``broker_sdk`` profile.

    Args:
        profile: A runner-capable ``broker_sdk`` profile (e.g.
            ``bybit-paper-trade``).

    Returns:
        The binding the runner and its halt sweep consume.
    """
    from src.trading import service

    profile_id = profile.id

    def read_positions() -> list[Any]:
        return _unwrap(service.get_positions(profile_id), "positions")

    def read_balance() -> Mapping[str, Any]:
        payload = service.get_account(profile_id)
        if str(payload.get("status", "")).lower() == "error":
            raise RuntimeError(str(payload.get("error") or "account read failed"))
        return payload

    def read_open_orders() -> list[Any]:
        return _unwrap(service.get_open_orders(profile_id), "open_orders")

    def submit_fn(order: Mapping[str, Any]) -> dict[str, Any]:
        action = str(order.get("action") or "").strip().lower()
        if action == "cancel":
            return service.cancel_order(
                str(order.get("order_id") or ""),
                profile_id,
                symbol=order.get("symbol"),
            )
        if action == "close":
            return service.place_order(
                str(order.get("symbol") or ""),
                profile_id,
                side=str(order.get("side") or ""),
                quantity=abs(float(order.get("qty"))),
                order_type="market",
                reduce_only=True,
            )
        return service.place_order(
            str(order.get("symbol") or ""),
            profile_id,
            side=str(order.get("side") or ""),
            quantity=abs(float(order.get("qty"))),
            order_type="market",
        )

    return SdkRunnerBinding(
        read_positions=read_positions,
        read_balance=read_balance,
        read_open_orders=read_open_orders,
        submit_fn=submit_fn,
    )
