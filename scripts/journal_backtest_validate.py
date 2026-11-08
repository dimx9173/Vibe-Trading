"""Validate the live decision journal replay statistically.

Every decision becomes one fixed-notional trade entered at the first 1H bar
after the decision tick and closed at the first bar after a fixed horizon. The
script reports the net expectancy, a block bootstrap interval, a sign-flip
permutation p-value, a variant grid over horizon, fee tier and entry price, and
a split by the preceding BTC 24h return.

Read-only: it never calls the broker and never writes runtime state.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import random
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "agent"))


NOTIONAL = 250.0
TAKER = 0.00055
MAKER = 0.0002
BLOCK = 3
BOOT_REPS = 2000
PERM_REPS = 5000
SEED = 20261108



def _journal() -> Any:
    spec = importlib.util.spec_from_file_location(
        "journalmod", REPO / "scripts" / "live_decision_journal.py"
    )
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules["journalmod"] = module
    spec.loader.exec_module(module)
    return module


def _decisions(j: Any, broker: str, limit: int) -> list[Any]:
    rows: list[Any] = j._load_decisions(broker, limit)
    run_map = j._run_dirs_for(rows)
    for row in rows:
        row.run_dir = run_map.get(row.decision_id, "")
        if row.run_dir:
            row.session_id = j._session_for_run(row.run_dir)
            row.calls, row.answer = j._read_session(row.session_id)
        if not row.symbols:
            row.symbols = j._symbols_in(row.answer) or j._symbols_in(" ".join(row.calls))
        row.orders = j._read_orders(broker, (row.started, row.started + dt.timedelta(hours=6)))
    return rows


def _bars_for(j: Any, cache: dict[tuple[str, dt.date], list[Any]], symbol: str, at: dt.datetime) -> list[Any]:
    key = (symbol, at.date())
    if key not in cache:
        cache[key] = j._bars(symbol, at)
    return cache[key]


def _entry_bar(bars: list[Any], at: dt.datetime) -> tuple | None:
    for bar in bars:
        if bar[0] > at:
            return bar
    return None


def _exit_bar(bars: list[Any], at: dt.datetime, hours: float) -> tuple | None:
    target = at + dt.timedelta(hours=hours)
    for bar in bars:
        if bar[0] > target:
            return bar
    return None


def _df(bar: tuple, prefer_open: bool = True) -> float | None:
    price = bar[1] if prefer_open else bar[2]
    if price is None:
        price = bar[2]
    return price


def _is_short(j: Any, row: Any) -> bool:
    for order in row.orders:
        request = order.get("broker_request")
        if isinstance(request, dict) and str(request.get("side", "")).lower() in ("sell", "short"):
            return True
    return False


def _prior_btc_move(j: Any, cache: dict[tuple[str, dt.date], list[Any]], at: dt.datetime) -> float | None:
    bars = _bars_for(j, cache, "BTC-USDT", at)
    before = [b for b in bars if b[0] <= at]
    target = at - dt.timedelta(hours=24)
    earlier = [b for b in before if b[0] <= target]
    if not before or not earlier:
        return None
    now = before[-1][2]
    then = earlier[-1][2]
    if not now or not then:
        return None
    return (now / then - 1.0) * 100.0


def _trades(j: Any, rows: list[Any], horizons=(24.0,), fee: float = TAKER, prefer_open: bool = True) -> list[dict[str, Any]]:
    cache: dict[tuple[str, dt.date], list[Any]] = {}
    trades: list[dict[str, Any]] = []
    for row in rows:
        verdict = j._decision_label(row.answer)
        if verdict not in ("ORDER", "HOLD") or not row.symbols:
            continue
        symbol = row.symbols[0]
        bars = _bars_for(j, cache, symbol, row.started)
        entry_bar = _entry_bar(bars, row.started)
        if entry_bar is None:
            continue
        entry = _df(entry_bar, prefer_open)
        if not entry or entry <= 0:
            continue
        short = verdict == "ORDER" and _is_short(j, row)
        for hours in horizons:
            exit_bar = _exit_bar(bars, row.started, hours)
            if exit_bar is None:
                continue
            exit_px = _df(exit_bar, prefer_open)
            qty = NOTIONAL / entry
            move = (entry - exit_px) if short else (exit_px - entry)
            fees = 2.0 * NOTIONAL * fee
            trades.append(
                {
                    "id": str(row.decision_id[:8]),
                    "started": row.started,
                    "symbol": symbol,
                    "verdict": verdict,
                    "short": short,
                    "hours": hours,
                    "entry": entry,
                    "exit": exit_px,
                    "pnl": move * qty - fees,
                    "fee": fees,
                    "prior24": _prior_btc_move(j, cache, row.started),
                }
            )
    trades.sort(key=lambda t: t["started"])
    return trades




def _stats(values: list[float]) -> dict:
    wins = [v for v in values if v > 0]
    losses = [v for v in values if v < 0]
    return {
        "n": len(values),
        "net": sum(values),
        "mean": sum(values) / len(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "win_rate": 100.0 * len(wins) / len(values) if values else 0.0,
        "pf": (sum(wins) / -sum(losses)) if losses else float("inf"),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _block_bootstrap(values: list[float], reps: int, block: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randrange(n)
            for offset in range(block):
                sample.append(values[(start + offset) % n])
        means.append(sum(sample[:n]) / n)
    means.sort()
    return means[int(0.05 * reps)], means[int(0.95 * reps) - 1]


def _permutation_p(values: list[float], reps: int, seed: int) -> float:
    rng = random.Random(seed)
    observed = abs(sum(values) / len(values))
    hits = 0
    for _ in range(reps):
        flipped = sum(v if rng.random() < 0.5 else -v for v in values) / len(values)
        if abs(flipped) >= observed:
            hits += 1
    return (hits + 1) / (reps + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default="bybit-demo")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args(argv)

    j = _journal()
    rows = _decisions(j, args.broker, args.limit)

    print(f"# Backtest validation — `{args.broker}` ({args.limit} decisions)")
    print()
    print("## Variant grid (single trade per decision)")
    print()
    print("| horizon | fee | entry | n | net USD | mean | PF | win% |")
    print("|---|---|---|---|---|---|---|---|")
    grid = []
    for hours in (12.0, 24.0, 48.0):
        for label, fee in (("taker", TAKER), ("maker", MAKER), ("none", 0.0)):
            for entry_label, prefer_open in (("next-open", True), ("next-close", False)):
                trades = _trades(j, rows, horizons=(hours,), fee=fee, prefer_open=prefer_open)
                values = [t["pnl"] for t in trades]
                if not values:
                    continue
                s = _stats(values)
                grid.append((hours, label, entry_label, s, values))
                print(
                    f"| {hours:.0f}h | {label} | {entry_label} | {s['n']} | "
                    f"{s['net']:+.2f} | {s['mean']:+.3f} | {s['pf']:.2f} | {s['win_rate']:.0f} |"
                )

    base_hours, base_fee, base_entry, base_stats, base_values = grid[
        [i for i, g in enumerate(grid) if g[0] == 24.0 and g[1] == "taker" and g[2] == "next-open"][0]
    ]
    lo, hi = _block_bootstrap(base_values, BOOT_REPS, BLOCK, SEED)
    p_value = _permutation_p(base_values, PERM_REPS, SEED + 1)

    print()
    print("## Statistical test on the 24h taker next-open series")
    print()
    print(f"- trades: {base_stats['n']}, net {base_stats['net']:+.2f} USD")
    print(
        f"- mean {base_stats['mean']:+.3f} USD/trade, median {base_stats['median']:+.2f}, "
        f"stdev {base_stats['stdev']:.2f}"
    )
    print(
        f"- block bootstrap (block={BLOCK}, {BOOT_REPS} reps): mean 90% CI "
        f"[{lo:+.3f}, {hi:+.3f}] USD"
    )
    print(f"- sign-flip permutation ({PERM_REPS} reps): two-sided p = {p_value:.3f}")

    print()
    print("## Split by preceding BTC 24h return")
    print()
    print("| prior 24h band | n | net USD | mean | win% |")
    print("|---|---|---|---|---|")
    bands = [(-1e9, -5.0), (-5.0, -2.0), (-2.0, 0.0), (0.0, 2.0), (2.0, 1e9)]
    for lo_b, hi_b in bands:
        values = [
            t["pnl"]
            for t in _trades(j, rows, horizons=(24.0,), fee=TAKER)
            if t["prior24"] is not None and lo_b <= t["prior24"] < hi_b
        ]
        if not values:
            continue
        s = _stats(values)
        print(
            f"| [{lo_b:+.0f}%, {hi_b:+.0f}%) | {s['n']} | {s['net']:+.2f} | "
            f"{s['mean']:+.3f} | {s['win_rate']:.0f} |"
        )

    missing = sum(
        1
        for t in _trades(j, rows, horizons=(24.0,), fee=TAKER)
        if t["prior24"] is None
    )
    if missing:
        print()
        print(f"{missing} trade(s) had no BTC history to classify.")

    print()
    print(
        "*Limits: one trade per decision, overlapping windows, 4h cadence. The "
        "bootstrap treats the ordered series as the population, so a p-value above "
        "0.05 only says this sample cannot distinguish the mean from zero.*"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
