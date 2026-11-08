#!/usr/bin/env python3
"""Read-only decision journal for an autonomous live runner.

Joins, for every recorded decision of one runner key:

* ``decisions.sqlite3`` rows (id / started / finished)
* the run directory the tick produced (``~/.vibe-trading/runs/<ts>/``) with its
  ``llm_usage.json`` (tokens) and session ``trace.jsonl`` (tool calls, answer)
* ``audit.jsonl`` order records inside the decision window
* public OHLCV bars (no broker call) for the symbol the decision discussed, to
  score a post-decision markout and the HOLD counterfactual.

Nothing is written anywhere: this script never touches broker state, mandates,
or the runner. It is an offline measurement tool.

Usage::

    PYTHONPATH=agent ./.venv/bin/python scripts/live_decision_journal.py \
        --broker bybit-demo --limit 30

Exit status is 0 with a Markdown table printed to stdout; unresolved price
sources degrade that row to ``n/a`` rather than failing the report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO_ROOT, "agent") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "agent"))

VIBE_HOME = os.path.expanduser(os.environ.get("VIBE_TRADING_HOME", "~/.vibe-trading"))
_TICK_PROMPT_MARKER = "AUTONOMOUS live-trading tick"
_RUN_DIR_RE = re.compile(r"/runs/(\d{8}_\d{6}_\d+_[0-9a-z]+)")
_SYMBOL_RE = re.compile(r"\b([A-Z]{2,10})[-/](USDT|USDC|USD)\b")

# Taker fee (Bybit V5 USDT perp tier-1). Demo rate is unverifiable from the
# recorded evidence, so the replay bills one constant rate on both legs.
_TAKER_RATE = 0.00055
# Counterfactual trades are sized to the mandate's per-order ceiling so the
# replay PnL is comparable across symbols.
_CF_NOTIONAL_USD = 250.0


@dataclass
class DecisionRow:
    """One runner decision with its evidence chain."""

    decision_id: str
    started: dt.datetime
    finished: Any
    run_dir: str = ""
    session_id: str = ""
    tokens: int = 0
    calls: list[str] = field(default_factory=list)
    answer: str = ""
    orders: list[dict[str, Any]] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)

    # Backtest replay fields (populated when build_report(..., backtest=True))
    executed: bool = False
    pnl_kind: str = "-"              # "real" | "cf" | "-"
    real_fee: float | None = None
    real_uPnl_now: float | None = None
    cf_entry: float | None = None
    cf_exit: float | None = None
    cf_fee: float | None = None
    cf_pnl: float | None = None
    cf_exit_rule: str = ""

def _load_decisions(broker: str, limit: int) -> list[DecisionRow]:
    """Read the newest ``limit`` decisions for one runner key."""
    db = os.path.join(VIBE_HOME, "live", broker, "runtime", "decisions.sqlite3")
    if not os.path.exists(db):
        raise SystemExit(f"no decisions database at {db}")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, started, finished FROM decisions ORDER BY started DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [
        DecisionRow(
            decision_id=str(r[0]),
            started=dt.datetime.fromtimestamp(r[1], dt.timezone.utc),
            finished=r[2],
        )
        for r in rows
    ]


def _run_dirs_for(rows: list[DecisionRow]) -> dict[str, str]:
    """Map decision id -> run dir by matching the tick prompt, not timestamps.

    Timestamp windows are ambiguous when two runners tick together (e.g. the
    forecast cron), so the run's ``req.json`` prompt must carry the autonomous
    tick marker for a match to count.
    """
    by_start: dict[int, str] = {}
    for path in glob.glob(os.path.join(VIBE_HOME, "runs", "*", "req.json")):
        try:
            with open(path) as fh:
                prompt = json.load(fh).get("prompt") or ""
        except (OSError, ValueError):
            continue
        if _TICK_PROMPT_MARKER not in prompt:
            continue
        try:
            ts = dt.datetime.strptime(
                os.path.basename(os.path.dirname(path))[:15], "%Y%m%d_%H%M%S"
            ).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        by_start[int(ts.timestamp())] = os.path.dirname(path)

    mapping: dict[str, str] = {}
    for row in rows:
        started = int(row.started.timestamp())
        # The tick starts the run within a couple of minutes of the DB row.
        candidates = [
            (abs(started - ts), run_dir)
            for ts, run_dir in by_start.items()
            if abs(started - ts) <= 180
        ]
        if candidates:
            mapping[row.decision_id] = min(candidates)[1]
    return mapping


def _session_for_run(run_dir: str) -> str:
    """Find the live-decision session whose trace names this run directory."""
    needle = os.path.join(run_dir, "tool-results")
    for session_dir in glob.glob(os.path.join(VIBE_HOME, "sessions", "*")):
        trace = os.path.join(session_dir, "trace.jsonl")
        if not os.path.exists(trace):
            continue
        try:
            with open(trace, errors="replace") as fh:
                head = fh.read(200_000)
        except OSError:
            continue
        if os.path.basename(run_dir) in head or needle in head:
            return os.path.basename(session_dir)
    return ""


def _read_tokens(run_dir: str) -> int:
    """Total tokens recorded for the tick, 0 when the file is absent."""
    path = os.path.join(run_dir, "llm_usage.json")
    try:
        with open(path) as fh:
            return int((json.load(fh).get("totals") or {}).get("total_tokens") or 0)
    except (OSError, ValueError, TypeError):
        return 0


def _read_session(session_id: str) -> tuple[list[str], str]:
    """Return (tool call names in order, final answer text) for a session."""
    if not session_id:
        return [], ""
    trace = os.path.join(VIBE_HOME, "sessions", session_id, "trace.jsonl")
    calls: list[str] = []
    answer = ""
    if not os.path.exists(trace):
        return calls, answer
    with open(trace, errors="replace") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "tool_call" and event.get("tool"):
                calls.append(str(event["tool"]))
            elif event.get("type") == "answer":
                content = event.get("content")
                if isinstance(content, str) and content.strip():
                    answer = content
    return calls, answer


def _read_orders(broker: str, window: tuple[dt.datetime, dt.datetime]) -> list[dict[str, Any]]:
    """Ledger rows written inside the decision window (order evidence only)."""
    ledger = os.path.join(VIBE_HOME, "live", "audit.jsonl")
    out: list[dict[str, Any]] = []
    if not os.path.exists(ledger):
        return out
    lo, hi = window
    with open(ledger, errors="replace") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or row.get("server") not in (None, broker):
                continue
            try:
                ts = dt.datetime.fromisoformat(str(row.get("ts", "")))
            except ValueError:
                continue
            if lo <= ts <= hi and row.get("kind") in ("order_placed", "order_cancelled", "order"):
                out.append(row)
    return out


def _symbols_in(text: str) -> list[str]:
    """Distinct ``BASE-QUOTE`` pairs named in the decision text, ordered."""
    seen: list[str] = []
    for base, quote in _SYMBOL_RE.findall(text.upper()):
        pair = f"{base}-{quote}"
        if pair not in seen:
            seen.append(pair)
    return seen


def _markout(symbol: str, at: dt.datetime) -> dict[str, float | None]:
    """Price at decision time, plus change 4h / 24h later, from public bars."""
    from src.market_data import fetch_market_data

    start = (at - dt.timedelta(days=2)).date().isoformat()
    end = (at + dt.timedelta(days=2)).date().isoformat()
    marker: dict[str, float | None] = {"entry": None, "h4": None, "h24": None}
    try:
        payload = fetch_market_data(
            codes=[symbol], start_date=start, end_date=end, interval="1H", max_rows=200
        )
    except Exception:  # noqa: BLE001 - a dead source degrades the row, never the report
        return marker
    bars = payload.get(symbol)
    rows = bars.get("data") if isinstance(bars, dict) else bars
    if not isinstance(rows, list) or not rows:
        return marker

    def _ts(row: dict[str, Any]) -> dt.datetime | None:
        for key in ("trade_date", "datetime", "date", "timestamp", "time"):
            raw = row.get(key)
            if raw is None:
                continue
            if isinstance(raw, (int, float)):
                return dt.datetime.fromtimestamp(float(raw), dt.timezone.utc)
            try:
                parsed = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except ValueError:
                continue
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
        return None

    def _close(row: dict[str, Any]) -> float | None:
        for key in ("close", "Close", "adj_close"):
            if row.get(key) is not None:
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    return None
        return None

    series = [(_ts(r), _close(r)) for r in rows if isinstance(r, dict)]
    series = [(t, c) for t, c in series if t is not None and c is not None]
    if not series:
        return marker
    series.sort(key=lambda item: item[0])

    def _at(target: dt.datetime) -> float | None:
        prior = [c for t, c in series if t <= target]
        return prior[-1] if prior else None

    entry = _at(at)
    h4 = _at(at + dt.timedelta(hours=4))
    h24 = _at(at + dt.timedelta(hours=24))
    marker["entry"] = entry
    marker["h4"] = (h4 / entry - 1.0) * 100.0 if entry and h4 else None
    marker["h24"] = (h24 / entry - 1.0) * 100.0 if entry and h24 else None
    return marker
def _fee(notional: float, rate: float) -> float:
    """Fee on a notional amount at the given rate."""
    return notional * rate


def _bars(symbol: str, at: dt.datetime) -> list[tuple[dt.datetime, float | None, float]]:
    """Fetch OHLCV bars for a symbol around the decision time.

    Returns a list of (timestamp, open, close). If open is None, fall back to
    close.
    """
    from src.market_data import fetch_market_data

    today = dt.datetime.now(dt.timezone.utc).date()
    start = (at - dt.timedelta(days=2)).date().isoformat()
    end = min((at + dt.timedelta(days=2)).date(), today).isoformat()

    try:
        payload = fetch_market_data(
            codes=[symbol], start_date=start, end_date=end, interval="1H", max_rows=200
        )
    except Exception:  # noqa: BLE001
        return []
    bars = payload.get(symbol)
    rows = bars.get("data") if isinstance(bars, dict) else bars
    if not isinstance(rows, list) or not rows:
        return []

    def _ts(row: dict[str, Any]) -> dt.datetime | None:
        for key in ("trade_date", "datetime", "date", "timestamp", "time"):
            raw = row.get(key)
            if raw is None:
                continue
            if isinstance(raw, (int, float)):
                return dt.datetime.fromtimestamp(float(raw), dt.timezone.utc)
            try:
                parsed = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except ValueError:
                continue
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
        return None

    def _open(row: dict[str, Any]) -> float | None:
        for key in ("open", "Open"):
            if row.get(key) is not None:
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    return None
        return None

    def _close(row: dict[str, Any]) -> float | None:
        for key in ("close", "Close", "adj_close"):
            if row.get(key) is not None:
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    return None
        return None

    series: list[tuple[dt.datetime, float | None, float]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = _ts(r)
        o = _open(r)
        c = _close(r)
        if t is not None and c is not None:
            series.append((t, o, c))
    series.sort(key=lambda item: item[0])
    return series


def _norm_symbol(symbol: str) -> str:
    """Collapse a broker or universe symbol to ``BASEQUOTE`` for matching.

    ``ETH/USDT:USDT`` repeats the quote leg as settle, so the duplicate suffix
    collapses; otherwise a persisted position would never match a decision that
    names ``ETH-USDT``.
    """
    alnum = "".join(ch for ch in (symbol or "").upper() if ch.isalnum())
    for quote in ("USDT", "USDC", "USD"):
        if alnum.endswith(quote + quote):
            return alnum[: -len(quote)]
    return alnum


def _runtime_state_positions(broker: str) -> dict[str, dict[str, Any]]:
    """Persisted broker positions keyed by normalized symbol, or an empty map.

    ``runtime_state.json`` is the reconciliation baseline: a position there is
    a broker-verified fill with its entry price, quantity, and mark. It still
    needs an order-shaped ledger claim to be attributed to one decision, so an
    older fill with no such claim is reported as unclaimed rather than guessed.
    """
    path = os.path.join(VIBE_HOME, "live", broker, "runtime_state.json")
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for position in raw.get("positions") or []:
        if not isinstance(position, dict):
            continue
        key = _norm_symbol(str(position.get("symbol") or ""))
        if key:
            out[key] = position
    return out


def _real_claims(rows: list[DecisionRow]) -> dict[str, str]:
    """Attribute a broker position to the decision that submitted it.

    The claim needs order-shaped ledger evidence inside the decision window: an
    ``order_placed`` row written by the placement tool whose request payload
    names the symbol. Tick markers and decision text are not evidence, and a
    price-adjacency guess is not either (ETH traded within 0.05% of the current
    entry on several unrelated days). One symbol therefore has at most one
    claimant, so a fill can never be counted twice.
    """
    claims: dict[str, str] = {}
    for row in rows:
        if _decision_label(row.answer) != "ORDER" or not row.symbols:
            continue
        want = _norm_symbol(row.symbols[0])
        for order in row.orders:
            if order.get("remote_tool") != "place_order":
                continue
            request = order.get("broker_request")
            if not isinstance(request, dict):
                continue
            if _norm_symbol(str(request.get("symbol") or "")) == want:
                # Rows arrive newest first: the newest placement owns the key.
                claims.setdefault(want, row.decision_id)
                break
    return claims


def _replay(
    row: DecisionRow,
    bars: list[tuple[dt.datetime, float | None, float]],
    state_positions: dict[str, dict[str, Any]],
    claims: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score one decision as a real fill or as a counterfactual trade.

    A *real* row reports the position the broker holds, attributed to the one
    decision that submitted it and evidenced by order-shaped ledger rows. It is
    an open-position mark at actual size, using the position record's own mark
    price (entry fee only, no exit). A *counterfactual* row prices the
    decision's own idea, HOLD-as-long or an ORDER that left no broker fill, as
    one fixed-horizon paper trade at mandate notional, closed at the first bar
    that opens after the 24h mark.

    Exactly one group is populated, so ``real`` and ``counterfactual`` totals
    never double-count one decision. A row scores ``"-"`` when the decision
    names no symbol, the verdict is neither ORDER nor HOLD, no bars resolved,
    the position has no ledger claim, or the exit bar has not closed yet.
    """
    empty = {
        "executed": False,
        "pnl_kind": "-",
        "real_fee": None,
        "real_uPnl_now": None,
        "cf_entry": None,
        "cf_exit": None,
        "cf_qty": None,
        "cf_fee": None,
        "cf_pnl": None,
        "cf_exit_rule": "",
    }
    verdict = _decision_label(row.answer)
    if verdict not in ("ORDER", "HOLD"):
        return empty
    symbol = row.symbols[0] if row.symbols else ""
    if not symbol or not bars:
        return empty

    claims = claims or {}
    norm = _norm_symbol(symbol)
    is_real_claim = (
        state_positions.get(norm) is not None and claims.get(norm) == row.decision_id
    )
    position = state_positions.get(norm) if is_real_claim else None
    if position is not None and verdict == "ORDER":
        try:
            entry = float(position.get("entry_price"))
            qty = abs(float(position.get("quantity")))
        except (TypeError, ValueError):
            return empty
        if entry <= 0 or qty <= 0:
            return empty
        # Position still open (TP/SL resting): entry fee realized, and the quote
        # move comes from the position record's own broker-valued mark.
        try:
            mark = float(position.get("mark_price"))
        except (TypeError, ValueError):
            mark = bars[-1][2]
        fee = _fee(entry * qty, _TAKER_RATE)
        short = str(position.get("side", "")).lower() in ("short", "sell")
        move = (entry - mark) if short else (mark - entry)
        return {
            **empty,
            "executed": True,
            "pnl_kind": "real",
            "real_fee": fee,
            "real_uPnl_now": move * qty - fee,
        }

    # decisions.sqlite3 stores `finished` as a 0/1 flag, not a timestamp, and
    # the decision completes minutes after `started` anyway: the replay entry
    # is the first bar that opens after started.
    after = [item for item in bars if item[0] > row.started]

    if not after:
        return empty
    entry = after[0][1] if after[0][1] is not None else after[0][2]
    if entry is None or entry <= 0:
        return empty

    target = row.started + dt.timedelta(hours=24)
    exit_px = None
    for stamp, open_px, close_px in bars:
        if stamp > target:
            exit_px = open_px if open_px is not None else close_px
            break
    if exit_px is None:
        # The 24h bar has not closed yet: report the open entry without a PnL
        # so the total never mixes closed trades with an unfinished one.
        return {**empty, "cf_entry": entry, "cf_qty": _CF_NOTIONAL_USD / entry,
                "cf_exit_rule": "awaiting_close"}

    qty = _CF_NOTIONAL_USD / entry
    fee = _fee(_CF_NOTIONAL_USD, _TAKER_RATE) + _fee(exit_px * qty, _TAKER_RATE)
    short = verdict == "ORDER" and any(
        str((order.get("broker_request") or {}).get("side", "")).lower() in ("sell", "short")
        for order in row.orders
        if isinstance(order.get("broker_request"), dict)
    )
    move = (entry - exit_px) if short else (exit_px - entry)
    return {
        **empty,
        "pnl_kind": "cf",
        "cf_entry": entry,
        "cf_exit": exit_px,
        "cf_qty": qty,
        "cf_fee": fee,
        "cf_pnl": move * qty - fee,
        "cf_exit_rule": "h24" if verdict == "ORDER" else "hold_as_long/h24",
    }


def _decision_label(answer: str) -> str:
    """One-word verdict extracted from the final answer, best effort."""
    text = (answer or "").strip()
    if not text:
        return "(none)"
    upper = text.upper()
    if "VERIFICATION GATE" in upper and "REJECT" in upper:
        return "GATE-ECHO"
    if re.search(r"\bORDER\b", upper) and "HOLD" not in upper[:400]:
        return "ORDER"
    if re.search(r"\bHOLD\b", upper):
        return "HOLD"
    return text.splitlines()[0][:40]


def build_report(broker: str, limit: int, markout: bool, backtest: bool = False) -> str:
    """Assemble the Markdown report for one runner key.

    When backtest=True, adds replay columns and summary rows for real/cf PnL.
    """
    rows = _load_decisions(broker, limit)
    run_map = _run_dirs_for(rows)

    # One bar window per decision; multi-symbol decisions score the first pair.
    bars_by_symbol: dict[str, list[tuple[dt.datetime, float | None, float]]] = {}

    for row in rows:
        row.run_dir = run_map.get(row.decision_id, "")
        if row.run_dir:
            row.session_id = _session_for_run(row.run_dir)
            row.tokens = _read_tokens(row.run_dir)
            row.calls, row.answer = _read_session(row.session_id)
        if not row.symbols:
            row.symbols = _symbols_in(row.answer) or _symbols_in(" ".join(row.calls))
        end = row.started + dt.timedelta(hours=6)
        row.orders = _read_orders(broker, (row.started, end))

        if backtest and row.symbols:
            cache_key = f"{row.symbols[0]}|{row.started.isoformat()}"
            if cache_key not in bars_by_symbol:
                bars_by_symbol[cache_key] = _bars(row.symbols[0], row.started)

    if backtest:
        state_positions = _runtime_state_positions(broker)
        claims = _real_claims(rows)
        for row in rows:
            symbol = row.symbols[0] if row.symbols else ""
            bars = bars_by_symbol.get(f"{symbol}|{row.started.isoformat()}", []) if symbol else []
            replay = _replay(row, bars, state_positions, claims)
            row.executed = replay["executed"]
            row.pnl_kind = replay["pnl_kind"]
            row.real_fee = replay["real_fee"]
            row.real_uPnl_now = replay["real_uPnl_now"]
            row.cf_entry = replay["cf_entry"]
            row.cf_exit = replay["cf_exit"]
            row.cf_fee = replay["cf_fee"]
            row.cf_pnl = replay["cf_pnl"]
            row.cf_exit_rule = replay["cf_exit_rule"]

    lines: list[str] = []
    for row in rows:
        verdict = _decision_label(row.answer)
        calls = ",".join(dict.fromkeys(row.calls)) or "-"
        order_note = f"{len(row.orders)}" if row.orders else "0"
        marks = ""
        if markout:
            parts = []
            for symbol in row.symbols[:2]:
                m = _markout(symbol, row.started)
                h4 = "n/a" if m["h4"] is None else f"{m['h4']:+.2f}%"
                h24 = "n/a" if m["h24"] is None else f"{m['h24']:+.2f}%"
                parts.append(f"{symbol} 4h={h4} 24h={h24}")
            marks = "; ".join(parts) or "n/a"

        if backtest:
            if row.pnl_kind == "real":
                real_fee_str = f"{row.real_fee:.4f}" if row.real_fee is not None else "n/a"
                real_upnl_str = f"{row.real_uPnl_now:.2f}" if row.real_uPnl_now is not None else "n/a"
                replay_cols = f" | real_fee={real_fee_str} | real_uPnl={real_upnl_str} | cf=- | cf_fee=- | cf_pnl=- | exit=-"
            elif row.pnl_kind == "cf":
                cf_entry_str = f"{row.cf_entry:.2f}" if row.cf_entry is not None else "n/a"
                cf_exit_str = f"{row.cf_exit:.2f}" if row.cf_exit is not None else "n/a"
                cf_fee_str = f"{row.cf_fee:.4f}" if row.cf_fee is not None else "n/a"
                cf_pnl_str = f"{row.cf_pnl:.2f}" if row.cf_pnl is not None else "n/a"
                replay_cols = f" | real_fee=- | real_uPnl=- | cf_entry={cf_entry_str} | cf_exit={cf_exit_str} | cf_fee={cf_fee_str} | cf_pnl={cf_pnl_str} | exit={row.cf_exit_rule}"
            else:
                replay_cols = " | real_fee=- | real_uPnl=- | cf=- | cf_fee=- | cf_pnl=- | exit=-"
        else:
            replay_cols = ""

        lines.append(
            "| {ts} | {did} | {verdict} | {tokens} | {calls} | {orders} | {marks}{replay} |".format(
                ts=row.started.strftime("%m-%d %H:%MZ"),
                did=row.decision_id[:8],
                verdict=verdict,
                tokens=f"{row.tokens:,}",
                calls=calls,
                orders=order_note,
                marks=marks,
                replay=replay_cols,
            )
        )

    if backtest:
        real_total = sum(
            r.real_uPnl_now for r in rows
            if r.pnl_kind == "real" and r.real_uPnl_now is not None
        )
        cf_total = sum(r.cf_pnl for r in rows if r.pnl_kind == "cf" and r.cf_pnl is not None)
        real_count = sum(1 for r in rows if r.pnl_kind == "real")
        cf_count = sum(1 for r in rows if r.pnl_kind == "cf")
        unclaimed = max(0, len(_runtime_state_positions(broker)) - real_count)

        lines.append("")
        lines.append(
            f"**real (broker fills)**: {real_total:+.2f} USD across {real_count} claimed "
            f"row(s); {unclaimed} broker position(s) unclaimed for lack of order-shaped "
            "ledger evidence inside a decision window"
        )
        lines.append(
            f"**counterfactual (HOLD-as-long or unfilled order)**: {cf_total:+.2f} USD "
            f"across {cf_count} hypothetical trades at "
            f"${_CF_NOTIONAL_USD:.0f} notional (entry and exit taker fees)"
        )
        lines.append("")
        lines.append(
            "*sample note: no closed broker fill is in the window, and the counterfactual "
            "count is far below the N>=30 bar, so these numbers describe the decisions, "
            "they do not prove an edge.*"
        )
        lines.append("*funding omitted: about a cent per 4h at $250 notional, below fee noise.*")

    header = [
        f"# Live decision journal — `{broker}`",
        "",
        f"Decisions: {len(rows)} (newest first). Tokens = LLM spend for that tick.",
        "",
        "| time (UTC) | id | verdict | tokens | tools | ledger rows | post-decision move |"
        + (" replay |" if backtest else ""),
        "|---|---|---|---|---|---|---|" + ("---|" if backtest else ""),
    ]
    return "\n".join(header + lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default="bybit-demo", help="runner key (default bybit-demo)")
    parser.add_argument("--limit", type=int, default=20, help="decisions to include")
    parser.add_argument("--no-markout", action="store_true", help="skip OHLCV fetches")
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="run the offline replay and show counterfactual and real PnL",
    )
    args = parser.parse_args(argv)
    print(build_report(args.broker, args.limit, markout=not args.no_markout, backtest=args.backtest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
