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
            if not isinstance(row, dict) or row.get("broker") not in (None, broker):
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


def build_report(broker: str, limit: int, markout: bool) -> str:
    """Assemble the Markdown report for one runner key."""
    rows = _load_decisions(broker, limit)
    run_map = _run_dirs_for(rows)
    lines: list[str] = []
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
        lines.append(
            "| {ts} | {did} | {verdict} | {tokens} | {calls} | {orders} | {marks} |".format(
                ts=row.started.strftime("%m-%d %H:%MZ"),
                did=row.decision_id[:8],
                verdict=verdict,
                tokens=f"{row.tokens:,}",
                calls=calls,
                orders=order_note,
                marks=marks,
            )
        )
    header = [
        f"# Live decision journal — `{broker}`",
        "",
        f"Decisions: {len(rows)} (newest first). Tokens = LLM spend for that tick.",
        "",
        "| time (UTC) | id | verdict | tokens | tools | ledger rows | post-decision move |",
        "|---|---|---|---|---|---|---|",
    ]
    return "\n".join(header + lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default="bybit-demo", help="runner key (default bybit-demo)")
    parser.add_argument("--limit", type=int, default=20, help="decisions to include")
    parser.add_argument("--no-markout", action="store_true", help="skip OHLCV fetches")
    args = parser.parse_args(argv)
    print(build_report(args.broker, args.limit, markout=not args.no_markout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
