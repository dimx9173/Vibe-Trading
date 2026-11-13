"""Non-price readings a trading tick observes must ground the numbers it cites.

A live tick answers with an indicator reading ("RSI 31.2") and an account fact
("balance 41,030"). Both arrived through tool calls the run can point at, yet
the observed-claim check only compared them against the instrument's price
envelope, so every such answer was rejected as ``numeric_claim_conflict`` and
the tick could never finish. These cases pin the reading pool that fixes it,
and keep the gate closed for a number nobody observed.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.grounding import GroundingLedger

QUOTE = {
    "status": "ok",
    "symbol": "BTC/USDT:USDT",
    "quote": {"bid": 73788.8, "ask": 73788.9, "last": 73788.9, "high": 75123.2, "low": 72686.6},
}
INDICATORS = {
    "ok": True,
    "symbol": "BTC-USDT",
    "interval": "4H",
    "latest_close": 73205.4,
    "indicators": {"rsi_14": 31.234, "macd": {"macd_line": -1801.0855}},
}
ACCOUNT = {
    "status": "ok",
    "balances": [{"asset": "USDT", "free": 41030.01918627, "used": 0.0, "total": 41030.01918627}],
}


def _tick_ledger(tmp_path: Path) -> GroundingLedger:
    """A ledger holding one tick's quote, indicator and account reads."""
    ledger = GroundingLedger(run_dir=tmp_path, user_message="live tick")
    for call_id, (tool, args, payload) in enumerate(
        [
            ("trading_quote", {"connection": "bybit-paper-trade", "symbol": "BTC-USDT"}, QUOTE),
            ("technical_indicators", {"symbol": "BTC-USDT", "interval": "4H"}, INDICATORS),
            ("trading_account", {"connection": "bybit-paper-trade"}, ACCOUNT),
        ],
        start=1,
    ):
        ledger.ingest_tool_result(
            tool_name=tool,
            arguments=args,
            result=json.dumps(payload),
            call_id=f"call_{call_id}",
            success=True,
        )
    return ledger


def test_indicator_reading_grounds_a_cited_rsi(tmp_path: Path) -> None:
    """RSI 31.234 came from technical_indicators, so quoting it is observed."""
    result = _tick_ledger(tmp_path).validate_final_answer(
        "BTC-USDT four-hour RSI 31.234, price 73788.9, structure still weak."
    )

    assert result.valid, [issue["message"] for issue in result.issues]


def test_account_balance_grounds_a_cited_balance(tmp_path: Path) -> None:
    """A balance the tick read is an observation, not a claim about BTC."""
    result = _tick_ledger(tmp_path).validate_final_answer(
        "BTC-USDT price 73788.9; USDT balance 41030.01918627; holding."
    )

    assert result.valid, [issue["message"] for issue in result.issues]


def test_unobserved_reading_is_still_rejected(tmp_path: Path) -> None:
    """The reading pool does not license a number no tool returned."""
    result = _tick_ledger(tmp_path).validate_final_answer(
        "BTC-USDT 4H RSI 77.777, price 73788.9, breakout confirmed."
    )

    assert not result.valid
    assert any(issue["code"] == "numeric_claim_conflict" for issue in result.issues)
