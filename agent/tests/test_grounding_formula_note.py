"""Descriptive punctuation must preserve arithmetic and evidence checks."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.agent.grounding import GroundingLedger
from src.agent.grounding.policies import _evaluate_formula, _formula_in_note

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("separator", [":", "："])
@pytest.mark.parametrize(
    "note",
    [
        "scenario result{separator} 125 / 10000 * 100",
        "任意说明{separator} 125 / 10000 * 100",
        "αποτέλεσμα{separator} 125 / 10000 * 100",
        "scenario result{separator}125 / 10000 * 100",
        "scenario result{separator} 125 / 10000 * 100 = 1.25",
        "scenario result{separator} 125 / 10000 * 100, explanation",
        "scenario result{separator} 125 units / 10000 units * 100",
    ],
)
def test_descriptive_colon_preserves_formula(note: str, separator: str) -> None:
    expected = _evaluate_formula("125 / 10000 * 100")
    actual = _formula_in_note(note.format(separator=separator))

    assert expected is not None
    assert actual is not None
    assert actual[:2] == expected[:2]
    assert ast.dump(actual[2]) == ast.dump(expected[2])


@pytest.mark.parametrize("separator", [":", "："])
@pytest.mark.parametrize(
    "expression",
    [
        "125",
        "125 / 0",
        "125 +",
        "(125 + 100",
        "125 + unknown",
        "125 // 100",
        "125 ** 4",
        "f(125) + 100",
        "__import__('os')",
        "[125, 100][0] + 100",
    ],
)
def test_descriptive_colon_does_not_extend_evaluator(
    expression: str, separator: str
) -> None:
    assert _formula_in_note(f"scenario result{separator} {expression}") is None


@pytest.mark.parametrize("separator", [":", "："])
@pytest.mark.parametrize(
    ("value", "expression", "ref", "with_evidence", "reason"),
    [
        ("12.4%", "(112.4 - 100.0) / 100.0", "quote", True, None),
        (
            "99.0%",
            "(112.4 - 100.0) / 100.0",
            "quote",
            True,
            "derivation_result_mismatch",
        ),
        ("12.4%", "(56.2 - 50.0) / 50.0", "quote", True, "formula_not_anchored"),
        ("12.4%", "(112.4 - 100.0) / 100.0", "quote", False, "no_evidence"),
        ("12.4%", "(112.4 - 100.0) / 0", "quote", True, "formula_not_evaluable"),
        (
            "13.4%",
            "(112.4 - 100.0 + 1) / 100.0",
            "quote",
            True,
            "additive_operand_not_observed",
        ),
    ],
)
def test_descriptive_colon_keeps_derived_grounding_checks(
    tmp_path: Path,
    separator: str,
    value: str,
    expression: str,
    ref: str,
    with_evidence: bool,
    reason: str | None,
) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="Analyze AAPL.US")
    if with_evidence:
        ledger.ingest_tool_result(
            tool_name="get_market_data",
            arguments={"codes": ["AAPL.US"]},
            result=json.dumps(
                {
                    "AAPL.US": [
                        {"trade_date": "2026-08-03", "close": 100.0},
                        {"trade_date": "2026-09-02", "close": 112.4},
                    ]
                }
            ),
            call_id="quote",
            success=True,
        )
    result = ledger.validate_final_answer(
        f"AAPL.US (USD) cumulative return: {value}.\n\n"
        f"```figures\n{value} | derived | scenario result{separator} {expression} | {ref}\n```"
    )

    assert result.valid is (reason is None), result.issues
    if reason is not None:
        assert reason in {issue.get("reason") for issue in result.issues}, result.issues


def test_a_rounding_call_restates_an_observed_measurement(tmp_path: Path) -> None:
    """``round(x, n)`` is presentation of an observation, not a new number.

    Live decisions quote indicators at the precision they reason with
    ("RSI 38.4") while the tool returns full float precision. Before this
    rule, every such note failed with ``formula_not_evaluable``, two revision
    rounds exhausted the tick, and the runner released its safe fallback.
    """
    ledger = GroundingLedger(run_dir=tmp_path, user_message="autonomous BTC-USDT tick")
    ledger.ingest_tool_result(
        tool_name="technical_indicators",
        arguments={"symbol": "BTC-USDT"},
        call_id="ti1",
        success=True,
        result=json.dumps({"ok": True, "symbol": "BTC-USDT", "indicators": {"rsi_14": 38.399791901175355}}),
    )
    answer = (
        "BTC-USDT RSI read 38.4.\n"
        "```figures\n38.4 | derived | round(38.399791901175355, 1) | ti1\n```"
    )

    verdict = ledger.validate_final_answer(answer)

    assert verdict.valid is True, verdict.issues


def test_a_rounding_call_cannot_anchor_an_unobserved_measurement(tmp_path: Path) -> None:
    """Rounding is not a way to invent a number: the operand must be evidence."""
    ledger = GroundingLedger(run_dir=tmp_path, user_message="autonomous BTC-USDT tick")
    ledger.ingest_tool_result(
        tool_name="technical_indicators",
        arguments={"symbol": "BTC-USDT"},
        call_id="ti1",
        success=True,
        result=json.dumps({"ok": True, "symbol": "BTC-USDT", "indicators": {"rsi_14": 38.399791901175355}}),
    )
    answer = (
        "BTC-USDT RSI read 12.3.\n"
        "```figures\n12.3 | derived | round(51.7712, 1) | ti1\n```"
    )

    verdict = ledger.validate_final_answer(answer)

    assert verdict.valid is False
    assert {issue.get("reason") for issue in verdict.issues} == {"formula_not_anchored"}
