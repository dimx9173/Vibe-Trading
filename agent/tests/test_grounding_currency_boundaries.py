"""Currency identity survives magnitude marks without inventing word matches."""
import json

import pytest

from src.agent.grounding import GroundingLedger
from src.agent.grounding.figures import parse_figures_block, scan_figures


@pytest.mark.parametrize(('text', 'code'), [
    ('Value 2.5B JPY.', 'JPY'),
    ('Value 2.5亿 CNY.', 'CNY'),
    ('Value CHF 2.5.', 'CHF'),
    ('Value NZD 2.5.', 'NZD'),
    ('Value ₩2.5.', 'KRW'),
    ('Value ₹2.5.', 'INR'),
    ('Value 2.5 美元素描', None),
])
def test_currency_shape_boundaries(text, code):
    [figure] = scan_figures(text, parse_figures_block(text))
    assert figure.currency_code == code


@pytest.mark.parametrize(('unit', 'claim', 'valid'), [
    ('USD', '2.5B USD', True),
    ('USD', '2.5B JPY', False),
    ('CHF', 'CHF 2,500,000,000', True),
    ('CHF', 'USD 2,500,000,000', False),
    ('THB', 'THB 2,500,000,000', True),
    ('shares', 'USD 2,500,000,000', False),
])
def test_sec_currency_units_and_scaled_claims(tmp_path, unit, claim, valid):
    ledger = GroundingLedger(run_dir=tmp_path, user_message='Analyze HE.US debt')
    ledger.ingest_tool_result(
        tool_name='get_sec_filings', arguments={'ticker': 'HE'}, call_id='sec1', success=True,
        result=json.dumps({'ok': True, 'market': 'US', 'source': 'sec_edgar', 'data': {
            'ticker': 'HE', 'metric': {'unit': unit, 'points': [{'val': 2500000000}]}}}),
    )
    declared = '2.5B' if '2.5B' in claim else '2500000000'
    answer = f'HE.US debt: {claim}.\n```figures\n{declared} | observed | debt | sec1\n```'
    verdict = ledger.validate_final_answer(answer)
    assert verdict.valid is valid, verdict.issues


@pytest.mark.parametrize('side', ['bid', 'ask'])
def test_broker_quote_side_grounds_a_currency_marked_figure(tmp_path, side):
    """A connector's bid/ask is a price: a $-figure quoting it must ground.

    The 2026-11-06 live decision quoted ``Bid/Ask: $76,019.2 / $76,019.3`` and
    the gate rejected the bid with ``not_in_referenced_call`` because only
    last/close-like leaves counted as money, so two revision rounds burned the
    whole tick and the runner released the deterministic safe fallback.
    """
    bid, ask = 76019.2, 76019.3
    ledger = GroundingLedger(run_dir=tmp_path, user_message='autonomous BTC-USDT tick')
    ledger.ingest_tool_result(
        tool_name='trading_quote',
        arguments={'symbol': 'BTC-USDT'},
        call_id='tq1',
        success=True,
        result=json.dumps({
            'status': 'ok',
            'symbol': 'BTC/USDT:USDT',
            'quote': {'bid': bid, 'ask': ask, 'last': 76019.0, 'high': 76380.5, 'low': 74810.0},
        }),
    )
    quoted = bid if side == 'bid' else ask
    answer = (
        f'BTC-USDT {side} is ${quoted:,.1f}.\n'
        f'```figures\n{quoted} | observed | BTC-USDT {side} | tq1\n```'
    )

    verdict = ledger.validate_final_answer(answer)

    assert verdict.valid is True, verdict.issues
