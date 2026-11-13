"""The sweep refuses to run for an idea that is not registered and open.

A grid searched before registration is the cherry-pick deflated Sharpe cannot
see. These cases pin the precondition at the CLI boundary and prove the happy
path still produces a verdict and links it to the hypothesis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import param_sweep  # noqa: E402

GRID = '{"lookback": [30], "skip": [7], "quantile": [0.1]}'


@pytest.fixture()
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A throwaway hypothesis registry so the real ledger is untouched."""
    monkeypatch.setenv("VIBE_TRADING_HYPOTHESES_PATH", str(tmp_path / "hypotheses.json"))
    from src.hypotheses.registry import HypothesisRegistry

    return HypothesisRegistry()


def _register(registry, status: str) -> str:
    hyp = registry.create(
        title="sweep gate fixture",
        thesis="a registered claim the sweep answers to",
        universe="TEST-USDT",
        signal_definition="unit fixture",
    )
    if status != "exploring":
        registry.update(hyp.hypothesis_id, status=status)
    return hyp.hypothesis_id


def _panel(tmp_path: Path) -> Path:
    """A panel wide and long enough for one weekly momentum observation."""
    panel = tmp_path / "panel"
    panel.mkdir()
    for index in range(25):
        rows = ["ts,o,h,l,c,v"]
        price = 1.0 + index * 0.01
        for day in range(230):
            price *= 1.001 + index * 0.0002
            stamp = 1_700_000_000_000 + day * 86_400_000
            rows.append(f"{stamp},{price},{price},{price},{price},1000")
        (panel / f"SYM{index:02d}_USDT-USDT.csv").write_text("\n".join(rows) + "\n")
    return panel


def test_unknown_hypothesis_refuses_the_sweep(tmp_path: Path, registry) -> None:
    code = param_sweep.main([
        "--strategy", "cross_sectional_momentum", "--grid", GRID,
        "--out", str(tmp_path / "out"), "--hypothesis-id", "hyp_missing",
    ])

    assert code == 2
    assert not (tmp_path / "out" / "verdict.json").exists()


def test_rejected_hypothesis_must_be_reopened(tmp_path: Path, registry) -> None:
    hypothesis_id = _register(registry, "rejected")

    code = param_sweep.main([
        "--strategy", "cross_sectional_momentum", "--grid", GRID,
        "--out", str(tmp_path / "out"), "--hypothesis-id", hypothesis_id,
    ])

    assert code == 2


def test_missing_hypothesis_flag_is_a_usage_error(tmp_path: Path, registry) -> None:
    with pytest.raises(SystemExit):
        param_sweep.main([
            "--strategy", "cross_sectional_momentum", "--grid", GRID,
            "--out", str(tmp_path / "out"),
        ])


def test_open_hypothesis_runs_and_records_the_verdict(tmp_path: Path, registry) -> None:
    hypothesis_id = _register(registry, "testing")
    out = tmp_path / "out"

    code = param_sweep.main([
        "--strategy", "cross_sectional_momentum", "--grid", GRID,
        "--out", str(out), "--hypothesis-id", hypothesis_id,
        "--panel-dir", str(_panel(tmp_path)),
    ])

    assert code in (0, 1)
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["verdict"]["promote"] is False
    linked = registry.list()
    assert linked[0].run_cards, "the sweep must link its verdict to the hypothesis"
