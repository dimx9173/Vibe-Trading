"""Monthly promotion gate for research hypotheses.

Runs the repo's own validation stack against a candidate's weekly return
series and refuses promotion unless every gate passes. This is the executable
form of the nine-round falsification protocol: trial count is an input, so a
winning parameter cannot be reported without paying for the search that found
it.

Usage:
    PYTHONPATH=agent ./.venv/bin/python scripts/hypothesis_gate.py \
        --hypothesis-id hyp_... --returns /tmp/weekly.csv --n-trials 27
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "agent")

from backtest.validation import bootstrap_sharpe_ci, walk_forward_analysis
from src.hypotheses.registry import HypothesisRegistry
from src.quantlib.multipletesting import deflated_sharpe_ratio

WEEKLY_BARS_PER_YEAR = 52
MIN_OBSERVATIONS = 104
MIN_TRIALS = 1


def per_observation_sharpe(returns: np.ndarray) -> float:
    """Sharpe in per-observation units.

    ``quantlib.deflated_sharpe_ratio`` expects a per-observation Sharpe, while
    ``backtest.validation`` reports annualised figures. Annualising here and
    comparing there is the classic unit trap; this helper is the only place
    the conversion happens.
    """
    sd = float(returns.std(ddof=1))
    return float(returns.mean() / sd) if sd else 0.0


def evaluate(returns: np.ndarray, n_trials: int, seed: int = 20261110, trial_sharpes: np.ndarray | None = None) -> dict:
    """Run every gate on a weekly return series and return the verdict."""
    if n_trials > 1 and trial_sharpes is None:
        raise ValueError("n_trials>1 requires trial_sharpes: the spread across "
                         "the searched trials decides the deflated Sharpe")
    trial_sharpe_std = float(np.std(trial_sharpes, ddof=1)) if trial_sharpes is not None else 0.0
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    gates: dict[str, object] = {}
    ok = True

    gates["n_observations"] = int(len(r))
    if len(r) < MIN_OBSERVATIONS:
        gates["n_observations_ok"] = False
        return {"promote": False, "gates": gates,
                "reason": f"need >= {MIN_OBSERVATIONS} weekly observations"}

    equity = pd.Series(np.cumprod(1.0 + r), index=pd.RangeIndex(len(r)))
    wf = walk_forward_analysis(equity, trades=[], n_windows=4,
                               bars_per_year=WEEKLY_BARS_PER_YEAR)
    gates["walk_forward"] = {"consistency_rate": wf.get("consistency_rate"),
                             "sharpe_mean": wf.get("sharpe_mean"),
                             "profitable_windows": wf.get("profitable_windows")}
    if float(wf.get("consistency_rate") or 0.0) < 0.6:
        ok = False

    boot = bootstrap_sharpe_ci(equity, n_bootstrap=2000, confidence=0.95,
                               bars_per_year=WEEKLY_BARS_PER_YEAR, seed=seed)
    gates["bootstrap"] = {"observed_sharpe": boot.get("observed_sharpe"),
                          "ci_lower": boot.get("ci_lower"),
                          "ci_upper": boot.get("ci_upper"),
                          "prob_positive": boot.get("prob_positive")}
    if float(boot.get("ci_lower") or -1.0) <= 0.0:
        ok = False
    # Deflated Sharpe: pay for the search that produced the winner. The spread
    # of Sharpes across the searched trials is an input, never a default: a
    # fabricated 0.02 would silently decide the verdict.
    dsr = deflated_sharpe_ratio(
        observed_sharpe=per_observation_sharpe(r),
        n_trials=max(MIN_TRIALS, int(n_trials)),
        n_observations=len(r),
        trial_sharpe_std=trial_sharpe_std,
        skew=float(pd.Series(r).skew()),
        kurtosis=float(pd.Series(r).kurt()) + 3.0,  # non-excess
    )
    gates["deflated_sharpe"] = {"observed": round(dsr.observed_sharpe, 4),
                                "expected_max": round(dsr.expected_maximum_sharpe, 4),
                                "dsr": round(dsr.deflated_sharpe_ratio, 4),
                                "survives": dsr.survives}
    if not dsr.survives:
        ok = False

    peak = equity.cummax()
    mdd = float((equity / peak - 1.0).min())
    gates["max_drawdown"] = round(mdd, 4)
    if mdd < -0.25:
        ok = False

    half = len(r) // 2
    gates["halves"] = {"h1_sharpe": round(per_observation_sharpe(r[:half]), 4),
                       "h2_sharpe": round(per_observation_sharpe(r[half:]), 4)}
    if gates["halves"]["h1_sharpe"] <= 0 or gates["halves"]["h2_sharpe"] <= 0:
        ok = False

    return {"promote": ok, "gates": gates}


def main(argv: list[str] | None = None) -> int:
    """CLI entry: evaluate a returns CSV and optionally record the verdict."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--returns", required=True,
                    help="CSV with a 'ret' column of per-period returns")
    ap.add_argument("--hypothesis-id", default="",
                    help="registry id to link the verdict to")
    ap.add_argument("--n-trials", type=int, default=1,
                    help="how many parameter combinations were searched")
    ap.add_argument("--trials", default="",
                    help="CSV whose columns are every searched trial's returns; "
                         "required when --n-trials > 1")
    ap.add_argument("--column", default="ret")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.returns)
    if args.column not in df:
        print(f"missing column {args.column!r} in {args.returns}", file=sys.stderr)
        return 2
    trial_sharpes = None
    if args.n_trials > 1:
        if not args.trials:
            print("--n-trials > 1 requires --trials (one column per searched trial)",
                  file=sys.stderr)
            return 2
        tdf = pd.read_csv(args.trials)
        trial_sharpes = np.array([per_observation_sharpe(tdf[c].to_numpy(dtype=float))
                                  for c in tdf.columns])
    result = evaluate(df[args.column].to_numpy(), args.n_trials, trial_sharpes=trial_sharpes)

    if args.hypothesis_id:
        reg = HypothesisRegistry()
        reg.link_backtest(
            args.hypothesis_id,
            run_card_path=str(Path(args.returns).resolve()),
            metrics={"n_trials": args.n_trials,
                     "promote": result["promote"],
                     "gates": result.get("gates", {})},
            validation=result.get("gates", {}),
            notes="hypothesis_gate.py monthly evaluation",
        )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["promote"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
