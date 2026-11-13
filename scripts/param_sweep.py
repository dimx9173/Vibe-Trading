"""Batch sweep: enumerate parameter grids, build the real trial matrix,
and pay for the search with deflated Sharpe. Blocks cherry-picking.

The nine falsification rounds all died the same way: a winner picked from a
grid, reported without paying for the grid. This CLI makes the grid itself
the artifact — every trial's return series is written to disk and the
deflated Sharpe sees the full spread, not a fabricated default.

Usage:
    PYTHONPATH=agent ./.venv/bin/python scripts/param_sweep.py \
        --strategy cross_sectional_momentum \
        --grid '{"lookback": [30, 60, 90], "skip": [3, 7, 14], "quantile": [0.1, 0.2, 0.3]}' \
        --out /tmp/sweep_out
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "agent")
sys.path.insert(0, "scripts")

from hypothesis_gate import evaluate

STRATEGIES: dict[str, str] = {
    "cross_sectional_momentum": (
        "Top/bottom-quantile cross-sectional momentum on a daily close panel, "
        "weekly rebalance, dollar-neutral."
    ),
}


def _load_panel(panel_dir: str) -> pd.DataFrame:
    """Load cached daily close CSVs into an aligned panel."""
    import glob
    import os

    frames: dict[str, pd.Series] = {}
    for path in glob.glob(os.path.join(panel_dir, "*.csv")):
        df = pd.read_csv(path)
        if len(df) < 120:
            continue
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.normalize()
        df = df.drop_duplicates("dt").set_index("dt")
        frames[os.path.basename(path)[:-4]] = df["c"].astype(float)
    return pd.DataFrame(frames).sort_index()


def _xsec_momentum_weekly(panel: pd.DataFrame, lookback: int, skip: int,
                          quantile: float, hold: int = 7,
                          fee_bps: float = 20.0) -> pd.Series:
    """Weekly net returns of a parameterised cross-sectional momentum leg."""
    dates = panel.index
    first = {c: panel[c].first_valid_index() for c in panel.columns}
    weeks: list[float] = []
    idx: list = []
    i = max(lookback, 180)
    while i + hold < len(dates):
        t0, t, t1 = dates[i - skip], dates[i], dates[i + hold]
        tb = dates[max(0, i - lookback)]
        elig = []
        for c in panel.columns:
            f = first[c]
            if f is None or (t - f).days < 180:
                continue
            a, b = panel[c].get(tb), panel[c].get(t0)
            nxt = panel[c].get(t1)
            if a is None or b is None or nxt is None:
                continue
            if bool(pd.isna(a) or pd.isna(b) or pd.isna(nxt)) or a <= 0:
                continue
            elig.append((c, float(b / a - 1), float(nxt / b - 1)))
        if len(elig) < 20:
            i += hold
            continue
        elig.sort(key=lambda r: r[1])
        k = max(1, int(len(elig) * quantile))
        lr = float(np.mean([r[2] for r in elig[-k:]]))
        sr = float(np.mean([r[2] for r in elig[:k]]))
        weeks.append(lr - sr - 2 * 2 * fee_bps / 1e4)
        idx.append(t1)
        i += hold
    return pd.Series(weeks, index=pd.DatetimeIndex(idx), name="ret")


def _open_registered_hypothesis(hypothesis_id: str) -> int | None:
    """Refuse the sweep unless the idea is registered and still open.

    A sweep that runs before registration is the cherry-pick the deflated
    Sharpe cannot see: the trial matrix pays for the search, but nothing links
    the search to the claim it is supposed to test, so the claim arrives after
    the data. Registration first makes every grid lookup a recorded test.

    Args:
        hypothesis_id: Registry identifier the sweep answers to.

    Returns:
        ``None`` when the hypothesis is open for testing, else exit code 2.
    """
    from src.hypotheses.registry import HypothesisRegistry

    open_states = {"exploring", "testing"}
    try:
        records = HypothesisRegistry().list()
    except Exception as exc:  # noqa: BLE001 - an unreadable registry cannot authorize a search
        print(f"hypothesis registry unreadable: {exc}", file=sys.stderr)
        return 2
    match = next((h for h in records if h.hypothesis_id == hypothesis_id), None)
    if match is None:
        print(
            f"unknown hypothesis {hypothesis_id!r}; register it first "
            f"(exploring -> testing) so the sweep answers to a recorded claim",
            file=sys.stderr,
        )
        return 2
    if match.status not in open_states:
        print(
            f"hypothesis {hypothesis_id!r} is {match.status!r}; reopen it as "
            f"'testing' before sweeping it again",
            file=sys.stderr,
        )
        return 2
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry: sweep, persist the trial matrix, judge the winner."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    ap.add_argument("--grid", required=True,
                    help="JSON mapping param name -> list of values")
    ap.add_argument("--panel-dir", default="/tmp/h5_panel")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--hypothesis-id", required=True)
    args = ap.parse_args(argv)

    gate = _open_registered_hypothesis(args.hypothesis_id)
    if gate is not None:
        return gate

    grid: dict[str, list] = json.loads(args.grid)
    names = list(grid)
    combos = list(itertools.product(*(grid[n] for n in names)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trials_dir = out / "trials"
    trials_dir.mkdir(exist_ok=True)

    panel = _load_panel(args.panel_dir)
    trial_frames: dict[str, pd.Series] = {}
    manifest = []
    for combo in combos:
        params = dict(zip(names, combo))
        tag = "_".join(f"{n}={v}" for n, v in params.items())
        s = _xsec_momentum_weekly(panel, int(params["lookback"]),
                                  int(params["skip"]),
                                  float(params["quantile"]))
        s.to_csv(trials_dir / f"{tag}.csv", index_label="date")
        trial_frames[tag] = s
        manifest.append({"params": params, "n_weeks": len(s),
                         "sharpe": round(float(s.mean() / s.std(ddof=1) * (52 ** 0.5)), 4)
                         if len(s) > 8 and s.std(ddof=1) else 0.0})

    matrix = pd.DataFrame(trial_frames)
    matrix.to_csv(out / "trial_matrix.csv")
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"strategy": args.strategy, "n_trials": len(combos),
                   "trials": manifest}, fh, indent=2, default=str)

    best = max(manifest, key=lambda m: m["sharpe"])
    n_trials = len(combos)
    trial_sharpes = np.array([
        float(matrix[c].mean() / matrix[c].std(ddof=1)) if matrix[c].std(ddof=1) else 0.0
        for c in matrix.columns
    ])
    winner = matrix[[c for c in matrix.columns
                     if all(f"{n}={v}" in c for n, v in best["params"].items())][0]]
    verdict = evaluate(winner.to_numpy(), n_trials, trial_sharpes=trial_sharpes)
    with open(out / "verdict.json", "w", encoding="utf-8") as fh:
        json.dump({"best_params": best, "verdict": verdict}, fh,
                  indent=2, default=str)

    from src.hypotheses.registry import HypothesisRegistry

    HypothesisRegistry().link_backtest(
        args.hypothesis_id,
        run_card_path=str((out / "verdict.json").resolve()),
        metrics={"n_trials": n_trials, "best": best,
                 "promote": verdict["promote"]},
        validation=verdict.get("gates", {}),
        notes=f"param_sweep.py {args.strategy} ({n_trials} trials)",
    )
    best_tag = "_".join(f"{n}={v}" for n, v in best["params"].items())
    print(json.dumps({"best": best_tag, "promote": verdict["promote"],
                      "dsr": verdict.get("gates", {}).get("deflated_sharpe"),
                      "n_trials": n_trials}, indent=2))
    return 0 if verdict["promote"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
