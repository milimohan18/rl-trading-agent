"""
Evaluation script — compare trained agents against a buy-and-hold baseline.

Usage::

    python evaluate.py
    python evaluate.py --ticker EURUSD=X

Loads previously trained DQN and PPO models, runs them on the **held-out
test set only**, computes a suite of performance metrics, and prints a
comparison table.  Results are persisted to ``results/metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CONFIG
from utils.data_loader import load_split, run_pipeline
from utils.indicators import prepare_features, get_feature_columns, add_indicators
from env.trading_env import TradingEnv
from agents.dqn_agent import load_dqn
from agents.ppo_agent import load_ppo


# ── Metric helpers ──────────────────────────────────────────────────────────

def _total_return(portfolio_values: np.ndarray) -> float:
    """Percentage total return."""
    return float((portfolio_values[-1] / portfolio_values[0] - 1.0) * 100.0)


def _sharpe_ratio(daily_returns: np.ndarray, risk_free: float) -> float:
    """Annualised Sharpe ratio."""
    if len(daily_returns) < 2 or daily_returns.std() < 1e-10:
        return 0.0
    daily_rf = risk_free / 252.0
    excess = daily_returns.mean() - daily_rf
    return float(excess / daily_returns.std() * np.sqrt(252))


def _max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum drawdown in percent."""
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    return float(drawdowns.min() * 100.0)


def _win_rate(trade_returns: List[float]) -> float:
    """Percentage of profitable trades."""
    if not trade_returns:
        return 0.0
    wins = sum(1 for r in trade_returns if r > 0)
    return float(wins / len(trade_returns) * 100.0)


def _calmar_ratio(total_ret_pct: float, max_dd_pct: float) -> float:
    """Calmar ratio = annualised return / |max drawdown|."""
    if abs(max_dd_pct) < 1e-10:
        return 0.0
    return float(total_ret_pct / abs(max_dd_pct))


# ── Run agent on environment ───────────────────────────────────────────────

def run_agent(
    model: Any,
    env: TradingEnv,
    deterministic: bool = True,
) -> Tuple[List[float], List[float], int]:
    """Roll out a trained agent on *env* and record portfolio values.

    Parameters
    ----------
    model : stable_baselines3 model
        Trained ``DQN`` or ``PPO`` model.
    env : TradingEnv
        Test environment.
    deterministic : bool
        Whether to use deterministic actions.

    Returns
    -------
    tuple
        ``(portfolio_values, trade_returns, trade_count)``
    """
    obs, info = env.reset(seed=CONFIG["seed"])
    portfolio_values: List[float] = [info["portfolio_value"]]
    trade_returns: List[float] = []
    prev_position = 0
    prev_value = info["portfolio_value"]
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        portfolio_values.append(info["portfolio_value"])

        # Track per-trade returns
        current_position = info["position"]
        if prev_position != 0 and current_position != prev_position:
            pnl = info["portfolio_value"] - prev_value
            trade_returns.append(pnl)
            prev_value = info["portfolio_value"]
        if current_position != 0 and prev_position == 0:
            prev_value = info["portfolio_value"]
        prev_position = current_position

        done = terminated or truncated

    return portfolio_values, trade_returns, info["trade_count"]


def buy_and_hold(
    prices: np.ndarray,
    initial_balance: float = CONFIG["initial_balance"],
    window_size: int = CONFIG["window_size"],
) -> List[float]:
    """Simulate a simple buy-and-hold strategy.

    Parameters
    ----------
    prices : np.ndarray
        Close prices for the test period.
    initial_balance : float
        Starting cash.
    window_size : int
        Observation window size (to align with agent start).

    Returns
    -------
    list[float]
        Portfolio values over time.
    """
    start_price = prices[window_size]
    shares = initial_balance / start_price
    return [float(shares * p) for p in prices[window_size:]]


# ── Main ────────────────────────────────────────────────────────────────────

def main(ticker: str | None = None) -> None:
    """Evaluate both agents and print the comparison table."""
    ticker = ticker or CONFIG["tickers"][0]
    results_dir = Path(CONFIG["results_dir"])

    print(f"\n{'='*60}")
    print(f"  Evaluation on held-out TEST data — {ticker}")
    print(f"{'='*60}\n")

    # ── Load data ───────────────────────────────────────────────────────────
    try:
        train_df = load_split(ticker, "train")
    except FileNotFoundError:
        print("Data not found — running pipeline …")
        run_pipeline()
        train_df = load_split(ticker, "train")

    test_df = load_split(ticker, "test")
    cols = get_feature_columns()

    # Features — scaler fitted on train only
    train_features, _, test_features, scaler = prepare_features(
        train_df, test_df=test_df
    )

    # Aligned prices
    test_with_ind = add_indicators(test_df.copy())
    test_with_ind = test_with_ind.dropna(subset=cols)
    test_prices = test_with_ind["Close"].values

    # Environment
    test_env = TradingEnv(test_features, test_prices)

    # ── Load models ─────────────────────────────────────────────────────────
    dqn_path = results_dir / "dqn_best.zip"
    ppo_path = results_dir / "ppo_best.zip"

    if not dqn_path.exists():
        # Check EvalCallback best path
        alt = results_dir / "dqn_best" / "best_model.zip"
        dqn_path = alt if alt.exists() else dqn_path

    if not ppo_path.exists():
        alt = results_dir / "ppo_best" / "best_model.zip"
        ppo_path = alt if alt.exists() else ppo_path

    dqn_model = load_dqn(dqn_path, env=test_env)
    ppo_model = load_ppo(ppo_path, env=test_env)

    # ── Run agents ──────────────────────────────────────────────────────────
    dqn_vals, dqn_trades_r, dqn_n_trades = run_agent(dqn_model, test_env)
    ppo_vals, ppo_trades_r, ppo_n_trades = run_agent(ppo_model, test_env)
    bh_vals = buy_and_hold(test_prices)

    # ── Compute metrics ─────────────────────────────────────────────────────
    rf = CONFIG["risk_free_rate"]

    def _metrics(vals: List[float], trades_r: List[float], n_trades: int, name: str) -> Dict[str, Any]:
        arr = np.array(vals)
        daily_ret = np.diff(arr) / arr[:-1]
        tr = _total_return(arr)
        sr = _sharpe_ratio(daily_ret, rf)
        md = _max_drawdown(arr)
        wr = _win_rate(trades_r)
        cr = _calmar_ratio(tr, md)
        return {
            "strategy": name,
            "total_return_pct": round(tr, 2),
            "sharpe_ratio": round(sr, 4),
            "max_drawdown_pct": round(md, 2),
            "win_rate_pct": round(wr, 2),
            "total_trades": n_trades,
            "calmar_ratio": round(cr, 4),
        }

    dqn_m = _metrics(dqn_vals, dqn_trades_r, dqn_n_trades, "DQN")
    ppo_m = _metrics(ppo_vals, ppo_trades_r, ppo_n_trades, "PPO")
    bh_m = _metrics(bh_vals, [], 1, "Buy & Hold")

    metrics_all = [dqn_m, ppo_m, bh_m]

    # ── Print table ─────────────────────────────────────────────────────────
    headers = [
        "Strategy",
        "Total Return (%)",
        "Sharpe Ratio",
        "Max Drawdown (%)",
        "Win Rate (%)",
        "Total Trades",
        "Calmar Ratio",
    ]
    rows = [
        [
            m["strategy"],
            m["total_return_pct"],
            m["sharpe_ratio"],
            m["max_drawdown_pct"],
            m["win_rate_pct"],
            m["total_trades"],
            m["calmar_ratio"],
        ]
        for m in metrics_all
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # ── Save to JSON ────────────────────────────────────────────────────────
    out_path = results_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"\n-> Metrics saved to {out_path}")

    # Also save portfolio value series for backtesting plot
    np.savez(
        str(results_dir / "portfolio_values.npz"),
        dqn=np.array(dqn_vals),
        ppo=np.array(ppo_vals),
        bh=np.array(bh_vals),
        prices=test_prices,
    )
    print(f"-> Portfolio series saved to {results_dir / 'portfolio_values.npz'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL trading agents")
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()
    main(ticker=args.ticker)
