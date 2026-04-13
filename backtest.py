"""
Backtesting visualisation — three-panel chart comparing strategies.

Usage::

    python backtest.py

Reads the portfolio-value arrays written by ``evaluate.py`` and produces
a publication-quality figure saved to ``results/backtest_plot.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CONFIG
from utils.data_loader import load_split
from utils.indicators import add_indicators, get_feature_columns
from env.trading_env import TradingEnv
from agents.dqn_agent import load_dqn
from agents.ppo_agent import load_ppo


def _collect_actions(
    model: Any,
    env: TradingEnv,
) -> Tuple[List[int], List[float]]:
    """Run a model on an env and collect the action at each step.

    Returns
    -------
    tuple[list[int], list[float]]
        ``(actions, portfolio_values)``
    """
    obs, info = env.reset(seed=CONFIG["seed"])
    actions: List[int] = []
    values: List[float] = [info["portfolio_value"]]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        values.append(info["portfolio_value"])
        done = terminated or truncated
    return actions, values


def _drawdown(values: np.ndarray) -> np.ndarray:
    """Compute the drawdown series (fraction, always ≤ 0)."""
    peak = np.maximum.accumulate(values)
    return (values - peak) / peak


def main(ticker: str | None = None) -> None:
    """Generate the three-panel backtest chart."""
    ticker = ticker or CONFIG["tickers"][0]
    results_dir = Path(CONFIG["results_dir"])

    # ── Load test data & features ───────────────────────────────────────────
    from utils.indicators import prepare_features
    train_df = load_split(ticker, "train")
    test_df = load_split(ticker, "test")
    cols = get_feature_columns()

    train_features, _, test_features, scaler = prepare_features(
        train_df, test_df=test_df
    )

    test_with_ind = add_indicators(test_df.copy())
    test_with_ind = test_with_ind.dropna(subset=cols)
    test_prices = test_with_ind["Close"].values
    test_dates = test_with_ind.index

    # Environments
    test_env = TradingEnv(test_features, test_prices)

    # Load models
    dqn_path = results_dir / "dqn_best.zip"
    ppo_path = results_dir / "ppo_best.zip"
    if not dqn_path.exists():
        alt = results_dir / "dqn_best" / "best_model.zip"
        dqn_path = alt if alt.exists() else dqn_path
    if not ppo_path.exists():
        alt = results_dir / "ppo_best" / "best_model.zip"
        ppo_path = alt if alt.exists() else ppo_path

    dqn_model = load_dqn(dqn_path, env=test_env)
    ppo_model = load_ppo(ppo_path, env=test_env)

    # Rollouts
    dqn_actions, dqn_vals = _collect_actions(dqn_model, test_env)
    ppo_actions, ppo_vals = _collect_actions(ppo_model, test_env)

    # Buy-and-hold
    window = CONFIG["window_size"]
    start_price = test_prices[window]
    shares = CONFIG["initial_balance"] / start_price
    bh_vals = [float(shares * p) for p in test_prices[window:]]

    # Align dates (window offset + 1 for initial value)
    plot_dates = test_dates[window:]
    n = min(len(plot_dates), len(dqn_vals), len(ppo_vals), len(bh_vals))
    plot_dates = plot_dates[:n]
    dqn_vals = np.array(dqn_vals[:n])
    ppo_vals = np.array(ppo_vals[:n])
    bh_vals_arr = np.array(bh_vals[:n])

    # Align actions — they start after the window
    action_dates = test_dates[window + 1 : window + 1 + len(dqn_actions)]

    # ── Create figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        f"Backtest Results — {ticker}",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )

    colours = {"DQN": "#2196F3", "PPO": "#FF9800", "B&H": "#4CAF50"}

    # ── Panel 1: Portfolio value ────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(plot_dates, dqn_vals, label="DQN", color=colours["DQN"], linewidth=1.4)
    ax1.plot(plot_dates, ppo_vals, label="PPO", color=colours["PPO"], linewidth=1.4)
    ax1.plot(plot_dates, bh_vals_arr, label="Buy & Hold", color=colours["B&H"], linewidth=1.4, linestyle="--")
    ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax1.set_title("Portfolio Value Over Time", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Price with Buy/Sell markers ────────────────────────────────
    ax2 = axes[1]
    price_plot = test_prices[window:][:n]
    ax2.plot(plot_dates, price_plot, color="#757575", linewidth=1.0, label="Price")

    # PPO markers (shown as example — can swap to DQN)
    for i, act in enumerate(ppo_actions[:n - 1]):
        dt = action_dates[i] if i < len(action_dates) else None
        if dt is None:
            continue
        if act == 1:  # Buy
            ax2.annotate(
                "▲",
                xy=(dt, test_prices[window + 1 + i]),
                fontsize=8,
                color="#4CAF50",
                ha="center",
            )
        elif act == 2:  # Sell
            ax2.annotate(
                "▼",
                xy=(dt, test_prices[window + 1 + i]),
                fontsize=8,
                color="#F44336",
                ha="center",
            )

    ax2.set_ylabel("Price", fontsize=12)
    ax2.set_title("Asset Price with PPO Buy (▲) / Sell (▼) Signals", fontsize=14)
    ax2.grid(alpha=0.3)

    # ── Panel 3: Drawdown ───────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(
        plot_dates,
        _drawdown(dqn_vals) * 100,
        0,
        alpha=0.35,
        color=colours["DQN"],
        label="DQN",
    )
    ax3.fill_between(
        plot_dates,
        _drawdown(ppo_vals) * 100,
        0,
        alpha=0.35,
        color=colours["PPO"],
        label="PPO",
    )
    ax3.fill_between(
        plot_dates,
        _drawdown(bh_vals_arr) * 100,
        0,
        alpha=0.25,
        color=colours["B&H"],
        label="Buy & Hold",
    )
    ax3.set_ylabel("Drawdown (%)", fontsize=12)
    ax3.set_xlabel("Date", fontsize=12)
    ax3.set_title("Drawdown Curve", fontsize=14)
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = results_dir / "backtest_plot.png"
    fig.savefig(out_path, dpi=CONFIG["plot_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"-> Backtest chart saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate backtest visualisation")
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()
    main(ticker=args.ticker)
