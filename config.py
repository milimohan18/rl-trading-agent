"""
Central configuration for the RL Trading Agent project.
All tuneable hyper-parameters, paths, and constants live here so that
nothing is hard-coded elsewhere in the codebase.
"""

from pathlib import Path
from typing import Any, Dict

# ── Project Root ────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ── Single config dictionary used by every module ───────────────────────────
CONFIG: Dict[str, Any] = {
    # ── Paths ───────────────────────────────────────────────────────────────
    "data_dir": str(PROJECT_ROOT / "data"),
    "results_dir": str(PROJECT_ROOT / "results"),

    # ── Data download ───────────────────────────────────────────────────────
    "tickers": ["BTC-USD", "EURUSD=X"],
    "period": "5y",
    "interval": "1d",

    # ── Train / Val / Test split ratios (must sum to 1.0) ───────────────────
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # ── Technical-indicator parameters ──────────────────────────────────────
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "ema_short": 10,
    "ema_long": 50,
    "atr_period": 14,
    "volatility_window": 10,

    # ── Environment ─────────────────────────────────────────────────────────
    "window_size": 20,
    "initial_balance": 10_000.0,
    "transaction_cost": 0.001,       # 0.1 %

    # ── DQN hyper-parameters ────────────────────────────────────────────────
    "dqn": {
        "policy": "MlpPolicy",
        "net_arch": [256, 256],
        "learning_rate": 1e-4,
        "buffer_size": 50_000,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.30,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },

    # ── PPO hyper-parameters ────────────────────────────────────────────────
    "ppo": {
        "policy": "MlpPolicy",
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    },

    # ── Training ────────────────────────────────────────────────────────────
    "total_timesteps": 200_000,
    "eval_freq": 10_000,
    "seed": 42,

    # ── Evaluation ──────────────────────────────────────────────────────────
    "risk_free_rate": 0.02,

    # ── Plotting ────────────────────────────────────────────────────────────
    "plot_dpi": 150,
}
