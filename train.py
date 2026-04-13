"""
Training script — train DQN and PPO agents on forex / crypto data.

Usage::

    python train.py                  # train both agents on BTC-USD
    python train.py --ticker EURUSD=X  # train on EUR/USD instead

All hyper-parameters are read from ``config.py``.  Training metrics are
logged via MLflow and the best checkpoints are persisted to ``results/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import mlflow
from stable_baselines3.common.callbacks import EvalCallback

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CONFIG
from utils.data_loader import load_split, run_pipeline
from utils.indicators import prepare_features, get_feature_columns, add_indicators
from env.trading_env import TradingEnv
from agents.dqn_agent import build_dqn, train_dqn
from agents.ppo_agent import build_ppo, train_ppo


def _set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _extract_prices(df, cols):
    """Get aligned close prices after indicator warm-up rows are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with indicators already added.
    cols : list[str]
        Feature column names used to determine which rows survive dropna.

    Returns
    -------
    np.ndarray
        Close prices aligned with feature rows.
    """
    df = add_indicators(df.copy())
    df = df.dropna(subset=cols)
    return df["Close"].values


def main(ticker: str | None = None) -> None:
    """Entry point for training.

    Parameters
    ----------
    ticker : str, optional
        Override the first ticker in the config.
    """
    seed: int = CONFIG["seed"]
    _set_seeds(seed)

    results_dir = Path(CONFIG["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    ticker = ticker or CONFIG["tickers"][0]
    print(f"\n{'='*60}")
    print(f"  RL Trading Agent — Training on {ticker}")
    print(f"{'='*60}\n")

    # ── 1. Load data ────────────────────────────────────────────────────────
    try:
        train_df = load_split(ticker, "train")
    except FileNotFoundError:
        print("Data splits not found — running data pipeline first …")
        run_pipeline()
        train_df = load_split(ticker, "train")

    val_df = load_split(ticker, "val")

    # ── 2. Feature engineering ──────────────────────────────────────────────
    train_features, val_features, _, scaler = prepare_features(
        train_df, val_df
    )
    cols = get_feature_columns()
    train_prices = _extract_prices(train_df, cols)
    val_prices = _extract_prices(val_df, cols)

    print(f"  Train features : {train_features.shape}")
    print(f"  Val features   : {val_features.shape}")

    # ── 3. Build environments ───────────────────────────────────────────────
    train_env = TradingEnv(train_features, train_prices)
    val_env = TradingEnv(val_features, val_prices)

    # ── 4. Callbacks ────────────────────────────────────────────────────────
    def _make_eval_cb(algo_name: str) -> EvalCallback:
        best_dir = str(results_dir / f"{algo_name}_best")
        os.makedirs(best_dir, exist_ok=True)
        return EvalCallback(
            val_env,
            best_model_save_path=best_dir,
            log_path=str(results_dir / f"{algo_name}_eval_logs"),
            eval_freq=CONFIG["eval_freq"],
            n_eval_episodes=1,
            deterministic=True,
            verbose=1,
        )

    # ── 5. Train DQN ────────────────────────────────────────────────────────
    print("\n-- Training DQN ------------------------------------------")
    mlflow.set_tracking_uri(f"file:///{results_dir.as_posix()}/mlruns")
    mlflow.set_experiment("rl_trading")

    with mlflow.start_run(run_name="DQN"):
        mlflow.log_params({f"dqn_{k}": v for k, v in CONFIG["dqn"].items()})
        mlflow.log_param("total_timesteps", CONFIG["total_timesteps"])
        mlflow.log_param("seed", seed)
        mlflow.log_param("ticker", ticker)

        dqn_model = build_dqn(train_env, seed=seed)
        dqn_cb = _make_eval_cb("dqn")
        train_dqn(dqn_model, callbacks=[dqn_cb])

        dqn_save_path = str(results_dir / "dqn_best")
        dqn_model.save(dqn_save_path)
        mlflow.log_artifact(dqn_save_path + ".zip")

        # Log validation Sharpe
        obs, info = val_env.reset(seed=seed)
        done = False
        while not done:
            action, _ = dqn_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = val_env.step(int(action))
            done = terminated or truncated
        mlflow.log_metric("val_sharpe_dqn", info["sharpe_ratio"])
        print(f"  DQN validation Sharpe: {info['sharpe_ratio']:.4f}")

    # ── 6. Train PPO ────────────────────────────────────────────────────────
    print("\n-- Training PPO ------------------------------------------")
    # Recreate envs so episode counters are fresh
    train_env = TradingEnv(train_features, train_prices)
    val_env = TradingEnv(val_features, val_prices)

    with mlflow.start_run(run_name="PPO"):
        mlflow.log_params({f"ppo_{k}": str(v) for k, v in CONFIG["ppo"].items()})
        mlflow.log_param("total_timesteps", CONFIG["total_timesteps"])
        mlflow.log_param("seed", seed)
        mlflow.log_param("ticker", ticker)

        ppo_model = build_ppo(train_env, seed=seed)
        ppo_cb = _make_eval_cb("ppo")
        train_ppo(ppo_model, callbacks=[ppo_cb])

        ppo_save_path = str(results_dir / "ppo_best")
        ppo_model.save(ppo_save_path)
        mlflow.log_artifact(ppo_save_path + ".zip")

        obs, info = val_env.reset(seed=seed)
        done = False
        while not done:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = val_env.step(int(action))
            done = terminated or truncated
        mlflow.log_metric("val_sharpe_ppo", info["sharpe_ratio"])
        print(f"  PPO validation Sharpe: {info['sharpe_ratio']:.4f}")

    print("\n-> Training complete.  Models saved to", results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL trading agents")
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Yahoo Finance ticker to train on (default: first in config)",
    )
    args = parser.parse_args()
    main(ticker=args.ticker)
