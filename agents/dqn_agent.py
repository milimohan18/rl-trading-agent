"""
DQN agent wrapper around ``stable_baselines3.DQN``.

Provides helper functions to build, train, and load a DQN agent
with the hyper-parameters defined in :pydata:`config.CONFIG`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from config import CONFIG
from env.trading_env import TradingEnv


def build_dqn(
    env: TradingEnv,
    seed: int = CONFIG["seed"],
    tensorboard_log: Optional[str] = None,
) -> DQN:
    """Construct a DQN model from the global config.

    Parameters
    ----------
    env : TradingEnv
        Training environment.
    seed : int
        Random seed.
    tensorboard_log : str, optional
        Path for TensorBoard logs.

    Returns
    -------
    DQN
    """
    cfg = CONFIG["dqn"]
    model = DQN(
        policy=cfg["policy"],
        env=env,
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        gamma=cfg["gamma"],
        exploration_fraction=cfg["exploration_fraction"],
        exploration_initial_eps=cfg["exploration_initial_eps"],
        exploration_final_eps=cfg["exploration_final_eps"],
        policy_kwargs={"net_arch": cfg["net_arch"]},
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )
    return model


def train_dqn(
    model: DQN,
    total_timesteps: int = CONFIG["total_timesteps"],
    callbacks: Optional[list[BaseCallback]] = None,
) -> DQN:
    """Train the DQN model.

    Parameters
    ----------
    model : DQN
        DQN model instance.
    total_timesteps : int
        Total training timesteps.
    callbacks : list[BaseCallback], optional
        Optional list of SB3 callbacks.

    Returns
    -------
    DQN
        The trained model.
    """
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )
    return model


def load_dqn(path: str | Path, env: Optional[TradingEnv] = None) -> DQN:
    """Load a saved DQN model from disk.

    Parameters
    ----------
    path : str or Path
        Path to the ``.zip`` model file.
    env : TradingEnv, optional
        Environment to attach.

    Returns
    -------
    DQN
    """
    return DQN.load(str(path), env=env)
