"""
PPO agent wrapper around ``stable_baselines3.PPO``.

Provides helper functions to build, train, and load a PPO agent
with the hyper-parameters defined in :pydata:`config.CONFIG`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from config import CONFIG
from env.trading_env import TradingEnv


def build_ppo(
    env: TradingEnv,
    seed: int = CONFIG["seed"],
    tensorboard_log: Optional[str] = None,
) -> PPO:
    """Construct a PPO model from the global config.

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
    PPO
    """
    cfg = CONFIG["ppo"]
    model = PPO(
        policy=cfg["policy"],
        env=env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        policy_kwargs={"net_arch": cfg["net_arch"]},
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )
    return model


def train_ppo(
    model: PPO,
    total_timesteps: int = CONFIG["total_timesteps"],
    callbacks: Optional[list[BaseCallback]] = None,
) -> PPO:
    """Train the PPO model.

    Parameters
    ----------
    model : PPO
        PPO model instance.
    total_timesteps : int
        Total training timesteps.
    callbacks : list[BaseCallback], optional
        Optional list of SB3 callbacks.

    Returns
    -------
    PPO
        The trained model.
    """
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )
    return model


def load_ppo(path: str | Path, env: Optional[TradingEnv] = None) -> PPO:
    """Load a saved PPO model from disk.

    Parameters
    ----------
    path : str or Path
        Path to the ``.zip`` model file.
    env : TradingEnv, optional
        Environment to attach.

    Returns
    -------
    PPO
    """
    return PPO.load(str(path), env=env)
