"""
Custom Gymnasium trading environment.

The agent observes a sliding window of normalised technical-indicator
features and issues Buy / Hold / Sell actions.  Portfolio value is
tracked with a simple long / flat / short model and a configurable
transaction cost.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import CONFIG


class TradingEnv(gym.Env):
    """A discrete-action trading environment compatible with Gymnasium.

    Attributes
    ----------
    metadata : dict
        Gymnasium rendering metadata.
    """

    metadata: dict = {"render_modes": ["human"]}

    # Action constants
    HOLD: int = 0
    BUY: int = 1
    SELL: int = 2

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        window_size: int = CONFIG["window_size"],
        initial_balance: float = CONFIG["initial_balance"],
        transaction_cost: float = CONFIG["transaction_cost"],
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialise the trading environment.

        Parameters
        ----------
        features : np.ndarray
            Normalised feature array of shape ``(T, n_features)``.
        prices : np.ndarray
            Raw close prices aligned with *features*, shape ``(T,)``.
        window_size : int
            Number of past timesteps in each observation.
        initial_balance : float
            Starting cash balance.
        transaction_cost : float
            Proportional cost applied per trade (e.g. 0.001 = 0.1 %).
        render_mode : str, optional
            ``"human"`` for console output.
        """
        super().__init__()

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.render_mode = render_mode

        self.n_features: int = features.shape[1]

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_features),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # Hold / Buy / Sell

        # Internal state (set properly in reset)
        self._current_step: int = self.window_size
        self._position: int = 0     # -1=short, 0=flat, 1=long
        self._entry_price: float = 0.0
        self._cash: float = initial_balance
        self._portfolio_value: float = initial_balance
        self._trade_count: int = 0
        self._returns: list[float] = []

    # ── Gym interface ───────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the beginning of the episode.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Unused; required by the Gymnasium API.

        Returns
        -------
        tuple[np.ndarray, dict]
            ``(observation, info)``
        """
        super().reset(seed=seed)

        self._current_step = self.window_size
        self._position = 0
        self._entry_price = 0.0
        self._cash = self.initial_balance
        self._portfolio_value = self.initial_balance
        self._trade_count = 0
        self._returns = []

        return self._get_observation(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep.

        Parameters
        ----------
        action : int
            0 = Hold, 1 = Buy, 2 = Sell.

        Returns
        -------
        tuple
            ``(observation, reward, terminated, truncated, info)``
        """
        prev_portfolio = self._portfolio_value
        current_price = self.prices[self._current_step]

        # ── Execute action ──────────────────────────────────────────────────
        self._execute_action(action, current_price)

        # ── Update portfolio value ──────────────────────────────────────────
        self._update_portfolio(current_price)

        # ── Reward: log return of portfolio ─────────────────────────────────
        if prev_portfolio > 0:
            step_return = np.log(self._portfolio_value / prev_portfolio)
        else:
            step_return = 0.0
        self._returns.append(step_return)
        reward: float = float(step_return)

        # ── Advance ─────────────────────────────────────────────────────────
        self._current_step += 1
        terminated: bool = self._current_step >= len(self.prices)
        truncated: bool = False

        obs = (
            self._get_observation()
            if not terminated
            else np.zeros(
                (self.window_size, self.n_features), dtype=np.float32
            )
        )

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self._get_info()

    def render(self) -> None:
        """Print a one-line status to the console."""
        action_map = {0: "HOLD", 1: "BUY ", 2: "SELL"}
        step = self._current_step
        price = self.prices[min(step, len(self.prices) - 1)]
        print(
            f"Step {step:>5d} | "
            f"Price {price:>12,.2f} | "
            f"Portfolio {self._portfolio_value:>12,.2f} | "
            f"Position {self._position:>2d} | "
            f"Trades {self._trade_count:>4d}"
        )

    # ── Private helpers ─────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Return the sliding-window observation."""
        start = self._current_step - self.window_size
        end = self._current_step
        return self.features[start:end].copy()

    def _get_info(self) -> Dict[str, Any]:
        """Build the info dict required by the spec."""
        return {
            "portfolio_value": self._portfolio_value,
            "position": self._position,
            "trade_count": self._trade_count,
            "sharpe_ratio": self._compute_sharpe(),
        }

    def _execute_action(self, action: int, price: float) -> None:
        """Translate a discrete action into a position change."""
        if action == self.BUY and self._position <= 0:
            # Close short if any, then go long
            if self._position == -1:
                self._close_position(price)
            self._open_position(1, price)
        elif action == self.SELL and self._position >= 0:
            # Close long if any, then go short
            if self._position == 1:
                self._close_position(price)
            self._open_position(-1, price)
        # else: HOLD — do nothing

    def _open_position(self, direction: int, price: float) -> None:
        """Open a new position (long=+1, short=-1)."""
        cost = self._cash * self.transaction_cost
        self._cash -= cost
        self._position = direction
        self._entry_price = price
        self._trade_count += 1

    def _close_position(self, price: float) -> None:
        """Close the current position and realise PnL."""
        if self._position == 0:
            return
        pnl_pct = (price - self._entry_price) / self._entry_price
        if self._position == -1:
            pnl_pct = -pnl_pct
        self._cash *= 1.0 + pnl_pct
        cost = self._cash * self.transaction_cost
        self._cash -= cost
        self._position = 0
        self._entry_price = 0.0
        self._trade_count += 1

    def _update_portfolio(self, price: float) -> None:
        """Recompute portfolio value including unrealised PnL."""
        if self._position == 0:
            self._portfolio_value = self._cash
        else:
            pnl_pct = (price - self._entry_price) / self._entry_price
            if self._position == -1:
                pnl_pct = -pnl_pct
            self._portfolio_value = self._cash * (1.0 + pnl_pct)

    def _compute_sharpe(self) -> float:
        """Annualised Sharpe ratio from episode returns so far."""
        if len(self._returns) < 2:
            return 0.0
        arr = np.array(self._returns)
        mean_r = arr.mean()
        std_r = arr.std()
        if std_r < 1e-10:
            return 0.0
        daily_rf = CONFIG["risk_free_rate"] / 252.0
        sharpe = (mean_r - daily_rf) / std_r
        return float(sharpe * np.sqrt(252))
