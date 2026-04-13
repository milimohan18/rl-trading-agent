"""
Feature engineering: compute technical indicators and normalise them.

Every indicator is computed from raw OHLCV columns.  A
``sklearn.preprocessing.RobustScaler`` is **fitted only on training data**
and applied to validation and test sets to avoid data leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import CONFIG


# ────────────────────────────────────────────────────────────────────────────
# Individual indicator functions
# ────────────────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    Parameters
    ----------
    series : pd.Series
        Close prices.
    period : int
        Look-back window.

    Returns
    -------
    pd.Series
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(
    series: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram.

    Parameters
    ----------
    series : pd.Series
        Close prices.
    fast, slow, signal : int
        EMA periods.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (macd_line, signal_line, histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(
    series: pd.Series,
    period: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (upper, middle, lower).

    Parameters
    ----------
    series : pd.Series
        Close prices.
    period : int
        Rolling-window length.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (upper, middle, lower)
    """
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + 2 * std
    lower = middle - 2 * std
    return upper, middle, lower


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Average True Range.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC columns.
    period : int
        Look-back window.

    Returns
    -------
    pd.Series
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (volume * direction).cumsum()


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators and append them to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Open, High, Low, Close, Volume`` columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame augmented with indicator columns.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # RSI
    df["RSI_14"] = _rsi(close, CONFIG["rsi_period"])

    # MACD
    macd_line, signal_line, histogram = _macd(
        close, CONFIG["macd_fast"], CONFIG["macd_slow"], CONFIG["macd_signal"]
    )
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = histogram

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = _bollinger_bands(close, CONFIG["bb_period"])
    df["BB_Upper"] = bb_upper
    df["BB_Middle"] = bb_middle
    df["BB_Lower"] = bb_lower

    # EMAs
    df["EMA_10"] = _ema(close, CONFIG["ema_short"])
    df["EMA_50"] = _ema(close, CONFIG["ema_long"])

    # ATR
    df["ATR_14"] = _atr(high, low, close, CONFIG["atr_period"])

    # OBV
    df["OBV"] = _obv(close, volume)

    # Daily return
    df["Daily_Return"] = close.pct_change()

    # Rolling volatility (10-day std of daily returns)
    df["Rolling_Vol_10"] = df["Daily_Return"].rolling(CONFIG["volatility_window"]).std()

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names used in the model.

    Returns
    -------
    list[str]
    """
    return [
        "Open", "High", "Low", "Close", "Volume",
        "RSI_14",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Middle", "BB_Lower",
        "EMA_10", "EMA_50",
        "ATR_14",
        "OBV",
        "Daily_Return",
        "Rolling_Vol_10",
    ]


def prepare_features(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], RobustScaler]:
    """Add indicators, drop NaNs from the warm-up period, and normalise.

    The ``RobustScaler`` is **fitted exclusively on training data** and then
    applied to validation and test sets — no data leakage.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training OHLCV data.
    val_df : pd.DataFrame, optional
        Validation OHLCV data.
    test_df : pd.DataFrame, optional
        Test OHLCV data.

    Returns
    -------
    tuple
        ``(train_features, val_features, test_features, scaler)``
        Each feature array has shape ``(timesteps, n_features)``.
    """
    cols = get_feature_columns()

    # ── Compute indicators ──────────────────────────────────────────────────
    train_df = add_indicators(train_df.copy())
    if val_df is not None:
        val_df = add_indicators(val_df.copy())
    if test_df is not None:
        test_df = add_indicators(test_df.copy())

    # ── Drop warm-up NaN rows ───────────────────────────────────────────────
    train_df = train_df.dropna(subset=cols)
    if val_df is not None:
        val_df = val_df.dropna(subset=cols)
    if test_df is not None:
        test_df = test_df.dropna(subset=cols)

    # ── Fit scaler on train only ────────────────────────────────────────────
    scaler = RobustScaler()
    train_arr = scaler.fit_transform(train_df[cols].values)

    val_arr: Optional[np.ndarray] = None
    test_arr: Optional[np.ndarray] = None
    if val_df is not None:
        val_arr = scaler.transform(val_df[cols].values)
    if test_df is not None:
        test_arr = scaler.transform(test_df[cols].values)

    return train_arr, val_arr, test_arr, scaler


def prepare_single_split(
    df: pd.DataFrame,
    scaler: RobustScaler,
) -> np.ndarray:
    """Compute indicators and normalise a single split using an existing scaler.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data for a single split.
    scaler : RobustScaler
        Previously fitted scaler.

    Returns
    -------
    np.ndarray
        Array of shape ``(timesteps, n_features)``.
    """
    cols = get_feature_columns()
    df = add_indicators(df.copy())
    df = df.dropna(subset=cols)
    return scaler.transform(df[cols].values)
