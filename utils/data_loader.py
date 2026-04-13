"""
Data-loading pipeline: download, clean, split, and persist OHLCV data.

Downloads real forex / crypto price data via ``yfinance``, forward-fills
missing values, splits into train / validation / test sets while
preserving chronological order, and writes the splits to CSV.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config import CONFIG


def download_ohlcv(ticker: str) -> pd.DataFrame:
    """Download daily OHLCV data for *ticker* using yfinance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. ``"BTC-USD"``).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``Date`` with columns
        ``Open, High, Low, Close, Volume``.
    """
    df: pd.DataFrame = yf.download(
        ticker,
        period=CONFIG["period"],
        interval=CONFIG["interval"],
        auto_adjust=True,
        progress=False,
    )
    # yfinance may return multi-level columns when a single ticker is passed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with forward-fill, then back-fill residuals.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with no NaN values.
    """
    df = df.ffill().bfill()
    # Drop any rows that are still NaN (shouldn't happen after bfill)
    df = df.dropna()
    return df


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically into train / val / test.

    No shuffling is performed — time order is preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)``
    """
    n: int = len(df)
    train_end: int = int(n * CONFIG["train_ratio"])
    val_end: int = int(n * (CONFIG["train_ratio"] + CONFIG["val_ratio"]))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def save_splits(
    ticker: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, str]:
    """Persist train / val / test DataFrames to CSV in ``data/``.

    Parameters
    ----------
    ticker : str
        Ticker symbol, used to name files.
    train_df, val_df, test_df : pd.DataFrame
        The three chronological splits.

    Returns
    -------
    dict[str, str]
        Mapping of split name → absolute file path.
    """
    data_dir = Path(CONFIG["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    safe_ticker = ticker.replace("=", "").replace("/", "_")
    paths: Dict[str, str] = {}
    for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        fp = data_dir / f"{safe_ticker}_{name}.csv"
        frame.to_csv(fp)
        paths[name] = str(fp)
        print(f"  [{name:>5}] {len(frame):>5} rows -> {fp}")

    return paths


def load_split(ticker: str, split: str) -> pd.DataFrame:
    """Load a previously saved CSV split.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.

    Returns
    -------
    pd.DataFrame
        DataFrame loaded from the CSV file.
    """
    safe_ticker = ticker.replace("=", "").replace("/", "_")
    fp = Path(CONFIG["data_dir"]) / f"{safe_ticker}_{split}.csv"
    return pd.read_csv(fp, index_col="Date", parse_dates=True)


def run_pipeline() -> None:
    """Execute the full data pipeline for every ticker in the config."""
    for ticker in CONFIG["tickers"]:
        print(f"\n{'='*60}")
        print(f"Processing {ticker}")
        print(f"{'='*60}")

        raw = download_ohlcv(ticker)
        print(f"  Downloaded {len(raw)} rows")

        clean = clean_data(raw)
        print(f"  After cleaning: {len(clean)} rows")

        train_df, val_df, test_df = split_data(clean)
        save_splits(ticker, train_df, val_df, test_df)


if __name__ == "__main__":
    run_pipeline()
