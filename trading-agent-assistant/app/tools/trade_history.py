"""
Trade history tool — fetches executed trades from the mock trade log.

The tool returns structured data (list of trade dicts) so the LLM can
compose its own natural-language answer.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Path to the trade data file (relative to this module)
_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "trades.json"

# Cached trade data — loaded once on first call
_trades_cache: Optional[List[Dict[str, Any]]] = None


def _load_trades() -> List[Dict[str, Any]]:
    """Load and cache the trade data from disk."""
    global _trades_cache
    if _trades_cache is None:
        with open(_DATA_FILE, "r") as f:
            _trades_cache = json.load(f)
    return _trades_cache


def _parse_date(date_str: str) -> date:
    """Parse an ISO date string (YYYY-MM-DD) into a date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def get_trade_history(
    start_date: str,
    end_date: str,
    symbol: Optional[str] = None,
    sort_by: Optional[str] = None,
    max_results: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch executed trades for a date range, optionally filtered by symbol.

    Parameters
    ----------
    start_date : str
        ISO date string, e.g. ``"2025-08-01"``.
    end_date : str
        ISO date string, e.g. ``"2025-08-31"``.
    symbol : str, optional
        Ticker symbol filter, e.g. ``"BTC-USD"``.
    sort_by : str, optional
        Sort results. Options: ``"pnl"`` (highest first), ``"pnl_asc"``
        (lowest first), ``"date"`` (chronological), ``"date_desc"`` (newest first).
    max_results : int, optional
        Maximum number of trade records to return (default: all).

    Returns
    -------
    list[dict]
        List of trade records. Each trade contains:
        - trade_id, symbol, action, position_type
        - entry_date, entry_price, exit_date, exit_price
        - pnl, pnl_pct
        - indicators_at_entry, indicators_at_exit
    """
    trades = _load_trades()

    try:
        start = _parse_date(start_date)
        end = _parse_date(end_date)
    except ValueError as e:
        return [{"error": f"Invalid date format: {e}. Use YYYY-MM-DD."}]

    if start > end:
        return [{"error": "start_date must be before or equal to end_date."}]

    results: List[Dict[str, Any]] = []
    for trade in trades:
        entry = _parse_date(trade["entry_date"])
        exit_ = _parse_date(trade["exit_date"])

        # Include trade if it overlaps with the requested date range
        if exit_ < start or entry > end:
            continue

        # Apply symbol filter if provided
        if symbol and trade["symbol"].upper() != symbol.upper():
            continue

        results.append(trade)

    # Apply sorting
    if sort_by == "pnl":
        results.sort(key=lambda t: t["pnl"], reverse=True)
    elif sort_by == "pnl_asc":
        results.sort(key=lambda t: t["pnl"])
    elif sort_by == "date_desc":
        results.sort(key=lambda t: t["entry_date"], reverse=True)
    # default "date" / None: trades.json is already chronological

    # Apply result cap
    if max_results is not None and max_results > 0:
        results = results[:max_results]

    return results


# ── Tool schema for the LLM (Anthropic format) ─────────────────────────────

TOOL_SCHEMA = {
    "name": "get_trade_history",
    "description": (
        "Fetch executed trades and their outcomes for a date range, "
        "optionally filtered by symbol. Returns structured data including "
        "entry/exit prices, P&L, and technical indicator values at the "
        "time of each trade. Use sort_by='pnl' with max_results=1 to get "
        "the most profitable trade; use sort_by='pnl_asc' with max_results=1 "
        "to get the worst trade."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "ISO date, e.g. 2025-08-01",
            },
            "end_date": {
                "type": "string",
                "description": "ISO date, e.g. 2025-08-31",
            },
            "symbol": {
                "type": "string",
                "description": "Optional ticker filter, e.g. BTC-USD",
            },
            "sort_by": {
                "type": "string",
                "enum": ["pnl", "pnl_asc", "date", "date_desc"],
                "description": (
                    "Sort order: 'pnl' = highest P&L first, 'pnl_asc' = lowest first, "
                    "'date' = chronological (default), 'date_desc' = newest first."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of records to return. Use 1 to get a single top result.",
            },
        },
        "required": ["start_date", "end_date"],
    },
}
