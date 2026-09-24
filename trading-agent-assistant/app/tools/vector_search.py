"""
RAG tool — semantic search over market indicator explanations and notes.

Phase 3 extension: uses ChromaDB as a local vector store to provide
context about technical indicators and market concepts.

To enable this tool, install chromadb:
    pip install chromadb>=0.5.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import chromadb
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Keep the embedding model inside the project instead of ~/.cache so a copy
# fetched at build time (python -m app.tools.vector_search) survives into
# the runtime on hosts like Render, where only the project dir is kept.
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / ".chroma_models"


# ── Sample documents ────────────────────────────────────────────────────────

INDICATOR_DOCS = [
    {
        "id": "rsi_overview",
        "text": (
            "RSI (Relative Strength Index) is a momentum oscillator that "
            "measures the speed and magnitude of price changes on a scale "
            "of 0 to 100. An RSI above 70 typically indicates overbought "
            "conditions, while an RSI below 30 indicates oversold conditions. "
            "The default period is 14 days."
        ),
    },
    {
        "id": "rsi_trading",
        "text": (
            "RSI divergence occurs when the price makes a new high/low but "
            "RSI does not confirm it. Bullish divergence (price lower low, "
            "RSI higher low) can signal a potential reversal upward. Bearish "
            "divergence (price higher high, RSI lower high) can signal a "
            "potential reversal downward."
        ),
    },
    {
        "id": "macd_overview",
        "text": (
            "MACD (Moving Average Convergence Divergence) is a trend-following "
            "momentum indicator. It consists of the MACD line (12-period EMA "
            "minus 26-period EMA), the signal line (9-period EMA of MACD), "
            "and the histogram (MACD minus signal). A bullish signal occurs "
            "when the MACD line crosses above the signal line."
        ),
    },
    {
        "id": "macd_histogram",
        "text": (
            "The MACD histogram represents the difference between the MACD "
            "line and the signal line. When the histogram is positive and "
            "growing, bullish momentum is increasing. When it's negative and "
            "shrinking, bearish momentum may be weakening. Zero-line crossovers "
            "of the histogram often precede signal-line crossovers."
        ),
    },
    {
        "id": "bollinger_bands",
        "text": (
            "Bollinger Bands consist of a middle band (20-period SMA) and "
            "upper/lower bands at 2 standard deviations. When price touches "
            "the upper band, the asset may be overbought. When it touches the "
            "lower band, it may be oversold. Band squeeze (narrowing) often "
            "precedes a significant price move."
        ),
    },
    {
        "id": "ema_crossover",
        "text": (
            "EMA (Exponential Moving Average) crossover is a common trading "
            "signal. When the short-term EMA (e.g., 10-day) crosses above "
            "the long-term EMA (e.g., 50-day), it's called a golden cross "
            "and is considered bullish. The opposite (death cross) is bearish. "
            "EMAs react faster to recent price changes than SMAs."
        ),
    },
    {
        "id": "atr_volatility",
        "text": (
            "ATR (Average True Range) measures market volatility. It's the "
            "average of true ranges over a period (typically 14 days). Higher "
            "ATR means higher volatility. Traders use ATR for position sizing "
            "and setting stop-loss levels — a common approach is to set stops "
            "at 1.5x or 2x the ATR from the entry price."
        ),
    },
    {
        "id": "obv_volume",
        "text": (
            "OBV (On-Balance Volume) is a cumulative volume indicator. It "
            "adds volume on up days and subtracts volume on down days. Rising "
            "OBV confirms an uptrend, while falling OBV confirms a downtrend. "
            "Divergence between OBV and price can signal a potential reversal."
        ),
    },
    {
        "id": "dqn_agent",
        "text": (
            "The DQN (Deep Q-Network) agent uses a neural network to "
            "approximate the Q-value function. It was trained with a "
            "256x256 MLP architecture, learning rate of 1e-4, and epsilon-"
            "greedy exploration (starting at 1.0, decaying to 0.05 over 30% "
            "of training). The replay buffer holds 50,000 transitions."
        ),
    },
    {
        "id": "ppo_agent",
        "text": (
            "The PPO (Proximal Policy Optimization) agent uses a separate "
            "actor-critic architecture with 256x256 MLPs. It was trained "
            "with learning rate 3e-4, GAE lambda 0.95, and 10 epochs per "
            "update. PPO tends to be more stable but sometimes makes fewer "
            "trades than DQN."
        ),
    },
    {
        "id": "trading_env",
        "text": (
            "The trading environment supports three actions: HOLD (0), "
            "BUY (1), and SELL (2). It tracks positions as long (+1), "
            "flat (0), or short (-1). Transaction costs are 0.1% per trade. "
            "The observation window is 20 days of normalized indicator values."
        ),
    },
    {
        "id": "sharpe_ratio",
        "text": (
            "The Sharpe Ratio measures risk-adjusted return. It's calculated "
            "as (mean return - risk-free rate) / standard deviation of returns, "
            "annualized by multiplying by sqrt(252). A Sharpe above 1.0 is "
            "considered good, above 2.0 is very good. The DQN agent achieved "
            "a Sharpe of 1.69 on the test set."
        ),
    },
    {
        "id": "max_drawdown",
        "text": (
            "Maximum drawdown is the largest peak-to-trough decline in "
            "portfolio value. It measures the worst-case scenario for an "
            "investor. The DQN agent had a max drawdown of -14.16%, "
            "significantly better than the buy-and-hold drawdown of -49.74%."
        ),
    },
    {
        "id": "btc_market_context",
        "text": (
            "BTC-USD experienced significant volatility during the test period "
            "(July 2025 – April 2026). The price ranged from approximately "
            "$68,000 to $120,000. The overall trend during this period was "
            "bearish, with buy-and-hold returning -36.73%. The DQN agent's "
            "ability to short and time entries/exits allowed it to profit."
        ),
    },
    {
        "id": "win_rate",
        "text": (
            "Win rate is the percentage of trades that were profitable. "
            "The DQN agent achieved a 71.43% win rate over 29 trades. "
            "While a high win rate is desirable, it must be considered "
            "alongside the average win/loss size — a lower win rate can "
            "still be profitable if winners are larger than losers."
        ),
    },
]


# ── ChromaDB setup ──────────────────────────────────────────────────────────

_collection = None


def _init_chroma():
    """Initialize ChromaDB collection with embeddings for the 15 indicator snippets."""
    global _collection
    if not CHROMADB_AVAILABLE:
        return None

    try:
        embedding_fn = ONNXMiniLM_L6_V2()
        embedding_fn.DOWNLOAD_PATH = _MODEL_DIR / ONNXMiniLM_L6_V2.MODEL_NAME

        # Ephemeral in-memory client: the collection is rebuilt from
        # INDICATOR_DOCS on every startup, so there is no on-disk state to lose.
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name="market_context",
            metadata={"description": "Indicator explanations and market notes"},
            embedding_function=embedding_fn,
        )

        if collection.count() == 0:
            collection.add(
                ids=[doc["id"] for doc in INDICATOR_DOCS],
                documents=[doc["text"] for doc in INDICATOR_DOCS],
            )
        _collection = collection
        return _collection
    except Exception as e:
        print(f"Warning: Failed to initialize ChromaDB collection: {e}")
        return None


def _get_collection():
    """Return the initialized ChromaDB collection, initializing if needed."""
    global _collection
    if _collection is not None:
        return _collection
    return _init_chroma()


# Initialize on import when ChromaDB is available
if CHROMADB_AVAILABLE:
    _init_chroma()


def search_market_context(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """Semantic search over stored indicator explanations and market notes.

    Parameters
    ----------
    query : str
        Natural-language search query.
    n_results : int
        Number of results to return (default 3).

    Returns
    -------
    list[dict]
        List of ``{"id": ..., "text": ..., "distance": ...}`` dicts.
    """
    collection = _get_collection()

    if collection is None:
        # ChromaDB not installed — fall back to simple keyword matching
        query_lower = query.lower()
        scored = []
        for doc in INDICATOR_DOCS:
            words = query_lower.split()
            score = sum(1 for w in words if w in doc["text"].lower())
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": doc["id"], "text": doc["text"], "distance": 1.0 / (score + 1)}
            for score, doc in scored[:n_results]
        ]

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    output = []
    if results and results["ids"]:
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })

    return output


# ── Tool schema for the LLM (Anthropic format) ─────────────────────────────

TOOL_SCHEMA = {
    "name": "search_market_context",
    "description": (
        "Semantic search over stored indicator explanations, agent "
        "configuration details, and market context notes. Use this to look up "
        "what indicators mean, how the agents work, or market background."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query about indicators, agents, or market context.",
            },
        },
        "required": ["query"],
    },
}


if __name__ == "__main__":
    # Build step: importing this module downloads the embedding model and
    # indexes the docs; fail the build if that didn't work.
    if _get_collection() is None:
        raise SystemExit("ChromaDB collection failed to initialize")
    print(f"ChromaDB ready: {_get_collection().count()} docs, model in {_MODEL_DIR}")
