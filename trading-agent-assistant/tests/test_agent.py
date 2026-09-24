"""
Tests for the Trading Agent Assistant.

All tests mock the LLM call so they run without an API key.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.tools.trade_history import get_trade_history
from app.tools.vector_search import search_market_context


# ── Trade history tool tests ────────────────────────────────────────────────


class TestGetTradeHistory:
    """Tests for the get_trade_history tool."""

    def test_returns_trades_in_date_range(self):
        """Trades within the requested date range are returned."""
        result = get_trade_history("2025-08-01", "2025-08-31")
        assert isinstance(result, list)
        assert len(result) > 0
        for trade in result:
            assert "trade_id" in trade
            assert "entry_date" in trade
            assert "exit_date" in trade
            assert "pnl" in trade
            assert "indicators_at_entry" in trade

    def test_empty_result_for_out_of_range(self):
        """No trades returned when date range has no trades."""
        result = get_trade_history("2020-01-01", "2020-01-31")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_symbol_filter(self):
        """Symbol filter returns only matching trades."""
        result = get_trade_history("2025-07-01", "2026-12-31", symbol="BTC-USD")
        assert isinstance(result, list)
        assert len(result) > 0
        for trade in result:
            assert trade["symbol"] == "BTC-USD"

    def test_symbol_filter_no_match(self):
        """Symbol filter with non-existent symbol returns empty."""
        result = get_trade_history("2025-07-01", "2026-12-31", symbol="ETH-USD")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_invalid_date_returns_error(self):
        """Invalid date format returns an error dict, not an exception."""
        result = get_trade_history("not-a-date", "2025-08-31")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]

    def test_start_after_end_returns_error(self):
        """start_date after end_date returns an error."""
        result = get_trade_history("2025-12-31", "2025-01-01")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]

    def test_trade_structure(self):
        """Each trade has all expected fields."""
        result = get_trade_history("2025-07-01", "2025-07-31")
        assert len(result) > 0
        trade = result[0]
        expected_keys = {
            "trade_id", "symbol", "action", "entry_date", "entry_price",
            "exit_date", "exit_price", "pnl", "pnl_pct", "position_type",
            "indicators_at_entry", "indicators_at_exit",
        }
        assert expected_keys.issubset(set(trade.keys()))

    def test_all_29_trades_loaded(self):
        """The full dataset of 29 trades is accessible."""
        result = get_trade_history("2025-01-01", "2027-01-01")
        assert len(result) == 29


# ── Vector search tool tests ───────────────────────────────────────────────


class TestSearchMarketContext:
    """Tests for the search_market_context tool."""

    def test_returns_results_for_rsi_query(self):
        """Searching for RSI returns relevant results."""
        result = search_market_context("What is RSI?")
        assert isinstance(result, list)
        assert len(result) > 0
        # At least one result should mention RSI
        texts = [r["text"] for r in result]
        assert any("RSI" in t for t in texts)

    def test_returns_results_for_macd_query(self):
        """Searching for MACD returns relevant results."""
        result = search_market_context("MACD crossover signal")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_result_structure(self):
        """Each result has id, text, and distance fields."""
        result = search_market_context("volatility")
        assert len(result) > 0
        for item in result:
            assert "id" in item
            assert "text" in item
            assert "distance" in item


# ── Agent tool dispatch tests ──────────────────────────────────────────────


class TestAgentToolDispatch:
    """Tests for the agent's tool execution logic (mocked LLM)."""

    def test_execute_known_tool(self):
        """Known tools execute correctly."""
        from app.agent import _execute_tool

        result_str = _execute_tool(
            "get_trade_history",
            {"start_date": "2025-08-01", "end_date": "2025-08-31"},
        )
        result = json.loads(result_str)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_execute_unknown_tool(self):
        """Unknown tool names return a graceful error, not a crash."""
        from app.agent import _execute_tool

        result_str = _execute_tool("nonexistent_tool", {"foo": "bar"})
        result = json.loads(result_str)
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_execute_tool_with_bad_args(self):
        """Tool called with wrong arguments returns an error, not a crash."""
        from app.agent import _execute_tool

        result_str = _execute_tool(
            "get_trade_history",
            {"wrong_param": "value"},
        )
        result = json.loads(result_str)
        assert "error" in result

    def test_ask_raises_without_api_key(self):
        """ask() raises RuntimeError if neither API key is set."""
        from app.agent import ask

        # Temporarily clear all API keys
        original_groq = os.environ.pop("GROQ_API_KEY", None)
        original_ant = os.environ.pop("ANTHROPIC_API_KEY", None)
        original_oai = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(RuntimeError, match="API_KEY"):
                ask("test question")
        finally:
            if original_groq:
                os.environ["GROQ_API_KEY"] = original_groq
            if original_ant:
                os.environ["ANTHROPIC_API_KEY"] = original_ant
            if original_oai:
                os.environ["OPENAI_API_KEY"] = original_oai



# ── Mock LLM & Multi-Tool Loop Tests ──────────────────────────────────────────


class MockTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    def __init__(self, block_id: str, name: str, tool_input: dict):
        self.type = "tool_use"
        self.id = block_id
        self.name = name
        self.input = tool_input


class MockMessageResponse:
    def __init__(self, stop_reason: str, content: list):
        self.stop_reason = stop_reason
        self.content = content


class TestAgentMockLoop:
    """Tests simulating full LLM tool loops with a mocked Anthropic client."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-mock-key", "GROQ_API_KEY": "", "OPENAI_API_KEY": ""})
    @patch("anthropic.Anthropic")
    def test_multi_tool_question(self, mock_anthropic_class):
        """Agent can execute multiple tool calls in a single turn."""
        from app.agent import ask

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Round 1: Claude decides to invoke both tools
        resp_round_1 = MockMessageResponse(
            stop_reason="tool_use",
            content=[
                MockToolUseBlock(
                    block_id="tool_1",
                    name="get_trade_history",
                    tool_input={"start_date": "2025-08-01", "end_date": "2025-08-31"},
                ),
                MockToolUseBlock(
                    block_id="tool_2",
                    name="search_market_context",
                    tool_input={"query": "RSI indicator meaning"},
                ),
            ],
        )

        # Round 2: Claude answers using the results
        resp_round_2 = MockMessageResponse(
            stop_reason="end_turn",
            content=[
                MockTextBlock("In August 2025, 4 trades were made. RSI indicates momentum."),
            ],
        )

        mock_client.messages.create.side_effect = [resp_round_1, resp_round_2]

        result = ask("What trades happened in August 2025 and what does RSI mean?")

        assert mock_client.messages.create.call_count == 2
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["tool"] == "get_trade_history"
        assert result["tool_calls"][1]["tool"] == "search_market_context"
        assert "In August 2025" in result["answer"]

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-mock-key", "GROQ_API_KEY": "", "OPENAI_API_KEY": ""})
    @patch("anthropic.Anthropic")
    def test_multi_turn_history_forwarding(self, mock_anthropic_class):
        """Agent forwards conversation_history in messages."""
        from app.agent import ask

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        resp = MockMessageResponse(
            stop_reason="end_turn",
            content=[MockTextBlock("The previous trade was profitable.")],
        )
        mock_client.messages.create.return_value = resp

        history = [
            {"role": "user", "content": "Show trade #1"},
            {"role": "assistant", "content": "Trade #1 had a PnL of $1,200."},
        ]

        result = ask("Was it profitable?", conversation_history=history)

        assert mock_client.messages.create.call_count == 1
        call_kwargs = mock_client.messages.create.call_args[1]
        sent_messages = call_kwargs["messages"]

        # Expect history + the new question
        assert len(sent_messages) == 3
        assert sent_messages[0] == history[0]
        assert sent_messages[1] == history[1]
        assert sent_messages[2] == {"role": "user", "content": "Was it profitable?"}
        assert result["answer"] == "The previous trade was profitable."

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-mock-key", "GROQ_API_KEY": "", "OPENAI_API_KEY": ""})
    @patch("anthropic.Anthropic")
    def test_max_tool_rounds_exhaustion(self, mock_anthropic_class):
        """Agent stops after max_tool_rounds and returns a graceful message."""
        from app.agent import ask

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Always returns tool_use
        continuous_tool = MockMessageResponse(
            stop_reason="tool_use",
            content=[
                MockToolUseBlock(
                    block_id="tool_loop",
                    name="get_trade_history",
                    tool_input={"start_date": "2025-08-01", "end_date": "2025-08-02"},
                )
            ],
        )
        mock_client.messages.create.return_value = continuous_tool

        result = ask("Infinite loop question", max_tool_rounds=2)

        assert mock_client.messages.create.call_count == 2
        assert len(result["tool_calls"]) == 2
        assert "within the allowed number of tool calls" in result["answer"]


# ── FastAPI Endpoint Tests ───────────────────────────────────────────────────


class TestFastAPIEndpoints:
    """Direct tests for the FastAPI application routes."""

    def test_health_endpoint(self):
        """GET /health returns service status."""
        import asyncio
        from app.main import health

        res = asyncio.run(health())
        assert res["status"] == "ok"
        assert res["service"] == "trading-agent-assistant"
        assert "api_key_configured" in res

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-mock-key"})
    @patch("app.main.ask")
    def test_ask_endpoint_success(self, mock_ask):
        """POST /ask returns answer and tool calls."""
        import asyncio
        from app.main import AskRequest, ask_endpoint

        mock_ask.return_value = {
            "answer": "Test answer from mocked agent.",
            "tool_calls": [
                {"tool": "get_trade_history", "input": {"start_date": "2025-08-01"}, "output": []}
            ],
        }

        req = AskRequest(
            question="What happened in August?",
            conversation_history=[{"role": "user", "content": "hi"}],
        )
        res = asyncio.run(ask_endpoint(req))

        assert res.answer == "Test answer from mocked agent."
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0].tool == "get_trade_history"
        mock_ask.assert_called_once_with(
            "What happened in August?",
            conversation_history=[{"role": "user", "content": "hi"}],
        )

    def test_ask_endpoint_missing_api_key(self):
        """POST /ask returns 500 when ANTHROPIC_API_KEY is missing."""
        import asyncio
        from fastapi import HTTPException
        from app.main import AskRequest, ask_endpoint

        with patch.dict(os.environ, {}, clear=True):
            req = AskRequest(question="Test question")
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(ask_endpoint(req))
            assert exc_info.value.status_code == 500
            assert "ANTHROPIC_API_KEY" in exc_info.value.detail


