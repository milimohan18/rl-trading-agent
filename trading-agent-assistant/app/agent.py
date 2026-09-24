"""
LLM agent orchestration — Anthropic Claude with native tool use.

Implements a single-turn tool-call loop: the user's question is sent to
the LLM along with tool schemas.  If the LLM responds with tool_use
blocks, the corresponding Python functions are called, their results
are sent back, and the loop repeats until the LLM returns a final
text answer.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import anthropic
from dotenv import load_dotenv

from app.tools.trade_history import TOOL_SCHEMA as TRADE_HISTORY_SCHEMA
from app.tools.trade_history import get_trade_history
from app.tools.vector_search import TOOL_SCHEMA as VECTOR_SEARCH_SCHEMA
from app.tools.vector_search import search_market_context

# Load .env from the project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# ── Tool registry ──────────────────────────────────────────────────────────

# Maps tool name → (callable, schema)
_TOOL_REGISTRY: Dict[str, Any] = {
    "get_trade_history": get_trade_history,
    "search_market_context": search_market_context,
}

# List of all tool schemas to send to the LLM
_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    TRADE_HISTORY_SCHEMA,
    VECTOR_SEARCH_SCHEMA,
]


def register_tool(name: str, fn: Any, schema: Dict[str, Any]) -> None:
    """Register an additional tool at runtime (used by Phase 3 RAG)."""
    _TOOL_REGISTRY[name] = fn
    _TOOL_SCHEMAS.append(schema)


# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert assistant for an RL-based cryptocurrency trading system.

The system uses DQN and PPO reinforcement learning agents trained on BTC-USD
historical data. It tracks technical indicators including RSI, MACD,
Bollinger Bands, EMA, ATR, OBV, and rolling volatility.

Your job is to answer questions about the trading agent's behavior, trade
history, and market indicators. You MUST use the available tools to fetch
real data before answering — do not make up trade data, prices, or indicator
values.

When presenting data to the user:
- Cite specific values from the tool results
- Explain indicator meanings in plain language when relevant
- Be honest about what the data shows, even if results are unfavorable

The trade data comes from a DQN agent's backtest on the BTC-USD test set
(July 2025 – April 2026). The agent made 29 trades with a ~70% total return
and ~71% win rate.
"""


# ── Agent execution ────────────────────────────────────────────────────────

def _execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Execute a tool by name and return a JSON string of the result."""
    fn = _TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return json.dumps({
            "error": f"Unknown tool '{tool_name}'. Available tools: "
                     f"{list(_TOOL_REGISTRY.keys())}"
        })
    try:
        result = fn(**tool_input)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})


def _ask_anthropic(
    question: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    model: str,
    max_tool_rounds: int,
    api_key: str,
) -> Dict[str, Any]:
    """Execute tool loop using Anthropic Claude."""
    client = anthropic.Anthropic(api_key=api_key)

    messages: List[Dict[str, Any]] = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    tool_calls_log: List[Dict[str, Any]] = []

    for _round in range(max_tool_rounds):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=_TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id

                    result_str = _execute_tool(tool_name, tool_input)

                    tool_calls_log.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "output": json.loads(result_str),
                    })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_str,
                    })

            messages.append({"role": "user", "content": tool_results})
        else:
            answer_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    answer_text += block.text

            return {
                "answer": answer_text,
                "tool_calls": tool_calls_log,
            }

    return {
        "answer": (
            "I wasn't able to fully answer your question within the allowed "
            "number of tool calls. Please try rephrasing or asking a more "
            "specific question."
        ),
        "tool_calls": tool_calls_log,
    }


def _ask_openai(
    question: str,
    conversation_history: Optional[List[Dict[str, Any]]],
    model: str,
    max_tool_rounds: int,
    api_key: str,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute tool loop using OpenAI or OpenAI-compatible (Groq) API."""
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in _TOOL_SCHEMAS
    ]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    tool_calls_log: List[Dict[str, Any]] = []

    for _round in range(max_tool_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
        )

        choice = response.choices[0]
        msg = choice.message
        messages.append(msg)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except Exception:
                    tool_input = {}

                result_str = _execute_tool(tool_name, tool_input)

                tool_calls_log.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "output": json.loads(result_str),
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
        else:
            return {
                "answer": msg.content or "",
                "tool_calls": tool_calls_log,
            }

    return {
        "answer": (
            "I wasn't able to fully answer your question within the allowed "
            "number of tool calls. Please try rephrasing or asking a more "
            "specific question."
        ),
        "tool_calls": tool_calls_log,
    }


def ask(
    question: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    max_tool_rounds: int = 5,
) -> Dict[str, Any]:
    """Send a question to the LLM agent and get an answer with tool calls.

    Supports Groq (default if GROQ_API_KEY set), Anthropic Claude, or OpenAI.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    conversation_history : list, optional
        Prior conversation turns for multi-turn support (Phase 4).
    model : str, optional
        Model identifier. Defaults to llama-3.3-70b-versatile for Groq,
        claude-sonnet-4-6 for Anthropic, or gpt-4o for OpenAI.
    max_tool_rounds : int
        Maximum number of tool-call round-trips before forcing a text answer.

    Returns
    -------
    dict
        ``{"answer": "...", "tool_calls": [...]}``
        Each tool_call is ``{"tool": "name", "input": {...}, "output": ...}``.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        return _ask_openai(
            question=question,
            conversation_history=conversation_history,
            model=model or "openai/gpt-oss-120b",
            max_tool_rounds=max_tool_rounds,
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
    elif anthropic_key:
        return _ask_anthropic(
            question=question,
            conversation_history=conversation_history,
            model=model or "claude-sonnet-4-6",
            max_tool_rounds=max_tool_rounds,
            api_key=anthropic_key,
        )
    elif openai_key:
        return _ask_openai(
            question=question,
            conversation_history=conversation_history,
            model=model or "gpt-4o",
            max_tool_rounds=max_tool_rounds,
            api_key=openai_key,
        )
    else:
        raise RuntimeError(
            "Neither GROQ_API_KEY, ANTHROPIC_API_KEY, nor OPENAI_API_KEY is set. "
            "Copy .env.example to .env and add your key."
        )


