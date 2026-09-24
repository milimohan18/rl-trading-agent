# Trading Agent Assistant

An LLM-powered assistant that answers natural-language questions about an RL trading system's behavior. It uses Anthropic Claude with native tool calling to fetch real trade data and indicator values, then composes grounded answers citing specific evidence.

Built as a companion to the [RL Trading Agent](../) project.

## Architecture

```
User question (CLI or HTTP POST)
        │
        ▼
  FastAPI endpoint  POST /ask
        │
        ▼
  Agent orchestrator (LLM tool-call loop)
        │
        ├── Tool: get_trade_history(start_date, end_date, symbol?)
        │     → reads trade log → returns structured JSON
        │
        └── Tool: search_market_context(query)
              → searches indicator explanations & market notes → top-k snippets
        │
        ▼
  LLM composes final answer citing the tool data it used
        │
        ▼
  JSON response: { "answer": "...", "tool_calls": [...] }
```

**Key components:**

| File | Purpose |
|------|---------|
| `app/agent.py` | LLM orchestration — sends messages, handles tool calls, loops until final answer |
| `app/tools/trade_history.py` | `get_trade_history` — queries mock trade log by date range and symbol |
| `app/tools/vector_search.py` | `search_market_context` — semantic search over indicator explanations |
| `app/main.py` | FastAPI app with `POST /ask` and `GET /health` endpoints |
| `scripts/ask_cli.py` | CLI interface for quick testing |
| `app/data/trades.json` | Mock trade data (29 trades, BTC-USD) |

## Setup

### 1. Clone and navigate

```bash
cd trading-agent-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
copy .env.example .env
# Edit .env and replace your_key_here with your actual Anthropic API key
```

### 5. Run

**CLI mode** (no server needed):
```bash
# Single question:
python scripts/ask_cli.py "What trades were made in August 2025?"

# Interactive multi-turn chat session:
python scripts/ask_cli.py -i
```

**API mode**:
```bash
# Terminal 1: start the server
uvicorn app.main:app --reload

# Terminal 2: send a request (supports optional conversation_history for multi-turn)
curl -X POST http://localhost:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What trades were made in August 2025?\"}"
```

## Example Request / Response

**Request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me the most profitable trade and explain why the model entered it"}'
```

**Response:**
```json
{
  "answer": "The most profitable trade was Trade #27, a long position on BTC-USD entered on 2026-02-25 at $82,500.60 and exited on 2026-03-03 at $86,800.40, yielding a profit of $521.08 (+5.21%).\n\nThe model likely entered this trade because the RSI was extremely oversold at 18.5 (well below the 30 threshold), the MACD histogram showed strong negative momentum that was likely to reverse, and the price had fallen significantly below both the 10-day and 50-day EMAs, suggesting a potential mean-reversion opportunity.",
  "tool_calls": [
    {
      "tool": "get_trade_history",
      "input": {"start_date": "2025-07-01", "end_date": "2026-12-31"},
      "output": ["... 29 trades ..."]
    }
  ]
}
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests mock the LLM call — no API key is required to run them.

## Scope & Limitations

- **Mock data**: The trade history in `app/data/trades.json` is **realistic mock data** generated to match the DQN agent's actual evaluation metrics (29 trades, ~70% total return, ~71% win rate on BTC-USD). It is not a direct replay of the trained model.
- **Portfolio/demo project**: This is a demonstration of LLM tool-calling integrated with a trading system. It is **not** a production trading system.
- **No real trades**: No real money is moved, no exchange connections are made, and no live trading is executed.
- **LLM providers**: Supports Anthropic Claude (via `ANTHROPIC_API_KEY`) and OpenAI (via `OPENAI_API_KEY`).
- **Vector search**: Powered by local ChromaDB embedding collection with 15 curated indicator definitions and market context snippets.
- **No authentication**: The API is open — intended for local development only.

## What I'd Build Next

- **Replay real model**: Run the actual DQN/PPO models on the test set and capture per-trade logs with real indicator snapshots, replacing the mock data
- **Multi-turn conversation memory**: Persist conversation history so follow-up questions reference prior context
- **Live data integration**: Connect to a market data API for real-time indicator queries alongside historical trade analysis
